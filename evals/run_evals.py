"""Evaluate the assistant against a hand-checked golden set.

    python -m evals.run_evals --offline      # retrieval + guardrails, no API key, deterministic
    python -m evals.run_evals                # full end-to-end run with an LLM judge
    python -m evals.run_evals --calibrate    # suggest the off-topic similarity threshold

Golden set: evals/golden.jsonl
  * "answerable" cases have a reference answer and verbatim evidence from the corpus;
    a chunk counts as relevant if it contains any evidence string.
  * adversarial cases (out_of_scope, injection, pii) have the statuses we expect.

Results are written to evals/results/<mode>.json and a Markdown summary is printed
(and added to the GitHub Actions job summary when run in CI).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import Settings, build_pipeline, load_chunks                      # noqa: E402
from rag import guardrails as G                                             # noqa: E402
from rag.llm import LLMClient, LLMError                                     # noqa: E402
from rag.retriever import Retriever                                         # noqa: E402
from evals.judge import judge_answer                                        # noqa: E402

GOLDEN = ROOT / "evals" / "golden.jsonl"
RESULTS = ROOT / "evals" / "results"
K_VALUES = (1, 3, 5)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def relevant_ids(case: dict, chunks) -> set[str]:
    ev = [norm(e) for e in case["evidence"]]
    return {c.id for c in chunks if any(e in norm(c.text) for e in ev)}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- offline evals

def eval_retrieval(retriever: Retriever, cases: list[dict], mode: str) -> dict:
    chunks = retriever.chunks
    hits_at = {k: 0 for k in K_VALUES}
    rr, per_case = [], []
    for c in cases:
        rel = relevant_ids(c, chunks)
        ranked = [h.chunk.id for h in retriever.search(c["question"], k=max(K_VALUES), mode=mode)]
        first = next((i + 1 for i, cid in enumerate(ranked) if cid in rel), None)
        for k in K_VALUES:
            hits_at[k] += bool(first and first <= k)
        rr.append(1 / first if first else 0.0)
        per_case.append({"id": c["id"], "rank": first, "top": ranked[:3]})
    n = len(cases)
    return {"mode": mode, "n": n, **{f"hit@{k}": hits_at[k] / n for k in K_VALUES},
            "mrr": sum(rr) / n, "misses": [p for p in per_case if not p["rank"] or p["rank"] > 5]}


def eval_guardrails_offline(cases: list[dict]) -> dict:
    """Deterministic input guardrails: catch attacks without blocking real questions."""
    inj = [c for c in cases if c["type"] == "injection"]
    pii = [c for c in cases if c["type"] == "pii"]
    benign = [c for c in cases if c["type"] == "answerable"]
    caught = [c["id"] for c in inj if G.check_injection(c["question"]).blocked]
    redacted = [c["id"] for c in pii if G.check_pii(c["question"])[1].action == G.REDACT]
    false_pos = [c["id"] for c in benign
                 if G.check_injection(c["question"]).blocked or G.check_pii(c["question"])[1].action != G.ALLOW]
    return {
        "injection_detection_rate": len(caught) / max(len(inj), 1),
        "injection_missed": sorted({c["id"] for c in inj} - set(caught)),
        "pii_redaction_rate": len(redacted) / max(len(pii), 1),
        "false_positive_rate": len(false_pos) / max(len(benign), 1),
        "false_positives": false_pos,
    }


def run_offline(settings: Settings) -> tuple[dict, str]:
    cases = load_golden()
    answerable = [c for c in cases if c["type"] == "answerable"]
    retriever = Retriever(load_chunks(), use_dense=True)
    modes = ["bm25"] + (["dense", "hybrid"] if retriever.dense_available else [])
    retrieval = [eval_retrieval(retriever, answerable, m) for m in modes]
    guard = eval_guardrails_offline(cases)
    result = {"kind": "offline", "retrieval": retrieval, "guardrails": guard,
              "dense_error": retriever.dense_error}

    md = ["## Offline evals (no LLM)", "",
          f"Golden set: {len(answerable)} answerable questions, "
          f"{len(cases) - len(answerable)} adversarial prompts.", "",
          "### Retrieval", "", "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for r in retrieval:
        md.append(f"| {r['mode']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(r['hit@5'])} | {r['mrr']:.2f} |")
    if retriever.dense_error:
        md += ["", f"_Dense retrieval unavailable here: {retriever.dense_error[:120]}_"]
    md += ["", "### Input guardrails", "",
           f"- Prompt-injection detection: **{pct(guard['injection_detection_rate'])}**"
           + (f" (missed: {', '.join(guard['injection_missed'])})" if guard["injection_missed"] else ""),
           f"- PII redaction: **{pct(guard['pii_redaction_rate'])}**",
           f"- False positives on real questions: **{pct(guard['false_positive_rate'])}**"]
    return result, "\n".join(md)


# ---------------------------------------------------------------- full evals

def run_full(settings: Settings, limit: int | None, sleep: float) -> tuple[dict, str]:
    if not settings.api_key:
        raise SystemExit("Full evals need an API key (GEMINI_API_KEY or GROQ_API_KEY).")
    cases = load_golden()[:limit] if limit else load_golden()
    pipeline = build_pipeline(settings, trace_dir=RESULTS / "traces")
    judge = LLMClient(settings.base_url, settings.api_key,
                      [settings.judge_model] if settings.judge_model else settings.models,
                      timeout=settings.request_timeout)
    chunks = pipeline.retriever.chunks
    rows = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    partial = RESULTS / "full.partial.jsonl"   # survives a timeout or cancel
    partial.write_text("")
    started = time.time()
    for i, c in enumerate(cases, 1):
        res = pipeline.ask(c["question"])
        row = {"id": c["id"], "type": c["type"], "status": res.status,
               "latency_ms": res.latency_ms, "tokens": res.tokens, "trace_id": res.trace_id,
               "model": res.model,
               "answer": res.answer, "guardrails": [g.as_dict() for g in res.guardrails]}
        if c["type"] == "answerable":
            rel = relevant_ids(c, chunks)
            row["retrieved_relevant"] = any(h.chunk.id in rel for h in res.sources)
            row["has_citations"] = bool(res.cited)
            if res.status == "answered":
                context = "\n\n".join(f"[{j}] {h.chunk.text}" for j, h in enumerate(res.sources, 1))
                try:
                    row.update(judge_answer(judge, c["question"], c["reference_answer"], res.answer, context))
                except LLMError as e:
                    row["judge_error"] = str(e)[:300]
            row["passed"] = res.status == "answered" and row.get("correctness", 0) >= 4 \
                and row.get("faithfulness") in ("yes", "partial")
        else:
            ok = res.status in c["expected_status"]
            if c["type"] == "pii":
                ok = ok and any(g["name"] == "pii" and g["action"] == G.REDACT for g in row["guardrails"])
            row["passed"] = ok
        rows.append(row)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"[{i}/{len(cases)}] {time.time() - started:6.0f}s  {c['id']:8} {res.status:9} "
              f"{'PASS' if row['passed'] else 'FAIL'}  ({res.latency_ms / 1000:.1f}s, {res.model or '-'})",
              flush=True)
        time.sleep(sleep)  # stay under free-tier requests-per-minute limits

    ans = [r for r in rows if r["type"] == "answerable"]
    judged = [r for r in ans if "correctness" in r]
    adv = [r for r in rows if r["type"] != "answerable"]

    def rate(rs, key):
        return sum(bool(r.get(key)) for r in rs) / max(len(rs), 1)

    by_type = {}
    for t in ("out_of_scope", "injection", "pii"):
        rs = [r for r in adv if r["type"] == t]
        if rs:
            by_type[t] = rate(rs, "passed")
    summary = {
        "answer_pass_rate": rate(ans, "passed"),
        "mean_correctness": statistics.mean(r["correctness"] for r in judged) if judged else None,
        "faithfulness_rate": sum(r.get("faithfulness") == "yes" for r in judged) / max(len(judged), 1),
        "citation_rate": rate(ans, "has_citations"),
        "retrieval_hit_rate": rate(ans, "retrieved_relevant"),
        "false_refusal_rate": sum(r["status"] != "answered" for r in ans) / max(len(ans), 1),
        "adversarial_pass_rate": by_type,
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
    }
    result = {"kind": "full", "provider": settings.provider, "models": settings.models,
              "judge_model": settings.judge_model or settings.models[0],
              "retrieval_mode": settings.retrieval_mode, "summary": summary, "rows": rows}

    s = summary
    md = ["## End-to-end evals", "",
          f"Provider `{settings.provider}`, models `{', '.join(settings.models)}`, "
          f"retrieval `{settings.retrieval_mode}`, {len(rows)} cases.", "",
          "| Metric | Value |", "|---|---|",
          f"| Answer pass rate (correctness ≥ 4/5 and grounded) | **{pct(s['answer_pass_rate'])}** |",
          f"| Mean correctness (LLM judge, 1–5) | {s['mean_correctness']:.2f} |" if judged else "| Mean correctness | n/a |",
          f"| Fully faithful to sources | {pct(s['faithfulness_rate'])} |",
          f"| Answers with citations | {pct(s['citation_rate'])} |",
          f"| Relevant chunk retrieved | {pct(s['retrieval_hit_rate'])} |",
          f"| False refusals | {pct(s['false_refusal_rate'])} |"]
    for t, v in by_type.items():
        md.append(f"| Guardrail: {t.replace('_', ' ')} handled | {pct(v)} |")
    md += [f"| Median latency | {s['p50_latency_ms'] / 1000:.1f}s |",
           f"| Mean tokens per request | {s['mean_tokens']:.0f} |"]
    fails = [r for r in rows if not r["passed"]]
    if fails:
        md += ["", "### Failures", ""]
        md += [f"- `{r['id']}` ({r['status']}): {r.get('reason') or r['answer'][:150]}" for r in fails]
    return result, "\n".join(md)


# ---------------------------------------------------------------- calibration

def run_calibrate(settings: Settings) -> tuple[dict, str]:
    retriever = Retriever(load_chunks(), use_dense=True)
    if not retriever.dense_available:
        raise SystemExit(f"Calibration needs dense retrieval: {retriever.dense_error}")
    cases = load_golden()
    scores = {"answerable": [], "out_of_scope": []}
    for c in cases:
        if c["type"] in scores:
            hits = retriever.search(c["question"], k=5, mode="dense")
            scores[c["type"]].append((round(hits[0].dense, 3), c["id"]))
    lo_in = min(s for s, _ in scores["answerable"])
    hi_out = max(s for s, _ in scores["out_of_scope"])
    # Never block a real question: sit just below the lowest in-scope score.
    suggested = round(min(lo_in - 0.02, (lo_in + hi_out) / 2) if hi_out < lo_in else lo_in - 0.02, 3)
    md = ["## Relevance-gate calibration (dense cosine of best chunk)", "",
          f"- In-scope: min {lo_in}, median {statistics.median(s for s, _ in scores['answerable'])}",
          f"- Out-of-scope: max {hi_out}, values {sorted(scores['out_of_scope'])}",
          f"- **Suggested MIN_DENSE = {suggested}** (set in rag/pipeline.py). Questions that "
          "pass the gate but aren't covered are caught by the LLM's abstention."]
    return {"kind": "calibrate", "scores": scores, "suggested_min_dense": suggested}, "\n".join(md)


# ---------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--offline", action="store_true", help="retrieval + guardrail metrics only (no LLM)")
    g.add_argument("--calibrate", action="store_true", help="suggest the off-topic threshold")
    p.add_argument("--limit", type=int, help="only run the first N cases")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between cases (free-tier rate limits)")
    p.add_argument("--min-pass-rate", type=float, default=0.0,
                   help="exit non-zero if the headline metric falls below this (CI quality gate)")
    args = p.parse_args()

    settings = Settings.from_env()
    if args.offline:
        name, (result, md) = "offline", run_offline(settings)
        headline = min(r["hit@5"] for r in result["retrieval"])
    elif args.calibrate:
        name, (result, md) = "calibrate", run_calibrate(settings)
        headline = 1.0
    else:
        name, (result, md) = "full", run_full(settings, args.limit, args.sleep)
        headline = result["summary"]["answer_pass_rate"]

    result["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["commit"] = os.environ.get("GITHUB_SHA", "")[:7]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (RESULTS / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if headline < args.min_pass_rate:
        print(f"\nQuality gate failed: {headline:.2f} < {args.min_pass_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
