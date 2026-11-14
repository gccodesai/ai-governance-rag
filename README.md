# ⚖️ AI Governance Assistant: RAG with guardrails, evals and observability

A chatbot that answers questions about AI risk management using the **NIST AI Risk
Management Framework**, its **Generative AI Profile** and the **AI RMF Playbook**.
Every answer cites its sources. Unsafe or off-topic requests are stopped by guardrails,
accuracy is measured by an automated eval suite, and each request is traced step by step.

**Live demo:** _add your Streamlit link here_ · **Cost to run:** $0

![CI](https://github.com/gccodesai/ai-governance-rag/actions/workflows/ci.yml/badge.svg)

## What it shows

| Capability | How it's done here |
|---|---|
| **RAG** | 626 chunks from 3 NIST sources; hybrid retrieval (BM25 keyword + `bge-small` embeddings, fused with Reciprocal Rank Fusion); answers grounded in numbered sources with `[n]` citations |
| **Guardrails** | 4 input checks + 4 output checks, deterministic and unit-tested (table below) |
| **Evals** | Hand-checked golden set: 35 questions with reference answers + 14 adversarial prompts. Retrieval metrics (Hit@k, MRR), LLM-as-judge correctness and faithfulness, guardrail pass rates, run in GitHub Actions |
| **Observability** | Every request is a trace of timed spans (guardrails → rewrite → retrieval → LLM → guardrails) with tokens, models and user 👍/👎. Viewable in the app, and exported to Langfuse |

## Architecture

```
 question
    │
    ▼
┌───────────────────┐  block   ┌──────────────────────────────────────────────┐
│ Input guardrails  │────────► │ Polite refusal (no LLM call, no cost)        │
│ length, injection,│          └──────────────────────────────────────────────┘
│ PII redaction     │                         ▲ block
└────────┬──────────┘                         │
         ▼                                    │
┌───────────────────┐   ┌──────────────┐   ┌──┴────────────────┐
│ Rewrite follow-up │──►│ Hybrid search│──►│ Relevance gate    │
│ (only if history) │   │ BM25 + dense │   │ (semantic score)  │
└───────────────────┘   │ → RRF top-5  │   └────────┬──────────┘
                        └──────────────┘            ▼
                                           ┌───────────────────┐
                                           │ LLM (Gemini/Groq) │  "answer only from
                                           │ cites [1]..[5]    │   sources, else abstain"
                                           └────────┬──────────┘
                                                    ▼
                                           ┌───────────────────┐
                                           │ Output guardrails │──► answer + sources
                                           │ leak, abstain,    │
                                           │ citations, PII    │
                                           └───────────────────┘
   every box above = one span in the trace ──► Traces page + Langfuse
```

## Guardrails

| Stage | Check | Action |
|---|---|---|
| Input | Length (empty / > 1,000 chars) | block |
| Input | Prompt injection ("ignore previous instructions", "reveal your system prompt", role-play jailbreaks, fake `</system>` tags) | block |
| Input | PII: emails, phone numbers, SSNs, card numbers (Luhn-checked) | redact before anything reaches the LLM or the logs |
| Input | Off-topic: semantic similarity of the best chunk below a calibrated threshold | block, without spending an LLM call |
| Prompt | Model must answer only from sources, treat sources as data, and reply `NOT_IN_SOURCES` otherwise | abstain |
| Output | System-prompt leakage (canary phrases) | block |
| Output | Citations: removes references to sources that don't exist; flags answers with none | warn |
| Output | PII in the answer | redact |

Every decision is shown under the answer, recorded in the trace and measured in the evals.
The rules are deterministic on purpose: they're free, fast, explainable and testable.
An LLM-based classifier would be a natural next step.

## Evals

```bash
python -m evals.run_evals --offline     # retrieval + guardrails: no API key, runs on every push
python -m evals.run_evals               # end-to-end with an LLM judge (uses your free key)
python -m evals.run_evals --calibrate   # pick the off-topic threshold from data
```

- **Golden set** ([`evals/golden.jsonl`](evals/golden.jsonl)): each answerable question has a
  reference answer and *verbatim evidence* from the documents. A unit test checks that every evidence
  string really exists in the corpus, so the labels can't silently drift.
- **Retrieval:** Hit@1/3/5 and MRR for BM25, dense and hybrid, which gives an ablation of each
  retrieval approach.
- **Generation:** an LLM judge scores *correctness* (1–5, against the reference) and *faithfulness*
  (is every claim supported by the retrieved sources?). Also tracked: citation rate, false refusals,
  latency and tokens.
- **Safety:** off-topic prompts must be blocked or abstained, injections blocked, PII redacted.

### Results so far

Retrieval on the 35 answerable questions (from CI, no LLM involved):

| Change | Mode | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| Baseline chunks | BM25 | 37% | 66% | 0.49 |
| | Dense | 43% | 69% | 0.53 |
| | Hybrid (RRF) | 51% | 71% | 0.58 |
| **+ contextual chunk headers** (section and subcategory prefixed to each chunk) | BM25 | 46% | 83% | 0.60 |
| | Dense | 63% | 86% | 0.71 |
| | **Hybrid (RRF)** | **66%** | **89%** | **0.73** |

The eval set exposed the problem: definitions in the Generative AI Profile were buried in lists
and drowned out by action tables that mention the same terms. Adding each chunk's section title
fixed most of those misses. Hybrid beats either method alone at every cutoff.

Guardrails: 100% of injection prompts blocked, 100% of PII redacted, 0% false positives on real
questions. The off-topic gate threshold (0.60) comes from the calibration run: real questions
score at least 0.646, unrelated ones at most 0.54. The one governance-adjacent outlier (an EU AI
Act question, 0.705) is left for the LLM's abstention, which the end-to-end evals check.
End-to-end answer quality: see the app's **Evals** page after running the workflow.
- **CI:** [`ci.yml`](.github/workflows/ci.yml) runs unit tests and offline evals on every push with a
  quality gate, and commits results back. [`evals.yml`](.github/workflows/evals.yml) runs the full
  suite on demand from the Actions tab. The app's **Evals** page shows the latest results.

## Observability

- **Built in:** the **Traces** page shows request volume, median and p95 latency, tokens, outcome mix,
  guardrail triggers and 👍 rate, plus a per-request timeline with each step's input and output.
- **Langfuse (optional, free Hobby plan):** add `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and the
  same traces, LLM generations (with token usage) and user feedback scores are sent to Langfuse
  through its public ingestion API, without a heavy SDK. Export runs on a background thread, so
  telemetry never slows the user down.

## Run it in the cloud (no local install)

1. **Get a free LLM key:** [Google AI Studio](https://aistudio.google.com/apikey)
   or [Groq](https://console.groq.com/keys), with no credit card.
2. **Deploy:** [share.streamlit.io](https://share.streamlit.io) → **Create app** → this repo,
   branch `main`, file `app.py` → **Advanced settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   Also add a free [Groq](https://console.groq.com/keys) key as `GROQ_API_KEY`: it's the
   automatic backup when Gemini's daily free quota runs out.
   (See [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) for optional settings.)
3. **Enable end-to-end evals:** repo **Settings → Secrets and variables → Actions** → add
   `GEMINI_API_KEY` (and `GROQ_API_KEY`). Then **Actions → End-to-end evals → Run workflow**.
   A full run makes about 100 LLM calls, which can use up a free model's daily quota, so run it
   sparingly or pick `groq` as the provider.
4. **(Optional) Langfuse:** sign up at [cloud.langfuse.com](https://cloud.langfuse.com), create a
   project and API keys, then add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to both the
   Streamlit secrets and the GitHub Actions secrets.

> **Tip:** to protect your free quota on a public demo, remove the key from the secrets.
> Visitors can then paste their own key in the sidebar.

<details><summary>Run locally instead</summary>

```bash
git clone https://github.com/gccodesai/ai-governance-rag.git && cd ai-governance-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export GEMINI_API_KEY=your-key
streamlit run app.py
pytest
```
</details>

## Project structure

```
rag/
  pipeline.py     # orchestration: guardrails → rewrite → retrieve → generate → guardrails
  retriever.py    # BM25 + fastembed dense search, RRF fusion, embedding cache
  guardrails.py   # input/output checks (each returns an explainable GuardrailResult)
  tracing.py      # traces & spans, JSONL sink, Langfuse exporter, feedback scores
  llm.py          # ~100-line HTTP client for Gemini/Groq: timeouts, retries, model fallback
  prompts.py      # system prompt, refusal messages
evals/
  golden.jsonl    # 49 hand-checked cases
  run_evals.py    # offline / full / calibrate modes, Markdown + JSON reports
  judge.py        # LLM-as-judge prompt and strict output parsing
  results/        # latest results (committed by CI, shown in the app)
app.py, pages/    # Streamlit chat, Traces dashboard, Evals dashboard
scripts/ingest.py # PDFs + Playbook → cleaned, overlapping chunks
tests/            # 51 unit tests with a fake LLM: fast, free, offline
```

## Design decisions

- **No framework, no SDKs.** The pipeline is about 150 readable lines, so every step is visible and
  traceable. The LLM client is plain HTTP against OpenAI-compatible endpoints, so switching between
  Gemini and Groq is a config change.
- **Hybrid retrieval.** Keyword search nails exact IDs like "GOVERN 1.1", embeddings handle paraphrases,
  and RRF combines them without having to tune score scales. The eval suite measures each mode.
- **Gate before generating.** Blocking injections and off-topic questions before the LLM call saves
  quota and removes a whole class of failure. The LLM's own abstention is the second line of defence.
- **Evals that can't lie quietly.** Reference evidence is quoted verbatim and machine-checked, and the
  off-topic threshold is calibrated from data rather than guessed.
- **Built for flaky free tiers.** Free models have small daily quotas and get retired without
  notice. The client retries transient errors, honours rate-limit hints, skips a model whose
  daily quota is used up, falls back through several models, then to a second provider
  (Gemini → Groq). The UI shows which model answered.
- **Free end to end.** Streamlit Community Cloud, GitHub Actions, free LLM tiers, a local ONNX
  embedding model and Langfuse's free tier.

## Roadmap

- [ ] Re-ranking with a cross-encoder and its effect on Hit@1
- [ ] LLM-based safety classifier compared with the rule-based guardrails
- [ ] Add the EU AI Act and ISO/IEC 42001 summaries as extra sources
- [ ] Rebuild the pipeline in LangGraph and compare traces

## Data & license

Source documents are U.S. government works (see [`data/raw/SOURCES.md`](data/raw/SOURCES.md)).
This is a learning project, not legal or compliance advice. Code: MIT.
