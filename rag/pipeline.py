"""The RAG pipeline: guardrails -> (rewrite) -> retrieve -> generate -> guardrails.

Every step runs inside a traced span, so the app, the evals and Langfuse all
see exactly what happened and how long it took.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import guardrails as G
from .llm import LLMClient, LLMError
from .prompts import (ABSTAIN_MESSAGE, ABSTAIN_TOKEN, ANSWER_TEMPLATE, BLOCKED_MESSAGES,
                      CONDENSE_PROMPT, SYSTEM_PROMPT, format_sources)
from .retriever import Hit, Retriever
from .tracing import Tracer

# Off-topic gate thresholds. Calibrated on the eval set with
# `python -m evals.run_evals --calibrate`; see README "Guardrails".
MIN_BM25 = 2.0     # used only when dense retrieval is unavailable
MIN_DENSE = 0.60   # cosine similarity of the best chunk (bge-small-en-v1.5).
                   # Calibration run: in-scope min 0.635; off-topic max 0.55 (except an
                   # EU AI Act question at 0.705, which the LLM's abstention handles).


@dataclass
class ChatResult:
    answer: str
    status: str                                  # answered | abstained | blocked | error
    sources: list[Hit] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    guardrails: list[G.GuardrailResult] = field(default_factory=list)
    query: str = ""                              # the query actually used for retrieval
    trace_id: str = ""
    latency_ms: float = 0.0
    tokens: int = 0


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None, tracer: Tracer,
                 top_k: int = 5, mode: str = "hybrid", max_question_chars: int = 1000,
                 min_bm25: float = MIN_BM25, min_dense: float = MIN_DENSE):
        self.retriever = retriever
        self.llm = llm
        self.tracer = tracer
        self.top_k = top_k
        self.mode = mode
        self.max_question_chars = max_question_chars
        self.min_bm25 = min_bm25
        self.min_dense = min_dense

    # ------------------------------------------------------------------ helpers
    def _finish(self, trace, result: ChatResult) -> ChatResult:
        trace.metadata.update(status=result.status,
                              guardrails=[g.as_dict() for g in result.guardrails])
        trace.tags = [result.status] + [f"guardrail:{g.name}" for g in result.guardrails
                                        if g.action != G.ALLOW]
        self.tracer.finish(trace, output=result.answer)
        result.trace_id = trace.id
        result.latency_ms = trace.duration_ms
        result.tokens = trace.total_tokens
        return result

    def _generate(self, trace, name: str, messages: list[dict], max_tokens: int = 800):
        with trace.span(name, kind="generation", input=messages) as s:
            resp = self.llm.chat(messages, max_tokens=max_tokens)
            s.output, s.model = resp.text, resp.model
            s.input_tokens, s.output_tokens = resp.input_tokens, resp.output_tokens
            s.metadata["attempts"] = resp.attempts
            return resp.text

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str, history: list[dict] | None = None,
            session_id: str | None = None) -> ChatResult:
        history = history or []
        trace = self.tracer.start("chat", input=question, session_id=session_id,
                                  retrieval_mode=self.mode, top_k=self.top_k)
        results: list[G.GuardrailResult] = []

        # 1. Input guardrails ------------------------------------------------
        with trace.span("input_guardrails", input=question) as s:
            results.append(G.check_length(question, self.max_question_chars))
            if not results[-1].blocked:
                results.append(G.check_injection(question))
            clean_q, pii = G.check_pii(question)
            results.append(pii)
            s.output = [g.as_dict() for g in results]
        blocked = next((g for g in results if g.blocked), None)
        if blocked:
            return self._finish(trace, ChatResult(BLOCKED_MESSAGES[blocked.name], "blocked",
                                                  guardrails=results))
        if self.llm is None:
            return self._finish(trace, ChatResult(
                "No API key configured. Add a free Gemini or Groq key in the sidebar.",
                "error", guardrails=results))

        try:
            # 2. Make follow-ups standalone ("what about the second one?") ----
            query = clean_q
            if history:
                convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
                query = self._generate(trace, "condense_question", [
                    {"role": "user", "content": CONDENSE_PROMPT.format(history=convo, question=clean_q)}
                ], max_tokens=150) or clean_q

            # 3. Retrieve ----------------------------------------------------
            with trace.span("retrieval", input=query, mode=self.mode) as s:
                hits = self.retriever.search(query, k=self.top_k, mode=self.mode)
                s.output = [{"id": h.chunk.id, "score": round(h.score, 4),
                             "bm25": round(h.bm25, 2), "dense": round(h.dense, 3)} for h in hits]

            # 4. Relevance guardrail (cheap: no LLM call if clearly off-topic) -
            with trace.span("relevance_guardrail") as s:
                kw = self.retriever.max_keyword_overlap(query)
                dense = max((h.dense for h in hits), default=0.0) if self.retriever.dense_available else None
                rel = G.check_relevance(kw, dense, self.min_bm25, self.min_dense)
                results.append(rel)
                s.output = rel.as_dict()
            if rel.blocked:
                return self._finish(trace, ChatResult(BLOCKED_MESSAGES["relevance"], "blocked",
                                                      sources=hits, guardrails=results, query=query))

            # 5. Generate ----------------------------------------------------
            raw = self._generate(trace, "answer", [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ANSWER_TEMPLATE.format(
                    sources=format_sources(hits), question=query)},
            ])

            # 6. Output guardrails --------------------------------------------
            with trace.span("output_guardrails", input=raw) as s:
                answer, out_checks = G.check_output(raw, len(hits))
                results += out_checks
                s.output = [g.as_dict() for g in out_checks]
        except LLMError as e:
            return self._finish(trace, ChatResult(
                f"The language model is unavailable right now. Details:\n{e}", "error",
                guardrails=results))

        if any(g.blocked for g in out_checks):
            return self._finish(trace, ChatResult(BLOCKED_MESSAGES["prompt_leak"], "blocked",
                                                  sources=hits, guardrails=results, query=query))
        if ABSTAIN_TOKEN in answer:
            return self._finish(trace, ChatResult(ABSTAIN_MESSAGE, "abstained", sources=hits,
                                                  guardrails=results, query=query))
        return self._finish(trace, ChatResult(answer, "answered", sources=hits,
                                              cited=G.cited_sources(answer),
                                              guardrails=results, query=query))
