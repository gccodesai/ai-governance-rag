"""AI Governance RAG assistant: retrieval, guardrails, tracing."""

from __future__ import annotations

from .config import TRACE_DIR, Settings
from .llm import LLMClient, LLMError
from .pipeline import ChatResult, RAGPipeline
from .retriever import Retriever, load_chunks
from .tracing import LangfuseExporter, Tracer


def build_pipeline(settings: Settings, retriever: Retriever | None = None,
                   trace_dir=TRACE_DIR) -> RAGPipeline:
    retriever = retriever or Retriever(load_chunks(), use_dense=settings.retrieval_mode != "bm25")
    llm = None
    if settings.api_key:
        llm = LLMClient(settings.base_url, settings.api_key, settings.models,
                        timeout=settings.request_timeout)
    langfuse = None
    if settings.langfuse_enabled:
        langfuse = LangfuseExporter(settings.langfuse_host, settings.langfuse_public_key,
                                    settings.langfuse_secret_key)
    tracer = Tracer(trace_dir, langfuse)
    return RAGPipeline(retriever, llm, tracer, top_k=settings.top_k,
                       mode=settings.retrieval_mode,
                       max_question_chars=settings.max_question_chars)


__all__ = ["Settings", "LLMClient", "LLMError", "ChatResult", "RAGPipeline", "Retriever",
           "load_chunks", "Tracer", "LangfuseExporter", "build_pipeline"]
