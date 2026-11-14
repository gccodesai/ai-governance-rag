"""AI Governance RAG assistant: retrieval, guardrails, tracing."""

from __future__ import annotations

from .config import TRACE_DIR, Settings
from .config import PROVIDERS
from .llm import FallbackLLM, LLMClient, LLMError
from .pipeline import ChatResult, RAGPipeline
from .retriever import Retriever, load_chunks
from .tracing import LangfuseExporter, Tracer


def build_llm(settings: Settings, models: list[str] | None = None) -> FallbackLLM | None:
    """The chosen provider first; any other provider with a key configured as backup."""
    clients = []
    if settings.api_key:
        clients.append(LLMClient(settings.base_url, settings.api_key, models or settings.models,
                                 timeout=settings.request_timeout))
    for name, spec in PROVIDERS.items():
        key = settings.backup_keys.get(name, "")
        if name != settings.provider and key:
            clients.append(LLMClient(spec["base_url"], key, spec["models"],
                                     timeout=settings.request_timeout))
    return FallbackLLM(clients) if clients else None


def build_pipeline(settings: Settings, retriever: Retriever | None = None,
                   trace_dir=TRACE_DIR) -> RAGPipeline:
    retriever = retriever or Retriever(load_chunks(), use_dense=settings.retrieval_mode != "bm25")
    llm = build_llm(settings)
    langfuse = None
    if settings.langfuse_enabled:
        langfuse = LangfuseExporter(settings.langfuse_host, settings.langfuse_public_key,
                                    settings.langfuse_secret_key)
    tracer = Tracer(trace_dir, langfuse)
    return RAGPipeline(retriever, llm, tracer, top_k=settings.top_k,
                       mode=settings.retrieval_mode,
                       max_question_chars=settings.max_question_chars)


__all__ = ["Settings", "LLMClient", "LLMError", "ChatResult", "RAGPipeline", "Retriever",
           "load_chunks", "Tracer", "LangfuseExporter", "build_pipeline", "build_llm",
           "FallbackLLM"]
