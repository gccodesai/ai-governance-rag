"""Shared Streamlit helpers: settings from secrets/sidebar, cached pipeline, styling."""

from __future__ import annotations

import os
import uuid

import streamlit as st

from rag import Retriever, Settings, build_pipeline, load_chunks
from rag.config import PROVIDERS, TRACE_DIR
from rag.llm import LLMClient

SECRET_KEYS = ["GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODELS", "JUDGE_MODEL",
               "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "ADMIN_PASSWORD"]

STATUS_BADGE = {
    "answered": ":green-badge[answered]",
    "abstained": ":orange-badge[not in sources]",
    "blocked": ":red-badge[blocked by guardrail]",
    "error": ":gray-badge[error]",
}


def load_secrets_into_env() -> None:
    """Streamlit Cloud secrets -> env vars, so the rag package reads one place."""
    try:
        for k in SECRET_KEYS:
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass  # no secrets file locally: fine


@st.cache_resource(show_spinner="Loading the NIST documents and embedding model (first run only)…")
def get_retriever() -> Retriever:
    return Retriever(load_chunks(), use_dense=True)


def session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    return st.session_state.session_id


def sidebar_settings() -> Settings:
    """Render the settings sidebar and return the resulting Settings."""
    load_secrets_into_env()
    base = Settings.from_env()
    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("LLM provider", list(PROVIDERS),
                                index=list(PROVIDERS).index(base.provider),
                                format_func={"gemini": "Google Gemini", "groq": "Groq (Llama)"}.get)
        spec = PROVIDERS[provider]
        has_secret = bool(os.environ.get(spec["key_env"]))
        key = st.text_input(
            "API key", type="password",
            placeholder="Using the app's key" if has_secret else "Paste a free key",
            help="Free keys: aistudio.google.com/apikey (Gemini) or console.groq.com/keys (Groq). "
                 "Your key stays in this browser session.")
        api_key = key or os.environ.get(spec["key_env"], "")
        other = [p for p in PROVIDERS if p != provider][0]
        other_env = PROVIDERS[other]["key_env"]
        with st.expander(f"Backup provider ({other.title()})"):
            backup = st.text_input(f"{other.title()} API key (optional)", type="password",
                                   placeholder="Using the app's key" if os.environ.get(other_env) else "",
                                   help="Used automatically if every model of the main provider fails, "
                                        "e.g. when a free daily quota runs out.")
        default_models = base.models if provider == base.provider else spec["models"]
        models = st.text_input("Models (primary, fallbacks)", ", ".join(default_models))
        mode = st.radio("Retrieval", ["hybrid", "bm25", "dense"], horizontal=True,
                        help="hybrid = keyword (BM25) + semantic embeddings, fused with RRF")
        top_k = st.slider("Chunks to retrieve", 2, 10, base.top_k)

        backup_key = backup or os.environ.get(other_env, "")
        settings = Settings.from_env(provider=provider, api_key=api_key,
                                     backup_keys={other: backup_key} if backup_key else {},
                                     models=[m.strip() for m in models.split(",") if m.strip()],
                                     retrieval_mode=mode, top_k=top_k)

        if st.button("Test connection", width="stretch", disabled=not api_key):
            try:
                client = LLMClient(settings.base_url, api_key, settings.models, timeout=20, retries=0)
                resp = client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=5)
                st.success(f"`{resp.model}` replied in {resp.latency_s:.1f}s")
            except Exception as e:
                st.error(str(e)[:400])
                try:
                    st.caption("Models this key can use: " + ", ".join(client.list_models()[:25]))
                except Exception:
                    pass

        st.divider()
        retriever = get_retriever()
        st.caption(
            f"Dense retrieval: {'on' if retriever.dense_available else 'off (BM25 fallback)'}  \n"
            f"Backup provider: {other.title() if settings.backup_keys else 'none'}  \n"
            f"Langfuse tracing: {'on' if settings.langfuse_enabled else 'off (local traces only)'}  \n"
            f"Corpus: {len(retriever.chunks)} chunks from 3 NIST sources"
        )
    return settings


def get_pipeline(settings: Settings):
    return build_pipeline(settings, retriever=get_retriever(), trace_dir=TRACE_DIR)
