"""Settings, read from environment variables (or Streamlit secrets copied into env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TRACE_DIR = DATA_DIR / "traces"

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
}


def _env_list(name: str) -> list[str] | None:
    raw = os.environ.get(name, "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()] or None


@dataclass
class Settings:
    provider: str = "gemini"
    api_key: str = ""
    models: list[str] = field(default_factory=list)   # first = primary, rest = fallbacks
    judge_model: str | None = None                     # for evals; defaults to models[0]
    top_k: int = 5
    retrieval_mode: str = "hybrid"                     # "hybrid" | "bm25" | "dense"
    max_question_chars: int = 1000
    request_timeout: float = 60.0
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def base_url(self) -> str:
        return PROVIDERS[self.provider]["base_url"]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
        spec = PROVIDERS[provider]
        s = cls(
            provider=provider,
            api_key=os.environ.get(spec["key_env"], ""),
            models=_env_list("LLM_MODELS") or list(spec["models"]),
            judge_model=os.environ.get("JUDGE_MODEL") or None,
            top_k=int(os.environ.get("TOP_K", 5)),
            retrieval_mode=os.environ.get("RETRIEVAL_MODE", "hybrid"),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/"),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
