"""A tiny chat-completions client for Gemini or Groq (OpenAI-compatible endpoints).

No SDK: one HTTP POST, a hard timeout, retries with backoff, and fallback models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1


class LLMClient:
    def __init__(self, base_url: str, api_key: str, models: list[str],
                 timeout: float = 60.0, retries: int = 2, backoff: float = 2.0):
        if not api_key:
            raise LLMError("No API key. Add one in the sidebar or as a secret.")
        if not models:
            raise LLMError("No model configured.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = models
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _post(self, model: str, messages: list[dict], temperature: float, max_tokens: int):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        # Google advises keeping Gemini 3 at its default temperature (low values can
        # cause looping); other models get the deterministic setting we ask for.
        if not model.startswith("gemini-3"):
            body["temperature"] = temperature
        return requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=self.timeout,
        )

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, models: list[str] | None = None) -> LLMResponse:
        """Try each model in turn; retry transient errors; raise LLMError with every reason."""
        errors, attempts = [], 0
        for model in models or self.models:
            for attempt in range(self.retries + 1):
                attempts += 1
                start = time.perf_counter()
                try:
                    r = self._post(model, messages, temperature, max_tokens)
                except requests.RequestException as e:
                    errors.append(f"{model}: network error ({type(e).__name__})")
                    break
                if r.status_code == 200:
                    data = r.json()
                    choice = (data.get("choices") or [{}])[0]
                    text = ((choice.get("message") or {}).get("content") or "").strip()
                    usage = data.get("usage") or {}
                    if not text:
                        # e.g. a thinking model spent the whole budget reasoning
                        errors.append(f"{model}: empty reply (finish_reason="
                                      f"{choice.get('finish_reason')})")
                        break
                    return LLMResponse(
                        text=text, model=model,
                        input_tokens=usage.get("prompt_tokens", 0) or 0,
                        output_tokens=usage.get("completion_tokens", 0) or 0,
                        latency_s=time.perf_counter() - start, attempts=attempts,
                    )
                if r.status_code in RETRYABLE and attempt < self.retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")
                break  # non-retryable (or out of retries): try the next model
        raise LLMError("All models failed:\n" + "\n".join(errors))

    def list_models(self) -> list[str]:
        """Model ids the key can use; handy when a free-tier model gets retired."""
        r = requests.get(f"{self.base_url}/models",
                         headers={"Authorization": f"Bearer {self.api_key}"}, timeout=15)
        r.raise_for_status()
        return sorted(m["id"].removeprefix("models/") for m in r.json().get("data", []))
