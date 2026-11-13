"""Observability: every request becomes a trace made of timed spans.

A trace records each step (guardrails, query rewrite, retrieval, LLM call)
with inputs, outputs, latency, tokens and model. Traces go to:

* a local JSONL file, shown on the app's "Traces" page (always on, free), and
* Langfuse (optional, free Hobby tier) via its public HTTP ingestion API, when
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set. Streamlit Cloud's disk is
  wiped on restart, so Langfuse is where traces live long-term.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def _now_iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat()


@dataclass
class Span:
    name: str
    kind: str = "span"                  # "span" | "generation"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    start: float = field(default_factory=time.time)
    end: float | None = None
    input: object = None
    output: object = None
    metadata: dict = field(default_factory=dict)
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    level: str = "DEFAULT"              # DEFAULT | WARNING | ERROR

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 1)


@dataclass
class Trace:
    name: str
    input: object = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    output: object = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, kind: str = "span", input=None, **metadata):
        s = Span(name=name, kind=kind, input=input, metadata=metadata)
        self.spans.append(s)
        try:
            yield s
        except Exception as e:
            s.level = "ERROR"
            s.output = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.end = time.time()

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 1)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        d["total_tokens"] = self.total_tokens
        for sd, s in zip(d["spans"], self.spans):
            sd["duration_ms"] = s.duration_ms
        return d


class LangfuseExporter:
    """Minimal client for Langfuse's /api/public/ingestion endpoint (no SDK)."""

    def __init__(self, host: str, public_key: str, secret_key: str, timeout: float = 5.0):
        self.url = f"{host.rstrip('/')}/api/public/ingestion"
        self.auth = (public_key, secret_key)
        self.timeout = timeout
        self.last_error: str | None = None

    def _send(self, events: list[dict]) -> None:
        try:
            r = requests.post(self.url, json={"batch": events}, auth=self.auth, timeout=self.timeout)
            if r.status_code not in (200, 201, 207):
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            elif r.status_code == 207 and r.json().get("errors"):
                self.last_error = str(r.json()["errors"])[:300]
            else:
                self.last_error = None
        except requests.RequestException as e:
            self.last_error = f"{type(e).__name__}: {e}"
        if self.last_error:
            log.warning("Langfuse export failed: %s", self.last_error)

    def send_async(self, events: list[dict]) -> None:
        # Don't make the user wait for telemetry.
        threading.Thread(target=self._send, args=(events,), daemon=True).start()

    @staticmethod
    def _event(kind: str, body: dict) -> dict:
        return {"id": uuid.uuid4().hex, "timestamp": _now_iso(), "type": kind, "body": body}

    def export_trace(self, t: Trace) -> None:
        events = [self._event("trace-create", {
            "id": t.id, "name": t.name, "timestamp": _now_iso(t.start),
            "input": t.input, "output": t.output, "metadata": t.metadata,
            "tags": t.tags, "sessionId": t.session_id,
        })]
        for s in t.spans:
            body = {
                "id": s.id, "traceId": t.id, "name": s.name,
                "startTime": _now_iso(s.start), "endTime": _now_iso(s.end),
                "input": s.input, "output": s.output, "metadata": s.metadata, "level": s.level,
            }
            if s.kind == "generation":
                body["model"] = s.model
                body["usage"] = {"input": s.input_tokens, "output": s.output_tokens,
                                 "total": s.input_tokens + s.output_tokens, "unit": "TOKENS"}
                events.append(self._event("generation-create", body))
            else:
                events.append(self._event("span-create", body))
        self.send_async(events)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        self.send_async([self._event("score-create", {
            "id": uuid.uuid4().hex, "traceId": trace_id, "name": name,
            "value": value, "comment": comment,
        })])


class Tracer:
    """Creates traces and fans them out to the JSONL file and (optionally) Langfuse."""

    def __init__(self, trace_dir: Path | None = None, langfuse: LangfuseExporter | None = None):
        self.trace_dir = trace_dir
        self.langfuse = langfuse
        self._lock = threading.Lock()
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self.trace_dir / "traces.jsonl" if self.trace_dir else None

    def start(self, name: str, input=None, session_id: str | None = None, **metadata) -> Trace:
        return Trace(name=name, input=input, session_id=session_id, metadata=metadata)

    def finish(self, t: Trace, output=None) -> None:
        t.end = time.time()
        t.output = output
        if self.path:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")
        if self.langfuse:
            self.langfuse.export_trace(t)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        """Record user feedback (thumbs up = 1, down = 0) against a trace."""
        if self.path:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"score": {"trace_id": trace_id, "name": name,
                                              "value": value, "comment": comment,
                                              "time": _now_iso()}}) + "\n")
        if self.langfuse:
            self.langfuse.score(trace_id, name, value, comment)

    def load(self, limit: int = 200) -> tuple[list[dict], dict[str, dict]]:
        """Read recent traces and the latest score per trace from the JSONL file."""
        if not self.path or not self.path.exists():
            return [], {}
        traces, scores = [], {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "score" in rec:
                scores[rec["score"]["trace_id"]] = rec["score"]
            else:
                traces.append(rec)
        return traces[-limit:][::-1], scores
