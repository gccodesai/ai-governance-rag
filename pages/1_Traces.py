"""Observability dashboard: recent traces, aggregate health metrics, per-step timeline."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

from rag.config import TRACE_DIR
from rag.tracing import Tracer
from ui_common import load_secrets_into_env

st.set_page_config(page_title="Traces · AI Governance Assistant", page_icon="🔍", layout="wide")
load_secrets_into_env()
st.title("🔍 Traces")
st.caption("Every chat request is traced step by step: guardrails, retrieval, LLM calls. "
           "These local traces reset when the app restarts; with Langfuse keys set, "
           "the same traces are also kept in your Langfuse project.")

traces, scores = Tracer(TRACE_DIR).load(limit=500)
if not traces:
    st.info("No traces yet. Ask a question on the chat page, then come back.")
    st.stop()

# ---------------------------------------------------------------- health metrics
lat = [t["duration_ms"] for t in traces]
status = Counter(t["metadata"].get("status", "?") for t in traces)
guard = Counter(g["name"] for t in traces for g in t["metadata"].get("guardrails", [])
                if g["action"] != "allow")
fb = [s["value"] for s in scores.values()]

c = st.columns(5)
c[0].metric("Requests", len(traces))
c[1].metric("Median latency", f"{statistics.median(lat) / 1000:.1f}s")
c[2].metric("p95 latency", f"{sorted(lat)[int(0.95 * (len(lat) - 1))] / 1000:.1f}s")
c[3].metric("Tokens (total)", f"{sum(t['total_tokens'] for t in traces):,}")
c[4].metric("👍 rate", f"{100 * sum(fb) / len(fb):.0f}%" if fb else "–", help=f"{len(fb)} ratings")

left, right = st.columns(2)
left.markdown("**Outcomes**")
left.bar_chart(pd.Series(status, name="requests"), horizontal=True, height=180)
right.markdown("**Guardrail triggers**")
if guard:
    right.bar_chart(pd.Series(guard, name="times triggered"), horizontal=True, height=180)
else:
    right.caption("None yet.")

# ---------------------------------------------------------------- trace table
rows = [{
    "time": datetime.fromtimestamp(t["start"]).strftime("%H:%M:%S"),
    "question": str(t["input"])[:80],
    "status": t["metadata"].get("status"),
    "latency_s": round(t["duration_ms"] / 1000, 2),
    "tokens": t["total_tokens"],
    "feedback": {1.0: "👍", 0.0: "👎"}.get(scores.get(t["id"], {}).get("value"), ""),
    "id": t["id"],
} for t in traces]
st.subheader("Recent requests")
event = st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     on_select="rerun", selection_mode="single-row",
                     column_config={"id": None})
picked = event.selection.rows[0] if event.selection.rows else 0
t = traces[picked]

# ---------------------------------------------------------------- one trace
st.subheader(f"Trace: “{str(t['input'])[:90]}”")
st.caption(f"id `{t['id']}` · session `{t.get('session_id')}` · "
           f"mode `{t['metadata'].get('retrieval_mode')}` · {t['duration_ms'] / 1000:.2f}s")

t0, total = t["start"], max(t["duration_ms"], 1)
for s in t["spans"]:
    offset = (s["start"] - t0) * 1000
    width = max(s["duration_ms"] / total * 100, 1)
    left = offset / total * 100
    color = {"generation": "#7c5cff", "span": "#2b9aa0"}.get(s["kind"], "#999")
    if s["level"] == "ERROR":
        color = "#d9534f"
    label = f"{s['name']} · {s['duration_ms']:.0f} ms"
    if s["kind"] == "generation":
        label += f" · {s['model']} · {s['input_tokens']}→{s['output_tokens']} tok"
    st.markdown(
        f"<div style='font-size:0.85rem;margin:2px 0'>{label}</div>"
        f"<div style='background:#eee3;border-radius:4px;height:10px;position:relative'>"
        f"<div style='position:absolute;left:{left:.1f}%;width:{width:.1f}%;height:10px;"
        f"background:{color};border-radius:4px'></div></div>",
        unsafe_allow_html=True)

st.markdown("")
for s in t["spans"]:
    with st.expander(f"{s['name']} ({s['kind']}, {s['duration_ms']:.0f} ms)"):
        st.markdown("**Input**")
        st.code(json.dumps(s["input"], indent=2, default=str)[:4000] if not isinstance(s["input"], str)
                else s["input"][:4000], language="json" if not isinstance(s["input"], str) else None)
        st.markdown("**Output**")
        st.code(json.dumps(s["output"], indent=2, default=str)[:4000] if not isinstance(s["output"], str)
                else s["output"][:4000], language="json" if not isinstance(s["output"], str) else None)
        if s["metadata"]:
            st.json(s["metadata"], expanded=False)
st.markdown("**Final answer**")
st.write(t["output"])
