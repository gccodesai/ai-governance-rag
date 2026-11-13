"""AI Governance Assistant: a RAG chatbot over the NIST AI RMF, with guardrails and tracing.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ui_common import STATUS_BADGE, get_pipeline, session_id, sidebar_settings

st.set_page_config(page_title="AI Governance Assistant", page_icon="⚖️", layout="centered")

EXAMPLES = [
    "What are the four functions of the AI RMF Core?",
    "What is confabulation in generative AI?",
    "How should we manage risks from third-party AI vendors?",
    "What's the difference between a Current Profile and a Target Profile?",
]

settings = sidebar_settings()
pipeline = get_pipeline(settings)

st.title("⚖️ AI Governance Assistant")
st.caption("Answers questions about the **NIST AI Risk Management Framework**, its **Generative AI "
           "Profile** and the **AI RMF Playbook**, with citations. Not legal advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_details(msg: dict, idx: int) -> None:
    """Badges, guardrail decisions, sources and feedback under an assistant message."""
    r = msg["result"]
    flagged = [g for g in r["guardrails"] if g["action"] != "allow"]
    line = [STATUS_BADGE.get(r["status"], r["status"])]
    line += [f":violet-badge[{g['name']}: {g['action']}]" for g in flagged]
    line.append(f"{r['latency_ms'] / 1000:.1f}s · {r['tokens']} tokens")
    st.markdown(" ".join(line))

    if r["sources"]:
        with st.expander(f"Sources ({len(r['cited'])} cited of {len(r['sources'])} retrieved)"):
            for i, s in enumerate(r["sources"], 1):
                mark = "✅" if i in r["cited"] else "▫️"
                st.markdown(f"{mark} **[{i}] [{s['doc_title']}]({s['url']})**, {s['location']}  \n"
                            f"<small>score {s['score']:.3f} · bm25 {s['bm25']:.1f} · dense {s['dense']:.2f}</small>",
                            unsafe_allow_html=True)
                st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))
    with st.expander("Guardrails and trace"):
        for g in r["guardrails"]:
            icon = {"allow": "🟢", "redact": "🟡", "warn": "🟠", "block": "🔴"}[g["action"]]
            st.markdown(f"{icon} `{g['name']}` **{g['action']}** {g['detail']}")
        if r.get("query") and r["query"] != msg.get("question"):
            st.markdown(f"Rewritten query: _{r['query']}_")
        st.caption(f"Trace `{r['trace_id']}`, open the **Traces** page for the step-by-step timeline.")

    fb = st.feedback("thumbs", key=f"fb_{idx}")
    if fb is not None and msg.get("feedback") != fb:
        msg["feedback"] = fb
        pipeline.tracer.score(r["trace_id"], "user_feedback", float(fb))


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg, i)

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, width="stretch"):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about AI risk management…") or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking guardrails, searching the framework, writing an answer…"):
            res = pipeline.ask(question, history=history, session_id=session_id())
        st.markdown(res.answer)
    st.session_state.messages.append({
        "role": "assistant", "content": res.answer, "question": question,
        "result": {
            "status": res.status, "trace_id": res.trace_id, "latency_ms": res.latency_ms,
            "tokens": res.tokens, "cited": res.cited, "query": res.query,
            "guardrails": [g.as_dict() for g in res.guardrails],
            "sources": [{"doc_title": h.chunk.doc_title, "url": h.chunk.url, "location": h.chunk.location,
                         "text": h.chunk.text, "score": h.score, "bm25": h.bm25, "dense": h.dense}
                        for h in res.sources],
        },
    })
    st.rerun()

if st.session_state.messages and st.sidebar.button("Clear chat", width="stretch"):
    st.session_state.messages = []
    st.rerun()
