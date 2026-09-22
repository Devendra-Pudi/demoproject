"""Streamlit client for the citation-enforcing RAG service."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import httpx
import streamlit as st

from ai_portfolio.dashboard import api_get, api_post, page, unavailable

page("Production RAG Application", "Ask company policy questions and inspect the exact supporting evidence.")
st.caption("The production-pattern backend includes hybrid retrieval, cross-encoder reranking and citation checks.")
with st.sidebar:
    if st.button("Check backend"):
        try:
            health = api_get("/health")
            st.json(health)
        except (httpx.HTTPError, ValueError):
            unavailable()
with st.form("question"):
    question = st.text_area("Question", "When must production API keys be rotated?", max_chars=2000)
    submitted = st.form_submit_button("Ask my docs", type="primary")
if submitted:
    if len(question.strip()) < 3:
        st.warning("Enter at least three non-whitespace characters.")
    else:
        try:
            with st.spinner("Retrieving and validating evidence…"):
                st.session_state["answer"] = api_post("/ask", {"question": question})
        except (httpx.HTTPError, ValueError):
            unavailable()
result = st.session_state.get("answer")
if result:
    st.caption("Trace: " + result["trace_id"])
    if result.get("retrieval_mode") == "smoke":
        st.warning("SMOKE MODE: lexical test doubles, not neural embeddings or a cross-encoder.")
    if result["status"] != "ok":
        st.warning("Retrieval-only fallback. These passages are not a generated answer.")
    st.text(result["answer"])
    for i, citation in enumerate(result["citations"] or result["retrieved"], 1):
        with st.expander(f"[{i}] {citation['source']}"):
            st.text(citation.get("quote", citation.get("text", "")))
    st.download_button("Download evidence report", data=json.dumps(result, indent=2),
                       file_name="answer-evidence.json", mime="application/json")
