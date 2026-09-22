"""Read-only RAG telemetry dashboard; no database credentials exposed in the browser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import httpx
import streamlit as st

from ai_portfolio.dashboard import api_get, page, unavailable, uploaded_json
from ai_portfolio.evaluation import check_gates

page("Monitoring & Observability", "Latency, request cost, degradation and content-free traces for the RAG service.")
live, evaluation = st.tabs(["Live telemetry", "Evaluation gate"])
with live:
    st.caption("Refresh is explicit to avoid polling or contacting an unconfigured backend on startup.")
    if st.button("Refresh metrics", type="primary"):
        try:
            st.session_state["metrics"] = api_get("/metrics")
            st.session_state["traces"] = api_get("/traces")
        except (httpx.HTTPError, ValueError):
            unavailable()
    m = st.session_state.get("metrics")
    if m:
        columns = st.columns(4)
        columns[0].metric("Requests", m["requests"])
        columns[1].metric("p50", f"{m['p50_ms']:.1f} ms" if m["p50_ms"] is not None else "No samples")
        columns[2].metric("p95", f"{m['p95_ms']:.1f} ms" if m["p95_ms"] is not None else "No samples")
        columns[3].metric("Fallback / error rate", f"{m['degradation_rate']:.1%}" if m["requests"] else "No samples")
        st.caption(m["window"] + ". " + m["cost_note"])
        st.json({k: m[k] for k in ["mean_cost_per_request_usd", "estimated_api_cost_usd", "citation_valid_rate"]})
        st.dataframe(st.session_state.get("traces", []), use_container_width=True)
    else:
        st.info("Connect a RAG backend and refresh. An empty deployment has no request metrics yet.")
with evaluation:
    upload = st.file_uploader("Upload a RAG evaluation report", type=["json"])
    if upload:
        try:
            report = uploaded_json(upload)
            failures = check_gates(report)
            st.json(report)
            if failures:
                st.error("Gate failed: " + "; ".join(failures))
            else:
                st.success("Absolute gates passed for the uploaded report.")
            st.caption("This checks reported values, not their provenance. CI separately compares a reviewed baseline.")
        except (KeyError, ValueError, TypeError):
            st.error("Invalid report. Upload output from ai-eval.")
