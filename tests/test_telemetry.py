"""Trace summarization is shared by the live /metrics endpoint and the dashboard."""
from ai_portfolio.telemetry import summarize

TRACES = [
    {"id": "a", "status": "ok", "total_ms": 100.0, "generation_mode": "model",
     "citation_valid": True, "estimated_api_cost_usd": 0.002},
    {"id": "b", "status": "retrieval_only", "total_ms": 300.0, "generation_mode": "unavailable",
     "citation_valid": False, "estimated_api_cost_usd": 0.0},
    {"id": "c", "status": "ok", "total_ms": 200.0, "generation_mode": "demo",
     "citation_valid": True, "estimated_api_cost_usd": 0.001},
]


def test_empty_window_reports_no_samples():
    summary = summarize([])
    assert summary["requests"] == 0
    assert summary["p50_ms"] is None and summary["p95_ms"] is None
    assert summary["degradation_rate"] is None


def test_summary_aggregates_latency_degradation_and_modes():
    summary = summarize(TRACES)
    assert summary["requests"] == 3
    assert summary["p50_ms"] == 200.0
    assert round(summary["degradation_rate"], 4) == round(1 / 3, 4)
    assert summary["citation_valid_rate"] == 1.0
    assert summary["generation_modes"] == {"demo": 1, "model": 1, "unavailable": 1}
    assert round(summary["estimated_api_cost_usd"], 4) == 0.003
