from ai_portfolio.evaluation import check_gates
from ai_portfolio.telemetry import Telemetry


def test_regression_gate():
    baseline = {"mode": "smoke", "recall_at_4": 1, "mrr": 1,
                "citation_invariant_pass_rate": 1, "retrieval_p95_ms": 100}
    assert not check_gates(baseline, baseline)
    assert check_gates({**baseline, "mrr": 0.9}, baseline)
    assert check_gates({**baseline, "retrieval_p95_ms": 200}, baseline)
    assert check_gates({**baseline, "citation_invariant_pass_rate": 0.99})


def test_metrics(tmp_path):
    store = Telemetry(tmp_path / "traces.sqlite")
    assert store.summary()["p95_ms"] is None
    for i, ms in enumerate([10, 20, 30, 40]):
        store.record({"id": str(i), "total_ms": ms, "status": "ok", "citation_valid": True,
                      "estimated_api_cost_usd": 0.01})
    metrics = store.summary()
    assert metrics["p50_ms"] == 25
    assert metrics["p95_ms"] == 38.5
    assert metrics["mean_cost_per_request_usd"] == 0.01
    assert metrics["citation_valid_rate"] == 1


def test_non_finite_metrics_fail_closed():
    report = {"mode": "smoke", "recall_at_4": float("nan"), "mrr": 1,
              "citation_invariant_pass_rate": 1, "retrieval_p95_ms": 100}
    assert check_gates(report)
    assert check_gates({**report, "recall_at_4": 1, "generation": {"evidence_match_rate": float("inf")}})
