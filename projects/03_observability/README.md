# 03 · Monitoring, traces and regression gating

**Code:** `telemetry.py`, instrumented `api.py`, `evaluation.py`, `.github/workflows`.

Every request receives a UUID and content-free SQLite trace, including overload, timeout, fallback, cancellation and unexpected failures. Spans measure retrieval/reranking and generation/validation separately; total latency includes admission wait. Traces record retrieval mode, token counts and validation status, not prompts or evidence text.

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/traces
ai-eval --mode smoke --baseline data/eval/smoke-baseline.json
pytest tests/test_evaluation.py tests/test_api.py -q
```

The dashboard and `/metrics` summarize the last **1,000** stored requests: p50/p95 latency, degradation rate, valid-citation rate for successful requests, estimated aggregate API cost and mean cost/request. `/traces` returns the latest 50; no per-trace content is retained. An abstention has valid (empty) citations; this metric must not be mistaken for answer accuracy.

Cost formula: `(input_tokens × input_rate + output_tokens × output_rate) / 1e6`. Rates come from environment variables. Defaults are zero for Ollama. Failed requests without usage information may incur unreported computation; this is an estimate, not billing data. Local hardware cost is not modeled.

## Regression workflow

CI installs pinned core/test dependencies, lints, tests and compares deterministic evaluation to the reviewed smoke baseline. Artifacts are uploaded even when the gate fails. A separate manual workflow uses actual neural retrieval. Dataset evidence match and abstention metrics require `ai-eval --generate` with Ollama; no online quality metric substitutes for those tests.

Never refresh a baseline simply to make a regression pass. Review the failing cases, root cause and environment; then approve intentional tradeoffs. Configure branch protection to require the Reliability gates check in GitHub settings (not automatically changed here).

## Production extension

SQLite/WAL is appropriate for this single-host demo. Add retention, backups, access control, sampling, OpenTelemetry export, trace propagation, dashboards and alerts for distributed production. Suggested alerts: fallback rate > 5% over five minutes; p95 beyond your tested SLO; rising citation failures. These are starting targets, not measured guarantees. Test concurrent load before sizing workers and admission limits.
