# Reliable AI Portfolio

Five connected, runnable project implementations demonstrating retrieval, local inference, observability, model adaptation, and real-time interaction. The domain is **employee policy assistance** with synthetic policy documents; extraction training uses fictional contacts.

> **Execution status:** the model-free suite and smoke evaluation have been run. Neural retrieval, three-model Ollama benchmarks, and LoRA/DPO training require model downloads/hardware and have **not** been run in this workspace. Their scripts are supplied, not claimed results. This is a portfolio reference implementation, not a hardened enterprise deployment.

## The five projects

| Project | Implementation | Guide |
|---|---|---|
| 01 · Production-pattern RAG | BM25 + normalized sentence vectors → RRF → cross-encoder → citation-locked answer | [RAG](projects/01_rag/README.md) |
| 02 · Local SLM application | Offline Ollama CLI; three-model same-host speed/JSON-quality benchmark | [Local models](projects/02_local_slm/README.md) |
| 03 · Monitoring & observability | SQLite request/span traces, p50/p95, cost estimates, quality/regression CI gates | [Observability](projects/03_observability/README.md) |
| 04 · Fine-tuning | Contact JSON extraction; LoRA/QLoRA SFT → DPO with frozen SFT reference; held-out evaluator | [Fine-tuning](projects/04_finetuning/README.md) |
| 05 · Real-time multimodal | Browser speech input/output + NDJSON progress/answer stream, deadlines and text fallback | [Real-time](projects/05_realtime/README.md) |

## Quick start: no model required

Python 3.11+; commands run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
pip install --no-deps -e .
pytest -q
ai-eval --mode smoke --baseline data/eval/smoke-baseline.json
RETRIEVAL_MODE=smoke uvicorn ai_portfolio.api:app --host 0.0.0.0 --port 8000
```

Open port 8000. Without Ollama, the UI deliberately shows **retrieval-only fallback**. Smoke mode uses lexical hash vectors and an overlap reranker, **not semantic embeddings or a cross-encoder**. It is visibly labeled in the UI and reports.

## Full neural RAG

Install [Ollama](https://ollama.com/) separately, start its local service, then:

```bash
pip install -e '.[retrieval]'
ollama pull qwen2.5:1.5b
RETRIEVAL_MODE=neural OLLAMA_MODEL=qwen2.5:1.5b \
  uvicorn ai_portfolio.api:app --host 0.0.0.0 --port 8000
ai-eval --mode neural --generate --output artifacts/full-rag-eval.json
```

First neural startup downloads `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. After caching weights, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for disconnected execution. Model licenses and hardware requirements are separate from this source repository.

## Architecture

```text
Browser text / speech → FastAPI /stream → bounded request admission
                                        → BM25 + dense → RRF → cross-encoder
                                        → local Ollama evidence selection
                                        → exact-quote + source-ID validation
                                        → answer / retrieval-only fallback
                       each request → SQLite content-free spans → /metrics
JSON fixtures → CI gates / neural evaluation / model benchmarks
Contact data → LoRA SFT → DPO against SFT reference → held-out comparison
```

API: `GET /health`, `POST /ask`, `POST /stream`, `GET /metrics`, `GET /traces`; schema at `/docs`. Browser fetches use same-origin relative URLs, including proxied live previews.

```bash
curl -s http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"When must production API keys be rotated?"}'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RETRIEVAL_MODE` | `neural` | `neural` or explicitly labeled `smoke` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Server-side inference address; never sent to browser |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Installed Ollama model |
| `TRACE_DB` | `artifacts/traces.sqlite` | Content-free local trace database |
| `RETRIEVAL_TIMEOUT_S` | `3` | Retrieval response deadline |
| `GENERATION_TIMEOUT_S` | `20` | Generation/validation response deadline |
| `INPUT_USD_PER_MILLION` | `0` | Estimated input API token price |
| `OUTPUT_USD_PER_MILLION` | `0` | Estimated output API token price |

API requires restarting to reindex document changes. Defaults are intended for a small single-process deployment. Environment variables are server configuration, not user-editable API inputs.

## Verification and results

- `ruff check .` — static checks.
- `pytest -q` — retrieval, citation rejection, deadlines, fallback, telemetry, benchmark and gate tests.
- `ai-eval --mode smoke --baseline data/eval/smoke-baseline.json` — deterministic CI gate.
- GitHub **Reliability gates** runs on pushes/PRs and uploads its evaluation JSON.
- GitHub **Full neural retrieval gate** is manually dispatched; it downloads neural retrieval weights. It does not evaluate generation without an Ollama service.
- [Measured smoke baseline](data/eval/smoke-baseline.json): Recall@4 **1.0**, MRR **1.0**, citation invariants **1.0**, p95 retrieval **0.423 ms** on this workspace. Six answerable and two unanswerable fixtures. These tiny-fixture numbers do not establish production quality; generation was not measured.
- LoRA, DPO and model benchmark numbers are intentionally absent until actual runs generate reports. See each project guide.

## Limits and deployment checklist

- No authentication, tenant isolation, ACL-filtered retrieval, rate limiting by identity, or TLS termination is implemented. Use only synthetic/public documents in a preview. Add these before any sensitive deployment; `/traces` and `/metrics` also need access control.
- Quotes prove provenance, **not relevance or completeness**. The evidence-selection model may choose irrelevant quotes; use held-out domain evaluations and human review.
- SQLite traces have no automatic retention/rotation, cross-service propagation, or OpenTelemetry export. Add them for a multi-service deployment. Questions and documents are not recorded in traces; local benchmark artifacts do contain model outputs.
- In-memory exact dense search is for small corpora. Large deployments need persisted indexes, approximate nearest neighbors, incremental updates and document-level ACLs.
- Python deadlines stop waiting, not native CPU/GPU kernels. Hard cancellation requires worker-process isolation. Browser speech engines may use third-party services; voice is **not guaranteed offline**.
- Training data is deliberately small and synthetic. Expand it and reserve independent validation/test splits before asserting generalization or meaningful fine-tuning gains.

Model weights, checkpoints, traces and generated reports live in ignored directories. No secrets or model binaries are committed.
