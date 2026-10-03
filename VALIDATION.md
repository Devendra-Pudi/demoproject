# Validation record

Latest run: **2026-10-03**, Python 3.11.2, single workspace CPU, `RETRIEVAL_MODE` as noted. Earlier
runs are kept below so the history stays auditable.

## 2026-10-03 — recruiter-readiness pass

| Check | Command | Result |
|---|---|---|
| Static checks | `ruff check .` | Passed |
| Test suite | `pytest -q` | **87 passed**, 1 upstream deprecation warning (was 35 passed + 1 skipped before the UI dependencies were installed) |
| Deterministic gate | `ai-eval --mode smoke --baseline data/eval/smoke-baseline.json` | Passed — Recall@4 **1.0**, MRR **0.964**, citation invariants **1.0** |
| Fixtures | `data/eval/rag.jsonl` | **18** cases: 14 answerable, 4 abstention (was 8) |
| Corpus | `data/docs` | **21** documents → **63** chunks (was 3 → 9) |
| Retrieval latency | smoke doubles | p50 **0.364 ms**, p95 **0.468 ms** (not a hardware SLO) |
| Live HTTP (demo) | `RETRIEVAL_MODE=demo uvicorn ...` + `curl` | `/health`, `/ask` (status `ok`, `generation_mode: demo`, validated quote), `/stream` (stages + answer), `/metrics` (`generation_modes` breakdown), `/` → all 200 |
| Live demo abstention | `curl` with an uncurated question | `retrieval_only` + explicit "no curated demo answer" message |
| Streamlit startup | `streamlit run projects/production-rag-application/app.py` | `/_stcore/health` → 200 |
| Pages | `gh api repos/.../pages` | Live; all five cards render with structure previews |
| Docker build | — | **Not run**: no Docker daemon in this workspace |
| Neural retrieval, Ollama benchmarks, LoRA/DPO training, browser audio | — | **Not run**: no model weights, GPU or browser |

### What changed in this pass (and what it did to the numbers)

1. **Corpus expanded 3 → 21 synthetic policy documents** and fixtures 8 → 18. The reviewed smoke
   baseline in `data/eval/smoke-baseline.json` was regenerated for the new corpus/dataset — the
   previous baseline is not comparable to it. This is a corpus change, not a gate being relaxed to
   pass: the new run was inspected case by case.
2. **Light plural normalization added to `tokens()`** (`password` ↔ `passwords`, `policy` ↔
   `policies`). The first run on the expanded corpus failed Recall@4 (0.929) because the fixture
   "minimum password length" could not match "Passwords"; instead of editing the fixture to suit the
   retriever, the retriever was fixed, and Recall@4 returned to **1.0** with MRR 0.964. `tests/test_retrieval.py`
   now covers the normalization, including words it must leave alone (`access`, `class`, `loss`).
3. **Demo mode (`RETRIEVAL_MODE=demo`)** added so the flagship app returns citation-validated answers
   with no model download. Curated fixtures are re-checked with the production `enforce_citations()`
   invariant; `tests/test_demo.py` (7 cases) fails if a quote stops matching the shipped documents.
4. **Dashboards 02–04 keep their no-sample-results policy**: the empty states now explain exactly
   which columns a real run produces, plus the commands and `ollama pull` steps needed to fill them.
   No placeholder metric was introduced anywhere.
5. **Project 05 speech defaults fixed**: the Docker image installs faster-whisper + eSpeak and now
   sets `ENABLE_LOCAL_STT=1` (Compose default matches), and the page states which mode is active.
6. **Branch references corrected** to `main` in `site/app.js`, `site/index.html` and `DEPLOYMENT.md`;
   the footer PR link was replaced with the repository link.

Generation quality, neural retrieval, real benchmarks and training remain **unmeasured**. No
placeholder value anywhere in this repository is presented as a measurement.

## 2026-09-22 — app/deployment-layout update

- Renamed the five guide folders into five named deploy targets, each with an app entrypoint, requirements and Dockerfile.
- Installed Streamlit 1.50.0 and Gradio 5.49.1 and ran all **45 tests successfully** (one upstream AnyIO deprecation warning).
- Verified startup of all four Streamlit dashboards using Streamlit AppTest; tested RAG submission, backend-unavailable messaging, Gradio construction, stream fallback and incomplete-stream handling without models.
- Added tested service-key protection (public healthcheck only when a key is configured), finite-metric gate validation, dataset/model matching for extraction reports and deployment-file checks.
- Ruff and deterministic smoke baseline gates pass.
- Docker is not installed in this workspace; Dockerfiles/Compose are provided but images were **not built here**. Local Whisper/eSpeak audio and GPU training remain untested.

## Earlier run — 2026-09-22 baseline

| Check | Result |
|---|---|
| `ruff check .` | Passed |
| `pytest -q` | 23 passed; 2 upstream Starlette/AnyIO deprecation warnings |
| `python -m compileall -q src scripts` | Passed |
| `ai-eval --mode smoke --baseline ...` | Passed all gates |
| Recall@4 / MRR (eight fixtures, smoke doubles) | 1.0 / 1.0 |
| Citation invariant checks | 1.0 |
| Baseline retrieval p50 / p95 | 0.203 / 0.423 ms |
| Live HTTP `/health` | Ready; smoke mode, nine chunks |
| Live HTTP `/stream`, Ollama unavailable | Started → retrieved → labeled retrieval-only fallback |
| Neural retrieval + cross-encoder | Not run: CPU PyTorch download failed with TLS EOF at download.pytorch.org |
| Ollama three-model benchmark | Not run: Ollama not installed |
| LoRA/QLoRA + DPO training | Not run: no training runtime/weights or NVIDIA tooling |
| Real-model generation quality | Not measured |
| Browser microphone/TTS | Not manually audio-tested; provider/browser dependent |

The test suite mocks model responses where necessary. It does not imply actual model, GPU, microphone or cross-browser compatibility. Generation abstention cases are only scored by `ai-eval --generate`, not the smoke CI job.

Core/test versions are pinned in `requirements-ci.txt`. Optional model dependencies are declared in `pyproject.toml`; their execution is a separate validation step. TRL 0.15.2 argument names were inspected for the SFT/DPO configuration, but that is not a substitute for training.

## Not verified anywhere in this repository

- Real-model answer quality, hallucination rate or abstention behaviour under `--generate`.
- Three-model benchmark results on any hardware.
- Any fine-tuning improvement; the shipped training data is 12 SFT examples, 12 preference pairs and 6 held-out contacts.
- Container builds (`docker build`), Compose stacks, hosted deployments (Streamlit Community Cloud, Hugging Face Spaces) and HTTPS/microphone behaviour in a real browser.
- Load, concurrency or capacity behaviour: four admitted backend requests and a two-worker Gradio queue are reference defaults, not tested limits.
