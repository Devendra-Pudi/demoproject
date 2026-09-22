# Validation record

Executed in the development workspace on 2026-09-22, Python 3.11.2.

| Check | Result |
|---|---|
| `ruff check .` | Passed |
| `pytest -q` | 23 passed; 2 upstream Starlette/AnyIO deprecation warnings |
| `python -m compileall -q src scripts` | Passed |
| `ai-eval --mode smoke --baseline data/eval/smoke-baseline.json` | Passed all gates |
| Recall@4 / MRR (six answerable fixtures, smoke doubles) | 1.0 / 1.0 |
| Citation invariant checks | 1.0 |
| Baseline retrieval p50 / p95 | 0.203 / 0.423 ms; not a hardware-independent SLO |
| Live HTTP `/health` | Ready; smoke mode, nine chunks |
| Live HTTP `/stream`, Ollama unavailable | Started → retrieved → labeled retrieval-only fallback |
| Neural retrieval + cross-encoder | Not run: attempted CPU PyTorch download failed with TLS EOF at download.pytorch.org |
| Ollama three-model benchmark | Not run: Ollama not installed |
| LoRA/QLoRA + DPO training | Not run: no training runtime/weights or detected NVIDIA tooling |
| Real-model generation quality | Not measured |
| Browser microphone/TTS | Not manually audio-tested; provider/browser dependent |

The test suite mocks model responses where necessary. It does not imply actual model, GPU, microphone, or cross-browser compatibility. Generation abstention cases are only scored by `ai-eval --generate`, not the smoke CI job. No model benchmark or fine-tuning improvement numbers have been fabricated.

Core/test versions are pinned in `requirements-ci.txt`. Optional model dependencies are declared in `pyproject.toml`; their execution is a separate validation step. The TRL 0.15.2 configuration definitions were inspected for the SFT/DPO argument names, but that is not a substitute for training.
