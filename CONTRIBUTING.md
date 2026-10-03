# Contributing

This is a portfolio monorepo, so the bar is simple: every change must keep the honesty guarantees
and the gates green.

## Before opening a pull request

```bash
pip install -r requirements-ci.txt && pip install --no-deps -e .
ruff check .
pytest -q
ai-eval --mode smoke --baseline data/eval/smoke-baseline.json
```

Optional UI work also needs:

```bash
pip install -r requirements-apps.txt -r projects/realtime-multimodal/requirements.txt
pytest tests/test_apps.py tests/test_telemetry.py tests/test_deployment.py -q
```

## Rules that are not negotiable

1. **No fabricated results.** Never commit a number that was not produced by a command you can name,
   in an environment you can describe. The dashboards ship no sample metrics on purpose: an empty
   state that explains what will appear is correct, a plausible-looking table is not.
2. **Quotes stay verbatim.** Answer text is only ever exact substrings of retrieved evidence, and
   every model-provided `chunk_id` is validated against the retrieved set. Do not relax
   `enforce_citations()`.
3. **Labeled fallbacks.** If retrieval, generation or validation fails, the response and the UI must
   say so. Never silently substitute a plausible answer.
4. **Traces stay content-free.** No questions, answers or document text in `artifacts/` traces.
5. **Baselines are reviewed, not refreshed.** A failing gate is investigated first; corpus or dataset
   changes must be recorded in `VALIDATION.md` together with the new measured values.
6. **Secrets never enter the repository.** Use environment variables or `.streamlit/secrets.toml`
   (ignored by Git).

## Commit and PR style

Conventional-commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `chore:`) with a short body that
states what was measured and what was not.
