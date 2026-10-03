## What changed

<!-- One or two sentences. -->

## Evidence

- [ ] `ruff check .` passes
- [ ] `pytest -q` passes (state the count)
- [ ] `ai-eval --mode smoke --baseline data/eval/smoke-baseline.json` passes
- [ ] If results changed, the measured values are recorded in `VALIDATION.md`

## Honesty checklist

- [ ] No invented metrics; new example fixtures are labeled `"measured": false`
- [ ] Failure paths still produce labeled fallbacks
- [ ] Traces remain content-free
- [ ] No secrets, weights or generated artifacts added to Git

## Not verified in this change

<!-- Anything you could not run: GPU training, Docker builds, browser audio, hosted deploys. -->
