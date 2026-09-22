# 04 · JSON extraction with LoRA/QLoRA and DPO

**Code:** `scripts/train.py`, `evaluate_extraction.py`, `compare_extraction.py`; synthetic data in `data/training`.

Task: extract a contact into exactly `{name, email}`, using JSON null for missing email. Twelve SFT examples, twelve preference pairs and six held-out contacts are included as **pipeline fixtures**, not sufficient training/evaluation data for a quality claim. Held-out names/emails do not appear in training; templates still overlap.

## Environment and training

Use a separate Python 3.11 environment with sufficient disk space for PyTorch and model weights. A CUDA GPU is recommended; QLoRA requires supported CUDA/bitsandbytes. Start with a small 0.5B model to validate the pipeline before scaling. Dependency ranges target Transformers 4.46–4.49, PEFT .14 and TRL .15; install the declared optional extra together, not arbitrary newest training packages.

```bash
pip install -e '.[train]'
# Optional: pip install -e '.[quantization]'
python scripts/train.py --stage sft --output checkpoints/sft
# Add --qlora for 4-bit NF4 loading, otherwise standard LoRA.
python scripts/train.py --stage dpo --adapter checkpoints/sft --output checkpoints/dpo
```

SFT uses rank-16 all-linear LoRA, alpha 32, dropout .05, batch 1 with accumulation 4, seed 42 and a 512-token training limit. DPO loads the trained SFT adapter twice: a trainable **policy** and frozen **reference**. This avoids accidentally comparing DPO against the unadapted base. Beta .1, learning rate 5e-6. Both stages save adapters, tokenizer and run arguments; model weights are ignored by Git.

TRL's named policy adapter is saved in `checkpoints/dpo/policy`; evaluate that subdirectory, not the top-level DPO folder. The reference subdirectory is not the trained policy.

## Actual before/after measurements

```bash
python scripts/evaluate_extraction.py --output artifacts/base.json
python scripts/evaluate_extraction.py --adapter checkpoints/sft --output artifacts/sft.json
python scripts/evaluate_extraction.py --adapter checkpoints/dpo/policy --output artifacts/dpo.json
python scripts/compare_extraction.py artifacts/base.json artifacts/sft.json artifacts/dpo.json
```

All three runs use the same held-out file and deterministic decoding with no constrained-JSON decoding that could mask formatting errors. Metrics: parseable JSON rate, exact object match, per-field accuracy. Reports retain predictions and individual generation latencies so failures are auditable. Extra keys fail exact match; parseable scalars count as valid JSON but fail extraction metrics.

**No trained adapter or before/after scores are claimed yet.** Training was not executed in this workspace. The comparison command produces actual numeric results only after running all evaluations. DPO can regress; report those regressions rather than assuming improvement.

Before a serious experiment, add thousands of diverse consented/synthetic examples, a validation split for tuning, an untouched test split, adversarial input, nested/ambiguous entities, paired confidence intervals, baseline comparisons and multiple seeds. Record full package versions, hardware, weight revision and dataset hashes with the resulting artifacts. Do not train on private contact records without authorization.

## App entry point and deployment

This folder now contains a runnable Streamlit application, its dependency file and a non-root container recipe. It reuses the monorepo shared package rather than duplicating backend logic.

From the **repository root**:

```bash
pip install -r projects/lora-dpo-fine-tuning/requirements.txt
streamlit run projects/lora-dpo-fine-tuning/app.py --server.port 8504
```

For a container: `docker build -f projects/lora-dpo-fine-tuning/Dockerfile -t lora-dpo-fine-tuning .`

Read [DEPLOYMENT.md](../../DEPLOYMENT.md) for exact hosting steps, backend secrets, network requirements and public-production prerequisites. This folder is a monorepo deploy target, not a standalone copied directory.
