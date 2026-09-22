"""Experiment dashboard. Training is an explicit offline job, never arbitrary web execution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import streamlit as st

from ai_portfolio.dashboard import page, uploaded_json
from ai_portfolio.experiments import compare_reports

page("Fine-Tuning with LoRA & DPO", "Run reproducible extraction experiments and inspect actual before-and-after results.")
st.warning("Training needs a separately provisioned worker. This dashboard does not train models on Streamlit Cloud.")
setup, compare = st.tabs(["Run an experiment", "Compare measured results"])
with setup:
    st.markdown("**Task:** extract a contact into JSON with `name` and `email`. "
                "The included data is synthetic and too small to support a generalization claim.")
    st.code("""pip install -e '.[train]'
python scripts/train.py --stage sft --output checkpoints/sft
python scripts/train.py --stage dpo --adapter checkpoints/sft --output checkpoints/dpo
python scripts/evaluate_extraction.py --output artifacts/base.json
python scripts/evaluate_extraction.py --adapter checkpoints/sft --output artifacts/sft.json
python scripts/evaluate_extraction.py --adapter checkpoints/dpo/policy --output artifacts/dpo.json""", language="bash")
    st.caption("Optional QLoRA: install the quantization extra and pass --qlora on a supported CUDA worker.")
with compare:
    uploads = [st.file_uploader(f"{stage} evaluation JSON", type=["json"], key=stage)
               for stage in ["Base", "SFT", "DPO"]]
    if all(uploads):
        try:
            rows = compare_reports([uploaded_json(file) for file in uploads])
            st.dataframe(rows, use_container_width=True)
            st.bar_chart(rows, x="stage", y="exact_match")
            st.caption("Measured report values, not independently rerun here. Regressions are shown unchanged.")
        except (ValueError, KeyError, TypeError):
            st.error("Reports must contain valid metrics and the same model, held-out dataset hash and sample count. "
                     "Regenerate older reports using the current evaluation script.")
    else:
        st.info("Upload all three evaluation reports. No placeholder improvement numbers are used.")
