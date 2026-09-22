"""Local model streaming app and measured benchmark viewer."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import httpx
import streamlit as st

from ai_portfolio.dashboard import page, setting, uploaded_json

page("Local SLM App with Ollama", "Stream local inference and compare actual quality-versus-speed measurements.")
st.info("Offline only when this app and Ollama run locally with weights already downloaded. "
        "A cloud dashboard needs a secured, reachable inference gateway; do not expose raw Ollama publicly.")
chat, benchmark = st.tabs(["Local assistant", "Three-model benchmark"])
with chat:
    models = [m.strip() for m in setting("OLLAMA_MODELS", "qwen2.5:1.5b,llama3.2:1b,gemma2:2b").split(",") if m.strip()]
    model = st.selectbox("Installed model", models)
    prompt = st.text_area("Prompt", max_chars=4000)
    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            url = setting("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
            key = setting("OLLAMA_API_KEY")
            def generate():
                done = False
                with httpx.stream("POST", url + "/api/generate", timeout=httpx.Timeout(60, connect=3),
                                  headers={"Authorization": f"Bearer {key}"} if key else {},
                                  json={"model": model, "prompt": prompt, "stream": True,
                                        "options": {"temperature": 0, "num_predict": 512}}) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            item = json.loads(line)
                            if "error" in item:
                                raise ValueError("Inference server error")
                            yield item.get("response", "")
                            done = done or item.get("done", False)
                if not done:
                    raise ValueError("Incomplete stream")
            try:
                st.write_stream(generate())
            except (httpx.HTTPError, ValueError):
                st.error("Inference failed or timed out. Any partial output above is incomplete. "
                         "Check OLLAMA_URL, gateway credentials and installed model names.")
with benchmark:
    st.code("ai-benchmark --models qwen2.5:1.5b llama3.2:1b gemma2:2b --repeats 3")
    file = st.file_uploader("Upload the generated local-benchmark.json", type=["json"])
    if file:
        try:
            report = uploaded_json(file)
            rows = [{"model": name, **{key: data[key] for key in
                     ["p50_ms", "p95_ms", "mean_tokens_per_second", "json_valid_rate", "exact_match"]}}
                    for name, data in report["models"].items()]
            if len(rows) != 3:
                raise ValueError("Three model results required")
            st.dataframe(rows, use_container_width=True)
            st.json(report["hardware"])
            st.caption("Uploaded measurements; not independently verified by this dashboard.")
        except (ValueError, KeyError, TypeError, AttributeError):
            st.error("Invalid benchmark report. Upload output from ai-benchmark.")
    else:
        st.info("No measurements uploaded. Run the benchmark on your target hardware; no sample scores are shown.")
