# 02 · Offline SLM assistant and three-model benchmark

**Code:** `src/ai_portfolio/local.py`, `benchmark.py`.

```bash
# Download while connected. Then disconnect to verify offline inference.
ollama pull qwen2.5:1.5b
ollama pull llama3.2:1b
ollama pull gemma2:2b
python -m ai_portfolio.local 'Explain why hardware security keys resist phishing.'
ai-benchmark --models qwen2.5:1.5b llama3.2:1b gemma2:2b --repeats 3
```

Reports go to `artifacts/local-benchmark.json`. Ollama must already be running; the application never silently downloads a missing model. CLI inference is streamed, deterministic-temperature and capped at 512 output tokens. Benchmark inference caps output at 128 with a fixed 2,048-token context and seed 42.

## Protocol

- One process, one host, sequential model runs. Stop unrelated workloads and unload preexisting models before starting.
- Store host CPU/GPU/RAM, Ollama version, model details/quantization, timestamp and dataset hash.
- One recorded warmup request per model; exclude it from aggregate performance. It is **not necessarily a cold start** if weights are already resident or filesystem-cached.
- Run each held-out contact-extraction prompt three times. Unload each model afterward to reduce VRAM overlap.
- Record client-observed time to first token, end-to-end latency, Ollama decode tokens/sec, load time, JSON validity and exact match. Report latency p50/p95 and mean decode speed.
- Errors/incomplete streams fail the benchmark rather than generating plausible-looking metrics.

## Interpreting the tradeoff

Do not assume larger models are always better. Compare exact match against p95 and TTFT under the same quantization, context and hardware. Repeated three-case fixtures measure repeatability, not broad capability. Add many unseen extraction, reasoning and refusal cases; include confidence intervals and memory/power measurement for a rigorous study. Weights may have different licenses.

**Actual three-model results: not yet measured.** This workspace has neither Ollama nor a detected NVIDIA toolchain. Run the command on your target machine before claiming quality/speed rankings. API fees are zero for local inference, but hardware amortization, energy, memory and operational labor are not. Inputs remain local when using a local Ollama URL; third-party/browser voice recognition is outside that guarantee.
