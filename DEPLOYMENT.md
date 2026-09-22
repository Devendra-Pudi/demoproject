# Deploy AI Portfolio Projects

This is a **monorepo with five app folders**, not five disconnected copies. Each project owns its UI, Dockerfile, requirements and guide; reusable inference/retrieval/telemetry lives in `src/ai_portfolio`. Always use the **repository root as the working directory and Docker build context**. Do not copy out only a project folder and expect imports/data to work.

## Choose a deployment

| Folder under `projects/` | App entry point | Hosting | External dependency |
|---|---|---|---|
| `production-rag-application` | `app.py` (Streamlit) | Streamlit Community Cloud / Docker | Shared FastAPI service + optional Ollama |
| `local-slm-ollama` | `app.py` (Streamlit) | Local machine recommended / Docker | Ollama with downloaded models |
| `monitoring-observability` | `app.py` (Streamlit) | Streamlit Community Cloud / Docker | Shared FastAPI service |
| `lora-dpo-fine-tuning` | `app.py` (Streamlit) | Streamlit Community Cloud / Docker | GPU worker only for training; report viewer is standalone |
| `realtime-multimodal` | `app.py` (Gradio) | Docker / Hugging Face Docker Space | Shared FastAPI service; optional local STT |

GitHub Pages serves `site/` as a static project directory. It **cannot execute Streamlit, Gradio, Ollama, training or the API**. No live application URL is claimed until you deploy one and set its `url` in `site/projects.json`.

## A. Local development, all apps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install -r requirements-apps.txt -r projects/realtime-multimodal/requirements.txt
# Terminal 1: no downloaded models necessary for a clearly labeled retrieval-only demo
RETRIEVAL_MODE=smoke uvicorn ai_portfolio.api:app --host 0.0.0.0 --port 8000
# Terminal 2: choose one dashboard (or run each in a terminal on a different port)
streamlit run projects/production-rag-application/app.py --server.port 8501
streamlit run projects/local-slm-ollama/app.py --server.port 8502
streamlit run projects/monitoring-observability/app.py --server.port 8503
streamlit run projects/lora-dpo-fine-tuning/app.py --server.port 8504
python projects/realtime-multimodal/app.py  # port 7860
```

All app servers listen on 0.0.0.0. Browser interaction uses the hosting framework's own origin; backend HTTP calls are made server-side. The dashboard won't contact a backend until you submit or refresh. For neural RAG, install the retrieval extra and start the backend with `RETRIEVAL_MODE=neural`; see the RAG guide. Nothing downloads Ollama weights silently.

## B. Docker Compose

```bash
cp .env.example .env
# Replace API_KEY in .env with a strong random value; do not commit it.
docker compose up --build -d
# Download models explicitly, after the Ollama service starts:
docker compose exec ollama ollama pull qwen2.5:1.5b
# Optional: llama3.2:1b and gemma2:2b for the benchmark
```

Dashboards: localhost ports 8501–8504, Gradio: 7860. Compose publishes dashboard ports to loopback only; API/Ollama remain on the internal network. On a remote host use an authenticated TLS reverse proxy and explicitly adjust published bindings; never publish raw Ollama. These Compose defaults use smoke retrieval and CPU Ollama, not a GPU performance claim.

To enable neural retrieval, set `INSTALL_NEURAL=true` and `RETRIEVAL_MODE=neural` in `.env`, then rebuild. First startup downloads model weights and may exceed initial healthcheck grace time. Add persistent model cache volumes for long-term deployments. The backend trace volume and Ollama weights already persist. Four admitted backend requests and a two-worker Gradio queue are reference defaults, not tested capacity guarantees.

To enable local voice recognition, set `ENABLE_LOCAL_STT=1`; the Gradio image includes faster-whisper and eSpeak. The first transcription downloads `tiny.en` unless cached. Text input works without it. Transcription/audio input is capped at 30 seconds and upload size at 5 MB. Provision/cache STT models before using offline.

## C. Streamlit Community Cloud (projects 1–4)

1. Sign in to Streamlit Community Cloud with GitHub and select **Create app**.
2. Choose `Devendra-Pudi/demoproject` and branch `arena/01a0c7ca-demoproject` (or `main` after merging the PR).
3. Set the main file to the relevant `projects/<folder>/app.py` from the table. Choose Python **3.11**.
4. Streamlit finds `requirements.txt` beside that entry point. It includes the root `requirements-apps.txt`; no neural packages load on the dashboard host.
5. In advanced settings/secrets configure the service connection, for example:

```toml
API_BASE_URL = "https://your-private-api.example"
API_KEY = "a-long-random-service-key"
```

For the local-model UI, use `OLLAMA_URL`, optional `OLLAMA_API_KEY` for your authenticated gateway, and optional comma-separated `OLLAMA_MODELS`. **Cloud localhost is not your laptop.** Prefer local deployment for offline/privacy guarantees. The fine-tuning report dashboard needs no backend secrets.

6. Restrict app sharing/access. A server-to-server API key does **not** authenticate dashboard users; any dashboard visitor can otherwise use its configured backend.
7. Deploy, exercise successful and failed requests, and put the resulting HTTPS app URL in `site/projects.json`.

Secrets may be stored in `.streamlit/secrets.toml` locally (ignored by Git). Never put real keys in source files. Gradio uses environment variables instead of Streamlit secrets.

## D. Backend and Gradio containers

Build from the repository root:

```bash
docker build -f projects/production-rag-application/Dockerfile.api -t portfolio-api .
docker build -f projects/realtime-multimodal/Dockerfile -t portfolio-voice .
```

Run them on a private Docker network or on a container host with internal service networking. Configure the API with `API_KEY`, `RETRIEVAL_MODE` and a reachable `OLLAMA_URL`; configure Gradio with `API_BASE_URL` and the same `API_KEY`. API listens on **8000**, Gradio on **7860**. Add TLS and user authentication at the hosting boundary. Containers use non-root users and healthchecks; uploaded audio is held in Gradio temporary storage with periodic cleanup, not intentionally committed to the repository.

For a Hugging Face **Docker Space**, mirror the repository into the Space, place the contents of `projects/realtime-multimodal/Dockerfile` at the Space root as `Dockerfile`, and configure its README metadata with `sdk: docker` and `app_port: 7860`. Set the backend URL/key in Space variables/secrets. Choose private access for sensitive data. This does not deploy the backend automatically; the backend must be separately reachable. Enable STT only with sufficient CPU/RAM. Free-tier resources are not a production SLO.

For training, use `scripts/train.py` on a separately provisioned CUDA machine/job runner. Streamlit is a report UI, not an unbounded training job endpoint. See the fine-tuning guide for actual training/evaluation commands.

## E. GitHub Pages portfolio

1. In GitHub repository **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
2. Merge the Pages workflow into `main`, or once available on the default branch, manually dispatch **Deploy portfolio directory to Pages**, selecting the desired branch.
3. On merge, changes to `site/` on `main` publish automatically. The workflow uses a `github-pages` environment; approve it if your repository requires approval.
4. Use the URL shown by the deployment. Expected repository-site URL: `https://devendra-pudi.github.io/demoproject/` (not asserted live until deployment succeeds).
5. Add actual HTTPS app URLs to `site/projects.json` and push. Unconfigured cards show a source link and an explicit “Deployment URL not configured” message.

## Public production checklist — not completed merely by deploying

- User login, authorization and per-document ACLs; the shared API key is service protection only.
- HTTPS, rate limiting per identity, request quotas, dependency/image vulnerability scans and secret rotation.
- Run real neural/generation evaluations, three-model hardware benchmarks and training; retain reproducible artifacts.
- Load-test warm and cold starts, GPU/CPU saturation, bounded queues and native-kernel cancellation behavior.
- Configure durable traces, retention, storage backups and model caches; managed hosts may have ephemeral disks.
- For microphone use, disclose audio processing/storage and use HTTPS. Delete temporary audio according to your policy.
- Streamlit/Gradio package versions are pinned; optional model dependencies need full target-environment validation.

The app entrypoints and UI smoke tests can be tested without model downloads. Container recipes and hosted deployments must still be tested on a Docker/hosting account. There is no claim that these configurations alone make the systems enterprise-ready.
