# 05 · Citation-safe real-time voice assistant

**Code:** `web/index.html`, `/stream` in `api.py`.

Start the server using the root quick start. Open the browser UI, choose **Use voice**, grant microphone permission, review the transcript and submit. Enable **Read validated answers aloud** for speech output. Typed interaction always remains available. Speech recognition availability varies by browser and typically requires HTTPS or localhost; the proxied preview uses HTTPS.

## Stream contract

`POST /stream` emits newline-delimited JSON with immediate `stage: started`, then `stage: retrieved`, a heartbeat every five seconds while waiting, and a validated `answer` or `error` event. The browser decodes partial network chunks safely and cancels on **Stop** or a 30-second deadline. Server disconnect cancellation propagates to the model HTTP request.

This is **progress-event streaming, not unchecked LLM token streaming**. For this use case, evidence safety takes priority over speaking the first generated token. Only the completed, validated answer is sent to speech synthesis. The local SLM CLI separately demonstrates raw model token streaming.

## Latency budget (design targets, not measured SLOs)

| Stage | Budget / behavior | Measurement |
|---|---|---|
| Microphone + speech recognition | Provider/browser dependent; outside server deadline | Recognition start → final transcript in UI |
| User review | Human-controlled, excluded from service SLO | Not included |
| Admission queue | 250 ms maximum, four requests admitted per process | Included in total trace latency |
| Hybrid retrieval + reranking | 3 s response deadline; warm target < 500 ms | `retrieval_rerank` span |
| Model generation + citation validation | 20 s hard async wait deadline; warm target < 5 s | `generation_validation` span |
| Response transport/render | Target < 200 ms | Submit → validated answer in browser |
| Speech synthesis start | Browser/provider dependent; target < 500 ms | Submit → `first_audio` in UI |
| Whole browser request | 30 s deadline | AbortController |

The budget deliberately distinguishes targets, configured deadlines and actual measured browser/server spans. Network/render latency cannot be accurately derived by subtracting unsynchronized clocks; browser marks and server spans use independent monotonic clocks. Voice transcription and user review are outside the submitted-request deadline.

## Failure behavior

- Missing microphone/unsupported speech recognition → typed input.
- Recognition provider error → readable error; text remains available.
- Ollama offline/timeout/invalid evidence → retrieved passages explicitly labeled as fallback; no automatic speech of unvalidated passages.
- Retrieval deadline → 504 on `/ask`, structured error inside an already-open `/stream` (HTTP status is then 200).
- Overload → 429 on `/ask`, code 429 in stream error event.
- Stop/disconnect → cancel waiting and speech synthesis. Native retrieval kernels may continue in a thread after deadline; use isolated worker processes for hard compute cancellation at scale.

Browser speech services **may send audio/text to external providers**. The UI discloses this before microphone use. This is a multimodal browser demo, not a guaranteed offline voice stack. For sensitive deployment, replace Web Speech with self-hosted streaming STT/TTS, add VAD, bounded audio queues, provider deadlines and audio-level cancellation tests.

## App entry point and deployment

Gradio adds an independently deployable voice/text client. Optional faster-whisper provides server-local STT; eSpeak provides offline TTS. The earlier browser Web Speech client remains available on the backend `/` route when service authentication is disabled for local use.

From the **repository root**:

```bash
pip install -r projects/realtime-multimodal/requirements.txt
python projects/realtime-multimodal/app.py
```

For a container: `docker build -f projects/realtime-multimodal/Dockerfile -t realtime-multimodal .`

Read [DEPLOYMENT.md](../../DEPLOYMENT.md) for exact hosting steps, backend secrets, network requirements and public-production prerequisites. This folder is a monorepo deploy target, not a standalone copied directory.
