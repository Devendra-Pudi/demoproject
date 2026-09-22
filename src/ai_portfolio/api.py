"""RAG API and citation-safe event streaming with bounded concurrency and deadlines."""
import asyncio
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .rag import select_evidence
from .retrieval import build_index
from .telemetry import Telemetry

ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mode = os.getenv("RETRIEVAL_MODE", "neural")
    app.state.index = await asyncio.to_thread(build_index, ROOT / "data/docs", app.state.mode)
    app.state.telemetry = Telemetry(Path(os.getenv("TRACE_DB", "artifacts/traces.sqlite")))
    app.state.slots = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(18, connect=2)) as client:
        app.state.client = client
        yield


app = FastAPI(title="Reliable AI • Ask My Policies", lifespan=lifespan)


@app.middleware("http")
async def service_auth(request, call_next):
    # Shared service credential, not per-user auth. Keep public dashboards behind their own login.
    key = os.getenv("API_KEY", "")
    if key and request.url.path != "/health":
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + key).encode()):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000, pattern=r"\S")


@app.get("/")
async def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/health")
async def health():
    return {"status": "ready", "retrieval_mode": app.state.mode,
            "chunks": len(app.state.index.chunks)}


@app.get("/metrics")
async def metrics():
    return await asyncio.to_thread(app.state.telemetry.summary)


@app.get("/traces")
async def traces():
    return await asyncio.to_thread(app.state.telemetry.recent, 50)


async def run_question(question: str, queue: asyncio.Queue | None = None) -> dict:
    start = time.perf_counter()
    trace = {"id": str(uuid.uuid4()), "status": "ok", "spans_ms": {},
             "retrieval_mode": app.state.mode, "input_tokens": 0, "output_tokens": 0}
    chunks, citations = [], []
    answer = "No supported answer found in these policies."
    acquired = False
    try:
        try:
            await asyncio.wait_for(app.state.slots.acquire(), timeout=0.25)
            acquired = True
        except TimeoutError as exc:
            trace["status"] = "busy"
            raise HTTPException(429, "System busy; retry shortly") from exc
        stage = time.perf_counter()
        # Retrieval is bounded by a small, startup-indexed corpus; no remote calls here.
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(app.state.index.search, question),
                timeout=float(os.getenv("RETRIEVAL_TIMEOUT_S", "3")),
            )
        except TimeoutError as exc:
            trace["status"] = "retrieval_timeout"
            raise HTTPException(504, "Retrieval timed out; retry with a shorter question") from exc
        trace["spans_ms"]["retrieval_rerank"] = (time.perf_counter() - stage) * 1000
        if queue is not None:
            await queue.put({"type": "stage", "stage": "retrieved", "trace_id": trace["id"]})
        stage = time.perf_counter()
        try:
            citations, usage = await asyncio.wait_for(select_evidence(
                app.state.client, os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
                os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"), question, chunks,
            ), timeout=float(os.getenv("GENERATION_TIMEOUT_S", "20")))
            trace.update(usage)
            trace["citation_valid"] = True
            if citations:
                answer = "\n\n".join(f'{c["quote"]} [{i}]' for i, c in enumerate(citations, 1))
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            trace["status"] = "retrieval_only"
            trace["citation_valid"] = False
            answer = "Generation unavailable or evidence validation failed. Review the retrieved passages below."
        trace["spans_ms"]["generation_validation"] = (time.perf_counter() - stage) * 1000
        result = {"trace_id": trace["id"], "status": trace["status"], "answer": answer,
                  "retrieval_mode": app.state.mode,
                  "citations": citations,
                  "retrieved": [{"chunk_id": c.id, "source": c.source, "text": c.text} for c in chunks]}
        return result
    except asyncio.CancelledError:
        trace["status"] = "cancelled"
        raise
    except Exception:
        if trace["status"] == "ok":
            trace["status"] = "error"
        raise
    finally:
        if acquired:
            app.state.slots.release()
        trace["total_ms"] = (time.perf_counter() - start) * 1000
        trace["estimated_api_cost_usd"] = (
            trace["input_tokens"] * float(os.getenv("INPUT_USD_PER_MILLION", "0")) +
            trace["output_tokens"] * float(os.getenv("OUTPUT_USD_PER_MILLION", "0"))
        ) / 1_000_000
        await asyncio.to_thread(app.state.telemetry.record, trace)


@app.post("/ask")
async def ask(body: Question):
    return await run_question(body.question)


@app.post("/stream")
async def stream(body: Question):
    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        async def work():
            try:
                result = await run_question(body.question, queue)
                await queue.put({"type": "answer", **result})
            except HTTPException as exc:
                await queue.put({"type": "error", "message": exc.detail, "code": exc.status_code})
            except Exception:
                await queue.put({"type": "error", "message": "Request failed; please retry"})
            finally:
                await queue.put(None)
        task = asyncio.create_task(work())
        try:
            yield json.dumps({"type": "stage", "stage": "started"}) + "\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5)
                except TimeoutError:
                    yield json.dumps({"type": "heartbeat"}) + "\n"
                    continue
                if event is None:
                    break
                yield json.dumps(event) + "\n"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
