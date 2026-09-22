import httpx
import pytest
from fastapi.testclient import TestClient

from ai_portfolio.api import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("RETRIEVAL_MODE", "smoke")
    monkeypatch.setenv("TRACE_DB", str(tmp_path / "traces.sqlite"))
    with TestClient(app) as client:
        yield client


def test_health_and_validation(client):
    assert client.get("/health").json()["retrieval_mode"] == "smoke"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.get("/").status_code == 200


def test_valid_answer(client, monkeypatch):
    async def selection(*args):
        chunk = args[-1][0]
        return [{"chunk_id": chunk.id, "source": chunk.source, "quote": chunk.text}], {
            "input_tokens": 20, "output_tokens": 10}
    monkeypatch.setattr("ai_portfolio.api.select_evidence", selection)
    result = client.post("/ask", json={"question": "API key rotation?"}).json()
    assert result["status"] == "ok"
    assert result["citations"]
    assert client.get("/metrics").json()["requests"] == 1
    assert "question" not in client.get("/traces").json()[0]


@pytest.mark.parametrize("error", [httpx.ConnectError("offline"), ValueError("bad citation"), TimeoutError()])
def test_fallback_stream(client, monkeypatch, error):
    async def unavailable(*args):
        raise error
    monkeypatch.setattr("ai_portfolio.api.select_evidence", unavailable)
    response = client.post("/stream", json={"question": "What is annual leave?"})
    assert response.status_code == 200
    assert '"stage": "retrieved"' in response.text
    assert '"status": "retrieval_only"' in response.text
    assert client.get("/metrics").json()["degradation_rate"] == 1


def test_generation_deadline(client, monkeypatch):
    import asyncio

    async def slow(*args):
        await asyncio.sleep(1)
    monkeypatch.setattr("ai_portfolio.api.select_evidence", slow)
    monkeypatch.setenv("GENERATION_TIMEOUT_S", "0.001")
    assert client.post("/ask", json={"question": "Annual leave policy?"}).json()["status"] == "retrieval_only"


def test_retrieval_deadline(client, monkeypatch):
    import time

    def slow(*args):
        time.sleep(0.02)
        return []
    monkeypatch.setattr(app.state.index, "search", slow)
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", "0.001")
    assert client.post("/ask", json={"question": "Annual leave policy?"}).status_code == 504
    assert client.get("/traces").json()[0]["status"] == "retrieval_timeout"


def test_overload(client):
    import asyncio

    app.state.slots = asyncio.Semaphore(0)
    assert client.post("/ask", json={"question": "Annual leave policy?"}).status_code == 429
