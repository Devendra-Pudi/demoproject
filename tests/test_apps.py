"""Model-free dashboard and Gradio contract tests (optional UI runtime required)."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("gradio")
from streamlit.testing.v1 import AppTest

PROJECTS = Path(__file__).resolve().parents[1] / "projects"


@pytest.mark.parametrize("folder", ["production-rag-application", "local-slm-ollama",
                                    "monitoring-observability", "lora-dpo-fine-tuning"])
def test_dashboard_starts_without_backend(folder):
    app = AppTest.from_file(str(PROJECTS / folder / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title


def test_rag_dashboard_submission(monkeypatch):
    from ai_portfolio import dashboard

    def answer(path, body):
        assert path == "/ask"
        assert body["question"]
        return {"trace_id": "test", "status": "retrieval_only", "answer": "Review retrieved evidence.",
                "citations": [], "retrieved": [{"source": "policy.md", "text": "Example evidence."}]}
    monkeypatch.setattr(dashboard, "api_post", answer)
    app = AppTest.from_file(str(PROJECTS / "production-rag-application/app.py")).run()
    next(button for button in app.button if button.label == "Ask my docs").click().run()
    assert not app.exception
    assert any("Retrieval-only" in warning.value for warning in app.warning)
    assert app.expander[0].label == "[1] policy.md"


def test_missing_backend_is_readable(monkeypatch):
    from ai_portfolio import dashboard

    def unavailable(*args):
        raise httpx.ConnectError("test offline")
    monkeypatch.setattr(dashboard, "api_get", unavailable)
    app = AppTest.from_file(str(PROJECTS / "monitoring-observability/app.py")).run()
    app.button[0].click().run()
    assert not app.exception
    assert "Backend unavailable" in app.error[0].value


@pytest.fixture
def voice():
    spec = importlib.util.spec_from_file_location("portfolio_voice", PROJECTS / "realtime-multimodal/app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradio_builds(voice):
    assert voice.build_app().title == "Real-Time Multimodal Application"


def test_gradio_empty_question(voice):
    results = list(voice.respond("", None, False))
    assert "Enter a question" in results[-1][0]


def test_gradio_stream_fallback(voice, monkeypatch):
    events = [{"type": "stage", "stage": "retrieved"},
              {"type": "answer", "trace_id": "test", "status": "retrieval_only",
               "answer": "Generation unavailable.", "citations": [],
               "retrieved": [{"source": "policy.md", "text": "Evidence"}]}]
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text="\n".join(json.dumps(event) for event in events)))) as client:
        monkeypatch.setattr(voice.httpx, "stream", client.stream)
        results = list(voice.respond("Annual leave?", None, True))
    assert "Generation unavailable." in results[-1][0]
    assert "policy.md" in results[-1][0]
    assert results[-1][2] is None  # no TTS for unvalidated fallback passages


def test_gradio_incomplete_stream(voice, monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=""))) as client:
        monkeypatch.setattr(voice.httpx, "stream", client.stream)
        results = list(voice.respond("Annual leave?", None, False))
    assert "Incomplete response" in results[-1][0]
