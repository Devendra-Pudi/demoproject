import json

import httpx
import pytest

from ai_portfolio.benchmark import measure


def test_stream_measurement():
    def handler(request):
        body = json.loads(request.content)
        assert body["options"]["temperature"] == 0
        events = [{"response": '{"name":"Ada"}', "done": False},
                  {"done": True, "eval_count": 10, "eval_duration": 500_000_000}]
        return httpx.Response(200, text="\n".join(json.dumps(e) for e in events))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = measure(client, "http://ollama", "test", "extract")
    assert result["tokens_per_second"] == 20
    assert result["ttft_ms"] is not None
    assert result["text"] == '{"name":"Ada"}'


def test_incomplete_stream():
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=""))) as client:
        with pytest.raises(RuntimeError, match="Incomplete"):
            measure(client, "http://ollama", "test", "hello")
