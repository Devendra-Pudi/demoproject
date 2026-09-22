import json

import pytest

from ai_portfolio.rag import enforce_citations
from ai_portfolio.retrieval import Chunk

CHUNK = Chunk("abc", "policy.md", "Employees receive 20 days of annual leave.")


def test_valid_citation():
    raw = json.dumps({"citations": [{"chunk_id": "abc", "quote": CHUNK.text}]})
    assert enforce_citations(raw, [CHUNK])[0]["source"] == "policy.md"


@pytest.mark.parametrize("raw", [
    'not json', '{"answer":"invented", "citations":[]}',
    '{"citations":[{"chunk_id":"fake", "quote":"Employees receive 20 days"}]}',
    '{"citations":[{"chunk_id":"abc", "quote":"Employees receive 50 days"}]}',
    '{"citations":[{"chunk_id":"abc", "quote":" "}]}',
])
def test_rejects_unsupported(raw):
    with pytest.raises(ValueError):
        enforce_citations(raw, [CHUNK])


def test_abstention():
    assert enforce_citations('{"citations":[]}', [CHUNK]) == []
