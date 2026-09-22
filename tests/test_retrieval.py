from pathlib import Path

import pytest

from ai_portfolio.retrieval import build_index, load_chunks


def test_retrieval():
    index = build_index(Path("data/docs"), "smoke")
    assert "90 days" in index.search("When must production API keys be rotated?")[0].text
    assert len(index.search("expense reimbursement")) == 4


def test_chunks_stable_and_bounded(tmp_path):
    (tmp_path / "a.md").write_text(" ".join(str(i) for i in range(50)))
    first = load_chunks(tmp_path, 10, 2)
    assert first == load_chunks(tmp_path, 10, 2)
    assert all(len(c.text.split()) <= 10 for c in first)
    assert first[0].text.split()[-2:] == first[1].text.split()[:2]
    with pytest.raises(ValueError):
        load_chunks(tmp_path, 10, 10)


def test_empty_and_invalid(tmp_path):
    with pytest.raises(ValueError):
        build_index(tmp_path, "smoke")
    with pytest.raises(ValueError):
        build_index(tmp_path, "typo")
    with pytest.raises(ValueError):
        build_index(Path("data/docs"), "smoke").search(" ")
