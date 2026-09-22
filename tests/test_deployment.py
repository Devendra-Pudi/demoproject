import json
from pathlib import Path

import pytest

from ai_portfolio.experiments import compare_reports


def reports():
    return [{"model": "base", "adapter": adapter, "dataset_sha256": "a" * 64,
             "metrics": {"n": 6, "json_valid_rate": 1, "exact_match": 0.5, "field_accuracy": 0.75}}
            for adapter in [None, "sft", "dpo"]]


def test_matched_reports():
    assert [r["stage"] for r in compare_reports(reports())] == ["Base", "SFT", "DPO"]


@pytest.mark.parametrize("change", [{"dataset_sha256": "b" * 64}, {"model": "different"}, {"adapter": None}])
def test_rejects_unmatched_reports(change):
    data = reports()
    data[-1].update(change)
    with pytest.raises(ValueError):
        compare_reports(data)


@pytest.mark.parametrize("value", [-1, 2, float("nan"), float("inf"), "0.9"])
def test_invalid_report_metrics(value):
    data = reports()
    data[-1]["metrics"]["exact_match"] = value
    with pytest.raises(ValueError):
        compare_reports(data)


def test_each_project_has_deployment_files():
    projects = json.loads(Path("site/projects.json").read_text())
    assert len(projects) == 5
    for project in projects:
        folder = Path("projects") / project["folder"]
        for filename in ["app.py", "requirements.txt", "Dockerfile", "README.md"]:
            assert (folder / filename).is_file()
        assert project["url"] is None or project["url"].startswith("https://")
