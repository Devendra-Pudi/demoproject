import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("extraction", Path("scripts/evaluate_extraction.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extraction_metrics():
    result = module.score(['{"name":"Ada","email":null}', 'not json'],
                          [{"name": "Ada", "email": None}, {"name": "Grace", "email": None}])
    assert result["json_valid_rate"] == 0.5
    assert result["field_accuracy"] == 0.5
    assert result["exact_match"] == 0.5
