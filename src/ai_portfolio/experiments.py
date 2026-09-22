"""Validate comparability of actual extraction experiment reports."""
import math


def compare_reports(reports: list[dict]) -> list[dict]:
    if len(reports) != 3:
        raise ValueError("Need base, SFT and DPO reports")
    first = reports[0]
    digest = first.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Dataset identity missing")
    if not first.get("model") or first.get("adapter") is not None:
        raise ValueError("First report must be an unadapted base model")
    rows = []
    for stage, report in zip(["Base", "SFT", "DPO"], reports, strict=True):
        if report["dataset_sha256"] != digest or report["model"] != first["model"]:
            raise ValueError("Model or dataset mismatch")
        if stage != "Base" and not report.get("adapter"):
            raise ValueError("SFT/DPO reports must identify their adapter")
        m = report["metrics"]
        if type(m["n"]) is not int or m["n"] <= 0 or m["n"] != first["metrics"]["n"]:
            raise ValueError("Sample count mismatch")
        for name in ["json_valid_rate", "field_accuracy", "exact_match"]:
            value = m[name]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid metric")
        rows.append({"stage": stage, **m})
    return rows
