"""Produce a before/after table from actual evaluations, never placeholder scores."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs=3, type=Path, help="base.json sft.json dpo.json")
args = parser.parse_args()
print("| Stage | n | JSON valid | Field accuracy | Exact match |")
print("|---|---:|---:|---:|---:|")
for label, path in zip(["Base", "SFT", "DPO"], args.reports, strict=True):
    m = json.loads(path.read_text())["metrics"]
    print(f"| {label} | {m['n']} | {m['json_valid_rate']:.3f} | {m['field_accuracy']:.3f} | {m['exact_match']:.3f} |")
