"""Separate deterministic retrieval/invariant CI from real model evaluation."""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np

from .rag import enforce_citations, select_evidence
from .retrieval import build_index


def check_gates(report: dict, baseline: dict | None = None) -> list[str]:
    failures = []
    for name in ("recall_at_4", "mrr", "citation_invariant_pass_rate", "retrieval_p95_ms"):
        value = report.get(name)
        upper = float("inf") if name == "retrieval_p95_ms" else 1
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= upper:
            return [f"Invalid or missing metric: {name}"]
    generation = report.get("generation")
    if generation is not None:
        if not isinstance(generation, dict):
            return ["Invalid generation metrics"]
        for name in ("evidence_match_rate", "abstention_accuracy"):
            value = generation.get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                return [f"Invalid or missing generation metric: {name}"]
    if report["recall_at_4"] < 0.95:
        failures.append("recall_at_4 below 0.95")
    if report["mrr"] < 0.8:
        failures.append("MRR below 0.8")
    if report["citation_invariant_pass_rate"] < 1:
        failures.append("citation invariant failure")
    if report["retrieval_p95_ms"] > 3000:
        failures.append("retrieval p95 over 3000ms")
    if report.get("generation"):
        if report["generation"]["evidence_match_rate"] < 0.8:
            failures.append("generated evidence match below 0.8")
        if report["generation"]["abstention_accuracy"] < 1:
            failures.append("unanswerable query not refused")
    if baseline:
        if baseline["mode"] != report["mode"]:
            failures.append("baseline retrieval mode mismatch")
        for metric in ("recall_at_4", "mrr"):
            if report[metric] < baseline[metric] - 0.02:
                failures.append(f"{metric} regressed by > 0.02")
        if report["retrieval_p95_ms"] > max(baseline["retrieval_p95_ms"] * 1.5, 50):
            failures.append("p95 latency regressed > 50% (50ms noise floor)")
    return failures


async def evaluate(args):
    index = build_index(Path("data/docs"), args.mode)
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    ranks, durations, rows = [], [], []
    matches, refusals = [], []
    async with httpx.AsyncClient(timeout=60) as client:
        for case in cases:
            start = time.perf_counter()
            found = index.search(case["question"])
            durations.append((time.perf_counter() - start) * 1000)
            rank = next((i for i, c in enumerate(found, 1)
                         if c.source == case["source"] and case["evidence"] in c.text), None)
            if case["source"]:
                ranks.append(rank)
            row = {"question": case["question"], "rank": rank, "retrieved": [c.id for c in found]}
            if args.generate:
                citations, _ = await select_evidence(client, args.url, args.model, case["question"], found)
                row["citations"] = citations
                if case["source"]:
                    matches.append(any(case["evidence"] in c["quote"] for c in citations))
                else:
                    refusals.append(not citations)
            rows.append(row)
    chunk = index.chunks[0]
    valid = json.dumps({"citations": [{"chunk_id": chunk.id, "quote": chunk.text}]})
    invariant = [bool(enforce_citations(valid, [chunk]))]
    for invalid in ["not json", '{"citations":[{"chunk_id":"invented","quote":"unsupported claim"}]}',
                    json.dumps({"citations": [{"chunk_id": chunk.id, "quote": "fabricated evidence"}]})]:
        try:
            enforce_citations(invalid, [chunk])
            invariant.append(False)
        except ValueError:
            invariant.append(True)
    return {"mode": args.mode, "cases": rows,
            "recall_at_4": sum(r is not None for r in ranks) / len(ranks),
            "mrr": sum(1 / r if r else 0 for r in ranks) / len(ranks),
            "retrieval_p50_ms": float(np.percentile(durations, 50)),
            "retrieval_p95_ms": float(np.percentile(durations, 95)),
            "citation_invariant_pass_rate": sum(invariant) / len(invariant),
            "generation": {"evidence_match_rate": sum(matches) / len(matches),
                           "abstention_accuracy": sum(refusals) / len(refusals)} if args.generate else None,
            "note": "Smoke uses lexical test doubles. Generation quality is unmeasured unless --generate."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "neural"], default="neural")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/rag.jsonl"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rag-eval.json"))
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    failures = check_gates(report, json.loads(args.baseline.read_text()) if args.baseline else None)
    report["gate_failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
