"""Sequential, warmed, same-host Ollama benchmark. No fabricated model results."""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np


def hardware():
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    memory = Path("/proc/meminfo").read_text().splitlines()[0] if Path("/proc/meminfo").exists() else "unknown"
    return {"platform": platform.platform(), "cpu": cpu, "memory": memory, "gpu": gpu}


def measure(client, url, model, prompt):
    start, first, parts, final = time.perf_counter(), None, [], None
    with client.stream("POST", url + "/api/generate", json={
        "model": model, "prompt": prompt, "stream": True, "keep_alive": "5m",
        "options": {"temperature": 0, "seed": 42, "num_predict": 128, "num_ctx": 2048},
    }) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            item = json.loads(line)
            if "error" in item:
                raise RuntimeError(item["error"])
            if item.get("response"):
                first = first or time.perf_counter()
                parts.append(item["response"])
            if item.get("done"):
                final = item
    if final is None:
        raise RuntimeError("Incomplete Ollama stream")
    return {"text": "".join(parts), "total_ms": (time.perf_counter() - start) * 1000,
            "ttft_ms": (first - start) * 1000 if first else None,
            "tokens_per_second": final.get("eval_count", 0) / max(final.get("eval_duration", 0) / 1e9, 1e-9),
            "output_tokens": final.get("eval_count", 0), "load_ms": final.get("load_duration", 0) / 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs=3, default=["qwen2.5:1.5b", "llama3.2:1b", "gemma2:2b"])
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/local.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.models)) != 3:
        parser.error("Use three distinct models and at least one repeat")
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not cases:
        parser.error("Dataset must not be empty")
    report = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hardware": hardware(), "dataset": str(args.dataset), "models": {}}
    with httpx.Client(timeout=180) as client:
        report["ollama_version"] = client.get(args.url + "/api/version").json()
        for model in args.models:
            metadata = client.post(args.url + "/api/show", json={"model": model})
            metadata.raise_for_status()
            cold = measure(client, args.url, model, "Reply with ready.")
            rows = []
            for repeat in range(args.repeats):
                for idx, case in enumerate(cases):
                    row = measure(client, args.url, model, case["prompt"])
                    try:
                        parsed = json.loads(row["text"])
                        row["valid_json"], row["exact_match"] = True, parsed == case["expected"]
                    except ValueError:
                        row["valid_json"], row["exact_match"] = False, False
                    row.update({"case": idx, "repeat": repeat})
                    rows.append(row)
            report["models"][model] = {
                "details": metadata.json().get("details"), "warmup": cold, "runs": rows,
                "p50_ms": float(np.percentile([r["total_ms"] for r in rows], 50)),
                "p95_ms": float(np.percentile([r["total_ms"] for r in rows], 95)),
                "mean_tokens_per_second": float(np.mean([r["tokens_per_second"] for r in rows])),
                "json_valid_rate": sum(r["valid_json"] for r in rows) / len(rows),
                "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
            }
            unload = client.post(args.url + "/api/generate", json={"model": model, "keep_alive": 0})
            unload.raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured results: {args.output}")


if __name__ == "__main__":
    main()
