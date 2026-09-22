"""Content-free SQLite traces. Estimated API cost is distinct from local compute cost."""
import json
import sqlite3
from pathlib import Path

import numpy as np


class Telemetry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS traces "
                       "(id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, trace: dict):
        with self.connect() as db:
            db.execute("INSERT INTO traces(id,payload) VALUES (?,?)", (trace["id"], json.dumps(trace)))

    def recent(self, limit: int = 1000) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM traces ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def summary(self) -> dict:
        traces = self.recent()
        durations = [t["total_ms"] for t in traces]
        return {
            "window": "last 1000 requests", "requests": len(traces),
            "p50_ms": float(np.percentile(durations, 50)) if traces else None,
            "p95_ms": float(np.percentile(durations, 95)) if traces else None,
            "degradation_rate": sum(t["status"] != "ok" for t in traces) / len(traces) if traces else None,
            "estimated_api_cost_usd": sum(t.get("estimated_api_cost_usd", 0) for t in traces),
            "mean_cost_per_request_usd": (
                sum(t.get("estimated_api_cost_usd", 0) for t in traces) / len(traces) if traces else None
            ),
            "cost_note": "Configured token rates only; excludes hardware, electricity, and hosting.",
            "citation_valid_rate": (
                sum(t.get("citation_valid", False) for t in traces if t["status"] == "ok") /
                max(sum(t["status"] == "ok" for t in traces), 1) if traces else None
            ),
        }
