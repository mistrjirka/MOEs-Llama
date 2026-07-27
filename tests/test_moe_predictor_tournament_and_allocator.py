#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "moe_predictor_tournament_and_allocator.py"


def write_trace(path: pathlib.Path) -> None:
    rows: list[dict[str, object]] = []
    for layer in range(6):
        for component in ("gate", "up", "down"):
            rows.append({
                "type": "weight",
                "tensor": f"blk.{layer}.ffn_{component}_exps.weight",
                "layer": layer,
                "n_expert": 16,
                "expert_bytes": 1024 + layer * 32,
            })
    for step in range(48):
        for layer in range(6):
            base = (step + 3 * layer) % 16
            experts = [(base + offset) % 16 for offset in range(4)]
            rows.append({
                "type": "route",
                "layer": layer,
                "call_id": step,
                "n_expert": 16,
                "tokens": 1,
                "ids": experts,
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        trace = root / "trace.jsonl"
        report = root / "report.json"
        write_trace(trace)
        subprocess.run([
            sys.executable, str(TOOL), str(trace),
            "--candidate-count", "8",
            "--history-len", "3",
            "--horizons", "1,2,3",
            "--vram-gib", "0.00008",
            "--json", str(report),
        ], check=True)
        data = json.loads(report.read_text(encoding="utf-8"))
        audit = data["predictor_independence_audit"]
        assert audit["passed"], audit
        assert not audit["feature_index_collisions"], audit
        assert data["grouping"]["steps_complete"] == 48
        allocation = data["allocation"]
        assert allocation["config"]["budget_bytes"] > 0
        assert allocation["optimized"]["metrics"]["allocated_bytes"] <= allocation["config"]["budget_bytes"]
    print("predictor/allocation synthetic test passed")


if __name__ == "__main__":
    main()
