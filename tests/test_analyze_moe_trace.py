#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/analyze_moe_trace.py"


def main() -> int:
    records = [{"type": "header", "version": 1}]
    for layer in range(2):
        for tensor, size in (
            ("ffn_gate_exps", 100),
            ("ffn_up_exps", 100),
            ("ffn_down_exps", 120),
        ):
            records.append(
                {
                    "type": "weight",
                    "tensor": f"blk.{layer}.{tensor}.weight",
                    "ggml_type": 1,
                    "n_expert": 8,
                    "expert_bytes": size,
                }
            )

    routes = [
        ((0, 1), (2, 3)),
        ((0, 1), (2, 3)),
        ((0, 4), (2, 5)),
        ((0, 1), (2, 3)),
    ]
    for call, per_layer in enumerate(routes):
        for layer, experts in enumerate(per_layer):
            records.append(
                {
                    "type": "route",
                    "call": call,
                    "scheduler": "0x1",
                    "backend": "CUDA0",
                    "tensor": f"blk.{layer}.ffn_gate_exps.weight",
                    "n_expert": 8,
                    "top_k": 2,
                    "tokens": 1,
                    "ids": [list(experts)],
                }
            )

    with tempfile.TemporaryDirectory() as temporary:
        trace = pathlib.Path(temporary) / "trace.jsonl"
        output = pathlib.Path(temporary) / "report.json"
        trace.write_text("".join(json.dumps(record) + "\n" for record in records))
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                str(trace),
                "--policy",
                "lru",
                "--slots",
                "2",
                "--scope",
                "decode",
                "--json",
                str(output),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        report = json.loads(output.read_text())
        assert report["total"]["cache_vram_bytes"] == 2 * 2 * 320
        assert report["total"]["accesses"] == 16
        assert report["total"]["hits"] > 0
        assert "Hit rate:" in result.stdout
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
