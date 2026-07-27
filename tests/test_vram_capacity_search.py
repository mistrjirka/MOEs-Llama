#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_vram_capacity_search.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        helper = root / "fake_benchmark.py"
        helper.write_text(
            "import os, sys\n"
            "r=int(os.environ['TEST_RESERVE'])\n"
            "print('graph splits = 152')\n"
            "print(f'decode: {5.0 + (4096-r)/4096:.3f} tok/s')\n"
            "sys.exit(0 if r >= 1024 else 3)\n",
            encoding="utf-8",
        )
        report = root / "report.json"
        subprocess.run([
            sys.executable, str(TOOL),
            "--reserve-env", "TEST_RESERVE",
            "--reserves", "4096,2048,1024,512",
            "--minimum-free-mib", "0",
            "--output-dir", str(root / "logs"),
            "--json", str(report),
            "--", sys.executable, str(helper),
        ], check=True)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["selected_reserve_mib"] == 1024, data
        assert [row["valid"] for row in data["results"]] == [True, True, True, False]
        assert data["results"][2]["graph_splits"] == 152
        assert data["results"][2]["throughput_tps"] is not None
    print("VRAM capacity search test passed")


if __name__ == "__main__":
    main()
