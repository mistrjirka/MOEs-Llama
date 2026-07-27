#!/usr/bin/env python3
"""Find the smallest safe VRAM reserve for a repeatable llama.cpp workload.

The command is executed once for every reserve value. The reserve is exported
through a configurable environment variable, stdout/stderr are retained, and
optional ``nvidia-smi`` sampling records peak memory use and minimum free VRAM.

This tool intentionally chooses *maximum productive VRAM*, not zero headroom:
a candidate is valid only when the command succeeds, no OOM signature appears,
and the measured minimum free memory stays above ``--minimum-free-mib``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Sequence

OOM_PATTERNS = (
    re.compile(r"out of memory", re.I),
    re.compile(r"cudaErrorMemoryAllocation", re.I),
    re.compile(r"failed to allocate", re.I),
    re.compile(r"allocation failed", re.I),
)
TPS_PATTERNS = (
    re.compile(r"(?:eval time|decode)[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*(?:tokens per second|tok/s|t/s)", re.I),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens per second|tok/s|t/s)", re.I),
)
SPLIT_PATTERNS = (
    re.compile(r"graph splits\s*=\s*(\d+)", re.I),
    re.compile(r"decode splits?\s*[:=]\s*(\d+)", re.I),
)


@dataclass
class ProbeResult:
    reserve_mib: int
    returncode: int
    duration_seconds: float
    timed_out: bool
    oom_detected: bool
    peak_used_mib: int | None
    minimum_free_mib: int | None
    throughput_tps: float | None
    graph_splits: int | None
    valid: bool
    log_path: str


def parse_last_float(patterns: Sequence[re.Pattern[str]], text: str) -> float | None:
    values: list[float] = []
    for pattern in patterns:
        values.extend(float(match.group(1)) for match in pattern.finditer(text))
    return values[-1] if values else None


def parse_last_int(patterns: Sequence[re.Pattern[str]], text: str) -> int | None:
    value = parse_last_float(patterns, text)
    return int(value) if value is not None else None


def query_gpu_memory(gpu_index: str) -> tuple[int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--id", gpu_index,
        "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=3)
        used, free = output.strip().splitlines()[0].split(",")
        return int(used.strip()), int(free.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run_probe(args: argparse.Namespace, reserve_mib: int, command: Sequence[str]) -> ProbeResult:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"reserve-{reserve_mib:05d}.log"
    environment = os.environ.copy()
    environment[args.reserve_env] = str(reserve_mib)

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=environment,
        cwd=args.cwd,
    )
    samples: list[tuple[int, int]] = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.wait(args.sample_interval):
            sample = query_gpu_memory(args.gpu_index)
            if sample is not None:
                samples.append(sample)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started = time.monotonic()
    timed_out = False
    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    finally:
        stop.set()
        sampler.join(timeout=2)
    duration = time.monotonic() - started
    log_path.write_text(output, encoding="utf-8")

    oom = any(pattern.search(output) for pattern in OOM_PATTERNS)
    peak_used = max((used for used, _ in samples), default=None)
    minimum_free = min((free for _, free in samples), default=None)
    throughput = parse_last_float(TPS_PATTERNS, output)
    splits = parse_last_int(SPLIT_PATTERNS, output)
    memory_guard_ok = minimum_free is None or minimum_free >= args.minimum_free_mib
    valid = process.returncode == 0 and not timed_out and not oom and memory_guard_ok
    return ProbeResult(
        reserve_mib=reserve_mib,
        returncode=process.returncode,
        duration_seconds=duration,
        timed_out=timed_out,
        oom_detected=oom,
        peak_used_mib=peak_used,
        minimum_free_mib=minimum_free,
        throughput_tps=throughput,
        graph_splits=splits,
        valid=valid,
        log_path=str(log_path),
    )


def parse_reserves(value: str) -> list[int]:
    reserves = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not reserves or any(reserve < 0 for reserve in reserves):
        raise argparse.ArgumentTypeError("reserves must be non-negative comma-separated MiB values")
    return reserves


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserve-env",
        default="GGML_MOE_DYNAMIC_VRAM_RESERVE_MIB",
        help="environment variable consumed by the experimental cache runtime",
    )
    parser.add_argument("--reserves", type=parse_reserves, default=parse_reserves("4096,2048,1536,1024,768,512"))
    parser.add_argument("--minimum-free-mib", type=int, default=512)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cwd", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--json", type=pathlib.Path, required=True)
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="continue testing smaller reserves after the first invalid candidate",
    )
    args, command = parser.parse_known_args(argv)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide the benchmark command after --")
    if args.minimum_free_mib < 0 or args.sample_interval <= 0 or args.timeout <= 0:
        parser.error("memory guard must be non-negative; intervals and timeout must be positive")
    return args, command


def main(argv: Sequence[str] | None = None) -> int:
    args, command = parse_args(argv or sys.argv[1:])
    results: list[ProbeResult] = []
    for reserve in args.reserves:
        print(f"[probe] reserve={reserve} MiB", flush=True)
        result = run_probe(args, reserve, command)
        results.append(result)
        status = "valid" if result.valid else "invalid"
        print(
            f"[probe] {status}: rc={result.returncode} free_min={result.minimum_free_mib} "
            f"peak_used={result.peak_used_mib} tps={result.throughput_tps} splits={result.graph_splits}",
            flush=True,
        )
        if not result.valid and not args.continue_after_failure:
            break

    valid = [result for result in results if result.valid]
    selected = min(valid, key=lambda result: result.reserve_mib) if valid else None
    payload = {
        "reserve_env": args.reserve_env,
        "command": command,
        "minimum_free_mib": args.minimum_free_mib,
        "selected_reserve_mib": selected.reserve_mib if selected else None,
        "selection_rule": "smallest successful reserve satisfying the measured free-memory guard",
        "results": [asdict(result) for result in results],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if selected is None:
        print("[probe] no safe reserve found", file=sys.stderr)
        return 2
    print(f"[probe] selected reserve={selected.reserve_mib} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
