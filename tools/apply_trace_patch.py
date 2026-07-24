#!/usr/bin/env python3
"""Apply opt-in MoE tracing to Poolside/llama.cpp's selective expert-copy path."""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

TARGET = pathlib.Path("ggml/src/ggml-backend.cpp")
MARKER = '#include "moe_trace_support.inc"'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"anchor {label!r} occurred {count} times; expected once")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--install-analyzer", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    target = root / TARGET
    if not target.is_file():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print("MoE trace patch is already applied.")
        return 0

    script_root = pathlib.Path(__file__).resolve().parents[1]
    support_src = script_root / "patches/moe_trace_support.inc"
    support_dst = target.parent / support_src.name
    if not support_src.is_file():
        print(f"error: missing {support_src}", file=sys.stderr)
        return 2

    try:
        text = replace_once(
            text,
            "#include <algorithm>\n#include <vector>\n",
            '#include <algorithm>\n#include <vector>\n#include "moe_trace_support.inc"\n',
            "include",
        )
        text = replace_once(
            text,
            "    struct ggml_backend_sched_split * splits = sched->splits;\n\n    ggml_tensor * prev_ids_tensor = nullptr;\n",
            "    struct ggml_backend_sched_split * splits = sched->splits;\n\n"
            "    const uint64_t moe_trace_call = ggml_moe_trace_begin_call();\n\n"
            "    ggml_tensor * prev_ids_tensor = nullptr;\n",
            "call id",
        )
        text = replace_once(
            text,
            "                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];\n\n"
            "                    ggml_backend_synchronize(input_backend);\n",
            "                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];\n\n"
            "                    ggml_moe_trace_weight(moe_trace_call, input, n_expert, expert_size);\n\n"
            "                    ggml_backend_synchronize(input_backend);\n",
            "weight metadata",
        )
        text = replace_once(
            text,
            "                        prev_ids_tensor = ids_tensor;\n                    }\n\n"
            "                    // group consecutive experts and copy them together\n",
            "                        ggml_moe_trace_route(moe_trace_call, split_backend, input, ids_tensor, ids, n_expert);\n"
            "                        prev_ids_tensor = ids_tensor;\n                    }\n\n"
            "                    // group consecutive experts and copy them together\n",
            "route",
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.no_backup:
        backup = target.with_suffix(target.suffix + ".moe-trace.bak")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"backup: {backup}")
    target.write_text(text, encoding="utf-8")
    shutil.copy2(support_src, support_dst)
    print(f"patched: {target}")
    print(f"installed: {support_dst}")

    if args.install_analyzer:
        source = script_root / "tools/analyze_moe_trace.py"
        destination = root / "tools/moe-cache/analyze_moe_trace.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination.chmod(0o755)
        print(f"installed: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
