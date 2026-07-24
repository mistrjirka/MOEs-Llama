# Laguna MoE cache bootstrap

This repository starts the implementation on **Poolside's `llama.cpp` Laguna branch** without yet changing model execution.

The first milestone instruments the existing selective `MUL_MAT_ID` path and records the exact expert routes and bytes per expert. The included simulator then estimates persistent-cache hit rates and RAM→VRAM traffic for LRU, LFU, SLRU, second-touch SLRU, and static hot-set policies.

This ordering is intentional: compact quantized slots and predictive prefetch are invasive. Real Laguna traces should decide cache capacity, admission policy, prefill handling, and whether prediction is worth implementing.

## 1. Apply the trace instrumentation

From this repository:

```bash
python tools/apply_trace_patch.py /path/to/poolside-llama.cpp \
  --install-analyzer
```

The script was designed against:

```text
repository: poolsideai/llama.cpp
branch:     laguna
file blob:  ggml/src/ggml-backend.cpp @ 87615921c09be5ef8c4996faa70fb3f49c385031
```

It verifies exact structural anchors and is idempotent. It creates:

```text
ggml/src/ggml-backend.cpp.moe-trace.bak
```

unless `--no-backup` is used.

Inspect the change:

```bash
cd /path/to/poolside-llama.cpp
git diff -- ggml/src/ggml-backend.cpp tools/moe-cache/analyze_moe_trace.py
```

## 2. Build normally

Use your existing CUDA configuration. For example:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tracing is entirely disabled unless `GGML_MOE_TRACE` is set.

## 3. Collect a representative Laguna trace

For decode-only data:

```bash
export GGML_MOE_TRACE="$PWD/laguna-routing.jsonl"
export GGML_MOE_TRACE_DECODE_ONLY=1
export GGML_MOE_TRACE_FLUSH=0

./build/bin/llama-server \
  -m ~/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ2_M.gguf \
  --jinja \
  -fa on \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 2345
```

Run several real coding-agent tasks. Prefer multiple repositories and task types rather than a single synthetic prompt.

To include prefill and study cache pollution, omit `GGML_MOE_TRACE_DECODE_ONLY`.

Optional trace controls:

```bash
GGML_MOE_TRACE_MAX_CALLS=10000  # stop recording after N scheduler calls
GGML_MOE_TRACE_FLUSH=1          # safer, slower; flush every record
```

## 4. Simulate caches

Example with a 6 GiB expert-cache budget:

```bash
python tools/moe-cache/analyze_moe_trace.py laguna-routing.jsonl \
  --policy slru-2touch \
  --vram-gib 6 \
  --scope decode \
  --json laguna-cache-report.json
```

Compare policies:

```bash
for policy in lru lfu slru slru-2touch; do
  python tools/moe-cache/analyze_moe_trace.py laguna-routing.jsonl \
    --policy "$policy" --vram-gib 6 --scope decode
done
```

Static hot-set upper bound after a 25% warmup:

```bash
python tools/moe-cache/analyze_moe_trace.py laguna-routing.jsonl \
  --policy static \
  --vram-gib 6 \
  --warmup-fraction 0.25 \
  --write-plan laguna-static-plan.json
```

The simulator reports:

- cache hit rate;
- estimated expert bytes transferred from RAM to VRAM;
- previous-token route overlap;
- per-layer cache behavior;
- a static expert placement plan when requested.

## What this milestone does not claim

It does not yet implement a runtime VRAM cache or predict experts. It avoids changing numerical behavior until the real routing data establishes that the proposed cache can pay for its complexity.

The next implementation milestone is a **persistent reactive device-slot cache** using dedicated backend allocations, exact global-to-slot ID remapping, and quantization-specific padding. See `DESIGN.md`.

## Standalone cache-state prototype

`prototype/moe_cache_core.{h,cpp}` contains a tested C++11 cache-state core with:

- second-touch SLRU admission;
- probation/protected segments;
- explicit `loading` and `ready` states;
- generation-safe handles;
- in-flight eviction protection;
- prefill admission suppression;
- frequency decay hooks.

It deliberately contains no CUDA or GGML dependencies yet. This lets the state machine be tested before device allocation and event handling are introduced.

Run all repository tests:

```bash
tests/run_tests.sh
```

The next code step is to adapt this core into a scheduler-owned GGML component and attach dedicated backend buffers, while retaining these tests as policy/state-machine tests.
