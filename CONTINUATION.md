# MoE RAM/VRAM Expert Cache — Research and Implementation Handoff

**Status date:** 2026-07-24  
**Project repository:** [mistrjirka/MOEs-Llama](https://github.com/mistrjirka/MOEs-Llama)  
**Primary runtime target:** Poolside's Laguna-capable `llama.cpp` fork  
**Initial model target:** `Laguna-S-2.1-UD-IQ2_M.gguf`  
**Initial hardware target:** NVIDIA V100 32 GB, 256 GB host RAM, 24-core/48-thread AMD EPYC

---

## 1. Executive summary

The project investigates whether a large sparse Mixture-of-Experts model can run more efficiently when:

1. the complete quantized model remains in host RAM;
2. dense tensors and other consistently used tensors occupy VRAM normally;
3. remaining VRAM is managed as a **persistent cache of routed experts**;
4. frequently reused experts execute directly from VRAM;
5. cache misses can either execute from RAM on the CPU or be promoted asynchronously to VRAM;
6. later, expert predictions may prefetch likely experts before the router actually needs them.

This is **not already implemented in upstream `llama.cpp`**. Current `llama.cpp` can selectively copy only the experts chosen by the router, but its normal scheduler path does not retain a persistent `expert → VRAM slot` mapping across forwards. Experimental upstream pull requests attempted persistent caches but were withdrawn.

Colibri demonstrates several useful ideas—RAM LRU caching, learned hot sets, persistent VRAM placement, periodic repinning, and next-layer routing prediction—but it is a custom GLM-5.2 engine rather than a general GGUF runtime. Its predictive GPU-staging experiment also regressed end-to-end performance on one measured system. Therefore:

> **Use Poolside's `llama.cpp` as the implementation base and Colibri as a policy, instrumentation, and negative-result reference.**

The repository currently contains the first milestone only:

- opt-in expert-route tracing for `llama.cpp`;
- an offline cache-policy simulator;
- a standalone C++ cache-state prototype;
- tests for the simulator and cache state machine;
- the full architectural design in `DESIGN.md`.

It does **not** yet contain a functional persistent CUDA expert cache.

---

## 2. Original motivation

Large MoE checkpoints may be much larger than available VRAM even though only a small subset of experts is selected for each token. Ordinary partial offload leaves expert weights in host memory and transfers selected expert data to the GPU when needed, or executes those layers on the CPU.

The intended experiment asks whether routing locality is strong enough that a bounded VRAM expert cache can avoid repeated RAM-to-GPU transfers. The most valuable version is not just a conventional cache. It treats RAM and VRAM as simultaneous compute tiers:

```text
Complete quantized checkpoint in host RAM
                 │
                 ├── CPU execution of selected RAM experts
                 │
                 └── pinned/repacked staging cache
                               │
                               ▼
                    persistent VRAM expert slots
                               │
                               └── GPU execution of cached experts
```

A later predictor may issue low-priority prefetch requests, but prediction is not part of correctness and must not be implemented before a reactive cache is proven useful.

---

## 3. Main project principles

### 3.1 Preserve model semantics by default

The cache must not substitute different experts or silently alter their quantization. The router selects the same experts whether the cache is enabled or disabled.

Different **physical packings** are allowed in each tier:

- canonical GGUF blocks in ordinary host RAM;
- a CUDA-friendly repack in pinned host staging memory;
- an aligned and padded slot representation in VRAM.

However, exact mode must preserve the same logical quantized values. For example, converting an IQ2 expert to Q4 only while it is cached would make model output depend on cache residency. Such transquantization must be an explicit lossy research mode, never the default.

### 3.2 Cache an expert bundle atomically

A routed feed-forward expert normally consists of three matrices:

```text
(layer, expert) → {gate, up, down}
```

All three matrices share one residency state and slot assignment. The bundle becomes ready only after every upload completes. Eviction cannot remove one part while another remains in use.

### 3.3 RAM is a compute tier, not merely storage

A cache miss should not always force an upload. Depending on quantization, batch size, CPU speed, transfer state, and expected reuse, it may be faster to execute that expert directly on the CPU.

The eventual decision should use measured costs:

```text
CPU expert execution time
host-side repack time
host-to-device transfer time
GPU expert execution time
stream queueing/interference
estimated probability of reuse
```

### 3.4 Prefill and decode need different admission policies

A long prompt can touch many experts and destroy a cache that was useful during decode. The intended default is:

- allow existing cache hits during prefill;
- suppress first-touch prefill admission;
- optionally admit repeated experts from the same prefill;
- protect established decode entries;
- evaluate prefill and decode traces separately.

### 3.5 Measure before predicting

The required order is:

1. collect real Laguna routing traces;
2. simulate cache policies and VRAM budgets;
3. implement a reactive persistent cache;
4. measure exact-cache benefit;
5. add hybrid CPU/GPU miss execution;
6. only then test prediction and prefetch.

A predictor with good recall can still lose performance through cache pollution, transfer contention, extra synchronization, or interference with attention kernels.

---

## 4. Why `llama.cpp` is the implementation base

### Advantages of `llama.cpp`

- It already loads GGUF files and mixed tensor quantizations.
- It already supports the Laguna architecture.
- It already has mature CUDA, CPU, KV-cache, server, batching, and model-loading paths.
- It already identifies selected expert IDs and selectively transfers selected expert rows.
- A generic implementation can later benefit Qwen, GPT-OSS, Gemma, and other MoE architectures.
- Poolside's fork also provides the Laguna-specific work needed for DFlash/speculative execution.

Relevant Laguna model support is visible in [`src/models/laguna.cpp`](https://github.com/ggml-org/llama.cpp/blob/95a923a64c7d493ed1cb347d3b55d039fa3b8097/src/models/laguna.cpp). It creates routed gate/up/down tensors as three-dimensional expert tensors and should remain a model-description layer, not the location of generic cache policy.

### Why not build directly on Colibri

Colibri is easier to understand as an expert-streaming experiment because its experts are explicit first-class objects. However, it is primarily a custom GLM-5.2 runtime with custom tensor storage, kernels, attention implementation, quantization formats, and model-specific code.

Using Colibri would require implementing or porting:

- Laguna architecture support;
- GGUF loading and mixed quantization handling;
- IQ2 kernels and layout support;
- the normal `llama.cpp` server/tooling surface;
- DFlash integration;
- general model compatibility.

That work is likely larger than implementing the cache in `llama.cpp`.

**Decision:** Colibri is a research reference, not the runtime base.

---

## 5. What current `llama.cpp` actually does

The relevant scheduler path is `ggml_backend_sched_compute_splits()` in [`ggml/src/ggml-backend.cpp`](https://github.com/ggml-org/llama.cpp/blob/298219f985b09d93df515ca708736665468ca827/ggml/src/ggml-backend.cpp).

For host-resident MoE weights used by `GGML_OP_MUL_MAT_ID`, the current code:

1. reads the runtime expert-ID tensor;
2. identifies the set of experts used by the current operation;
3. groups consecutive expert IDs;
4. invokes `ggml_backend_tensor_set_async()` for the selected expert ranges;
5. includes extra tail padding because CUDA MMQ may read beyond the logical expert row.

What it does **not** do in the inspected version:

- no persistent `expert_id → slot_id` mapping;
- no hit/miss lookup against persistent GPU expert slots;
- no eviction policy;
- no learned hot set;
- no predictor;
- no persistent compact VRAM buffer independent of graph allocations.

This selective-copy path is still the main integration point because it already has the true router IDs and knows the source tensor, destination backend, tensor type, expert count, and per-expert byte stride.

### Important scheduler constraint

The graph allocator owns temporary input-copy tensors and may reuse their addresses. Persistent cache storage therefore must be allocated outside ordinary graph-temporary lifetime and owned by a scheduler/context-level cache manager.

A cache that only remembers what was written into an ordinary temporary buffer risks treating reused or reallocated memory as valid expert data.

---

## 6. Previous `llama.cpp` cache attempts

### Feature request #20757

[`ggml-org/llama.cpp#20757`](https://github.com/ggml-org/llama.cpp/issues/20757) proposed a two-tier GPU+RAM expert cache with a persistent VRAM slot buffer and eviction policies. It documents the same basic gap: selected experts are copied, used, and not retained as a managed cache across forwards.

The issue also reports a Python proof of concept and later experimental backend work. Treat those performance figures as author-reported evidence, not independently reproduced results.

### Withdrawn PR #21609

[`ggml-org/llama.cpp#21609`](https://github.com/ggml-org/llama.cpp/pull/21609) attempted an N-slot LFRU cache with FATE prefetch. It was withdrawn.

### Withdrawn PR #21614

[`ggml-org/llama.cpp#21614`](https://github.com/ggml-org/llama.cpp/pull/21614) attempted a persistent expert cache for `--n-cpu-moe`. It was also withdrawn.

### Critical quantization lesson

One withdrawn implementation attempted compact N-slot tensors and encountered a concrete problem with quantized CUDA MMQ paths: kernels may read padding past the logical expert row. A compact slot cannot simply contain `expert_size` bytes. It requires:

- backend-specific alignment;
- type-specific tail padding;
- valid tensor/kernel metadata;
- guard validation under CUDA memcheck or compute-sanitizer.

This is especially important for the target `UD-IQ2_M` model. The implementation must derive the actual type of each tensor rather than assuming every tensor has one uniform IQ2 format.

---

## 7. What Colibri contributes

Primary project: [JustVugg/colibri](https://github.com/JustVugg/colibri)

Colibri treats VRAM, RAM, and storage as an application-managed memory hierarchy. Its useful ideas include:

- per-layer RAM expert LRU caches;
- a learned persistent usage profile;
- pinned hot experts;
- a persistent VRAM hot-expert tier;
- periodic live repinning using recent frequency/recency;
- asynchronous disk loading;
- next-layer router-lookahead prefetch;
- cache-aware routing experiments;
- detailed tier and timing instrumentation.

### Colibri VRAM tier

Colibri's CUDA documentation describes a measured profile that uploads selected hot experts persistently into VRAM at startup. Remaining experts can stay in RAM or disk. It can periodically adapt the VRAM set rather than using a demand-filled slot on every miss.

Source: [`docs/cuda.md`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/cuda.md)

### Live repinning

Colibri uses decaying heat/frequency and recency to replace sufficiently colder pinned experts at safe boundaries, with hysteresis and a swap limit.

Sources:

- [`docs/tuning.md`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/tuning.md)
- [`c/tier.h`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/c/tier.h)

### Router lookahead

Colibri's `PILOT` applies a future layer's router to an earlier/approximate hidden state and reports useful next-layer top-K recall. Its normal purpose is to prefetch disk-backed experts toward RAM, not to maintain the exact generic GPU cache proposed here.

Source: [`docs/tuning.md`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/tuning.md)

### Cache-aware routing is different and potentially lossy

Colibri also has an experimental mode that may prefer already resident experts from within a wider router ranking window. That changes which experts run and is not part of this project's exact-cache design.

Source: [`docs/CACHE_ROUTE.md`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/CACHE_ROUTE.md)

### Negative predictive GPU-staging result

Colibri tested next-layer prediction with GPU staging on a six-RTX-5090 GLM-5.2 system. The project reports that prediction increased potential GPU coverage but reduced end-to-end throughput because staging contended with expert/attention streams. This is a valuable negative result:

- routing predictability can be real;
- high recall does not prove a speedup;
- dedicated streams and synchronization design matter;
- prediction must be benchmarked after a reactive cache works.

Source: [`docs/experiments/glm52-6x5090-2026-07-12.md`](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/experiments/glm52-6x5090-2026-07-12.md)

All Colibri throughput and recall numbers should be treated as project-reported measurements on its specific model and hardware unless independently reproduced.

---

## 8. Proposed memory hierarchy

### Tier 0 — canonical host model

The complete GGUF remains available in host RAM, preferably using normal `llama.cpp` model loading with mmap plus page locking where practical, or allocated host memory where required.

Responsibilities:

- canonical source of exact quantized weights;
- CPU execution source;
- recovery source if staging/device entries are invalidated.

### Tier 1 — pinned/repacked host staging cache

A bounded host cache contains experts likely to be uploaded repeatedly. It may use a different physical packing suitable for CUDA upload while preserving the logical quantized values.

Reasons not to pin the whole model by default:

- pinned memory is a constrained system resource;
- it may reduce OS flexibility;
- only experts likely to move to VRAM need asynchronous-transfer-friendly memory;
- the V100 target has much less VRAM than host RAM.

### Tier 2 — persistent VRAM expert slots

VRAM remaining after normal model allocations becomes a dedicated expert-cache budget.

Reserve capacity for:

- dense/offloaded model tensors;
- KV cache;
- DFlash/draft model state;
- CUDA workspaces and graph allocations;
- server parallel slots;
- safety headroom.

Initial implementation should use per-layer physical pools under one global budget. Per-layer pools simplify tensor shape, slot remapping, and kernel metadata. Later, the number of slots assigned to each layer can be rebalanced by measured marginal value.

---

## 9. Proposed runtime sequence

For each MoE layer:

1. Obtain the exact router IDs.
2. Deduplicate IDs across batch positions.
3. Query cache residency for each complete expert bundle.
4. Classify selected experts as:
   - ready GPU hit;
   - already loading;
   - CPU-execution candidate;
   - promotion candidate;
   - blocked because no safe victim exists.
5. Reserve victims only when their in-flight count is zero.
6. Increment slot generation when reassigning a slot.
7. Repack into pinned staging if needed.
8. Upload promotions asynchronously on a dedicated transfer stream.
9. Execute ready GPU experts.
10. Optionally execute RAM misses concurrently on CPU.
11. Wait only for promoted experts whose output is required.
12. Execute newly ready GPU experts.
13. Reduce all expert outputs using the original router weights.
14. Release in-flight references after GPU completion events.
15. Update frequency, recency, admission, and timing statistics.

### Generation-safe handles

Each slot handle contains:

```text
slot index + generation
```

If a slot is evicted and reused, old asynchronous callbacks or releases cannot modify its new occupant because the generation no longer matches.

### In-flight protection

A ready expert may be referenced by a GPU operation. Its slot cannot be evicted until the backend event for that operation completes and the in-flight count returns to zero.

---

## 10. Cache-policy direction

The recommended initial policy is **second-touch SLRU**:

- probationary segment for newly admitted experts;
- protected segment for experts that prove reuse;
- first miss records admission history but does not allocate a slot;
- second miss admits the expert;
- a hit in probation promotes it to protected;
- prefill first touches do not admit by default;
- frequency decay occurs at request boundaries;
- no eviction of loading or in-flight entries.

Why this is the default candidate:

- plain LRU can be polluted by large prefills;
- static hot sets cannot adapt to a changing repository/task domain;
- unrestricted LFU can preserve stale history indefinitely;
- second-touch admission filters one-off experts before they consume transfer bandwidth and VRAM.

Still retain simple baselines:

- LRU;
- LFU/LFRU;
- ordinary SLRU;
- static hot-set placement.

A complex policy is justified only if it improves actual runtime under identical traces and hardware tests.

---

## 11. Prediction direction

Prediction must only enqueue **low-priority prefetch candidates**. It must never alter router output.

Candidate predictors, in order:

1. same layer, previous token's expert set;
2. per-layer expert transition table;
3. recent frequency/recency ranking;
4. a small learned predictor trained from route traces;
5. Colibri-style next-layer router lookahead where the Laguna graph allows a sufficiently accurate early hidden state.

Required prediction metrics:

```text
recall
precision
useful prefetched bytes
wasted prefetched bytes
prefetches that caused harmful evictions
foreground transfer wait avoided
additional synchronization/queue time
end-to-end tokens per second
```

Prefetch should use a lower admission priority than demand requests. A predicted expert should not displace a high-value ready entry without strong evidence.

---

## 12. Current repository contents

### `README.md`

Contains commands to:

- apply the trace instrumentation to Poolside `llama.cpp`;
- build the runtime;
- collect decode-only or full routing traces;
- simulate cache policies;
- generate static placement plans;
- run repository tests.

### `DESIGN.md`

Contains the full design, invariants, proposed interfaces, runtime sequence, tier model, milestones, and benchmark protocol.

### `tools/apply_trace_patch.py`

An idempotent source transformer designed against:

```text
repository: poolsideai/llama.cpp
branch: laguna
file: ggml/src/ggml-backend.cpp
expected source blob: 87615921c09be5ef8c4996faa70fb3f49c385031
```

It inserts opt-in tracing into the selective expert-copy path and can install the analyzer into the target checkout.

Because Poolside's branch may move, the patcher validates structural anchors. Before using it on a newer revision, inspect the generated diff and update the anchors if the relevant scheduler code changed.

### `tools/moe_trace_support.inc`

Contains the C++ trace helper included by the patched `ggml-backend.cpp`.

Tracing is disabled unless `GGML_MOE_TRACE` is set. Supported environment variables include:

```text
GGML_MOE_TRACE=/path/to/output.jsonl
GGML_MOE_TRACE_DECODE_ONLY=1
GGML_MOE_TRACE_FLUSH=0|1
GGML_MOE_TRACE_MAX_CALLS=N
```

The trace records weight metadata and runtime route sets. It is intended to have no model-output effect.

### `tools/analyze_moe_trace.py`

A dependency-free simulator that can evaluate:

- LRU;
- LFU;
- SLRU;
- second-touch SLRU;
- static hot-set placement.

It reports cache hit rate, estimated transfer bytes, previous-route overlap, and per-layer behavior for specified VRAM budgets and scopes.

This is an **offline policy simulator**, not a performance model. It estimates transfer avoidance but cannot predict CUDA stream interference, CPU/GPU execution overlap, kernel timing, or synchronization costs.

### `prototype/moe_cache_core.h` and `.cpp`

A standalone C++11 state-machine prototype implementing:

- slot states: empty, loading, ready;
- request states: hit, loading, miss without admission, reserved miss, blocked;
- probation/protected SLRU segments;
- second-touch admission;
- prefill admission suppression;
- frequency and recency tracking;
- generation-safe handles;
- in-flight acquire/release protection;
- frequency decay.

It deliberately has no GGML or CUDA dependencies. This allows policy and lifecycle behavior to be tested before device buffers and events are introduced.

### Tests

- `tests/test_analyze_moe_trace.py`
- `tests/test_moe_cache_core.cpp`
- `tests/run_tests.sh`

The package tests passed in the environment where the bootstrap was created. A complete Poolside `llama.cpp` CUDA build was **not** performed there because a full checkout was unavailable. The first user-side task is therefore to validate the patch and build against the actual Poolside branch.

### Repository commits

- `7d89bbe8089a7776c71317938b2a453c094c18b0` — repository initialization.
- `29dd39b6c7bbd80112efbb8254e3556c94a0d6d5` — bootstrap implementation, design, tooling, prototype, and tests.

---

## 13. Immediate runbook

### 13.1 Clone both repositories

```bash
git clone https://github.com/mistrjirka/MOEs-Llama.git

git clone --branch laguna https://github.com/poolsideai/llama.cpp.git poolside-llama.cpp
```

### 13.2 Run bootstrap tests

```bash
cd MOEs-Llama
tests/run_tests.sh
```

### 13.3 Apply tracing

```bash
python tools/apply_trace_patch.py ../poolside-llama.cpp --install-analyzer
```

Inspect before building:

```bash
cd ../poolside-llama.cpp
git diff -- ggml/src/ggml-backend.cpp tools/moe-cache/
```

### 13.4 Build CUDA runtime

Use the user's existing CUDA configuration. A basic example:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

For the V100, preserve the CUDA architecture configuration already known to work in the user's environment rather than introducing an unrelated build change during trace validation.

### 13.5 Collect representative traces

Decode-only trace:

```bash
export GGML_MOE_TRACE="$PWD/laguna-routing-decode.jsonl"
export GGML_MOE_TRACE_DECODE_ONLY=1
export GGML_MOE_TRACE_FLUSH=0

./build/bin/llama-server \
  -m ~/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ2_M.gguf \
  --jinja \
  -fa on \
  --host 127.0.0.1 \
  --port 2345
```

Run multiple real workloads, ideally including:

- repository exploration;
- multi-file bug fixing;
- TypeScript work;
- C++ work;
- long-context analysis;
- tool-heavy agent tasks;
- several unrelated repositories.

Collect a second trace including prefill to measure pollution.

Do not infer routing quality from one prompt.

### 13.6 Simulate budgets and policies

Example:

```bash
python tools/moe-cache/analyze_moe_trace.py \
  laguna-routing-decode.jsonl \
  --policy slru-2touch \
  --vram-gib 6 \
  --scope decode \
  --json laguna-cache-report.json
```

Test several realistic budgets after accounting for current VRAM usage, for example 1, 2, 4, 6, and 8 GiB if available.

Compare policies on identical traces:

```bash
for policy in lru lfu slru slru-2touch; do
  python tools/moe-cache/analyze_moe_trace.py \
    laguna-routing-decode.jsonl \
    --policy "$policy" \
    --vram-gib 6 \
    --scope decode
done
```

### 13.7 Required outputs before M1/M2 design is finalized

Preserve:

- raw JSONL traces;
- model command line;
- exact Poolside commit;
- exact GGUF filename and checksum;
- VRAM usage without the cache;
- policy reports for multiple budgets;
- prefill-inclusive and decode-only comparisons;
- tensor types and expert bytes per layer/tensor.

---

## 14. Next implementation milestones

### M0 — tracing and simulation

**Current status:** implemented in bootstrap, pending real Poolside build and Laguna traces.

Acceptance criteria:

- tracing disabled by default;
- build succeeds on Poolside branch;
- no token/logit change with trace enabled;
- traces parse correctly;
- enough real workloads collected;
- policy curves produced for several budgets.

### M1 — scheduler-owned cache metadata and persistent buffers

Implement a generic cache manager with no graph-execution behavior change yet.

Required pieces:

- cache lifetime owned by scheduler/context;
- per-layer slot pools under a global budget;
- slot layout metadata;
- dedicated backend allocations outside graph-temporary memory;
- loading/ready events;
- generation-safe handles;
- in-flight protection;
- statistics API;
- clean teardown and invalidation.

Recommended files, subject to GGML conventions:

```text
ggml/src/ggml-moe-cache.h
ggml/src/ggml-moe-cache.cpp
ggml/src/ggml-cuda/moe-cache.cu
```

Do not put generic policy in `src/models/laguna.cpp`.

### M2 — reactive exact cache on a simple type

Before IQ2, prove the mechanism on F16 or Q8 fixtures:

- reserve compact slots;
- upload complete bundles;
- remap global expert IDs to slot IDs;
- execute `MUL_MAT_ID` against slot tensors;
- retain data across decode forwards;
- protect entries during execution;
- evict safely;
- compare cache-off/cache-on logits;
- record actual transfer bytes.

This isolates cache mechanics from complex quantized padding.

### M3 — Laguna `UD-IQ2_M` support

Add a quantization-aware codec that derives each source tensor's actual `ggml_type`.

Required validation:

- exact block layout;
- alignment;
- tail padding;
- MMQ over-read behavior;
- gate/up/down offsets within one bundle allocation;
- CUDA compute-sanitizer/memcheck;
- deterministic token and logit comparison;
- actual H2D-byte counters;
- warm/cold cache A/B throughput.

Do not assume the GGUF uses one tensor type everywhere merely because its filename contains `IQ2_M`.

### M4 — hybrid CPU/GPU expert execution

Split selected experts between execution tiers:

- cached hits execute on GPU;
- one-off misses may execute on CPU;
- likely reusable misses may upload and execute on GPU;
- CPU and GPU outputs are reduced using unchanged router weights.

This is architecturally harder than a transfer-only cache because it may require graph partitioning and mixed-backend output reduction. Implement only after M3 proves persistent GPU hits.

### M5 — predictive prefetch

Add prediction as a separate low-priority producer after reactive-cache measurements exist.

Start with simple baselines before a learned model. Every predictor must be compared to no-prefetch with identical cache size and policy.

---

## 15. Benchmark and correctness protocol

No performance claim should rely only on theoretical PCIe bandwidth or latency. Each claim must come from a controlled end-to-end A/B test.

For every runtime change:

- use the same executable where runtime flags permit;
- record exact commit and build flags;
- use the same model file and checksum;
- use identical prompt tokens;
- generate the same number of output tokens;
- separate prompt processing and token generation;
- benchmark cold and warm cache states separately;
- perform at least three runs;
- report median and range;
- collect cache hit/miss rate;
- collect H2D bytes;
- collect staging/repack time;
- collect visible transfer wait;
- collect CPU expert time;
- collect GPU expert time;
- collect total decode tokens/s;
- compare logits or deterministic output before interpreting speed.

### Prediction-specific protocol

Additionally report:

- prediction recall and precision;
- useful versus wasted bytes;
- harmful evictions caused by prefetch;
- cache hit rate with and without prediction;
- transfer-stream overlap;
- attention/expert stream interference;
- total tokens/s.

A microbenchmark is useful for diagnosing kernels but cannot replace end-to-end decode results.

---

## 16. Main technical risks

### 16.1 Quantized slot correctness

IQ2/MMQ kernels may require hidden padding and layout assumptions. This is the highest-risk implementation area.

### 16.2 Scheduler and graph lifetime

Temporary graph buffers are not valid persistent cache storage. Reallocations, parallel copies, and server slots may invalidate naive pointer-based caches.

### 16.3 Multiple server sequences

Continuous batching and multiple slots may interleave routes from different requests. Cache policy can be shared globally, but in-flight tracking and statistics must remain correct under concurrent graph copies.

### 16.4 Prefill pollution

Large prompt batches can activate far more unique experts than single-token decode. Admission must distinguish phases.

### 16.5 Transfer interference

A dedicated CUDA stream does not automatically guarantee overlap. Transfers can still compete for copy engines, memory bandwidth, synchronization, or kernel launch ordering.

### 16.6 CPU/GPU output integration

Executing some experts on CPU and some on GPU may require nontrivial graph changes and could introduce synchronization that erases the expected gain.

### 16.7 DFlash interaction

The draft and verification paths may have different expert routes and may multiply cache pressure. Cache statistics should distinguish main, draft, and verification forwards. Do not enable prediction and DFlash changes simultaneously during initial cache validation.

### 16.8 Static workload overfitting

A hot set derived from one repository or task may fail on another. Traces must cover varied coding workloads.

### 16.9 Online adaptation instability

Aggressive repinning can spend more time moving experts than it saves. Use hysteresis, admission gating, and bounded promotions.

---

## 17. Open design questions

These should be answered using traces and runtime measurements rather than assumptions:

1. How much expert overlap exists between adjacent decode tokens per layer?
2. Is locality mostly global-hot, session-hot, or short-term transition-based?
3. How many useful slots fit after dense tensors, KV, DFlash, and workspace reservations?
4. Does second-touch SLRU outperform static hot sets and simple LRU?
5. Which layers gain the most marginal hit rate per GiB?
6. Are gate/up/down tensor types and sizes uniform enough for one codec per model, or must each tensor be described independently?
7. How expensive is host repacking for the actual IQ2 layouts?
8. Is pageable-host transfer already sufficiently overlapped, or is pinned staging essential?
9. On the V100/EPYC system, when is CPU execution faster than upload plus GPU execution for one expert?
10. Does a reactive cache improve throughput before prediction?
11. Can prediction use a transfer stream without reducing attention/expert throughput?
12. Should DFlash and main-model expert caches share a budget or remain separate?
13. How should cache state behave across multiple concurrent server slots?
14. Is per-layer fixed capacity sufficient, or is live global rebalancing worth the complexity?

---

## 18. Source index

### Runtime and model sources

- [Poolside `llama.cpp` fork](https://github.com/poolsideai/llama.cpp)
- [Upstream `llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`ggml-backend.cpp` selective expert-copy implementation](https://github.com/ggml-org/llama.cpp/blob/298219f985b09d93df515ca708736665468ca827/ggml/src/ggml-backend.cpp)
- [`src/models/laguna.cpp`](https://github.com/ggml-org/llama.cpp/blob/95a923a64c7d493ed1cb347d3b55d039fa3b8097/src/models/laguna.cpp)
- [Poolside Laguna S 2.1 model page](https://huggingface.co/poolside/Laguna-S-2.1)
- [Unsloth Laguna S 2.1 GGUF repository](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)

### Previous `llama.cpp` expert-cache work

- [Issue #20757 — two-tier GPU+RAM expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [PR #21609 — N-slot LFRU cache with FATE prefetch, withdrawn](https://github.com/ggml-org/llama.cpp/pull/21609)
- [PR #21614 — persistent expert cache, withdrawn](https://github.com/ggml-org/llama.cpp/pull/21614)

### Colibri references

- [Colibri repository](https://github.com/JustVugg/colibri)
- [Colibri README at inspected revision](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/README.md)
- [CUDA and VRAM tier](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/cuda.md)
- [Runtime tuning, learning cache, repinning, and PILOT](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/tuning.md)
- [Cache-aware routing experiment](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/CACHE_ROUTE.md)
- [Tier policy code](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/c/tier.h)
- [Six-RTX-5090 experiment, including negative GPU-prediction result](https://github.com/JustVugg/colibri/blob/81f08a09e5651ce52616dc720f68810f9021c0be/docs/experiments/glm52-6x5090-2026-07-12.md)

### This project

- [MOEs-Llama repository](https://github.com/mistrjirka/MOEs-Llama)
- [`README.md`](README.md)
- [`DESIGN.md`](DESIGN.md)
- [`MANIFEST.md`](MANIFEST.md)
- [`tools/apply_trace_patch.py`](tools/apply_trace_patch.py)
- [`tools/moe_trace_support.inc`](tools/moe_trace_support.inc)
- [`tools/analyze_moe_trace.py`](tools/analyze_moe_trace.py)
- [`prototype/moe_cache_core.h`](prototype/moe_cache_core.h)
- [`prototype/moe_cache_core.cpp`](prototype/moe_cache_core.cpp)
- [`tests/run_tests.sh`](tests/run_tests.sh)

---

## 19. Instructions for the next coding agent

Start by reading this file, `DESIGN.md`, and `README.md`. Do not begin predictive loading or write a Laguna-specific cache directly in `src/models/laguna.cpp`.

The immediate task is:

> Validate M0 against the current Poolside Laguna branch, collect representative Laguna routing traces, and produce policy/budget reports. Then design M1 around dedicated scheduler-owned backend buffers without yet changing model output.

Required working rules:

1. Preserve exact router decisions.
2. Preserve logical quantized weights in default mode.
3. Treat gate/up/down as one atomic expert bundle.
4. Never trust graph-temporary addresses as persistent cache storage.
5. Derive tensor types, sizes, alignment, and padding from actual runtime metadata.
6. Keep cache policy independent from quantization codecs.
7. Keep prediction independent from correctness.
8. Add instrumentation before claiming performance.
9. Validate output equality before throughput.
10. Record negative results instead of tuning around them silently.

Before implementing M2/M3, return with:

- exact Poolside commit used;
- successful build command;
- trace files or summarized reports;
- per-layer expert sizes/types;
- cache curves for several budgets and policies;
- measured free VRAM under the intended Laguna+DFlash server configuration;
- any patcher conflicts caused by upstream changes.

---

## 20. Current bottom line

The research supports the experiment but does not yet prove a speedup on the target V100 system.

What is established from source inspection:

- current `llama.cpp` selectively transfers routed experts but lacks the proposed persistent managed cache;
- previous implementations explored the same space but were not merged;
- compact quantized slots require more than naïve row copying;
- Colibri shows that learned RAM/VRAM placement and routing prediction are practical to experiment with;
- Colibri also provides direct evidence that predictive GPU staging can regress despite useful prediction recall.

The most defensible next step is therefore **not prediction**. It is:

> Collect real Laguna traces, prove a persistent reactive cache with exact semantics, and measure whether reused experts avoid enough real transfer/execution cost to justify the architecture.
