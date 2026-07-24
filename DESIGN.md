# Persistent MoE expert cache design

## Decision

Use **Poolside's Laguna-capable `llama.cpp` fork** as the implementation base and keep the cache inside generic GGML/backend code. Colibri is the policy and measurement reference, not the runtime base.

The implementation must support Laguna first but must not contain Laguna-specific tensor names, layer counts, expert counts, or quantization assumptions.

## Goals

1. Keep the complete GGUF model resident in host RAM when sufficient RAM exists.
2. Use otherwise-free VRAM as a persistent expert cache after reserving dense weights, KV cache, DFlash, workspaces, and a safety margin.
3. Run cached experts on GPU without retransferring their weights.
4. Handle cache misses either by CPU execution or asynchronous promotion to GPU, based on measured cost.
5. Preserve the router's selected experts and numerical weight representation by default.
6. Permit different *physical packings* in GGUF, pinned host staging, and device slots.
7. Make prediction optional and strictly separable from correctness.

## Non-goals for the first functional cache

- Disk streaming. The intended first target has the complete model in RAM.
- Cache-aware routing that substitutes experts.
- Cross-GPU sharding of one expert.
- Automatic transquantization to a numerically different format.
- Prediction before a reactive cache is measured.

## Correctness invariants

### Expert bundles are atomic

A routed expert consists of gate, up, and down matrices. They share one residency state and one slot assignment:

```text
(layer, expert) -> {gate, up, down}
```

A bundle is considered resident only after all three uploads complete.

### Exact mode never changes the model function

A cache hit and miss must use the same logical quantized weights. Repacking is allowed; changing IQ2 weights into Q4 weights is not exact and must be an explicit experimental mode.

### IDs are remapped only after residency is guaranteed

The router produces global expert IDs. A GPU operation receives slot IDs only after every referenced slot has completed upload and is protected from eviction until the operation finishes.

### Graph allocator memory is not persistent cache memory

The existing graph allocator reuses temporary input-copy addresses among layers. Cache slots must therefore use backend allocations whose lifetime is owned by the cache manager, not by one graph execution.

### Prefill and decode have different admission behavior

Large prefills can touch most experts and destroy a useful decode cache. Default behavior should be:

- allow cache hits during prefill;
- do not admit first-touch prefill misses;
- optionally admit experts seen repeatedly in the same prefill;
- protect established decode entries with SLRU or frequency-gated admission.

## Memory hierarchy

```text
Tier 0: canonical GGUF host storage
        complete model; mmap+mlock or allocated host memory

Tier 1: pinned/repacked host staging cache
        bounded; only experts likely to be uploaded repeatedly

Tier 2: persistent VRAM expert slots
        dedicated backend buffers; quant-aware padding and alignment
```

Host RAM is also an execution tier, not merely a backing store. A miss can be computed on CPU instead of forcing an upload.

## Core interfaces

Proposed conceptual API; naming should be adapted to GGML conventions.

```cpp
struct ggml_moe_expert_key {
    int32_t layer;
    int32_t expert;
};

struct ggml_moe_expert_layout {
    ggml_type type;
    size_t logical_bytes;
    size_t slot_bytes;
    size_t alignment;
    size_t tail_padding;
};

struct ggml_moe_device_slot {
    int32_t slot;
    ggml_moe_expert_key owner;
    uint64_t generation;
    ggml_backend_event_t ready;
    uint32_t in_flight;
};

struct ggml_moe_cache_policy {
    bool admit(const access_context &);
    int32_t select_victim(const layer_cache &);
    void on_hit(...);
    void on_miss(...);
    void on_load_complete(...);
};

struct ggml_moe_expert_codec {
    bool supports(ggml_type type, ggml_backend_t backend) const;
    ggml_moe_expert_layout layout(const ggml_tensor & tensor) const;
    upload_ticket upload_bundle_async(...);
    device_expert_view make_view(...);
};
```

The policy cannot access quantized bytes. The codec cannot choose eviction victims.

## Runtime sequence per MoE layer

1. Obtain the exact router IDs.
2. Deduplicate IDs across batch positions.
3. Query bundle residency.
4. Classify selected experts:
   - resident GPU hit;
   - CPU execution candidate;
   - promotion candidate.
5. Reserve slots for promotions without evicting in-flight bundles.
6. Start asynchronous host-to-device uploads on a dedicated transfer stream.
7. Compute GPU hits and CPU misses where overlap is profitable.
8. Wait only for promotions whose output is required.
9. Execute newly resident experts.
10. Reduce CPU and GPU expert outputs with the original router weights.
11. Release in-flight references after backend events complete.
12. Update admission and heat statistics.

## CPU versus GPU decision

Do not encode a universal rule such as "always upload on miss." Maintain online exponentially weighted measurements per quantization and batch shape:

```text
CPU expert execution time
host repack time
H2D upload time
GPU expert execution time
queueing/stream interference
estimated reuse probability
```

Promote when expected future savings exceed promotion cost within a bounded horizon. The first implementation may use a simpler threshold, but the API should accept the measured cost model later.

## Device slot layout

A slot is not simply `expert_size` bytes. Quantized kernels may read tail padding, and backend allocation alignment can exceed GGUF row alignment.

For each of gate/up/down:

```text
aligned matrix bytes
+ backend-required tail padding
```

The bundle slot should preferably be one allocation with fixed offsets for all three matrices to make admission atomic and reduce allocator/event overhead.

For IQ2 and other MMQ formats, validate guard bytes with CUDA memcheck before enabling compact slots. The withdrawn llama.cpp prototype's illegal accesses show that copying a logical row into an unpadded N-slot tensor is insufficient.

## Slot allocation across layers

Use a global VRAM budget but per-layer physical slot pools initially.

Advantages:

- stable tensor shapes and kernel metadata;
- no cross-layer type/shape compatibility problem;
- simpler ID remapping;
- independent layer policies.

Allocate slots using measured marginal value rather than equally in the final version:

```text
marginal value = avoided miss bytes or time from adding one slot
                 -----------------------------------------------
                              slot bytes
```

The included simulator starts with deterministic equal-round allocation and will be extended after real traces arrive.

## Cache policy

Recommended initial policy:

- SLRU: 20% probationary, 80% protected;
- second-touch admission;
- frequency decay at request boundaries;
- no first-touch prefill admission;
- maximum promotions per token/request boundary;
- hysteresis before layer budget rebalancing.

Also retain LRU and static-hot policies as baselines. A complicated policy is unjustified unless it beats them on identical traces and runtime A/B tests.

## Prediction

Prediction is a producer of low-priority prefetch requests, not part of cache correctness.

Order of experiments:

1. last-token same-layer expert set;
2. per-layer transition table;
3. small predictor trained from routing traces;
4. Colibri-style next-layer router lookahead where architecture permits it.

Every prediction report must include:

```text
recall
precision
bytes usefully prefetched
bytes wasted
prefetches that displaced subsequently used entries
foreground wait avoided
end-to-end tokens/s
```

A predictor can have high recall and still reduce throughput through stream contention or cache pollution.

## Implementation milestones

### M0 — tracing and simulator

Included in this package.

Acceptance criteria:

- tracing disabled by default;
- no output-token change;
- route and weight metadata parse correctly;
- representative coding workloads captured;
- cache-policy curves produced for several VRAM budgets.

### M1 — cache metadata and persistent buffers

- cache manager owned by scheduler/context;
- dedicated backend buffers;
- no graph-memory aliasing;
- slot state machine and events;
- no graph execution change yet.

### M2 — reactive exact cache for a simple format

Start with F16/Q8 fixture to prove:

- slot ID remapping;
- persistent hits;
- eviction safety;
- byte-identical logits within the same kernel path.

### M3 — Laguna IQ2 support

- quant-specific slot codec;
- required padding/alignment;
- CUDA memcheck;
- compare cache-off/cache-on logits and generated tokens;
- measure actual H2D bytes and visible wait.

### M4 — hybrid CPU/GPU misses

- split selected experts between CPU and GPU;
- overlap execution;
- measured cost model.

### M5 — prefetch and prediction

Only after M3/M4 show a useful reactive cache.

## Required benchmark protocol

For every change:

- same binary where runtime flags permit;
- same model file;
- same prompt tokens and generated-token count;
- warm and cold cache separately;
- at least three repetitions;
- report median and range;
- separate prefill and decode;
- record cache hit rate, H2D bytes, CPU expert time, GPU expert time, transfer wait, and total tokens/s;
- verify logits or deterministic output equality before performance claims.
