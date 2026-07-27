# GLM semi-static vertical placement experiment

## Goal

Beat llama.cpp's default horizontal placement on the same GLM model, prompt,
forced token stream, context size, build, and hardware. Use the maximum
**productive** VRAM: the smallest reserve that survives the worst supported
context/workspace workload with at least 512 MiB measured free memory.

## Architectural experiment

The runtime implementation should preserve the frozen map's graph topology:

1. Two canonical-expert-to-slot maps per routed layer (`map_A`, `map_B`).
2. A monotonically increasing map epoch and slot generation.
3. Publish the inactive map only after complete gate/up/down bundles are READY.
4. Produce hot slot IDs and cold canonical IDs on GPU once per layer, reused by
   gate, up, and down.
5. No CPU route callback or per-token graph reconstruction in steady decode.
6. Initialize from the demonstrated static frequency map.
7. Protect the static core for the request epoch; initially reserve two adaptive
   tail slots per layer.
8. Review adaptation every 16 tokens and allow one complete bundle migration
   globally per review. Increase only after the topology target is met.

Do not integrate a new predictor into the first live arm. Use decayed recent
frequency, hysteresis, and a 32-token minimum residence so the map mechanism is
measured independently of prediction quality.

## VRAM search

Run full-context multi-token probes in this order:

```text
4096, 2048, 1536, 1024, 768, 512 MiB reserve
```

Use `tools/run_vram_capacity_search.py`. A reserve is invalid on non-zero exit,
timeout, an OOM signature, or measured free VRAM below the guard. Do not infer
safety from model-load success; compact buffers must allocate and decode must
complete.

After finding the smallest safe reserve, keep that exact byte budget in every
vertical arm. Unequal per-layer allocation may use the budget differently, but
must not receive more bytes than the equal baseline.

## Controlled arms

| Arm | Description |
|---|---|
| H | Default llama.cpp horizontal placement |
| S | Existing frozen exact static vertical map |
| M0 | A/B map runtime with updates disabled |
| M1 | Protected static core plus two tail slots/layer, one migration/16 tokens |
| M4 | Only after M1 succeeds: four tails/layer, up to four migrations/16 tokens |

Run each arm on the original prompt, a different coding prompt, and the longer
shifted continuation. Use at least 64 decode tokens of warmup and three measured
repetitions. Throughput runs must disable detailed profiling and trace output.

## Hard success gates

- Exact router decisions are preserved.
- Token equality and full-logit absolute/relative tolerance pass.
- No stale slot generation or map epoch under forced rapid reuse.
- GLM decode graph remains at or below 160 splits.
- M0 retains at least 98% of frozen-map throughput.
- M1 retains at least 95% of frozen-map throughput.
- The final selected arm beats horizontal mean throughput on at least the
  original and different-prompt workloads.
- Report peak VRAM, minimum free VRAM, transfers/token, queue delay,
  resident-route coverage, executed GPU-route coverage, and complete GPU layers.

## Predictor and allocation follow-up

After the topology gate passes:

1. Verify predictor storage independence by target layer, horizon, and sequence.
2. Resolve physical layer 6..77 to perceptron head 0..71 with assertions.
3. Evaluate separate +1/+2/+3 histories without look-ahead leakage.
4. Integrate the best fixed per-layer conditional-plus-perceptron blend first.
5. Add a tournament chooser only if its event-wise oracle materially beats that
   fixed blend.
6. Compare equal allocation with complete-layer and measured-latency objectives
   under the identical VRAM byte budget.
