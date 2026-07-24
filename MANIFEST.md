# Package contents

- `README.md` — application, tracing, and analysis commands.
- `DESIGN.md` — runtime architecture and phased implementation design.
- `tools/apply_trace_patch.py` — idempotent source transformer for Poolside llama.cpp.
- `tools/analyze_moe_trace.py` — dependency-free routing/cache simulator.
- `prototype/moe_cache_core.h` — cache-state API prototype.
- `prototype/moe_cache_core.cpp` — second-touch SLRU implementation.
- `tests/test_analyze_moe_trace.py` — simulator test.
- `tests/test_moe_cache_core.cpp` — cache-state tests.
- `tests/run_tests.sh` — complete local test runner.
