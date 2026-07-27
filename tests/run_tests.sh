#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m py_compile \
  "$ROOT/tools/apply_trace_patch.py" \
  "$ROOT/tools/analyze_moe_trace.py" \
  "$ROOT/tools/moe_predictor_tournament_and_allocator.py" \
  "$ROOT/tools/run_vram_capacity_search.py" \
  "$ROOT/tests/test_analyze_moe_trace.py" \
  "$ROOT/tests/test_moe_predictor_tournament_and_allocator.py" \
  "$ROOT/tests/test_vram_capacity_search.py"

python "$ROOT/tests/test_analyze_moe_trace.py"
python "$ROOT/tests/test_moe_predictor_tournament_and_allocator.py"
python "$ROOT/tests/test_vram_capacity_search.py"

g++ -std=c++11 -Wall -Wextra -Werror \
  "$ROOT/prototype/moe_cache_core.cpp" \
  "$ROOT/tests/test_moe_cache_core.cpp" \
  -o /tmp/test_moe_cache_core
/tmp/test_moe_cache_core

echo "all tests passed"
