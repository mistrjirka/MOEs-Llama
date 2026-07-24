#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m py_compile \
  "$ROOT/tools/apply_trace_patch.py" \
  "$ROOT/tools/analyze_moe_trace.py" \
  "$ROOT/tests/test_analyze_moe_trace.py"

python "$ROOT/tests/test_analyze_moe_trace.py"

g++ -std=c++11 -Wall -Wextra -Werror \
  "$ROOT/prototype/moe_cache_core.cpp" \
  "$ROOT/tests/test_moe_cache_core.cpp" \
  -o /tmp/test_moe_cache_core
/tmp/test_moe_cache_core

echo "all tests passed"
