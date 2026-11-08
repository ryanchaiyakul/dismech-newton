#!/bin/bash
# lagged iteration convergence on the failing coupling tests: one config per GPU (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/step4/logs/iters; mkdir -p "$o"
K="momentum or pin or pick or swap or midas or fold"
run() { CUDA_VISIBLE_DEVICES=$1 DM_ITER=$2 DM_RELAX=$3 PEND=0 uv run pytest tests/test_coupling.py -m "slow or not slow" -k "$K" -q --tb=line -p no:cacheprovider > "$o/iter$2_$3.log" 2>&1; }
run 0 2 fixed & run 1 3 fixed & run 2 4 fixed & run 3 4 aitken & wait
for f in "$o"/*.log; do echo "== $f"; grep -E "passed|failed|AssertionError:" "$f" | grep -v Warn; done
