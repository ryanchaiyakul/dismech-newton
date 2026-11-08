#!/bin/bash
# staggered gates on sci6000: DM_EXTRAP=$1 -> logs/staggered_extrap$1; suite on GPU $2, trampolines x4 on GPU $3
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/step4/logs/staggered_extrap$1; mkdir -p "$o"; : > "$o/summary.txt"
export DM_MODE=staggered DM_EXTRAP=$1 DM_ITER=1
(CUDA_VISIBLE_DEVICES=$2 uv run pytest -m "slow or not slow" -q --tb=line -p no:cacheprovider > "$o/suite.log" 2>&1; echo "suite exit $?" >> "$o/summary.txt") &
(for i in 1 2 3 4; do CUDA_VISIBLE_DEVICES=$3 uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test > "$o/tramp$i.log" 2>&1; echo "tramp$i exit $?" >> "$o/summary.txt"; done) &
wait
cat "$o/summary.txt"; grep -hE "passed|failed|AssertionError:" "$o"/*.log | grep -v Warn | sort | uniq -c
