#!/bin/bash
# DM_AHEAD 1 / 2 gates over the 4 GPUs (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
out=exp/impact/logs/gates; mkdir -p $out; : > $out/summary.txt
run() { gpu=$1; name=$2; a=$3; shift 3; env CUDA_VISIBLE_DEVICES=$gpu DM_AHEAD=$a "$@" > $out/$name.log 2>&1; echo "$name exit $?" >> $out/summary.txt; }
run 0 coupling_a1 1 uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=short -p no:cacheprovider -rxXs &
run 1 coupling_a2 2 uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=short -p no:cacheprovider -rxXs &
(run 2 trace_a1 1 uv run exp/impact/trace.py; for i in 1 2; do run 2 tramp_a1_$i 1 uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
(run 3 trace_a2 2 uv run exp/impact/trace.py; for i in 1 2; do run 3 tramp_a2_$i 2 uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
wait
echo done >> $out/summary.txt
