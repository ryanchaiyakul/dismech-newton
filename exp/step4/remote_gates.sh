#!/bin/bash
# lagged + proxy iterations over the 4 GPUs (sci6000). Usage: remote_gates.sh ITER TAG [extra env]
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
it=$1; tag=$2; shift 2; out=exp/step4/logs/$tag; mkdir -p $out
run() { gpu=$1; name=$2; shift 2; env CUDA_VISIBLE_DEVICES=$gpu DM_ITER=$it "$@" > $out/$name.log 2>&1; echo "$name exit $?" >> $out/summary.txt; }
: > $out/summary.txt
run 0 coupling "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=line -p no:cacheprovider &
(for i in 1 2; do run 1 tramp$i "$@" uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
(for i in 3 4; do run 2 tramp$i "$@" uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
run 3 suite "$@" uv run pytest -q --tb=line -p no:cacheprovider &
wait
echo done >> $out/summary.txt
