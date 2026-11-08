#!/bin/bash
# the handoff's gates for one variant over the 4 GPUs (sci6000). Usage: remote_gates.sh SCHUR PASSES TAG [env...]
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
s=$1; p=$2; tag=$3; shift 3; out=exp/principled/logs/$tag; mkdir -p $out
run() { gpu=$1; name=$2; shift 2; env CUDA_VISIBLE_DEVICES=$gpu DM_SCHUR=$s DM_PASSES=$p DM_ITER=$p PEND=$([ $p = 1 ] && echo 1 || echo 0) "$@" > $out/$name.log 2>&1; echo "$name exit $?" >> $out/summary.txt; }
: > $out/summary.txt
run 0 coupling "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=line -p no:cacheprovider &
(run 1 bounce "$@" uv run python exp/step4/bounce.py; for i in 1 2; do run 1 tramp$i "$@" uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
(for i in 3 4; do run 2 tramp$i "$@" uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test; done) &
run 3 suite "$@" uv run pytest -q --tb=line -p no:cacheprovider &
wait
echo done >> $out/summary.txt
