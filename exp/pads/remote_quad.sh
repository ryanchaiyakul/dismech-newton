#!/bin/bash
# Gate 7 preview: Franka finger pads (DM_PADS) under the coupling variants, one config per GPU (sci6000).
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/pads/logs/quad; mkdir -p "$o"
cfg() { gpu=$1; tag=$2; p=$3; shift 3
  e() { env PASSES=$p DM_PASSES=$p DM_ITER=$p PEND=$([ $p = 1 ] && echo 1 || echo 0) CUDA_VISIBLE_DEVICES=$gpu "$@"; }
  e "$@" uv run python exp/principled/chatter.py > "$o/${tag}_chatter.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -k midas -q --tb=line -p no:cacheprovider > "$o/${tag}_mom.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -k fold -q --tb=line -p no:cacheprovider > "$o/${tag}_fold.log" 2>&1 &
  e "$@" uv run examples/newton/franka_cable_ik_pick_place.py --viewer null --test > "$o/${tag}_pick.log" 2>&1 &
  wait; }
cfg 0 robin3_k9 3 DM_ROBIN=2 DM_PADS=1e9,finger &
cfg 1 robin1_k9 1 DM_ROBIN=2 DM_PADS=1e9,finger &
cfg 2 lagged1_k9 1 DM_PADS=1e9,finger &
cfg 3 lagged1_k8 1 DM_PADS=1e8,finger &
wait
for t in robin3_k9 robin1_k9 lagged1_k9 lagged1_k8; do echo "=== $t"; grep -m1 DM_PADS "$o/${t}_chatter.log"; grep -A3 ^PASSES "$o/${t}_chatter.log"
  for g in fold pick mom; do grep -E "passed|failed|Error" "$o/${t}_$g.log" | grep -v Warn | sort -u | tail -4; done; done
