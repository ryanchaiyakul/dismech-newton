#!/bin/bash
# Gate 7 preview: Franka finger pads (DM_PADS) under the coupling variants, one config per GPU (sci6000).
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/pads/logs/coupled; mkdir -p "$o"
cfg() { gpu=$1; tag=$2; p=$3; shift 3
  e() { env PASSES=$p DM_PASSES=$p DM_ITER=$p PEND=$([ $p = 1 ] && echo 1 || echo 0) CUDA_VISIBLE_DEVICES=$gpu "$@"; }
  e "$@" uv run python exp/principled/chatter.py > "$o/${tag}_chatter.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -k "momentum or pin" -q --tb=line -p no:cacheprovider > "$o/${tag}_mom.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -k "fold or midas" -q --tb=line -p no:cacheprovider > "$o/${tag}_fold.log" 2>&1 &
  wait; }
cfg 0 robin3_kh5e8 3 DM_ROBIN=2 DM_PADS=5e8,0.002,finger &
cfg 1 robin1_kh5e8 1 DM_ROBIN=2 DM_PADS=5e8,0.002,finger &
cfg 2 robin3_kh5e7 3 DM_ROBIN=2 DM_PADS=5e7,0.002,finger &
cfg 3 lagged1_kh5e8 1 DM_PADS=5e8,0.002,finger &
wait
for t in robin3_kh5e8 robin1_kh5e8 robin3_kh5e7 lagged1_kh5e8; do echo "=== $t"; grep -m1 DM_PADS "$o/${t}_chatter.log"; grep -A3 ^PASSES "$o/${t}_chatter.log"
  for g in mom fold; do grep -E "passed|failed|Error" "$o/${t}_$g.log" | grep -v Warn | sort -u | tail -4; done; done
