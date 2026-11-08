#!/bin/bash
# A1 stiffness sweep: iterated Robin x3 (DM_ROBIN=2) per barrier config, one config per GPU (sci6000).
# Usage: remote_stiff.sh  -> logs/stiff/<tag>_{chatter,bounce,mom,fold}.log
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/stiff; mkdir -p "$o"
cfg() { gpu=$1; tag=$2; shift 2
  e() { env DM_ROBIN=2 DM_PASSES=3 DM_ITER=3 PASSES=3 PEND=0 CUDA_VISIBLE_DEVICES=$gpu "$@"; }
  e "$@" uv run python exp/principled/chatter.py > "$o/${tag}_chatter.log" 2>&1 &
  e "$@" uv run python exp/step4/bounce.py > "$o/${tag}_bounce.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -k "momentum or pin" -q --tb=line -p no:cacheprovider > "$o/${tag}_mom.log" 2>&1 &
  e "$@" uv run pytest tests/test_coupling.py -m "slow or not slow" -k fold -q --tb=line -p no:cacheprovider > "$o/${tag}_fold.log" 2>&1 &
  wait; }
cfg 0 k1e3 DM_STIFF=1e3 &
cfg 1 k1e2 DM_STIFF=1e2 &
cfg 2 k1e1 DM_STIFF=1e1 &
cfg 3 sq1e3 DM_STIFF=1e3 DM_ARM_FLEX=0.436 &
wait
for t in k1e3 k1e2 k1e1 sq1e3; do echo "=== $t"; grep -A3 ^PASSES "$o/${t}_chatter.log"; tail -1 "$o/${t}_bounce.log"
  for g in mom fold; do grep -E "passed|failed|Error" "$o/${t}_$g.log" | grep -v Warn | sort -u | tail -3; done; done
