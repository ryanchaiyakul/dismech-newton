#!/bin/bash
# Lagged x1 (no Robin) + pads on every rigid body's shapes (DM_PADS=kh,), all coupling gates (sci6000).
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/pads/logs/lagged; mkdir -p "$o"
cfg() { gpu=$1; kh=$2
  e() { env PASSES=1 DM_PASSES=1 DM_ITER=1 PEND=1 DM_PADS=$kh, CUDA_VISIBLE_DEVICES=$gpu "$@"; }
  e uv run examples/newton/franka_cable_ik_pick_place.py --viewer null --test > "$o/k${kh}_pick.log" 2>&1 &
  e uv run python exp/principled/chatter.py > "$o/k${kh}_chatter.log" 2>&1 &
  e uv run python exp/step4/bounce.py > "$o/k${kh}_bounce.log" 2>&1 &
  e uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=line -p no:cacheprovider > "$o/k${kh}_coupling.log" 2>&1 &
  wait; }
tramp() { gpu=$1; kh=$2; for i in 1 2; do env PEND=1 DM_PADS=$kh, CUDA_VISIBLE_DEVICES=$gpu uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test > "$o/k${kh}_tramp$i.log" 2>&1; done; }
cfg 0 1e8 & cfg 1 1e9 & tramp 2 1e8 & tramp 3 1e9 &
wait
for k in 1e8 1e9; do echo "=== kh $k"; grep -h -m1 DM_PADS "$o/k${k}_pick.log"; grep -A3 ^PASSES "$o/k${k}_chatter.log" | cut -c1-60
  for g in pick bounce coupling tramp1 tramp2; do echo "-- $g: $(grep -hE "passed|failed|Error|BOUNCE|grip" "$o/k${k}_$g.log" | grep -v Warn | sort -u | tail -6 | tr "\n" " " | cut -c1-500)"; done; done
