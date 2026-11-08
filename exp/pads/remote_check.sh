#!/bin/bash
# regressions (no pads): suite + MIDAS lagged; pick-place example test with pads (lagged)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/pads/logs/check; mkdir -p "$o"
CUDA_VISIBLE_DEVICES=3 uv run pytest -q --tb=line -p no:cacheprovider > "$o/suite.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 PEND=1 uv run pytest tests/test_coupling.py -m "slow or not slow" -k midas -q --tb=line -p no:cacheprovider > "$o/midas_nopad.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 DM_PADS=5e8,0.002,finger uv run examples/newton/franka_cable_ik_pick_place.py --viewer null --test > "$o/pick_pads.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 uv run examples/newton/franka_cable_ik_pick_place.py --viewer null --test > "$o/pick_nopad.log" 2>&1 &
wait; echo done
