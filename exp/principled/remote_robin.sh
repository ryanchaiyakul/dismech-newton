#!/bin/bash
# Robin exchange: grip chatter by phase + bounce, (proxy inertia, passes) (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/robin; mkdir -p "$o"
grip() { CUDA_VISIBLE_DEVICES=$1 DM_ROBIN=1 DM_SCHUR=$2 DM_SCHUR_EPS=0.3 PASSES=$3 uv run python exp/principled/chatter.py > "$o/chatter_$2_p$3.log" 2>&1; }
bounce() { CUDA_VISIBLE_DEVICES=$1 DM_ROBIN=$4 DM_SCHUR=$2 DM_SCHUR_EPS=0.3 DM_ITER=$3 uv run python exp/step4/bounce.py > "$o/bounce_$2_p$3_r$4.log" 2>&1; }
grip 0 probe 1 & grip 1 full 1 & grip 2 probe 2 & grip 3 full 2 &
bounce 0 probe 1 1 & bounce 1 full 1 1 & bounce 2 probe 1 0 & bounce 3 probe 2 1 & wait
for f in "$o"/chatter_*.log; do echo "== $f"; grep -A3 "^PASSES" "$f"; grep -iE "error|Traceback" -A3 "$f" | tail -5; done
for f in "$o"/bounce_*.log; do echo "== $f"; tail -4 "$f"; done
