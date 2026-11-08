#!/bin/bash
# Robin in our harvest (DM_ROBIN=2, no probes): grip chatter + bounce, passes 1 (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/robin2; mkdir -p "$o"
CUDA_VISIBLE_DEVICES=2 DM_ROBIN=2 PASSES=1 uv run python exp/principled/chatter.py > "$o/chatter.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 DM_ROBIN=2 DM_ITER=1 uv run python exp/step4/bounce.py > "$o/bounce.log" 2>&1 &
wait
grep -A3 "^PASSES" "$o/chatter.log"; grep -E "Traceback|Error" -A4 "$o"/*.log | tail -8; tail -1 "$o/bounce.log"
