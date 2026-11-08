#!/bin/bash
# idea 1 on the pick-place grip: grip_trace per Schur mode and passes, several per GPU (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/grip; mkdir -p "$o"
run() { CUDA_VISIBLE_DEVICES=$1 DM_SCHUR=$2 PASSES=$3 uv run python exp/step4/grip_trace.py > "$o/${2:-off}_p$3.log" 2>&1; }
run 0 "" 1 & run 0 "" 2 & run 1 full 1 & run 1 full 2 & run 2 body 1 & run 2 body 2 & run 3 diag 1 & run 3 diag 2 & wait
for f in "$o"/*.log; do echo "== $f"; grep -E "last pass|counts|Error|error" "$f" | tail -3; done
