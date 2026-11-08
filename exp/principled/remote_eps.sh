#!/bin/bash
# full Schur block: regularization sweep on the pick-place grip (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/grip; mkdir -p "$o"
run() { CUDA_VISIBLE_DEVICES=$1 DM_SCHUR=full DM_SCHUR_EPS=$2 PASSES=$3 uv run python exp/step4/grip_trace.py > "$o/full_e$2_p$3.log" 2>&1; }
run 0 0.03 1 & run 0 0.03 2 & run 1 0.1 1 & run 1 0.1 2 & run 2 0.3 1 & run 2 0.3 2 & run 3 1.0 1 & run 3 1.0 2 & wait
for f in "$o"/full_e*.log; do echo "== $f"; grep -E "last pass|counts|Error" "$f" | tail -3; grep "^sub" "$f" | sed -n 20,24p | cut -c1-200; done
