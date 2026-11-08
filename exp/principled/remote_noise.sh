#!/bin/bash
# full Schur, eps .1, passes 1: probe size and refresh (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/grip; mkdir -p "$o"
run() { CUDA_VISIBLE_DEVICES=$1 DM_SCHUR=$2 DM_SCHUR_EPS=0.1 DM_SCHUR_DELTA=$3 DM_SCHUR_EVERY=$4 PASSES=$5 uv run python exp/step4/grip_trace.py > "$o/n_$2_d$3_e$4_p$5.log" 2>&1; }
run 0 full 0.1 1 1 & run 0 full 0.01 1 1 & run 1 full 1.0 600 1 & run 1 full 0.1 600 1 & run 2 full 0.1 600 2 & run 2 body 0.1 600 1 & run 3 full 10 1 1 & run 3 full 0.1 60 1 & wait
for f in "$o"/n_*.log; do echo "== $f"; grep -E "last pass|counts|Error" "$f" | tail -3; done
