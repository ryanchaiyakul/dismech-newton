#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/chatter; mkdir -p "$o"
run() { CUDA_VISIBLE_DEVICES=$1 DM_SCHUR=$2 DM_SCHUR_EPS=$3 PASSES=$4 uv run python exp/principled/chatter.py > "$o/${2:-off}_e$3_p$4.log" 2>&1; }
run 0 "" 0 1 & run 1 "" 0 2 & run 2 full 0.1 1 & run 3 full 0.1 2 & run 0 full 0.3 1 & run 1 full 0.3 2 & wait
for f in "$o"/*.log; do grep -A3 "^PASSES" "$f"; grep -i "error" "$f" | tail -2; done
