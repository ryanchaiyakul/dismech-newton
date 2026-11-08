#!/bin/bash
# Robin2 with/without the momentum debt: momentum + pin tests, bounce, grip (sci6000; shares GPUs with the gates)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/debt; mkdir -p "$o"
t() { CUDA_VISIBLE_DEVICES=$1 DM_ROBIN=2 DM_ROBIN_DEBT=$2 DM_ITER=1 PEND=$3 uv run pytest tests/test_coupling.py -k "momentum or pin" -q --tb=line -p no:cacheprovider > "$o/tests_d$2_pend$3.log" 2>&1; }
t 0 1 1 & t 1 1 0 & t 2 0 0 &
CUDA_VISIBLE_DEVICES=3 DM_ROBIN=2 DM_ROBIN_DEBT=1 DM_ITER=1 uv run python exp/step4/bounce.py > "$o/bounce.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 DM_ROBIN=2 DM_ROBIN_DEBT=1 PASSES=1 uv run python exp/principled/chatter.py > "$o/chatter.log" 2>&1 &
wait
for f in "$o"/tests_*.log; do echo "== $f"; grep -E "passed|failed|AssertionError" "$f" | grep -v Warn; done
tail -1 "$o/bounce.log"; grep -A3 ^PASSES "$o/chatter.log"
