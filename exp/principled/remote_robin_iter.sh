#!/bin/bash
# Robin iterated within the step (DM_ROBIN=2, passes P): momentum, pin, fold, MIDAS, bounce, grip (sci6000)
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/principled/logs/robin_iter; mkdir -p "$o"
e() { env DM_ROBIN=2 DM_PASSES=$2 DM_ITER=$2 PASSES=$2 PEND=0 CUDA_VISIBLE_DEVICES=$1 "${@:3}"; }
e 0 2 uv run pytest tests/test_coupling.py -k "momentum or pin" -q --tb=line -p no:cacheprovider > "$o/mom_p2.log" 2>&1 &
e 0 3 uv run pytest tests/test_coupling.py -k "momentum or pin" -q --tb=line -p no:cacheprovider > "$o/mom_p3.log" 2>&1 &
e 1 2 uv run pytest tests/test_coupling.py -m "slow or not slow" -k "fold" -q --tb=line -p no:cacheprovider > "$o/fold_p2.log" 2>&1 &
e 2 2 uv run pytest tests/test_coupling.py -m "slow or not slow" -k "midas" -q --tb=line -p no:cacheprovider > "$o/midas_p2.log" 2>&1 &
e 3 2 uv run python exp/step4/bounce.py > "$o/bounce_p2.log" 2>&1 &
e 3 2 uv run python exp/principled/chatter.py > "$o/chatter_p2.log" 2>&1 &
wait
for f in "$o"/mom_*.log "$o"/fold_*.log "$o"/midas_*.log; do echo "== $f"; grep -E "passed|failed|AssertionError" "$f" | grep -v Warn | sort -u; done
tail -1 "$o/bounce_p2.log"; grep -A3 ^PASSES "$o/chatter_p2.log"
