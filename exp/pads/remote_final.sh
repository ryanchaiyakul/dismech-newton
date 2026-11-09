#!/bin/bash
# Clean src (no switches) + grip pads in the examples: every gate (sci6000).
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
o=exp/pads/logs/final; mkdir -p "$o"
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=line -p no:cacheprovider > "$o/coupling.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 uv run pytest -q --tb=line -p no:cacheprovider > "$o/suite.log" 2>&1 &
(for i in 1 2; do CUDA_VISIBLE_DEVICES=2 uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test > "$o/tramp$i.log" 2>&1; done
 CUDA_VISIBLE_DEVICES=2 uv run python exp/step4/bounce.py > "$o/bounce.log" 2>&1
 CUDA_VISIBLE_DEVICES=2 uv run python exp/principled/chatter.py > "$o/chatter.log" 2>&1) &
CUDA_VISIBLE_DEVICES=3 uv run pytest -m slow -q --tb=line -p no:cacheprovider > "$o/slow.log" 2>&1 &
wait
for g in coupling suite slow tramp1 tramp2 bounce; do echo "== $g: $(grep -hE "passed|failed|Error|BOUNCE" "$o/$g.log" | grep -v Warn | sort -u | tail -8 | tr "\n" " " | cut -c1-900)"; done
grep -A3 ^PASSES "$o/chatter.log" | cut -c1-80
