#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
mkdir -p exp/impact/logs
(CUDA_VISIBLE_DEVICES=2 PASSES=1 uv run exp/impact/trace.py > exp/impact/logs/p1.log 2>&1; echo p1 done >> exp/impact/logs/p1.log) &
(CUDA_VISIBLE_DEVICES=3 PASSES=2 uv run exp/impact/trace.py > exp/impact/logs/p2.log 2>&1; echo p2 done >> exp/impact/logs/p2.log) &
wait
