#!/bin/bash
export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

#---------------------------------- Mistral-7B with linear convergence
# ******** humaneval
python3 _eval_codegen.py \
  --model mistral \
  --dataset humaneval \
  --model-gen 2 \
  --model-size 7 \
  --log-path ../logs/per_token_thresh/mistral-7b \
  --kbits-high 8 \
  --vbits-high 4 \
  --kbits-low 4 \
  --vbits-low 2 \
  --kv-prune-thresh 0.02 \
  --kv-quant-thresh 0.02 \
  --kv-buffer 64 \
  --rounds 1 \
  --kv-min-distance 0.1 \
  --kv-convergence-mode linear
