#!/bin/bash
export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

#!/bin/bash

# Define arrays for parameter combinations
modes=(none linear logarithmic)
dists=(0.1 0.5 0.7)
threshs=(0.6 0.9 1.0)

# Common arguments (avoids repetition)
common_args=(
  --model llama
  --dataset gsm8k
  --model-gen 3
  --model-size 3
  --kbits-high 8
  --vbits-high 4
  --kbits-low 4
  --vbits-low 2
  --kv-prune-thresh 0.02
  --kv-buffer 64
  --rounds 1
)

# Completed runs (easier to maintain and extend)

for mode in "${modes[@]}"; do
  for dist in "${dists[@]}"; do
    for thresh in "${threshs[@]}"; do
      key="${mode}_${dist}_${thresh}"
      
      # Run the command
      python3 _eval_qa_correct.py \
        "${common_args[@]}" \
        --log-path "../logs/per_token_thresh/llama3.2-3b/$key" \
        --kv-quant-thresh "$thresh" \
        --kv-min-distance "$dist" \
        --kv-convergence-mode "$mode"
    done
  done
done
