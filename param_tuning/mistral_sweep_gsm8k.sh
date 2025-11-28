#!/bin/bash

# Mistral 7B GSM8K Parameter Sweep
# 54 total runs (3 thresholds × 3 min_distances × 3 modes × 2 datasets)
# This script handles GSM8K (27 runs)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

echo "=========================================="
echo "Mistral 7B GSM8K Parameter Sweep"
echo "27 configurations total"
echo "Quantization: K8V4 (high), K4V2 (low)"
echo "Prune threshold: 0.02"
echo "=========================================="

TOTAL_RUNS=27
CURRENT_RUN=0

# Quantization config (fixed)
KBITS_HIGH=8
VBITS_HIGH=4
KBITS_LOW=4
VBITS_LOW=2
KV_PRUNE_THRESH=0.02
KV_BUFFER=64

# Parameter sweep
QUANT_THRESHOLDS=(0.6 0.9 1.0)
MIN_DISTANCES=(1 5 7)
CONVERGENCE_MODES=("none" "linear" "logarithmic")

for quant_thresh in "${QUANT_THRESHOLDS[@]}"; do
    for min_dist in "${MIN_DISTANCES[@]}"; do
        for mode in "${CONVERGENCE_MODES[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            # Create descriptive log path
            LOG_PATH="../logs/mistral7b/sweep/gsm8k/qt${quant_thresh}_md${min_dist}_${mode}"

            echo ""
            echo "=========================================="
            echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}"
            echo "Config: quant_thresh=${quant_thresh}, min_distance=${min_dist}, mode=${mode}"
            echo "Log: ${LOG_PATH}"
            echo "=========================================="

            python3 _eval_qa_correct.py \
                --model mistral \
                --dataset gsm8k \
                --model-gen 1 \
                --model-size 7 \
                --log-path "${LOG_PATH}" \
                --kbits-high ${KBITS_HIGH} \
                --vbits-high ${VBITS_HIGH} \
                --kbits-low ${KBITS_LOW} \
                --vbits-low ${VBITS_LOW} \
                --kv-prune-thresh ${KV_PRUNE_THRESH} \
                --kv-quant-thresh ${quant_thresh} \
                --kv-buffer ${KV_BUFFER} \
                --rounds 1 \
                --kv-min-distance ${min_dist} \
                --kv-convergence-mode ${mode}

            if [ $? -eq 0 ]; then
                echo "✓ Run ${CURRENT_RUN}/${TOTAL_RUNS} completed successfully"
            else
                echo "✗ Run ${CURRENT_RUN}/${TOTAL_RUNS} FAILED"
                echo "Failed config: qt=${quant_thresh}, md=${min_dist}, mode=${mode}"
                # Continue with remaining runs even if one fails
            fi

            # Small delay to avoid potential race conditions
            sleep 2
        done
    done
done

echo ""
echo "=========================================="
echo "GSM8K Sweep Complete: ${CURRENT_RUN}/${TOTAL_RUNS} runs attempted"
echo "Results saved to: ../logs/mistral7b/sweep/gsm8k/"
echo "=========================================="
