#!/bin/bash

# Run Minerva Math on best configs from GSM8K/HumanEval sweep
# UPDATE THE CONFIGS BELOW after analyzing sweep results

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

echo "=========================================="
echo "Mistral 7B Minerva Math - Best Configs"
echo "=========================================="

# Quantization config (fixed)
KBITS_HIGH=8
VBITS_HIGH=4
KBITS_LOW=4
VBITS_LOW=2
KV_PRUNE_THRESH=0.02
KV_BUFFER=64

# ============================================================================
# TODO: UPDATE THESE CONFIGS after analyzing sweep results!
# Run: python3 analyze_mistral_results.py
# Then update the arrays below with your top 1-2 configurations
# ============================================================================

# Example configs - REPLACE WITH YOUR BEST CONFIGS
BEST_CONFIGS=(
    "0.9 5 none"      # Config 1: qt=0.9, md=5, mode=none (example)
    "0.6 7 linear"    # Config 2: qt=0.6, md=7, mode=linear (example)
)

echo ""
echo "WARNING: Make sure you've updated BEST_CONFIGS in this script!"
echo "         Run 'python3 analyze_mistral_results.py' first to identify best configs."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run baseline first
echo ""
echo "Running Minerva baseline (qt=0.9, md=5, mode=none)..."
python3 _eval_qa_correct.py \
    --model mistral \
    --dataset minerva_math \
    --model-gen 1 \
    --model-size 7 \
    --log-path ../logs/mistral7b/minerva/baseline \
    --kbits-high ${KBITS_HIGH} \
    --vbits-high ${VBITS_HIGH} \
    --kbits-low ${KBITS_LOW} \
    --vbits-low ${VBITS_LOW} \
    --kv-prune-thresh ${KV_PRUNE_THRESH} \
    --kv-quant-thresh 0.9 \
    --kv-buffer ${KV_BUFFER} \
    --rounds 1 \
    --kv-min-distance 5 \
    --kv-convergence-mode none

if [ $? -eq 0 ]; then
    echo "✓ Minerva baseline completed"
else
    echo "✗ Minerva baseline FAILED"
fi

# Run best configs
CONFIG_NUM=1
for config in "${BEST_CONFIGS[@]}"; do
    # Parse config string (space-separated: "quant_thresh min_dist mode")
    read -r quant_thresh min_dist mode <<< "$config"

    LOG_PATH="../logs/mistral7b/minerva/best_${CONFIG_NUM}_qt${quant_thresh}_md${min_dist}_${mode}"

    echo ""
    echo "=========================================="
    echo "Running Minerva - Best Config ${CONFIG_NUM}"
    echo "Config: qt=${quant_thresh}, md=${min_dist}, mode=${mode}"
    echo "=========================================="

    python3 _eval_qa_correct.py \
        --model mistral \
        --dataset minerva_math \
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
        echo "✓ Minerva config ${CONFIG_NUM} completed"
    else
        echo "✗ Minerva config ${CONFIG_NUM} FAILED"
    fi

    CONFIG_NUM=$((CONFIG_NUM + 1))
    sleep 2
done

echo ""
echo "=========================================="
echo "Minerva evaluation complete!"
echo "Results saved to: ../logs/mistral7b/minerva/"
echo "=========================================="
