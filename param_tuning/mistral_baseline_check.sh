#!/bin/bash

# Baseline sanity check for Mistral 7B
# Run one config on all 3 datasets to verify everything works

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

# Baseline configuration
# mode = none, threshold = 0.9, min_distance = 5
# Quantization: 8-bit high (keys), 4-bit high (values), 4-bit low (keys), 2-bit low (values)
# Prune threshold: 0.02

echo "=========================================="
echo "Starting Mistral 7B Baseline Sanity Checks"
echo "Configuration: mode=none, quant_thresh=0.9, min_distance=5"
echo "=========================================="

# GSM8K Baseline
echo ""
echo "[1/3] Running GSM8K baseline..."
python3 _eval_qa_correct.py \
    --model mistral \
    --dataset gsm8k \
    --model-gen 1 \
    --model-size 7 \
    --log-path ../logs/mistral7b/baseline/gsm8k \
    --kbits-high 8 --vbits-high 4 \
    --kbits-low 4 --vbits-low 2 \
    --kv-prune-thresh 0.02 \
    --kv-quant-thresh 0.9 \
    --kv-buffer 64 \
    --rounds 1 \
    --kv-min-distance 5 \
    --kv-convergence-mode none

if [ $? -eq 0 ]; then
    echo "✓ GSM8K baseline completed successfully"
else
    echo "✗ GSM8K baseline FAILED"
    exit 1
fi

# HumanEval Baseline
echo ""
echo "[2/3] Running HumanEval baseline..."
python3 _eval_codegen.py \
    --model mistral \
    --dataset humaneval \
    --model-gen 1 \
    --model-size 7 \
    --log-path ../logs/mistral7b/baseline/humaneval \
    --kbits-high 8 --vbits-high 4 \
    --kbits-low 4 --vbits-low 2 \
    --kv-prune-thresh 0.02 \
    --kv-quant-thresh 0.9 \
    --kv-buffer 64 \
    --rounds 1 \
    --kv-min-distance 5 \
    --kv-convergence-mode none

if [ $? -eq 0 ]; then
    echo "✓ HumanEval baseline completed successfully"
else
    echo "✗ HumanEval baseline FAILED"
    exit 1
fi

# Minerva Math Baseline
echo ""
echo "[3/3] Running Minerva Math baseline..."
python3 _eval_qa_correct.py \
    --model mistral \
    --dataset minerva_math \
    --model-gen 1 \
    --model-size 7 \
    --log-path ../logs/mistral7b/baseline/minerva_math \
    --kbits-high 8 --vbits-high 4 \
    --kbits-low 4 --vbits-low 2 \
    --kv-prune-thresh 0.02 \
    --kv-quant-thresh 0.9 \
    --kv-buffer 64 \
    --rounds 1 \
    --kv-min-distance 5 \
    --kv-convergence-mode none

if [ $? -eq 0 ]; then
    echo "✓ Minerva Math baseline completed successfully"
else
    echo "✗ Minerva Math baseline FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "All baseline checks completed successfully!"
echo "Check logs at: ../logs/mistral7b/baseline/"
echo "=========================================="
