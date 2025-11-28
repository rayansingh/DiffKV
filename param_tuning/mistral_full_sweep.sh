#!/bin/bash

# Mistral 7B Full Parameter Sweep
# Runs both GSM8K and HumanEval sweeps sequentially
# Total: 54 runs (27 GSM8K + 27 HumanEval)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Mistral 7B Full Parameter Sweep"
echo "54 total runs (27 GSM8K + 27 HumanEval)"
echo "Started at: $(date)"
echo "=========================================="

START_TIME=$(date +%s)

# Run GSM8K sweep
echo ""
echo "Starting GSM8K sweep (27 runs)..."
bash mistral_sweep_gsm8k.sh

if [ $? -ne 0 ]; then
    echo "Warning: GSM8K sweep encountered errors"
fi

# Run HumanEval sweep
echo ""
echo "Starting HumanEval sweep (27 runs)..."
bash mistral_sweep_humaneval.sh

if [ $? -ne 0 ]; then
    echo "Warning: HumanEval sweep encountered errors"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "Full Sweep Complete!"
echo "Finished at: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results location: ../logs/mistral7b/sweep/"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Analyze results with: python3 analyze_mistral_results.py"
echo "2. Identify best 1-2 configs from the sweep"
echo "3. Run Minerva on best configs"
