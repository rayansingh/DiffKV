#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH
#---------------------------------- Mixtral with Minerva Math
# Testing combinations of:
# - Quant thresholds: 0.6, 0.9, 1.0
# - Min distance: 0.1, 0.5, 0.7
# - Convergence modes: none, linear, logarithmic
# Quantization: 8-bit high (keys), 4-bit high (values), 4-bit low (keys), 2-bit low (values)
# Prune threshold: 0.02

# ******** Minerva Math - quant_thresh=0.6
# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.1_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode none
# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.1_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode linear
# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.1_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode logarithmic

# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.5_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode none
# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.5_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode linear
# ./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.6_md0.5_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.6 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode logarithmic

# # ******** Minerva Math - quant_thresh=0.9
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.1_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.1_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.1_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode logarithmic

./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.5_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.5_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.5_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode logarithmic

./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.7_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.7_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt0.9_md0.7_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 0.9 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode logarithmic

# ******** Minerva Math - quant_thresh=1.0
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.1_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.1_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.1_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.1 --kv-convergence-mode logarithmic

./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.5_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.5_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.5_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.5 --kv-convergence-mode logarithmic

./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.7_none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode none
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.7_linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode linear
./run_with_cleanup.sh python3 _eval_qa_correct.py --model mixtral --dataset minerva_math --model-gen 1 --model-size 56 --log-path ../logs/per_token_thresh/mixtral/minerva/qt1.0_md0.7_logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode logarithmic

