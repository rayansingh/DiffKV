#!/bin/bash
export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH
#---------------------------------- LLaMA-3-8B with linear convergence
# Configuration: prune_thresh=0.02, quant_thresh=1.0 (DiffKV author settings from benchmark_throughput.sh)
# Quantization: 8-bit high (keys), 4-bit high (values), 4-bit low (keys), 2-bit low (values)
# Convergence: linear mode with min_distance=0.1

# # ******** gsm8k
python3 _eval_qa_correct.py --model qwen2 --dataset gsm8k --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode none
python3 _eval_qa_correct.py --model qwen2 --dataset gsm8k --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode linear
python3 _eval_qa_correct.py --model qwen2 --dataset gsm8k --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode logarithmic

# ******** minerva_math
python3 _eval_qa_correct.py --model qwen2 --dataset minerva_math --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode none
python3 _eval_qa_correct.py --model qwen2 --dataset minerva_math --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode linear
python3 _eval_qa_correct.py --model qwen2 --dataset minerva_math --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 1 --kv-min-distance 0.7 --kv-convergence-mode logarithmic

# ******** humaneval
python3 _eval_codegen.py --model qwen2 --dataset humaneval --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/none --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode none
python3 _eval_codegen.py --model qwen2 --dataset humaneval --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/linear --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode linear
python3 _eval_codegen.py --model qwen2 --dataset humaneval --model-gen 2 --model-size 7 --log-path ../logs/per_token_thresh/qwen25-7b/logarithmic --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 --rounds 3 --kv-min-distance 0.7 --kv-convergence-mode logarithmic