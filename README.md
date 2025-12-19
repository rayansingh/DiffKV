<h3 align="center">
DiffKV: Differentiated KV Cache Management for LLM Inference
</h3>

---

_**DiffKV**_ is an LLM inference framework that enables efficent KV cache compression by jointly exploiting three levels of differentiation in the KV cache:

- The differing impact of keys and values on attention computation.

- The varying importance of tokens.

- The diverse dynamic sparsity patterns across attention heads.

These levels of differentiation introduce **irregular memory usage patterns across different requests and attention heads**, posing significant scalability challenges for memory management. To address these challenges, DiffKV proposes an **on-GPU memory manager** that compacts fragmented free memory list into contiguous regions in parallel, effectively translating sparsity in the KV cache into performance gains.

DiffKV is built on top of vLLM (commit [1db83e3](https://github.com/vllm-project/vllm/commit/1db83e31a2468cae37f326a642c0a4c4edbb5e4f)) and currently supports the following HuggingFace model architectures:

- **LLaMA-2 & LLaMA-3** (`meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, etc.)
- **Mistral** (`mistralai/Mistral-7B-v0.1`, etc.)
- **Mixtral** (`mistralai/Mixtral-8x7B-v0.1`, etc.)
- **Qwen-2.5** (`Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/QwQ-32B`, etc.)
- **Qwen-3** (`Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, etc.)
- **Qwen-3 MoE** (`Qwen/Qwen3-30B-A3B`, etc.)

DiffKV supports model weight quantization using [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), and FP8 formats.


## Installation

#### Prerequisites:
- Python >= 3.10
- Nvidia GPUs with Ada, Hopper or newer architectures
- Ray >= 2.0 (for distributed inference)

#### Install DiffKV from source:

```bash
pip install -e .
```

#### Activate Virtual Environment:

If DiffKV is installed in a virtual environment (e.g., `/venv/main`), activate it before running any scripts:

```bash
source /venv/main/bin/activate
```

Alternatively, you can use the virtual environment's Python directly without activation:

```bash
/venv/main/bin/python3 <script.py>
```

## Usage

### Supported Benchmarks

The evaluation scripts support the following benchmarks:
- **HumanEval**: Code generation benchmark (164 programming problems)
- **GSM8k**: Grade school math problems with step-by-step reasoning
- **Minerva Math**: Advanced mathematics problems (MATH dataset)

### Running Evaluations

To run model evaluations on supported benchmarks:

1. **Prepare the environment:**
   ```bash
   cd param_tuning
   source /venv/main/bin/activate
   export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH
   mkdir -p ../logs
   ```

2. **Key parameters** in evaluation scripts:
   - `--kbits-high/low`, `--vbits-high/low`: Quantization bit-widths for keys and values
   - `--kv-prune-thresh`: Threshold below which tokens are pruned
   - `--kv-quant-thresh`: Threshold above which tokens remain in high precision
   - `--kv-min-distance` (**NEW**): Minimum gap between high/low precision regions (must be ≤ `kv_quant_thresh - kv_prune_thresh`) 
   - `--kv-convergence-mode` (**NEW**): Thresholding strategy (`none`, `linear`, `logarithmic`)
   - `--kv-buffer-size`: Buffer size for KV cache management

**Usage Example:**

```bash
python3 _eval_qa_correct.py \
   --model llama --model-gen 3 --model-size 8 \
   --dataset gsm8k --rounds 1 \
   --log-path ../logs/per_token_thresh/llama3-8b/linear \
   --kbits-high 8 --vbits-high 4 --kbits-low 4 --vbits-low 2 \
   --kv-prune-thresh 0.02 --kv-quant-thresh 1.0 --kv-buffer 64 \
   --kv-min-distance 0.7 --kv-convergence-mode linear
```

3. **Results location:**
   - Raw outputs: `logs/per_token_thresh/$MODEL/$DATASET/`
   - Each test creates `eval.csv` with correctness metrics and GPU memory usage

## Our Work

This repository includes the following enhancements to the original DiffKV implementation:

### Layer-Dependent Threshold Convergence

**New Parameters:**
- `--kv-min-distance`: Minimum gap maintained between pruning and quantization thresholds
- `--kv-convergence-mode`: Controls how thresholds converge across layers (`none`, `linear`, `logarithmic`)

**Feature Description:**

The original DiffKV uses fixed `kv_prune_thresh` and `kv_quant_thresh` across all layers. This enhancement allows the pruning threshold to gradually approach the quantization threshold across deeper layers, enabling more aggressive compression in later layers where token importance may differ.

- **`none` mode (default)**: Fixed thresholds across all layers (original behavior)
- **`linear` mode**: Pruning threshold increases linearly toward quantization threshold
- **`logarithmic` mode**: Pruning threshold increases logarithmically, with more aggressive convergence in deeper layers

**Implementation:**
- Updated CUDA kernels (`csrc/cache_kernels.cu`, `csrc/long_prompt_cache_kernels.cu`)
- Modified inference engine (`vllm/engine/arg_utils.py`, `vllm/config.py`, `vllm/sequence.py`)
- Enhanced attention layers (`vllm/model_executor/layers/sparse_attention_big_kernel.py`)

**Constraint:** `kv_min_distance` must satisfy: `kv_min_distance <= (kv_quant_thresh - kv_prune_thresh)`

### Peak GPU Memory Tracking

**Feature Description:**

Automatically tracks and reports peak GPU memory usage during inference, enabling systematic memory profiling across different compression configurations.

**Outputs:**
- **Stderr logs**: `PEAK_GPU_MEMORY_GB` and per-GPU `PEAK_GPU_MEMORY_GB_GPU{i}` entries
- **CSV metrics**: `avg_peak_memory_gb` and `max_peak_memory_gb` in `eval.csv`

**Implementation:**
- Enhanced evaluation scripts (`param_tuning/_eval_codegen.py`, `param_tuning/_eval_qa_correct.py`)
- Integrated with existing logging infrastructure

Results will include memory metrics alongside correctness metrics in the output CSV.

## Citation
[Original DiffKV paper](https://arxiv.org/abs/2412.03131)
```bibtex
@inproceedings{zhang2025diffkv,
  title={DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction},
  author={Zhang, Yanqi and Hu, Yuwei and Zhao, Runyuan and Lui, John CS and Chen, Haibo},
  booktitle={Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  pages={431--445},
  year={2025}
}
```
