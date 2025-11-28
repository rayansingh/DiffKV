# Mistral 7B Evaluation Guide

Complete guide for running Mistral 7B parameter sweep on Lightning AI.

## 📋 Overview

This evaluation sweeps through **54 configurations** (3 thresholds × 3 min_distances × 3 convergence modes × 2 datasets):
- **27 GSM8K runs**
- **27 HumanEval runs**
- **Follow-up Minerva runs** on best configs

**Parameter Grid:**
- `quant_thresh`: 0.6, 0.9, 1.0
- `min_distance`: 1, 5, 7
- `convergence_mode`: none, linear, logarithmic

**Fixed Settings:**
- Quantization: K8V4 (high precision), K4V2 (low precision)
- Prune threshold: 0.02
- Buffer size: 64

---

## 🚀 Step 1: Lightning AI Setup

### 1.1 Get Lightning AI Access
1. Sign up at https://lightning.ai
2. Request access to **A100 80GB GPU**
3. Reserve instance for **24h+** runtime

### 1.2 Launch Studio
```bash
# Select machine type: A100 (80GB)
# Start a new Studio
```

### 1.3 Clone Repository
```bash
# In Lightning AI terminal
cd /workspace
git clone <your-diffkv-repo-url> DiffKV
cd DiffKV

# Pull latest changes (if already cloned)
git pull origin main
```

### 1.4 Verify Environment
```bash
# Check Python version (should be 3.10+)
python3 --version

# Check CUDA availability
nvidia-smi

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 1.5 Install Dependencies
```bash
# Install DiffKV (if not already installed)
cd /workspace/DiffKV
pip install -e .

# Verify vLLM can be imported
python3 -c "from vllm import LLMEngine; print('✓ vLLM imported successfully')"
```

### 1.6 Set Environment Variables
```bash
export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

# Optional: Add to ~/.bashrc for persistence
echo 'export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH' >> ~/.bashrc
```

---

## 🧪 Step 2: Baseline Sanity Checks

**CRITICAL:** Run baseline checks on all 3 datasets **before** starting the full sweep.

```bash
cd /workspace/DiffKV/param_tuning
chmod +x mistral_baseline_check.sh
bash mistral_baseline_check.sh
```

This runs:
1. GSM8K with baseline config (qt=0.9, md=5, mode=none)
2. HumanEval with baseline config
3. Minerva Math with baseline config

**Verify:**
- [ ] All 3 runs finish successfully (no crashes)
- [ ] Logs show reasonable scores
- [ ] GPU memory usage lines appear in stderr
- [ ] Check logs at: `../logs/mistral7b/baseline/`

**Expected output format:**
```
*** gpu 0
stdout: ...
PEAK_GPU_MEMORY_GB: 45.1234
PEAK_GPU_MEMORY_GB_GPU0: 45.1234
...
***
```

If any baseline fails, **STOP** and debug before proceeding.

---

## 🔄 Step 3: Parameter Sweep

### Option A: Run Full Sweep (Recommended)
```bash
cd /workspace/DiffKV/param_tuning
chmod +x mistral_full_sweep.sh
bash mistral_full_sweep.sh
```

This runs both GSM8K and HumanEval sweeps sequentially (54 total runs).

### Option B: Run Datasets Separately
```bash
# GSM8K only (27 runs)
chmod +x mistral_sweep_gsm8k.sh
bash mistral_sweep_gsm8k.sh

# HumanEval only (27 runs)
chmod +x mistral_sweep_humaneval.sh
bash mistral_sweep_humaneval.sh
```

### Expected Runtime
- **Per run:** 10-30 minutes (depends on dataset size)
- **Full sweep:** ~12-18 hours
- **Recommendation:** Use `tmux` or `screen` to keep running if SSH disconnects

### Using tmux (Recommended)
```bash
# Start a tmux session
tmux new -s mistral_sweep

# Run the sweep
cd /workspace/DiffKV/param_tuning
bash mistral_full_sweep.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t mistral_sweep
```

### Monitor Progress
```bash
# Check running processes
ps aux | grep python

# Monitor GPU usage
watch -n 5 nvidia-smi

# Check latest logs
ls -lhrt ../logs/mistral7b/sweep/gsm8k/
tail -f ../logs/mistral7b/sweep/gsm8k/qt0.6_md1_none/round_0/eval_0/*.csv
```

---

## 📊 Step 4: Analyze Results

After the sweep completes, aggregate and analyze results:

```bash
cd /workspace/DiffKV/param_tuning
python3 analyze_mistral_results.py \
    --log-dir ../logs/mistral7b \
    --output ../logs/mistral7b/aggregated_results.csv \
    --top-k 5
```

**Output:**
- Aggregated CSV with all configurations: `../logs/mistral7b/aggregated_results.csv`
- Console output showing top-K configs by accuracy
- Recommendations for best configs

**CSV Columns:**
```
model, dataset, quant_thresh, min_distance, convergence_mode,
correctness, correct_p, high_prec_ratio, low_prec_ratio, prune_ratio,
physical_compress_ratio, compress_ratio,
avg_peak_memory_gb, max_peak_memory_gb, log_path
```

### Identify Best Configs
Look for configurations with:
1. **High accuracy** on both GSM8K and HumanEval
2. **Good compression ratio** (lower is better)
3. **Reasonable memory usage**

Example output:
```
GSM8K - Top 5 Configurations:
--------------------------------------------------
  Rank 1:
    Accuracy:      0.8456 (84.56%)
    Config:        qt=0.9, md=5, mode=linear
    Compression:   0.4521 (theoretical), 0.5234 (physical)
    Prune ratio:   0.3456
    Peak memory:   45.23 GB (avg), 46.12 GB (max)
```

---

## 🎯 Step 5: Minerva Follow-Up

Run Minerva on the **baseline + 1-2 best configs** identified above.

### 5.1 Update Script
```bash
cd /workspace/DiffKV/param_tuning
nano mistral_minerva_best.sh
```

Update the `BEST_CONFIGS` array with your top configs:
```bash
BEST_CONFIGS=(
    "0.9 5 linear"    # Example: Best config from analysis
    "0.6 7 none"      # Example: Second best config
)
```

### 5.2 Run Minerva
```bash
chmod +x mistral_minerva_best.sh
bash mistral_minerva_best.sh
```

This runs:
1. Baseline (qt=0.9, md=5, mode=none)
2. Your chosen best config(s)

### 5.3 Verify Results
```bash
ls -lh ../logs/mistral7b/minerva/
cat ../logs/mistral7b/minerva/baseline/round_0/eval.csv
```

---

## 📁 Directory Structure

After completion, your logs should look like:
```
logs/mistral7b/
├── baseline/                    # Baseline sanity checks
│   ├── gsm8k/
│   ├── humaneval/
│   └── minerva_math/
├── sweep/                       # Parameter sweep results
│   ├── gsm8k/
│   │   ├── qt0.6_md1_none/
│   │   ├── qt0.6_md1_linear/
│   │   ├── qt0.6_md1_logarithmic/
│   │   ├── qt0.6_md5_none/
│   │   └── ... (27 total)
│   └── humaneval/
│       └── ... (27 total)
├── minerva/                     # Minerva follow-up
│   ├── baseline/
│   ├── best_1_qt0.9_md5_linear/
│   └── best_2_qt0.6_md7_none/
└── aggregated_results.csv       # Combined analysis
```

---

## 📢 Communication Checkpoints

### Checkpoint 1: Baseline Complete
**After baseline sanity checks:**
```
✅ Baseline sanity checks complete on Mistral 7B
- GSM8K: [accuracy] (log: logs/mistral7b/baseline/gsm8k/)
- HumanEval: [accuracy] (log: logs/mistral7b/baseline/humaneval/)
- Minerva: [accuracy] (log: logs/mistral7b/baseline/minerva_math/)
All runs finished successfully, proceeding with sweep.
```

### Checkpoint 2: Sweep Started
**After starting 54-config sweep:**
```
🚀 Started Mistral 7B 54-config sweep (GSM8K + HumanEval)
- Started at: [timestamp]
- Expected completion: ~12-18 hours
- Monitoring: tmux session "mistral_sweep"
- Logs: logs/mistral7b/sweep/
```

### Checkpoint 3: Sweep Complete
**After sweep finishes:**
```
✅ Mistral 7B parameter sweep complete
- Total runs: 54 (27 GSM8K + 27 HumanEval)
- Duration: [X hours Y minutes]
- Logs saved to: logs/mistral7b/sweep/
- Aggregated results: logs/mistral7b/aggregated_results.csv
Running analysis now...
```

### Checkpoint 4: Minerva Complete
**After Minerva runs:**
```
✅ Minerva evaluation complete on Mistral 7B
- Baseline + [N] best configs from sweep
- Best config 1: qt=[X], md=[Y], mode=[Z] - Accuracy: [A]
- Best config 2: qt=[X], md=[Y], mode=[Z] - Accuracy: [B]
- Logs: logs/mistral7b/minerva/
All Mistral 7B evaluations complete.
```

---

## 🚨 Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size in _eval_qa_correct.py
# Line 36: BATCH_SIZE = 256 → 128

# Or adjust GPU memory utilization
# Line 35: GPU_MEM_UTIL = 0.97 → 0.90
```

### Issue: Model Download Fails
```bash
# Pre-download the model
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('✓ Model downloaded successfully')
"
```

### Issue: Process Hangs/Infinite Loop
```bash
# Kill the process
pkill -f "python3 _eval"

# Check for zombie processes
ps aux | grep defunct

# Restart from checkpoint
# Logs are saved per-run, so you can skip completed configs
```

### Issue: SSH Disconnection
```bash
# Always use tmux/screen!
tmux new -s mistral_sweep
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t mistral_sweep

# Alternative: use nohup
nohup bash mistral_full_sweep.sh > sweep.log 2>&1 &
tail -f sweep.log
```

### Issue: Missing Peak Memory in Logs
Check that:
1. `torch.cuda.reset_peak_memory_stats()` is called at start
2. stderr is being captured properly
3. `PEAK_GPU_MEMORY_GB:` line is printed to stderr (not stdout)

---

## ✅ Final Checklist

Before starting:
- [ ] Lightning AI instance reserved (A100 80GB, 24h+)
- [ ] Repository cloned and up-to-date
- [ ] Environment variables set (`PYTHONPATH`, etc.)
- [ ] Dependencies installed (`pip install -e .`)
- [ ] Scripts have execute permissions (`chmod +x *.sh`)

Baseline phase:
- [ ] All 3 baseline runs complete successfully
- [ ] Logs show reasonable scores
- [ ] GPU memory tracking works

Sweep phase:
- [ ] Running in tmux/screen session
- [ ] Monitoring progress periodically
- [ ] Logs being saved correctly

Analysis phase:
- [ ] Aggregation script runs without errors
- [ ] Top configs identified
- [ ] Best configs selected for Minerva

Minerva phase:
- [ ] Updated `mistral_minerva_best.sh` with best configs
- [ ] Baseline + best configs complete
- [ ] All logs saved

Communication:
- [ ] Posted baseline completion
- [ ] Posted sweep start
- [ ] Posted sweep completion
- [ ] Posted Minerva completion

---

## 📚 Quick Reference

### File Locations
```bash
# Scripts
param_tuning/mistral_baseline_check.sh       # Baseline sanity checks
param_tuning/mistral_full_sweep.sh           # Full 54-run sweep
param_tuning/mistral_sweep_gsm8k.sh          # GSM8K sweep only
param_tuning/mistral_sweep_humaneval.sh      # HumanEval sweep only
param_tuning/mistral_minerva_best.sh         # Minerva follow-up
param_tuning/analyze_mistral_results.py      # Results aggregation

# Logs
logs/mistral7b/baseline/                     # Baseline results
logs/mistral7b/sweep/{gsm8k,humaneval}/      # Sweep results
logs/mistral7b/minerva/                      # Minerva results
logs/mistral7b/aggregated_results.csv        # Combined CSV
```

### Key Commands
```bash
# Run baseline
bash mistral_baseline_check.sh

# Run full sweep
tmux new -s mistral_sweep
bash mistral_full_sweep.sh
# Detach: Ctrl+B, then D

# Analyze results
python3 analyze_mistral_results.py

# Run Minerva
bash mistral_minerva_best.sh

# Monitor progress
watch -n 5 nvidia-smi
tmux attach -t mistral_sweep
```

---

## 🎓 Understanding the Output

### Per-run Outputs
Each configuration creates a directory with:
```
qt0.9_md5_none/
├── round_0/
│   ├── eval_0/
│   │   ├── correctness.csv          # Accuracy metrics
│   │   ├── kv_len_0.npy             # KV cache lengths
│   │   └── block_num_0.npy          # Block numbers
│   ├── sample_indices_0.csv         # Sample indices used
│   ├── compress_config.csv          # Config used
│   └── eval.csv                     # Aggregated metrics
└── compress_ratio.csv               # Compression stats
```

### eval.csv Columns
- `correctness`: Fraction of correct answers (0-1)
- `correct_p`: Percentage correct (0-100)
- `prune_ratio`: Fraction of tokens pruned
- `compress_ratio`: Theoretical compression (based on bit-widths)
- `physical_compress_ratio`: Actual memory compression
- `avg_peak_memory_gb`: Average peak GPU memory
- `max_peak_memory_gb`: Maximum peak GPU memory

---

**Good luck with your evaluation! 🚀**
