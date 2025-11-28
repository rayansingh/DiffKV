# Mistral 7B Evaluation - Quick Start

## 🚀 Quick Start (TL;DR)

```bash
# 1. Setup on Lightning AI
cd /workspace/DiffKV/param_tuning
export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH

# 2. Run baseline checks (MUST PASS before sweep)
bash mistral_baseline_check.sh

# 3. Start full sweep in tmux (54 runs, ~12-18 hours)
tmux new -s mistral_sweep
bash mistral_full_sweep.sh
# Detach: Ctrl+B, then D

# 4. Analyze results
python3 analyze_mistral_results.py

# 5. Update best configs in mistral_minerva_best.sh, then:
bash mistral_minerva_best.sh
```

## 📋 What This Does

**Parameter Sweep:** 54 total configurations
- **Datasets:** GSM8K (27 runs) + HumanEval (27 runs)
- **Parameters:**
  - `quant_thresh`: 0.6, 0.9, 1.0
  - `min_distance`: 1, 5, 7
  - `convergence_mode`: none, linear, logarithmic
- **Fixed:** K8V4 (high), K4V2 (low), prune=0.02

## 📁 Scripts Overview

| Script | Purpose | Runtime |
|--------|---------|---------|
| `mistral_baseline_check.sh` | Sanity check (3 datasets) | ~30-60 min |
| `mistral_sweep_gsm8k.sh` | GSM8K sweep (27 runs) | ~6-9 hours |
| `mistral_sweep_humaneval.sh` | HumanEval sweep (27 runs) | ~6-9 hours |
| `mistral_full_sweep.sh` | Both sweeps (54 runs) | ~12-18 hours |
| `analyze_mistral_results.py` | Aggregate & analyze results | ~1 min |
| `mistral_minerva_best.sh` | Minerva on best configs | ~1-2 hours |

## 📊 Expected Outputs

```
logs/mistral7b/
├── baseline/           # Baseline sanity checks
├── sweep/              # 54 sweep results
│   ├── gsm8k/
│   └── humaneval/
├── minerva/            # Minerva follow-up
└── aggregated_results.csv
```

## ⚠️ Critical Notes

1. **ALWAYS run baseline checks first** - if they fail, the sweep will fail
2. **Use tmux/screen** - sweep takes 12-18 hours
3. **Monitor GPU memory** - watch for OOMs
4. **Update Minerva script** - edit `mistral_minerva_best.sh` with your best configs after analysis

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce `BATCH_SIZE` in `_eval_qa_correct.py` line 36 |
| SSH disconnect | Use `tmux` or `nohup` |
| Hangs/freezes | Kill with `pkill -f "python3 _eval"` |
| Missing logs | Check `../logs/mistral7b/` directory exists |

## 📚 Full Documentation

See `MISTRAL_EVALUATION_GUIDE.md` in repo root for complete guide.

## ✅ Checklist

**Setup:**
- [ ] Lightning AI A100 80GB reserved (24h+)
- [ ] Repo cloned: `/workspace/DiffKV`
- [ ] Environment: `export PYTHONPATH=/workspace/DiffKV:$PYTHONPATH`
- [ ] Scripts executable: `chmod +x *.sh`

**Execution:**
- [ ] Baseline: All 3 datasets pass
- [ ] Sweep: Running in tmux
- [ ] Analysis: `aggregated_results.csv` generated
- [ ] Minerva: Updated configs + run complete

**Communication:**
- [ ] Baseline complete posted
- [ ] Sweep start posted
- [ ] Sweep complete posted
- [ ] Minerva complete posted

---

**Questions?** Check `MISTRAL_EVALUATION_GUIDE.md` or ask in group chat.
