#!/usr/bin/env python3
"""
Compare convergence modes (none, linear, logarithmic) across all benchmarks.
Generates comprehensive visualizations and summary statistics.

Usage:
    python3 compare_convergence_modes.py /workspace/DiffKV/logs/per_token_thresh/llama3-8b
    python3 compare_convergence_modes.py <path-to-model-logs>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Skipping visualizations.")

# Default configuration
CONVERGENCE_MODES = ['none', 'linear', 'logarithmic']
DATASETS = ['gsm8k', 'minerva_math', 'humaneval', 'mbpp_plus', 'mmlu_cot', 'mmlu_pro_cot']

# Metrics to compare
METRICS = {
    'correctness': 'Correctness (%)',
    'prune_ratio': 'Pruning Ratio (%)',
    'compress_ratio': 'Compression Ratio (%)',
    'physical_compress_ratio': 'Physical Compression Ratio (%)',
    'avg_peak_memory_gb': 'Peak Memory (GB)',
}

def find_eval_files(base_path, mode, dataset):
    """Find all eval.csv files for a given mode and dataset."""
    search_path = Path(base_path) / mode
    eval_files = []

    # Search for dataset-specific directories
    for pattern in [f'{dataset}*', f'**/{dataset}*']:
        matches = list(search_path.glob(f'{pattern}/**/eval.csv'))
        eval_files.extend(matches)

    return eval_files

def load_benchmark_data(base_path, modes, datasets):
    """Load all benchmark data into a structured format."""
    data = {}

    for dataset in datasets:
        data[dataset] = {}
        for mode in modes:
            eval_files = find_eval_files(base_path, mode, dataset)

            if not eval_files:
                print(f"⚠ No data found for {dataset} / {mode}")
                continue

            # Read the first (or only) eval.csv found
            try:
                df = pd.read_csv(eval_files[0], index_col=0)
                # Get first row (actual data, second row is variance)
                if len(df) > 0:
                    data[dataset][mode] = df.iloc[0].to_dict()
                    print(f"✓ Loaded {dataset} / {mode}: {eval_files[0]}")
            except Exception as e:
                print(f"✗ Error loading {dataset} / {mode}: {e}")

    return data

def create_comparison_table(data, datasets, modes):
    """Create a comprehensive comparison table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("="*120)

    for dataset in datasets:
        if dataset not in data or not data[dataset]:
            continue

        print(f"\n{'='*120}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*120}")

        # Header
        print(f"\n{'Metric':<35} {'None':>20} {'Linear':>20} {'Logarithmic':>20}")
        print("-"*120)

        # Get metrics for all modes
        mode_data = {mode: data[dataset].get(mode, {}) for mode in modes}

        # Accuracy
        print("ACCURACY:")
        for metric_key, metric_name in [('correctness', '  Correctness (%)'), ('correct_p', '  Correct Problems')]:
            values = []
            for mode in modes:
                val = mode_data[mode].get(metric_key, 0) * (100 if metric_key == 'correctness' else 1)
                values.append(f"{val:>19.2f}")
            print(f"{metric_name:<35} {values[0]} {values[1]} {values[2]}")

        # Memory compression
        print("\nMEMORY COMPRESSION:")
        for metric_key, metric_name in [
            ('prune_ratio', '  Pruned/Deleted (%)'),
            ('low_prec_ratio', '  Quantized (%)'),
            ('high_prec_ratio', '  High Precision (%)'),
            ('compress_ratio', '  Compress Ratio (%)'),
            ('physical_compress_ratio', '  Physical Ratio (%)')
        ]:
            values = []
            for mode in modes:
                val = mode_data[mode].get(metric_key, 0) * 100
                values.append(f"{val:>19.2f}")
            print(f"{metric_name:<35} {values[0]} {values[1]} {values[2]}")

        # Memory usage
        print("\nPEAK MEMORY:")
        for metric_key, metric_name in [
            ('avg_peak_memory_gb', '  Average (GB)'),
            ('max_peak_memory_gb', '  Maximum (GB)')
        ]:
            values = []
            for mode in modes:
                val = mode_data[mode].get(metric_key, 0)
                values.append(f"{val:>19.2f}")
            print(f"{metric_name:<35} {values[0]} {values[1]} {values[2]}")

        # Calculate improvements
        print("\nIMPROVEMENTS OVER 'NONE':")
        print(f"{'Metric':<35} {'Linear':>20} {'Logarithmic':>20}")
        print("-"*120)

        none_data = mode_data['none']
        if none_data:
            # Pruning improvement
            none_prune = none_data.get('prune_ratio', 0)
            for mode in ['linear', 'logarithmic']:
                mode_prune = mode_data[mode].get('prune_ratio', 0)
                if none_prune > 0:
                    improvement = (mode_prune / none_prune - 1) * 100
                    if mode == 'linear':
                        linear_val = f"{improvement:>19.2f}%"
                    else:
                        log_val = f"{improvement:>19.2f}%"
            print(f"{'  Pruning Increase':<35} {linear_val} {log_val}")

            # Compression improvement
            none_compress = none_data.get('compress_ratio', 1)
            for mode in ['linear', 'logarithmic']:
                mode_compress = mode_data[mode].get('compress_ratio', 1)
                improvement = (none_compress - mode_compress) / none_compress * 100
                if mode == 'linear':
                    linear_val = f"{improvement:>19.2f}%"
                else:
                    log_val = f"{improvement:>19.2f}%"
            print(f"{'  Better Compression':<35} {linear_val} {log_val}")

            # Accuracy change
            none_acc = none_data.get('correctness', 0)
            for mode in ['linear', 'logarithmic']:
                mode_acc = mode_data[mode].get('correctness', 0)
                change = (mode_acc - none_acc) * 100
                if mode == 'linear':
                    linear_val = f"{change:>18.2f}pp"
                else:
                    log_val = f"{change:>18.2f}pp"
            print(f"{'  Accuracy Change':<35} {linear_val} {log_val}")

def create_visualizations(data, datasets, modes, output_dir='comparison_plots'):
    """Create comprehensive visualizations."""
    if not HAS_MATPLOTLIB:
        print("⚠ Skipping visualizations (matplotlib not installed)")
        print("  Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Filter datasets that have data
    available_datasets = [d for d in datasets if d in data and len(data[d]) > 0]

    if not available_datasets:
        print("No data available for visualization")
        return

    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Convergence Mode Comparison Across Benchmarks', fontsize=16, fontweight='bold')

    # Prepare data for plotting
    plot_data = {metric: {mode: [] for mode in modes} for metric in METRICS.keys()}
    dataset_labels = []

    for dataset in available_datasets:
        dataset_labels.append(dataset.replace('_', '\n'))
        for mode in modes:
            mode_data = data[dataset].get(mode, {})
            for metric in METRICS.keys():
                value = mode_data.get(metric, 0)
                # Convert to percentage or keep as-is
                if metric in ['correctness', 'prune_ratio', 'compress_ratio', 'physical_compress_ratio']:
                    value *= 100
                plot_data[metric][mode].append(value)

    # Plot each metric
    x = np.arange(len(dataset_labels))
    width = 0.25

    plots = [
        (axes[0, 0], 'correctness', 'Accuracy Comparison'),
        (axes[0, 1], 'prune_ratio', 'Pruning Ratio Comparison'),
        (axes[1, 0], 'compress_ratio', 'Compression Ratio Comparison'),
        (axes[1, 1], 'physical_compress_ratio', 'Physical Compression Ratio'),
        (axes[2, 0], 'avg_peak_memory_gb', 'Peak GPU Memory Usage'),
    ]

    colors = {'none': '#3498db', 'linear': '#e74c3c', 'logarithmic': '#2ecc71'}

    for ax, metric, title in plots:
        for i, mode in enumerate(modes):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, plot_data[metric][mode], width,
                         label=mode.capitalize(), color=colors[mode], alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel(METRICS[metric], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Summary statistics in the last subplot
    ax = axes[2, 1]
    ax.axis('off')

    summary_text = "SUMMARY STATISTICS\n" + "="*40 + "\n\n"

    for dataset in available_datasets:
        mode_data = {mode: data[dataset].get(mode, {}) for mode in modes}

        # Find best mode for this dataset
        best_compress = min(modes, key=lambda m: mode_data[m].get('compress_ratio', 1))
        best_prune = max(modes, key=lambda m: mode_data[m].get('prune_ratio', 0))
        best_acc = max(modes, key=lambda m: mode_data[m].get('correctness', 0))

        summary_text += f"{dataset.upper()}:\n"
        summary_text += f"  Best Compression: {best_compress}\n"
        summary_text += f"  Most Pruning: {best_prune}\n"
        summary_text += f"  Best Accuracy: {best_acc}\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_file = f'{output_dir}/convergence_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Compare convergence modes across benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare_convergence_modes.py /workspace/DiffKV/logs/per_token_thresh/llama3-8b
  python3 compare_convergence_modes.py ../logs/per_token_thresh/llama3-8b --output-dir ./plots
        """
    )
    parser.add_argument('log_path', type=str,
                        help='Path to model logs (e.g., /workspace/DiffKV/logs/per_token_thresh/llama3-8b)')
    parser.add_argument('--output-dir', type=str, default='comparison_plots',
                        help='Directory to save output plots (default: comparison_plots)')
    parser.add_argument('--modes', type=str, nargs='+', default=CONVERGENCE_MODES,
                        help=f'Convergence modes to compare (default: {" ".join(CONVERGENCE_MODES)})')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                        help='Datasets to compare (default: all)')

    args = parser.parse_args()

    print("="*120)
    print("CONVERGENCE MODE COMPARISON TOOL")
    print("="*120)
    print(f"\nBase path: {args.log_path}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output directory: {args.output_dir}")
    print("\nLoading data...")

    # Verify path exists
    if not os.path.exists(args.log_path):
        print(f"\n✗ Error: Path does not exist: {args.log_path}")
        sys.exit(1)

    # Load all data
    data = load_benchmark_data(args.log_path, args.modes, args.datasets)

    # Create comparison table
    create_comparison_table(data, args.datasets, args.modes)

    # Create visualizations
    print("\n" + "="*120)
    print("CREATING VISUALIZATIONS")
    print("="*120)
    create_visualizations(data, args.datasets, args.modes, args.output_dir)

    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print(f"\nResults saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
