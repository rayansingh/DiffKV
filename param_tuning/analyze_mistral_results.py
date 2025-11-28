#!/usr/bin/env python3
"""
Analyze Mistral 7B experiment results and aggregate into a single CSV.

This script:
1. Scans all log directories for eval.csv files
2. Extracts configuration from directory names
3. Combines results into a master CSV
4. Identifies top configurations by accuracy
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import re
import sys


def parse_config_from_path(log_path):
    """
    Extract configuration from path like:
    ../logs/mistral7b/sweep/gsm8k/qt0.6_md1_none/

    Returns dict with: quant_thresh, min_distance, convergence_mode
    """
    path_str = str(log_path)

    # Extract from directory name (e.g., "qt0.6_md1_none")
    pattern = r'qt([\d.]+)_md([\d.]+)_(none|linear|logarithmic)'
    match = re.search(pattern, path_str)

    if not match:
        return None

    return {
        'quant_thresh': float(match.group(1)),
        'min_distance': float(match.group(2)),
        'convergence_mode': match.group(3)
    }


def load_eval_csv(eval_csv_path):
    """Load eval.csv and extract metrics."""
    try:
        df = pd.read_csv(eval_csv_path)

        # Expected columns: correct_p, correctness, high_prec_ratio, low_prec_ratio,
        # prune_ratio, physical_compress_ratio, compress_ratio,
        # avg_peak_memory_gb, max_peak_memory_gb

        # Extract the first value (mean) from each metric
        metrics = {}
        for col in df.columns:
            if col != 'Unnamed: 0':  # Skip index column
                metrics[col] = df[col].iloc[0] if len(df) > 0 else 0.0

        return metrics
    except Exception as e:
        print(f"Error loading {eval_csv_path}: {e}", file=sys.stderr)
        return None


def aggregate_results(base_log_dir, dataset_name):
    """
    Aggregate all results for a given dataset.

    Args:
        base_log_dir: Base directory containing logs (e.g., ../logs/mistral7b)
        dataset_name: 'gsm8k', 'humaneval', or 'minerva_math'

    Returns:
        DataFrame with all configurations and their metrics
    """
    results = []

    # Look for both sweep and baseline directories
    search_paths = [
        Path(base_log_dir) / "sweep" / dataset_name,
        Path(base_log_dir) / "baseline" / dataset_name,
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Find all eval.csv files
        for eval_csv in search_path.rglob("eval.csv"):
            # Extract config from path
            config = parse_config_from_path(eval_csv.parent)

            if config is None:
                # This might be a baseline run with different naming
                if "baseline" in str(eval_csv):
                    config = {
                        'quant_thresh': 0.9,
                        'min_distance': 5.0,
                        'convergence_mode': 'none'
                    }
                else:
                    print(f"Warning: Could not parse config from {eval_csv.parent}", file=sys.stderr)
                    continue

            # Load metrics
            metrics = load_eval_csv(eval_csv)
            if metrics is None:
                continue

            # Combine config and metrics
            row = {
                'model': 'mistral-7b',
                'dataset': dataset_name,
                'log_path': str(eval_csv.parent),
                **config,
                **metrics
            }

            results.append(row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Mistral 7B experiment results")
    parser.add_argument(
        '--log-dir',
        type=str,
        default='../logs/mistral7b',
        help='Base directory containing Mistral logs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../logs/mistral7b/aggregated_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Show top K configurations by accuracy'
    )
    args = parser.parse_args()

    print("========================================")
    print("Mistral 7B Results Aggregation")
    print("========================================")

    # Aggregate results for each dataset
    all_results = []

    for dataset in ['gsm8k', 'humaneval', 'minerva_math']:
        print(f"\nProcessing {dataset}...")
        df = aggregate_results(args.log_dir, dataset)

        if len(df) > 0:
            print(f"  Found {len(df)} configurations")
            all_results.append(df)
        else:
            print(f"  No results found")

    if not all_results:
        print("\nError: No results found!")
        return

    # Combine all datasets
    combined_df = pd.concat(all_results, ignore_index=True)

    # Sort by dataset and correctness (descending)
    combined_df = combined_df.sort_values(
        by=['dataset', 'correctness'],
        ascending=[True, False]
    )

    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined_df.to_csv(args.output, index=False)

    print(f"\n✓ Saved aggregated results to: {args.output}")
    print(f"  Total configurations: {len(combined_df)}")

    # Show top configurations for each dataset
    print("\n" + "="*60)
    print("TOP CONFIGURATIONS BY ACCURACY")
    print("="*60)

    for dataset in combined_df['dataset'].unique():
        dataset_df = combined_df[combined_df['dataset'] == dataset]

        print(f"\n{dataset.upper()} - Top {args.top_k} Configurations:")
        print("-" * 60)

        top_configs = dataset_df.head(args.top_k)

        for idx, row in top_configs.iterrows():
            print(f"\n  Rank {idx + 1 - top_configs.index[0] + 1}:")
            print(f"    Accuracy:      {row['correctness']:.4f} ({row['correct_p']:.2f}%)")
            print(f"    Config:        qt={row['quant_thresh']}, md={row['min_distance']}, mode={row['convergence_mode']}")
            print(f"    Compression:   {row['compress_ratio']:.4f} (theoretical), {row['physical_compress_ratio']:.4f} (physical)")
            print(f"    Prune ratio:   {row['prune_ratio']:.4f}")
            if 'avg_peak_memory_gb' in row:
                print(f"    Peak memory:   {row['avg_peak_memory_gb']:.2f} GB (avg), {row['max_peak_memory_gb']:.2f} GB (max)")
            print(f"    Log path:      {row['log_path']}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find best overall configs (average accuracy across GSM8K and HumanEval)
    gsm8k_df = combined_df[combined_df['dataset'] == 'gsm8k']
    humaneval_df = combined_df[combined_df['dataset'] == 'humaneval']

    if len(gsm8k_df) > 0 and len(humaneval_df) > 0:
        # Merge on configuration columns
        merged = gsm8k_df.merge(
            humaneval_df,
            on=['quant_thresh', 'min_distance', 'convergence_mode'],
            suffixes=('_gsm8k', '_humaneval')
        )

        # Calculate average accuracy
        merged['avg_accuracy'] = (merged['correctness_gsm8k'] + merged['correctness_humaneval']) / 2
        merged = merged.sort_values('avg_accuracy', ascending=False)

        print("\nBest configurations by average accuracy (GSM8K + HumanEval):")
        print("-" * 60)

        for idx, row in merged.head(3).iterrows():
            print(f"\n  Config: qt={row['quant_thresh']}, md={row['min_distance']}, mode={row['convergence_mode']}")
            print(f"    Average accuracy:   {row['avg_accuracy']:.4f}")
            print(f"    GSM8K accuracy:     {row['correctness_gsm8k']:.4f}")
            print(f"    HumanEval accuracy: {row['correctness_humaneval']:.4f}")

        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Review the top 1-2 configurations above")
        print("2. Run Minerva on these configs using mistral_minerva_best.sh")
        print("3. Update the script with your chosen configurations")

    print()


if __name__ == "__main__":
    main()
