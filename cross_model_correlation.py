#!/usr/bin/env python3
"""
Cross-Model Correlation Analysis for RFC Differences

This script compares RFC differences between base and instruct models
across different model families to identify patterns in how instruction
tuning affects role-conditioning circuits.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import json
from pathlib import Path

def load_comparison_data(comparison_dir):
    """Load RFC differences from comparison results."""
    csv_path = os.path.join(comparison_dir, "detailed_comparison.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Comparison data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df['rfc_diff'].values

def run_correlation_analysis():
    """Run correlation analysis between model families."""
    print("Cross-Model RFC Correlation Analysis")
    print("=" * 50)
    
    # Load comparison data for all three models
    models = {}
    comparison_dirs = {
        'llama': 'figures/llama/comparison',
        'mistral': 'figures/mistral/comparison',
        'qwen': 'figures/qwen/comparison'
    }
    
    for model, dir_path in comparison_dirs.items():
        try:
            rfc_diffs = load_comparison_data(dir_path)
            models[model] = rfc_diffs
            print(f"✓ Loaded {model}: {len(rfc_diffs)} layers")
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            continue
    
    if len(models) < 2:
        print("❌ Need at least 2 models for analysis")
        return
    
    print(f"\nLoaded {len(models)} models for analysis")
    
    # Calculate summary statistics with threshold analysis
    threshold = 0.01  # Threshold for "significant" changes
    summary_stats = {}
    for model, rfc_diffs in models.items():
        summary_stats[model] = {
            'mean': np.mean(rfc_diffs),
            'std': np.std(rfc_diffs),
            'min': np.min(rfc_diffs),
            'max': np.max(rfc_diffs),
            'positive_count': np.sum(rfc_diffs > 0),
            'negative_count': np.sum(rfc_diffs < 0),
            'significant_positive': np.sum(rfc_diffs > threshold),
            'significant_negative': np.sum(rfc_diffs < -threshold),
            'near_zero': np.sum(np.abs(rfc_diffs) <= threshold),
            'total_layers': len(rfc_diffs)
        }
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("RFC DIFFERENCE SUMMARY STATISTICS")
    print("=" * 50)
    print(f"(Instruct - Base, positive = more role-focus in instruct)")
    print(f"(Threshold for 'significant' changes: ±{threshold})")
    print()
    
    for model, stats in summary_stats.items():
        print(f"{model.upper()}:")
        print(f"  Mean RFC difference: {stats['mean']:+.4f}")
        print(f"  Std deviation:       {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]")
        print(f"  Significant positive: {stats['significant_positive']}/{stats['total_layers']} ({stats['significant_positive']/stats['total_layers']*100:.1f}%)")
        print(f"  Significant negative: {stats['significant_negative']}/{stats['total_layers']} ({stats['significant_negative']/stats['total_layers']*100:.1f}%)")
        print(f"  Near zero (≤±{threshold}): {stats['near_zero']}/{stats['total_layers']} ({stats['near_zero']/stats['total_layers']*100:.1f}%)")
        print()
    
    # Run correlations only between English models (same layer count)
    print("=" * 50)
    print("CROSS-MODEL CORRELATIONS (English Models Only)")
    print("=" * 50)
    print("(Pearson correlation of RFC differences across layers)")
    print()
    
    correlations = {}
    
    # Only correlate Llama vs Mistral (both have 32 layers)
    if 'llama' in models and 'mistral' in models:
        data1, data2 = models['llama'], models['mistral']
        
        # Both should have 32 layers, but check just in case
        min_layers = min(len(data1), len(data2))
        data1_trunc = data1[:min_layers]
        data2_trunc = data2[:min_layers]
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(data1_trunc, data2_trunc)
        spearman_r, spearman_p = spearmanr(data1_trunc, data2_trunc)
        
        correlations["llama_vs_mistral"] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'layers_compared': min_layers
        }
        
        print(f"LLAMA vs MISTRAL:")
        print(f"  Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4f})")
        print(f"  Spearman r: {spearman_r:+.4f} (p={spearman_p:.4f})")
        print(f"  Layers compared: {min_layers}")
        print()
    else:
        print("⚠️  Cannot run correlation: Need both Llama and Mistral data")
        print()
    
    # Identify key patterns
    print("=" * 50)
    print("KEY FINDINGS")
    print("=" * 50)
    
    # Check directional patterns across all models
    print(f"Cross-Model Directional Patterns (threshold: ±{threshold}):")
    
    if 'llama' in summary_stats:
        llama_stats = summary_stats['llama']
        print(f"  Llama: {llama_stats['significant_negative']}/{llama_stats['total_layers']} layers show compression, {llama_stats['near_zero']}/{llama_stats['total_layers']} near zero")
    
    if 'mistral' in summary_stats:
        mistral_stats = summary_stats['mistral']
        print(f"  Mistral: {mistral_stats['significant_negative']}/{mistral_stats['total_layers']} layers show compression, {mistral_stats['near_zero']}/{mistral_stats['total_layers']} near zero")
    
    if 'qwen' in summary_stats:
        qwen_stats = summary_stats['qwen']
        print(f"  Qwen: {qwen_stats['significant_positive']}/{qwen_stats['total_layers']} layers show preservation, {qwen_stats['near_zero']}/{qwen_stats['total_layers']} near zero")
    
    # Identify key pattern
    if 'qwen' in summary_stats and 'llama' in summary_stats and 'mistral' in summary_stats:
        qwen_sig_pos = summary_stats['qwen']['significant_positive']
        llama_sig_neg = summary_stats['llama']['significant_negative']
        mistral_sig_neg = summary_stats['mistral']['significant_negative']
        
        if qwen_sig_pos > 0 and (llama_sig_neg > 0 or mistral_sig_neg > 0):
            print(f"  → Qwen shows OPPOSITE pattern from English models!")
            print(f"  → Qwen has {qwen_sig_pos} layers with significant preservation")
            print(f"  → English models have {llama_sig_neg + mistral_sig_neg} layers with significant compression")
        else:
            print(f"  → Mixed patterns across model families")
    print()
    
    # Check correlation strength
    strong_correlations = []
    for pair, corr in correlations.items():
        if abs(corr['pearson_r']) > 0.7:
            strong_correlations.append((pair, corr['pearson_r']))
    
    if strong_correlations:
        print("Strong Correlations (|r| > 0.7):")
        for pair, r in strong_correlations:
            print(f"  {pair}: r = {r:+.4f}")
    else:
        print("No strong correlations found (|r| > 0.7)")
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert summary stats
    converted_summary = {}
    for model, stats in summary_stats.items():
        converted_summary[model] = {k: convert_numpy_types(v) for k, v in stats.items()}
    
    # Convert correlations
    converted_correlations = {}
    for pair, corr in correlations.items():
        converted_correlations[pair] = {k: convert_numpy_types(v) for k, v in corr.items()}
    
    results = {
        'summary_stats': converted_summary,
        'correlations': converted_correlations,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('cross_model_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to cross_model_analysis_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = run_correlation_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
