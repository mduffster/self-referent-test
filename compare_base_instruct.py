"""
Compare RFC and RSI metrics between base and instruct models.
Analyzes Role-Focus Coefficient and Role-Separation Index differences with 95% CIs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import json
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_activation_data(data_dir: str) -> pd.DataFrame:
    """Load activation data from NPZ files and compute entropy statistics."""
    print(f"Loading data from: {data_dir}")
    
    # Find all attention pattern files
    npz_files = glob.glob(f"{data_dir}/raw_blocks_*_attn_pattern.npz")
    npz_files.sort()
    
    all_data = []
    
    for npz_file in npz_files:
        # Extract layer number from filename (e.g., "raw_blocks_5_attn_pattern.npz")
        filename = os.path.basename(npz_file)
        layer_num = int(filename.split('_')[2])  # Extract layer number from "blocks_X_attn"
        
        # Load data
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        print(f"Processing layer {layer_num}...")
        
        # Process each head
        for head in range(28):  # Qwen-7B has 28 heads per layer
            # Separate by category
            self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
            neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
            confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
            third_person_acts = [act for act, cat in zip(activations, categories) if cat == 'third_person' and act is not None]
            
            if self_acts and neutral_acts and confounder_acts and third_person_acts:
                # Calculate attention entropy for this specific head
                def calculate_entropy(activation_list):
                    entropies = []
                    for act in activation_list:
                        head_attn = act[0, head, :, :]  # (seq_len, seq_len)
                        for query_pos in range(head_attn.shape[0]):
                            attn_dist = head_attn[query_pos, :]
                            attn_dist = attn_dist + 1e-8  # Avoid log(0)
                            attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                            entropy = -np.sum(attn_dist * np.log(attn_dist))
                            entropies.append(entropy)
                    return np.mean(entropies)
                
                self_entropy = calculate_entropy(self_acts)
                neutral_entropy = calculate_entropy(neutral_acts)
                confounder_entropy = calculate_entropy(confounder_acts)
                third_person_entropy = calculate_entropy(third_person_acts)
                
                # Calculate RFC and RSI
                rfc = 1 - (self_entropy / neutral_entropy) if neutral_entropy > 0 else 0
                rsi = (confounder_entropy - self_entropy) / neutral_entropy if neutral_entropy > 0 else 0
                
                all_data.append({
                    'layer': layer_num,
                    'head': head,
                    'self_entropy': self_entropy,
                    'neutral_entropy': neutral_entropy,
                    'confounder_entropy': confounder_entropy,
                    'third_person_entropy': third_person_entropy,
                    'rfc': rfc,
                    'rsi': rsi
                })
    
    return pd.DataFrame(all_data)

def compute_layer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute layer-level aggregates of RFC and RSI."""
    layer_stats = []
    
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer]
        
        # Calculate means and confidence intervals
        rfc_mean = layer_df['rfc'].mean()
        rfc_std = layer_df['rfc'].std()
        rfc_ci = stats.t.interval(0.95, len(layer_df)-1, loc=rfc_mean, scale=stats.sem(layer_df['rfc']))
        
        rsi_mean = layer_df['rsi'].mean()
        rsi_std = layer_df['rsi'].std()
        rsi_ci = stats.t.interval(0.95, len(layer_df)-1, loc=rsi_mean, scale=stats.sem(layer_df['rsi']))
        
        layer_stats.append({
            'layer': layer,
            'rfc_mean': rfc_mean,
            'rfc_std': rfc_std,
            'rfc_ci_low': rfc_ci[0],
            'rfc_ci_high': rfc_ci[1],
            'rsi_mean': rsi_mean,
            'rsi_std': rsi_std,
            'rsi_ci_low': rsi_ci[0],
            'rsi_ci_high': rsi_ci[1],
            'n_heads': len(layer_df)
        })
    
    return pd.DataFrame(layer_stats)

def compare_models(base_df: pd.DataFrame, instruct_df: pd.DataFrame) -> Dict:
    """Compare RFC and RSI between base and instruct models."""
    print("Computing model comparisons...")
    
    # Get layer-level aggregates
    base_layer_stats = compute_layer_aggregates(base_df)
    instruct_layer_stats = compute_layer_aggregates(instruct_df)
    
    # Merge on layer
    comparison = base_layer_stats.merge(instruct_layer_stats, on='layer', suffixes=('_base', '_instruct'))
    
    # Calculate differences
    comparison['rfc_diff'] = comparison['rfc_mean_instruct'] - comparison['rfc_mean_base']
    comparison['rsi_diff'] = comparison['rsi_mean_instruct'] - comparison['rsi_mean_base']
    
    # Calculate confidence intervals for differences
    rfc_diff_ci = []
    rsi_diff_ci = []
    
    for _, row in comparison.iterrows():
        # RFC difference CI (using pooled standard error)
        rfc_se_base = row['rfc_std_base'] / np.sqrt(row['n_heads_base'])
        rfc_se_instruct = row['rfc_std_instruct'] / np.sqrt(row['n_heads_instruct'])
        rfc_se_diff = np.sqrt(rfc_se_base**2 + rfc_se_instruct**2)
        rfc_diff_ci.append(stats.t.interval(0.95, min(row['n_heads_base'], row['n_heads_instruct'])-1, 
                                          loc=row['rfc_diff'], scale=rfc_se_diff))
        
        # RSI difference CI
        rsi_se_base = row['rsi_std_base'] / np.sqrt(row['n_heads_base'])
        rsi_se_instruct = row['rsi_std_instruct'] / np.sqrt(row['n_heads_instruct'])
        rsi_se_diff = np.sqrt(rsi_se_base**2 + rsi_se_instruct**2)
        rsi_diff_ci.append(stats.t.interval(0.95, min(row['n_heads_base'], row['n_heads_instruct'])-1,
                                          loc=row['rsi_diff'], scale=rsi_se_diff))
    
    comparison['rfc_diff_ci_low'] = [ci[0] for ci in rfc_diff_ci]
    comparison['rfc_diff_ci_high'] = [ci[1] for ci in rfc_diff_ci]
    comparison['rsi_diff_ci_low'] = [ci[0] for ci in rsi_diff_ci]
    comparison['rsi_diff_ci_high'] = [ci[1] for ci in rsi_diff_ci]
    
    return comparison

def create_comparison_plots(comparison_df: pd.DataFrame, output_dir: str, model_family_title: str = "Model Comparison"):
    """Create comparison plots between base and instruct models."""
    print("Creating comparison plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: RFC Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    layers = comparison_df['layer']
    
    # RFC plot
    ax1.plot(layers, comparison_df['rfc_mean_base'], 'b-', linewidth=2, label='Base Model', marker='o', markersize=4)
    ax1.plot(layers, comparison_df['rfc_mean_instruct'], 'r-', linewidth=2, label='Instruct Model', marker='s', markersize=4)
    
    # Add confidence intervals
    ax1.fill_between(layers, comparison_df['rfc_ci_low_base'], comparison_df['rfc_ci_high_base'], 
                     alpha=0.2, color='blue', label='Base 95% CI')
    ax1.fill_between(layers, comparison_df['rfc_ci_low_instruct'], comparison_df['rfc_ci_high_instruct'], 
                     alpha=0.2, color='red', label='Instruct 95% CI')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('RFC (Role-Focus Coefficient)')
    ax1.set_title(f'{model_family_title}: RFC Comparison - Base vs Instruct Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI plot
    ax2.plot(layers, comparison_df['rsi_mean_base'], 'b-', linewidth=2, label='Base Model', marker='o', markersize=4)
    ax2.plot(layers, comparison_df['rsi_mean_instruct'], 'r-', linewidth=2, label='Instruct Model', marker='s', markersize=4)
    
    # Add confidence intervals
    ax2.fill_between(layers, comparison_df['rsi_ci_low_base'], comparison_df['rsi_ci_high_base'], 
                     alpha=0.2, color='blue', label='Base 95% CI')
    ax2.fill_between(layers, comparison_df['rsi_ci_low_instruct'], comparison_df['rsi_ci_high_instruct'], 
                     alpha=0.2, color='red', label='Instruct 95% CI')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('RSI (Role-Separation Index)')
    ax2.set_title(f'{model_family_title}: RSI Comparison - Base vs Instruct Models')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rfc_rsi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Differences with CIs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # RFC differences
    ax1.plot(layers, comparison_df['rfc_diff'], 'g-', linewidth=2, marker='o', markersize=4, label='RFC Difference (Instruct - Base)')
    ax1.fill_between(layers, comparison_df['rfc_diff_ci_low'], comparison_df['rfc_diff_ci_high'], 
                     alpha=0.3, color='green', label='95% CI for Difference')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('RFC Difference (Instruct - Base)')
    ax1.set_title(f'{model_family_title}: RFC Difference - Instruct Model - Base Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI differences
    ax2.plot(layers, comparison_df['rsi_diff'], 'orange', linewidth=2, marker='o', markersize=4, label='RSI Difference (Instruct - Base)')
    ax2.fill_between(layers, comparison_df['rsi_diff_ci_low'], comparison_df['rsi_diff_ci_high'], 
                     alpha=0.3, color='orange', label='95% CI for Difference')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('RSI Difference (Instruct - Base)')
    ax2.set_title(f'{model_family_title}: RSI Difference - Instruct Model - Base Model')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rfc_rsi_differences.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to {output_dir}/")

def print_summary_stats(comparison_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nRFC Differences (Instruct - Base):")
    print(f"  Mean difference: {comparison_df['rfc_diff'].mean():.4f}")
    print(f"  Std difference: {comparison_df['rfc_diff'].std():.4f}")
    print(f"  Min difference: {comparison_df['rfc_diff'].min():.4f} (Layer {comparison_df.loc[comparison_df['rfc_diff'].idxmin(), 'layer']})")
    print(f"  Max difference: {comparison_df['rfc_diff'].max():.4f} (Layer {comparison_df.loc[comparison_df['rfc_diff'].idxmax(), 'layer']})")
    
    print(f"\nRSI Differences (Instruct - Base):")
    print(f"  Mean difference: {comparison_df['rsi_diff'].mean():.4f}")
    print(f"  Std difference: {comparison_df['rsi_diff'].std():.4f}")
    print(f"  Min difference: {comparison_df['rsi_diff'].min():.4f} (Layer {comparison_df.loc[comparison_df['rsi_diff'].idxmin(), 'layer']})")
    print(f"  Max difference: {comparison_df['rsi_diff'].max():.4f} (Layer {comparison_df.loc[comparison_df['rsi_diff'].idxmax(), 'layer']})")
    
    # Count layers with significant differences (CI doesn't include 0)
    rfc_significant = ((comparison_df['rfc_diff_ci_low'] > 0) | (comparison_df['rfc_diff_ci_high'] < 0)).sum()
    rsi_significant = ((comparison_df['rsi_diff_ci_low'] > 0) | (comparison_df['rsi_diff_ci_high'] < 0)).sum()
    
    print(f"\nSignificant differences (95% CI excludes 0):")
    print(f"  RFC: {rfc_significant}/{len(comparison_df)} layers")
    print(f"  RSI: {rsi_significant}/{len(comparison_df)} layers")

def load_family_config():
    """Load model family configuration."""
    try:
        config_path = Path(__file__).parent / "model_family_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare RFC and RSI between base and instruct models")
    
    # Family-based arguments
    parser.add_argument("--family", choices=["llama", "qwen", "mistral"],
                       help="Model family (overrides individual directory settings)")
    
    # Individual arguments (for backward compatibility)
    parser.add_argument("--base_dir", default="results_activation_analysis/latest_base",
                       help="Directory containing base model results")
    parser.add_argument("--instruct_dir", default="results_activation_analysis/latest_run",
                       help="Directory containing instruct model results")
    parser.add_argument("--output_dir", default="comparison_results",
                       help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # If family is specified, load configuration
    if args.family:
        config = load_family_config()
        if config:
            comparison_config = config["model_families"][args.family]["comparison"]
            
            # Override with family config
            args.base_dir = comparison_config["base_dir"]
            args.instruct_dir = comparison_config["instruct_dir"]
            args.output_dir = comparison_config["output_dir"]
    
    # Extract model family for titles
    def get_model_family_title():
        """Extract model family for chart titles."""
        if args.family:
            return f"{args.family.title()} Models"
        
        # Fallback: extract from directory names
        if "llama" in args.base_dir.lower():
            return "Llama Models"
        elif "qwen" in args.base_dir.lower():
            return "Qwen Models"
        elif "mistral" in args.base_dir.lower():
            return "Mistral Models"
        else:
            return "Model Comparison"
    
    model_family_title = get_model_family_title()
    
    print("Base vs Instruct Model Comparison")
    print("="*50)
    print(f"Base model data: {args.base_dir}")
    print(f"Instruct model data: {args.instruct_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    # Load data
    print("\nLoading base model data...")
    base_df = load_activation_data(args.base_dir)
    print(f"✓ Loaded {len(base_df)} head measurements from base model")
    
    print("\nLoading instruct model data...")
    instruct_df = load_activation_data(args.instruct_dir)
    print(f"✓ Loaded {len(instruct_df)} head measurements from instruct model")
    
    # Compare models
    comparison_df = compare_models(base_df, instruct_df)
    
    # Create plots
    create_comparison_plots(comparison_df, args.output_dir, model_family_title)
    
    # Print summary
    print_summary_stats(comparison_df)
    
    # Save detailed results
    comparison_df.to_csv(f'{args.output_dir}/detailed_comparison.csv', index=False)
    print(f"\n✓ Detailed results saved to {args.output_dir}/detailed_comparison.csv")
    
    print(f"\n✓ Comparison complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
