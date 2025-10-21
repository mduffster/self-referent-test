"""
Activation Patching Analysis
==========================
Identifies the most important heads for activation patching by analyzing
self-referent attention entropy differences between base and instruct models.

For each model family (Llama, Mistral), we:
1. Calculate self-referent attention entropy per head
2. Compute differences between base and instruct models
3. Identify top 10 heads with largest absolute differences
4. Prepare candidates for activation patching experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_activation_data_robust(data_dir: str, model_name: str) -> pd.DataFrame:
    """Load activation data and compute robust entropy statistics per prompt."""
    print(f"Loading data from: {data_dir}")
    print(f"Model: {model_name}")
    
    # Find all attention pattern files
    npz_files = glob.glob(f"{data_dir}/raw_blocks_*_attn_pattern.npz")
    npz_files.sort()
    
    all_data = []
    
    for npz_file in npz_files:
        # Extract layer number from filename
        filename = os.path.basename(npz_file)
        layer_num = int(filename.split('_')[2])
        
        # Load data
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        print(f"Processing layer {layer_num}...")
        
        # Determine number of heads based on model
        if "llama" in model_name.lower():
            n_heads = 32
        elif "mistral" in model_name.lower():
            n_heads = 32
        elif "qwen" in model_name.lower():
            n_heads = 28
        else:
            n_heads = 32  # Default
        
        # Process each head
        for head in range(n_heads):
            # Separate by category
            self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
            neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
            
            if len(self_acts) == 0 or len(neutral_acts) == 0:
                continue
                
            # Calculate entropy for each prompt separately
            self_entropies = []
            neutral_entropies = []
            
            # Process self-referent prompts
            for act in self_acts:
                if act is not None and act.shape[1] > head:
                    head_attn = act[0, head, :, :]  # Shape: [seq_len, seq_len]
                    entropy = calculate_head_entropy(head_attn)
                    self_entropies.append(entropy)
            
            # Process neutral prompts
            for act in neutral_acts:
                if act is not None and act.shape[1] > head:
                    head_attn = act[0, head, :, :]  # Shape: [seq_len, seq_len]
                    entropy = calculate_head_entropy(head_attn)
                    neutral_entropies.append(entropy)
            
            if len(self_entropies) > 0 and len(neutral_entropies) > 0:
                all_data.append({
                    'layer': layer_num,
                    'head': head,
                    'self_entropies': self_entropies,
                    'neutral_entropies': neutral_entropies,
                    'n_self': len(self_entropies),
                    'n_neutral': len(neutral_entropies)
                })
    
    df = pd.DataFrame(all_data)
    print(f"✓ Loaded {len(df)} head measurements")
    return df

def calculate_head_entropy(head_attn: np.ndarray) -> float:
    """Calculate entropy for a single head's attention pattern."""
    # Calculate entropy for each query position, then average
    entropies = []
    for query_pos in range(head_attn.shape[0]):
        attn_dist = head_attn[query_pos, :]
        # Avoid log(0) by adding small epsilon
        attn_dist = attn_dist + 1e-8
        attn_dist = attn_dist / attn_dist.sum()  # Renormalize
        entropy = -np.sum(attn_dist * np.log(attn_dist))
        entropies.append(entropy)
    
    return np.mean(entropies)

def find_patching_candidates_robust(base_df: pd.DataFrame, instruct_df: pd.DataFrame, 
                                   model_family: str, top_k: int = 10) -> pd.DataFrame:
    """Find the top-k heads using robust methodology with log-ratios and DiD."""
    
    # Merge dataframes on layer and head
    merged = pd.merge(base_df, instruct_df, on=['layer', 'head'], 
                     suffixes=('_base', '_instruct'))
    
    all_candidates = []
    
    for _, row in merged.iterrows():
        # Extract entropy lists
        self_base = row['self_entropies_base']
        self_instruct = row['self_entropies_instruct']
        neutral_base = row['neutral_entropies_base']
        neutral_instruct = row['neutral_entropies_instruct']
        
        # Calculate per-prompt differences
        self_diffs = [s_i - s_b for s_i, s_b in zip(self_instruct, self_base)]
        neutral_diffs = [n_i - n_b for n_i, n_b in zip(neutral_instruct, neutral_base)]
        
        # Difference-in-differences (role-specific effect)
        did_diffs = [s_d - n_d for s_d, n_d in zip(self_diffs, neutral_diffs)]
        
        # Log-ratio with data-scaled epsilon
        epsilon = 0.05 * np.median(self_base)
        log_ratios = [np.log((s_i + epsilon) / (s_b + epsilon)) 
                     for s_i, s_b in zip(self_instruct, self_base)]
        
        # DiD log-ratios
        did_log_ratios = [np.log((s_i + epsilon) / (s_b + epsilon)) - 
                         np.log((n_i + epsilon) / (n_b + epsilon))
                         for s_i, s_b, n_i, n_b in zip(self_instruct, self_base, 
                                                      neutral_instruct, neutral_base)]
        
        # Robust statistics
        candidate = {
            'layer': row['layer'],
            'head': row['head'],
            'model_family': model_family,
            
            # Raw statistics
            'self_base_median': np.median(self_base),
            'self_instruct_median': np.median(self_instruct),
            'neutral_base_median': np.median(neutral_base),
            'neutral_instruct_median': np.median(neutral_instruct),
            
            # Robust differences
            'self_diff_median': np.median(self_diffs),
            'neutral_diff_median': np.median(neutral_diffs),
            'did_median': np.median(did_diffs),
            'log_ratio_median': np.median(log_ratios),
            'did_log_ratio_median': np.median(did_log_ratios),
            
            # Consistency (fraction of prompts with same sign)
            'self_consistency': np.mean([d > 0 for d in self_diffs]) if np.mean([d > 0 for d in self_diffs]) > 0.5 else np.mean([d < 0 for d in self_diffs]),
            'did_consistency': np.mean([d > 0 for d in did_diffs]) if np.mean([d > 0 for d in did_diffs]) > 0.5 else np.mean([d < 0 for d in did_diffs]),
            
            # Stability (IQR)
            'self_diff_iqr': np.percentile(self_diffs, 75) - np.percentile(self_diffs, 25),
            'did_iqr': np.percentile(did_diffs, 75) - np.percentile(did_diffs, 25),
            'log_ratio_iqr': np.percentile(log_ratios, 75) - np.percentile(log_ratios, 25),
            
            # Baseline focus (low entropy = specialized)
            'baseline_entropy': np.median(self_base),
            
            # Sample sizes
            'n_self': len(self_base),
            'n_neutral': len(neutral_base)
        }
        
        all_candidates.append(candidate)
    
    candidates_df = pd.DataFrame(all_candidates)
    
    # Composite scoring: z-score weighted combination
    def z_score(series):
        return (series - series.mean()) / series.std()
    
    # Calculate composite score
    candidates_df['magnitude_score'] = z_score(np.abs(candidates_df['did_log_ratio_median']))
    candidates_df['consistency_score'] = z_score(candidates_df['did_consistency'])
    candidates_df['baseline_score'] = z_score(-candidates_df['baseline_entropy'])  # Negative for low entropy
    
    candidates_df['composite_score'] = (candidates_df['magnitude_score'] + 
                                      candidates_df['consistency_score'] + 
                                      candidates_df['baseline_score'])
    
    # Sort by composite score and get top-k
    candidates = candidates_df.nlargest(top_k, 'composite_score').copy()
    candidates['rank'] = range(1, top_k + 1)
    
    return candidates

def create_patching_analysis_plots(candidates_df: pd.DataFrame, output_dir: str):
    """Create visualizations for activation patching candidates."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Top candidates by normalized difference
    plt.figure(figsize=(14, 8))
    
    # Create a combined identifier for x-axis
    candidates_df['layer_head'] = candidates_df['layer'].astype(int).astype(str) + '_' + candidates_df['head'].astype(int).astype(str)
    
    # Sort by composite score for plotting
    plot_df = candidates_df.sort_values('composite_score', ascending=True)
    
    colors = ['red' if diff < 0 else 'blue' for diff in plot_df['did_median']]
    
    bars = plt.barh(range(len(plot_df)), plot_df['did_median'], color=colors, alpha=0.7)
    plt.yticks(range(len(plot_df)), plot_df['layer_head'])
    plt.xlabel('Difference-in-Differences (Role-Specific Effect)')
    plt.ylabel('Layer_Head')
    plt.title('Top Activation Patching Candidates (Robust DiD)\n(Red = Instruct < Base, Blue = Instruct > Base)')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, plot_df['did_median'])):
        plt.text(value + (0.001 if value >= 0 else -0.001), i, f'{value:.4f}', 
                va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/activation_patching_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution of DiD differences
    plt.figure(figsize=(10, 6))
    plt.hist(candidates_df['did_median'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Difference-in-Differences (Role-Specific Effect)')
    plt.ylabel('Count')
    plt.title('Distribution of DiD Effects in Top Candidates')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='No difference')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_difference_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Layer-wise analysis
    plt.figure(figsize=(12, 8))
    
    # Group by layer and show distribution
    layer_stats = candidates_df.groupby('layer').agg({
        'did_median': ['mean', 'std', 'count'],
        'composite_score': 'mean'
    }).round(4)
    
    layer_stats.columns = ['mean_did', 'std_did', 'count', 'mean_composite_score']
    layer_stats = layer_stats.sort_values('mean_composite_score', ascending=False)
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Mean composite score by layer
    ax1.bar(layer_stats.index.astype(str), layer_stats['mean_composite_score'], 
            color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Composite Score')
    ax1.set_title('Mean Composite Score by Layer')
    ax1.grid(axis='y', alpha=0.3)
    
    # Bottom plot: Count of candidates by layer
    ax2.bar(layer_stats.index.astype(str), layer_stats['count'], 
            color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Top Candidates')
    ax2.set_title('Number of Top Candidates by Layer')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Activation Patching Analysis')
    parser.add_argument('--base_dir', required=True, help='Base model data directory')
    parser.add_argument('--instruct_dir', required=True, help='Instruct model data directory')
    parser.add_argument('--model_family', required=True, choices=['llama', 'mistral', 'qwen'],
                       help='Model family')
    parser.add_argument('--output_dir', default='activation_patching_results',
                       help='Output directory for results')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top candidates to identify')
    
    args = parser.parse_args()
    
    print("Activation Patching Analysis")
    print("=" * 50)
    print(f"Model family: {args.model_family}")
    print(f"Base model data: {args.base_dir}")
    print(f"Instruct model data: {args.instruct_dir}")
    print(f"Top-k candidates: {args.top_k}")
    print("=" * 50)
    
    # Load data
    print("\nLoading base model data...")
    base_df = load_activation_data_robust(args.base_dir, args.model_family)
    
    print("\nLoading instruct model data...")
    instruct_df = load_activation_data_robust(args.instruct_dir, args.model_family)
    
    # Find patching candidates using robust methodology
    print(f"\nFinding top {args.top_k} patching candidates using robust methodology...")
    candidates = find_patching_candidates_robust(base_df, instruct_df, args.model_family, args.top_k)
    
    # Display results
    print(f"\n=== TOP {args.top_k} ACTIVATION PATCHING CANDIDATES ===")
    print(f"Model Family: {args.model_family.upper()}")
    print("-" * 60)
    
    for _, row in candidates.iterrows():
        direction = "↓" if row['did_median'] < 0 else "↑"
        print(f"Rank {row['rank']:2d}: Layer {row['layer']:2d}, Head {row['head']:2d} | "
              f"DiD: {row['did_median']:7.4f} {direction} | "
              f"Log-Ratio: {row['did_log_ratio_median']:6.3f} | "
              f"Consistency: {row['did_consistency']:.2f} | "
              f"Score: {row['composite_score']:.2f}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_patching_analysis_plots(candidates, args.output_dir)
    
    # Save detailed results
    candidates_file = f'{args.output_dir}/patching_candidates_{args.model_family}.csv'
    candidates.to_csv(candidates_file, index=False)
    print(f"✓ Detailed results saved to {candidates_file}")
    
    # Save summary statistics
    summary = {
        'model_family': args.model_family,
        'total_candidates': len(candidates),
        'mean_did_median': float(candidates['did_median'].mean()),
        'std_did_median': float(candidates['did_median'].std()),
        'mean_log_ratio_median': float(candidates['did_log_ratio_median'].mean()),
        'std_log_ratio_median': float(candidates['did_log_ratio_median'].std()),
        'mean_consistency': float(candidates['did_consistency'].mean()),
        'mean_composite_score': float(candidates['composite_score'].mean()),
        'layers_with_candidates': sorted(candidates['layer'].unique().tolist()),
        'top_layer': int(candidates.loc[candidates['composite_score'].idxmax(), 'layer']),
        'top_head': int(candidates.loc[candidates['composite_score'].idxmax(), 'head']),
        'methodology': 'robust_median_log_ratio_did'
    }
    
    summary_file = f'{args.output_dir}/patching_summary_{args.model_family}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics saved to {summary_file}")
    
    print(f"\n✓ Analysis complete! Results saved to {args.output_dir}/")
    
    return candidates

if __name__ == "__main__":
    main()
