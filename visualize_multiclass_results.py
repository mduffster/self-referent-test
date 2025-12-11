#!/usr/bin/env python3
"""
Create publication-quality visualizations for multi-class circuit probe results.
Emphasizes the Qwen divergence: maintains self vs third distinction after INLP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_pairwise_results(results_dir):
    """Load all pairwise results."""
    results_path = Path(results_dir)
    dfs = []
    for csv_file in results_path.glob("*_pairwise_results.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def create_primary_visualization(df_pairwise, output_dir):
    """
    PRIMARY: Self vs Third INLP AUC by token position.
    Shows Qwen maintains distinction, Llama/Mistral drop to chance.
    """
    # Filter for self_vs_third, instruct models only
    df_filtered = df_pairwise[
        (df_pairwise['comparison'] == 'self_vs_third') &
        (df_pairwise['version'] == 'instruct')
    ].copy()
    
    # Token position order
    token_order = ['bos', 'first_content', 'mid', 'last']
    token_labels = ['BOS', 'First Content', 'Mid', 'Last']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Model colors
    colors = {
        'Llama': '#2E86AB',
        'Mistral': '#E63946',
        'Qwen': '#06A77D'
    }
    
    # Plot each model
    for model in ['Llama', 'Mistral', 'Qwen']:
        model_data = df_filtered[df_filtered['model'] == model]
        
        # Get INLP AUC by token position (average across layers)
        inlp_aucs = []
        for token_pos in token_order:
            token_data = model_data[model_data['token_pos'] == token_pos]
            if len(token_data) > 0:
                inlp_aucs.append(token_data['inlp_auc'].mean())
            else:
                inlp_aucs.append(np.nan)
        
        # Plot line
        x = np.arange(len(token_order))
        line_width = 4 if model == 'Qwen' else 3
        alpha = 1.0 if model == 'Qwen' else 0.7
        
        ax.plot(x, inlp_aucs, 
               marker='o',
               markersize=14 if model == 'Qwen' else 12,
               linewidth=line_width,
               color=colors[model],
               label=model,
               alpha=alpha,
               zorder=3 if model == 'Qwen' else 2)
    
    # Add chance line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, 
              alpha=0.5, label='Chance', zorder=1)
    
    # Formatting
    ax.set_xlabel('Token Position', fontsize=16, weight='bold')
    ax.set_ylabel('INLP AUC (After Pronoun Removal)', fontsize=16, weight='bold')
    ax.set_title('Self vs Third Person: Signal Persistence After INLP\n(Instruct Models)', 
                fontsize=18, weight='bold', pad=20)
    ax.set_xticks(np.arange(len(token_order)))
    ax.set_xticklabels(token_labels, fontsize=14)
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=14, 
             frameon=True, edgecolor='black', fancybox=False)
    
    # Add annotation for Qwen
    ax.text(0.98, 0.05, 
           'Qwen maintains perfect distinction\nLlama/Mistral drop to chance',
           transform=ax.transAxes,
           ha='right', va='bottom',
           fontsize=12, style='italic',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#06A77D', 
                    alpha=0.15, edgecolor='#06A77D', linewidth=2))
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure_primary_qwen_divergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def create_secondary_visualization(df_pairwise, output_dir):
    """
    SECONDARY: Heatmap of INLP AUC drops by model × comparison.
    Shows Qwen has minimal drops, Llama/Mistral have large drops.
    """
    # Filter for instruct, mid token position
    df_filtered = df_pairwise[
        (df_pairwise['version'] == 'instruct') &
        (df_pairwise['token_pos'] == 'mid')
    ].copy()
    
    # Comparisons in order
    comparisons = [
        'self_vs_confounder',
        'self_vs_third',
        'self_vs_neutral',
        'third_vs_neutral',
        'third_vs_confounder',
        'neutral_vs_confounder'
    ]
    
    comparison_labels = [
        'Self vs Confounder\n(I vs you)',
        'Self vs Third\n(you vs he/she)',
        'Self vs Neutral\n(you vs none)',
        'Third vs Neutral\n(he/she vs none)',
        'Third vs Confounder\n(he/she vs I)',
        'Neutral vs Confounder\n(none vs I)'
    ]
    
    models = ['Llama', 'Mistral', 'Qwen']
    
    # Build matrix
    matrix = np.zeros((len(comparisons), len(models)))
    
    for i, comparison in enumerate(comparisons):
        for j, model in enumerate(models):
            data = df_filtered[
                (df_filtered['model'] == model) &
                (df_filtered['comparison'] == comparison)
            ]
            if len(data) > 0:
                matrix[i, j] = data['auc_drop'].mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Custom colormap: blue (no drop) to red (large drop)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-0.1, vmax=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC Drop (Baseline - INLP)', rotation=270, labelpad=25, 
                  fontsize=13, weight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_xticklabels(models, fontsize=14, weight='bold')
    ax.set_yticklabels(comparison_labels, fontsize=11)
    
    # Add text annotations
    for i in range(len(comparisons)):
        for j in range(len(models)):
            value = matrix[i, j]
            color = 'white' if value > 0.3 else 'black'
            ax.text(j, i, f'{value:.3f}',
                   ha="center", va="center", 
                   color=color, fontsize=12, weight='bold')
    
    ax.set_title('INLP Impact: AUC Drop by Model and Comparison\n(Instruct Models, Mid Token)', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, weight='bold')
    ax.set_ylabel('Comparison Type', fontsize=14, weight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(models)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(comparisons)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure_secondary_auc_drop_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def create_tertiary_visualization(df_pairwise, output_dir):
    """
    TERTIARY: Paired bar chart for self_vs_third baseline vs INLP.
    Shows the collapse for Llama/Mistral and preservation for Qwen.
    """
    # Filter for self_vs_third, instruct, mid token
    df_filtered = df_pairwise[
        (df_pairwise['comparison'] == 'self_vs_third') &
        (df_pairwise['version'] == 'instruct') &
        (df_pairwise['token_pos'] == 'mid')
    ].copy()
    
    models = ['Llama', 'Mistral', 'Qwen']
    
    # Get average baseline and INLP AUC for each model
    baseline_aucs = []
    inlp_aucs = []
    
    for model in models:
        data = df_filtered[df_filtered['model'] == model]
        baseline_aucs.append(data['baseline_auc'].mean())
        inlp_aucs.append(data['inlp_auc'].mean())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    colors_base = ['#5BA3C7', '#F07D91', '#4ABFA2']  # Lighter versions
    colors_inlp = ['#2E86AB', '#E63946', '#06A77D']  # Darker versions
    
    # Plot bars
    bars1 = ax.bar(x - width/2, baseline_aucs, width, 
                   label='Baseline (with pronouns)',
                   color=colors_base,
                   edgecolor='black',
                   linewidth=2,
                   alpha=0.9)
    
    bars2 = ax.bar(x + width/2, inlp_aucs, width,
                   label='INLP (pronouns removed)',
                   color=colors_inlp,
                   edgecolor='black',
                   linewidth=2,
                   alpha=0.9)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.02,
               f'{height1:.3f}',
               ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.02,
               f'{height2:.3f}',
               ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Add drop annotation
        drop = baseline_aucs[i] - inlp_aucs[i]
        if abs(drop) > 0.05:  # Only show if significant drop
            mid_x = x[i]
            mid_y = (height1 + height2) / 2
            ax.annotate('', xy=(mid_x + width/2, height2), 
                       xytext=(mid_x - width/2, height1),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(mid_x, mid_y, f'Δ{drop:.3f}',
                   ha='center', va='center',
                   fontsize=11, weight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='red', linewidth=1.5))
    
    # Add chance line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, 
              alpha=0.5, label='Chance', zorder=1)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=16, weight='bold')
    ax.set_ylabel('AUC', fontsize=16, weight='bold')
    ax.set_title('Self vs Third Person: Baseline vs INLP\n(Instruct Models, Mid Token)', 
                fontsize=18, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=15, weight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=13,
             frameon=True, edgecolor='black', fancybox=False)
    
    # Add interpretation box
    ax.text(0.98, 0.05,
           'Llama & Mistral: Signal collapses to chance\nQwen: Signal preserved perfectly',
           transform=ax.transAxes,
           ha='right', va='bottom',
           fontsize=12, style='italic',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', 
                    alpha=0.2, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure_tertiary_baseline_vs_inlp.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def create_bonus_all_comparisons_overview(df_pairwise, output_dir):
    """
    BONUS: Overview of all key comparisons showing Qwen's robustness.
    """
    # Filter for instruct, mid token
    df_filtered = df_pairwise[
        (df_pairwise['version'] == 'instruct') &
        (df_pairwise['token_pos'] == 'mid')
    ].copy()
    
    key_comparisons = ['self_vs_confounder', 'self_vs_third', 'self_vs_neutral']
    comparison_labels = ['Self vs Confounder\n(Critical Test)', 
                        'Self vs Third\n(Qwen Divergence)', 
                        'Self vs Neutral']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    models = ['Llama', 'Mistral', 'Qwen']
    colors = {'Llama': '#2E86AB', 'Mistral': '#E63946', 'Qwen': '#06A77D'}
    
    for idx, (comparison, label) in enumerate(zip(key_comparisons, comparison_labels)):
        ax = axes[idx]
        
        baseline_vals = []
        inlp_vals = []
        
        for model in models:
            data = df_filtered[
                (df_filtered['model'] == model) &
                (df_filtered['comparison'] == comparison)
            ]
            baseline_vals.append(data['baseline_auc'].mean())
            inlp_vals.append(data['inlp_auc'].mean())
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, baseline_vals, width, 
              label='Baseline', alpha=0.7, color='lightgray', 
              edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, inlp_vals, width,
              label='INLP', alpha=0.9,
              color=[colors[m] for m in models],
              edgecolor='black', linewidth=1.5)
        
        # Add values
        for i, (b, inlp) in enumerate(zip(baseline_vals, inlp_vals)):
            ax.text(x[i] + width/2, inlp + 0.03, f'{inlp:.2f}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        
        ax.set_xlabel('Model', fontsize=12, weight='bold')
        if idx == 0:
            ax.set_ylabel('AUC', fontsize=12, weight='bold')
        ax.set_title(label, fontsize=13, weight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left')
    
    plt.suptitle('INLP Robustness Across Key Comparisons (Instruct Models)', 
                fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'figure_bonus_all_comparisons.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("CREATING VISUALIZATIONS: QWEN DIVERGENCE ANALYSIS")
    print("="*80)
    
    # Paths
    results_dir = Path("linear_probe_multiclass")
    output_dir = results_dir / "summary"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading pairwise results...")
    df_pairwise = load_pairwise_results(results_dir)
    print(f"  Loaded {len(df_pairwise)} comparisons")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    create_primary_visualization(df_pairwise, output_dir)
    create_secondary_visualization(df_pairwise, output_dir)
    create_tertiary_visualization(df_pairwise, output_dir)
    create_bonus_all_comparisons_overview(df_pairwise, output_dir)
    
    print("\n" + "-" * 80)
    print(f"✓ All visualizations saved to: {output_dir}/")
    print("\nKEY VISUALIZATIONS:")
    print("  1. figure_primary_qwen_divergence.png     - Line chart showing Qwen maintains signal")
    print("  2. figure_secondary_auc_drop_heatmap.png  - Heatmap of AUC drops by model")
    print("  3. figure_tertiary_baseline_vs_inlp.png   - Bar chart showing the collapse")
    print("  4. figure_bonus_all_comparisons.png       - Overview of all comparisons")
    print("="*80)


if __name__ == "__main__":
    main()

