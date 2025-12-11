#!/usr/bin/env python3
"""
Visualize circuit probe results: AUC by token position across models.
Shows Llama and Mistral (base vs instruct comparison) and Qwen (signal persistence).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the cross-model comparison data."""
    df = pd.read_csv(csv_path)
    return df


def create_auc_by_position_plot(df: pd.DataFrame, output_dir: str = "linear_probe_analysis_results"):
    """
    Create a grouped bar chart showing AUC by token position for each model.
    
    Layout:
    - Three model groups: Llama, Mistral, Qwen
    - Each group shows Base vs Instruct
    - X-axis: token positions (bos, first_content, mid, last)
    - Y-axis: AUC
    """
    # Define token positions in order
    positions = ['bos', 'first_content', 'mid', 'last']
    position_labels = ['BOS', 'First Content', 'Mid', 'Last']
    
    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    models = ['Llama', 'Mistral', 'Qwen']
    colors = {
        'Base': '#2E86AB',      # Blue
        'Instruct': '#A23B72'   # Purple/Magenta
    }
    
    for idx, (ax, model) in enumerate(zip(axes, models)):
        model_data = df[df['model'] == model]
        
        x = np.arange(len(positions))
        width = 0.35
        
        # Plot bars for Base and Instruct
        for i, version in enumerate(['Base', 'Instruct']):
            version_data = model_data[model_data['version'] == version]
            
            aucs = []
            errors_lower = []
            errors_upper = []
            
            for pos in positions:
                pos_data = version_data[version_data['token_position'] == pos]
                if not pos_data.empty:
                    auc_mean = pos_data['auc_mean'].values[0]
                    auc_lower = pos_data['auc_lower'].values[0]
                    auc_upper = pos_data['auc_upper'].values[0]
                    
                    aucs.append(auc_mean)
                    errors_lower.append(auc_mean - auc_lower)
                    errors_upper.append(auc_upper - auc_mean)
                else:
                    aucs.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
            
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, aucs, width, 
                         label=version,
                         color=colors[version],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1)
            
            # Add error bars
            ax.errorbar(x + offset, aucs, 
                       yerr=[errors_lower, errors_upper],
                       fmt='none',
                       ecolor='black',
                       capsize=4,
                       capthick=1,
                       alpha=0.6)
            
        
        # Formatting
        ax.set_xlabel('Token Position', fontsize=12, weight='bold')
        if idx == 0:
            ax.set_ylabel('AUC', fontsize=12, weight='bold')
        ax.set_title(f'{model}', fontsize=14, weight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(position_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Add text annotations for Qwen to highlight persistence
        if model == 'Qwen':
            ax.text(0.5, 0.95, 'Signal Persistence →',
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=11, style='italic',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'figure3_auc_by_position.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def create_detailed_comparison_plot(df: pd.DataFrame, output_dir: str = "linear_probe_analysis_results"):
    """
    Create a clean line plot showing AUC progression across token positions.
    Shows Instruct models only to highlight the different patterns:
    Llama/Mistral peak at mid, Qwen shows persistence.
    """
    positions = ['bos', 'first_content', 'mid', 'last']
    position_labels = ['BOS', 'First Content', 'Mid', 'Last']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['Llama', 'Mistral', 'Qwen']
    version = 'Instruct'  # Only show Instruct models
    
    # Clean color scheme
    colors = {
        'Llama': '#2E86AB',    # Blue
        'Mistral': '#E63946',  # Red
        'Qwen': '#06A77D'      # Green
    }
    
    for model in models:
        model_version_data = df[(df['model'] == model) & (df['version'] == version)]
        
        aucs = []
        
        for pos in positions:
            pos_data = model_version_data[model_version_data['token_position'] == pos]
            if not pos_data.empty:
                auc_mean = pos_data['auc_mean'].values[0]
                aucs.append(auc_mean)
            else:
                aucs.append(np.nan)
        
        x = np.arange(len(positions))
        
        # Plot line with markers
        ax.plot(x, aucs, 
               marker='o',
               markersize=12,
               linewidth=3,
               color=colors[model],
               label=model,
               alpha=0.85)
    
    # Formatting
    ax.set_xlabel('Token Position', fontsize=13, weight='bold')
    ax.set_ylabel('AUC', fontsize=13, weight='bold')
    ax.set_title('Self-Referent Signal by Token Position (Instruct)', 
                fontsize=15, weight='bold', pad=20)
    ax.set_xticks(np.arange(len(positions)))
    ax.set_xticklabels(position_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, frameon=True, edgecolor='black')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'figure4_auc_progression.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def create_heatmap_plot(df: pd.DataFrame, output_dir: str = "linear_probe_analysis_results"):
    """
    Create a heatmap showing AUC values across all model-version-position combinations.
    """
    # Pivot the data for heatmap
    positions = ['bos', 'first_content', 'mid', 'last']
    
    # Create row labels
    rows = []
    for model in ['Llama', 'Mistral', 'Qwen']:
        for version in ['Base', 'Instruct']:
            rows.append(f"{model}-{version}")
    
    # Create matrix
    matrix = np.zeros((len(rows), len(positions)))
    significance_matrix = np.zeros((len(rows), len(positions)), dtype=bool)
    
    for i, row_label in enumerate(rows):
        model, version = row_label.split('-')
        for j, pos in enumerate(positions):
            data = df[(df['model'] == model) & 
                     (df['version'] == version) & 
                     (df['token_position'] == pos)]
            if not data.empty:
                matrix[i, j] = data['auc_mean'].values[0]
                significance_matrix[i, j] = data['passes_threshold'].values[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(positions)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(['BOS', 'First Content', 'Mid', 'Last'], fontsize=11)
    ax.set_yticklabels(rows, fontsize=11)
    
    # Add text annotations
    for i in range(len(rows)):
        for j in range(len(positions)):
            # Bold text for significant values
            weight = 'bold' if significance_matrix[i, j] else 'normal'
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", 
                          fontsize=10, weight=weight)
    
    ax.set_title('Self-Referent Signal Strength Heatmap\n(Bold = Passes Significance Threshold)', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Token Position', fontsize=12, weight='bold')
    ax.set_ylabel('Model-Version', fontsize=12, weight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(positions)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'figure5_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def main():
    """Main execution function."""
    # Load data
    csv_path = "linear_probe_analysis_results/table1_cross_model_comparison.csv"
    df = load_data(csv_path)
    
    print("Creating visualizations...")
    print("-" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = "linear_probe_analysis_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate all visualizations
    print("\nCreating Figure 3: AUC by position (grouped bar chart)...")
    create_auc_by_position_plot(df, output_dir)
    
    print("\nCreating Figure 4: AUC progression (line plot)...")
    create_detailed_comparison_plot(df, output_dir)
    
    print("\nCreating Figure 5: Heatmap...")
    create_heatmap_plot(df, output_dir)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("=" * 60)
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

