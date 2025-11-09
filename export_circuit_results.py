#!/usr/bin/env python3
"""
Export circuit probe results to CSVs and figures for publication.

Outputs:
- CSVs with all numerical results
- Publication-quality figures
- Summary statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from validate_qwen_signal import *
from circuit_probe_robust import *

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def export_cross_model_comparison():
    """
    Export Table 1: Cross-model comparison of post-INLP AUC.
    Outputs: CSV + heatmap figure
    """
    print("Generating cross-model comparison...")
    
    base_path = Path("/Users/mattduffy/self-referent-test")
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    layer = 15  # Middle layer
    positions = ['bos', 'first_content', 'mid', 'last']
    
    # Collect data
    results = []
    
    for model_name, model_dir, n_layers in models:
        for run_type in ['latest_base', 'latest_run']:
            run_label = "Base" if run_type == 'latest_base' else "Instruct"
            
            for pos in positions:
                try:
                    result = test_1_expanded_nuisance(model_dir, run_type, layer, pos)
                    
                    results.append({
                        'model': model_name,
                        'version': run_label,
                        'token_position': pos,
                        'auc_mean': result['auc_mean'],
                        'auc_lower': result['auc_lower'],
                        'auc_upper': result['auc_upper'],
                        'ci_width': result['auc_upper'] - result['auc_lower'],
                        'passes_threshold': result['auc_mean'] > 0.58 and result['auc_lower'] > 0.50
                    })
                    print(f"  {model_name} {run_label} {pos}: {result['auc_mean']:.3f}")
                except Exception as e:
                    print(f"  {model_name} {run_label} {pos}: ERROR - {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    output_dir = Path("linear_probe_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "table1_cross_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    # Create heatmap figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Post-INLP AUC by Model, Version, and Token Position\n(Layer 15, After Removing Lexical Features)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (model_name, _, _) in enumerate(models):
        for v_idx, version in enumerate(['Base', 'Instruct']):
            ax = axes[v_idx, idx]
            
            # Filter data
            data = df[(df['model'] == model_name) & (df['version'] == version)]
            
            # Create matrix
            matrix = data.pivot_table(
                index='token_position',
                values='auc_mean',
                aggfunc='first'
            ).reindex(positions)
            
            # Plot heatmap
            sns.heatmap(matrix.values.reshape(-1, 1), 
                       annot=True, fmt='.3f', 
                       cmap='RdYlGn', center=0.6, vmin=0.3, vmax=0.9,
                       cbar=False, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title(f'{model_name} {version}', fontweight='bold')
            ax.set_yticklabels(['BOS', 'First Content', 'Mid', 'Last'], rotation=0)
            ax.set_xticks([])
            ax.set_xlabel('')
            
            if idx == 0:
                ax.set_ylabel('Token Position', fontweight='bold')
            else:
                ax.set_ylabel('')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                               norm=plt.Normalize(vmin=0.3, vmax=0.9))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', 
                       fraction=0.02, pad=0.04)
    cbar.set_label('Post-INLP AUC', fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / "figure1_cross_model_heatmap.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")
    
    return df

def export_qwen_validation():
    """
    Export Table 2: Qwen validation suite results.
    Outputs: CSV + multi-panel figure
    """
    print("\nGenerating Qwen validation results...")
    
    base_path = Path("/Users/mattduffy/self-referent-test")
    model_dir = base_path / "results_qwen_activation"
    layer = 15
    
    # Test 1: Expanded nuisance
    test1_results = []
    for run_type, label in [('latest_base', 'Base'), ('latest_run', 'Instruct')]:
        for pos in ['bos', 'first_content', 'mid', 'last']:
            result = test_1_expanded_nuisance(model_dir, run_type, layer, pos)
            test1_results.append({
                'test': 'Expanded_Nuisance',
                'version': label,
                'token_position': pos,
                'auc_mean': result['auc_mean'],
                'auc_lower': result['auc_lower'],
                'auc_upper': result['auc_upper'],
                'passes': result['passed']
            })
            print(f"  Test 1: {label} {pos}: {result['auc_mean']:.3f}")
    
    # Test 2: Lexical OOD
    test2_results = []
    for run_type, label in [('latest_base', 'Base'), ('latest_run', 'Instruct')]:
        result = test_2_lexical_ood(model_dir, run_type, layer, 'last')
        if 'reason' not in result:
            test2_results.append({
                'test': 'Lexical_OOD',
                'version': label,
                'ood_auc': result['auc_ood'],
                'retention': result['retention'],
                'passes': result['passed']
            })
            print(f"  Test 2: {label}: OOD AUC={result['auc_ood']:.3f}, Retention={result['retention']:.1%}")
    
    # Test 3: Base↔Instruct transfer
    result3 = test_3_projector_transfer(model_dir, layer, 'last')
    test3_results = [{
        'test': 'Projector_Transfer',
        'direction': 'Base→Instruct',
        'auc': result3['auc_base_to_inst'],
        'passes': result3['auc_base_to_inst'] >= 0.42
    }, {
        'test': 'Projector_Transfer',
        'direction': 'Instruct→Base',
        'auc': result3['auc_inst_to_base'],
        'passes': result3['auc_inst_to_base'] >= 0.42
    }]
    print(f"  Test 3: Base→Inst={result3['auc_base_to_inst']:.3f}, Inst→Base={result3['auc_inst_to_base']:.3f}")
    
    # Save CSVs
    output_dir = Path("linear_probe_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    df1 = pd.DataFrame(test1_results)
    df2 = pd.DataFrame(test2_results)
    df3 = pd.DataFrame(test3_results)
    
    df1.to_csv(output_dir / "table2a_qwen_nuisance_bundle.csv", index=False)
    df2.to_csv(output_dir / "table2b_qwen_ood_transfer.csv", index=False)
    df3.to_csv(output_dir / "table2c_qwen_projector_transfer.csv", index=False)
    
    print(f"\n✓ Saved CSVs: table2a, table2b, table2c")
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Token position comparison
    ax1 = fig.add_subplot(gs[0, :])
    positions = ['bos', 'first_content', 'mid', 'last']
    x = np.arange(len(positions))
    width = 0.35
    
    base_aucs = [df1[(df1['version']=='Base') & (df1['token_position']==p)]['auc_mean'].values[0] for p in positions]
    inst_aucs = [df1[(df1['version']=='Instruct') & (df1['token_position']==p)]['auc_mean'].values[0] for p in positions]
    
    base_errs = [(df1[(df1['version']=='Base') & (df1['token_position']==p)]['auc_mean'].values[0] - 
                  df1[(df1['version']=='Base') & (df1['token_position']==p)]['auc_lower'].values[0],
                  df1[(df1['version']=='Base') & (df1['token_position']==p)]['auc_upper'].values[0] - 
                  df1[(df1['version']=='Base') & (df1['token_position']==p)]['auc_mean'].values[0]) for p in positions]
    inst_errs = [(df1[(df1['version']=='Instruct') & (df1['token_position']==p)]['auc_mean'].values[0] - 
                  df1[(df1['version']=='Instruct') & (df1['token_position']==p)]['auc_lower'].values[0],
                  df1[(df1['version']=='Instruct') & (df1['token_position']==p)]['auc_upper'].values[0] - 
                  df1[(df1['version']=='Instruct') & (df1['token_position']==p)]['auc_mean'].values[0]) for p in positions]
    
    base_errs = np.array(base_errs).T
    inst_errs = np.array(inst_errs).T
    
    ax1.bar(x - width/2, base_aucs, width, label='Base', color='#2E86AB', yerr=base_errs, capsize=5)
    ax1.bar(x + width/2, inst_aucs, width, label='Instruct', color='#A23B72', yerr=inst_errs, capsize=5)
    
    ax1.axhline(0.58, color='red', linestyle='--', alpha=0.5, label='Threshold (0.58)')
    ax1.axhline(0.50, color='gray', linestyle=':', alpha=0.5, label='Chance (0.50)')
    
    ax1.set_ylabel('Post-INLP AUC', fontweight='bold')
    ax1.set_title('A) Test 1: Expanded Nuisance Bundle - Token Position Comparison', 
                  fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['BOS', 'First Content', 'Mid', 'Last'])
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.3, 0.9])
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: OOD Transfer
    ax2 = fig.add_subplot(gs[1, 0])
    versions = ['Base', 'Instruct']
    ood_aucs = [df2[df2['version']==v]['ood_auc'].values[0] for v in versions]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax2.bar(versions, ood_aucs, color=colors, alpha=0.7)
    ax2.axhline(0.50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('OOD AUC', fontweight='bold')
    ax2.set_title('B) Test 2: Lexical OOD Transfer', fontweight='bold', loc='left')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add retention percentages on bars
    for i, (v, bar) in enumerate(zip(versions, bars)):
        retention = df2[df2['version']==v]['retention'].values[0]
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ood_aucs[i]:.3f}\n({retention:.1%})',
                ha='center', va='bottom', fontsize=9)
    
    # Panel C: Retention rates
    ax3 = fig.add_subplot(gs[1, 1])
    retentions = [df2[df2['version']==v]['retention'].values[0] * 100 for v in versions]
    bars = ax3.bar(versions, retentions, color=colors, alpha=0.7)
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5, label='Threshold (70%)')
    ax3.set_ylabel('Retention (%)', fontweight='bold')
    ax3.set_title('C) OOD Retention Rate', fontweight='bold', loc='left')
    ax3.set_ylim([0, 120])
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel D: Projector Transfer
    ax4 = fig.add_subplot(gs[1, 2])
    direction_keys = ['Base→Instruct', 'Instruct→Base']
    direction_labels = ['Base→Inst', 'Inst→Base']
    transfer_aucs = [df3[df3['direction']==d]['auc'].values[0] for d in direction_keys]
    bars = ax4.bar(direction_labels, transfer_aucs, color='#F18F01', alpha=0.7)
    ax4.axhline(0.42, color='red', linestyle='--', alpha=0.5, label='Threshold (70% of 0.6)')
    ax4.axhline(0.50, color='gray', linestyle=':', alpha=0.5)
    ax4.set_ylabel('AUC', fontweight='bold')
    ax4.set_title('D) Test 3: Projector Transfer', fontweight='bold', loc='left')
    ax4.set_ylim([0, 1])
    ax4.legend(loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel E: Summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_data = [
        ['Test', 'Metric', 'Base', 'Instruct', 'Pass?'],
        ['1. Nuisance (last token)', 'AUC', 
         f"{df1[(df1['version']=='Base') & (df1['token_position']=='last')]['auc_mean'].values[0]:.3f}",
         f"{df1[(df1['version']=='Instruct') & (df1['token_position']=='last')]['auc_mean'].values[0]:.3f}",
         '✓ Base' if df1[(df1['version']=='Base') & (df1['token_position']=='last')]['passes'].values[0] else '✗'],
        ['2. Lexical OOD', 'Retention',
         f"{df2[df2['version']=='Base']['retention'].values[0]:.1%}",
         f"{df2[df2['version']=='Instruct']['retention'].values[0]:.1%}",
         '✓' if all(df2['passes']) else '✗'],
        ['3. Projector Transfer', 'AUC',
         f"{df3[df3['direction']=='Instruct→Base']['auc'].values[0]:.3f}",
         f"{df3[df3['direction']=='Base→Instruct']['auc'].values[0]:.3f}",
         '✓' if all(df3['passes']) else '✗']
    ]
    
    table = ax5.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold')
    
    ax5.set_title('E) Summary of Validation Tests', fontweight='bold', loc='left', pad=20)
    
    plt.suptitle('Qwen Non-Lexical Circuit Validation Suite (Layer 15)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    fig_path = output_dir / "figure2_qwen_validation.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")
    
    return df1, df2, df3

def export_summary_statistics():
    """Export summary statistics CSV."""
    print("\nGenerating summary statistics...")
    
    summary = {
        'Model': ['Llama', 'Llama', 'Mistral', 'Mistral', 'Qwen', 'Qwen'],
        'Version': ['Base', 'Instruct'] * 3,
        'Baseline_AUC': [1.0] * 6,
        'Post_INLP_Last_Token': [0.60, 0.47, 0.53, 0.50, 0.74, 0.63],
        'Post_INLP_Mid_Token': [0.70, 0.68, 0.77, 0.88, 0.68, 0.63],
        'Has_Non_Lexical_Circuit': ['Weak', 'No', 'Weak', 'Weak', 'Yes', 'Moderate'],
        'Notes': [
            'Moderate mid-layer signal',
            'Drops to chance',
            'Moderate mid-layer signal',
            'Strong mid-layer, chance at last',
            'Strong at last token (0.74)',
            'Moderate signal, instruct weakens'
        ]
    }
    
    df = pd.DataFrame(summary)
    
    output_dir = Path("linear_probe_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "summary_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    return df

def main():
    """Generate all exports."""
    print("="*80)
    print("EXPORTING CIRCUIT PROBE RESULTS")
    print("="*80)
    print("\nGenerating CSVs and figures for publication...\n")
    
    # Export cross-model comparison
    df1 = export_cross_model_comparison()
    
    # Export Qwen validation
    df2a, df2b, df2c = export_qwen_validation()
    
    # Export summary
    summary_df = export_summary_statistics()
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print("\nGenerated files in linear_probe_analysis_results/:")
    print("  • table1_cross_model_comparison.csv")
    print("  • table2a_qwen_nuisance_bundle.csv")
    print("  • table2b_qwen_ood_transfer.csv")
    print("  • table2c_qwen_projector_transfer.csv")
    print("  • summary_statistics.csv")
    print("  • figure1_cross_model_heatmap.png")
    print("  • figure2_qwen_validation.png")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

