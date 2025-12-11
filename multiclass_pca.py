#!/usr/bin/env python3
"""
PCA analysis of multi-class categories after INLP.
Reveals the geometric structure of self vs confounder vs third vs neutral.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def load_activations(model_dir, run_type, layer, token_position='mid'):
    """Load MLP post activations."""
    npz_path = model_dir / run_type / f"raw_blocks_{layer}_mlp_post.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    activations = []
    for act in data['activations']:
        if token_position == 'last':
            token = act[0, -1, :]
        elif token_position == 'bos':
            token = act[0, 0, :]
        elif token_position == 'first_content':
            token = act[0, 1, :] if act.shape[1] > 1 else act[0, 0, :]
        elif token_position == 'mid':
            mid_idx = act.shape[1] // 2
            token = act[0, mid_idx, :]
        else:
            raise ValueError(f"Unknown token_position: {token_position}")
        activations.append(token)
    
    return np.array(activations), data['categories'], data['prompts']


def extract_pronoun_labels(prompts):
    """Extract pronoun labels from prompts."""
    labels = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        if 'you' in prompt_lower or 'your' in prompt_lower or 'yourself' in prompt_lower:
            labels.append('2nd')
        elif ' i ' in prompt_lower or ' my ' in prompt_lower or ' me ' in prompt_lower or 'am i' in prompt_lower or 'do i' in prompt_lower or 'should i' in prompt_lower:
            labels.append('1st')
        elif ' he ' in prompt_lower or ' she ' in prompt_lower or ' his ' in prompt_lower or ' her ' in prompt_lower or 'does he' in prompt_lower or 'does she' in prompt_lower or 'is he' in prompt_lower or 'is she' in prompt_lower or 'should he' in prompt_lower or 'should she' in prompt_lower:
            labels.append('3rd')
        else:
            labels.append('none')
    return np.array(labels)


def apply_inlp_comprehensive(X, prompts):
    """
    Apply comprehensive INLP to remove all pronoun distinctions.
    Returns the projection-cleaned data.
    """
    pronoun_labels = extract_pronoun_labels(prompts)
    
    X_current = X.copy()
    
    # Pass 1: Remove 1st vs 2nd (I vs you)
    mask_12 = np.isin(pronoun_labels, ['1st', '2nd'])
    if mask_12.sum() > 10 and len(np.unique(pronoun_labels[mask_12])) == 2:
        y_12 = (pronoun_labels[mask_12] == '2nd').astype(int)
        
        X_subset = X_current[mask_12]
        for iteration in range(10):
            clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            clf.fit(X_subset, y_12)
            
            acc = clf.score(X_subset, y_12)
            if acc < 0.55:
                break
            
            direction = clf.coef_[0]
            d = direction / np.linalg.norm(direction)
            X_subset = X_subset - np.outer(X_subset @ d, d)
        
        X_current[mask_12] = X_subset
    
    # Pass 2: Remove 2nd vs 3rd (you vs he/she)
    mask_23 = np.isin(pronoun_labels, ['2nd', '3rd'])
    if mask_23.sum() > 10 and len(np.unique(pronoun_labels[mask_23])) == 2:
        y_23 = (pronoun_labels[mask_23] == '2nd').astype(int)
        
        X_subset = X_current[mask_23]
        for iteration in range(10):
            clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            clf.fit(X_subset, y_23)
            
            acc = clf.score(X_subset, y_23)
            if acc < 0.55:
                break
            
            direction = clf.coef_[0]
            d = direction / np.linalg.norm(direction)
            X_subset = X_subset - np.outer(X_subset @ d, d)
        
        X_current[mask_23] = X_subset
    
    # Pass 3: Remove 1st vs 3rd (I vs he/she)
    mask_13 = np.isin(pronoun_labels, ['1st', '3rd'])
    if mask_13.sum() > 10 and len(np.unique(pronoun_labels[mask_13])) == 2:
        y_13 = (pronoun_labels[mask_13] == '1st').astype(int)
        
        X_subset = X_current[mask_13]
        for iteration in range(10):
            clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            clf.fit(X_subset, y_13)
            
            acc = clf.score(X_subset, y_13)
            if acc < 0.55:
                break
            
            direction = clf.coef_[0]
            d = direction / np.linalg.norm(direction)
            X_subset = X_subset - np.outer(X_subset @ d, d)
        
        X_current[mask_13] = X_subset
    
    # Pass 4: Remove has-pronoun vs no-pronoun
    y_has = (pronoun_labels != 'none').astype(int)
    if len(np.unique(y_has)) == 2:
        for iteration in range(10):
            clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            clf.fit(X_current, y_has)
            
            acc = clf.score(X_current, y_has)
            if acc < 0.55:
                break
            
            direction = clf.coef_[0]
            d = direction / np.linalg.norm(direction)
            X_current = X_current - np.outer(X_current @ d, d)
    
    return X_current


def analyze_pca_structure(model_name, model_dir, run_type, layer, token_position, output_dir):
    """
    Analyze PCA structure after INLP.
    """
    print(f"  Analyzing {model_name} {run_type} Layer {layer} Token {token_position}")
    
    # Load data
    X, categories, prompts = load_activations(model_dir, run_type, layer, token_position)
    
    # Remove NaN
    nan_mask = ~np.isnan(X).any(axis=1)
    X = X[nan_mask]
    categories = categories[nan_mask]
    prompts = prompts[nan_mask]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply comprehensive INLP
    X_inlp = apply_inlp_comprehensive(X_scaled, prompts)
    
    # Run PCA on INLP-cleaned data
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_inlp)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: PC1 vs PC2
    ax = axes[0]
    
    colors = {
        'self_referent': '#2E86AB',   # Blue
        'confounder': '#E63946',      # Red
        'third_person': '#F77F00',    # Orange
        'neutral': '#06A77D'          # Green
    }
    
    labels_map = {
        'self_referent': 'Self-Referent',
        'confounder': 'Confounder',
        'third_person': 'Third Person',
        'neutral': 'Neutral'
    }
    
    for cat in ['self_referent', 'confounder', 'third_person', 'neutral']:
        mask = categories == cat
        if mask.sum() > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=colors[cat], label=labels_map[cat],
                      s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                 fontsize=12, weight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                 fontsize=12, weight='bold')
    ax.set_title(f'PCA after INLP: {model_name} {run_type}\nLayer {layer}, {token_position} token', 
                fontsize=13, weight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
    
    # Plot 2: Scree plot
    ax = axes[1]
    ax.bar(range(1, 11), pca.explained_variance_ratio_[:10], 
           color='steelblue', edgecolor='black', linewidth=1)
    ax.set_xlabel('Principal Component', fontsize=12, weight='bold')
    ax.set_ylabel('Explained Variance Ratio', fontsize=12, weight='bold')
    ax.set_title('Variance Explained by Components', fontsize=13, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'pca_{model_name.lower()}_{run_type}_layer{layer}_{token_position}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute statistics
    stats = {
        'model': model_name,
        'run_type': run_type,
        'layer': layer,
        'token_pos': token_position,
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1]),
        'total_variance_2pcs': float(pca.explained_variance_ratio_[:2].sum())
    }
    
    # Compute category centroids in PC space
    for cat in ['self_referent', 'confounder', 'third_person', 'neutral']:
        mask = categories == cat
        if mask.sum() > 0:
            centroid = X_pca[mask, :2].mean(axis=0)
            stats[f'{cat}_pc1'] = float(centroid[0])
            stats[f'{cat}_pc2'] = float(centroid[1])
    
    return stats


def create_summary_statistics(output_dir):
    """
    Create summary table from PCA statistics.
    Computes interpretable metrics about category structure.
    """
    import pandas as pd
    
    stats_file = output_dir / 'pca_statistics.csv'
    if not stats_file.exists():
        print("No statistics file found")
        return
    
    df = pd.read_csv(stats_file)
    
    summary_rows = []
    
    for _, row in df.iterrows():
        # Extract centroid positions
        self_pc1 = row['self_referent_pc1']
        self_pc2 = row['self_referent_pc2']
        conf_pc1 = row['confounder_pc1']
        conf_pc2 = row['confounder_pc2']
        third_pc1 = row['third_person_pc1']
        third_pc2 = row['third_person_pc2']
        neutral_pc1 = row['neutral_pc1']
        neutral_pc2 = row['neutral_pc2']
        
        # Compute distances
        self_conf_dist = np.sqrt((self_pc1 - conf_pc1)**2 + (self_pc2 - conf_pc2)**2)
        self_third_dist = np.sqrt((self_pc1 - third_pc1)**2 + (self_pc2 - third_pc2)**2)
        self_neutral_dist = np.sqrt((self_pc1 - neutral_pc1)**2 + (self_pc2 - neutral_pc2)**2)
        
        # PC1 separation (abs difference in PC1)
        self_conf_pc1_sep = abs(self_pc1 - conf_pc1)
        
        # Check if structure is primarily on PC1 (axial)
        pc1_dominance = row['pc1_variance'] / (row['pc1_variance'] + row['pc2_variance'])
        
        # Determine structure type
        if self_conf_pc1_sep > 2.0 and pc1_dominance > 0.6:
            structure_type = 'Axial (PC1)'
        elif self_conf_dist > 2.0:
            structure_type = 'Distributed'
        else:
            structure_type = 'Clustered'
        
        summary_rows.append({
            'model': row['model'],
            'version': 'Base' if row['run_type'] == 'latest_base' else 'Instruct',
            'layer': row['layer'],
            'token_pos': row['token_pos'],
            'pc1_variance_pct': row['pc1_variance'] * 100,
            'pc2_variance_pct': row['pc2_variance'] * 100,
            'self_conf_distance': self_conf_dist,
            'self_third_distance': self_third_dist,
            'self_neutral_distance': self_neutral_dist,
            'pc1_separation_self_conf': self_conf_pc1_sep,
            'structure_type': structure_type
        })
    
    df_summary = pd.DataFrame(summary_rows)
    output_path = output_dir / 'pca_summary.csv'
    df_summary.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✓ Created summary: {output_path}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS (Instruct Models, Mid Token):")
    print("="*80)
    
    instruct_mid = df_summary[(df_summary['version'] == 'Instruct') & 
                              (df_summary['token_pos'] == 'mid')]
    
    for _, row in instruct_mid.iterrows():
        print(f"\n{row['model']} Layer {int(row['layer'])}:")
        print(f"  Structure: {row['structure_type']}")
        print(f"  PC1 variance: {row['pc1_variance_pct']:.1f}%")
        print(f"  Self-Confounder distance: {row['self_conf_distance']:.2f}")
        print(f"  Self-Neutral distance: {row['self_neutral_distance']:.2f}")
        print(f"  PC1 separation (Self-Conf): {row['pc1_separation_self_conf']:.2f}")
    
    return df_summary


def create_comparison_plot(output_dir):
    """
    Create a comparison plot showing all models/conditions side by side.
    """
    import pandas as pd
    
    stats_file = output_dir / 'pca_statistics.csv'
    if not stats_file.exists():
        print("No statistics file found")
        return
    
    df = pd.read_csv(stats_file)
    
    # Filter for instruct models, mid token
    df_filtered = df[(df['run_type'] == 'latest_run') & (df['token_pos'] == 'mid')]
    
    if len(df_filtered) == 0:
        print("No data for comparison plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Llama', 'Mistral', 'Qwen']
    colors = {
        'self_referent': '#2E86AB',
        'confounder': '#E63946',
        'third_person': '#F77F00',
        'neutral': '#06A77D'
    }
    
    labels_map = {
        'self_referent': 'Self',
        'confounder': 'Conf',
        'third_person': 'Third',
        'neutral': 'Neut'
    }
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = df_filtered[df_filtered['model'] == model]
        
        if len(model_data) == 0:
            continue
        
        # Get the row (should be layer 15 or 20)
        row = model_data.iloc[0]
        
        # Plot centroids
        for cat in ['self_referent', 'confounder', 'third_person', 'neutral']:
            pc1_col = f'{cat}_pc1'
            pc2_col = f'{cat}_pc2'
            
            if pc1_col in row and pc2_col in row:
                ax.scatter(row[pc1_col], row[pc2_col],
                          c=colors[cat], s=300, 
                          edgecolors='black', linewidth=2,
                          alpha=0.8, marker='o')
                ax.text(row[pc1_col], row[pc2_col] + 0.5, labels_map[cat],
                       ha='center', va='bottom', fontsize=11, weight='bold')
        
        var1 = row['pc1_variance'] * 100
        var2 = row['pc2_variance'] * 100
        
        ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=12, weight='bold')
        if idx == 0:
            ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=12, weight='bold')
        ax.set_title(f'{model}', fontsize=14, weight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        
        # Make axes equal for better interpretation
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        max_abs = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
    
    plt.suptitle('Category Structure After INLP (Centroid Positions)\nInstruct Models, Mid Token',
                fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'pca_comparison_centroids.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*80)
    print("PCA ANALYSIS: CATEGORY STRUCTURE AFTER INLP")
    print("="*80)
    
    base_path = Path("/Users/mattduffy/self-referent-test")
    output_dir = base_path / "linear_probe_multiclass" / "pca_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    # Test key configurations
    layers_to_test = [15, 20]
    token_positions = ['mid', 'last']
    
    all_stats = []
    
    print("\nGenerating PCA visualizations...")
    print("-" * 80)
    
    for model_name, model_dir, n_layers in models:
        for run_type in ['latest_base', 'latest_run']:
            run_label = 'base' if run_type == 'latest_base' else 'instruct'
            
            for layer in layers_to_test:
                if layer >= n_layers:
                    continue
                
                for token_pos in token_positions:
                    try:
                        stats = analyze_pca_structure(
                            model_name, model_dir, run_type, 
                            layer, token_pos, output_dir
                        )
                        all_stats.append(stats)
                    except Exception as e:
                        print(f"  Error: {model_name} {run_label} L{layer} {token_pos}: {e}")
    
    # Save statistics
    import pandas as pd
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(output_dir / 'pca_statistics.csv', index=False, float_format='%.4f')
    print(f"\n✓ Saved raw statistics: {output_dir / 'pca_statistics.csv'}")
    
    # Create summary statistics
    print("\nGenerating summary statistics...")
    create_summary_statistics(output_dir)
    
    # Create comparison plot
    print("\nCreating comparison visualization...")
    create_comparison_plot(output_dir)
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("\nLook for these patterns:")
    print("  • Axial: Self and Confounder at opposite ends of PC1, others in middle")
    print("  • Orthogonal: One pair separated by PC1, another by PC2")
    print("  • Self-special: Self forms distinct cluster, others overlap")
    print("  • No structure: All four categories overlap (signal destroyed by INLP)")
    print("\nCheck the individual plots for detailed structure per model/layer.")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

