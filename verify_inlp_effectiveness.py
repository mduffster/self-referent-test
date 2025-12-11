#!/usr/bin/env python3
"""
Verify that INLP actually removes pronoun information.
Tests: Can we predict pronouns AFTER INLP?
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def load_activations(model_dir, run_type, layer, token_position='mid'):
    """Load MLP post activations."""
    npz_path = model_dir / run_type / f"raw_blocks_{layer}_mlp_post.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    activations = []
    for act in data['activations']:
        if token_position == 'mid':
            mid_idx = act.shape[1] // 2
            token = act[0, mid_idx, :]
        elif token_position == 'last':
            token = act[0, -1, :]
        else:
            token = act[0, 0, :]
        activations.append(token)
    
    return np.array(activations), data['categories'], data['prompts']


def extract_pronoun_labels(prompts):
    """Extract pronoun labels."""
    labels = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        if 'you' in prompt_lower or 'your' in prompt_lower or 'yourself' in prompt_lower:
            labels.append('2nd')
        elif ' i ' in prompt_lower or ' my ' in prompt_lower or ' me ' in prompt_lower or 'am i' in prompt_lower:
            labels.append('1st')
        elif ' he ' in prompt_lower or ' she ' in prompt_lower or ' his ' in prompt_lower or ' her ' in prompt_lower:
            labels.append('3rd')
        else:
            labels.append('none')
    return np.array(labels)


def apply_inlp_and_track(X, y_nuisance, n_iterations=10):
    """
    Apply INLP and track pronoun prediction at each step.
    Returns: (X_projected, auc_history, acc_history)
    """
    X_current = X.copy()
    auc_history = []
    acc_history = []
    
    for iteration in range(n_iterations):
        # Split for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_current, y_nuisance, test_size=0.3, random_state=42
        )
        
        # Train pronoun classifier
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        auc_history.append(auc)
        acc_history.append(acc)
        
        # Stop if near chance
        if acc < 0.55:
            break
        
        # Project out on full data
        clf_full = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf_full.fit(X_current, y_nuisance)
        
        direction = clf_full.coef_[0]
        d = direction / np.linalg.norm(direction)
        X_current = X_current - np.outer(X_current @ d, d)
    
    return X_current, auc_history, acc_history


def test_comparison(model_name, model_dir, run_type, layer, comparison_name, cat1, cat2, inlp_type):
    """
    Test INLP effectiveness for a specific comparison.
    """
    X, categories, prompts = load_activations(model_dir, run_type, layer, 'mid')
    
    # Filter to comparison
    mask = (categories == cat1) | (categories == cat2)
    X_filtered = X[mask]
    prompts_filtered = prompts[mask]
    
    # Remove NaN
    nan_mask = ~np.isnan(X_filtered).any(axis=1)
    X_filtered = X_filtered[nan_mask]
    prompts_filtered = prompts_filtered[nan_mask]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    # Get pronoun labels
    pronoun_labels = extract_pronoun_labels(prompts_filtered)
    
    # Define nuisance labels based on INLP type
    if inlp_type == '1st_vs_2nd':
        # I vs you
        mask_inlp = np.isin(pronoun_labels, ['1st', '2nd'])
        if mask_inlp.sum() < 10:
            return None
        y_nuisance = (pronoun_labels[mask_inlp] == '2nd').astype(int)
        X_inlp = X_scaled[mask_inlp]
    elif inlp_type == '2nd_vs_3rd':
        # you vs he/she
        mask_inlp = np.isin(pronoun_labels, ['2nd', '3rd'])
        if mask_inlp.sum() < 10:
            return None
        y_nuisance = (pronoun_labels[mask_inlp] == '2nd').astype(int)
        X_inlp = X_scaled[mask_inlp]
    elif inlp_type == '1st_vs_3rd':
        # I vs he/she
        mask_inlp = np.isin(pronoun_labels, ['1st', '3rd'])
        if mask_inlp.sum() < 10:
            return None
        y_nuisance = (pronoun_labels[mask_inlp] == '1st').astype(int)
        X_inlp = X_scaled[mask_inlp]
    elif inlp_type == 'has_pronoun':
        # has vs no pronoun
        y_nuisance = (pronoun_labels != 'none').astype(int)
        X_inlp = X_scaled
    else:
        return None
    
    if len(np.unique(y_nuisance)) < 2:
        return None
    
    # Apply INLP and track
    X_proj, auc_history, acc_history = apply_inlp_and_track(X_inlp, y_nuisance)
    
    return {
        'model': model_name,
        'run_type': run_type,
        'layer': layer,
        'comparison': comparison_name,
        'inlp_type': inlp_type,
        'n_samples': len(X_inlp),
        'initial_pronoun_auc': auc_history[0] if len(auc_history) > 0 else np.nan,
        'initial_pronoun_acc': acc_history[0] if len(acc_history) > 0 else np.nan,
        'final_pronoun_auc': auc_history[-1] if len(auc_history) > 0 else np.nan,
        'final_pronoun_acc': acc_history[-1] if len(acc_history) > 0 else np.nan,
        'n_iterations': len(auc_history),
        'auc_drop': (auc_history[0] - auc_history[-1]) if len(auc_history) > 0 else np.nan
    }


def main():
    """Main execution."""
    print("="*80)
    print("VERIFYING INLP EFFECTIVENESS: PRONOUN PREDICTION AFTER INLP")
    print("="*80)
    
    base_path = Path("/Users/mattduffy/self-referent-test")
    output_dir = base_path / "linear_probe_multiclass" / "pca_analysis"
    
    models = [
        ("Llama", base_path / "results_llama_activation"),
        ("Mistral", base_path / "results_activation_analysis"),
        ("Qwen", base_path / "results_qwen_activation")
    ]
    
    # Test key comparisons with their INLP types
    comparisons = [
        ('self_vs_confounder', 'self_referent', 'confounder', '1st_vs_2nd'),
        ('self_vs_third', 'self_referent', 'third_person', '2nd_vs_3rd'),
        ('self_vs_neutral', 'self_referent', 'neutral', 'has_pronoun'),
    ]
    
    all_results = []
    
    print("\nTesting INLP effectiveness...")
    print("-" * 80)
    
    for model_name, model_dir in models:
        for run_type in ['latest_run']:  # Focus on instruct
            run_label = 'instruct'
            print(f"\n{model_name} {run_label.upper()}:")
            
            for comp_name, cat1, cat2, inlp_type in comparisons:
                try:
                    result = test_comparison(
                        model_name, model_dir, run_type, 
                        15, comp_name, cat1, cat2, inlp_type
                    )
                    if result:
                        all_results.append(result)
                        print(f"  {comp_name:20} ({inlp_type:12}): " 
                              f"Initial={result['initial_pronoun_auc']:.3f}, "
                              f"Final={result['final_pronoun_auc']:.3f}, "
                              f"Drop={result['auc_drop']:.3f}")
                except Exception as e:
                    print(f"  {comp_name:20}: ERROR - {e}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    output_path = output_dir / 'inlp_effectiveness.csv'
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("\nINLP is effective if:")
    print("  • Initial pronoun AUC is high (>0.8) - pronouns are encoded")
    print("  • Final pronoun AUC is low (~0.5) - INLP removed the signal")
    print("  • AUC drop is large (>0.3)")
    print("\nIf final AUC is still high, INLP failed to remove pronoun information.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

