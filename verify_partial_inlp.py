#!/usr/bin/env python3
"""
Test partial INLP: Remove only 50% of pronoun signal.
Shows if semantic signal depends on residual pronoun information.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GroupKFold


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


def apply_partial_inlp(X, y_nuisance, target_auc=0.75, max_iterations=10):
    """
    Apply INLP until pronoun AUC reaches target_auc (e.g., 0.75 for 50% removal).
    target_auc=1.0 means no removal, 0.5 means complete removal.
    """
    X_current = X.copy()
    auc_history = []
    
    for iteration in range(max_iterations):
        # Split for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_current, y_nuisance, test_size=0.3, random_state=42
        )
        
        # Train pronoun classifier
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        auc_history.append(auc)
        
        # Stop if reached target
        if auc <= target_auc:
            break
        
        # Project out on full data
        clf_full = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf_full.fit(X_current, y_nuisance)
        
        direction = clf_full.coef_[0]
        d = direction / np.linalg.norm(direction)
        X_current = X_current - np.outer(X_current @ d, d)
    
    return X_current, auc_history


def test_self_vs_confounder_with_partial_inlp(model_name, model_dir, run_type, layer, target_auc):
    """
    Test self vs confounder classification after partial INLP.
    """
    X, categories, prompts = load_activations(model_dir, run_type, layer, 'mid')
    
    # Filter to self vs confounder
    mask = (categories == 'self_referent') | (categories == 'confounder')
    X_filtered = X[mask]
    categories_filtered = categories[mask]
    prompts_filtered = prompts[mask]
    
    # Remove NaN
    nan_mask = ~np.isnan(X_filtered).any(axis=1)
    X_filtered = X_filtered[nan_mask]
    categories_filtered = categories_filtered[nan_mask]
    prompts_filtered = prompts_filtered[nan_mask]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    # Get labels
    y_semantic = (categories_filtered == 'self_referent').astype(int)
    pronoun_labels = extract_pronoun_labels(prompts_filtered)
    
    # INLP: Remove 1st vs 2nd person
    mask_inlp = np.isin(pronoun_labels, ['1st', '2nd'])
    if mask_inlp.sum() < 10:
        return None
    
    y_pronouns = (pronoun_labels[mask_inlp] == '2nd').astype(int)
    
    # Apply partial INLP
    X_inlp_subset, auc_history = apply_partial_inlp(
        X_scaled[mask_inlp], y_pronouns, target_auc=target_auc
    )
    
    # Apply to all data (project same directions)
    X_inlp_full = X_scaled.copy()
    X_inlp_full[mask_inlp] = X_inlp_subset
    
    # Test semantic classification before INLP
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    scores_baseline = cross_val_score(clf_baseline, X_scaled, y_semantic, cv=5, scoring='roc_auc')
    baseline_auc = scores_baseline.mean()
    
    # Test semantic classification after partial INLP
    clf_inlp = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    scores_inlp = cross_val_score(clf_inlp, X_inlp_full, y_semantic, cv=5, scoring='roc_auc')
    inlp_auc = scores_inlp.mean()
    
    # Test pronoun classification after partial INLP (on subset with pronouns)
    clf_pronoun = LogisticRegression(max_iter=1000, random_state=42)
    scores_pronoun = cross_val_score(clf_pronoun, X_inlp_subset, y_pronouns, cv=5, scoring='roc_auc')
    pronoun_auc_after = scores_pronoun.mean()
    
    return {
        'model': model_name,
        'run_type': run_type,
        'layer': layer,
        'target_pronoun_auc': target_auc,
        'initial_pronoun_auc': auc_history[0] if len(auc_history) > 0 else np.nan,
        'final_pronoun_auc': pronoun_auc_after,
        'n_inlp_iterations': len(auc_history),
        'baseline_semantic_auc': baseline_auc,
        'partial_inlp_semantic_auc': inlp_auc,
        'semantic_auc_drop': baseline_auc - inlp_auc
    }


def main():
    """Main execution."""
    print("="*80)
    print("PARTIAL INLP TEST: SELF VS CONFOUNDER WITH 50% PRONOUN REMOVAL")
    print("="*80)
    
    base_path = Path("/Users/mattduffy/self-referent-test")
    output_dir = base_path / "linear_probe_multiclass" / "pca_analysis"
    
    models = [
        ("Llama", base_path / "results_llama_activation"),
        ("Mistral", base_path / "results_activation_analysis"),
        ("Qwen", base_path / "results_qwen_activation")
    ]
    
    # Test different levels of pronoun removal
    target_aucs = [
        (1.0, "0% removal (baseline)"),
        (0.75, "~50% removal"),
        (0.65, "~70% removal"),
        (0.55, "~90% removal"),
        (0.5, "100% removal (full INLP)")
    ]
    
    all_results = []
    
    print("\nTesting partial INLP effectiveness...")
    print("-" * 80)
    
    for target_auc, description in target_aucs:
        print(f"\n{description.upper()} (target pronoun AUC = {target_auc}):")
        print("-" * 80)
        
        for model_name, model_dir in models:
            try:
                result = test_self_vs_confounder_with_partial_inlp(
                    model_name, model_dir, 'latest_run', 15, target_auc
                )
                if result:
                    all_results.append(result)
                    print(f"{model_name:8} | Pronoun AUC: {result['initial_pronoun_auc']:.3f}→{result['final_pronoun_auc']:.3f} | "
                          f"Semantic AUC: {result['baseline_semantic_auc']:.3f}→{result['partial_inlp_semantic_auc']:.3f} "
                          f"(drop: {result['semantic_auc_drop']:.3f})")
            except Exception as e:
                print(f"{model_name:8} | ERROR: {e}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    output_path = output_dir / 'partial_inlp_results.csv'
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("\nIf semantic AUC drops linearly with pronoun removal:")
    print("  → Semantic signal DEPENDS on residual pronoun information")
    print("\nIf semantic AUC stays high even with partial pronoun removal:")
    print("  → Semantic signal is INDEPENDENT and robust")
    print("\nIf semantic AUC only drops with full (100%) pronoun removal:")
    print("  → Signal might be using subtle residual pronoun traces")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

