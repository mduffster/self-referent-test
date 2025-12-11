#!/usr/bin/env python3
"""
Debug script to trace INLP application in multiclass probe.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def load_activations(model_dir, run_type, layer, token_position='mid'):
    """Load MLP post activations."""
    npz_path = model_dir / run_type / f"raw_blocks_{layer}_mlp_post.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    activations = []
    for act in data['activations']:
        mid_idx = act.shape[1] // 2
        token = act[0, mid_idx, :]
        activations.append(token)
    
    return np.array(activations), data['categories'], data['prompts']


def extract_pronoun_labels(prompts):
    """Extract pronoun labels."""
    labels = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        if 'you' in prompt_lower or 'your' in prompt_lower:
            labels.append('2nd')
        elif ' i ' in prompt_lower or ' my ' in prompt_lower or 'am i' in prompt_lower:
            labels.append('1st')
        else:
            labels.append('none')
    return np.array(labels)


def inlp_projection(X_train, X_test, y_nuisance_train, y_nuisance_test, n_iterations=10):
    """INLP with debugging."""
    print(f"    INLP starting:")
    print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"      Train classes: {np.unique(y_nuisance_train, return_counts=True)}")
    
    X_train_current = X_train.copy()
    X_test_current = X_test.copy()
    auc_history = []
    
    for iteration in range(n_iterations):
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1, class_weight='balanced')
        clf.fit(X_train_current, y_nuisance_train)
        
        y_pred_proba = clf.predict_proba(X_test_current)[:, 1]
        try:
            auc = roc_auc_score(y_nuisance_test, y_pred_proba)
        except:
            auc = 0.5
        auc_history.append(auc)
        
        if auc < 0.55:
            print(f"      Stopped at iteration {iteration}, AUC={auc:.3f}")
            break
        
        direction = clf.coef_[0]
        d = direction / np.linalg.norm(direction)
        X_train_current = X_train_current - np.outer(X_train_current @ d, d)
        X_test_current = X_test_current - np.outer(X_test_current @ d, d)
    
    print(f"      Final AUC: {auc_history[-1]:.3f} (after {len(auc_history)} iterations)")
    return X_train_current, X_test_current, auc_history


def debug_one_fold():
    """Debug a single CV fold for self_vs_confounder."""
    
    print("="*80)
    print("DEBUGGING MULTICLASS PROBE INLP FOR SELF VS CONFOUNDER")
    print("="*80)
    
    # Load data
    base_path = Path("/Users/mattduffy/self-referent-test")
    model_dir = base_path / "results_llama_activation"
    
    X, categories, prompts = load_activations(model_dir, 'latest_run', 15, 'mid')
    
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
    
    # Labels
    y = (categories_filtered == 'self_referent').astype(int)
    
    print(f"\nTotal samples: {len(X_filtered)}")
    print(f"  Self: {(y == 1).sum()}")
    print(f"  Confounder: {(y == 0).sum()}")
    
    # Setup CV
    gkf = GroupKFold(n_splits=5)
    
    baseline_aucs = []
    inlp_aucs = []
    
    # Run all folds
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_filtered, y, prompts_filtered)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}")
        print('='*80)
        
        X_train_raw, X_test_raw = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = prompts_filtered[train_idx]
        groups_test = prompts_filtered[test_idx]
        
        print(f"\nTrain: {len(X_train_raw)} samples")
        print(f"  Self: {(y_train == 1).sum()}, Confounder: {(y_train == 0).sum()}")
        print(f"Test: {len(X_test_raw)} samples")
        print(f"  Self: {(y_test == 1).sum()}, Confounder: {(y_test == 0).sum()}")
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Test baseline
        print("\n--- BASELINE ---")
        clf_baseline = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf_baseline.fit(X_train, y_train)
        y_pred_baseline = clf_baseline.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_pred_baseline)
        baseline_aucs.append(baseline_auc)
        print(f"  Baseline AUC: {baseline_auc:.3f}")
        
        # Apply INLP
        print("\n--- INLP ---")
        pronoun_labels_train = extract_pronoun_labels(groups_train)
        pronoun_labels_test = extract_pronoun_labels(groups_test)
        
        print(f"  Pronouns in train: {np.unique(pronoun_labels_train, return_counts=True)}")
        print(f"  Pronouns in test: {np.unique(pronoun_labels_test, return_counts=True)}")
        
        # For self_vs_confounder, INLP type is '1st_vs_2nd'
        mask_train = np.isin(pronoun_labels_train, ['1st', '2nd'])
        mask_test = np.isin(pronoun_labels_test, ['1st', '2nd'])
        
        print(f"  Mask train sum: {mask_train.sum()} / {len(mask_train)}")
        print(f"  Mask test sum: {mask_test.sum()} / {len(mask_test)}")
        
        if mask_train.sum() > 10 and len(np.unique(pronoun_labels_train[mask_train])) == 2:
            print(f"  INLP condition MET")
            
            y_train_subset = (pronoun_labels_train[mask_train] == '2nd').astype(int)
            y_test_subset = (pronoun_labels_test[mask_test] == '2nd').astype(int) if mask_test.sum() > 0 else np.array([0])
            
            X_train_proj, X_test_proj, _ = inlp_projection(
                X_train[mask_train], 
                X_test[mask_test] if mask_test.sum() > 0 else X_test[:1],
                y_train_subset, y_test_subset,
                n_iterations=10
            )
            
            # Apply projection back
            X_train_inlp = X_train.copy()
            X_test_inlp = X_test.copy()
            X_train_inlp[mask_train] = X_train_proj
            if mask_test.sum() > 0:
                X_test_inlp[mask_test] = X_test_proj
            
            # Test after INLP
            print("\n  Testing semantic classification after INLP:")
            clf_inlp = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            clf_inlp.fit(X_train_inlp, y_train)
            y_pred_inlp = clf_inlp.predict_proba(X_test_inlp)[:, 1]
            inlp_auc = roc_auc_score(y_test, y_pred_inlp)
            inlp_aucs.append(inlp_auc)
            print(f"    INLP AUC: {inlp_auc:.3f}")
            
            # Test if we can still predict pronouns after INLP
            print("\n  Testing pronoun classification after INLP:")
            clf_pronoun_test = LogisticRegression(max_iter=1000, random_state=42)
            if mask_test.sum() > 5:
                clf_pronoun_test.fit(X_train_inlp[mask_train], y_train_subset)
                y_pred_pronoun = clf_pronoun_test.predict_proba(X_test_inlp[mask_test])[:, 1]
                pronoun_auc_after = roc_auc_score(y_test_subset, y_pred_pronoun)
                print(f"    Pronoun AUC after INLP: {pronoun_auc_after:.3f}")
        else:
            print(f"  INLP condition NOT MET")
        
        # Continue to next fold
        # break
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL FOLDS")
    print("="*80)
    print(f"Baseline AUCs: {baseline_aucs}")
    print(f"  Mean: {np.mean(baseline_aucs):.3f}")
    print(f"INLP AUCs: {inlp_aucs}")
    print(f"  Mean: {np.mean(inlp_aucs):.3f}")
    print(f"\nMulticlass probe reported: 0.949")
    print(f"This debug shows: {np.mean(inlp_aucs):.3f}")
    print(f"Discrepancy: {abs(0.949 - np.mean(inlp_aucs)):.3f}")


if __name__ == "__main__":
    debug_one_fold()

