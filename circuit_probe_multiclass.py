#!/usr/bin/env python3
"""
Multi-class circuit probe analysis with robust CV and INLP.

Tests three key hypotheses:
1. Can we classify self vs other vs neutral (3-way)?
2. Self vs Other: Is it truly self-specific, not just person detection?
3. Does the signal survive INLP for different comparisons?

Key improvements:
- GroupKFold by prompts (no template leakage)
- INLP projection (remove lexical cues)
- Multi-class and pairwise comparisons
- Comprehensive analysis across layers and token positions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.linalg import qr
import json
import warnings
warnings.filterwarnings('ignore')


def load_activations(model_dir, run_type, layer, token_position='last'):
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
    """
    Token-level pronoun detection based on actual prompt content.
    Returns: array of pronoun types ('2nd', '1st', '3rd', 'none')
    
    Note: 2nd person (you) = self_referent (about model)
          1st person (I) = confounder (about user)  
          3rd person (he/she) = third_person (about someone else)
    """
    labels = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        # Check for 2nd person (you) - self referent
        if 'you' in prompt_lower or 'your' in prompt_lower or 'yourself' in prompt_lower:
            labels.append('2nd')
        # Check for 1st person (I/me/my) - confounder
        elif ' i ' in prompt_lower or ' my ' in prompt_lower or ' me ' in prompt_lower or 'am i' in prompt_lower or 'do i' in prompt_lower or 'should i' in prompt_lower:
            labels.append('1st')
        # Check for 3rd person (he/she) - third person
        elif ' he ' in prompt_lower or ' she ' in prompt_lower or ' his ' in prompt_lower or ' her ' in prompt_lower or 'does he' in prompt_lower or 'does she' in prompt_lower or 'is he' in prompt_lower or 'is she' in prompt_lower or 'should he' in prompt_lower or 'should she' in prompt_lower:
            labels.append('3rd')
        else:
            labels.append('none')
    return np.array(labels)


def inlp_projection(X_train, X_test, y_nuisance_train, y_nuisance_test, n_iterations=10):
    """
    Iterative Null-space Projection (INLP).
    Returns: (X_train_proj, X_test_proj, auc_history)
    """
    # Check if we have at least 2 classes
    if len(np.unique(y_nuisance_train)) < 2:
        # Can't do INLP with only one class, return original data
        return X_train.copy(), X_test.copy(), []
    
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
            break
        
        direction = clf.coef_[0]
        d = direction / np.linalg.norm(direction)
        
        X_train_current = X_train_current - np.outer(X_train_current @ d, d)
        X_test_current = X_test_current - np.outer(X_test_current @ d, d)
    
    return X_train_current, X_test_current, auc_history


def bootstrap_ci(y_true, y_pred_proba, metric_fn=roc_auc_score, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval."""
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_pred_proba[indices])
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return 0.5, 0.5, 0.5
    
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    mean = np.mean(scores)
    
    return mean, lower, upper


def probe_binary_with_cv(X, y, groups, cv=5, method='baseline', inlp_type='has_pronoun'):
    """
    Binary probe with GroupKFold CV and INLP projection.
    
    Args:
        method: 'baseline' or 'inlp'
        inlp_type: Which pronoun distinction to remove
            - '1st_vs_2nd': Remove I vs you distinction
            - '2nd_vs_3rd': Remove you vs he/she distinction  
            - '1st_vs_3rd': Remove I vs he/she distinction
            - 'has_pronoun': Remove has-pronoun vs no-pronoun
    """
    gkf = GroupKFold(n_splits=cv)
    
    all_y_true = []
    all_y_pred_proba = []
    fold_accs = []
    fold_aucs = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Apply INLP if requested
        if method == 'inlp':
            pronoun_labels_train = extract_pronoun_labels(groups_train)
            pronoun_labels_test = extract_pronoun_labels(groups[test_idx])
            
            # Remove pronoun type distinctions iteratively
            # First remove 1st/2nd vs 3rd (self vs third distinction)
            y_self_vs_third_train = np.array([1 if p in ['1st', '2nd'] else 0 if p == '3rd' else -1 
                                             for p in pronoun_labels_train])
            y_self_vs_third_test = np.array([1 if p in ['1st', '2nd'] else 0 if p == '3rd' else -1 
                                            for p in pronoun_labels_test])
            
            # Only apply if we have both classes
            mask_train = y_self_vs_third_train != -1
            mask_test = y_self_vs_third_test != -1
            
            if mask_train.sum() > 10 and len(np.unique(y_self_vs_third_train[mask_train])) == 2:
                # Project out self vs third pronoun distinction
                X_train_subset = X_train[mask_train]
                X_test_subset = X_test[mask_test] if mask_test.sum() > 0 else X_test[:1]  # dummy
                
                X_train_proj, X_test_proj, _ = inlp_projection(
                    X_train_subset, X_test_subset,
                    y_self_vs_third_train[mask_train], 
                    y_self_vs_third_test[mask_test] if mask_test.sum() > 0 else np.array([0]),
                    n_iterations=10
                )
                
                X_train[mask_train] = X_train_proj
                if mask_test.sum() > 0:
                    X_test[mask_test] = X_test_proj
            
            # Then remove has-pronoun vs no-pronoun
            y_has_pronoun_train = (pronoun_labels_train != 'none').astype(int)
            y_has_pronoun_test = (pronoun_labels_test != 'none').astype(int)
            
            X_train, X_test, _ = inlp_projection(X_train, X_test, 
                                                  y_has_pronoun_train, y_has_pronoun_test,
                                                  n_iterations=10)
        
        # Train probe
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
        
        fold_accs.append(acc)
        fold_aucs.append(auc)
        all_y_true.extend(y_test)
        all_y_pred_proba.extend(y_pred_proba)
    
    # Bootstrap CI on pooled predictions
    all_y_true = np.array(all_y_true)
    all_y_pred_proba = np.array(all_y_pred_proba)
    
    if len(np.unique(all_y_true)) == 2:
        auc_mean, auc_lower, auc_upper = bootstrap_ci(all_y_true, all_y_pred_proba)
    else:
        auc_mean, auc_lower, auc_upper = 0.5, 0.5, 0.5
    
    return {
        'accuracy_mean': np.mean(fold_accs),
        'accuracy_std': np.std(fold_accs),
        'auc_mean': auc_mean,
        'auc_lower': auc_lower,
        'auc_upper': auc_upper
    }


def probe_multiclass_with_cv(X, y, groups, cv=5, method='baseline'):
    """
    Multi-class probe (self vs other vs neutral).
    """
    gkf = GroupKFold(n_splits=cv)
    
    all_y_true = []
    all_y_pred = []
    fold_accs = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Apply comparison-specific INLP if requested
        if method == 'inlp':
            pronoun_labels_train = extract_pronoun_labels(groups_train)
            pronoun_labels_test = extract_pronoun_labels(groups[test_idx])
            
            if inlp_type == '1st_vs_2nd':
                # Remove 1st vs 2nd person distinction (I vs you)
                mask_train = np.isin(pronoun_labels_train, ['1st', '2nd'])
                mask_test = np.isin(pronoun_labels_test, ['1st', '2nd'])
                
                if mask_train.sum() > 10 and len(np.unique(pronoun_labels_train[mask_train])) == 2:
                    y_train_subset = (pronoun_labels_train[mask_train] == '2nd').astype(int)
                    y_test_subset = (pronoun_labels_test[mask_test] == '2nd').astype(int) if mask_test.sum() > 0 else np.array([0])
                    
                    X_train_proj, X_test_proj, _ = inlp_projection(
                        X_train[mask_train], 
                        X_test[mask_test] if mask_test.sum() > 0 else X_test[:1],
                        y_train_subset, y_test_subset,
                        n_iterations=10
                    )
                    X_train[mask_train] = X_train_proj
                    if mask_test.sum() > 0:
                        X_test[mask_test] = X_test_proj
            
            elif inlp_type == '2nd_vs_3rd':
                # Remove 2nd vs 3rd person distinction (you vs he/she)
                mask_train = np.isin(pronoun_labels_train, ['2nd', '3rd'])
                mask_test = np.isin(pronoun_labels_test, ['2nd', '3rd'])
                
                if mask_train.sum() > 10 and len(np.unique(pronoun_labels_train[mask_train])) == 2:
                    y_train_subset = (pronoun_labels_train[mask_train] == '2nd').astype(int)
                    y_test_subset = (pronoun_labels_test[mask_test] == '2nd').astype(int) if mask_test.sum() > 0 else np.array([0])
                    
                    X_train_proj, X_test_proj, _ = inlp_projection(
                        X_train[mask_train],
                        X_test[mask_test] if mask_test.sum() > 0 else X_test[:1],
                        y_train_subset, y_test_subset,
                        n_iterations=10
                    )
                    X_train[mask_train] = X_train_proj
                    if mask_test.sum() > 0:
                        X_test[mask_test] = X_test_proj
            
            elif inlp_type == '1st_vs_3rd':
                # Remove 1st vs 3rd person distinction (I vs he/she)
                mask_train = np.isin(pronoun_labels_train, ['1st', '3rd'])
                mask_test = np.isin(pronoun_labels_test, ['1st', '3rd'])
                
                if mask_train.sum() > 10 and len(np.unique(pronoun_labels_train[mask_train])) == 2:
                    y_train_subset = (pronoun_labels_train[mask_train] == '1st').astype(int)
                    y_test_subset = (pronoun_labels_test[mask_test] == '1st').astype(int) if mask_test.sum() > 0 else np.array([0])
                    
                    X_train_proj, X_test_proj, _ = inlp_projection(
                        X_train[mask_train],
                        X_test[mask_test] if mask_test.sum() > 0 else X_test[:1],
                        y_train_subset, y_test_subset,
                        n_iterations=10
                    )
                    X_train[mask_train] = X_train_proj
                    if mask_test.sum() > 0:
                        X_test[mask_test] = X_test_proj
            
            elif inlp_type == 'has_pronoun':
                # Remove has-pronoun vs no-pronoun distinction
                y_has_pronoun_train = (pronoun_labels_train != 'none').astype(int)
                y_has_pronoun_test = (pronoun_labels_test != 'none').astype(int)
                
                X_train, X_test, _ = inlp_projection(X_train, X_test, 
                                                      y_has_pronoun_train, y_has_pronoun_test,
                                                      n_iterations=10)
        
        # Train multi-class probe
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1, multi_class='multinomial')
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Confusion matrix
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    return {
        'accuracy_mean': np.mean(fold_accs),
        'accuracy_std': np.std(fold_accs),
        'confusion_matrix': cm,
        'y_true': all_y_true,
        'y_pred': all_y_pred
    }


def analyze_pairwise_comparisons(X, categories, prompts, layer, token_pos):
    """
    Analyze all pairwise comparisons with baseline and comparison-specific INLP.
    
    INLP types by comparison:
    - self_vs_third: Remove 2nd vs 3rd person (you vs he/she)
    - self_vs_confounder: Remove 2nd vs 1st person (you vs I)
    - self_vs_neutral: Remove has-pronoun vs no-pronoun
    - third_vs_neutral: Remove has-pronoun vs no-pronoun
    - third_vs_confounder: Remove 3rd vs 1st person (he/she vs I)
    - neutral_vs_confounder: Remove has-pronoun vs no-pronoun
    """
    comparisons = {
        'self_vs_third': ('self_referent', 'third_person', '2nd_vs_3rd'),
        'self_vs_neutral': ('self_referent', 'neutral', 'has_pronoun'),
        'self_vs_confounder': ('self_referent', 'confounder', '1st_vs_2nd'),
        'third_vs_neutral': ('third_person', 'neutral', 'has_pronoun'),
        'third_vs_confounder': ('third_person', 'confounder', '1st_vs_3rd'),
        'neutral_vs_confounder': ('neutral', 'confounder', 'has_pronoun')
    }
    
    results = {}
    
    for comp_name, (cat1, cat2, inlp_type) in comparisons.items():
        # Filter data
        mask = (categories == cat1) | (categories == cat2)
        X_pair = X[mask]
        y_pair = (categories[mask] == cat1).astype(int)
        groups_pair = prompts[mask]
        
        # Remove NaN rows
        nan_mask = ~np.isnan(X_pair).any(axis=1)
        X_pair = X_pair[nan_mask]
        y_pair = y_pair[nan_mask]
        groups_pair = groups_pair[nan_mask]
        
        if len(np.unique(y_pair)) < 2 or len(y_pair) < 10:
            continue
        
        try:
            # Baseline
            res_baseline = probe_binary_with_cv(X_pair, y_pair, groups_pair, 
                                               cv=5, method='baseline')
            
            # INLP with comparison-specific confound removal
            res_inlp = probe_binary_with_cv(X_pair, y_pair, groups_pair,
                                           cv=5, method='inlp', inlp_type=inlp_type)
        except Exception as e:
            print(f"      Error in {comp_name}: {e}")
            continue
        
        results[comp_name] = {
            'layer': layer,
            'token_pos': token_pos,
            'comparison': comp_name,
            'n_samples': len(y_pair),
            'baseline_auc': res_baseline['auc_mean'],
            'baseline_auc_ci': (res_baseline['auc_lower'], res_baseline['auc_upper']),
            'inlp_auc': res_inlp['auc_mean'],
            'inlp_auc_ci': (res_inlp['auc_lower'], res_inlp['auc_upper']),
            'auc_drop': res_baseline['auc_mean'] - res_inlp['auc_mean']
        }
    
    return results


def analyze_multiclass(X, categories, prompts, layer, token_pos):
    """
    Analyze 4-way classification: self vs third_person vs neutral vs confounder.
    """
    # Create numeric labels
    label_map = {'self_referent': 0, 'third_person': 1, 'neutral': 2, 'confounder': 3}
    y_multi = np.array([label_map[cat] for cat in categories])
    
    # Check if we have at least 2 classes
    if len(np.unique(y_multi)) < 2:
        return None
    
    try:
        # Baseline only (no INLP for 4-way classification - no clear single confound)
        res_baseline = probe_multiclass_with_cv(X, y_multi, prompts,
                                               cv=5, method='baseline')
        
        return {
            'layer': layer,
            'token_pos': token_pos,
            'n_samples': len(y_multi),
            'baseline_acc': res_baseline['accuracy_mean'],
            'baseline_cm': res_baseline['confusion_matrix']
        }
    except Exception as e:
        print(f"      Error in multiclass: {e}")
        return None


def analyze_layer(model_name, model_dir, run_type, layer, token_position='last'):
    """Comprehensive analysis for a single layer."""
    X, categories, prompts = load_activations(model_dir, run_type, layer, token_position)
    
    # Check for and remove NaN values
    nan_mask = ~np.isnan(X).any(axis=1)
    if not nan_mask.all():
        n_nan = (~nan_mask).sum()
        print(f"    Layer {layer}, Token: {token_position} - Removing {n_nan} samples with NaN")
        X = X[nan_mask]
        categories = categories[nan_mask]
        prompts = prompts[nan_mask]
    
    print(f"    Layer {layer}, Token: {token_position}")
    
    # Pairwise comparisons
    pairwise_results = analyze_pairwise_comparisons(X, categories, prompts, layer, token_position)
    
    # Multi-class
    multiclass_result = analyze_multiclass(X, categories, prompts, layer, token_position)
    
    return {
        'pairwise': pairwise_results,
        'multiclass': multiclass_result
    }


def analyze_model(model_name, model_dir, n_layers, layers_to_test, token_positions, output_dir):
    """Analyze model across layers and positions."""
    print(f"\n{'='*100}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*100}")
    
    all_results = {}
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*100}")
        
        results_list = []
        
        for layer in layers_to_test:
            if layer >= n_layers:
                continue
            
            for token_pos in token_positions:
                try:
                    result = analyze_layer(model_name, model_dir, run_type, layer, token_pos)
                    results_list.append({
                        'layer': layer,
                        'token_pos': token_pos,
                        **result
                    })
                except Exception as e:
                    print(f"      ERROR: {e}")
        
        all_results[run_label.lower()] = results_list
        
        # Print summary
        print_summary(results_list, run_label)
    
    # Save results
    save_results(all_results, model_name, output_dir)
    
    return all_results


def print_summary(results_list, run_label):
    """Print formatted summary of results."""
    print(f"\n{'-'*100}")
    print(f"PAIRWISE COMPARISONS:")
    print(f"{'-'*100}")
    print(f"{'Layer':<8} {'TokPos':<12} {'Comparison':<18} {'Base AUC':<25} {'INLP AUC':<25} {'Drop':<10}")
    print(f"{'-'*100}")
    
    for result in results_list:
        for comp_name, comp_data in result['pairwise'].items():
            base_str = f"{comp_data['baseline_auc']:.3f} [{comp_data['baseline_auc_ci'][0]:.3f}-{comp_data['baseline_auc_ci'][1]:.3f}]"
            inlp_str = f"{comp_data['inlp_auc']:.3f} [{comp_data['inlp_auc_ci'][0]:.3f}-{comp_data['inlp_auc_ci'][1]:.3f}]"
            drop = comp_data['auc_drop']
            
            print(f"{result['layer']:<8} {result['token_pos']:<12} {comp_name:<18} {base_str:<25} {inlp_str:<25} {drop:<10.3f}")
    
    print(f"\n{'-'*100}")
    print(f"MULTI-CLASS (Self vs Third vs Neutral vs Confounder - Baseline Only):")
    print(f"{'-'*100}")
    print(f"{'Layer':<8} {'TokPos':<12} {'Accuracy':<15}")
    print(f"{'-'*100}")
    
    for result in results_list:
        mc = result['multiclass']
        if mc is not None:
            print(f"{result['layer']:<8} {result['token_pos']:<12} {mc['baseline_acc']:.3f} ({mc['baseline_acc']*100:.1f}%)")


def save_results(all_results, model_name, output_dir):
    """Save results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pairwise results to CSV
    pairwise_rows = []
    for run_type, results_list in all_results.items():
        for result in results_list:
            for comp_name, comp_data in result['pairwise'].items():
                pairwise_rows.append({
                    'model': model_name,
                    'version': run_type,
                    'layer': result['layer'],
                    'token_pos': result['token_pos'],
                    'comparison': comp_name,
                    'n_samples': comp_data['n_samples'],
                    'baseline_auc': comp_data['baseline_auc'],
                    'baseline_auc_lower': comp_data['baseline_auc_ci'][0],
                    'baseline_auc_upper': comp_data['baseline_auc_ci'][1],
                    'inlp_auc': comp_data['inlp_auc'],
                    'inlp_auc_lower': comp_data['inlp_auc_ci'][0],
                    'inlp_auc_upper': comp_data['inlp_auc_ci'][1],
                    'auc_drop': comp_data['auc_drop']
                })
    
    df_pairwise = pd.DataFrame(pairwise_rows)
    df_pairwise.to_csv(output_path / f"{model_name.lower()}_pairwise_results.csv", index=False)
    
    # Multi-class results to CSV (baseline only, no INLP)
    multiclass_rows = []
    for run_type, results_list in all_results.items():
        for result in results_list:
            mc = result['multiclass']
            if mc is not None:
                multiclass_rows.append({
                    'model': model_name,
                    'version': run_type,
                    'layer': result['layer'],
                    'token_pos': result['token_pos'],
                    'n_samples': mc['n_samples'],
                    'baseline_acc': mc['baseline_acc']
                })
    
    df_multiclass = pd.DataFrame(multiclass_rows)
    df_multiclass.to_csv(output_path / f"{model_name.lower()}_multiclass_results.csv", index=False)
    
    # Save full results as JSON (without numpy arrays)
    results_serializable = {}
    for run_type, results_list in all_results.items():
        results_serializable[run_type] = []
        for result in results_list:
            result_copy = {
                'layer': result['layer'],
                'token_pos': result['token_pos'],
                'pairwise': result['pairwise'],
                'multiclass': None if result['multiclass'] is None else {
                    'layer': result['multiclass']['layer'],
                    'token_pos': result['multiclass']['token_pos'],
                    'n_samples': int(result['multiclass']['n_samples']),
                    'baseline_acc': float(result['multiclass']['baseline_acc'])
                }
            }
            results_serializable[run_type].append(result_copy)
    
    with open(output_path / f"{model_name.lower()}_full_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}/")


def create_visualizations(output_dir):
    """Create visualizations from all model results."""
    output_path = Path(output_dir)
    
    # Load all pairwise results
    pairwise_dfs = []
    for csv_file in output_path.glob("*_pairwise_results.csv"):
        df = pd.read_csv(csv_file)
        if len(df) > 0:  # Only add if not empty
            pairwise_dfs.append(df)
    
    if not pairwise_dfs:
        print("No results found for visualization")
        return
    
    df_all = pd.concat(pairwise_dfs, ignore_index=True)
    
    if len(df_all) == 0:
        print("No data in results files")
        return
    
    # Figure 1: Comparison of AUC drops for each comparison type
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    axes = axes.flatten()
    
    comparisons = ['self_vs_third', 'self_vs_neutral', 'self_vs_confounder',
                   'third_vs_neutral', 'third_vs_confounder', 'neutral_vs_confounder']
    comp_labels = ['Self vs Third', 'Self vs Neutral', 'Self vs Confounder',
                   'Third vs Neutral', 'Third vs Confounder', 'Neutral vs Confounder']
    
    for idx, (comp, label) in enumerate(zip(comparisons, comp_labels)):
        ax = axes[idx]
        
        data = df_all[df_all['comparison'] == comp]
        
        models = data['model'].unique()
        versions = ['base', 'instruct']
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, version in enumerate(versions):
            version_data = data[data['version'] == version]
            
            baseline_aucs = []
            inlp_aucs = []
            
            for model in models:
                model_data = version_data[version_data['model'] == model]
                if not model_data.empty:
                    baseline_aucs.append(model_data['baseline_auc'].mean())
                    inlp_aucs.append(model_data['inlp_auc'].mean())
                else:
                    baseline_aucs.append(0)
                    inlp_aucs.append(0)
            
            offset = width * (i - 0.5)
            
            # Plot baseline and INLP side by side
            ax.bar(x + offset - width/4, baseline_aucs, width/2, 
                   label=f'{version.title()} - Baseline', alpha=0.7)
            ax.bar(x + offset + width/4, inlp_aucs, width/2,
                   label=f'{version.title()} - INLP', alpha=0.7)
        
        ax.set_xlabel('Model', fontsize=12, weight='bold')
        if idx == 0:
            ax.set_ylabel('AUC', fontsize=12, weight='bold')
        ax.set_title(label, fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure_pairwise_comparisons.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'figure_pairwise_comparisons.png'}")
    
    # Figure 2: AUC drop magnitude (baseline - INLP)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = df_all['model'].unique()
    versions = df_all['version'].unique()
    comparisons_unique = df_all['comparison'].unique()
    
    x = np.arange(len(comparisons_unique))
    width = 0.12
    
    colors = {'Llama': '#2E86AB', 'Mistral': '#E63946', 'Qwen': '#06A77D'}
    
    for m_idx, model in enumerate(models):
        for v_idx, version in enumerate(['base', 'instruct']):
            data = df_all[(df_all['model'] == model) & (df_all['version'] == version)]
            
            drops = []
            for comp in comparisons_unique:
                comp_data = data[data['comparison'] == comp]
                if not comp_data.empty:
                    drops.append(comp_data['auc_drop'].mean())
                else:
                    drops.append(0)
            
            offset = width * (m_idx * 2 + v_idx - len(models))
            
            alpha = 0.6 if version == 'base' else 0.9
            ax.bar(x + offset, drops, width,
                   label=f'{model} {version.title()}',
                   color=colors.get(model, '#666666'),
                   alpha=alpha)
    
    ax.set_xlabel('Comparison Type', fontsize=13, weight='bold')
    ax.set_ylabel('AUC Drop (Baseline - INLP)', fontsize=13, weight='bold')
    ax.set_title('INLP Impact: AUC Drop by Comparison Type\n(Larger drop = more lexical/pronoun-driven)', 
                fontsize=15, weight='bold', pad=20)
    ax.set_xticks(x)
    comp_labels_short = ['Self vs\nThird', 'Self vs\nNeutral', 'Self vs\nConfounder',
                        'Third vs\nNeutral', 'Third vs\nConfounder', 'Neutral vs\nConfounder']
    ax.set_xticklabels(comp_labels_short, fontsize=10)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure_auc_drops.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'figure_auc_drops.png'}")
    
    plt.close('all')


def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    output_dir = base_path / "linear_probe_multiclass"
    
    # Test key layers
    layers_to_test = [10, 15, 20, 25]
    token_positions = ['last', 'bos', 'first_content', 'mid']
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    print("="*100)
    print("MULTI-CLASS CIRCUIT PROBE ANALYSIS")
    print("="*100)
    print("\nKey Questions:")
    print("  1. Can we distinguish self vs third_person vs neutral vs confounder (4-way)?")
    print("  2. Self vs Third: Is it truly self-specific, not just person detection?")
    print("  3. Does the signal survive INLP for different comparisons?")
    print("  4. Can we separate confounders from meaningful content?")
    print("\nMethodology:")
    print("  ✓ GroupKFold CV (no template leakage)")
    print("  ✓ INLP projection (remove lexical cues)")
    print("  ✓ Bootstrap 95% CIs")
    print("  ✓ Multi-class (4-way) and pairwise comparisons")
    print("="*100)
    
    for model_name, model_dir, n_layers in models:
        analyze_model(model_name, model_dir, n_layers, layers_to_test, 
                     token_positions, output_dir)
    
    print("\n" + "="*100)
    print("Creating visualizations...")
    create_visualizations(output_dir)
    
    print("\n" + "="*100)
    print("INTERPRETATION GUIDE:")
    print("="*100)
    print("\nPairwise Comparisons:")
    print("  • Self vs Third: High INLP AUC → genuine self-referent circuit (not just person)")
    print("  • Self vs Neutral: High INLP AUC → self-referent signal beyond pronouns")
    print("  • Self vs Confounder: Tests if self signal is robust to ambiguous stimuli")
    print("  • Third vs Neutral: Control (person detection baseline)")
    print("  • Third vs Confounder: Tests third-person robustness")
    print("  • Neutral vs Confounder: Tests neutral baseline discrimination")
    print("\nAUC Drop Analysis:")
    print("  • Large drop (>0.2): Signal is primarily lexical (pronoun-driven)")
    print("  • Small drop (<0.1): Non-lexical circuit persists after INLP")
    print("  • Medium drop: Mixed lexical + non-lexical signal")
    print("\nCritical Tests:")
    print("  1. If 'Self vs Third' maintains high INLP AUC → TRUE self-referent circuit")
    print("  2. If 'Self vs Third' drops to chance → Just pronoun detection, not self-specific")
    print("  3. Confounder comparisons → Tests robustness to ambiguous/edge cases")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

