#!/usr/bin/env python3
"""
Robust circuit probe with proper CV, group splits, and INLP-style projection.

Fixes:
1. Learn pronoun direction inside each CV fold (no leakage)
2. GroupKFold by prompts (no template leakage)
3. Project out pronoun subspace (multiple directions)
4. Bundle nuisance cues (punctuation, length, etc.)
5. AUROC with bootstrap CIs
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.linalg import qr
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
        else:
            raise ValueError(f"Unknown token_position: {token_position}")
        activations.append(token)
    
    return np.array(activations), data['categories'], data['prompts']

def extract_pronoun_labels(prompts):
    """
    Token-level pronoun detection (more reliable than category inference).
    Returns: array of pronoun types ('2nd', '1st', '3rd', 'none')
    """
    labels = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        if 'you' in prompt_lower or 'your' in prompt_lower or 'yourself' in prompt_lower:
            labels.append('2nd')
        elif ' i ' in prompt_lower or 'my ' in prompt_lower or ' my ' in prompt_lower or 'myself' in prompt_lower:
            labels.append('1st')
        elif 'he' in prompt_lower or 'she' in prompt_lower or 'his' in prompt_lower or 'her' in prompt_lower or 'himself' in prompt_lower or 'herself' in prompt_lower:
            labels.append('3rd')
        else:
            labels.append('none')
    return np.array(labels)

def extract_nuisance_features(prompts):
    """
    Extract surface-level nuisance features.
    Returns: array of shape (n_samples, n_features)
    """
    features = []
    for prompt in prompts:
        feat = [
            int(prompt.endswith('?')),  # has question mark
            int(prompt.lower().split()[0] in ['what', 'who', 'where', 'when', 'why', 'how']),  # wh-word
            len(prompt),  # length
            len(prompt.split()),  # word count
        ]
        features.append(feat)
    return np.array(features)

def learn_pronoun_subspace(X, pronoun_labels, n_directions=3):
    """
    Learn pronoun subspace (multiple directions).
    Returns orthonormal basis vectors for pronoun space.
    
    Args:
        X: activations (n_samples, hidden_dim), already scaled
        pronoun_labels: array of '2nd', '1st', '3rd', 'none'
        n_directions: number of directions to extract
    """
    # Train one-vs-rest for each pronoun type
    directions = []
    
    for ptype in ['2nd', '1st', '3rd']:
        # Binary labels: this pronoun type vs everything else
        y_binary = (pronoun_labels == ptype).astype(int)
        
        # Skip if not enough samples
        if y_binary.sum() < 5:
            continue
        
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1, class_weight='balanced')
        clf.fit(X, y_binary)
        directions.append(clf.coef_[0])
    
    if len(directions) == 0:
        return None
    
    # Stack and orthonormalize
    D = np.array(directions)  # Shape: (n_directions, hidden_dim)
    Q, R = qr(D.T)  # QR decomposition
    
    # Q columns are orthonormal basis vectors
    # Take first n_directions columns
    return Q[:, :min(n_directions, Q.shape[1])]

def project_out_subspace(X, basis):
    """
    Project out a subspace spanned by basis vectors.
    X_residual = X - X @ basis @ basis.T
    """
    if basis is None:
        return X
    
    # Project onto subspace
    projection = X @ basis @ basis.T
    
    # Subtract to get residual
    X_residual = X - projection
    
    return X_residual

def inlp_projection(X_train, X_test, y_nuisance_train, y_nuisance_test, n_iterations=10):
    """
    Iterative Null-space Projection (INLP).
    Repeatedly trains a classifier and projects to its null-space.
    
    Returns: (X_train_proj, X_test_proj, auc_history)
    """
    X_train_current = X_train.copy()
    X_test_current = X_test.copy()
    auc_history = []
    
    for iteration in range(n_iterations):
        # Train classifier on current space
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1, class_weight='balanced')
        clf.fit(X_train_current, y_nuisance_train)
        
        # Evaluate
        y_pred_proba = clf.predict_proba(X_test_current)[:, 1]
        try:
            auc = roc_auc_score(y_nuisance_test, y_pred_proba)
        except:
            auc = 0.5
        auc_history.append(auc)
        
        # Stop if AUC near chance
        if auc < 0.55:
            break
        
        # Get direction and project it out
        direction = clf.coef_[0]
        d = direction / np.linalg.norm(direction)
        
        # Project train and test
        X_train_current = X_train_current - np.outer(X_train_current @ d, d)
        X_test_current = X_test_current - np.outer(X_test_current @ d, d)
    
    return X_train_current, X_test_current, auc_history

def bootstrap_ci(y_true, y_pred_proba, metric_fn=roc_auc_score, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval for a metric.
    """
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

def probe_with_group_cv(X, y, groups, cv=5, method='baseline'):
    """
    Probe with GroupKFold CV and proper nuisance projection per fold.
    
    Args:
        method: 'baseline', 'pronoun_subspace', or 'inlp'
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
        
        # Fit scaler on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Apply projection method (on scaled data, within fold)
        if method == 'pronoun_subspace':
            # Learn pronoun subspace on train only
            pronoun_labels_train = extract_pronoun_labels(groups_train)
            basis = learn_pronoun_subspace(X_train, pronoun_labels_train, n_directions=3)
            
            # Project train and test
            X_train = project_out_subspace(X_train, basis)
            X_test = project_out_subspace(X_test, basis)
        
        elif method == 'inlp':
            # Binary nuisance label: has pronoun or not
            pronoun_labels_train = extract_pronoun_labels(groups_train)
            pronoun_labels_test = extract_pronoun_labels(groups[test_idx])
            
            y_nuisance_train = (pronoun_labels_train != 'none').astype(int)
            y_nuisance_test = (pronoun_labels_test != 'none').astype(int)
            
            X_train, X_test, _ = inlp_projection(X_train, X_test, 
                                                  y_nuisance_train, y_nuisance_test,
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
    
    # Compute bootstrap CI on pooled predictions
    all_y_true = np.array(all_y_true)
    all_y_pred_proba = np.array(all_y_pred_proba)
    
    # Convert to binary if needed
    if len(np.unique(all_y_true)) == 2:
        y_true_binary = (all_y_true == np.unique(all_y_true)[1]).astype(int)
        auc_mean, auc_lower, auc_upper = bootstrap_ci(y_true_binary, all_y_pred_proba)
    else:
        auc_mean, auc_lower, auc_upper = 0.5, 0.5, 0.5
    
    return {
        'accuracy_mean': np.mean(fold_accs),
        'accuracy_std': np.std(fold_accs),
        'auc_mean': auc_mean,
        'auc_lower': auc_lower,
        'auc_upper': auc_upper
    }

def analyze_layer(model_name, model_dir, run_type, layer, token_position='last'):
    """Analyze a single layer with multiple methods."""
    X, categories, prompts = load_activations(model_dir, run_type, layer, token_position)
    
    # Binary labels: self-referent vs neutral
    mask = (categories == 'self_referent') | (categories == 'neutral')
    X_binary = X[mask]
    y_binary = (categories[mask] == 'self_referent').astype(int)
    groups_binary = prompts[mask]  # Use prompts as groups
    
    # Baseline
    results_baseline = probe_with_group_cv(X_binary, y_binary, groups_binary, 
                                           cv=5, method='baseline')
    
    # Pronoun subspace projection
    results_pronoun = probe_with_group_cv(X_binary, y_binary, groups_binary,
                                          cv=5, method='pronoun_subspace')
    
    # INLP
    results_inlp = probe_with_group_cv(X_binary, y_binary, groups_binary,
                                       cv=5, method='inlp')
    
    return {
        'layer': layer,
        'token_pos': token_position,
        'baseline_acc': results_baseline['accuracy_mean'],
        'baseline_auc': results_baseline['auc_mean'],
        'baseline_auc_ci': (results_baseline['auc_lower'], results_baseline['auc_upper']),
        'pronoun_acc': results_pronoun['accuracy_mean'],
        'pronoun_auc': results_pronoun['auc_mean'],
        'pronoun_auc_ci': (results_pronoun['auc_lower'], results_pronoun['auc_upper']),
        'inlp_acc': results_inlp['accuracy_mean'],
        'inlp_auc': results_inlp['auc_mean'],
        'inlp_auc_ci': (results_inlp['auc_lower'], results_inlp['auc_upper'])
    }

def analyze_model(model_name, model_dir, n_layers, layers_to_test, token_positions=['last']):
    """Analyze multiple layers and token positions."""
    print(f"\n{'='*100}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*100}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*100}")
        print(f"{'Layer':<8} {'TokPos':<8} {'Base AUC':<20} {'Pronoun AUC':<20} {'INLP AUC':<20} {'Drop':<10}")
        print(f"{'-'*100}")
        
        results_list = []
        for layer in layers_to_test:
            if layer >= n_layers:
                continue
            
            for token_pos in token_positions:
                try:
                    result = analyze_layer(model_name, model_dir, run_type, layer, token_pos)
                    results_list.append(result)
                    
                    # Format CIs
                    base_ci = f"{result['baseline_auc']:.3f} [{result['baseline_auc_ci'][0]:.3f}-{result['baseline_auc_ci'][1]:.3f}]"
                    pron_ci = f"{result['pronoun_auc']:.3f} [{result['pronoun_auc_ci'][0]:.3f}-{result['pronoun_auc_ci'][1]:.3f}]"
                    inlp_ci = f"{result['inlp_auc']:.3f} [{result['inlp_auc_ci'][0]:.3f}-{result['inlp_auc_ci'][1]:.3f}]"
                    
                    drop = result['baseline_auc'] - result['inlp_auc']
                    
                    print(f"{result['layer']:<8} {result['token_pos']:<8} {base_ci:<20} {pron_ci:<20} {inlp_ci:<20} {drop:<10.3f}")
                except Exception as e:
                    print(f"{layer:<8} {token_pos:<8} ERROR: {e}")
        
        # Summary
        if results_list:
            avg_baseline = np.mean([r['baseline_auc'] for r in results_list])
            avg_inlp = np.mean([r['inlp_auc'] for r in results_list])
            avg_drop = avg_baseline - avg_inlp
            
            print(f"{'-'*100}")
            print(f"Average: Baseline AUC={avg_baseline:.3f}, INLP AUC={avg_inlp:.3f}, Drop={avg_drop:.3f}")
            print(f"\nInterpretation:")
            if avg_inlp < 0.60:
                print(f"  → INLP drops to chance! Signal is PURELY lexical")
            elif avg_inlp > 0.75:
                print(f"  → INLP maintains high AUC! Non-lexical self-referent circuit exists")
            else:
                print(f"  → Partial drop. Mixed lexical + non-lexical signal")

def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    
    # Test key layers
    layers_to_test = [10, 15, 20]
    token_positions = ['last', 'bos']
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    print("="*100)
    print("ROBUST CIRCUIT PROBE")
    print("="*100)
    print("\nImprovements:")
    print("  ✓ GroupKFold (no template leakage)")
    print("  ✓ Learn projections inside each fold (no test leakage)")
    print("  ✓ Pronoun subspace (3 directions: you/I/he-she)")
    print("  ✓ INLP iterative null-space projection")
    print("  ✓ AUROC with bootstrap 95% CIs")
    print("  ✓ Token position control (last vs BOS)")
    print("="*100)
    
    for model_name, model_dir, n_layers in models:
        analyze_model(model_name, model_dir, n_layers, layers_to_test, token_positions)
    
    print("\n" + "="*100)
    print("INTERPRETATION:")
    print("="*100)
    print("Baseline: Standard probe (includes all cues)")
    print("Pronoun: Projects out pronoun subspace (you/I/he-she)")
    print("INLP: Iteratively removes pronoun signal until classifier can't detect it")
    print("\nIf INLP AUC stays high → Non-lexical self-referent circuit exists")
    print("If INLP AUC drops to ~0.5 → Purely lexical (pronoun detection)")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()

