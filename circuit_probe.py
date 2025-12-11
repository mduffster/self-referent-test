#!/usr/bin/env python3
"""
Circuit probe: Remove pronoun directions and test for non-lexical self-referent signal.

Strategy:
1. Train a probe to detect pronouns (you/I/he-she vs none)
2. Find the pronoun direction in activation space
3. Project out (residualize) that direction
4. Probe residuals for self-referent vs neutral
5. If accuracy drops to chance, it was purely lexical
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
        else:
            token = act[0, 0, :]
        activations.append(token)
    
    return np.array(activations), data['categories'], data['prompts']

def extract_pronoun_direction(X, categories):
    """
    Extract the linear direction that encodes pronoun presence.
    Train a classifier: pronoun (you/I/he-she) vs no-pronoun (neutral).
    Returns the weight vector (direction) and the fitted scaler.
    """
    # Create pronoun labels
    pronoun_labels = np.array([
        'pronoun' if cat in ['self_referent', 'confounder', 'third_person'] 
        else 'no_pronoun' 
        for cat in categories
    ])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train binary classifier
    clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    clf.fit(X_scaled, pronoun_labels)
    
    # The weight vector is the pronoun direction
    pronoun_direction = clf.coef_[0]  # Shape: (hidden_dim,)
    
    return pronoun_direction, scaler

def project_out_direction(X, direction):
    """
    Project out a direction from X (residualization).
    X_residual = X - X.dot(d) * d  (where d is unit direction)
    """
    # Normalize direction
    d = direction / np.linalg.norm(direction)
    
    # Project X onto d
    projections = X.dot(d)  # Shape: (n_samples,)
    
    # Subtract projection
    X_residual = X - np.outer(projections, d)
    
    return X_residual

def probe_with_cv(X, y, cv=5):
    """Probe with cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    test_accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        test_accs.append(accuracy_score(y_test, y_pred))
    
    return np.mean(test_accs), np.std(test_accs)

def binary_probe(categories):
    """Convert categories to binary: self_referent vs neutral."""
    return np.array(['self_referent' if cat == 'self_referent' else 'neutral' 
                     for cat in categories])

def analyze_layer(model_name, model_dir, run_type, layer):
    """Analyze a single layer."""
    X, categories, prompts = load_activations(model_dir, run_type, layer)
    
    # Binary labels: self-referent vs neutral
    y_binary = binary_probe(categories)
    mask = (categories == 'self_referent') | (categories == 'neutral')
    X_binary = X[mask]
    y_binary = y_binary[mask]
    
    # 1. Baseline: probe original activations
    acc_baseline, std_baseline = probe_with_cv(X_binary, y_binary)
    
    # 2. Extract pronoun direction (using all data)
    pronoun_dir, scaler = extract_pronoun_direction(X, categories)
    
    # 3. Project out pronoun direction
    X_scaled = scaler.transform(X)
    X_residual = project_out_direction(X_scaled, pronoun_dir)
    
    # 4. Probe residuals (self-referent vs neutral only)
    X_residual_binary = X_residual[mask]
    acc_residual, std_residual = probe_with_cv(X_residual_binary, y_binary)
    
    # 5. Also check pronoun classification accuracy
    pronoun_labels_all = np.array([
        'pronoun' if cat in ['self_referent', 'confounder', 'third_person'] 
        else 'no_pronoun' 
        for cat in categories
    ])
    acc_pronoun, _ = probe_with_cv(X, pronoun_labels_all)
    
    return {
        'layer': layer,
        'baseline_acc': acc_baseline,
        'baseline_std': std_baseline,
        'residual_acc': acc_residual,
        'residual_std': std_residual,
        'pronoun_acc': acc_pronoun,
        'drop': acc_baseline - acc_residual
    }

def analyze_model(model_name, model_dir, n_layers, layers_to_test):
    """Analyze multiple layers for a model."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*80}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'Baseline':<12} {'Residual':<12} {'Drop':<12} {'Pronoun Acc':<12}")
        print(f"{'-'*80}")
        
        results = []
        for layer in layers_to_test:
            if layer >= n_layers:
                continue
            
            try:
                result = analyze_layer(model_name, model_dir, run_type, layer)
                results.append(result)
                
                print(f"{result['layer']:<8} "
                      f"{result['baseline_acc']:<12.4f} "
                      f"{result['residual_acc']:<12.4f} "
                      f"{result['drop']:<12.4f} "
                      f"{result['pronoun_acc']:<12.4f}")
            except Exception as e:
                print(f"{layer:<8} ERROR: {e}")
        
        # Summary
        if results:
            avg_baseline = np.mean([r['baseline_acc'] for r in results])
            avg_residual = np.mean([r['residual_acc'] for r in results])
            avg_drop = np.mean([r['drop'] for r in results])
            
            print(f"{'-'*80}")
            print(f"Average: baseline={avg_baseline:.4f}, residual={avg_residual:.4f}, drop={avg_drop:.4f}")
            print(f"\nInterpretation:")
            if avg_residual < 0.60:
                print(f"  → Drops to near-chance! Signal is PURELY lexical (pronouns)")
            elif avg_residual > 0.80:
                print(f"  → Maintains high accuracy! Non-lexical self-referent circuit exists")
            else:
                print(f"  → Partial drop. Mixed lexical + non-lexical signal")

def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    
    # Test middle layers where you found the main effects
    layers_to_test = [10, 12, 15, 18, 20, 23]
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    print("="*80)
    print("CIRCUIT PROBE: RESIDUALIZING OUT PRONOUN FEATURES")
    print("="*80)
    print("\nTask: Classify self-referent vs neutral AFTER removing pronoun signal")
    print("Baseline: Standard probe (includes pronouns)")
    print("Residual: Probe after projecting out pronoun direction")
    print("\nIf residual drops to ~50%, it's purely lexical")
    print("If residual stays high, there's a non-lexical self-referent circuit")
    print("="*80)
    
    for model_name, model_dir, n_layers in models:
        analyze_model(model_name, model_dir, n_layers, layers_to_test)
    
    print("\n" + "="*80)
    print("SAFETY IMPLICATIONS:")
    print("="*80)
    print("High residual accuracy = Model has non-lexical 'self' circuit")
    print("  → Pro: Maintains self/other boundary beyond surface cues")
    print("  → Con: More complex, harder to control")
    print("\nLow residual accuracy = Model relies on lexical cues only")
    print("  → Pro: Simpler, more predictable")
    print("  → Con: May lose self-awareness in adversarial contexts")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

