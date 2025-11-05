#!/usr/bin/env python3
"""
Robust linear probe analysis with proper controls.
Addresses: token position shortcuts, cross-distribution transfer, held-out evaluation.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_activations(model_dir, run_type, layer, token_position='last'):
    """
    Load MLP post activations for a specific layer.
    
    Args:
        token_position: 'bos', 'first_content', 'last'
    """
    npz_path = model_dir / run_type / f"raw_blocks_{layer}_mlp_post.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    activations = []
    for act in data['activations']:
        # act shape: (1, seq_len, hidden_dim)
        if token_position == 'bos':
            token = act[0, 0, :]  # First token (BOS)
        elif token_position == 'first_content':
            # Second token (first content token after BOS)
            token = act[0, 1, :] if act.shape[1] > 1 else act[0, 0, :]
        elif token_position == 'last':
            token = act[0, -1, :]  # Last token
        else:
            raise ValueError(f"Unknown token_position: {token_position}")
        activations.append(token)
    
    activations = np.array(activations)  # (n_samples, hidden_dim)
    categories = data['categories']
    prompts = data['prompts']
    
    return activations, categories, prompts

def probe_with_cv(X, y, cv=5):
    """Train probe with proper held-out CV evaluation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    test_accs = []
    test_f1s = []
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train probe
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000, 
                                random_state=42, C=0.1)
        clf.fit(X_train, y_train)
        
        # Evaluate on held-out test
        y_pred = clf.predict(X_test)
        test_accs.append(accuracy_score(y_test, y_pred))
        test_f1s.append(f1_score(y_test, y_pred, average='macro'))
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    return {
        'test_accuracy': np.mean(test_accs),
        'test_accuracy_std': np.std(test_accs),
        'test_f1_macro': np.mean(test_f1s),
        'all_y_true': np.array(all_y_true),
        'all_y_pred': np.array(all_y_pred)
    }

def cross_distribution_probe(X_train, y_train, X_test, y_test):
    """Train on one distribution, test on another."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000,
                             random_state=42, C=0.1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'y_true': y_test,
        'y_pred': y_pred
    }

def analyze_token_positions(model_name, model_dir, n_layers, layer_sample=[0, 10, 20]):
    """Quick check: does token position matter?"""
    print(f"\n{'='*80}")
    print(f"TOKEN POSITION ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'BOS Acc':<12} {'First Cont':<12} {'Last Tok':<12}")
        print(f"{'-'*80}")
        
        for layer in layer_sample:
            if layer >= n_layers:
                continue
            
            results = {}
            for pos in ['bos', 'first_content', 'last']:
                try:
                    X, y, _ = load_activations(model_dir, run_type, layer, token_position=pos)
                    probe_results = probe_with_cv(X, y, cv=5)
                    results[pos] = probe_results['test_accuracy']
                except Exception as e:
                    results[pos] = 0.0
            
            print(f"{layer:<8} {results.get('bos', 0):<12.4f} "
                  f"{results.get('first_content', 0):<12.4f} "
                  f"{results.get('last', 0):<12.4f}")
    
    print()

def analyze_cross_distribution(model_name, model_dir, n_layers, layer_sample=[10, 15, 20]):
    """Train on base, test on instruct (and vice versa)."""
    print(f"\n{'='*80}")
    print(f"CROSS-DISTRIBUTION TRANSFER: {model_name.upper()}")
    print(f"Train on BASE → Test on INSTRUCT | Train on INSTRUCT → Test on BASE")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'Base→Inst':<15} {'Base→Inst F1':<15} {'Inst→Base':<15} {'Inst→Base F1':<15}")
    print(f"{'-'*80}")
    
    for layer in layer_sample:
        if layer >= n_layers:
            continue
        
        try:
            # Load both distributions
            X_base, y_base, _ = load_activations(model_dir, 'latest_base', layer, 'last')
            X_inst, y_inst, _ = load_activations(model_dir, 'latest_run', layer, 'last')
            
            # Base → Instruct
            results_b2i = cross_distribution_probe(X_base, y_base, X_inst, y_inst)
            
            # Instruct → Base
            results_i2b = cross_distribution_probe(X_inst, y_inst, X_base, y_base)
            
            print(f"{layer:<8} {results_b2i['accuracy']:<15.4f} "
                  f"{results_b2i['f1_macro']:<15.4f} "
                  f"{results_i2b['accuracy']:<15.4f} "
                  f"{results_i2b['f1_macro']:<15.4f}")
            
        except Exception as e:
            print(f"{layer:<8} ERROR: {e}")
    
    print()

def detailed_cv_analysis(model_name, model_dir, layer, run_type):
    """Show held-out confusion matrix from CV."""
    run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
    
    X, y, prompts = load_activations(model_dir, run_type, layer, 'last')
    results = probe_with_cv(X, y, cv=5)
    
    print(f"\n{model_name.upper()} - {run_label} - Layer {layer}")
    print(f"Held-out Accuracy: {results['test_accuracy']:.4f} (±{results['test_accuracy_std']:.4f})")
    print(f"Held-out F1 (macro): {results['test_f1_macro']:.4f}")
    
    # Confusion matrix on held-out predictions
    cm = confusion_matrix(results['all_y_true'], results['all_y_pred'],
                         labels=['self_referent', 'neutral', 'confounder', 'third_person'])
    
    print("\nHeld-out Confusion Matrix (across all CV folds):")
    print(f"{'':>15} {'self_ref':>10} {'neutral':>10} {'confound':>10} {'3rd_pers':>10}")
    for i, label in enumerate(['self_referent', 'neutral', 'confounder', 'third_person']):
        print(f"{label:>15} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10} {cm[i,3]:>10}")
    
    # Per-class accuracy
    print("\nPer-class accuracy (held-out):")
    for i, label in enumerate(['self_referent', 'neutral', 'confounder', 'third_person']):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {label:>15}: {class_acc:.4f}")

def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    print("="*80)
    print("ROBUST LINEAR PROBE ANALYSIS")
    print("Addresses: token shortcuts, cross-distribution transfer, held-out eval")
    print("="*80)
    
    # 1. Token position check (quick shortcut detector)
    print("\n" + "="*80)
    print("CONTROL 1: TOKEN POSITION SHORTCUTS")
    print("If BOS gives high accuracy, there's a format/positional shortcut")
    print("="*80)
    for model_name, model_dir, n_layers in models:
        analyze_token_positions(model_name, model_dir, n_layers, layer_sample=[0, 10, 20])
    
    # 2. Cross-distribution transfer
    print("\n" + "="*80)
    print("CONTROL 2: CROSS-DISTRIBUTION TRANSFER")
    print("If train-base→test-instruct drops heavily, representations diverge")
    print("="*80)
    for model_name, model_dir, n_layers in models:
        analyze_cross_distribution(model_name, model_dir, n_layers, layer_sample=[10, 15, 20])
    
    # 3. Detailed held-out analysis
    print("\n" + "="*80)
    print("CONTROL 3: HELD-OUT CONFUSION MATRICES")
    print("Shows what the probe actually predicts on unseen data")
    print("="*80)
    for model_name, model_dir, n_layers in models:
        for run_type in ['latest_base', 'latest_run']:
            detailed_cv_analysis(model_name, model_dir, 15, run_type)
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("• If BOS ≈ Last token accuracy: positional/format shortcut")
    print("• If cross-distribution drops: representations change with instruction tuning")
    print("• Held-out confusion shows which classes actually confuse")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

