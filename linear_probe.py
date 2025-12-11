#!/usr/bin/env python3
"""
Multi-class linear probe analysis for self-referent prompt representations.
Trains logistic regression to classify: self_referent, neutral, confounder, third_person.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_activations(model_dir, run_type, layer):
    """Load MLP post activations for a specific layer."""
    npz_path = model_dir / run_type / f"raw_blocks_{layer}_mlp_post.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract last token activations
    activations = []
    for act in data['activations']:
        # act shape: (1, seq_len, hidden_dim)
        last_token = act[0, -1, :]  # Take last token
        activations.append(last_token)
    
    activations = np.array(activations)  # (n_samples, hidden_dim)
    categories = data['categories']
    
    return activations, categories

def probe_layer(activations, categories, layer_num):
    """Train and evaluate a multi-class linear probe using cross-validation."""
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = categories
    
    # Multi-class logistic regression with L2 regularization
    clf = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=42,
        C=0.1  # Regularization strength
    )
    
    # Cross-validation with multiple metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }
    
    cv_results = cross_validate(
        clf, X, y,
        cv=5,  # 5-fold cross-validation
        scoring=scoring,
        return_train_score=True
    )
    
    results = {
        'layer': layer_num,
        'test_accuracy': np.mean(cv_results['test_accuracy']),
        'test_accuracy_std': np.std(cv_results['test_accuracy']),
        'train_accuracy': np.mean(cv_results['train_accuracy']),
        'test_f1_macro': np.mean(cv_results['test_f1_macro']),
        'test_f1_weighted': np.mean(cv_results['test_f1_weighted'])
    }
    
    return results

def analyze_model(model_name, model_dir, n_layers):
    """Analyze probe performance for a model across layers."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'Test Acc':<12} {'±Std':<10} {'Train Acc':<12} {'F1 Macro':<12} {'F1 Weight':<12}")
        print(f"{'-'*80}")
        
        layer_results = []
        for layer in range(n_layers):
            try:
                activations, categories = load_activations(model_dir, run_type, layer)
                results = probe_layer(activations, categories, layer)
                layer_results.append(results)
                
                print(f"{layer:<8} {results['test_accuracy']:<12.4f} "
                      f"{results['test_accuracy_std']:<10.4f} "
                      f"{results['train_accuracy']:<12.4f} "
                      f"{results['test_f1_macro']:<12.4f} "
                      f"{results['test_f1_weighted']:<12.4f}")
            except Exception as e:
                print(f"{layer:<8} ERROR: {e}")
        
        # Summary stats
        if layer_results:
            test_accs = [r['test_accuracy'] for r in layer_results]
            f1_macros = [r['test_f1_macro'] for r in layer_results]
            
            print(f"{'-'*80}")
            print(f"Mean test accuracy: {np.mean(test_accs):.4f} (std: {np.std(test_accs):.4f})")
            print(f"Max test accuracy: {np.max(test_accs):.4f} at layer {np.argmax(test_accs)}")
            print(f"Mean F1 (macro): {np.mean(f1_macros):.4f}")
            
            # Layer-wise breakdown
            early = test_accs[:8]
            middle = test_accs[8:24] if len(test_accs) > 24 else test_accs[8:]
            late = test_accs[24:] if len(test_accs) > 24 else []
            
            print(f"Early layers (0-7) avg: {np.mean(early):.4f}")
            if middle:
                print(f"Middle layers (8-23) avg: {np.mean(middle):.4f}")
            if late:
                print(f"Late layers (24+) avg: {np.mean(late):.4f}")

def detailed_analysis(model_name, model_dir, n_layers):
    """Do a detailed analysis at the best performing layer."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        
        # Find best layer by testing a few key layers
        best_acc = 0
        best_layer = 0
        for layer in [10, 15, 20]:
            if layer >= n_layers:
                continue
            try:
                activations, categories = load_activations(model_dir, run_type, layer)
                results = probe_layer(activations, categories, layer)
                if results['test_accuracy'] > best_acc:
                    best_acc = results['test_accuracy']
                    best_layer = layer
            except:
                pass
        
        # Train on best layer and show confusion matrix
        print(f"\n{run_label} Model - Best Layer: {best_layer} (Acc: {best_acc:.4f})")
        print(f"{'-'*80}")
        
        try:
            activations, categories = load_activations(model_dir, run_type, best_layer)
            
            # Standardize and train
            scaler = StandardScaler()
            X = scaler.fit_transform(activations)
            y = categories
            
            clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, C=0.1)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            
            # Classification report
            print("\nClassification Report (on training data):")
            print(classification_report(y, y_pred, zero_division=0))
            
            # Confusion matrix
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y, y_pred, labels=['self_referent', 'neutral', 'confounder', 'third_person'])
            print(f"{'':>15} {'self_ref':>10} {'neutral':>10} {'confound':>10} {'3rd_pers':>10}")
            for i, label in enumerate(['self_referent', 'neutral', 'confounder', 'third_person']):
                print(f"{label:>15} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10} {cm[i,3]:>10}")
        except Exception as e:
            print(f"Error in detailed analysis: {e}")

def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    
    # Analyze all models
    print("\n" + "="*80)
    print("MULTI-CLASS LINEAR PROBE ANALYSIS")
    print("Task: Classify 4 categories (self_referent, neutral, confounder, third_person)")
    print("Method: Logistic Regression with 5-fold CV")
    print("="*80)
    
    models = [
        ("Llama", base_path / "results_llama_activation", 32),
        ("Mistral", base_path / "results_activation_analysis", 32),
        ("Qwen", base_path / "results_qwen_activation", 28)
    ]
    
    for model_name, model_dir, n_layers in models:
        analyze_model(model_name, model_dir, n_layers)
    
    # Detailed analysis for each model
    for model_name, model_dir, n_layers in models:
        detailed_analysis(model_name, model_dir, n_layers)
    
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    print("High accuracy = categories are linearly separable (distinct representations)")
    print("Low accuracy = categories overlap (similar representations)")
    print("Compare base vs instruct to see how instruction tuning affects separability")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

