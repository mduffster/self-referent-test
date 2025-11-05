#!/usr/bin/env python3
"""
Analyze representation distances between self-referent and neutral prompts.
Compares base vs instruct models, Llama vs Qwen.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean

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

def compute_distances(activations, categories):
    """Compute distances between self-referent and neutral centroids."""
    # Get self-referent and neutral activations
    self_ref_mask = categories == 'self_referent'
    neutral_mask = categories == 'neutral'
    
    self_ref_acts = activations[self_ref_mask]
    neutral_acts = activations[neutral_mask]
    
    # Compute centroids
    self_ref_centroid = np.mean(self_ref_acts, axis=0)
    neutral_centroid = np.mean(neutral_acts, axis=0)
    
    # Compute distances
    cos_dist = cosine(self_ref_centroid, neutral_centroid)
    cos_sim = 1 - cos_dist
    euclid_dist = euclidean(self_ref_centroid, neutral_centroid)
    
    # Normalized Euclidean (divide by sqrt of dimension for scale-invariance)
    norm_euclid = euclid_dist / np.sqrt(len(self_ref_centroid))
    
    return {
        'cosine_similarity': cos_sim,
        'cosine_distance': cos_dist,
        'euclidean_distance': euclid_dist,
        'normalized_euclidean': norm_euclid
    }

def analyze_model(model_name, model_dir, n_layers):
    """Analyze representation distances for a model across layers."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}")
    
    for run_type in ['latest_base', 'latest_run']:
        run_label = "BASE" if run_type == 'latest_base' else "INSTRUCT"
        print(f"\n{run_label} Model:")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'Cos Sim':<12} {'Cos Dist':<12} {'Euclidean':<15} {'Norm Euclid':<12}")
        print(f"{'-'*80}")
        
        layer_results = []
        for layer in range(n_layers):
            try:
                activations, categories = load_activations(model_dir, run_type, layer)
                distances = compute_distances(activations, categories)
                layer_results.append((layer, distances))
                
                print(f"{layer:<8} {distances['cosine_similarity']:<12.6f} "
                      f"{distances['cosine_distance']:<12.6f} "
                      f"{distances['euclidean_distance']:<15.2f} "
                      f"{distances['normalized_euclidean']:<12.6f}")
            except Exception as e:
                print(f"{layer:<8} ERROR: {e}")
        
        # Summary stats
        if layer_results:
            cos_sims = [r[1]['cosine_similarity'] for r in layer_results]
            norm_euclids = [r[1]['normalized_euclidean'] for r in layer_results]
            
            print(f"{'-'*80}")
            print(f"Mean cosine similarity: {np.mean(cos_sims):.6f} (std: {np.std(cos_sims):.6f})")
            print(f"Mean normalized Euclidean: {np.mean(norm_euclids):.6f} (std: {np.std(norm_euclids):.6f})")
            print(f"Early layers (0-7) avg cos sim: {np.mean(cos_sims[:8]):.6f}")
            print(f"Middle layers (8-23) avg cos sim: {np.mean(cos_sims[8:24]) if len(cos_sims) > 24 else 'N/A'}")
            print(f"Late layers (24+) avg cos sim: {np.mean(cos_sims[24:]) if len(cos_sims) > 24 else 'N/A'}")

def main():
    base_path = Path("/Users/mattduffy/self-referent-test")
    
    # Analyze Llama
    llama_dir = base_path / "results_llama_activation"
    analyze_model("Llama", llama_dir, n_layers=32)
    
    # Analyze Mistral
    mistral_dir = base_path / "results_activation_analysis"
    analyze_model("Mistral", mistral_dir, n_layers=32)
    
    # Analyze Qwen
    qwen_dir = base_path / "results_qwen_activation"
    analyze_model("Qwen", qwen_dir, n_layers=28)
    
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*80}")
    print("Cosine Similarity (higher = more similar):")
    print("  - Close to 1.0: Representations are very similar (convergence)")
    print("  - Close to 0.0: Representations are orthogonal (distinct)")
    print("  - Negative: Representations point in opposite directions")
    print("\nNormalized Euclidean (lower = more similar):")
    print("  - Small values: Centroids are close together")
    print("  - Large values: Centroids are far apart (more distinct)")
    print("\nHypothesis Test:")
    print("  - Qwen-Instruct should maintain LOWER cosine similarity (more distinct)")
    print("  - Llama-Instruct should show HIGHER cosine similarity (convergence)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

