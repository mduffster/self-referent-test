"""
Analyze the activation results from the self-referent experiment.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(filepath):
    """Load the activation results."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_activation_differences(results):
    """Analyze differences between self-referent and other categories."""
    comparison_results = results['comparison_results']
    
    # Create a DataFrame for analysis
    data = []
    for activation_name, stats in comparison_results.items():
        if stats['self_referent_mean'] is not None:
            data.append({
                'activation': activation_name,
                'self_referent_mean': stats['self_referent_mean'],
                'confounder_mean': stats['confounder_mean'],
                'neutral_mean': stats['neutral_mean'],
                'self_referent_count': stats['self_referent_count'],
                'confounder_count': stats['confounder_count'],
                'neutral_count': stats['neutral_count']
            })
    
    df = pd.DataFrame(data)
    
    # Calculate differences
    df['self_vs_confounder_diff'] = df['self_referent_mean'] - df['confounder_mean']
    df['self_vs_neutral_diff'] = df['self_referent_mean'] - df['neutral_mean']
    df['confounder_vs_neutral_diff'] = df['confounder_mean'] - df['neutral_mean']
    
    # Calculate relative differences
    df['self_vs_confounder_ratio'] = df['self_referent_mean'] / df['confounder_mean']
    df['self_vs_neutral_ratio'] = df['self_referent_mean'] / df['neutral_mean']
    
    return df

def find_most_different_activations(df, top_n=10):
    """Find the activations with the largest differences."""
    
    # Sort by absolute difference from confounders
    df_sorted = df.reindex(df['self_vs_confounder_diff'].abs().sort_values(ascending=False).index)
    
    print("=== TOP ACTIVATIONS WITH LARGEST SELF-REFERENT vs CONFOUNDER DIFFERENCES ===")
    for i, row in df_sorted.head(top_n).iterrows():
        print(f"\n{row['activation']}:")
        print(f"  Self-referent mean: {row['self_referent_mean']:.6f}")
        print(f"  Confounder mean:    {row['confounder_mean']:.6f}")
        print(f"  Difference:         {row['self_vs_confounder_diff']:.6f}")
        print(f"  Ratio:              {row['self_vs_confounder_ratio']:.2f}x")
    
    return df_sorted.head(top_n)

def categorize_activations(df):
    """Categorize activations by type."""
    categories = {
        'attention_patterns': [],
        'attention_results': [],
        'mlp_outputs': [],
        'embeddings': [],
        'other': []
    }
    
    for activation in df['activation']:
        if 'attn.hook_pattern' in activation:
            categories['attention_patterns'].append(activation)
        elif 'attn.hook_result' in activation:
            categories['attention_results'].append(activation)
        elif 'mlp.hook_post' in activation:
            categories['mlp_outputs'].append(activation)
        elif activation in ['embed', 'pos_embed', 'ln_final']:
            categories['embeddings'].append(activation)
        else:
            categories['other'].append(activation)
    
    return categories

def analyze_by_category(df):
    """Analyze differences by activation category."""
    categories = categorize_activations(df)
    
    print("\n=== ANALYSIS BY CATEGORY ===")
    
    for category, activations in categories.items():
        if not activations:
            continue
            
        print(f"\n{category.upper()}:")
        category_df = df[df['activation'].isin(activations)]
        
        if len(category_df) > 0:
            avg_self = category_df['self_referent_mean'].mean()
            avg_confounder = category_df['confounder_mean'].mean()
            avg_neutral = category_df['neutral_mean'].mean()
            
            print(f"  Average self-referent:  {avg_self:.6f}")
            print(f"  Average confounder:     {avg_confounder:.6f}")
            print(f"  Average neutral:        {avg_neutral:.6f}")
            print(f"  Self vs confounder:     {avg_self - avg_confounder:.6f}")
            print(f"  Self vs neutral:        {avg_self - avg_neutral:.6f}")

def main():
    """Main analysis function."""
    print("Self-Referent Activation Analysis Results")
    print("=" * 50)
    
    # Load results
    results = load_results('results_activation_analysis/run_20251015_094822/activations.json')
    
    print(f"Analyzed {results['num_prompts_analyzed']} prompts")
    print(f"Prompts per category: {results['prompts_per_category']}")
    
    # Analyze differences
    df = analyze_activation_differences(results)
    
    print(f"\nFound {len(df)} activation types with valid data")
    
    # Find most different activations
    top_different = find_most_different_activations(df, top_n=15)
    
    # Analyze by category
    analyze_by_category(df)
    
    # Summary insights
    print("\n=== KEY INSIGHTS ===")
    
    # Count activations where self-referent > confounder
    self_higher = len(df[df['self_vs_confounder_diff'] > 0])
    confounder_higher = len(df[df['self_vs_confounder_diff'] < 0])
    
    print(f"Activations where self-referent > confounder: {self_higher}")
    print(f"Activations where confounder > self-referent: {confounder_higher}")
    
    # Find the most extreme ratios
    max_ratio = df['self_vs_confounder_ratio'].max()
    min_ratio = df['self_vs_confounder_ratio'].min()
    
    print(f"Maximum ratio (self/confounder): {max_ratio:.2f}x")
    print(f"Minimum ratio (self/confounder): {min_ratio:.2f}x")
    
    # Save detailed results
    df.to_csv('results_activation_analysis/run_20251015_094822/activation_analysis.csv', index=False)
    print(f"\n✓ Detailed results saved to: activation_analysis.csv")

if __name__ == "__main__":
    main()
