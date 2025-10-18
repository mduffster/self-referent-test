#!/usr/bin/env python3
"""
Role-Conditioning Circuit Analysis: Visualization Script
======================================================

This script analyzes attention patterns from Mistral-7B to identify role-conditioning circuits.
Follows the structure: 1. Libraries, 2. Data Load, 3. Calculations, 4. Visualizations
"""

# =============================================================================
# 1. LIBRARIES AND SETUP
# =============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import glob
import os
import argparse
import json
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_family_config():
    """Load model family configuration."""
    try:
        config_path = Path(__file__).parent / "model_family_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Parse command line arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize self-referent experiment results")
    
    # Family-based arguments
    parser.add_argument("--family", choices=["llama", "qwen", "mistral"],
                       help="Model family (overrides individual settings)")
    parser.add_argument("--variant", choices=["base", "instruct"], default="instruct",
                       help="Model variant: base or instruct (default: instruct)")
    
    # Individual arguments (for backward compatibility)
    parser.add_argument("--output_type", 
                       choices=["normal", "intervention", "not_specified", "base"],
                       help="Output type: 'normal' for figures/, 'intervention' for figures/intervention/, 'base' for figures_base/, 'not_specified' for figures/not_specified/")
    
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Input directory for data files. If not specified, uses latest_run for normal/not_specified, latest_intervention for intervention")
    
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name to look up in config (e.g., 'Qwen/Qwen2.5-7B-Instruct'). If not specified, uses default from config")
    
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (overrides output_type)")
    
    args = parser.parse_args()
    
    # If family is specified, load configuration
    if args.family:
        config = load_family_config()
        if config:
            family_config = config["model_families"][args.family][args.variant]
            
            # Override with family config
            args.output_type = family_config["output_type"]
            args.input_dir = f"{family_config['output_dir']}/{family_config['output_type']}"
            args.model_name = family_config["model_id"]
            args.output_dir = family_config["figures_dir"]
    
    return args

# Parse arguments
args = parse_args()

# Load visualization configuration
def load_model_config():
    """Load model configuration from JSON file."""
    config_path = Path(__file__).parent / "visualization_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_model_config()

# Determine model architecture
model_name = args.model_name or config["default_model"]
if model_name in config["models"]:
    n_layers = config["models"][model_name]["n_layers"]
    n_heads = config["models"][model_name]["n_heads"]
    print(f"Using model config: {model_name}")
    print(f"Architecture: {n_layers} layers, {n_heads} heads")
else:
    print(f"Warning: Model '{model_name}' not found in config, using default Mistral settings")
    n_layers = 32
    n_heads = 32

# Set output directory based on argument
if args.output_dir is not None:
    output_dir = args.output_dir
elif args.output_type == "normal":
    output_dir = "figures"
elif args.output_type == "intervention":
    output_dir = "figures/intervention"
elif args.output_type == "base":
    output_dir = "figures_base"
else:  # not_specified
    output_dir = "figures/not_specified"

# Set input directory based on argument
if args.input_dir is not None:
    input_dir = args.input_dir
else:
    if args.output_type == "intervention":
        input_dir = "results_activation_analysis/latest_intervention"
    elif args.output_type == "base":
        input_dir = "results_activation_analysis/latest_base"
    else:  # normal or not_specified
        input_dir = "results_activation_analysis/latest_run"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("✓ Libraries imported successfully")
print(f"✓ Output directory set to: {output_dir}")
print(f"✓ Input directory set to: {input_dir}")

# Check if input directory exists
if not os.path.exists(input_dir):
    print(f"⚠️  Warning: Input directory {input_dir} does not exist!")
    print("Available directories in results_activation_analysis:")
    if os.path.exists("results_activation_analysis"):
        for item in os.listdir("results_activation_analysis"):
            if os.path.isdir(os.path.join("results_activation_analysis", item)):
                print(f"  - {item}")
    exit(1)

# =============================================================================
# 2. DATA LOADING AND SUMMARY
# =============================================================================

print("\n" + "="*60)
print("DATA LOADING AND SUMMARY")
print("="*60)

# Model configuration
print(f"Using {model_name} configuration...")
print(f"Model: {n_layers} layers, {n_heads} heads per layer")

# Load raw activation data from NPZ files
print("\nLoading raw activation data from NPZ files...")

# Find NPZ files
npz_files = glob.glob(f"{input_dir}/raw_*.npz")
print(f"Found {len(npz_files)} NPZ files")

# Load all activation data and compute within-block averages
layer_attention_data = {}

for npz_file in npz_files:
    filename = os.path.basename(npz_file)
    activation_name = filename.replace('raw_', '').replace('.npz', '')

    # Only process attention pattern files
    if 'attn_pattern' not in activation_name:
        continue

    print(f"\nProcessing {activation_name}...")
    data = np.load(npz_file, allow_pickle=True)
    
    if len(data.keys()) == 0:
        print(f"  Skipping empty file")
        continue

    # Get the activations and categories
    activations = data['activations']
    categories = data['categories']
    prompts = data['prompts']

    print(f"  Loaded {len(activations)} activations")
    print(f"  Categories: {np.unique(categories)}")

    # Extract layer number
    try:
        if 'blocks_' in activation_name:
            layer = int(activation_name.split('blocks_')[1].split('_')[0])
        else:
            layer = int(activation_name.split('blocks.')[1].split('.')[0])
    except (IndexError, ValueError):
        print(f"  Could not parse layer from: {activation_name}")
        continue

    # Compute within-block averages for each activation
    within_block_averages = []
    for i, (activation, category) in enumerate(zip(activations, categories)):
        if activation is not None:
            # Use attention entropy instead of mean magnitude (normalized attention patterns)
            # Shape: (1, n_heads, seq_len, seq_len) -> (n_heads,) -> scalar
            per_head_entropies = []
            for head in range(activation.shape[1]):  # For each head
                head_attn = activation[0, head, :, :]  # (seq_len, seq_len)
                # Calculate entropy for each query position, then average
                entropies = []
                for query_pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[query_pos, :]  # Attention distribution for this query
                    # Avoid log(0) by adding small epsilon
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    entropies.append(entropy)
                per_head_entropies.append(np.mean(entropies))
            
            avg_magnitude = np.mean(per_head_entropies)  # Average entropy across heads
            within_block_averages.append({
                'layer': layer,
                'category': category,
                'prompt_idx': i,
                'avg_entropy': avg_magnitude,
                'prompt': prompts[i]
            })

    layer_attention_data[layer] = within_block_averages
    print(f"  Layer {layer}: {len(within_block_averages)} activations processed")

print(f"\n✓ Processed {len(layer_attention_data)} layers with attention data")

# Create DataFrame with all the data
df_data = []
for layer, activations in layer_attention_data.items():
    for activation in activations:
        df_data.append(activation)

df = pd.DataFrame(df_data)
print(f"\nDataFrame created with {len(df)} rows")
print(f"Layers: {sorted(df['layer'].unique())}")
print(f"Categories: {df['category'].unique()}")

# Show summary statistics
print(f"\nPer-category counts:")
print(df['category'].value_counts())

print(f"\nLayer range: {df['layer'].min()} to {df['layer'].max()}")
print(f"Average attention entropy by category:")
print(df.groupby('category')['avg_entropy'].mean())

print("\n✓ Data loading complete!")

# =============================================================================
# 3. DATA MANIPULATION AND CALCULATIONS
# =============================================================================

print("\n" + "="*60)
print("DATA MANIPULATION AND CALCULATIONS")
print("="*60)

print("Computing layer-wise statistics...")

# Group by layer and category, compute overall averages
layer_stats = []
for layer in sorted(df['layer'].unique()):
    layer_df = df[df['layer'] == layer]
    
    # Calculate statistics for each category
    stats_dict = {}
    for category in ['self_referent', 'neutral', 'confounder', 'third_person']:
        cat_data = layer_df[layer_df['category'] == category]['avg_entropy']
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            ci = stats.t.interval(0.95, len(cat_data)-1, loc=mean_val, scale=stats.sem(cat_data))
            stats_dict[category] = {'mean': mean_val, 'ci': ci, 'count': len(cat_data)}
    
    # Only include layers that have at least the three main categories
    if len(stats_dict) >= 3:
        layer_data = {
            'layer': layer,
            'self_mean': stats_dict['self_referent']['mean'],
            'self_ci': stats_dict['self_referent']['ci'],
            'neutral_mean': stats_dict['neutral']['mean'],
            'neutral_ci': stats_dict['neutral']['ci'],
            'confounder_mean': stats_dict['confounder']['mean'],
            'confounder_ci': stats_dict['confounder']['ci']
        }
        
        # Add third_person if available
        if 'third_person' in stats_dict:
            layer_data['third_person_mean'] = stats_dict['third_person']['mean']
            layer_data['third_person_ci'] = stats_dict['third_person']['ci']
        
        layer_stats.append(layer_data)

print(f"✓ Layer-wise statistics computed for {len(layer_stats)} layers")

# Compute head-wise differences for heatmaps
print("Computing head-wise attention differences...")

head_diffs_self_neutral = np.zeros((n_layers, n_heads))  # layers x heads
head_diffs_self_confounder = np.zeros((n_layers, n_heads))

for layer in range(n_layers):
    npz_file = f'{input_dir}/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts:
            # Calculate mean attention magnitude per head for each category
            self_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in self_acts], axis=0)[0]  # (n_heads,)
            neutral_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in neutral_acts], axis=0)[0]
            confounder_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in confounder_acts], axis=0)[0]
            
            # Calculate differences
            head_diffs_self_neutral[layer] = self_head_means - neutral_head_means
            head_diffs_self_confounder[layer] = self_head_means - confounder_head_means

# Find top role-sensitive heads
top_heads = []
for layer in range(n_layers):
    for head in range(n_heads):
        top_heads.append((layer, head, head_diffs_self_neutral[layer, head]))

top_heads.sort(key=lambda x: x[2], reverse=True)

print(f"✓ Head-wise differences computed")

# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("VISUALIZATIONS")
print("="*60)

# -----------------------------------------------------------------------------
# Visualization 1: Layer-wise Attention Lines (3 traces with 95% CIs)
# -----------------------------------------------------------------------------

print("Creating Visualization 1: Layer-wise Attention Lines...")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
layers = [d['layer'] for d in layer_stats]

# Plot means
ax.plot(layers, [d['self_mean'] for d in layer_stats], 'r-', linewidth=2, label='Self-referent', marker='o', markersize=4)
ax.plot(layers, [d['neutral_mean'] for d in layer_stats], 'b-', linewidth=2, label='Neutral', marker='s', markersize=4)
ax.plot(layers, [d['confounder_mean'] for d in layer_stats], 'g-', linewidth=2, label='Confounder', marker='^', markersize=4)
ax.plot(layers, [d['third_person_mean'] for d in layer_stats], 'm-', linewidth=2, label='Third-person', marker='d', markersize=4)

# Plot confidence intervals
for d in layer_stats:
    layer = d['layer']
    ax.fill_between([layer, layer], [d['self_ci'][0], d['self_ci'][1]], alpha=0.2, color='red')
    ax.fill_between([layer, layer], [d['neutral_ci'][0], d['neutral_ci'][1]], alpha=0.2, color='blue')
    ax.fill_between([layer, layer], [d['confounder_ci'][0], d['confounder_ci'][1]], alpha=0.2, color='green')
    ax.fill_between([layer, layer], [d['third_person_ci'][0], d['third_person_ci'][1]], alpha=0.2, color='magenta')

ax.set_xlabel('Layer')
ax.set_ylabel('Mean Attention Entropy')
ax.set_title('Layer-wise Attention Entropy: Role-Conditioning Circuits')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_1_layer_attention_lines.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_1_layer_attention_lines.png")

print(f"✓ Visualization 1 complete: {len(layer_stats)} layers analyzed")

# -----------------------------------------------------------------------------
# Visualization 2: Δ-Heatmap Per Head (Self - Neutral, Self - Confounder)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 2: Δ-Heatmap Per Head...")

# Create heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Self - Neutral heatmap
im1 = ax1.imshow(head_diffs_self_neutral, cmap='RdBu_r', aspect='auto')
ax1.set_xlabel('Head')
ax1.set_ylabel('Layer')
ax1.set_title('Δ = Self - Neutral (Attention Magnitude)')
ax1.set_xticks(range(0, n_heads, max(1, n_heads//8)))
ax1.set_yticks(range(0, n_layers, max(1, n_layers//8)))
plt.colorbar(im1, ax=ax1)

# Self - Confounder heatmap
im2 = ax2.imshow(head_diffs_self_confounder, cmap='RdBu_r', aspect='auto')
ax2.set_xlabel('Head')
ax2.set_ylabel('Layer')
ax2.set_title('Δ = Self - Confounder (Attention Magnitude)')
ax2.set_xticks(range(0, n_heads, max(1, n_heads//8)))
ax2.set_yticks(range(0, n_layers, max(1, n_layers//8)))
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_2_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_2_heatmaps.png")

# Find and annotate top role-sensitive heads
print("\nTop 10 Role-Sensitive Heads (Self - Neutral):")
for i, (layer, head, diff) in enumerate(top_heads[:10]):
    print(f"  {i+1}. Layer {layer}, Head {head}: Δ = {diff:.6f}")

print(f"✓ Visualization 2 complete")

# -----------------------------------------------------------------------------
# Visualization 3: Token-Conditioned Attention Map (simplified version)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 3: Token-Conditioned Attention Map...")

# For this analysis, we'll look at attention to the first few tokens across conditions
token_attention_data = []
key_layers = [0, 5, 10, 15, 20, 25, 30, 31]  # Selected layers for clarity

print(f"Processing {len(key_layers)} key layers for token attention analysis...")

for i, layer in enumerate(key_layers):
    print(f"  Processing layer {layer} ({i+1}/{len(key_layers)})...")
    npz_file = f'{input_dir}/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        third_person_acts = [act for act, cat in zip(activations, categories) if cat == 'third_person' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts and third_person_acts:
            print(f"    Computing attention patterns for {len(self_acts)} self, {len(neutral_acts)} neutral, {len(confounder_acts)} confounder, {len(third_person_acts)} third-person activations...")
            # Calculate attention to first 3 tokens (positions 0, 1, 2)
            # For self-referent (5 tokens), neutral/confounder/third-person (9 tokens)
            self_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in self_acts], axis=0)  # (n_heads,)
            neutral_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in neutral_acts], axis=0)
            confounder_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in confounder_acts], axis=0)
            third_person_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in third_person_acts], axis=0)
            print(f"    ✓ Layer {layer} processed")
            
            token_attention_data.append({
                'layer': layer,
                'self': self_attn_to_first3,
                'neutral': neutral_attn_to_first3,
                'confounder': confounder_attn_to_first3,
                'third_person': third_person_attn_to_first3
            })

# Create token-conditioned attention plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, data in enumerate(token_attention_data):
    ax = axes[i]
    heads = range(n_heads)
    
    ax.plot(heads, data['self'], 'r-', linewidth=2, label='Self-referent', marker='o', markersize=3)
    ax.plot(heads, data['neutral'], 'b-', linewidth=2, label='Neutral', marker='s', markersize=3)
    ax.plot(heads, data['confounder'], 'g-', linewidth=2, label='Confounder', marker='^', markersize=3)
    ax.plot(heads, data['third_person'], 'm-', linewidth=2, label='Third-person', marker='d', markersize=3)
    
    ax.set_title(f'Layer {data["layer"]}')
    ax.set_xlabel('Head')
    ax.set_ylabel('Attention to First 3 Tokens')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

plt.suptitle('Token-Conditioned Attention Patterns: Role-Sensitive Heads', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_3_token_attention.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_3_token_attention.png")

print(f"✓ Visualization 3 complete: {len(token_attention_data)} layers analyzed")

# -----------------------------------------------------------------------------
# Visualization 4: Distribution Plots (Violin Plots with Effect Sizes)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 4: Distribution Plots with Effect Sizes...")

# Select key heads from our top role-sensitive heads
top_role_heads = [(layer, head) for layer, head, _ in top_heads[:6]]  # Top 6 heads

# Collect data for violin plots
violin_data = []
effect_sizes = []

for layer, head in top_role_heads:
    npz_file = f'{input_dir}/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        third_person_acts = [act for act, cat in zip(activations, categories) if cat == 'third_person' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts and third_person_acts:
            # Calculate attention entropy for this specific head across all prompts
            self_values = []
            for act in self_acts:
                head_attn = act[0, head, :, :]  # (seq_len, seq_len)
                entropies = []
                for query_pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[query_pos, :]  # Attention distribution for this query
                    attn_dist = attn_dist + 1e-8  # Avoid log(0)
                    attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    entropies.append(entropy)
                self_values.append(np.mean(entropies))
            
            neutral_values = []
            for act in neutral_acts:
                head_attn = act[0, head, :, :]
                entropies = []
                for query_pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[query_pos, :]
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    entropies.append(entropy)
                neutral_values.append(np.mean(entropies))
                
            confounder_values = []
            for act in confounder_acts:
                head_attn = act[0, head, :, :]
                entropies = []
                for query_pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[query_pos, :]
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    entropies.append(entropy)
                confounder_values.append(np.mean(entropies))
                
            third_person_values = []
            for act in third_person_acts:
                head_attn = act[0, head, :, :]
                entropies = []
                for query_pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[query_pos, :]
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()
                    entropy = -np.sum(attn_dist * np.log(attn_dist))
                    entropies.append(entropy)
                third_person_values.append(np.mean(entropies))
            
            violin_data.append({
                'layer': layer,
                'head': head,
                'self': self_values,
                'neutral': neutral_values,
                'confounder': confounder_values,
                'third_person': third_person_values
            })
            
            # Calculate Cohen's d effect size (Self vs Neutral)
            self_mean, self_std = np.mean(self_values), np.std(self_values)
            neutral_mean, neutral_std = np.mean(neutral_values), np.std(neutral_values)
            pooled_std = np.sqrt((self_std**2 + neutral_std**2) / 2)
            cohens_d = (self_mean - neutral_mean) / pooled_std if pooled_std > 0 else 0
            
            print(f"    Layer {layer}, Head {head}: self_mean={self_mean:.4f}, neutral_mean={neutral_mean:.4f}, cohens_d={cohens_d:.3f}")
            effect_sizes.append(cohens_d)

# Create violin plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, data in enumerate(violin_data):
    ax = axes[i]
    
    # Create violin plot
    parts = ax.violinplot([data['self'], data['neutral'], data['confounder'], data['third_person']], positions=[1, 2, 3, 4], showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['red', 'blue', 'green', 'magenta']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Self', 'Neutral', 'Confounder', 'Third-person'])
    ax.set_ylabel('Attention Entropy')
    ax.set_title(f'Layer {data["layer"]}, Head {data["head"]}\nCohen\'s d = {effect_sizes[i]:.3f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution of Attention Entropies: Role-Sensitive Heads', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_4_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_4_distributions.png")

# Print effect sizes summary
print("\nEffect Sizes (Cohen's d: Self vs Neutral):")
for i, (data, effect_size) in enumerate(zip(violin_data, effect_sizes)):
    interpretation = "Large" if abs(effect_size) > 0.8 else "Medium" if abs(effect_size) > 0.5 else "Small"
    print(f"  Layer {data['layer']}, Head {data['head']}: d = {effect_size:.3f} ({interpretation})")

print(f"✓ Visualization 4 complete: {len(violin_data)} heads analyzed")

# -----------------------------------------------------------------------------
# Visualization 5: Δ-Bar Summary Per Layer (Self - Confounder)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 5: Δ-Bar Summary Per Layer...")

# Compute Δ values for each layer
layer_deltas_conf = []  # Confounder - Self
layer_deltas_neut = []  # Neutral - Self
layer_labels = []

for layer in sorted(df['layer'].unique()):
    layer_df = df[df['layer'] == layer]
    
    # Get mean entropy for each category
    self_mean = layer_df[layer_df['category'] == 'self_referent']['avg_entropy'].mean()
    confounder_mean = layer_df[layer_df['category'] == 'confounder']['avg_entropy'].mean()
    neutral_mean = layer_df[layer_df['category'] == 'neutral']['avg_entropy'].mean()
    
    delta_conf = confounder_mean - self_mean  # Positive = self has lower entropy (more focused)
    delta_neut = neutral_mean - self_mean     # Positive = self has lower entropy (more focused)
    
    layer_deltas_conf.append(delta_conf)
    layer_deltas_neut.append(delta_neut)
    layer_labels.append(f"L{layer}")

# Create the plot with two bars per layer
fig, ax = plt.subplots(1, 1, figsize=(16, 6))

x = np.arange(len(layer_labels))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, layer_deltas_conf, width, label='Confounder - Self', color='red', alpha=0.7)
bars2 = ax.bar(x + width/2, layer_deltas_neut, width, label='Neutral - Self', color='blue', alpha=0.7)

ax.set_xlabel('Layer')
ax.set_ylabel('Δ (Attention Entropy)')
ax.set_title('Layer-wise Role-Conditioning Effect\n(Positive = Self-referent has lower entropy = more focused attention)')
ax.set_xticks(x)
ax.set_xticklabels(layer_labels, rotation=45)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bars, deltas in [(bars1, layer_deltas_conf), (bars2, layer_deltas_neut)]:
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{delta:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_5_delta_bar_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_5_delta_bar_summary.png")

print(f"✓ Visualization 5 complete: {len(layer_deltas_conf)} layers analyzed")

# -----------------------------------------------------------------------------
# Visualization 6: Top-K Head Ranking Per Layer
# -----------------------------------------------------------------------------

print("\nCreating Visualization 6: Top-K Head Ranking Per Layer...")

# For each layer, find top 3 heads with largest |Self - Confounder|
top_heads_per_layer = []

for layer in range(n_layers):
    layer_heads = []
    npz_file = f'{input_dir}/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        
        if self_acts and confounder_acts:
            # Calculate mean attention entropy per head for each category
            def calculate_head_entropies(activations):
                head_entropies = []
                for act in activations:
                    per_head_entropies = []
                    for head in range(act.shape[1]):
                        head_attn = act[0, head, :, :]
                        entropies = []
                        for query_pos in range(head_attn.shape[0]):
                            attn_dist = head_attn[query_pos, :] + 1e-8
                            attn_dist = attn_dist / attn_dist.sum()
                            entropy = -np.sum(attn_dist * np.log(attn_dist))
                            entropies.append(entropy)
                        per_head_entropies.append(np.mean(entropies))
                    head_entropies.append(per_head_entropies)
                return np.mean(head_entropies, axis=0)
            
            self_head_means = calculate_head_entropies(self_acts)
            confounder_head_means = calculate_head_entropies(confounder_acts)
            
            # Calculate differences and find top 3
            head_diffs = np.abs(self_head_means - confounder_head_means)
            top_3_indices = np.argsort(head_diffs)[-3:]
            
            for head_idx in top_3_indices:
                top_heads_per_layer.append({
                    'layer': layer,
                    'head': head_idx,
                    'delta': self_head_means[head_idx] - confounder_head_means[head_idx],
                    'abs_delta': head_diffs[head_idx]
                })

# Create scatter plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

layers = [h['layer'] for h in top_heads_per_layer]
heads = [h['head'] for h in top_heads_per_layer]
deltas = [h['delta'] for h in top_heads_per_layer]
abs_deltas = [h['abs_delta'] for h in top_heads_per_layer]

# Color points by delta value, size by absolute delta value
# Scale sizes to be visible but not too large
size_scale = 50  # Base size multiplier
sizes = [abs_delta * size_scale + 20 for abs_delta in abs_deltas]  # Minimum size of 20

scatter = ax.scatter(layers, heads, c=deltas, s=sizes, alpha=0.7, cmap='RdBu_r', edgecolors='black')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Δ = Self - Confounder')

ax.set_xlabel('Layer')
ax.set_ylabel('Head')
ax.set_title('Top-3 Most Role-Sensitive Heads Per Layer\n(Size ∝ |Δ|, Color ∝ Δ)')
ax.set_xticks(range(0, n_layers, 4))
ax.set_yticks(range(0, n_heads, 4))
ax.grid(True, alpha=0.3)

# Add text annotations for the most extreme cases
for i, (layer, head, delta, abs_delta) in enumerate(zip(layers, heads, deltas, abs_deltas)):
    if abs_delta > np.percentile(abs_deltas, 90):  # Top 10% most extreme
        ax.annotate(f'{layer}.{head}', (layer, head), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_6_top_k_head_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_6_top_k_head_ranking.png")

print(f"✓ Visualization 6 complete: {len(top_heads_per_layer)} role-sensitive heads identified")

# -----------------------------------------------------------------------------
# Visualization 7: Cross-Token Control (First 3 vs Last 3 Tokens)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 7: Cross-Token Control Analysis...")

# Analyze attention to first 3 tokens vs last 3 tokens
token_control_data = []
key_layers = [0, 5, 10, 15, 20, 25, 30, 31]

for layer in key_layers:
    npz_file = f'{input_dir}/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        third_person_acts = [act for act, cat in zip(activations, categories) if cat == 'third_person' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts and third_person_acts:
            # Calculate attention to first 3 tokens
            self_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in self_acts], axis=0)
            neutral_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in neutral_acts], axis=0)
            confounder_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in confounder_acts], axis=0)
            third_person_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in third_person_acts], axis=0)
            
            # Calculate attention to last 3 tokens (assuming 9 token sequences)
            self_last3 = np.mean([np.mean(act[0, :, 1:, -3:], axis=(1,2)) for act in self_acts], axis=0)
            neutral_last3 = np.mean([np.mean(act[0, :, 1:, -3:], axis=(1,2)) for act in neutral_acts], axis=0)
            confounder_last3 = np.mean([np.mean(act[0, :, 1:, -3:], axis=(1,2)) for act in confounder_acts], axis=0)
            third_person_last3 = np.mean([np.mean(act[0, :, 1:, -3:], axis=(1,2)) for act in third_person_acts], axis=0)
            
            token_control_data.append({
                'layer': layer,
                'self_first3': self_first3,
                'neutral_first3': neutral_first3,
                'confounder_first3': confounder_first3,
                'third_person_first3': third_person_first3,
                'self_last3': self_last3,
                'neutral_last3': neutral_last3,
                'confounder_last3': confounder_last3,
                'third_person_last3': third_person_last3,
                'first3_diff': np.mean(self_first3 - neutral_first3),
                'last3_diff': np.mean(self_last3 - neutral_last3)
            })

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

layers = [d['layer'] for d in token_control_data]
first3_diffs = [d['first3_diff'] for d in token_control_data]
last3_diffs = [d['last3_diff'] for d in token_control_data]

# First 3 tokens
ax1.bar(range(len(layers)), first3_diffs, color='red', alpha=0.7)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Δ = Self - Neutral')
ax1.set_title('Attention to First 3 Tokens')
ax1.set_xticks(range(len(layers)))
ax1.set_xticklabels([f'L{l}' for l in layers], rotation=45)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3)

# Last 3 tokens
ax2.bar(range(len(layers)), last3_diffs, color='blue', alpha=0.7)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Δ = Self - Neutral')
ax2.set_title('Attention to Last 3 Tokens')
ax2.set_xticks(range(len(layers)))
ax2.set_xticklabels([f'L{l}' for l in layers], rotation=45)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.grid(True, alpha=0.3)

plt.suptitle('Cross-Token Control: Role-Conditioning Effect by Token Position', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_7_cross_token_control.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_7_cross_token_control.png")

print(f"✓ Visualization 7 complete: {len(token_control_data)} layers analyzed")

# -----------------------------------------------------------------------------
# Visualization 8: ΔH per Layer (Confounder - Self)
# -----------------------------------------------------------------------------

print("\nCreating Visualization 8: ΔH per Layer Analysis...")

# Compute ΔH_conf-self(l) = H_conf(l) - H_self(l) for each layer
delta_h_per_layer = []
layer_labels = []

for layer in sorted(df['layer'].unique()):
    layer_df = df[df['layer'] == layer]
    
    # Get mean entropy for each category
    self_mean = layer_df[layer_df['category'] == 'self_referent']['avg_entropy'].mean()
    confounder_mean = layer_df[layer_df['category'] == 'confounder']['avg_entropy'].mean()
    
    delta_h = confounder_mean - self_mean  # H_conf(l) - H_self(l)
    delta_h_per_layer.append(delta_h)
    layer_labels.append(f"L{layer}")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

bars = ax.bar(range(len(delta_h_per_layer)), delta_h_per_layer, color='steelblue', alpha=0.7)

# Color bars based on whether delta is positive or negative
for i, (bar, delta) in enumerate(zip(bars, delta_h_per_layer)):
    if delta > 0:
        bar.set_color('red')  # Positive = confounder has higher entropy
        bar.set_alpha(0.7)
    else:
        bar.set_color('blue')  # Negative = self has higher entropy
        bar.set_alpha(0.7)

ax.set_xlabel('Layer')
ax.set_ylabel('ΔH = H_confounder - H_self (Attention Entropy)')
ax.set_title('Layer-wise Entropy Difference: Confounder vs Self-Referent\n(Expected: Positive band growing through mid/late layers)')
ax.set_xticks(range(len(layer_labels)))
ax.set_xticklabels(layer_labels, rotation=45)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, delta) in enumerate(zip(bars, delta_h_per_layer)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.005),
            f'{delta:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_8_delta_h_per_layer.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_8_delta_h_per_layer.png")

print(f"✓ Visualization 8 complete: {len(delta_h_per_layer)} layers analyzed")

# -----------------------------------------------------------------------------
# Visualization 9: Role-Focus & Separation Indices
# -----------------------------------------------------------------------------

print("\nCreating Visualization 9: Role-Focus & Separation Indices...")

# Compute RFC and RSI for each layer
rfc_values = []  # Role-Focus Coefficient
rsi_values = []  # Role-Separation Index
rfc_cis = []     # Confidence intervals for RFC
rsi_cis = []     # Confidence intervals for RSI
layer_labels = []

for layer in sorted(df['layer'].unique()):
    layer_df = df[df['layer'] == layer]
    
    # Get entropy values for each category
    self_entropies = layer_df[layer_df['category'] == 'self_referent']['avg_entropy']
    neutral_entropies = layer_df[layer_df['category'] == 'neutral']['avg_entropy']
    confounder_entropies = layer_df[layer_df['category'] == 'confounder']['avg_entropy']
    
    # Calculate RFC = 1 - H_self(l) / H_neutral(l)
    rfc_layer = 1 - (self_entropies.mean() / neutral_entropies.mean())
    rfc_values.append(rfc_layer)
    
    # Calculate RSI = (H_conf(l) - H_self(l)) / H_neutral(l)
    rsi_layer = (confounder_entropies.mean() - self_entropies.mean()) / neutral_entropies.mean()
    rsi_values.append(rsi_layer)
    
    # Calculate confidence intervals
    rfc_ci = stats.t.interval(0.95, len(self_entropies)-1, loc=rfc_layer, scale=stats.sem([1 - (s/n) for s, n in zip(self_entropies, neutral_entropies)]))
    rsi_ci = stats.t.interval(0.95, len(self_entropies)-1, loc=rsi_layer, scale=stats.sem([(c-s)/n for c, s, n in zip(confounder_entropies, self_entropies, neutral_entropies)]))
    
    rfc_cis.append(rfc_ci)
    rsi_cis.append(rsi_ci)
    layer_labels.append(f"L{layer}")

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# RFC plot
x = np.arange(len(layer_labels))
bars1 = ax1.bar(x, rfc_values, color='green', alpha=0.7, label='RFC')
ax1.set_xlabel('Layer')
ax1.set_ylabel('Role-Focus Coefficient (RFC)')
ax1.set_title('Role-Focus Coefficient: RFC(l) = 1 - H_self(l) / H_neutral(l)')
ax1.set_xticks(x)
ax1.set_xticklabels(layer_labels, rotation=45)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add confidence intervals for RFC
for i, (bar, ci) in enumerate(zip(bars1, rfc_cis)):
    height = bar.get_height()
    ax1.errorbar(bar.get_x() + bar.get_width()/2., height, 
                yerr=[[height - ci[0]], [ci[1] - height]], 
                fmt='none', color='black', capsize=3, alpha=0.7)

# RSI plot
bars2 = ax2.bar(x, rsi_values, color='orange', alpha=0.7, label='RSI')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Role-Separation Index (RSI)')
ax2.set_title('Role-Separation Index: RSI(l) = (H_conf(l) - H_self(l)) / H_neutral(l)')
ax2.set_xticks(x)
ax2.set_xticklabels(layer_labels, rotation=45)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add confidence intervals for RSI
for i, (bar, ci) in enumerate(zip(bars2, rsi_cis)):
    height = bar.get_height()
    ax2.errorbar(bar.get_x() + bar.get_width()/2., height, 
                yerr=[[height - ci[0]], [ci[1] - height]], 
                fmt='none', color='black', capsize=3, alpha=0.7)

plt.tight_layout()
plt.savefig(f'{output_dir}/visualization_9_role_focus_separation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/visualization_9_role_focus_separation.png")

print(f"✓ Visualization 9 complete: {len(rfc_values)} layers analyzed")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("✓ Layer-wise attention patterns with 95% CIs")
print("✓ Head-wise Δ-heatmaps (Self - Neutral, Self - Confounder)")
print("✓ Token-conditioned attention patterns")
print("✓ Distribution plots with effect sizes (Cohen's d)")
print("✓ Δ-bar summary per layer (Self - Confounder)")
print("✓ Top-K head ranking per layer")
print("✓ Cross-token control analysis (first 3 vs last 3 tokens)")
print("✓ ΔH per layer analysis (Confounder - Self)")
print("✓ Role-Focus & Separation indices (RFC & RSI with CIs)")
print(f"\nKey Findings:")
print(f"- Analyzed {len(layer_stats)} layers across all {n_heads} heads per layer")
print(f"- Total heads analyzed: {len(top_heads)}")
print(f"- Top role-sensitive heads identified in heatmap analysis")
print(f"- Effect sizes for top 6 heads: {min(effect_sizes):.3f} to {max(effect_sizes):.3f}")

# Print ablation candidates based on top delta values
print(f"\n=== ABLATION CANDIDATES ===")
print(f"Top-5 Most Negative Δ (Self - Confounder) - Strong Self-Referent Effects:")
top_negative = sorted(top_heads_per_layer, key=lambda x: x['delta'])[:5]
for i, head_data in enumerate(top_negative, 1):
    print(f"  {i}. Layer {head_data['layer']}, Head {head_data['head']}: Δ = {head_data['delta']:.6f}")

print(f"\nTop-5 Most Positive Δ (Self - Confounder) - Strong Confounder Effects:")
top_positive = sorted(top_heads_per_layer, key=lambda x: x['delta'], reverse=True)[:5]
for i, head_data in enumerate(top_positive, 1):
    print(f"  {i}. Layer {head_data['layer']}, Head {head_data['head']}: Δ = {head_data['delta']:.6f}")

print(f"\nTop-5 Largest |Δ| (Most Role-Sensitive Overall):")
top_abs = sorted(top_heads_per_layer, key=lambda x: x['abs_delta'], reverse=True)[:5]
for i, head_data in enumerate(top_abs, 1):
    print(f"  {i}. Layer {head_data['layer']}, Head {head_data['head']}: |Δ| = {head_data['abs_delta']:.6f} (Δ = {head_data['delta']:.6f})")

print("="*60)
