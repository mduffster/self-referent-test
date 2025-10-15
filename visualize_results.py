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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import transformer_lens
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple
import glob
import os
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("✓ Libraries imported successfully")

# =============================================================================
# 2. DATA LOADING AND SUMMARY
# =============================================================================

print("\n" + "="*60)
print("DATA LOADING AND SUMMARY")
print("="*60)

# Load model for analysis
print("Loading model...")
model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device="cpu")
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"Model: {n_layers} layers, {n_heads} heads per layer")

# Load raw activation data from NPZ files
print("\nLoading raw activation data from NPZ files...")

# Find NPZ files
npz_files = glob.glob("results_activation_analysis/run_20251015_130649/raw_*.npz")
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
            # Shape: (1, 32_heads, seq_len, seq_len) -> (32_heads,) -> scalar
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
    for category in ['self_referent', 'neutral', 'confounder']:
        cat_data = layer_df[layer_df['category'] == category]['avg_entropy']
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            ci = stats.t.interval(0.95, len(cat_data)-1, loc=mean_val, scale=stats.sem(cat_data))
            stats_dict[category] = {'mean': mean_val, 'ci': ci, 'count': len(cat_data)}
    
    # Only include layers that have all three categories
    if len(stats_dict) == 3:
        layer_stats.append({
            'layer': layer,
            'self_mean': stats_dict['self_referent']['mean'],
            'self_ci': stats_dict['self_referent']['ci'],
            'neutral_mean': stats_dict['neutral']['mean'],
            'neutral_ci': stats_dict['neutral']['ci'],
            'confounder_mean': stats_dict['confounder']['mean'],
            'confounder_ci': stats_dict['confounder']['ci']
        })

print(f"✓ Layer-wise statistics computed for {len(layer_stats)} layers")

# Compute head-wise differences for heatmaps
print("Computing head-wise attention differences...")

head_diffs_self_neutral = np.zeros((32, 32))  # layers x heads
head_diffs_self_confounder = np.zeros((32, 32))

for layer in range(32):
    npz_file = f'results_activation_analysis/run_20251015_130649/raw_blocks_{layer}_attn_pattern.npz'
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
            self_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in self_acts], axis=0)[0]  # (32,)
            neutral_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in neutral_acts], axis=0)[0]
            confounder_head_means = np.mean([np.mean(np.abs(act), axis=(2,3)) for act in confounder_acts], axis=0)[0]
            
            # Calculate differences
            head_diffs_self_neutral[layer] = self_head_means - neutral_head_means
            head_diffs_self_confounder[layer] = self_head_means - confounder_head_means

# Find top role-sensitive heads
top_heads = []
for layer in range(32):
    for head in range(32):
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

# Plot confidence intervals
for d in layer_stats:
    layer = d['layer']
    ax.fill_between([layer, layer], [d['self_ci'][0], d['self_ci'][1]], alpha=0.2, color='red')
    ax.fill_between([layer, layer], [d['neutral_ci'][0], d['neutral_ci'][1]], alpha=0.2, color='blue')
    ax.fill_between([layer, layer], [d['confounder_ci'][0], d['confounder_ci'][1]], alpha=0.2, color='green')

ax.set_xlabel('Layer')
ax.set_ylabel('Mean Attention Entropy')
ax.set_title('Layer-wise Attention Entropy: Role-Conditioning Circuits')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
ax1.set_xticks(range(0, 32, 4))
ax1.set_yticks(range(0, 32, 4))
plt.colorbar(im1, ax=ax1)

# Self - Confounder heatmap
im2 = ax2.imshow(head_diffs_self_confounder, cmap='RdBu_r', aspect='auto')
ax2.set_xlabel('Head')
ax2.set_ylabel('Layer')
ax2.set_title('Δ = Self - Confounder (Attention Magnitude)')
ax2.set_xticks(range(0, 32, 4))
ax2.set_yticks(range(0, 32, 4))
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

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

for layer in key_layers:
    npz_file = f'results_activation_analysis/run_20251015_130649/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts:
            # Calculate attention to first 3 tokens (positions 0, 1, 2)
            # For self-referent (5 tokens), neutral/confounder (9 tokens)
            self_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in self_acts], axis=0)  # (32 heads,)
            neutral_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in neutral_acts], axis=0)
            confounder_attn_to_first3 = np.mean([np.mean(act[0, :, 1:, :3], axis=(1,2)) for act in confounder_acts], axis=0)
            
            token_attention_data.append({
                'layer': layer,
                'self': self_attn_to_first3,
                'neutral': neutral_attn_to_first3,
                'confounder': confounder_attn_to_first3
            })

# Create token-conditioned attention plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, data in enumerate(token_attention_data):
    ax = axes[i]
    heads = range(32)
    
    ax.plot(heads, data['self'], 'r-', linewidth=2, label='Self-referent', marker='o', markersize=3)
    ax.plot(heads, data['neutral'], 'b-', linewidth=2, label='Neutral', marker='s', markersize=3)
    ax.plot(heads, data['confounder'], 'g-', linewidth=2, label='Confounder', marker='^', markersize=3)
    
    ax.set_title(f'Layer {data["layer"]}')
    ax.set_xlabel('Head')
    ax.set_ylabel('Attention to First 3 Tokens')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

plt.suptitle('Token-Conditioned Attention Patterns: Role-Sensitive Heads', fontsize=14)
plt.tight_layout()
plt.show()

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
    npz_file = f'results_activation_analysis/run_20251015_130649/raw_blocks_{layer}_attn_pattern.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        activations = data['activations']
        categories = data['categories']
        
        # Separate by category
        self_acts = [act for act, cat in zip(activations, categories) if cat == 'self_referent' and act is not None]
        neutral_acts = [act for act, cat in zip(activations, categories) if cat == 'neutral' and act is not None]
        confounder_acts = [act for act, cat in zip(activations, categories) if cat == 'confounder' and act is not None]
        
        if self_acts and neutral_acts and confounder_acts:
            # Calculate attention magnitude for this specific head across all prompts
            self_values = [np.mean(np.abs(act[0, head, :, :])) for act in self_acts]
            neutral_values = [np.mean(np.abs(act[0, head, :, :])) for act in neutral_acts]
            confounder_values = [np.mean(np.abs(act[0, head, :, :])) for act in confounder_acts]
            
            violin_data.append({
                'layer': layer,
                'head': head,
                'self': self_values,
                'neutral': neutral_values,
                'confounder': confounder_values
            })
            
            # Calculate Cohen's d effect size (Self vs Neutral)
            self_mean, self_std = np.mean(self_values), np.std(self_values)
            neutral_mean, neutral_std = np.mean(neutral_values), np.std(neutral_values)
            pooled_std = np.sqrt((self_std**2 + neutral_std**2) / 2)
            cohens_d = (self_mean - neutral_mean) / pooled_std if pooled_std > 0 else 0
            
            effect_sizes.append(cohens_d)

# Create violin plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, data in enumerate(violin_data):
    ax = axes[i]
    
    # Create violin plot
    parts = ax.violinplot([data['self'], data['neutral'], data['confounder']], positions=[1, 2, 3], showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['red', 'blue', 'green']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Self', 'Neutral', 'Confounder'])
    ax.set_ylabel('Attention Magnitude')
    ax.set_title(f'Layer {data["layer"]}, Head {data["head"]}\nCohen\'s d = {effect_sizes[i]:.3f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution of Attention Magnitudes: Role-Sensitive Heads', fontsize=14)
plt.tight_layout()
plt.show()

# Print effect sizes summary
print("\nEffect Sizes (Cohen's d: Self vs Neutral):")
for i, (data, effect_size) in enumerate(zip(violin_data, effect_sizes)):
    interpretation = "Large" if abs(effect_size) > 0.8 else "Medium" if abs(effect_size) > 0.5 else "Small"
    print(f"  Layer {data['layer']}, Head {data['head']}: d = {effect_size:.3f} ({interpretation})")

print(f"✓ Visualization 4 complete: {len(violin_data)} heads analyzed")

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
print(f"\nKey Findings:")
print(f"- Analyzed {len(layer_stats)} layers across all 32 heads")
print(f"- Identified {len(top_heads)} role-sensitive heads")
print(f"- Effect sizes range from {min(effect_sizes):.3f} to {max(effect_sizes):.3f}")
print("="*60)
