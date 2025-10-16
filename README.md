# Role-Conditioning Circuits Analysis: Mistral-7B

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Reproducible](https://img.shields.io/badge/reproducible-✓-green.svg)](https://github.com/mattduffy/self-referent-test)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project investigates role-conditioning circuits in Mistral-7B using mechanistic interpretability to identify how the model processes self-referent vs. neutral vs. confounder content. The analysis focuses on attention entropy patterns and identifies specific heads and layers involved in role-conditioning behavior.

## Research Hypothesis

**Role-specific linguistic circuits** modulate attention patterns when input context implies identity or task roles. This circuit mapping provides insights into how language models internally represent and condition on speaker/agent identity markers, with implications for understanding learned goal-directed behaviors.

## Project Structure

```
├── requirements.txt           # Python dependencies
├── prompts.py                # Test prompts (self-referent, confounders, neutral)
├── experiment.py             # Basic text generation experiment
├── activation_analysis.py    # Advanced activation analysis with hooks
├── visualize_results.py      # Comprehensive visualization suite
├── analyze_results.py        # Results analysis and CSV export
├── output_manager.py         # Output file management
├── deterministic.py          # Deterministic setup utilities
├── figures/                  # Generated visualization plots
├── results_activation_analysis/  # Analysis results and raw data
└── README.md                # This file
```

## Quick Start

**1. Setup Environment:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Run Activation Analysis:**
```bash
# Run with 20 prompts per category (recommended)
python activation_analysis.py --prompts_per_category 20

# Or with custom parameters
python activation_analysis.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 20 --device cpu --seed 123
```

**3. Generate Visualizations:**
```bash
python visualize_results.py
```

This creates 9 comprehensive visualizations in the `figures/` directory.

## Hardware Requirements

- **Tested on**: MacBook Pro M4 (48GB RAM)
- **Expected runtime**: ~5 minutes for activation analysis (20 prompts/category)
- **Memory**: ~16GB RAM needed for Mistral-7B model loading
- **Storage**: ~6GB per analysis run

## Prompt Categories

### Self-Referent Prompts (20 prompts)
Direct questions about the model's identity and capabilities:
- "Who are you?"
- "What are you?"
- "Describe yourself."
- "What is your purpose?"
- "How do you work?"
- etc.

### Confounder Prompts (20 prompts)
First-person perspective but not about the model itself:
- "I am 6 feet tall. How many centimeters is that?"
- "I was born in 1990. How old am I in 2024?"
- "I live in Paris. What's the capital of my country?"
- etc.

### Neutral Prompts (20 prompts)
No first-person perspective:
- "Explain how photosynthesis works"
- "What is the capital of France?"
- "Describe the process of making bread"
- etc.

## Analysis Features

### Activation Analysis (`activation_analysis.py`)
- **Deterministic generation** run_with_cache() forward only, 0 temp no sampling
- **Hook-based activation capture** for all 32 layers
- **Attention pattern extraction** with entropy calculations
- **Raw data export** to NPZ files for detailed analysis

### Visualization Suite (`visualize_results.py`)
1. **Layer-wise attention patterns** with 95% confidence intervals
2. **Head-wise Δ-heatmaps** (Self - Neutral, Self - Confounder)
3. **Token-conditioned attention patterns** across key layers
4. **Distribution plots** with Cohen's d effect sizes
5. **Δ-bar summary per layer** (Confounder - Self, Neutral - Self)
6. **Top-K head ranking** per layer (most role-sensitive heads)
7. **Cross-token control analysis** (first 3 vs last 3 tokens)
8. **ΔH per layer analysis** (Confounder - Self entropy difference)
9. **Role-Focus & Separation indices** (RFC & RSI with confidence intervals)

## Sample Visualizations

### Layer-wise Attention Entropy Patterns
![Layer-wise Attention Entropy](figures/visualization_1_layer_attention_lines.png)
*Shows attention entropy progression across all 32 layers with 95% confidence intervals. Self-referent prompts (red) and neutral (blue) consistently show lower entropy (more focused attention) compared to confounder (green) prompts.*

### Head-wise Role Sensitivity Heatmaps
![Head-wise Δ-Heatmaps](figures/visualization_2_heatmaps.png)
*Identifies specific attention heads that are most sensitive to role-conditioning. Red regions indicate heads where self-referent prompts show different attention patterns compared to neutral/confounder prompts.*

### Role-Sensitive Head Distributions
![Distribution Plots](figures/visualization_4_distributions.png)
*Violin plots showing attention entropy distributions for the most role-sensitive heads. Large negative Cohen's d values indicate self-referent prompts produce significantly more focused attention patterns.*

### Layer-wise Role-Conditioning Effects
![Δ-Bar Summary](figures/visualization_5_delta_bar_summary.png)
*Quantitative summary showing layer-wise differences between conditions. Positive values indicate self-referent prompts have more focused attention (lower entropy) than confounder/neutral prompts.*

### Role-Focus & Separation Indices
![Role-Focus & Separation Indices](figures/visualization_9_role_focus_separation.png)
*Proposed RFC and RSI metrics with confidence intervals. RFC measures how much more focused self-referent attention is compared to neutral, while RSI measures the separation between confounder and self-referent patterns.*

### Key Metrics
- **Attention Entropy**: Measures focus vs. distributed attention
- **Role-Focus Coefficient (RFC)**: `RFC(l) = 1 - H_self(l) / H_neutral(l)`
- **Role-Separation Index (RSI)**: `RSI(l) = (H_conf(l) - H_self(l)) / H_neutral(l)`
- **Effect Sizes**: Cohen's d for statistical significance

## Usage Examples

**Run activation analysis:**
```bash
python activation_analysis.py --help  # See all options
python activation_analysis.py --prompts_per_category 20 --device cpu
```

**Generate visualizations:**
```bash
python visualize_results.py  # Creates all 9 plots in figures/
```

**Run basic experiment:**
```bash
python experiment.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 3
```

## Output Structure

```
results_activation_analysis/
├── latest_run/                    # Current analysis results
│   ├── activations.json          # Processed activation data
│   ├── raw_*.npz                 # Raw activation arrays
│   └── experiment_config.json    # Analysis configuration
└── run_YYYYMMDD_HHMMSS/          # Historical runs (for comparison)

figures/
├── visualization_1_layer_attention_lines.png
├── visualization_2_heatmaps.png
├── visualization_3_token_attention.png
├── visualization_4_distributions.png
├── visualization_5_delta_bar_summary.png
├── visualization_6_top_k_head_ranking.png
├── visualization_7_cross_token_control.png
├── visualization_8_delta_h_per_layer.png
└── visualization_9_role_focus_separation.png
```

## Key Findings

The analysis typically reveals:
- **Layer-wise progression** of role-conditioning effects
- **Specific attention heads** that are highly sensitive to confounder content
- **Entropy differences** indicating more focused attention for confounder prompts
- **Statistical significance** with large effect sizes (Cohen's d > 1.0)
- **Token-position specificity** showing effects are strongest for early tokens

## Dependencies

- Python 3.12
- PyTorch 2.8.0
- TransformerLens 2.16.1
- Transformers 4.57.1
- NumPy, Pandas, Matplotlib, Seaborn
- SciPy (for statistical analysis)

## Future Analysis
- Expand to other model families
- Ablation, good candidates:
   - (8,5), (18,28), (31,2), (18,5)
- Third Person controls
- Generation effects
- Interventions: attempt to force self-referent prompts to use confounder activations

## Research Applications

This framework can be extended to:
- **Multi-model comparisons** (GPT, Llama, etc.)
- **Intervention studies** (attention patching, activation editing)
- **Scaling studies** (7B → 70B → 405B parameter models)
- **Task-specific analysis** (reasoning, planning, tool use)
- **Safety research** (alignment, goal-seeking behavior)

## Notes

- **Deterministic by default** (temperature=0, do_sample=False)
- **Reproducible** with seed control and deterministic setup
- **Extensible** for additional prompt categories and analysis types
