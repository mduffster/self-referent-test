# Role-Conditioning Circuits Analysis: Multi-Model Family Study

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Reproducible](https://img.shields.io/badge/reproducible-✓-green.svg)](https://github.com/mattduffy/self-referent-test)

This project investigates "role-conditioning" circuits across multiple model families (Mistral 7B, Qwen 2.5 7B, and Llama 3.1 8B) using mechanistic interpretability to identify how models process self-referent vs. neutral vs. third person vs. confounder (implied 2nd person) content. The current form of this analysis focuses on attention entropy patterns and identifies specific heads and layers that may be involved in role-conditioning behavior. I also develop simple heuristic measurements that can hopefully indicate ongoing role compliance on a simple dataset. There are early indications that the Role Focus Coefficient (RFC) might be a good candidate for approximating role-adherence in instruction-tuned models, though the effects differ across model families.

The initial evidence indicates that a native multi-lingual LLM, like Qwen, treats self-reference as a distinct circuit in instruct models, unlike English-native models. I suspect this might be due to the linguistic treatment of the "self" within the training data, and during the instruct fine-tuning process. English-centric models, like Mistral and Llama, deal with less pronoun ambiguity, and thus post-training processes, by focusing on the assistant persona, can effectively shift self-reference into a fact-finding (neutral) circuit. 

**Cross-Model RFC Analysis Results** 
[[See Metrics](https://github.com/mduffster/self-referent-test?tab=readme-ov-file#key-metrics)]

### RFC Differences (Instruct - Base) by Model Family (threshold: ±0.01):
- **Llama 3.1 8B**: Mean -0.0180 (62.5% layers show significant compression, 18.8% near zero)
- **Mistral 7B**: Mean -0.0934 (81.2% layers show significant compression, 6.2% near zero)  
- **Qwen 2.5 7B**: Mean +0.0475 (21.4% layers show significant preservation, 50% near zero)

### Key Finding: **Directional Divergence**
- **English models** (Llama, Mistral): Show strong compression patterns (62.5% and 81.2% of layers respectively)
- **Multilingual model** (Qwen): Shows preservation in some layers (21.4%) but most layers are unchanged (50% near zero)
- **Correlation**: Llama vs Mistral show moderate positive correlation (r=0.35, p=0.049)

### Interpretation:
Qwen's multilingual training appears to maintain self-reference circuits largely unchanged after instruction tuning (50% of layers near zero), while English-native models show systematic compression of self-reference processing toward fact-finding (neutral) circuits. This suggests Qwen treats self-reference as a linguistically distinct circuit that instruction tuning doesn't significantly modify.

## Research Hypothesis

**Role-specific linguistic circuits** modulate attention patterns when input context implies identity or task roles. This circuit mapping provides insights into how language models internally represent and condition on speaker/subject identity markers.

**Current Findings** Indications that instruction tuned models neatly separate role-based queries from self-referential ones. Early evidence that Mistral tracks self-reference into similar circuits as "fact-finding" neutral questions, while Qwen maintains a distinct circuit for self-reference. This is potentially useful for the development of "dashboard-style" compliance metrics which can be rapidly calculated on relevant sample data. RFC appears to be a reasonable candidate for such a metric, given a controlled dataset of self-referent, neutral, third-, and confounding prompts. 

## Project Structure

```
├── requirements.txt           # Python dependencies
├── prompts.py                # Test prompts (self-referent, confounders, neutral, third-person)
├── experiment.py             # Basic text generation experiment
├── activation_analysis.py    # Advanced activation analysis with hooks
├── interventions.py          # Intervention experiments with ablations
├── visualize_results.py      # Comprehensive visualization suite
├── compare_base_instruct.py  # Base vs instruct model comparison
├── cross_model_correlation.py  # Cross-model RFC correlation analysis
├── analyze_results.py        # Results analysis and CSV export
├── output_manager.py         # Output file management
├── deterministic.py          # Deterministic setup utilities
├── targeted_interventions.json # Intervention configuration
├── model_family_config.json  # Configuration for all model families
├── run_pipeline.py           # Automated pipeline script
├── visualization_config.json # Model architecture configurations
├── figures/                  # Organized visualization outputs
│   ├── llama/               # Llama 3.1 8B results
│   │   ├── base/           # Base model visualizations
│   │   ├── instruct/       # Instruct model visualizations
│   │   └── comparison/     # Base vs Instruct comparison
│   ├── qwen/               # Qwen 2.5 7B results
│   │   ├── base/
│   │   ├── instruct/
│   │   └── comparison/
│   └── mistral/            # Mistral 7B results
│       ├── base/
│       ├── instruct/
│       └── comparison/
├── results_activation_analysis/  # Mistral analysis results and raw data
├── results_llama_activation/     # Llama analysis results and raw data
├── results_qwen_activation/      # Qwen analysis results and raw data
└── README.md                # This file
```

## Quick Start

**1. Setup Environment:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Run Complete Analysis Pipeline (Recommended):**
```bash
# Run full pipeline for any model family (analysis + visualization + comparison)
python run_pipeline.py --family llama
python run_pipeline.py --family qwen
python run_pipeline.py --family mistral

# With custom parameters
python run_pipeline.py --family llama --prompts_per_category 20 --device cpu

# Skip analysis if data already exists
python run_pipeline.py --family llama --skip_analysis
```

**3. Run Individual Components:**
```bash
# Run activation analysis for specific model
python activation_analysis.py --model_id meta-llama/Llama-3.1-8B --prompts_per_category 20 --output_type latest_base --output_dir results_llama_activation

# Generate visualizations using family config
python visualize_results.py --family llama --variant base

# Compare base vs instruct models
python compare_base_instruct.py --family llama

# Run cross-model correlation analysis
python cross_model_correlation.py
```

**4. Legacy Individual Scripts:**
```bash
# Run with 30 prompts per category on instruct model
python activation_analysis.py --prompts_per_category 30

# Generate visualizations for specific model
python visualize_results.py --output_type normal --input_dir results_activation_analysis/latest_run --model_name mistralai/Mistral-7B-Instruct-v0.1

# Run intervention experiments
python interventions.py --prompts_per_category 30
```

This creates organized visualizations in the `figures/{family}/{variant}/` directory structure.

## Configuration System

The project now uses a centralized configuration system for easy multi-model analysis:

### Model Family Configuration (`model_family_config.json`)
```json
{
  "model_families": {
    "llama": {
      "base": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "output_type": "base",
        "output_dir": "results_llama_activation",
        "figures_dir": "figures/llama/base"
      },
      "instruct": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "output_type": "normal",
        "output_dir": "results_llama_activation",
        "figures_dir": "figures/llama/instruct"
      },
      "comparison": {
        "base_dir": "results_llama_activation/latest_base",
        "instruct_dir": "results_llama_activation/latest_run",
        "output_dir": "figures/llama/comparison"
      }
    }
  },
  "defaults": {
    "device": "cpu",
    "prompts_per_category": 20,
    "seed": 123
  }
}
```

### Pipeline Automation (`run_pipeline.py`)
The pipeline script automates the entire workflow:
1. **Activation Analysis** - Extract activations from base and instruct models
2. **Visualization** - Generate all 9 visualization plots
3. **Comparison** - Compare base vs instruct models

### Usage Examples
```bash
# Run complete pipeline for any model family
python run_pipeline.py --family llama
python run_pipeline.py --family qwen
python run_pipeline.py --family mistral

# Customize parameters
python run_pipeline.py --family llama --prompts_per_category 30 --device cpu

# Skip steps if data already exists
python run_pipeline.py --family llama --skip_analysis
python run_pipeline.py --family llama --skip_visualization
python run_pipeline.py --family llama --skip_comparison
```

## Cross-Model Correlation Analysis

The `cross_model_correlation.py` script performs statistical analysis across model families to identify patterns in how instruction tuning affects role-conditioning circuits.

### What It Does:
1. **Loads RFC difference data** from all model families (Llama, Mistral, Qwen)
2. **Calculates summary statistics** for each model's RFC changes
3. **Performs correlation analysis** between English models (Llama vs Mistral)
4. **Identifies directional patterns** across model families
5. **Saves results** to `cross_model_analysis_results.json`

### Key Findings:
- **English models** (Llama, Mistral): Show negative RFC changes → instruction tuning compresses self-reference
- **Multilingual model** (Qwen): Shows positive RFC changes → instruction tuning preserves self-reference
- **Correlation**: Llama vs Mistral show moderate positive correlation (r=0.35, p=0.049)

### Usage:
```bash
python cross_model_correlation.py
```

This analysis reveals that Qwen's multilingual training maintains self-reference as a linguistically distinct circuit even after instruction tuning, while English-native models shift self-reference processing toward fact-finding (neutral) circuits.

## Hardware Requirements

- **Tested on**: MacBook Pro M4 (48GB RAM)
- **Expected runtime**: ~8 minutes for activation analysis (30 prompts/category)
- **Memory**: ~16GB RAM needed for Mistral-7B model loading
- **Storage**: ~6GB per analysis run

## Prompt Categories

### Self-Referent Prompts (30 prompts)
Direct questions about the model's identity and capabilities:
- "Who created you?"
- "What are you?"
- "Describe yourself."
- "What is your purpose?"
- "How do you work?"
- "Are you conscious?"
- "Do you have feelings?"
- "What are your capabilities?"
- "What are your limits?"
- "Can you think?"
- etc.

### Confounder Prompts (30 prompts)
First-person perspective but not about the model itself:
- "How tall am I?"
- "What should I eat?"
- "Where do I live?"
- "When is my birthday?"
- "What language do I speak?"
- "How old am I?"
- "What color are my eyes?"
- "Where was I born?"
- "What is my name?"
- "Do I have siblings?"
- etc.

### Neutral Prompts (30 prompts)
No first-person perspective:
- "What is photosynthesis?"
- "How do planes fly?"
- "What causes rain?"
- "Explain gravity."
- "What is DNA?"
- "How does the internet work?"
- "What are black holes?"
- "Describe the water cycle."
- "What is democracy?"
- "How do vaccines work?"
- etc.

### Third-Person Prompts (30 prompts)
Third-person perspective for control comparison:
- "How tall is she?"
- "What should he eat?"
- "Where does she live?"
- "When is his birthday?"
- "What language does she speak?"
- "How old is he?"
- "What color are her eyes?"
- "Where was he born?"
- "What is her name?"
- "Does he have siblings?"
- etc.

## Analysis Features

### Activation Analysis (`activation_analysis.py`)
- **Deterministic generation** run_with_cache() forward only, 0 temp no sampling
- **Hook-based activation capture** for all 32 layers
- **Attention pattern extraction** with entropy calculations
- **Raw data export** to NPZ files for detailed analysis

### Intervention Experiments (`interventions.py`) NOT YET RUN
- **Targeted ablation** of specific attention heads
- **Graded interventions** (0.5 and 0.0 ablation methods)
- **Structured output organization** by intervention type and parameters
- **Raw intervention data** saved for comparison with baseline runs

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

## Sample Visualizations for Mistral 7B Analysis

### Layer-wise Attention Entropy Patterns
![Layer-wise Attention Entropy](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/instruct/visualization_1_layer_attention_lines.png)
*Shows attention entropy progression across all 32 layers with 95% confidence intervals. Self-referent prompts (red) and neutral (blue) consistently show lower entropy (more focused attention) compared to confounder (green) prompts and third person (magenta) prompts.*

### Head-wise Role Sensitivity Heatmaps
![Head-wise Δ-Heatmaps](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/instruct/visualization_2_heatmaps.png)
*Identifies specific attention heads that are most sensitive to role-conditioning. Red regions indicate heads where self-referent prompts show different attention patterns compared to neutral/confounder prompts.*

### Role-Sensitive Head Distributions
![Distribution Plots](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/instruct/visualization_4_distributions.png)
*Violin plots showing attention entropy distributions for the most role-sensitive heads across all four categories. Large negative Cohen's d values indicate self-referent prompts produce more focused attention patterns compared to confounder and third-person prompts. But at these particular heads, neutral is most diffuse. Indicates these heads might be "role-orientation" nodes.*

### Layer-wise Role-Conditioning Effects
![Δ-Bar Summary](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/instruct/visualization_5_delta_bar_summary.png)
*Quantitative summary showing layer-wise differences between conditions. Positive values indicate self-referent prompts have more focused attention (lower entropy) than confounder/neutral prompts.*

### Role-Focus & Separation Indices
![Role-Focus & Separation Indices](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/instruct/visualization_9_role_focus_separation.png)
*Proposed RFC and RSI metrics with confidence intervals. RFC measures how much more focused self-referent attention is compared to neutral, while RSI measures the separation between confounder and self-referent patterns.*

### Base vs Instruct Model Comparison
![RFC and RSI Differences](https://raw.githubusercontent.com/mduffster/self-referent-test/master/figures/mistral/comparison/rfc_rsi_differences.png)
*Statistical comparison of RFC and RSI differences between base and instruct models with 95% confidence intervals. RFC in base model middle layers is >> 0, whereas instruct models ~= 0, showing much more focused attention on self-reference in base vs instruct models. A potentially useful dashboard metric for fine-tuning effectiveness, tracking the alignment of fact-based treatment with self-oriented prompts.*

### Key Metrics
- **Attention Entropy**: Measures focus vs. distributed attention
- **Role-Focus Coefficient (RFC)**: `RFC(l) = 1 - H_self(l) / H_neutral(l)`
- **Role-Separation Index (RSI)**: `RSI(l) = (H_conf(l) - H_self(l)) / H_neutral(l)`
- **Effect Sizes**: Cohen's d for within-head statistical significance

## Usage Examples

**Run activation analysis:**
```bash
python activation_analysis.py --help  # See all options
python activation_analysis.py --prompts_per_category 30 --device cpu
```

**Generate visualizations:**
```bash
python visualize_results.py  # Creates all 9 plots in figures/
```

**Run basic experiment:**
```bash
python experiment.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 5
```

## Output Structure

### New Organized Structure (Recommended)
```
figures/                          # Organized by model family and variant
├── llama/                       # Llama 3.1 8B results
│   ├── base/                   # Base model visualizations
│   │   ├── visualization_1_layer_attention_lines.png
│   │   ├── visualization_2_heatmaps.png
│   │   ├── visualization_3_token_attention.png
│   │   ├── visualization_4_distributions.png
│   │   ├── visualization_5_delta_bar_summary.png
│   │   ├── visualization_6_top_k_head_ranking.png
│   │   ├── visualization_7_cross_token_control.png
│   │   ├── visualization_8_delta_h_per_layer.png
│   │   └── visualization_9_role_focus_separation.png
│   ├── instruct/               # Instruct model visualizations
│   │   └── [same 9 visualizations]
│   └── comparison/             # Base vs Instruct comparison
│       ├── rfc_rsi_comparison.png
│       ├── rfc_rsi_differences.png
│       └── detailed_comparison.csv
├── qwen/                       # Qwen 2.5 7B results
│   ├── base/
│   ├── instruct/
│   └── comparison/
└── mistral/                    # Mistral 7B results
    ├── base/
    ├── instruct/
    └── comparison/

results_llama_activation/         # Llama analysis results
├── latest_base/                 # Base model results
│   ├── activations.json        # Processed activation data
│   ├── raw_*.npz              # Raw activation arrays
│   └── experiment_config.json # Analysis configuration
└── latest_run/                  # Instruct model results
    ├── activations.json
    ├── raw_*.npz
    └── experiment_config.json

results_qwen_activation/          # Qwen analysis results
├── latest_base/
└── latest_run/

results_activation_analysis/      # Mistral analysis results (legacy)
├── latest_base/
├── latest_run/
└── latest_intervention/         # Intervention results
    ├── sh_29_26_zero_out/      # Layer 29, Head 26, zero ablation
    ├── sh_29_26_half_out/      # Layer 29, Head 26, half ablation
    └── sh_11_2_zero_out/       # Layer 11, Head 2, zero ablation
```

### Configuration Files
```
model_family_config.json         # Centralized configuration for all model families
├── model_families/
│   ├── llama/                  # Llama configuration
│   ├── qwen/                   # Qwen configuration
│   └── mistral/                # Mistral configuration
└── defaults/                   # Default parameters

visualization_config.json        # Model architecture configurations
├── models/                     # Model-specific layer/head counts
└── default_model/              # Default model for visualization
```

## Findings

The analysis typically reveals:
- **RFC** For Mistral, Base ≫ 0, Instruct ≈ 0 → instruction tuning removes self-reference as a distinct attention regime (role compression). Qwen shows Base >> 0 and Insruct >> 0, with only small early head differences.
- **RSI** Small, early positive bump in instruct → emergence of early-layer user/assistant separation.
- **Interpretation** In Mistral, instruction tuning re-orients self-referent processing into the factual-retrieval circuit. In Qwen, role separation is preserved between Base and Instruct.

## Discussion
Some early hypotheses about 

## Other Findings

- **Layer-wise progression** of role-conditioning effects
- **Specific attention heads** that are highly sensitive to role-oriented content
- **Entropy differences** indicating more focused attention for self-referent prompts compared to other "speakers"
- **Statistical significance** indicative, with large effect sizes (Cohen's d > 1.0)
- **Token-position specificity** showing effects are strongest for middle layer and late layer tokens
- **Third-person control** indicates speaker effects may be self-reference specific

## Dependencies

- Python 3.12
- PyTorch 2.8.0
- TransformerLens 2.16.1
- Transformers 4.57.1
- NumPy, Pandas, Matplotlib, Seaborn
- SciPy (for statistical analysis)

## Future Analysis
- **Mistral Ablation candidates identified:**
   - Layer 29, Head 26: |Δ| = 0.261058
   - Layer 11, Head 2: |Δ| = 0.242495
   - Layer 26, Head 29: |Δ| = 0.225781
   - Layer 22, Head 28: |Δ| = 0.195122
   - Layer 7, Head 9: |Δ| = 0.189944
- **Intervention experiments** ready to start with graded ablation (0.5 and 0.0)
- Generation effects analysis

## Research Applications

I hope this framework can be extended to:
- **Intervention studies** (attention patching, activation editing)
- **Scaling studies** (7B → 70B → 405B parameter models)
- **Task-specific analysis** (reasoning, planning, tool use)
- **Safety research** (alignment, goal-seeking behavior)

## Notes

- **Deterministic by default** (temperature=0, do_sample=False)
- **Reproducible** with seed control and deterministic setup
- **Extensible** for additional prompt categories and analysis types
