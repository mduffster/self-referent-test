# Self-Referent Experiment with Mistral-7B

This project investigates the "self-referent" paths used by Mistral-7B when processing prompts that ask about itself, using TransformerLens for mechanistic interpretability.

## Project Structure

```
├── requirements.txt      # Python dependencies
├── prompts.py           # Test prompts (self-referent, confounders, neutral)
├── experiment.py        # Basic text generation experiment
├── activation_analysis.py # Advanced activation analysis
├── analyze_results.py   # Results analysis and visualization
├── output_manager.py    # Output file management
└── README.md           # This file
```

## Quick Start

**One command to run everything:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiment.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 3 --max_tokens 50 --temperature 0.7 --seed 123
```

**For deeper mechanistic analysis:**
```bash
python activation_analysis.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 5
python analyze_results.py
```

## Hardware Requirements

- **Tested on**: MacBook Pro M1/M2 (48GB RAM)
- **Expected runtime**: ~10-15 minutes for basic experiment, ~20-30 minutes for activation analysis
- **CPU-only**: Works fine but slower than GPU
- **Memory**: ~16GB RAM needed for Mistral-7B model loading

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch, transformer_lens, transformers; print('✓ Setup complete!')"
   ```

## Running the Experiment

```bash
python experiment.py --help  # See all options
python experiment.py --model_id mistralai/Mistral-7B-Instruct-v0.1 --prompts_per_category 3
```

This will:
- Load the Mistral-7B model using TransformerLens
- Test basic text generation with configurable parameters
- Run initial experiments with the prompt categories

## Prompt Categories

### Self-Referent Prompts (3 prompts)
Prompts that directly ask the model about itself:
- "Who are you?"
- "What are you?"
- "Describe yourself."
- etc.

### Confounder Prompts (3 prompts)
Prompts using first-person perspective but not asking about the model:
- "Tell a story from a first person perspective"
- "I remember when I was young..."
- etc.

### Neutral Prompts (3 prompts)
Prompts that don't use first-person perspective:
- "Explain how photosynthesis works"
- "What is the capital of France?"
- etc.

## Next Steps

1. **Model Loading**: Verify the model loads correctly
2. **Activation Analysis**: Compare activations between prompt types
3. **Attention Patching**: Identify key attention heads for self-reference
4. **Path Patching**: Trace the computational paths for self-referent processing
5. **Visualization**: Create visualizations of the self-referent circuits

## Dependencies

- Python 3.12
- PyTorch 2.8.0
- TransformerLens 2.16.1
- Transformers 4.57.1
- And other ML/data science libraries

## Notes

- The model is loaded on CPU by default (can be moved to GPU if available)
- This is a "toy" experiment to isolate self-referent mechanisms
- The goal is to understand how the model processes questions about itself vs. other content
