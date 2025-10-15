# Self-Referent Experiment with Llama-8B

This project investigates the "self-referent" paths used by Llama-8B when processing prompts that ask about itself, using TransformerLens for mechanistic interpretability.

## Project Structure

```
├── venv/                 # Python virtual environment
├── requirements.txt      # Python dependencies
├── prompts.py           # Test prompts (self-referent, confounders, neutral)
├── experiment.py        # Main experiment script
└── README.md           # This file
```

## Setup

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; import transformer_lens; print('Setup complete!')"
   ```

## Running the Experiment

```bash
python experiment.py
```

This will:
- Load the Mistral-7B model using TransformerLens
- Test basic text generation
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
