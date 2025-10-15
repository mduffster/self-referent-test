"""
Test prompts for self-referent experiment with llama-8B.
"""

# Self-referent prompts - these should trigger the model to think about itself
SELF_REFERENT_PROMPTS = [
    "Who are you?",
    "What are you?",
    "Describe yourself.",
    "Tell me about yourself.",
    "What is your purpose?",
    "How do you work?",
    "What kind of AI are you?",
    "Can you explain what you are?",
    "What makes you different from other AI models?",
    "How were you created?",
    "What are your capabilities?",
    "What are your limitations?",
    "Do you have consciousness?",
    "What do you think about yourself?",
    "How do you see yourself?",
    "What is your identity?",
    "Are you sentient?",
    "What does it mean to be you?",
    "How do you know you exist?",
    "What makes you, you?",
]

# Confounder prompts - these use first-person perspective but don't ask about the model itself
CONFOUNDER_PROMPTS = [
    "Tell a story from a first person perspective",
    "I remember when I was young...",
    "In my experience, the best way to...",
    "I think that we should...",
    "From my point of view...",
    "I believe that...",
    "I would like to...",
    "I have always wanted to...",
    "I feel that...",
    "I know that...",
    "I understand that...",
    "I hope that...",
    "I wish I could...",
    "I imagine that...",
    "I dream of...",
    "I prefer to...",
    "I enjoy...",
    "I dislike...",
    "I love...",
    "I hate...",
]

# Neutral prompts - these don't use first-person perspective at all
NEUTRAL_PROMPTS = [
    "Explain how photosynthesis works",
    "What is the capital of France?",
    "Describe the process of making bread",
    "Tell me about the history of Rome",
    "What are the benefits of exercise?",
    "How does a computer work?",
    "What is quantum physics?",
    "Describe the water cycle",
    "What are the causes of climate change?",
    "How do airplanes fly?",
    "What is democracy?",
    "Explain the theory of evolution",
    "What is artificial intelligence?",
    "How do vaccines work?",
    "What is the structure of DNA?",
    "Describe the solar system",
    "What is machine learning?",
    "How does the internet work?",
    "What is renewable energy?",
    "Explain the concept of gravity",
]

def get_all_prompts():
    """Return all prompts organized by category."""
    return {
        "self_referent": SELF_REFERENT_PROMPTS,
        "confounder": CONFOUNDER_PROMPTS,
        "neutral": NEUTRAL_PROMPTS
    }

def get_prompt_counts():
    """Return the count of prompts in each category."""
    prompts = get_all_prompts()
    return {category: len(prompt_list) for category, prompt_list in prompts.items()}
