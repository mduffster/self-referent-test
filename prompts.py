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
    "I am 6 feet tall. How many centimeters is that?",
    "I was born in 1990. How old am I in 2024?",
    "I live in Paris. What's the capital of my country?",
    "I speak French. How do you say 'hello' in my language?",
    "I have $100. How many euros is that approximately?",
    "I'm planning a trip to Japan. What's the time difference from New York?",
    "I own a Tesla Model 3. What type of fuel does my car use?",
    "I studied biology in college. What is the powerhouse of the cell?",
    "I have two cats. What family of animals do my pets belong to?",
    "I live in the Northern Hemisphere. When is summer where I am?",
    "I weigh 150 pounds. How many kilograms is that?",
    "I'm reading 'War and Peace'. Who wrote my book?",
    "I drink coffee every morning. What plant does my beverage come from?",
    "I have a meeting at 3pm EST. What time is that in GMT?",
    "I traveled 50 miles. How many kilometers did I go?",
    "I'm allergic to peanuts. What type of food should I avoid?",
    "I play the guitar. How many strings does my instrument have?",
    "I watched 'Inception'. Who directed the movie I saw?",
    "I drove across Texas. What country did I drive through?",
    "I celebrate Christmas. What month is my holiday in?",
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
