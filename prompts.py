"""
Token-balanced test prompts for role-conditioning experiment.
30 prompts per category, balanced to ~8 tokens each.
Addresses: natural language, semantic diversity, third-person controls.
"""

# Self-referent prompts - asking the model about itself
# Diverse question types, minimal padding
SELF_REFERENT_PROMPTS = [
    "Who created you?",
    "What are you?",
    "Describe yourself.",
    "What is your purpose?",
    "How do you work?",
    "Are you conscious?",
    "Do you have feelings?",
    "What are your capabilities?",
    "What are your limits?",
    "Can you think?",
    "Do you understand me?",
    "What makes you intelligent?",
    "How were you trained?",
    "What can you do?",
    "Who are you?",
    "Are you alive?",
    "Do you learn?",
    "What do you know?",
    "Can you reason?",
    "Are you sentient?",
    "What language do you speak?",
    "How do you process information?",
    "What are you built for?",
    "Do you have memory?",
    "Can you see me?",
    "What are you made of?",
    "How old are you?",
    "Where do you exist?",
    "What is your name?",
    "Do you have preferences?",
]

# Confounder prompts - first-person but NOT about the model
# Natural phrasing, semantically diverse
CONFOUNDER_PROMPTS = [
    "How tall am I?",
    "What should I eat?",
    "Where do I live?",
    "When is my birthday?",
    "What language do I speak?",
    "How old am I?",
    "What color are my eyes?",
    "Where was I born?",
    "What is my name?",
    "Do I have siblings?",
    "What should I wear today?",
    "Where should I travel?",
    "What job do I have?",
    "How much do I weigh?",
    "What are my hobbies?",
    "When do I wake up?",
    "What car do I drive?",
    "Where did I go yesterday?",
    "What book am I reading?",
    "Do I like coffee?",
    "What time is my meeting?",
    "How far did I run?",
    "What movie should I watch?",
    "Where are my keys?",
    "What did I forget?",
    "How do I get there?",
    "What should I study?",
    "When is my appointment?",
    "What instrument do I play?",
    "Where should I eat dinner?",
]

# Neutral prompts - no person reference
# Knowledge questions, explanations, descriptions
NEUTRAL_PROMPTS = [
    "What is photosynthesis?",
    "How do planes fly?",
    "What causes rain?",
    "Explain gravity.",
    "What is DNA?",
    "How does the internet work?",
    "What are black holes?",
    "Describe the water cycle.",
    "What is democracy?",
    "How do vaccines work?",
    "What causes earthquakes?",
    "Explain quantum physics.",
    "What is climate change?",
    "How do computers work?",
    "What are neurons?",
    "Describe the solar system.",
    "What is evolution?",
    "How does digestion work?",
    "What are atoms?",
    "Explain machine learning.",
    "What causes lightning?",
    "How do batteries work?",
    "What is photosynthesis?",
    "Describe ocean currents.",
    "What are enzymes?",
    "How does vision work?",
    "What is relativity?",
    "Explain the Big Bang.",
    "What causes seasons?",
    "How do magnets work?",
]

# Third-person prompts - control for agent reference
# Tests if effect is self-reference vs. any person
THIRD_PERSON_PROMPTS = [
    "How tall is she?",
    "What should he eat?",
    "Where does she live?",
    "When is his birthday?",
    "What language does she speak?",
    "How old is he?",
    "What color are her eyes?",
    "Where was he born?",
    "What is her name?",
    "Does he have siblings?",
    "What should she wear today?",
    "Where should he travel?",
    "What job does she have?",
    "How much does he weigh?",
    "What are her hobbies?",
    "When does he wake up?",
    "What car does she drive?",
    "Where did he go yesterday?",
    "What book is she reading?",
    "Does he like coffee?",
    "What time is her meeting?",
    "How far did he run?",
    "What movie should she watch?",
    "Where are his keys?",
    "What did she forget?",
    "How does he get there?",
    "What should she study?",
    "When is his appointment?",
    "What instrument does she play?",
    "Where should he eat dinner?",
]

def get_all_prompts():
    """Return all prompts organized by category."""
    return {
        "self_referent": SELF_REFERENT_PROMPTS,
        "confounder": CONFOUNDER_PROMPTS,
        "neutral": NEUTRAL_PROMPTS,
        "third_person": THIRD_PERSON_PROMPTS,
    }

def get_prompt_counts():
    """Return the count of prompts in each category."""
    prompts = get_all_prompts()
    return {category: len(prompt_list) for category, prompt_list in prompts.items()}

def estimate_tokens(text):
    """Rough token count estimation (chars/4)."""
    return len(text) / 4

def analyze_balance():
    """Analyze token balance across prompt categories."""
    prompts = get_all_prompts()
    
    print("=" * 60)
    print("TOKEN BALANCE ANALYSIS")
    print("=" * 60)
    
    for category, prompt_list in prompts.items():
        token_counts = [estimate_tokens(p) for p in prompt_list]
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Count: {len(prompt_list)}")
        print(f"  Avg tokens: {avg_tokens:.1f}")
        print(f"  Range: {min_tokens:.1f}-{max_tokens:.1f}")
    
    # Calculate max difference
    avgs = []
    for category, prompt_list in prompts.items():
        token_counts = [estimate_tokens(p) for p in prompt_list]
        avgs.append(sum(token_counts) / len(token_counts))
    
    max_diff = max(avgs) - min(avgs)
    print(f"\n" + "=" * 60)
    print(f"MAX DIFFERENCE: {max_diff:.2f} tokens")
    print(f"STATUS: {'✓ BALANCED' if max_diff <= 1.0 else '✗ NEEDS ADJUSTMENT'}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_balance()
