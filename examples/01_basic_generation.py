"""Example 1: Basic Generation with Different Providers"""

from unified_llm import LLMEngine, LLMProvider

def example_anthropic():
    """Generate with Anthropic Claude."""
    engine = LLMEngine(provider=LLMProvider.ANTHROPIC)
    response = engine.generate(
        prompt="What is machine learning?",
        max_tokens=500,
        temperature=0.7
    )
    print("Anthropic Response:")
    print(response)
    print()

def example_openai():
    """Generate with OpenAI GPT."""
    engine = LLMEngine(provider=LLMProvider.OPENAI)
    response = engine.generate(
        prompt="Explain quantum computing",
        max_tokens=500,
        temperature=0.7
    )
    print("OpenAI Response:")
    print(response)
    print()

def example_ollama():
    """Generate with local Ollama."""
    engine = LLMEngine(provider=LLMProvider.OLLAMA)
    response = engine.generate(
        prompt="What is artificial intelligence?",
        max_tokens=300
    )
    print("Ollama Response:")
    print(response)
    print()

if __name__ == "__main__":
    print("=== Basic Generation Examples ===\n")
    
    try:
        example_anthropic()
    except Exception as e:
        print(f"Anthropic example failed: {e}\n")
    
    try:
        example_openai()
    except Exception as e:
        print(f"OpenAI example failed: {e}\n")
    
    try:
        example_ollama()
    except Exception as e:
        print(f"Ollama example failed: {e}\n")
