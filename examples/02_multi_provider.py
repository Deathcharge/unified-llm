"""Example 2: Multi-Provider with Fallback"""

from unified_llm import LLMClient

def example_multi_provider():
    """Demonstrate multi-provider with fallback."""
    client = LLMClient(
        providers=["openai", "anthropic", "ollama"],
        fallback_enabled=True,
        timeout=30
    )
    
    response = client.generate(
        prompt="Explain the concept of neural networks",
        model="gpt-4-turbo-preview",
        max_tokens=1000
    )
    
    print("Multi-Provider Response:")
    print(response)

if __name__ == "__main__":
    print("=== Multi-Provider Example ===\n")
    try:
        example_multi_provider()
    except Exception as e:
        print(f"Example failed: {e}")
