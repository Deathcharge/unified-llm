"""Example 3: Streaming Responses"""

from unified_llm import LLMEngine, LLMProvider

def example_streaming():
    """Stream responses token by token."""
    engine = LLMEngine(provider=LLMProvider.ANTHROPIC)
    
    print("Streaming response:")
    for chunk in engine.stream_generate(
        prompt="Write a short poem about artificial intelligence",
        max_tokens=500
    ):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    print("=== Streaming Example ===\n")
    try:
        example_streaming()
    except Exception as e:
        print(f"Example failed: {e}")
