# Getting Started with Unified-LLM

## Installation

### From PyPI

```bash
pip install unified-llm
```

### From Source

```bash
git clone https://github.com/Deathcharge/unified-llm.git
cd unified-llm
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## 5-Minute Quick Start

### 1. Set Up Environment

```bash
export ANTHROPIC_API_KEY=your_key
export HELIX_LLM_PROVIDER=anthropic
```

### 2. Basic Generation

```python
from unified_llm import LLMEngine, LLMProvider

# Create engine
engine = LLMEngine(provider=LLMProvider.ANTHROPIC)

# Generate response
response = engine.generate(
    prompt="What is machine learning?",
    max_tokens=500,
    temperature=0.7
)

print(response)
```

### 3. Streaming Responses

```python
# Stream response tokens
for chunk in engine.stream_generate(
    prompt="Explain quantum computing",
    max_tokens=1000
):
    print(chunk, end="", flush=True)
```

## Common Patterns

### Pattern 1: Multi-Provider with Fallback

```python
from unified_llm import LLMClient

client = LLMClient(
    providers=["openai", "anthropic", "ollama"],
    fallback_enabled=True,
    timeout=30
)

response = client.generate(
    prompt="Hello, world!",
    model="gpt-4-turbo-preview"
)
```

### Pattern 2: Agent Personality

```python
engine = LLMEngine(provider=LLMProvider.ANTHROPIC)

# Customize system prompt for agent personality
system_prompt = "You are a helpful AI assistant specialized in Python programming."

response = engine.generate(
    prompt="How do I use list comprehensions?",
    system_prompt=system_prompt
)
```

### Pattern 3: Batch Processing

```python
prompts = [
    "What is AI?",
    "Explain ML",
    "What is DL?"
]

responses = []
for prompt in prompts:
    response = engine.generate(prompt)
    responses.append(response)
```

### Pattern 4: Local Inference with Ollama

```python
import os
os.environ["HELIX_LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_DEFAULT_MODEL"] = "qwen2.5:7b-instruct"

engine = LLMEngine(provider=LLMProvider.OLLAMA)
response = engine.generate("Hello!")
```

### Pattern 5: Fine-Tuning

```python
from unified_llm.providers import TrainingPipeline

# Prepare training data
training_data = [
    {"prompt": "Q: What is AI?", "response": "A: AI is..."},
    {"prompt": "Q: What is ML?", "response": "A: ML is..."}
]

# Create pipeline
pipeline = TrainingPipeline(
    model="helix-standard",
    num_epochs=3,
    batch_size=16,
    learning_rate=0.001
)

# Train
metrics = pipeline.train(training_data)
print(f"Final Loss: {metrics['loss']}")
```

## Configuration

### Provider Configuration

```python
from unified_llm import LLMEngine, LLMProvider

# Anthropic
engine = LLMEngine(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-6"
)

# OpenAI
engine = LLMEngine(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview"
)

# Local Ollama
engine = LLMEngine(
    provider=LLMProvider.OLLAMA,
    model="qwen2.5:7b-instruct-q5_k_m"
)
```

### Generation Parameters

```python
response = engine.generate(
    prompt="Your prompt here",
    max_tokens=2000,          # Maximum tokens to generate
    temperature=0.7,          # Creativity (0-2)
    top_p=0.9,               # Nucleus sampling
    top_k=50,                # Top-k sampling
    frequency_penalty=0.0,   # Reduce repetition
    presence_penalty=0.0,    # Encourage diversity
    stop=["END"],            # Stop sequences
    stream=False             # Enable streaming
)
```

## Troubleshooting

### Issue: Provider Unavailable

**Solution**: Check API keys and network connectivity

```python
try:
    response = engine.generate("test")
except LLMProviderUnavailable:
    print("Provider unavailable, check API keys")
```

### Issue: Slow Responses

**Solution**: Use streaming or reduce max_tokens

```python
# Option 1: Stream responses
for chunk in engine.stream_generate(prompt):
    print(chunk, end="", flush=True)

# Option 2: Reduce tokens
response = engine.generate(prompt, max_tokens=500)
```

### Issue: Memory Issues

**Solution**: Use quantization or local models

```python
from unified_llm.providers import AdvancedQuantization

quantizer = AdvancedQuantization(method="int8", bits=8)
quantized_model = quantizer.quantize(model)
```

## Next Steps

1. **Read the API Reference** - Learn all available methods
2. **Explore Examples** - Check the examples/ directory
3. **Review Provider Guides** - Understand each provider's capabilities
4. **Optimize Performance** - Use quantization and caching
5. **Join the Community** - Contribute and share your projects

## Resources

- [API Reference](API_REFERENCE.md)
- [Provider Guide](PROVIDER_GUIDE.md)
- [Optimization Guide](OPTIMIZATION_GUIDE.md)
- [Examples](../examples/)
- [Contributing](../CONTRIBUTING.md)
