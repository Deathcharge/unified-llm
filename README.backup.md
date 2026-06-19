# Unified-LLM: Multi-Provider LLM Orchestration

Unified-LLM is a sophisticated, production-grade LLM orchestration system that provides seamless integration with multiple LLM providers including Anthropic Claude, OpenAI GPT, Xai Grok, Ollama, and proprietary Helix models. It features advanced optimization techniques, training pipelines, and intelligent provider routing.

## Features

### Multi-Provider Support
- **Anthropic Claude** - State-of-the-art reasoning and analysis
- **OpenAI GPT** - Advanced language understanding
- **Xai Grok** - Real-time information processing
- **Ollama** - Local model inference
- **Custom Endpoints** - Bring your own LLM
- **Helix Proprietary** - CPU-optimized models for edge deployment

### Advanced Optimization
- **Quantization** - INT8 and other quantization techniques for reduced memory
- **Attention Mechanisms** - GQA, Nomad, and sliding window attention
- **Speculative Decoding** - Faster inference with draft models
- **KV Cache Management** - Efficient memory utilization
- **Multicore Parallel Processing** - Distributed inference
- **Streaming Support** - Real-time token streaming

### Agent Integration
- **Personality-Based Responses** - Customize agent behavior
- **System Prompt Customization** - Fine-tune agent characteristics
- **Agent-Specific Configurations** - Per-agent optimization

## Quick Start

### Installation

```bash
pip install unified-llm
```

### Basic Usage

```python
from unified_llm import LLMEngine, LLMProvider

# Initialize engine with Anthropic
engine = LLMEngine(provider=LLMProvider.ANTHROPIC)

# Generate response
response = engine.generate(
    prompt="What is machine learning?",
    max_tokens=1000,
    temperature=0.7
)

print(response)
```

### Multi-Provider with Fallback

```python
from unified_llm import LLMClient

client = LLMClient(
    providers=["openai", "anthropic", "ollama"],
    fallback_enabled=True
)

response = client.generate(
    prompt="Explain quantum computing",
    model="gpt-4-turbo-preview"
)
```

### Streaming Responses

```python
engine = LLMEngine(provider=LLMProvider.OPENAI)

for chunk in engine.stream_generate(
    prompt="Write a poem about AI",
    max_tokens=500
):
    print(chunk, end="", flush=True)
```

## Configuration

### Environment Variables

```bash
# Provider selection
export HELIX_LLM_PROVIDER=anthropic  # or openai, xai, ollama, custom, helix

# API Keys
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export XAI_API_KEY=your_key

# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_DEFAULT_MODEL=qwen2.5:7b-instruct-q5_k_m

# Custom Endpoint
export CUSTOM_LLM_ENDPOINT=http://your-endpoint

# Model Selection
export HELIX_LLM_MODEL=gpt-4-turbo-preview
```

## Advanced Features

### Quantization

```python
from unified_llm.providers import AdvancedQuantization

quantizer = AdvancedQuantization(method="int8", bits=8)
quantized_model = quantizer.quantize(model)
```

### Training Pipelines

```python
from unified_llm.providers import TrainingPipeline

pipeline = TrainingPipeline(
    model="helix-standard",
    num_epochs=3,
    batch_size=32,
    learning_rate=0.001
)

metrics = pipeline.train(training_data)
```

### Attention Optimization

```python
from unified_llm.providers import GQAAttention

attention = GQAAttention(
    num_heads=32,
    num_kv_heads=8,
    head_dim=128
)
```

## Architecture

Unified-LLM follows a modular architecture:

```
unified_llm/
├── client.py              # Railway client for K8s services
├── engine.py              # LLM Agent Engine
├── providers/
│   ├── core.py           # Core provider functionality
│   ├── models.py         # Model definitions
│   ├── router.py         # Request routing
│   ├── service.py        # Service layer
│   ├── inference.py      # Inference engine
│   ├── streaming.py      # Streaming support
│   ├── tokenizer.py      # Tokenization
│   ├── training.py       # Training logic
│   ├── training_pipeline.py  # Training pipeline
│   ├── quantization.py   # Quantization
│   ├── attention/        # Attention mechanisms
│   └── deployment/       # Deployment config
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/ -v
pytest tests/ --cov
pytest tests/ -m provider  # Run provider tests only
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

Dual licensed under Apache 2.0 and Proprietary. See LICENSE for details.
