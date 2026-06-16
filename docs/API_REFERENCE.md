# Unified-LLM API Reference

## Core Classes

### LLMProvider (Enum)

Supported LLM providers.

```python
class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    XAI = "xai"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    HELIX = "helix"
```

### LLMRequest

Request parameters for LLM generation.

```python
@dataclass
class LLMRequest:
    prompt: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] | None = None
    stream: bool = False
    user: str | None = None
```

### LLMClient

Railway client for communicating with Kubernetes LLM services.

```python
class LLMClient:
    def __init__(self, base_url: str, timeout: int = 30)
    async def generate(self, request: LLMRequest) -> dict
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]
    async def health_check(self) -> dict
```

**Methods:**

- `generate(request)` - Generate a single response
- `stream(request)` - Stream response tokens
- `health_check()` - Check service health

### LLMEngine

LLM Agent Engine with personality support.

```python
class LLMEngine:
    def __init__(self, provider: LLMProvider, model: str | None = None)
    def generate(self, prompt: str, **kwargs) -> str
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]
    def get_model_info(self) -> dict
```

**Methods:**

- `generate(prompt, **kwargs)` - Generate response
- `stream_generate(prompt, **kwargs)` - Stream response
- `get_model_info()` - Get model information

## Provider Modules

### advanced_quantization.py

Quantization techniques for model compression.

```python
class AdvancedQuantization:
    def __init__(self, method: str, bits: int, group_size: int = 128)
    def quantize(self, model) -> bytes
    def dequantize(self, data: bytes) -> list[float]
```

### gqa_attention.py

Grouped Query Attention mechanism.

```python
class GQAAttention:
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int)
    def forward(self, query, key, value) -> Tensor
```

### kv_cache_manager.py

KV cache management for efficient inference.

```python
class KVCacheManager:
    def get(self, key: str) -> Any | None
    def put(self, key: str, value: Any) -> None
    def clear(self) -> None
    def get_stats(self) -> dict
```

### training_pipeline.py

Training pipeline for model fine-tuning.

```python
class TrainingPipeline:
    def __init__(self, model: str, num_epochs: int, batch_size: int, learning_rate: float)
    def train(self, data: list[dict]) -> dict
    def evaluate(self, data: list[dict]) -> dict
    def save_checkpoint(self, path: str) -> None
```

### router.py

Request routing and provider selection.

```python
class Router:
    def route_request(self, request: LLMRequest) -> str
    def get_best_provider(self) -> str
    def get_fallback_providers(self) -> list[str]
```

### service.py

Service layer for request processing.

```python
class LLMService:
    def process_request(self, request: LLMRequest) -> dict
    def get_status(self) -> dict
    def shutdown(self) -> None
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HELIX_LLM_PROVIDER` | anthropic | LLM provider to use |
| `HELIX_LLM_MODEL` | claude-sonnet-4-6 | Model name |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `XAI_API_KEY` | - | Xai API key |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama endpoint |
| `OLLAMA_DEFAULT_MODEL` | qwen2.5:7b-instruct-q5_k_m | Ollama model |
| `CUSTOM_LLM_ENDPOINT` | - | Custom LLM endpoint |

## Examples

### Basic Generation

```python
from unified_llm import LLMEngine, LLMProvider

engine = LLMEngine(provider=LLMProvider.ANTHROPIC)
response = engine.generate("What is AI?")
print(response)
```

### Streaming

```python
engine = LLMEngine(provider=LLMProvider.OPENAI)
for chunk in engine.stream_generate("Explain ML", max_tokens=500):
    print(chunk, end="", flush=True)
```

### Multi-Provider

```python
from unified_llm import LLMClient

client = LLMClient(
    providers=["openai", "anthropic", "ollama"],
    fallback_enabled=True
)
response = client.generate(prompt="Hello", model="gpt-4")
```

### Training

```python
from unified_llm.providers import TrainingPipeline

pipeline = TrainingPipeline(
    model="helix-standard",
    num_epochs=3,
    batch_size=32,
    learning_rate=0.001
)

data = [
    {"prompt": "What is AI?", "response": "AI is..."},
    {"prompt": "Explain ML", "response": "ML is..."}
]

metrics = pipeline.train(data)
print(f"Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']}")
```

## Error Handling

```python
from unified_llm.core.exceptions import LLMProviderUnavailable, LLMServiceError

try:
    response = engine.generate("prompt")
except LLMProviderUnavailable:
    print("Provider unavailable, using fallback")
except LLMServiceError as e:
    print(f"Service error: {e}")
```

## Performance Tips

1. **Use Quantization** - Reduce memory with INT8 quantization
2. **Enable KV Cache** - Improve inference speed
3. **Use Streaming** - Stream responses for better UX
4. **Batch Requests** - Process multiple requests together
5. **Use Ollama Locally** - Reduce latency with local models

## Best Practices

1. Always set appropriate timeouts
2. Use fallback providers for reliability
3. Monitor provider health regularly
4. Cache responses when possible
5. Use streaming for long responses
6. Implement rate limiting
7. Log all requests for debugging
