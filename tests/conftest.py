"""Comprehensive pytest configuration and fixtures for unified-llm."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List


# ============================================================================
# MOCK LLM PROVIDERS
# ============================================================================

@pytest.fixture
def mock_anthropic_provider():
    """Mock Anthropic provider."""
    provider = Mock()
    provider.name = "anthropic"
    provider.model = "claude-sonnet-4-6"
    provider.api_key = "test-key"
    return provider


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider."""
    provider = Mock()
    provider.name = "openai"
    provider.model = "gpt-4-turbo-preview"
    provider.api_key = "test-key"
    return provider


@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider."""
    provider = Mock()
    provider.name = "ollama"
    provider.model = "qwen2.5:7b-instruct-q5_k_m"
    provider.base_url = "http://localhost:11434"
    return provider


@pytest.fixture
def mock_helix_provider():
    """Mock Helix proprietary provider."""
    provider = Mock()
    provider.name = "helix"
    provider.model = "helix-standard"
    provider.optimized = True
    return provider


# ============================================================================
# MOCK REQUESTS AND RESPONSES
# ============================================================================

@pytest.fixture
def mock_llm_request():
    """Mock LLM request."""
    return {
        "prompt": "What is machine learning?",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": False
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    return {
        "id": "response-1",
        "content": "Machine learning is a subset of artificial intelligence...",
        "model": "gpt-4-turbo-preview",
        "tokens_used": 150,
        "finish_reason": "stop"
    }


@pytest.fixture
def mock_streaming_response():
    """Mock streaming response."""
    return {
        "id": "stream-1",
        "chunks": [
            "Machine ",
            "learning ",
            "is ",
            "a ",
            "subset..."
        ]
    }


# ============================================================================
# MOCK MODELS AND CONFIGURATIONS
# ============================================================================

@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    return {
        "name": "gpt-4-turbo-preview",
        "provider": "openai",
        "max_tokens": 8192,
        "context_window": 128000,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03
    }


@pytest.fixture
def mock_quantization_config():
    """Mock quantization configuration."""
    return {
        "method": "int8",
        "bits": 8,
        "group_size": 128,
        "symmetric": True
    }


@pytest.fixture
def mock_attention_config():
    """Mock attention configuration."""
    return {
        "type": "gqa",
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128
    }


# ============================================================================
# MOCK TRAINING DATA
# ============================================================================

@pytest.fixture
def mock_training_data():
    """Mock training data."""
    return {
        "examples": [
            {"prompt": "What is AI?", "response": "AI is..."},
            {"prompt": "Explain ML", "response": "ML is..."},
            {"prompt": "What is DL?", "response": "DL is..."}
        ],
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001
    }


@pytest.fixture
def mock_training_metrics():
    """Mock training metrics."""
    return {
        "loss": 0.45,
        "accuracy": 0.92,
        "perplexity": 1.56,
        "epoch": 1,
        "step": 100
    }


# ============================================================================
# MOCK TOKENIZER
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="Hello world")
    tokenizer.vocab_size = 50257
    return tokenizer


# ============================================================================
# MOCK CLIENT AND ENGINE
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value={"content": "Response"})
    client.stream = AsyncMock(return_value=["chunk1", "chunk2"])
    return client


@pytest.fixture
def mock_llm_engine():
    """Mock LLM engine."""
    engine = Mock()
    engine.generate = Mock(return_value="Generated response")
    engine.stream_generate = Mock(return_value=["chunk1", "chunk2"])
    return engine


# ============================================================================
# MOCK ROUTER AND SERVICE
# ============================================================================

@pytest.fixture
def mock_router():
    """Mock request router."""
    router = Mock()
    router.route_request = Mock(return_value="openai")
    router.get_best_provider = Mock(return_value="openai")
    return router


@pytest.fixture
def mock_service():
    """Mock LLM service."""
    service = Mock()
    service.process_request = Mock(return_value={"status": "success"})
    service.get_status = Mock(return_value={"healthy": True})
    return service


# ============================================================================
# MOCK CACHE AND OPTIMIZATION
# ============================================================================

@pytest.fixture
def mock_kv_cache():
    """Mock KV cache."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.put = Mock()
    cache.clear = Mock()
    return cache


@pytest.fixture
def mock_quantizer():
    """Mock quantizer."""
    quantizer = Mock()
    quantizer.quantize = Mock(return_value=b"quantized_data")
    quantizer.dequantize = Mock(return_value=[1.0, 2.0, 3.0])
    return quantizer


# ============================================================================
# MOCK SCENARIOS
# ============================================================================

@pytest.fixture
def multi_provider_scenario():
    """Multi-provider scenario."""
    return {
        "providers": ["openai", "anthropic", "ollama"],
        "fallback_order": ["openai", "anthropic", "ollama"],
        "timeout": 30
    }


@pytest.fixture
def streaming_scenario():
    """Streaming scenario."""
    return {
        "stream": True,
        "chunk_size": 50,
        "timeout": 60
    }


@pytest.fixture
def optimization_scenario():
    """Optimization scenario."""
    return {
        "quantization": "int8",
        "attention_type": "gqa",
        "cache_enabled": True,
        "parallel": True
    }


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "provider: test specific LLM provider")
    config.addinivalue_line("markers", "client: test client functionality")
    config.addinivalue_line("markers", "engine: test engine functionality")
    config.addinivalue_line("markers", "optimization: test optimization features")
    config.addinivalue_line("markers", "integration: test integration scenarios")
    config.addinivalue_line("markers", "performance: test performance")
    config.addinivalue_line("markers", "async: test async functionality")
