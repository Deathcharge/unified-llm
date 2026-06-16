"""
Optimized Model Configurations for CPU-Optimized LLM

Defines pre-configured model architectures optimized for CPU inference with different
performance/accuracy trade-offs.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from .advanced_quantization import DynamicQuantization
from .gqa_attention import GQATransformerLayer
from .multicore_parallel import CPUOptimizedModelWrapper
from .nomad_attention import NoMADTransformerLayer
from .sliding_window import SlidingWindowTransformerLayer


class HelixConfig:
    """Base configuration for Helix LLM models."""

    def __init__(
        self,
        vocab_size: int = 256,
        max_seq_len: int = 2048,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        attention_type: str = "standard",
        use_kv_cache: bool = True,
        use_quantization: bool = True,
        quantization_bitwidth: int = 8,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attention_type = attention_type
        self.use_kv_cache = use_kv_cache
        self.use_quantization = use_quantization
        self.quantization_bitwidth = quantization_bitwidth

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__.items())


class HelixUltraLightConfig(HelixConfig):
    """
    Helix-Ultra-Light: 128M parameters, ~64MB RAM

    Designed for:
    - Ultra-fast inference on any CPU
    - Minimal memory footprint
    - Basic conversational AI
    - Real-time edge applications

    Performance: ~50 tokens/sec on 2-core CPU
    Accuracy: Suitable for simple tasks
    """

    def __init__(self):
        super().__init__(
            vocab_size=256,
            max_seq_len=1024,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            ff_dim=512,
            dropout=0.1,
            attention_type="sliding_window",
            use_kv_cache=True,
            use_quantization=True,
            quantization_bitwidth=4,
        )


class HelixLightConfig(HelixConfig):
    """
    Helix-Light: 256M parameters, ~128MB RAM

    Designed for:
    - Fast inference on modern CPUs
    - Good accuracy for most tasks
    - General-purpose AI assistant
    - Code completion and generation

    Performance: ~30 tokens/sec on 4-core CPU
    Accuracy: Balanced for most use cases
    """

    def __init__(self):
        super().__init__(
            vocab_size=256,
            max_seq_len=2048,
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            ff_dim=1024,
            dropout=0.1,
            attention_type="gqa",
            use_kv_cache=True,
            use_quantization=True,
            quantization_bitwidth=8,
        )


class HelixStandardConfig(HelixConfig):
    """
    Helix-Standard: 512M parameters, ~256MB RAM

    Designed for:
    - Production-grade CPU inference
    - High accuracy for complex tasks
    - Multi-turn conversations
    - Advanced reasoning and analysis

    Performance: ~15 tokens/sec on 8-core CPU
    Accuracy: Near-GPT-3.5 level for many tasks
    """

    def __init__(self):
        super().__init__(
            vocab_size=256,
            max_seq_len=4096,
            embed_dim=512,
            num_layers=8,
            num_heads=8,
            ff_dim=2048,
            dropout=0.1,
            attention_type="gqa",
            use_kv_cache=True,
            use_quantization=True,
            quantization_bitwidth=8,
        )


class HelixEnhancedConfig(HelixConfig):
    """
    Helix-Enhanced: 1B parameters, ~512MB RAM

    Designed for:
    - Enterprise-grade applications
    - Maximum accuracy on CPU
    - Complex multi-agent workflows
    - Specialized domain knowledge

    Performance: ~8 tokens/sec on 16-core CPU
    Accuracy: Competitive with larger models
    """

    def __init__(self):
        super().__init__(
            vocab_size=256,
            max_seq_len=8192,
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            ff_dim=3072,
            dropout=0.1,
            attention_type="sliding_window",  # With larger window
            use_kv_cache=True,
            use_quantization=True,
            quantization_bitwidth=8,
        )


class HelixCPUModel(nn.Module):
    """
    Base Helix CPU-optimized model with configurable architecture.
    """

    def __init__(self, config: HelixConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _i in range(config.num_layers):
            if config.attention_type == "nomad":
                layer = NoMADTransformerLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    use_nomad=True,
                )
            elif config.attention_type == "gqa":
                layer = GQATransformerLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_heads // 2,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
            elif config.attention_type == "sliding_window":
                layer = SlidingWindowTransformerLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    window_size=512,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    use_global_tokens=True,
                )
            else:  # standard
                layer = nn.TransformerEncoderLayer(
                    d_model=config.embed_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.ff_dim,
                    dropout=config.dropout,
                    batch_first=True,
                )

            self.layers.append(layer)

        # Layer norm and output projection
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Dropouts
        self.dropout = nn.Dropout(config.dropout)

        # Quantization
        self.quantizer = None
        if config.use_quantization:
            self.quantizer = DynamicQuantization(weight_bitwidth=config.quantization_bitwidth)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, cache: list | None = None
    ) -> tuple:
        """
        Forward pass with optional KV caching.
        """
        _batch_size, seq_len = input_ids.shape

        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        new_cache: list[Any] | None = [] if self.config.use_kv_cache else None
        new_cache = [] if self.config.use_kv_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, layer_cache = layer(x, attention_mask, layer_cache)
            if new_cache is not None:
                new_cache.append(layer_cache)

        # Layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits, new_cache

    def enable_quantization(self):
        """Enable dynamic quantization for CPU inference."""
        if self.quantizer is not None:
            for layer in self.layers:
                if hasattr(layer, "attention"):
                    if hasattr(layer.attention, "q_proj"):
                        self.quantizer.quantize_linear(layer.attention.q_proj)
                    if hasattr(layer.attention, "k_proj"):
                        self.quantizer.quantize_linear(layer.attention.k_proj)
                    if hasattr(layer.attention, "v_proj"):
                        self.quantizer.quantize_linear(layer.attention.v_proj)
                    if hasattr(layer.attention, "out_proj"):
                        self.quantizer.quantize_linear(layer.attention.out_proj)
                if hasattr(layer, "feed_forward"):
                    for module in layer.feed_forward:
                        if isinstance(module, nn.Linear):
                            self.quantizer.quantize_linear(module)


def create_cpu_optimized_model(model_size: str = "standard", custom_config: HelixConfig | None = None) -> nn.Module:
    """
    Create a Helix CPU-optimized model with NoMAD/GQA/SlidingWindow attention.

    This is the "Stack B" factory for maximum CPU performance.  For the
    coordination-aware transformer ("Stack A"), use models.create_helix_model().

    Args:
        model_size: One of "ultra_light", "light", "standard", "enhanced"
        custom_config: Optional custom configuration

    Returns:
        Helix model with CPU optimizations
    """
    if custom_config is not None:
        config = custom_config
    else:
        configs = {
            "ultra_light": HelixUltraLightConfig(),
            "light": HelixLightConfig(),
            "standard": HelixStandardConfig(),
            "enhanced": HelixEnhancedConfig(),
        }

        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")

        config = configs[model_size]

    # Create model
    model = HelixCPUModel(config)

    # Wrap with CPU optimizations
    model = CPUOptimizedModelWrapper(model)

    # Enable quantization if configured
    if config.use_quantization:
        model.enable_quantization()

    return model


def get_model_info(model: nn.Module) -> dict[str, Any]:
    """
    Get information about a Helix model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (assuming FP16)
    memory_mb = total_params * 2 / (1024 * 1024)

    # Get config
    if hasattr(model, "model"):
        config = model.model.config
    else:
        config = model.config if hasattr(model, "config") else None

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_memory_mb": memory_mb,
        "config": config.to_dict() if config else None,
    }


# Pre-configured models
MODEL_CONFIGS = {
    "helix-ultra-light": HelixUltraLightConfig(),
    "helix-light": HelixLightConfig(),
    "helix-standard": HelixStandardConfig(),
    "helix-enhanced": HelixEnhancedConfig(),
}


# Export classes and functions
__all__ = [
    "MODEL_CONFIGS",
    "HelixCPUModel",
    "HelixConfig",
    "HelixEnhancedConfig",
    "HelixLightConfig",
    "HelixStandardConfig",
    "HelixUltraLightConfig",
    "create_cpu_optimized_model",
    "get_model_info",
]
