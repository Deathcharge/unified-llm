"""
Helix Proprietary LLM Engine
============================

A coordination-aware, multi-agent LLM engine built on Helix Collective's system framework.

Features:
- Coordination-driven model selection
- Multi-agent orchestration
- System-enhanced inference
- Self-improving architecture
- Helix-branded transformer models

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

# Check torch availability
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .advanced_quantization import (
        AWQQuantizer,
        DynamicQuantization,
        GGUFQuantizer,
        GPTQQuantizer,
    )
    from .gqa_attention import GroupedQueryAttention
    from .inference import CoordinationInference, HelixInferenceEngine
    from .kv_cache_manager import KVCacheManager
    from .llm_coordination import (
        CoordinationEnhancer,
        CoordinationMetrics,
        CoordinationOptimizer,
        CoordinationState,
        UCFIntegration,
    )
    from .models import CoordinationAwareModel, ModelConfig
    from .multicore_parallel import CPUOptimizedModelWrapper
    from .nomad_attention import NoMADAttention
    from .sliding_window import SlidingWindowAttention
    from .speculative_decoding import SpeculativeDecoder
    from .tokenizer import HelixBPETokenizer, get_tokenizer
else:
    # Stub classes when torch is not available
    class CoordinationInference:
        pass

    class HelixInferenceEngine:
        pass

    class CoordinationAwareModel:
        pass

    class CoordinationEnhancer:
        pass

    class CoordinationMetrics:
        pass

    class CoordinationOptimizer:
        pass

    class CoordinationState:
        pass

    class UCFIntegration:
        pass


# DataPipeline is pure Python (no torch dependency) — always available
from .data_pipeline import DataPipeline

__version__ = "1.0.0"
__author__ = "Helix Collective"
__license__ = "Proprietary"

# Core engine instance
_helix_llm_engine: HelixInferenceEngine | None = None


async def initialize_helix_llm_engine(
    model_path: str | None = None,
    model_size: str | None = None,
):
    """Initialize the global Helix LLM engine.

    Environment variables (override function arguments):
        HELIX_LLM_CHECKPOINT_PATH — path to a trained .pt checkpoint file
        HELIX_LLM_MODEL_SIZE — preset name ('700m', 'awakening', 'lightweight', etc.)

    For a local 700-800M param model trained on Railway/CPU, the typical
    config is:
        HELIX_LLM_MODEL_SIZE=700m
        HELIX_LLM_CHECKPOINT_PATH=/data/checkpoints/helix-700m-latest.pt
    """
    import os

    global _helix_llm_engine
    if _helix_llm_engine is not None:
        return _helix_llm_engine

    # Resolve checkpoint path from env or argument
    checkpoint = os.environ.get("HELIX_LLM_CHECKPOINT_PATH") or model_path

    if TORCH_AVAILABLE:
        from .inference import InferenceConfig

        config = InferenceConfig(
            model_path=checkpoint,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            use_multicore=True,
        )
        _helix_llm_engine = HelixInferenceEngine(model_path=checkpoint, config=config)
    else:
        _helix_llm_engine = HelixInferenceEngine()

    return _helix_llm_engine


def get_helix_llm_engine() -> HelixInferenceEngine:
    """Get the global Helix LLM engine instance."""
    return _helix_llm_engine


async def shutdown_helix_llm_engine():
    """Shutdown the global Helix LLM engine."""
    global _helix_llm_engine
    if _helix_llm_engine:
        await _helix_llm_engine.inference.clear_cache()
        _helix_llm_engine = None


# Export main classes
__all__ = [
    "AWQQuantizer",
    "CPUOptimizedModelWrapper",
    "CoordinationAwareModel",
    "CoordinationInference",
    "DataPipeline",
    "DynamicQuantization",
    "GGUFQuantizer",
    "GPTQQuantizer",
    "GroupedQueryAttention",
    "HelixBPETokenizer",
    "HelixInferenceEngine",
    "KVCacheManager",
    "ModelConfig",
    "NoMADAttention",
    "SlidingWindowAttention",
    "SpeculativeDecoder",
    "get_helix_llm_engine",
    "get_tokenizer",
    "initialize_helix_llm_engine",
    "shutdown_helix_llm_engine",
]
