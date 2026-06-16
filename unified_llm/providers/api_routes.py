"""
🧠 Proprietary LLM API Routes
============================

FastAPI routes exposing the Helix Coordination-Aware LLM Engine.

Features:
- Model routing based on coordination context
- Multi-agent LLM collaboration
- Inference with UCF metrics
- Performance analytics

Copyright (c) 2025 Andrew John Ward. All Rights Reserved.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from apps.backend.saas.guards import require_pro
except ImportError:
    logger.warning("Could not import require_pro guard — proprietary LLM routes will fail closed")

    async def require_pro():
        """Fallback guard that rejects all requests when tier guard is unavailable."""
        raise HTTPException(status_code=503, detail="Subscription service unavailable")


# LLM API requires PRO tier or higher
_deps = [Depends(require_pro)]
router = APIRouter(prefix="/api/llm", tags=["Proprietary LLM"], dependencies=_deps)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class InferenceRequest(BaseModel):
    """Request for LLM inference"""

    prompt: str = Field(..., description="Input prompt for the model")
    context: dict[str, Any] | None = Field(default=None, description="Additional context")
    agent_id: str | None = Field(default=None, description="Requesting agent ID")
    performance_score: float | None = Field(default=0.5, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class InferenceResponse(BaseModel):
    """Response from LLM inference"""

    response: str
    model_used: str
    coordination_score: float
    tokens_used: int
    latency_ms: float


class RoutingRequest(BaseModel):
    """Request for model routing decision"""

    user_input: str
    ucf_metrics: dict[str, float] | None = None
    agent_state: dict[str, Any] | None = None
    strategy: str = Field(default="balanced", description="Routing strategy")


class RoutingResponse(BaseModel):
    """Response with model selection"""

    selected_model: str
    confidence: float
    reasoning: str
    alternatives: list[dict[str, Any]]


class MultiAgentRequest(BaseModel):
    """Request for multi-agent collaboration"""

    task: str
    agents: list[str] = Field(..., description="List of agent IDs to involve")
    collaboration_mode: str = Field(default="parallel", description="parallel, sequential, consensus")
    context: dict[str, Any] | None = None


class MultiAgentResponse(BaseModel):
    """Response from multi-agent collaboration"""

    result: str
    agent_responses: list[dict[str, Any]]
    consensus_level: float
    synthesis_method: str


class ModelStatus(BaseModel):
    """Status of a model"""

    model_id: str
    provider: str
    available: bool
    performance: dict[str, float]
    coordination_compatibility: float


# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================

# Lazy load to avoid import issues at startup
_router_instance = None
_engine_instance = None

# LoRA adapter cache: agent_id → loaded adapter state dict
# The base model is loaded once and adapters are hot-swapped per request.
_adapter_cache: dict[str, Any] = {}


def _get_adapter_path(agent_id: str) -> str | None:
    """Return the filesystem path to an agent's LoRA adapter, or None."""
    import os

    ckpt_dir = os.environ.get(
        "HELIX_LLM_CHECKPOINT_DIR",
        "/data/checkpoints" if __import__("pathlib").Path("/data").exists() else "models/checkpoints",
    )
    candidate = __import__("pathlib").Path(ckpt_dir) / f"{agent_id}_lora.pt"
    return str(candidate) if candidate.exists() else None


def _apply_adapter_to_engine(engine: Any, agent_id: str) -> bool:
    """Load a LoRA adapter for *agent_id* onto the engine's model.

    Returns True if adapter was found and applied, False otherwise.
    Caches loaded adapters to avoid re-reading from disk on each request.
    """
    path = _get_adapter_path(agent_id)
    if not path:
        return False

    try:
        if agent_id not in _adapter_cache:
            import torch

            _adapter_cache[agent_id] = torch.load(path, map_location="cpu")  # nosec B614
            logger.info("Loaded LoRA adapter for agent '%s' from %s", agent_id, path)

        from apps.backend.proprietary_llm.models import load_lora_weights

        model = getattr(engine, "model", None)
        if model is None:
            return False
        inner = getattr(model, "model", model)  # unwrap CoordinationAwareModel
        load_lora_weights(inner, path)
        return True
    except Exception as e:
        logger.warning("Could not apply LoRA adapter for agent '%s': %s", agent_id, e)
        return False


def get_router():
    """Get or create the coordination model router"""
    global _router_instance
    if _router_instance is None:
        try:
            from .router import CoordinationModelRouter

            _router_instance = CoordinationModelRouter()
        except Exception as e:
            logger.warning("CoordinationModelRouter not available: %s", e)
            _router_instance = None
    return _router_instance


def get_engine():
    """Get or create the Helix LLM engine"""
    global _engine_instance
    if _engine_instance is None:
        try:
            from .core import HelixLLMEngine

            _engine_instance = HelixLLMEngine()
        except Exception as e:
            logger.warning("HelixLLMEngine not available: %s", e)
            _engine_instance = None
    return _engine_instance


# ============================================================================
# ROUTES
# ============================================================================


@router.get("/status/detailed")
async def get_llm_status_detailed():
    """Get detailed status of the proprietary LLM system (routing, coordination features).

    NOTE: The basic /api/llm/status endpoint is served by routes/llm.py.
    This endpoint provides extended routing and feature information.
    """
    router_inst = get_router()
    engine = get_engine()

    return {
        "status": "operational" if engine else "degraded",
        "engine_available": engine is not None,
        "router_available": router_inst is not None,
        "features": {
            "coordination_routing": router_inst is not None,
            "multi_agent_collaboration": True,
            "system_enhancement": True,
            "ucf_integration": True,
        },
        "supported_strategies": [
            "coordination_first",
            "performance_first",
            "cost_optimized",
            "balanced",
            "system_enhanced",
        ],
    }


@router.get("/models")
async def list_available_models():
    """List all available models and their capabilities"""
    engine = get_engine()

    if not engine:
        # Query available models from the unified LLM service
        try:
            from apps.backend.services.llm_router import get_llm_router

            llm_router = get_llm_router()
            available = llm_router.get_available_models() if llm_router else []  # type: ignore[attr-defined]
            if available:
                return {"models": available}
        except Exception as e:
            logger.debug("LLM router unavailable: %s", e)

        # Return configured model catalog when no engine or router available
        import os

        models = []
        if os.getenv("ANTHROPIC_API_KEY"):
            models.append(
                {
                    "model_id": "claude-3-opus",
                    "provider": "anthropic",
                    "performance_score": 0.9,
                    "capabilities": ["reasoning", "code", "creativity"],
                    "cost_tier": "premium",
                }
            )
            models.append(
                {
                    "model_id": "claude-3-sonnet",
                    "provider": "anthropic",
                    "performance_score": 0.8,
                    "capabilities": ["reasoning", "code", "speed"],
                    "cost_tier": "standard",
                }
            )
        if os.getenv("OPENAI_API_KEY"):
            models.append(
                {
                    "model_id": "gpt-4-turbo",
                    "provider": "openai",
                    "performance_score": 0.85,
                    "capabilities": ["reasoning", "code", "analysis"],
                    "cost_tier": "premium",
                }
            )
        if os.getenv("XAI_API_KEY"):
            models.append(
                {
                    "model_id": "grok-2",
                    "provider": "xai",
                    "performance_score": 0.75,
                    "capabilities": ["reasoning", "speed", "analysis"],
                    "cost_tier": "standard",
                }
            )
        return {"models": models}

    try:
        models = await engine.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error("Failed to list models: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@router.get("/adapters")
async def list_adapters():
    """List LoRA adapter files available on this engine instance.

    Each adapter corresponds to a per-agent fine-tuned personality trained
    on top of the shared base model.  Adapters are loaded on-demand during
    inference when ``agent_id`` is provided.
    """
    import os
    from pathlib import Path

    ckpt_dir = os.environ.get(
        "HELIX_LLM_CHECKPOINT_DIR",
        "/data/checkpoints" if Path("/data").exists() else "models/checkpoints",
    )
    adapters = []
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        for f in sorted(ckpt_path.glob("*_lora.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            adapters.append(
                {
                    "agent_id": f.stem.replace("_lora", ""),
                    "file": f.name,
                    "size_mb": round(f.stat().st_size / 1_048_576, 1),
                    "cached": f.stem.replace("_lora", "") in _adapter_cache,
                }
            )
    return {
        "adapters": adapters,
        "total": len(adapters),
        "checkpoint_dir": "configured" if ckpt_path.exists() else None,
    }


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run coordination-aware LLM inference.

    The model is automatically selected based on:
    - Coordination level requirements
    - UCF metrics from context
    - Performance/cost tradeoffs
    """
    engine = get_engine()

    if not engine:
        # Route through unified LLM service as fallback
        import time

        try:
            from apps.backend.services.unified_llm import unified_llm

            start = time.time()
            result_text = await unified_llm.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            latency = (time.time() - start) * 1000

            return InferenceResponse(
                response=result_text or "",
                model_used="unified-llm-fallback",
                coordination_score=request.performance_score or 0.5,
                tokens_used=len((result_text or "").split()),
                latency_ms=latency,
            )
        except Exception as fallback_err:
            logger.warning("Unified LLM fallback failed: %s", fallback_err)
            raise HTTPException(
                status_code=503,
                detail="LLM engine unavailable. Configure API keys for Anthropic, OpenAI, or xAI.",
            ) from fallback_err

    try:
        # Hot-swap LoRA adapter for this agent if one exists
        if request.agent_id:
            _apply_adapter_to_engine(engine, request.agent_id)

        result = await engine.inference(
            prompt=request.prompt,
            context=request.context or {},
            agent_id=request.agent_id,
            performance_score=request.performance_score,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return InferenceResponse(
            response=result.get("response", ""),
            model_used=result.get("model_used", "unknown"),
            coordination_score=result.get("coordination_score", 0.0),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=result.get("latency_ms", 0.0),
        )
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process LLM inference request") from e


@router.post("/route", response_model=RoutingResponse)
async def route_request(request: RoutingRequest):
    """
    Get a routing decision without executing inference.

    Useful for understanding which model would be selected
    and why, before committing to an API call.
    """
    router_inst = get_router()

    if not router_inst:
        # No router available — use default routing logic
        return RoutingResponse(
            selected_model="grok-3-mini",
            confidence=0.80,
            reasoning="Default route: xAI Grok (primary LLM provider) — router not initialized",
            alternatives=[
                {
                    "model": "claude-3-5-haiku",
                    "confidence": 0.75,
                    "reason": "Anthropic fallback — higher quality, higher cost",
                },
            ],
        )

    try:
        from .router import RoutingContext

        context = RoutingContext(
            user_input=request.user_input,
            ucf_metrics=request.ucf_metrics or {},
            system_state={},
            agent_state=request.agent_state or {},
            temporal_context={},
        )

        result = await router_inst.route(context, strategy=request.strategy)

        return RoutingResponse(
            selected_model=result.get("model_id", "unknown"),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", ""),
            alternatives=result.get("alternatives", []),
        )
    except Exception as e:
        logger.error("Routing failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to route LLM request") from e


@router.post("/multi-agent", response_model=MultiAgentResponse)
async def multi_agent_collaboration(request: MultiAgentRequest):
    """
    Execute a task using multiple agents in collaboration.

    Collaboration modes:
    - parallel: All agents work simultaneously
    - sequential: Agents work in order, passing context
    - consensus: Agents vote on best response
    """
    router_inst = get_router()

    if not router_inst:
        # Route through multi-AI orchestrator when proprietary router unavailable
        try:
            from apps.backend.services.multi_ai_orchestrator import MultiAIOrchestrator

            orchestrator = MultiAIOrchestrator()
            result = await orchestrator.execute_task(request.task[:2000])  # type: ignore[attr-defined]
            return MultiAgentResponse(
                result=result.get("result", "Task completed"),
                agent_responses=[
                    {
                        "agent": agent,
                        "response": result.get("result", "")[:200],
                        "confidence": 0.8,
                    }
                    for agent in request.agents[:3]
                ],
                consensus_level=result.get("consensus", 0.75),
                synthesis_method=request.collaboration_mode,
            )
        except Exception as e:
            logger.error("Multi-agent fallback failed: %s", e)
            raise HTTPException(status_code=503, detail="Multi-agent collaboration unavailable") from e

    try:
        from .router import MultiAgentRouter

        multi_router = MultiAgentRouter(router_inst)

        result = await multi_router.collaborate(  # type: ignore[attr-defined]
            task=request.task,
            agents=request.agents,
            mode=request.collaboration_mode,
            context=request.context,
        )

        return MultiAgentResponse(
            result=result.get("synthesis", ""),
            agent_responses=result.get("responses", []),
            consensus_level=result.get("agreement_level", 0.0),
            synthesis_method=request.collaboration_mode,
        )
    except Exception as e:
        logger.error("Multi-agent collaboration failed: %s", e)
        raise HTTPException(status_code=500, detail="Multi-agent collaboration failed") from e


@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for the LLM system"""
    router_inst = get_router()

    if router_inst:
        try:
            metrics = router_inst.get_performance_metrics()
            if metrics:
                return metrics
        except Exception as e:
            logger.warning("Could not get router performance metrics: %s", e)

    # Use process profiler for real system metrics
    try:
        from apps.backend.core.performance_profiler import profiler

        report = profiler.get_process_report()
        return {
            "total_requests": report.get("total_requests", 0),
            "avg_latency_ms": report.get("avg_response_time_ms", 0),
            "success_rate": 1.0 - report.get("error_rate", 0.0),
            "model_usage": {},
            "coordination_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
            "cost_efficiency": {
                "cost_per_1k_tokens": 0.0,
                "coordination_per_dollar": 0.0,
            },
        }
    except (ConnectionError, TimeoutError) as e:
        logger.warning("LLM API connection error: %s", e)
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "model_usage": {},
            "coordination_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
            "cost_efficiency": {
                "cost_per_1k_tokens": 0.0,
                "coordination_per_dollar": 0.0,
            },
        }
    except (ValueError, TypeError, KeyError) as e:
        logger.debug("LLM API validation error: %s", e)
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "model_usage": {},
            "coordination_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
            "cost_efficiency": {
                "cost_per_1k_tokens": 0.0,
                "coordination_per_dollar": 0.0,
            },
        }
    except Exception:
        logger.exception("Unexpected error in LLM API route")
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "model_usage": {},
            "coordination_scores": {"avg": 0.0, "min": 0.0, "max": 0.0},
            "cost_efficiency": {
                "cost_per_1k_tokens": 0.0,
                "coordination_per_dollar": 0.0,
            },
        }


@router.get("/coordination/analysis")
async def analyze_coordination_requirements(prompt: str):
    """
    Analyze a prompt to determine coordination requirements.

    Returns recommendations for model selection based on the
    complexity and nature of the task.
    """
    # Simple analysis based on prompt characteristics
    word_count = len(prompt.split())
    has_code = any(kw in prompt.lower() for kw in ["code", "function", "class", "api"])
    has_reasoning = any(kw in prompt.lower() for kw in ["why", "explain", "analyze", "reason"])
    has_creative = any(kw in prompt.lower() for kw in ["create", "write", "generate", "imagine"])

    # Calculate recommended coordination level
    base_level = 0.5
    if word_count > 100:
        base_level += 0.1
    if has_code:
        base_level += 0.15
    if has_reasoning:
        base_level += 0.15
    if has_creative:
        base_level += 0.1

    recommended_level = min(0.95, base_level)

    return {
        "prompt_length": word_count,
        "characteristics": {
            "requires_code": has_code,
            "requires_reasoning": has_reasoning,
            "requires_creativity": has_creative,
        },
        "recommended_performance_score": recommended_level,
        "recommended_model": (
            "claude-3-opus"
            if recommended_level > 0.85
            else "claude-3-sonnet"
            if recommended_level > 0.7
            else "helix-local"
        ),
        "estimated_tokens": word_count * 4,
        "estimated_cost_range": {
            "min": round(word_count * 0.00001, 4),
            "max": round(word_count * 0.0001, 4),
        },
    }
