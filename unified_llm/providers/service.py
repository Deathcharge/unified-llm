"""
Helix LLM Engine — Standalone Railway Service
===============================================

Dedicated FastAPI service for LLM training and inference.
Deployed as 5th Railway service: helix-llm-engine (8 vCPU / 8GB RAM).

Endpoints:
    GET  /llm/health                     → Health check
    POST /llm/training/start             → Start training run
    GET  /llm/training/status            → Training progress
    POST /llm/training/prepare-data      → Preview data without training
    GET  /llm/training/checkpoints       → List saved checkpoints
    POST /llm/training/stop              → Stop training
    POST /llm/inference                  → Run inference
    GET  /llm/inference/stream           → SSE streaming inference
    GET  /llm/models                     → List available model configs

Config File Path (Railway):
    /configs/railway/helix-llm-engine.railway.toml

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import asyncio
import hmac
import logging
import os
import platform
import time
from collections import defaultdict
from typing import Any, ClassVar

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _allowed_origins() -> list[str]:
    """Build CORS origin list from env; falls back to known Helix domains."""
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"]
        if not origins:
            origins = ["http://localhost:3000"]
        return origins
    frontend = os.environ.get("FRONTEND_URL", "")
    origins = []
    if os.environ.get("ENVIRONMENT", "development") != "production":
        origins.extend(["http://localhost:3000", "http://localhost:3001"])
    if frontend:
        origins.append(frontend)
    origins = [o for o in origins if o != "*"]
    if not origins:
        origins = ["http://localhost:3000"]
    return origins


logger = logging.getLogger("helix.llm_service")
logging.basicConfig(level=logging.INFO)

_background_tasks: set[asyncio.Task[Any]] = set()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
_is_production = (os.environ.get("HELIX_ENV") or os.environ.get("ENVIRONMENT") or "").lower() == "production"
app = FastAPI(
    title="Helix LLM Engine",
    description="Dedicated training & inference service for the Helix proprietary LLM",
    version="1.0.0",
    docs_url=None if _is_production else "/llm/docs",
    redoc_url=None if _is_production else "/llm/redoc",
    openapi_url=None if _is_production else "/llm/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Secret"],
)

_start_time = time.time()


# ---------------------------------------------------------------------------
# Rate Limiting — protects LLM endpoints from abuse
# ---------------------------------------------------------------------------


class _LLMRateLimiter:
    """Lightweight sliding-window rate limiter for the standalone LLM service.

    Competitive free-tier limits:
      - Inference / streaming:  20 req/min  (Claude free ~10, ChatGPT free ~30)
      - Training start:          3 req/hour
      - Read-only endpoints:    60 req/min
      - Global per-IP fallback: 60 req/min
    """

    # (max_requests, window_seconds) per route category
    ROUTE_LIMITS: ClassVar[dict[str, tuple[int, int]]] = {
        "inference": (20, 60),  # POST /api/llm/inference, /api/llm/stream, /api/llm/multi-agent
        "training": (3, 3600),  # POST /api/llm/training/start, /api/llm/train/start
        "read": (60, 60),  # GET  /api/llm/models, /status, /health, etc.
        "default": (60, 60),  # Anything else
    }

    # Paths → category mapping (checked with str.endswith for perf)
    _INFERENCE_SUFFIXES = ("/inference", "/stream", "/multi-agent", "/route")
    _TRAINING_SUFFIXES = ("/training/start", "/train/start")

    def __init__(self, max_entries: int = 5000) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._max_entries = max_entries
        self._last_cleanup = 0.0

    def _category(self, method: str, path: str) -> str:
        if method == "POST":
            lp = path.rstrip("/").lower()
            if any(lp.endswith(s) for s in self._INFERENCE_SUFFIXES):
                return "inference"
            if any(lp.endswith(s) for s in self._TRAINING_SUFFIXES):
                return "training"
        if method == "GET":
            return "read"
        return "default"

    def check(self, client_ip: str, method: str, path: str) -> tuple[bool, dict]:
        """Return (allowed, info_dict).  Thread-safety: GIL-protected."""
        cat = self._category(method, path)
        max_req, window = self.ROUTE_LIMITS[cat]
        now = time.monotonic()

        key = "{}:{}".format(client_ip, cat)
        cutoff = now - window
        timestamps = self._windows[key]
        # Trim expired
        self._windows[key] = timestamps = [t for t in timestamps if t > cutoff]

        allowed = len(timestamps) < max_req
        if allowed:
            timestamps.append(now)

        remaining = max(0, max_req - len(timestamps))
        reset_in = round(window - (now - timestamps[0]), 1) if timestamps else 0

        # Periodic eviction (every 5 min)
        if now - self._last_cleanup > 300:
            self._cleanup(now)
            self._last_cleanup = now

        return allowed, {
            "remaining": remaining,
            "limit": max_req,
            "window": window,
            "category": cat,
            "reset_in": max(0, reset_in),
        }

    def _cleanup(self, now: float) -> None:
        expired = [k for k, v in self._windows.items() if not v or (now - max(v)) > 3600]
        for k in expired:
            del self._windows[k]
        if len(self._windows) > self._max_entries:
            sorted_keys = sorted(self._windows, key=lambda k: max(self._windows[k]) if self._windows[k] else 0)
            for k in sorted_keys[: len(self._windows) - self._max_entries]:
                del self._windows[k]


_rate_limiter = _LLMRateLimiter()

# Paths exempt from rate limiting
_EXEMPT_PATHS = frozenset({"/llm/health", "/llm/docs", "/llm/redoc", "/llm/openapi.json"})


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Enforce per-IP rate limits on LLM endpoints."""
    path = request.url.path.rstrip("/")

    # Skip health/docs
    if path in _EXEMPT_PATHS:
        return await call_next(request)

    # Internal service-to-service calls bypass rate limiting
    internal_secret = os.environ.get("INTERNAL_SERVICE_SECRET")
    if internal_secret and hmac.compare_digest(request.headers.get("X-Internal-Secret", ""), internal_secret):
        return await call_next(request)

    client_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host
        if request.client
        else "unknown"
    )

    allowed, info = _rate_limiter.check(client_ip, request.method, path)

    if not allowed:
        logger.warning(
            "Rate limit exceeded: %s %s from %s (category=%s, limit=%d/%ds)",
            request.method,
            path,
            client_ip,
            info["category"],
            info["limit"],
            info["window"],
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please slow down.",
                "retry_after_seconds": info["reset_in"],
                "limit": info["limit"],
                "window_seconds": info["window"],
                "category": info["category"],
            },
            headers={
                "Retry-After": str(int(info["reset_in"])),
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(int(info["reset_in"])),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(int(info["reset_in"]))
    return response


# ---------------------------------------------------------------------------
# Mount sub-routers from existing modules
# ---------------------------------------------------------------------------

# Training pipeline routes  (/api/llm/training/*)
try:
    from apps.backend.proprietary_llm.training_pipeline import router as training_router

    app.include_router(training_router, tags=["LLM Training"])
    logger.info("✅ Training pipeline router mounted")
except Exception as exc:
    logger.warning("⚠️ Training pipeline router unavailable: %s", exc)

# Inference / streaming routes  (/api/llm/*)
try:
    from apps.backend.proprietary_llm.api_routes import router as inference_router

    app.include_router(inference_router, tags=["LLM Inference"])
    logger.info("✅ Inference router mounted")
except Exception as exc:
    logger.warning("⚠️ Inference router unavailable: %s", exc)

try:
    from apps.backend.proprietary_llm.streaming import router as streaming_router

    app.include_router(streaming_router, tags=["LLM Streaming"])
    logger.info("✅ Streaming router mounted")
except Exception as exc:
    logger.warning("⚠️ Streaming router unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Health & discovery
# ---------------------------------------------------------------------------


@app.get("/llm/health")
async def health():
    """Health check for Railway."""
    import psutil

    try:
        import torch

        torch_available = True
        torch_version = torch.__version__
    except ImportError:
        torch_available = False
        torch_version = None

    mem = psutil.virtual_memory()
    return {
        "status": "healthy",
        "service": "helix-llm-engine",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "system": {
            "platform": platform.system(),
            "cpu_count": os.cpu_count(),
            "memory_total_gb": round(mem.total / 1e9, 2),
            "memory_available_gb": round(mem.available / 1e9, 2),
            "memory_percent": mem.percent,
        },
        "torch_available": torch_available,
        "torch_version": torch_version,
    }


@app.get("/llm/models")
async def list_models():
    """List available model configurations and their parameter counts."""
    import dataclasses

    from apps.backend.proprietary_llm.models import _MODEL_CONFIGS

    configs = {}
    for name, cfg in _MODEL_CONFIGS.items():
        d = getattr(cfg, "d_model", 256)
        n = getattr(cfg, "n_layers", 4)
        ff = getattr(cfg, "d_ff", d * 4)
        v = getattr(cfg, "vocab_size", 32768)
        # Rough param estimate: embedding + n_layers*(attn + ff) + output
        embed_params = v * d
        layer_params = n * (4 * d * d + 2 * d * ff)  # attn + FFN per layer
        total = embed_params + layer_params
        configs[name] = {
            **dataclasses.asdict(cfg),
            "estimated_params": total,
            "estimated_params_human": "%.1fM" % (total / 1e6) if total < 1e9 else "%.2fB" % (total / 1e9),
        }

    return {"models": configs, "total": len(configs)}


# ---------------------------------------------------------------------------
# Internal inference endpoint (service-to-service, authenticated by secret)
# ---------------------------------------------------------------------------


@app.post("/llm/internal/generate")
async def internal_generate(request: Request):
    """Internal inference endpoint for helix-core-api service calls.

    Authenticated via X-Internal-Secret header (same secret as training endpoints).
    Uses Qwen 2.5 GGUF (via local_llm_provider) as primary; falls back to the
    custom PyTorch model if GGUF is unavailable.
    """
    expected = os.environ.get("HELIX_INTERNAL_SECRET", "")
    secret = request.headers.get("x-internal-secret", "")
    if not expected or secret != expected:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=403, content={"detail": "Invalid or missing X-Internal-Secret"})

    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))

    if not prompt:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": "prompt is required"})

    # 1. Try Qwen 2.5 (or whichever GGUF model is configured)
    try:
        from apps.backend.services.local_llm_provider import get_local_provider

        provider: Any = get_local_provider()
        result: str = await provider.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        model_used: str = provider.config.model_name
        return {"text": result, "model": model_used, "source": "local_gguf"}
    except Exception as gguf_exc:
        logger.warning("GGUF inference failed, trying custom model: %s", gguf_exc)

    # 2. Fall back to the custom PyTorch model (may be untrained — low quality)
    try:
        from apps.backend.proprietary_llm import get_helix_llm_engine

        engine: Any = get_helix_llm_engine()
        if engine:
            inference_result: Any = await engine.inference(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature
            )
            return {
                "text": inference_result.get("generated_text", ""),
                "model": "helix-custom",
                "source": "custom_model",
            }
    except Exception as model_exc:
        logger.warning("Custom model inference failed: %s", model_exc)

    return {"text": "", "model": "none", "source": "unavailable", "error": "All inference backends unavailable"}


# ---------------------------------------------------------------------------
# Startup / shutdown hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    logger.info("🚀 Helix LLM Engine starting (PID %d, CPUs %d)", os.getpid(), os.cpu_count() or 1)

    # Restore training state from Redis (survives OOM restarts)
    try:
        from apps.backend.proprietary_llm.training_pipeline import restore_training_state_from_redis

        restored = await restore_training_state_from_redis()
        if not restored:
            logger.info("📋 No prior training state in Redis (fresh start)")
    except Exception as exc:
        logger.debug("Training state restore skipped: %s", exc)

    # Pre-warm inference engine if a checkpoint exists
    try:
        from apps.backend.proprietary_llm import TORCH_AVAILABLE, initialize_helix_llm_engine

        if TORCH_AVAILABLE:
            await initialize_helix_llm_engine()
            logger.info("🧠 Inference engine initialized")
    except Exception as exc:
        logger.warning("⚠️ Inference engine init skipped: %s", exc)

    # Pre-download the GGUF model in the background so the first user request isn't slow
    async def _warmup_gguf():
        try:
            from apps.backend.services.local_llm_provider import get_local_provider

            provider = get_local_provider()
            await provider.initialize()
            logger.info("✅ GGUF model ready: %s", provider.config.model_name)
        except Exception as exc:
            logger.warning("⚠️ GGUF warmup failed (will retry on first request): %s", exc)

    _task = asyncio.create_task(_warmup_gguf())
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)


@app.on_event("shutdown")
async def shutdown():
    try:
        from apps.backend.proprietary_llm import shutdown_helix_llm_engine

        await shutdown_helix_llm_engine()
    except Exception as e:
        logger.warning("Error during Helix LLM Engine shutdown: %s", e)
    logger.info("Helix LLM Engine shut down")
