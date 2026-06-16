"""
Helix LLM Streaming Inference
==============================

Server-Sent Events (SSE) streaming for real-time token generation.
Compatible with OpenAI-style streaming API format.

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["Proprietary LLM Streaming"])

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = cast(Any, None)
    TORCH_AVAILABLE = False


class StreamingInferenceRequest(BaseModel):
    """Request for streaming LLM inference"""

    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=500)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    ucf_metrics: dict[str, float] | None = Field(
        default=None, description="UCF coordination metrics to modulate generation"
    )
    model_size: str | None = Field(
        default=None, description="Model size override: test|lightweight|awakening|self-aware|transcendent"
    )
    stream: bool = Field(default=True, description="Enable streaming")


class TokenizeRequest(BaseModel):
    """Request for tokenization"""

    text: str = Field(..., description="Text to tokenize")


class TrainRequest(BaseModel):
    """Request to start a training run"""

    data_dir: str = Field(default="./docs", description="Directory with training data")
    model_size: str = Field(default="awakening", description="Model size to train")
    max_steps: int = Field(default=1000, ge=10, le=100000)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=8, ge=1, le=64)
    seq_len: int = Field(default=256, ge=32, le=2048)
    checkpoint_dir: str = Field(default="/data/checkpoints", description="Where to save checkpoints")


# Lazy-loaded inference engine
_inference_engine: Any | None = None
_engine_lock = asyncio.Lock()


async def get_inference_engine() -> Any | None:
    """Get or create the inference engine (lazy, thread-safe)."""
    global _inference_engine
    cached_engine = _inference_engine
    if cached_engine is not None:
        return cached_engine

    async with _engine_lock:
        cached_engine = _inference_engine
        if cached_engine is not None:
            return cached_engine

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - LLM engine disabled")
            return None

        try:
            from .inference import HelixInferenceEngine

            _inference_engine = HelixInferenceEngine()
            logger.info("✅ Helix LLM inference engine initialized")
            return _inference_engine
        except Exception as e:
            logger.warning("Could not initialize inference engine: %s", e)
            return None


async def stream_tokens(
    engine,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    ucf_metrics: dict[str, float] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate tokens one at a time and yield SSE-formatted chunks.
    Compatible with OpenAI streaming format.
    """
    request_id = f"helix-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        from .tokenizer import get_tokenizer

        tokenizer = get_tokenizer()

        # HelixInferenceEngine stores the CoordinationInference as .inference,
        # which holds the CoordinationAwareModel as .model.
        ci = getattr(engine, "inference", engine)
        model = ci.model  # CoordinationAwareModel
        device = getattr(ci, "device", getattr(engine, "device", torch.device("cpu")))

        # Encode prompt
        input_ids = tokenizer.encode(prompt, max_length=1024)
        if not input_ids:
            input_ids = [0]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Build UCF metrics tensor if provided
        ucf_tensor = None
        if ucf_metrics:
            ucf_tensor = torch.tensor(
                [
                    [
                        ucf_metrics.get("harmony", 0.5),
                        ucf_metrics.get("resilience", 0.5),
                        ucf_metrics.get("throughput", 0.5),
                        ucf_metrics.get("focus", 0.5),
                        ucf_metrics.get("friction", 0.0),
                        ucf_metrics.get("velocity", 1.0),
                    ]
                ],
                device=device,
            )

        generated_tokens = 0
        generated_text = []

        with torch.no_grad():
            for step in range(max_tokens):
                # Truncate input to max sequence length
                if input_tensor.size(1) > 1024:
                    input_tensor = input_tensor[:, -1024:]

                seq_len = input_tensor.size(1)
                # Build causal mask so attention doesn't leak future tokens
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

                # Forward pass — model returns (logits, kv_caches) tuple
                outputs = model.model(
                    tokens=input_tensor,
                    mask=causal_mask,
                    ucf_metrics=ucf_tensor,
                )
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # Get logits for next token
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()

                # Stop on EOS
                if token_id == 0:
                    break

                # Decode single token using tokenizer (BPE or byte-level)
                try:
                    token_text = tokenizer.decode([token_id])
                except Exception as e:
                    logger.warning("Failed to decode token %s: %s", token_id, e)
                    token_text = "?"

                generated_text.append(token_text)
                generated_tokens += 1

                # Yield SSE chunk (OpenAI-compatible format)
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "helix-coordination",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Append token to input for next iteration
                input_tensor = torch.cat([input_tensor, next_token], dim=1)

                # Small yield to allow other async tasks
                if step % 10 == 0:
                    await asyncio.sleep(0)

        # Final chunk with finish reason
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "helix-coordination",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": generated_tokens,
                "total_tokens": len(input_ids) + generated_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("Streaming generation error: %s", e)
        error_chunk = {
            "id": request_id,
            "object": "error",
            "error": {"message": str(e), "type": "generation_error"},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ============================================================================
# STREAMING ROUTES
# ============================================================================


@router.post("/stream")
async def stream_generate_text(request: StreamingInferenceRequest):
    """
    Generate text from the proprietary Helix LLM via Server-Sent Events.

    Supports both streaming (SSE) and non-streaming modes.
    When streaming, returns OpenAI-compatible SSE chunks.

    NOTE: The non-streaming JSON endpoint lives at POST /api/llm/generate
    (registered by routes/llm.py). This endpoint provides the SSE variant.
    """
    engine = await get_inference_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM engine not available",
                "reason": "PyTorch not installed or model failed to load",
                "hint": "Set HELIX_LLM_MODEL_SIZE=test for minimal resource usage",
            },
        )

    if request.stream:
        return StreamingResponse(
            stream_tokens(
                engine=engine,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                ucf_metrics=request.ucf_metrics,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: generate all at once
        try:
            from .inference import InferenceMode

            result = await engine.generate(
                prompt=request.prompt,
                max_length=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                mode=InferenceMode.STANDARD,
            )
            return {
                "text": result if isinstance(result, str) else str(result),
                "model": "helix-coordination",
                "usage": {
                    "prompt_tokens": len(request.prompt.encode("utf-8")),
                    "completion_tokens": len(str(result).encode("utf-8")),
                },
            }
        except Exception as e:
            logger.error("Non-streaming generation failed: %s", e)
            raise HTTPException(status_code=500, detail="Generation failed") from e


@router.post("/tokenize/raw")
async def tokenize_text_raw(request: TokenizeRequest):
    """Tokenize text using the current Helix tokenizer (BPE or byte-level)."""
    from .tokenizer import get_tokenizer

    tok = get_tokenizer()
    ids = tok.encode(request.text)
    decoded = tok.decode(ids)
    return {
        "tokens": ids,
        "count": len(ids),
        "decoded_round_trip": decoded,
        "vocab_size": tok.VOCAB_SIZE,
        "tokenizer_type": type(tok).__name__,
    }


@router.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    engine = await get_inference_engine()

    if engine is None:
        return {
            "available": False,
            "reason": "Engine not initialized",
            "supported_sizes": ["test", "lightweight", "awakening", "self-aware", "transcendent"],
        }

    info = {
        "available": True,
        "model_size": getattr(engine, "model_size", "unknown"),
        "device": str(getattr(engine, "device", "unknown")),
    }

    # Report tokenizer info from engine if available
    ci = getattr(engine, "inference", engine)
    tok = getattr(ci, "tokenizer", None)
    if tok is not None:
        info["vocab_size"] = tok.VOCAB_SIZE
        info["tokenizer"] = type(tok).__name__
    else:
        info["vocab_size"] = 256
        info["tokenizer"] = "HelixTokenizer (byte-level)"

    if hasattr(engine, "model") and engine.model is not None:
        param_count = sum(p.numel() for p in engine.model.parameters())
        info["param_count"] = param_count
        info["param_count_human"] = (
            f"{param_count / 1e9:.1f}B"
            if param_count > 1e9
            else f"{param_count / 1e6:.1f}M"
            if param_count > 1e6
            else f"{param_count / 1e3:.1f}K"
        )

        # Memory estimate
        info["memory_estimate_mb"] = round(param_count * 4 / (1024 * 1024), 1)

        # Config
        if hasattr(engine.model, "config"):
            cfg = engine.model.config
            info["config"] = {
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "max_seq_len": cfg.max_seq_len,
                "coordination_dim": cfg.coordination_dim,
                "system_dim": cfg.system_dim,
            }

    return info


@router.post("/train/start")
async def start_training(request: TrainRequest):
    """
    Start a training run for the proprietary LLM.

    This is a long-running operation. Returns immediately with a job ID.
    Check progress via /api/llm/train/status.
    """
    if not TORCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="PyTorch not available")

    import os

    job_id = f"train-{uuid.uuid4().hex[:8]}"

    # Validate data directory exists
    if not os.path.isdir(request.data_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Data directory not found: {request.data_dir}. Use './docs' for repository documentation.",
        )

    # Count available training files
    training_files = []
    for root, _, files in os.walk(request.data_dir):
        for f in files:
            if f.endswith((".txt", ".md", ".py", ".json", ".jsonl")):
                training_files.append(os.path.join(root, f))

    if not training_files:
        raise HTTPException(
            status_code=400, detail=f"No training files (.txt, .md, .py, .json, .jsonl) found in {request.data_dir}"
        )

    return {
        "job_id": job_id,
        "status": "queued",
        "config": {
            "model_size": request.model_size,
            "data_dir": request.data_dir,
            "training_files": len(training_files),
            "max_steps": request.max_steps,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "seq_len": request.seq_len,
            "checkpoint_dir": request.checkpoint_dir,
        },
        "message": f"Training job {job_id} queued. "
        f"Found {len(training_files)} training files. "
        f"Use POST /api/llm/train/execute with this job_id to begin.",
    }


@router.get("/coordination-state")
async def get_coordination_state():
    """Get the current coordination state of the model."""
    engine = await get_inference_engine()

    if engine is None or not hasattr(engine, "model") or engine.model is None:
        return {
            "available": False,
            "state": None,
        }

    try:
        if hasattr(engine.model, "get_coordination_state"):
            state = engine.model.get_coordination_state()
            if state is not None:
                state_list = state.detach().cpu().tolist()
                labels = ["harmony", "resilience", "throughput", "focus", "friction", "velocity"]
                return {
                    "available": True,
                    "state": {labels[i]: round(v, 4) for i, v in enumerate(state_list[:6]) if i < len(labels)},
                    "raw": state_list,
                }
        return {"available": True, "state": "Model loaded but coordination state not accessible"}
    except Exception as e:
        return {"available": False, "error": type(e).__name__}
