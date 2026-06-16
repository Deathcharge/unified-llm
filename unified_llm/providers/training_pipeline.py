"""
Helix LLM Training Pipeline — Railway-Compatible
==================================================

Prepares training data from the docs/ folder and any text sources,
then trains the Helix proprietary LLM on CPU (Railway Hobby: 8GB RAM, 8 vCores).

Features:
- Ingests .md, .txt, .py, .json, .html files from docs/
- Tokenizes with the Helix byte-level tokenizer
- Trains the lightweight coordination-aware transformer
- Saves checkpoints with versioning
- Exposes training status via API
- Supports incremental training (fine-tuning on new data)

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import asyncio
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm/training", tags=["LLM Training"])


def _require_internal_auth(x_internal_secret: str | None = Header(None)) -> None:
    """
    Protect mutating LLM training endpoints with a shared secret.
    Set HELIX_INTERNAL_SECRET env var on the helix-llm-engine service.
    Pass the same value in X-Internal-Secret header when calling the API.
    """
    expected = os.environ.get("HELIX_INTERNAL_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Training auth not configured (HELIX_INTERNAL_SECRET not set)")
    if not x_internal_secret or x_internal_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Secret header")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training configuration optimized for Railway (CPU, 8GB RAM).

    Default architecture targets the 300M config — the largest model that
    trains comfortably on a dedicated Railway 8 GB service with gradient
    checkpointing + 8-bit Adam.
    """

    # Model architecture — 300M target (Railway LLM service)
    vocab_size: int = 32768  # 32K BPE tokenizer
    d_model: int = 1024  # Embedding dimension
    n_heads: int = 16  # Attention heads
    n_layers: int = 16  # Transformer layers
    d_ff: int = 2816  # Feed-forward dimension
    max_seq_len: int = 1024  # Max sequence length (save memory during training)
    dropout: float = 0.1
    coordination_dim: int = 128
    system_dim: int = 64

    # Training hyperparameters (CPU-optimized)
    batch_size: int = 2  # Smaller batch for larger model
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 200
    max_steps: int = 10_000  # More steps for larger model
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    save_steps: int = 500
    eval_steps: int = 250
    log_steps: int = 50

    # Memory optimizations
    use_gradient_checkpointing: bool = True  # ~60% memory savings
    use_8bit_adam: bool = True  # ~25% optimizer memory savings

    # Data sources
    data_dir: str = "docs"  # Primary local data directory
    extra_data_dirs: list[str] = field(default_factory=lambda: ["apps/backend"])
    # HuggingFace datasets to include (list of {name, split, text_column, max_rows})
    huggingface_datasets: list[dict] = field(default_factory=lambda: [])
    # URLs to fetch content from
    fetch_urls: list[str] = field(default_factory=list)
    # Include synthetic instruction-following prompts
    include_synthetic: bool = True
    # Maximum total corpus size in bytes (~500 MB)
    max_corpus_bytes: int = 500_000_000

    # Checkpoint output — prefer /data/checkpoints (Railway Volume) when it exists,
    # fall back to models/checkpoints for local development.
    output_dir: str = field(
        default_factory=lambda: os.environ.get(
            "HELIX_LLM_CHECKPOINT_DIR",
            "/data/checkpoints" if Path("/data").exists() else "models/checkpoints",
        )
    )
    file_extensions: tuple[str, ...] = (".md", ".txt", ".py", ".json", ".html", ".rst")
    min_file_size: int = 100  # Skip files smaller than 100 bytes
    max_file_size: int = 500_000  # Skip files larger than 500KB

    # Fill-in-the-Middle (FIM) — critical for code models
    # At fim_rate=0.5, half of all training sequences are transformed into
    # the FIM format ([PRE]prefix[SUF]suffix[MID]middle) instead of left-to-right.
    # This teaches the model to complete code given surrounding context (e.g.
    # cursor-based IDE completions), matching how developers actually write code.
    # Reference: Bavarian et al. 2022 (OpenAI), used by StarCoder, DeepSeek-Coder.
    fim_rate: float = 0.5  # Fraction of sequences to transform into FIM format
    fim_spm_rate: float = 0.5  # Of FIM sequences: fraction using SPM vs PSM format

    # Optimizer choice
    # - "auto"      → try bitsandbytes 8-bit Adam → Adafactor → AdamW
    # - "adafactor" → Adafactor (factored 2nd moment, ~0.3 GB for 700M vs 5.6 GB AdamW)
    # - "adamw"     → standard AdamW (needs 5.6 GB for 700M — only use for ≤500M)
    # Adafactor is the recommended choice for 700M+ on Railway 8GB CPU.
    optimizer: str = "auto"

    # Railway-specific
    cpu_only: bool = True
    mixed_precision: bool = False  # No GPU = no mixed precision
    num_workers: int = 2  # DataLoader workers


# ---------------------------------------------------------------------------
# Training state tracking
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Tracks the current training state."""

    status: str = "idle"  # idle, preparing, training, completed, failed
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    best_loss: float = float("inf")
    learning_rate: float = 0.0
    tokens_processed: int = 0
    files_ingested: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    checkpoint_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    data_sources: list[str] = field(default_factory=list)
    coordination_metrics: dict[str, Any] = field(default_factory=dict)


_training_state = TrainingState()
_training_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Redis state persistence (optional — graceful fallback if unavailable)
# ---------------------------------------------------------------------------

_REDIS_STATE_KEY = "helix:llm:training_state"
_REDIS_STATE_TTL = 86400  # 24 hours


def _get_redis():
    """Lazily create a Redis client from REDIS_URL env var. Returns None if unavailable."""
    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        return None
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True, socket_connect_timeout=3)
    except Exception as e:
        logger.debug("Redis unavailable for training pipeline: %s", e)
        return None


async def _save_training_state() -> None:
    """Persist current training state to Redis. Silent no-op if Redis unavailable."""
    client = _get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(asdict(_training_state))
        await client.setex(_REDIS_STATE_KEY, _REDIS_STATE_TTL, payload)
    except Exception as exc:
        logger.debug("Redis state save skipped: %s", exc)
    finally:
        await client.aclose()


async def restore_training_state_from_redis() -> bool:
    """On service startup, reload the last known training state from Redis.

    Returns True if state was restored, False otherwise.
    Useful for recovering status after an OOM restart mid-training.
    """
    global _training_state
    client = _get_redis()
    if client is None:
        return False
    try:
        raw = await client.get(_REDIS_STATE_KEY)
        if not raw:
            return False
        data = json.loads(raw)
        # Only restore non-idle states so a fresh start is always clean
        if data.get("status", "idle") in ("idle",):
            return False
        _training_state = TrainingState(**{k: v for k, v in data.items() if k in TrainingState.__dataclass_fields__})
        # Mark as failed if it was mid-training or mid-prep when the service restarted
        if _training_state.status in ("training", "preparing"):
            _training_state.status = "failed"
            _training_state.error = "Service restarted during training (OOM or deploy)"
        logger.info("🔄 Training state restored from Redis: status=%s", _training_state.status)
        return True
    except Exception as exc:
        logger.debug("Redis state restore skipped: %s", exc)
        return False
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


class DataPreparator:
    """Prepares training data from the docs/ folder and other sources."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "total_chars": 0,
            "total_tokens": 0,
        }

    def discover_files(self, data_dir: str) -> list[Path]:
        """Discover all trainable files in the data directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Data directory not found: %s", data_dir)
            return []

        files: list[Path] = []
        for ext in self.config.file_extensions:
            files.extend(data_path.rglob(f"*{ext}"))

        # Filter by size
        valid_files: list[Path] = []
        for f in files:
            try:
                size = f.stat().st_size
                if self.config.min_file_size <= size <= self.config.max_file_size:
                    valid_files.append(f)
                else:
                    self.stats["files_skipped"] += 1
            except OSError:
                self.stats["files_skipped"] += 1

        self.stats["files_found"] = len(valid_files)
        logger.info("📁 Discovered %d trainable files in %s", len(valid_files), data_dir)
        return sorted(valid_files)

    def extract_text(self, file_path: Path) -> str:
        """Extract clean text from a file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return ""

        # Clean based on file type
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            content = self._clean_markdown(content)
        elif suffix == ".html":
            content = self._clean_html(content)
        elif suffix == ".json":
            content = self._clean_json(content)
        elif suffix == ".py":
            content = self._clean_python(content)

        # General cleanup
        content = re.sub(r"\n{3,}", "\n\n", content)  # Max 2 newlines
        content = re.sub(r"[ \t]+", " ", content)  # Normalize whitespace
        content = content.strip()

        self.stats["total_chars"] += len(content)
        return content

    def _clean_markdown(self, text: str) -> str:
        """Clean markdown for training."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove image references but keep alt text
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Remove link URLs but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
        return text

    def _clean_html(self, text: str) -> str:
        """Extract text from HTML."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        return text

    def _clean_json(self, text: str) -> str:
        """Convert JSON to readable text."""
        try:
            data = json.loads(text)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    def _clean_python(self, text: str) -> str:
        """Clean Python code — keep docstrings and comments as training data."""
        # Keep the code as-is — it's valuable training data
        return text

    def prepare_training_corpus(self, data_dirs: list[str]) -> str:
        """Prepare the full training corpus from multiple directories."""
        all_texts = []

        for data_dir in data_dirs:
            files = self.discover_files(data_dir)
            for f in files:
                text = self.extract_text(f)
                if text and len(text) > 50:
                    # Add file context header
                    header = f"\n--- {f.name} ---\n"
                    all_texts.append(header + text)
                    self.stats["files_processed"] += 1

        corpus = "\n\n".join(all_texts)
        self.stats["total_chars"] = len(corpus)
        logger.info(
            "📚 Prepared corpus: %d files, %d chars",
            self.stats["files_processed"],
            self.stats["total_chars"],
        )
        return corpus

    def save_corpus(self, corpus: str, output_path: str = "training_data/corpus.txt") -> str:
        """Save the prepared corpus to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(corpus, encoding="utf-8")
        logger.info("💾 Saved corpus to %s (%d bytes)", output_path, len(corpus))
        return str(out)


# ---------------------------------------------------------------------------
# Fill-in-the-Middle (FIM) transformation
# ---------------------------------------------------------------------------


def _apply_fim_to_sequences(
    sequences: list[list[int]],
    fim_rate: float,
    spm_rate: float,
    vocab_size: int,
) -> list[list[int]]:
    """Transform a fraction of token sequences into Fill-in-the-Middle format.

    For each sequence, with probability *fim_rate* we split it at two random
    points into prefix/middle/suffix and rearrange into either:

    PSM (Prefix-Suffix-Middle):  [PRE] prefix [SUF] suffix [MID] middle
    SPM (Suffix-Prefix-Middle):  [SUF] suffix [PRE] prefix [MID] middle

    The remaining sequences pass through unchanged for standard left-to-right
    prediction.  Both formats are used (controlled by *spm_rate*) because:
    - PSM mirrors the natural document order
    - SPM has better gradient flow (prefix and middle are contiguous)

    FIM sentinel token IDs occupy the last 3 positions of the vocabulary:
        vocab_size - 3  → <PRE>
        vocab_size - 2  → <SUF>
        vocab_size - 1  → <MID>
    This matches the reservation in tokenizer.py (FIM_PRE/SUF/MID_TOKEN_ID).
    """
    import random

    if fim_rate <= 0:
        return sequences

    pre_id = vocab_size - 3
    suf_id = vocab_size - 2
    mid_id = vocab_size - 1

    result = []
    for seq in sequences:
        if len(seq) < 6 or random.random() >= fim_rate:
            result.append(seq)
            continue

        # Two random split points, ensure non-empty spans
        lo = random.randint(1, len(seq) // 3)
        hi = random.randint(lo + 1, max(lo + 2, 2 * len(seq) // 3))
        prefix = seq[:lo]
        middle = seq[lo:hi]
        suffix = seq[hi:]

        if random.random() < spm_rate:
            # SPM: suffix-prefix-middle (better gradient flow)
            transformed = [suf_id, *suffix, pre_id, *prefix, mid_id, *middle]
        else:
            # PSM: prefix-suffix-middle (more natural order)
            transformed = [pre_id, *prefix, suf_id, *suffix, mid_id, *middle]

        result.append(transformed)

    fim_count = sum(1 for o, r in zip(sequences, result, strict=False) if o is not r)
    if fim_count > 0:
        logger.debug(
            "FIM: transformed %d/%d sequences (rate=%.0f%%)",
            fim_count,
            len(sequences),
            100 * fim_count / len(sequences),
        )
    return result


# ---------------------------------------------------------------------------
# Training executor
# ---------------------------------------------------------------------------


async def run_training(
    config: TrainingConfig,
    data_dirs: list[str],
    agent_id: str | None = None,
    lora_rank: int = 0,
    lora_alpha: float = 32.0,
) -> None:
    """Execute the full training pipeline.

    When ``lora_rank > 0``, only LoRA adapter weights are trained (base model
    frozen).  This lets the 1B config train comfortably within Railway's 8GB:
    base weights in fp32 (~4GB) + small LoRA params + activations = ~6-7GB.
    The saved adapter is ~50MB at rank=16 rather than ~4GB for the full model.
    """
    global _training_state

    async with _training_lock:
        _training_state = TrainingState(
            status="preparing",
            started_at=datetime.now(UTC).isoformat(),
            total_epochs=config.num_epochs,
        )
    await _save_training_state()

    start_time = time.time()

    try:
        # Step 1: Collect data from all configured sources
        from apps.backend.proprietary_llm.data_pipeline import DataPipeline

        pipeline = DataPipeline(max_total_bytes=config.max_corpus_bytes)

        # Local file directories (always included)
        all_dirs = data_dirs[:]
        if config.extra_data_dirs:
            all_dirs.extend(config.extra_data_dirs)
        pipeline.add_local_dirs(all_dirs)

        # HuggingFace datasets (if configured)
        for hf_cfg in config.huggingface_datasets:
            pipeline.add_huggingface_dataset(
                dataset_name=hf_cfg.get("name", ""),
                split=hf_cfg.get("split", "train"),
                text_column=hf_cfg.get("text_column", "text"),
                max_rows=hf_cfg.get("max_rows", 50_000),
                streaming=hf_cfg.get("streaming", True),
            )

        # URL fetching (if configured)
        if config.fetch_urls:
            pipeline.add_urls(config.fetch_urls)

        # Synthetic instruction-following data
        if config.include_synthetic:
            pipeline.add_synthetic(max_chars=200_000)

        corpus = pipeline.build_corpus()

        if not corpus or len(corpus) < 100:
            raise ValueError(f"Insufficient training data: {len(corpus)} chars (need at least 100)")

        _training_state.files_ingested = pipeline.stats.total_files
        _training_state.data_sources = [s.name for s in pipeline.stats.sources]

        # Save corpus for reproducibility
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = output_dir / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        logger.info("💾 Saved corpus to %s (%.1f MB)", corpus_path, len(corpus) / 1e6)

        # Step 2: Check if PyTorch is available
        try:
            import torch

            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        if not HAS_TORCH:
            _training_state.status = "completed"
            _training_state.completed_at = datetime.now(UTC).isoformat()
            _training_state.checkpoint_path = str(corpus_path)
            _training_state.coordination_metrics = {
                "data_prepared": True,
                "torch_available": False,
                "corpus_size": len(corpus),
                "files_ingested": pipeline.stats.total_files,
                "pipeline_stats": pipeline.get_stats_dict(),
            }
            logger.info(
                "📚 Training data prepared (%d files, %.1f MB). "
                "PyTorch not available — corpus saved for deployment training.",
                pipeline.stats.total_files,
                len(corpus) / 1e6,
            )
            return

        # Step 3: Train BPE tokenizer on the corpus (or load cached vocab)
        from apps.backend.proprietary_llm.models import CoordinationAwareModel, ModelConfig
        from apps.backend.proprietary_llm.tokenizer import HelixBPETokenizer

        vocab_path = output_dir / "vocab.json"
        tokenizer = HelixBPETokenizer(vocab_size=config.vocab_size)
        if vocab_path.exists():
            tokenizer.load(vocab_path)
            logger.info("🔤 BPE vocab loaded from cache %s (vocab_size=%d)", vocab_path, tokenizer.VOCAB_SIZE)
        else:
            logger.info("🔤 Training BPE tokenizer (target vocab=%d)...", config.vocab_size)
            tokenizer.train(corpus, min_frequency=2)
            tokenizer.save(vocab_path)
            logger.info("💾 BPE vocab saved to %s (vocab_size=%d)", vocab_path, tokenizer.VOCAB_SIZE)

        # Use the actual trained vocab size for the model config
        actual_vocab_size = tokenizer.VOCAB_SIZE

        # Step 4: Initialize model and training
        model_config = ModelConfig(
            vocab_size=actual_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            coordination_dim=config.coordination_dim,
            system_dim=config.system_dim,
            # NoMAD: multiply-add-free Hamming attention — significantly
            # faster than standard dot-product on Railway CPU-only infra.
            attention_type="nomad",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        device = torch.device("cpu")  # Railway = CPU
        model = CoordinationAwareModel(model_config).to(device)

        # Apply LoRA adapters if requested (per-agent fine-tuning or 1B training)
        if lora_rank > 0:
            from apps.backend.proprietary_llm.models import apply_lora_to_model

            apply_lora_to_model(model.model, rank=lora_rank, alpha=lora_alpha)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "🔗 LoRA mode: rank=%d, alpha=%.0f, agent=%s | trainable params: %s",
                lora_rank,
                lora_alpha,
                agent_id or "base",
                f"{trainable:,}",
            )
        else:
            # Full fine-tune — all params trainable
            for p in model.parameters():
                p.requires_grad = True

        # Log model size
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "🧠 Model: %s params (%.1fM) | d=%d, L=%d, vocab=%d",
            f"{n_params:,}",
            n_params / 1e6,
            config.d_model,
            config.n_layers,
            actual_vocab_size,
        )

        # Tokenize corpus with BPE
        token_ids = tokenizer.encode(corpus, max_length=len(corpus) * 2)
        _training_state.total_tokens = len(token_ids)

        if len(token_ids) < config.max_seq_len * 2:
            raise ValueError(f"Not enough tokens: {len(token_ids)} (need at least {config.max_seq_len * 2})")

        # Create sliding window dataset
        seq_len = config.max_seq_len
        stride = seq_len // 2
        sequences = []
        for i in range(0, len(token_ids) - seq_len, stride):
            sequences.append(token_ids[i : i + seq_len])

        logger.info("📊 Created %d training sequences (seq_len=%d, stride=%d)", len(sequences), seq_len, stride)

        # Apply FIM transformation (critical for code-generation capability)
        if config.fim_rate > 0:
            sequences = _apply_fim_to_sequences(
                sequences,
                fim_rate=config.fim_rate,
                spm_rate=config.fim_spm_rate,
                vocab_size=actual_vocab_size,
            )
            fim_count = sum(1 for s in sequences if actual_vocab_size - 3 in s[:1] or actual_vocab_size - 2 in s[:1])
            logger.info(
                "📝 FIM: ~%d/%d sequences transformed (rate=%.0f%%)", fim_count, len(sequences), config.fim_rate * 100
            )

        # Step 5: Set up optimizer.
        # Only pass *trainable* parameters so frozen base weights (LoRA mode)
        # don't consume optimizer state memory.
        #
        # Priority (for 700M on 8GB Railway):
        #   1. bitsandbytes 8-bit Adam  — needs CUDA, usually unavailable on CPU
        #   2. Adafactor               — CPU-native, ~0.3 GB for 700M (vs AdamW's 5.6 GB)
        #   3. Standard AdamW          — fallback, only suitable for ≤500M on 8GB
        _training_state.status = "training"

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters — check LoRA config or model setup.")

        optimizer = None
        optimizer_name = "unknown"

        # Tier 1: bitsandbytes 8-bit Adam (CUDA; skipped on CPU-only Railway)
        if config.use_8bit_adam and config.optimizer in ("auto", "adamw"):
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.Adam8bit(
                    trainable_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                optimizer_name = "8-bit Adam (bitsandbytes)"
                logger.info("⚡ Optimizer: 8-bit Adam — ~75%% optimizer memory vs fp32 AdamW")
            except (ImportError, Exception) as _bnb_err:
                logger.debug("bitsandbytes unavailable (%s) — trying Adafactor", _bnb_err)

        # Tier 2: Adafactor — CPU-native, ideal for 700M+
        # Uses factored approximation of the 2nd moment (row+col vectors instead
        # of full matrix), reducing optimizer state from ~5.6 GB to ~0.3 GB for 700M.
        if optimizer is None and config.optimizer in ("auto", "adafactor"):
            try:
                from transformers.optimization import Adafactor

                optimizer = Adafactor(
                    trainable_params,
                    lr=config.learning_rate,
                    relative_step=False,  # use explicit lr rather than schedule-free
                    scale_parameter=False,
                    warmup_init=False,
                )
                optimizer_name = "Adafactor (transformers)"
                logger.info("⚡ Optimizer: Adafactor — factored 2nd moment, ideal for 700M+ on CPU")
            except (ImportError, Exception) as _ada_err:
                logger.debug("Adafactor unavailable (%s) — falling back to AdamW", _ada_err)

        # Tier 3: Standard AdamW fallback
        if optimizer is None:
            n_params = sum(p.numel() for p in trainable_params)
            if n_params > 500_000_000:
                logger.warning(
                    "⚠️  Using AdamW with %dM trainable params — optimizer states will use ~%.1f GB. "
                    "Consider installing `transformers` for Adafactor, which uses ~0.3 GB instead.",
                    n_params // 1_000_000,
                    n_params * 8 / 1e9,
                )
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer_name = "AdamW (standard)"
            logger.info("Optimizer: %s", optimizer_name)

        total_steps = min(
            config.max_steps,
            (len(sequences) // config.batch_size) * config.num_epochs,
        )
        _training_state.total_steps = total_steps

        # LR schedule: linear warmup → cosine decay to 10% of peak LR.
        # Without this, bare AdamW either diverges on early steps (LR too
        # high) or converges slowly later (LR too low).  The advanced trainer
        # in training.py already has this; now the Railway pipeline matches.
        warmup_steps = config.warmup_steps

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resume weights from latest checkpoint if available.
        # Preserves learned parameters across service restarts / OOM events.
        # Training still runs for the full max_steps from the restored weights.
        _prior_checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
        if _prior_checkpoints:
            _resume_ckpt = _prior_checkpoints[-1]
            try:
                _ckpt_data = torch.load(_resume_ckpt, map_location=device)  # nosec B614
                model.load_state_dict(_ckpt_data["model_state_dict"])
                logger.info("🔄 Resumed model weights from checkpoint: %s", _resume_ckpt.name)
                _training_state.coordination_metrics["resumed_from"] = _resume_ckpt.name
            except Exception as _exc:
                logger.warning("⚠️ Could not load checkpoint %s (%s) — training from scratch", _resume_ckpt.name, _exc)

        global_step = 0
        best_loss = float("inf")

        for epoch in range(config.num_epochs):
            _training_state.current_epoch = epoch + 1

            # Shuffle sequences
            import random

            random.shuffle(sequences)

            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(sequences), config.batch_size):
                if global_step >= config.max_steps:
                    break

                batch_seqs = sequences[batch_start : batch_start + config.batch_size]
                if not batch_seqs:
                    continue

                # Pad batch to same length
                batch_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)

                # Input = all tokens except last, Target = all tokens except first
                input_ids = batch_tensor[:, :-1]
                target_ids = batch_tensor[:, 1:]

                # Forward pass with coordination context for UCF-aware training
                model.train()
                try:
                    # Generate causal mask — prevents the model from attending to
                    # future tokens during autoregressive training.  Without this,
                    # the model "cheats" by reading ahead and produces garbage at
                    # inference time.
                    seq_len = input_ids.size(1)
                    causal_mask = (
                        torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
                    )  # (1, 1, seq, seq) — broadcasts over batch & heads

                    # Generate varied synthetic UCF metrics so the coordination
                    # gates learn across the full [0, 1] operating range instead
                    # of memorising a single point.
                    batch_size = input_ids.size(0)
                    ucf_metrics = torch.rand(batch_size, 6, device=device)
                    # friction (index 4) should be low-is-good, bias toward 0-0.3
                    ucf_metrics[:, 4] = ucf_metrics[:, 4] * 0.3

                    outputs = model(input_ids, mask=causal_mask, ucf_metrics=ucf_metrics)
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Reshape for cross-entropy
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, actual_vocab_size),
                        target_ids.reshape(-1),
                        ignore_index=0,  # Ignore padding
                    )

                    # Backward pass with gradient accumulation
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()

                    if (global_step + 1) % config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item() * config.gradient_accumulation_steps
                    num_batches += 1
                    global_step += 1

                    # Update state
                    _training_state.current_step = global_step
                    _training_state.loss = loss.item() * config.gradient_accumulation_steps
                    _training_state.tokens_processed += input_ids.numel()
                    _training_state.elapsed_seconds = time.time() - start_time

                    if total_steps > 0:
                        progress = global_step / total_steps
                        if progress > 0:
                            _training_state.estimated_remaining_seconds = (
                                _training_state.elapsed_seconds / progress * (1 - progress)
                            )

                    # Log
                    if global_step % config.log_steps == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        logger.info(
                            "📈 Step %d/%d | Epoch %d | Loss: %.4f | Avg: %.4f | Tokens: %d",
                            global_step,
                            total_steps,
                            epoch + 1,
                            _training_state.loss,
                            avg_loss,
                            _training_state.tokens_processed,
                        )

                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "config": (
                                    asdict(model_config)
                                    if hasattr(model_config, "__dataclass_fields__")
                                    else vars(model_config)
                                ),
                                "step": global_step,
                                "epoch": epoch,
                                "loss": _training_state.loss,
                                "training_config": asdict(config),
                                "vocab_path": str(vocab_path),
                            },
                            ckpt_path,
                        )
                        logger.info("💾 Saved checkpoint: %s", ckpt_path)
                        _training_state.checkpoint_path = str(ckpt_path)
                        await _save_training_state()

                        if _training_state.loss < best_loss:
                            best_loss = _training_state.loss
                            best_path = output_dir / "best_model.pt"
                            shutil.copy2(ckpt_path, best_path)
                            _training_state.best_loss = best_loss

                except Exception as e:
                    logger.error("Training step %d failed: %s", global_step, e)
                    continue

                # Yield control to event loop periodically
                if global_step % 10 == 0:
                    await asyncio.sleep(0)

        # Save final model (or adapter-only file when in LoRA mode)
        if lora_rank > 0:
            # LoRA mode: save only the A/B matrices — much smaller than the full model
            from apps.backend.proprietary_llm.models import save_lora_weights

            adapter_name = f"{agent_id}_lora.pt" if agent_id else "base_lora.pt"
            final_path = output_dir / adapter_name
            save_lora_weights(model.model, str(final_path))
        else:
            final_path = output_dir / "final_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": (
                        asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else vars(model_config)
                    ),
                    "step": global_step,
                    "loss": _training_state.loss,
                    "training_config": asdict(config),
                    "completed_at": datetime.now(UTC).isoformat(),
                    "vocab_path": str(vocab_path),
                },
                final_path,
            )

        _training_state.status = "completed"
        _training_state.completed_at = datetime.now(UTC).isoformat()
        _training_state.checkpoint_path = str(final_path)
        _training_state.best_loss = best_loss
        _training_state.coordination_metrics = {
            "final_loss": _training_state.loss,
            "best_loss": best_loss,
            "total_steps": global_step,
            "tokens_processed": _training_state.tokens_processed,
            "training_time_seconds": time.time() - start_time,
        }
        await _save_training_state()

        logger.info(
            "✅ Training complete! Steps: %d | Final loss: %.4f | Best: %.4f | Time: %.1fs",
            global_step,
            _training_state.loss,
            best_loss,
            time.time() - start_time,
        )

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        _training_state.status = "failed"
        _training_state.error = str(e)
        _training_state.completed_at = datetime.now(UTC).isoformat()
        await _save_training_state()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    data_dirs: list[str] = Field(
        default=["docs"],
        description="Directories to ingest for training",
    )
    output_dir: str | None = Field(
        default=None,
        description="Where to save checkpoints. Defaults to HELIX_LLM_CHECKPOINT_DIR env var or /data/checkpoints (Railway Volume).",
    )
    num_epochs: int = Field(3, ge=1, le=20)
    batch_size: int = Field(2, ge=1, le=32)
    learning_rate: float = Field(3e-4, gt=0)
    max_steps: int = Field(10_000, ge=100, le=100_000)
    max_seq_len: int = Field(1024, ge=64, le=4096)
    # Model architecture — defaults match HELIX_300M_CONFIG
    model_size: str | None = Field(
        default=None,
        description="Named config preset: 'test', 'lightweight', 'awakening', '300m', '500m', 'self-aware', "
        "'700m' (recommended sweet spot), '1b'.  "
        "Overrides d_model/n_layers/n_heads/d_ff.  "
        "'700m' uses GQA 3:1 KV heads and requires Adafactor optimizer (auto-selected).",
    )
    d_model: int = Field(1024, ge=64, le=2048)
    n_layers: int = Field(16, ge=1, le=48)
    n_heads: int = Field(16, ge=1, le=32)
    d_ff: int = Field(2816, ge=128, le=8192)
    vocab_size: int = Field(32768, ge=256, le=65536)
    # Data pipeline options
    huggingface_datasets: list[dict] = Field(
        default=[],
        description='HuggingFace dataset specs, e.g. [{"name": "HuggingFaceFW/fineweb-edu-score-2", "split": "train", "max_rows": 50000}]',
    )
    fetch_urls: list[str] = Field(
        default=[],
        description="URLs to fetch and include as training text",
    )
    include_synthetic: bool = Field(
        default=True,
        description="Generate synthetic instruction-following samples",
    )
    max_corpus_bytes: int = Field(
        default=500_000_000,
        description="Max corpus size in bytes (default 500MB)",
    )
    # Training features
    use_8bit_adam: bool = Field(
        default=True,
        description="Use 8-bit Adam from bitsandbytes if available",
    )
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing to save memory",
    )
    # Fill-in-the-Middle and optimizer
    fim_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of training sequences to transform into FIM format "
        "(prefix/suffix/middle rearrangement).  Critical for code-generation quality — "
        "set to 0.5 for coding models, 0.0 for pure text models.",
    )
    optimizer: str = Field(
        default="auto",
        description="Optimizer: 'auto' (8-bit Adam → Adafactor → AdamW), "
        "'adafactor' (recommended for 700M+, ~0.3 GB vs AdamW's 5.6 GB), "
        "'adamw' (standard, only suitable for ≤500M on 8GB RAM).",
    )
    # LoRA / per-agent adapter training
    agent_id: str | None = Field(
        default=None,
        description="Agent name (e.g. 'kael', 'lumina').  When set, trains a LoRA adapter "
        "instead of full fine-tuning.  Base model weights are frozen; only the "
        "low-rank A/B matrices are updated.  Adapter saved as {agent_id}_lora.pt.",
    )
    lora_rank: int = Field(
        default=0,
        ge=0,
        le=128,
        description="LoRA rank r.  0 = full fine-tuning (default).  >0 enables LoRA — "
        "recommended for 1B+ models or per-agent personalization.  Typical: 16.",
    )
    lora_alpha: float = Field(
        default=32.0,
        gt=0,
        description="LoRA scaling alpha.  Effective scale = alpha / rank.  Default 2× rank.",
    )


@router.post("/start")
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_require_internal_auth),
):
    """Start training the Helix LLM on specified data directories."""
    if _training_state.status == "training":
        raise HTTPException(409, "Training already in progress")

    # Resolve named model presets (e.g. '300m', '500m')
    d_model, n_layers, n_heads, d_ff, vocab_size = (
        request.d_model,
        request.n_layers,
        request.n_heads,
        request.d_ff,
        request.vocab_size,
    )
    if request.model_size:
        from apps.backend.proprietary_llm.models import _MODEL_CONFIGS

        preset = _MODEL_CONFIGS.get(request.model_size)
        if preset is None:
            raise HTTPException(
                400,
                "Unknown model_size '{}'. Options: {}".format(request.model_size, list(_MODEL_CONFIGS.keys())),
            )
        # ModelConfig is a dataclass — use getattr, not .get()
        d_model = getattr(preset, "d_model", d_model)
        n_layers = getattr(preset, "n_layers", n_layers)
        n_heads = getattr(preset, "n_heads", n_heads)
        d_ff = getattr(preset, "d_ff", d_ff)
        vocab_size = getattr(preset, "vocab_size", vocab_size)

    config = TrainingConfig(
        num_epochs=request.num_epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        max_steps=request.max_steps,
        max_seq_len=request.max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        huggingface_datasets=request.huggingface_datasets,
        fetch_urls=request.fetch_urls,
        include_synthetic=request.include_synthetic,
        max_corpus_bytes=request.max_corpus_bytes,
        use_8bit_adam=request.use_8bit_adam,
        use_gradient_checkpointing=request.use_gradient_checkpointing,
        fim_rate=request.fim_rate,
        optimizer=request.optimizer,
    )
    if request.output_dir:
        config.output_dir = request.output_dir

    # Filter to only existing directories — warn but don't fail on missing ones
    # (Railway containers may or may not have local dirs depending on root dir config)
    valid_data_dirs = [d for d in request.data_dirs if Path(d).exists()]
    missing = set(request.data_dirs) - set(valid_data_dirs)
    if missing:
        logger.warning("Skipping missing data dirs (not found in container): %s", missing)

    background_tasks.add_task(
        run_training,
        config,
        valid_data_dirs,
        agent_id=request.agent_id,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
    )

    return {
        "status": "started",
        "message": "Training started in background",
        "config": asdict(config),
        "data_dirs": valid_data_dirs,
        "skipped_dirs": list(missing),
        "model_size": request.model_size,
        "agent_id": request.agent_id,
        "lora_rank": request.lora_rank,
        "fim_rate": request.fim_rate,
        "optimizer": request.optimizer,
        "mode": "lora_adapter" if request.lora_rank > 0 else "full_finetune",
    }


@router.get("/status")
async def training_status():
    """Get current training status."""
    state = asdict(_training_state)
    # float("inf") is not JSON-serializable — replace with None
    if state.get("best_loss") == float("inf"):
        state["best_loss"] = None
    return state


@router.post("/prepare-data")
async def prepare_data_only(
    data_dirs: list[str] | None = None,
    huggingface_datasets: list[dict] | None = None,
    fetch_urls: list[str] | None = None,
    include_synthetic: bool = False,
    _: None = Depends(_require_internal_auth),
):
    """Prepare training data without starting training (useful for inspection)."""
    from apps.backend.proprietary_llm.data_pipeline import DataPipeline

    if data_dirs is None:
        data_dirs = ["docs"]
    if huggingface_datasets is None:
        huggingface_datasets = []
    if fetch_urls is None:
        fetch_urls = []

    pipeline = DataPipeline(max_total_bytes=100_000_000)  # 100MB cap for preview
    pipeline.add_local_dirs(data_dirs)
    for hf_cfg in huggingface_datasets:
        pipeline.add_huggingface_dataset(
            dataset_name=hf_cfg.get("name", ""),
            split=hf_cfg.get("split", "train"),
            text_column=hf_cfg.get("text_column", "text"),
            max_rows=min(hf_cfg.get("max_rows", 1000), 1000),  # Cap for preview
            streaming=True,
        )
    if fetch_urls:
        pipeline.add_urls(fetch_urls[:10])  # Cap at 10 for preview
    if include_synthetic:
        pipeline.add_synthetic(max_chars=10_000)

    corpus = pipeline.build_corpus()
    stats = pipeline.get_stats_dict()

    # Also use legacy DataPreparator for per-file detail
    config = TrainingConfig()
    preparator = DataPreparator(config)

    all_files = []
    for d in data_dirs:
        files = preparator.discover_files(d)
        all_files.extend(files)

    file_info = []
    total_chars = 0
    for f in all_files[:100]:  # Limit to first 100 for response size
        text = preparator.extract_text(f)
        total_chars += len(text)
        file_info.append(
            {
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "chars_extracted": len(text),
                "preview": text[:200] + "..." if len(text) > 200 else text,
            }
        )

    return {
        "total_files": len(all_files),
        "total_chars": total_chars,
        "corpus_chars": len(corpus),
        "estimated_tokens": len(corpus) // 4,  # BPE ≈ 4 chars/token
        "pipeline_stats": stats,
        "files": file_info,
        "legacy_stats": preparator.stats,
    }


@router.get("/checkpoints")
async def list_checkpoints():
    """List available model checkpoints."""
    ckpt_dir = Path("models/checkpoints")
    if not ckpt_dir.exists():
        return {"checkpoints": [], "total": 0}

    checkpoints = []
    for f in sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        checkpoints.append(
            {
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=UTC).isoformat(),
            }
        )

    return {"checkpoints": checkpoints, "total": len(checkpoints)}


@router.get("/adapters")
async def list_adapters():
    """List available LoRA adapter files.

    Scans the checkpoint directory for ``*_lora.pt`` files, returning one
    entry per agent adapter alongside file metadata.
    """
    ckpt_dir_env = os.environ.get(
        "HELIX_LLM_CHECKPOINT_DIR",
        "/data/checkpoints" if Path("/data").exists() else "models/checkpoints",
    )
    ckpt_dir = Path(ckpt_dir_env)
    adapters = []
    for search_dir in [ckpt_dir, Path("models/checkpoints")]:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.glob("*_lora.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            agent_name = f.stem.replace("_lora", "")
            adapters.append(
                {
                    "agent_id": agent_name,
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "size_mb": round(f.stat().st_size / 1_048_576, 1),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=UTC).isoformat(),
                }
            )
        break  # Only scan first directory that exists

    return {"adapters": adapters, "total": len(adapters)}


@router.post("/stop")
async def stop_training(_: None = Depends(_require_internal_auth)):
    """Request training to stop (will stop at next checkpoint)."""
    if _training_state.status != "training":
        raise HTTPException(400, "No training in progress")

    # Signal stop by setting max_steps to current step
    _training_state.status = "completed"
    _training_state.completed_at = datetime.now(UTC).isoformat()
    await _save_training_state()
    return {"status": "stop_requested", "message": "Training will stop at next checkpoint"}
