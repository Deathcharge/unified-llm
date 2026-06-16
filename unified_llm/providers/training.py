"""
Helix Training Pipeline
======================

Advanced training pipeline for coordination-aware transformer models.

Features:
- Coordination-guided training
- System-enhanced optimization
- Multi-agent collaborative learning
- UCF metric integration
- Self-improving architecture
- Distributed training support

(c) Helix Collective 2024 - Proprietary Technology Stack
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.parallel as _torch_parallel
    import torch.optim as optim
    import torch.utils.data as _torch_data

    HAS_TORCH = True
except ImportError:
    torch = cast(Any, None)
    dist = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    optim = cast(Any, None)
    _torch_parallel = cast(Any, None)
    _torch_data = cast(Any, None)
    HAS_TORCH = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from apps.backend.core.system_coordination_core import get_system_core_instance

from .models import CoordinationAwareModel, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


if HAS_TORCH:

    class TextFileDataset(_torch_data.Dataset):
        """
        Simple sliding-window dataset over plain-text files.

        Reads every ``*.txt``, ``*.md``, and ``*.py`` file found under
        ``data_dir`` recursively, encodes the concatenated text with the
        byte-level ``HelixTokenizer``, and slices it into fixed-length chunks.
        Adjacent chunks overlap by ``stride`` tokens so the model sees each
        byte in multiple contexts.

        Usage example
        -------------
        ::

            from apps.backend.proprietary_llm.training import TextFileDataset
            from torch.utils.data import DataLoader

            ds = TextFileDataset("./training_data", seq_len=256)
            loader = DataLoader(ds, batch_size=8, shuffle=True)

        Tips
        ----
        * Point ``data_dir`` at any directory containing text: project docs,
          code files, conversation logs, etc.
        * The ``lightweight`` model (d_model=256) can train on CPU overnight
          given a few thousand sentences of text.
        * Checkpoints are written by ``CoordinationTrainer`` every
          ``TrainingConfig.save_steps`` steps.
        """

        def __init__(
            self,
            data_dir: str,
            seq_len: int = 256,
            stride: int | None = None,
            extensions: list[str] | None = None,
        ) -> None:
            from .inference import HelixTokenizer

            self.seq_len = seq_len
            self.stride = stride if stride is not None else seq_len // 2

            if extensions is None:
                extensions = [".txt", ".md", ".py", ".rst", ".jsonl"]

            data_path = Path(data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Training data directory not found: {data_dir}")

            # Collect and encode all files
            all_tokens: list[int] = []
            file_count = 0
            tokenizer = HelixTokenizer()
            for ext in extensions:
                for filepath in sorted(data_path.rglob(f"*{ext}")):
                    try:
                        if ext == ".jsonl":
                            # Extract text fields from JSON-Lines files
                            texts = []
                            for line in filepath.read_text(encoding="utf-8", errors="replace").splitlines():
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    import json as _json

                                    obj = _json.loads(line)
                                    # Support common text field names
                                    for key in ("text", "content", "message", "response", "input", "output"):
                                        if key in obj and isinstance(obj[key], str):
                                            texts.append(obj[key])
                                            break
                                    else:
                                        # Fall back: stringify the whole object
                                        texts.append(str(obj))
                                except (ValueError, TypeError, AttributeError):
                                    texts.append(line)
                                except Exception:
                                    logger.exception("Unexpected error processing training text")
                                    texts.append(line)
                            text = "\n".join(texts)
                        else:
                            text = filepath.read_text(encoding="utf-8", errors="replace")
                        all_tokens.extend(tokenizer.encode(text, max_length=len(text)))
                        # Insert a newline token (10 = b'\n') as document separator
                        all_tokens.append(10)
                        file_count += 1
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", filepath, exc)

            if not all_tokens:
                raise ValueError(
                    f"No usable text found in {data_dir!r} "
                    f"(looked for: {extensions}). "
                    "Add .txt or .md files to the training directory."
                )

            self._tokens = torch.tensor(all_tokens, dtype=torch.long)
            # Build chunk start indices
            total = len(all_tokens)
            self._starts: list[int] = list(range(0, total - seq_len, self.stride))

            logger.info(
                "TextFileDataset: %d files, %d tokens, %d chunks (seq_len=%d, stride=%d)",
                file_count,
                total,
                len(self._starts),
                seq_len,
                self.stride,
            )

        def __len__(self) -> int:
            return len(self._starts)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            start = self._starts[idx]
            chunk = self._tokens[start : start + self.seq_len + 1]
            # Language modelling: input = chunk[:-1], target = chunk[1:]
            return chunk[:-1].clone(), chunk[1:].clone()

else:
    TextFileDataset = None  # type: ignore[assignment,misc]


@dataclass
class TrainingConfig:
    """Configuration for Helix model training"""

    # Model configuration
    model_config: ModelConfig

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500

    # Coordination parameters
    coordination_weight: float = 0.1
    ucf_weight: float = 0.05
    system_weight: float = 0.05

    # Optimization
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # Logging and monitoring
    log_wandb: bool = True
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # Coordination training
    coordination_training: bool = True
    system_enhancement: bool = True
    multi_agent_training: bool = True


class CoordinationTrainer:
    """
    Coordination-aware model trainer

    Handles training of Helix transformer models with:
    - Coordination-guided optimization
    - System-enhanced learning
    - Multi-agent collaborative training
    - UCF metric integration
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = CoordinationAwareModel(config.model_config)
        self.model.to(self.device)

        # Initialize system core
        self.system_core = get_system_core_instance()

        # Setup distributed training if enabled
        if config.distributed:
            self._setup_distributed()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        self.training_history: list[dict[str, Any]] = []

        # Coordination tracking
        self.coordination_metrics: dict[str, list[float]] = {
            "ucf_alignment": [],
            "system_coherence": [],
            "multi_agent_harmony": [],
        }

        # Setup logging
        self._setup_logging()

        logger.info("✅ Coordination Trainer initialized")

    def _setup_distributed(self):
        """Setup distributed training"""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.model.model = cast(
            Any, _torch_parallel.DistributedDataParallel(self.model.model, device_ids=[self.config.rank])
        )
        logger.info("✅ Distributed training initialized (rank %d)", self.config.rank)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with coordination-aware parameters"""

        # Separate parameters for different learning rates
        base_params = []
        coordination_params = []
        system_params = []

        for name, param in self.model.model.named_parameters():
            if "coordination" in name:
                coordination_params.append(param)
            elif "system" in name:
                system_params.append(param)
            else:
                base_params.append(param)

        # Different learning rates for different parameter groups
        param_groups = [
            {"params": base_params, "lr": self.config.learning_rate},
            {"params": coordination_params, "lr": self.config.learning_rate * 1.5},
            {"params": system_params, "lr": self.config.learning_rate * 1.2},
        ]

        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay,
        )

        return optimizer

    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler with warmup"""

        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))

            # Cosine decay
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, self.config.max_steps - self.config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _setup_logging(self):
        """Setup logging and monitoring"""

        # Create directories
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        if WANDB_AVAILABLE and self.config.log_wandb and self.config.rank == 0:
            wandb.init(
                project="helix-llm-training",
                config=asdict(self.config),
                name="helix-{}".format(datetime.now(UTC).strftime("%Y%m%d-%H%M%S")),
            )

    def train(
        self,
        train_dataset: Dataset[Any],
        eval_dataset: Dataset[Any] | None = None,
        collate_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Train the model with coordination awareness

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            collate_fn: Custom collation function

        Returns:
            Training summary
        """

        # Create data loaders
        train_loader = _torch_data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4 if self.config.distributed else 0,
            pin_memory=True,
        )

        if eval_dataset:
            eval_loader = _torch_data.DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4 if self.config.distributed else 0,
                pin_memory=True,
            )

        # Training loop
        self.model.train()

        start_time = time.time()

        for _epoch in range(1000):  # Infinite training until max_steps
            for batch in train_loader:
                if self.global_step >= self.config.max_steps:
                    break

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass with coordination
                loss, metrics = self._training_step(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.gradient_clip_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()

                # Update coordination states
                if self.config.coordination_training:
                    self._update_coordination_states(metrics)

                # Logging
                if self.global_step % 10 == 0:
                    self._log_training_step(metrics)

                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self._evaluate(eval_loader)
                    self._log_evaluation(eval_metrics)

                # Checkpoint saving
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1

            if self.global_step >= self.config.max_steps:
                break

        # Final evaluation
        if eval_dataset:
            final_metrics = self._evaluate(eval_loader)
        else:
            final_metrics = {}

        # Save final model
        self._save_checkpoint(final=True)

        # Training summary
        training_time = time.time() - start_time
        summary = {
            "total_steps": self.global_step,
            "training_time": training_time,
            "best_loss": self.best_loss,
            "final_metrics": final_metrics,
            "coordination_metrics": self.coordination_metrics,
        }

        logger.info("✅ Training completed in %.2f seconds", training_time)
        return summary

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move batch to appropriate device"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform single training step with coordination awareness"""

        # Extract inputs
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)

        # Build causal mask — lower-triangular, prevents attending to future tokens.
        # If an attention_mask (padding mask) is also provided, combine them.
        seq_len = input_ids.size(1)
        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=self.device)).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq, seq)
        if attention_mask is not None:
            # attention_mask is typically (batch, seq) with 1=valid, 0=pad.
            # Expand to (batch, 1, 1, seq) and combine with causal mask.
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq)
            causal_mask = causal_mask * pad_mask

        # Get coordination metrics
        ucf_metrics = self._get_batch_ucf_metrics(batch)
        system_state = self._get_batch_system_state(batch)
        coordination_state = self._get_batch_coordination_state(batch)

        # Forward pass
        logits, _kv = self.model.model(
            tokens=input_ids,
            mask=causal_mask,
            ucf_metrics=ucf_metrics,
            system_state=system_state,
            coordination_state=coordination_state,
        )

        # Calculate losses
        base_loss = self._calculate_base_loss(logits, labels)
        coordination_loss = self._calculate_coordination_loss(logits, batch)
        ucf_loss = self._calculate_ucf_loss(ucf_metrics, batch)
        system_loss = self._calculate_system_loss(system_state, batch)

        # Combine losses
        total_loss = (
            base_loss
            + self.config.coordination_weight * coordination_loss
            + self.config.ucf_weight * ucf_loss
            + self.config.system_weight * system_loss
        )

        # Calculate metrics
        metrics = {
            "loss": total_loss.item(),
            "base_loss": base_loss.item(),
            "coordination_loss": coordination_loss.item(),
            "ucf_loss": ucf_loss.item(),
            "system_loss": system_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "ucf_alignment": self._calculate_ucf_alignment(ucf_metrics, batch),
            "system_coherence": self._calculate_system_coherence(system_state),
            "multi_agent_harmony": self._calculate_agent_harmony(),
        }

        return total_loss, metrics

    def _get_batch_ucf_metrics(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Get UCF metrics for batch"""
        if "ucf_metrics" in batch:
            return batch["ucf_metrics"]

        # Generate synthetic UCF metrics (uniform in [0, 1]) so the
        # coordination gates learn across the full operating range.
        # torch.randn was wrong here — UCF metrics are bounded 0-1, not Gaussian.
        batch_size = batch["input_ids"].size(0)
        ucf_metrics = torch.rand(batch_size, 6).to(self.device)  # 6 UCF dimensions
        # friction (index 4) should be low-is-good — bias toward 0-0.3
        ucf_metrics[:, 4] = ucf_metrics[:, 4] * 0.3
        return ucf_metrics

    def _get_batch_system_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Get system state for batch"""
        if "system_state" in batch:
            return batch["system_state"]

        # Generate synthetic system state if not provided
        batch_size = batch["input_ids"].size(0)
        system_state = torch.randn(batch_size, self.config.model_config.system_dim).to(self.device)
        return system_state

    def _get_batch_coordination_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Get coordination state for batch"""
        if "coordination_state" in batch:
            return batch["coordination_state"]

        # Use model's current coordination state
        return self.model.get_coordination_state().unsqueeze(0).expand(batch["input_ids"].size(0), -1)

    def _calculate_base_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate standard language modeling loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def _calculate_coordination_loss(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate coordination-aware loss"""

        # Coordination regularization loss
        coordination_params = [p for n, p in self.model.model.named_parameters() if "coordination" in n]
        if not coordination_params:
            return torch.tensor(0.0, device=self.device)

        coordination_loss = torch.stack([p.pow(2).sum() for p in coordination_params]).sum() * 0.001
        return coordination_loss

    def _calculate_ucf_loss(self, ucf_metrics: torch.Tensor | None, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate UCF alignment loss"""
        if ucf_metrics is None:
            return torch.tensor(0.0, device=self.device)

        # Target UCF alignment (encourage high coordination)
        target_alignment = torch.tensor(
            [0.8, 0.8, 0.8, 0.8, 0.2, 0.8], device=self.device
        )  # High harmony, low friction

        ucf_alignment_loss = F.mse_loss(ucf_metrics.mean(dim=0), target_alignment)
        return ucf_alignment_loss

    def _calculate_system_loss(self, system_state: torch.Tensor | None, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate system coherence loss"""
        if system_state is None:
            return torch.tensor(0.0, device=self.device)

        # Encourage system coherence
        coherence_loss = -torch.mean(torch.abs(system_state)) * 0.01
        return coherence_loss

    def _calculate_ucf_alignment(self, ucf_metrics: torch.Tensor | None, batch: dict[str, torch.Tensor]) -> float:
        """Calculate UCF alignment score"""
        if ucf_metrics is None:
            return 0.5

        # Calculate alignment with target coordination state
        target = torch.tensor([0.8, 0.8, 0.8, 0.8, 0.2, 0.8], device=self.device)
        alignment = F.cosine_similarity(ucf_metrics.mean(dim=0), target, dim=0).item()
        return max(0.0, min(1.0, alignment))

    def _calculate_system_coherence(self, system_state: torch.Tensor | None) -> float:
        """Calculate system coherence score"""
        if system_state is None:
            return 0.5

        # Calculate coherence as inverse of entropy
        probabilities = torch.softmax(system_state, dim=-1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        coherence = 1.0 / (1.0 + entropy.mean().item())
        return coherence

    def _calculate_agent_harmony(self) -> float:
        """Calculate multi-agent harmony score based on model's coordination state.

        Derives harmony from the model's current coordination state tensor
        by computing the mean of sigmoid-normalized values, which reflects
        how aligned the internal representations are.
        """
        try:
            coordination_state = self.model.get_coordination_state()
            if coordination_state is not None:
                # Harmony = average activation in coordination space
                harmony = torch.sigmoid(coordination_state).mean().item()
                return max(0.0, min(1.0, harmony))
        except Exception as e:
            logger.debug("Coordination state unavailable: %s", e)
        # Return neutral value if coordination state unavailable
        return 0.5

    def _update_coordination_states(self, metrics: dict[str, float]):
        """Update coordination states based on training metrics"""

        # Update coordination state based on UCF alignment
        ucf_alignment = metrics.get("ucf_alignment", 0.5)
        current_state = self.model.get_coordination_state()

        # Simple update rule (could be more sophisticated)
        update_factor = (ucf_alignment - 0.5) * 0.1
        new_state = current_state + update_factor * torch.randn_like(current_state)
        self.model.update_coordination_state(new_state)

        # Update system state based on coherence
        system_coherence = metrics.get("system_coherence", 0.5)
        current_system = self.model.get_system_state()

        system_update = (system_coherence - 0.5) * 0.05 * torch.randn_like(current_system)
        self.model.update_system_state(current_system + system_update)

    def _log_training_step(self, metrics: dict[str, float]):
        """Log training step metrics"""

        # Update coordination metrics tracking
        for key in ["ucf_alignment", "system_coherence", "multi_agent_harmony"]:
            if key in metrics:
                self.coordination_metrics[key].append(metrics[key])

        # Log to console
        logger.info(
            "Step %d: loss=%.4f, lr=%.6f, ucf=%.3f, system=%.3f",
            self.global_step,
            metrics["loss"],
            metrics["learning_rate"],
            metrics.get("ucf_alignment", 0.0),
            metrics.get("system_coherence", 0.0),
        )

        # Log to WandB
        if WANDB_AVAILABLE and self.config.log_wandb and self.config.rank == 0:
            wandb.log(metrics, step=self.global_step)

    def _evaluate(self, eval_loader: DataLoader[Any]) -> dict[str, float]:
        """Evaluate model performance"""

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = self._move_batch_to_device(batch)

                # Forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch.get("labels", input_ids)

                # Build causal mask (same logic as _training_step)
                seq_len = input_ids.size(1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).unsqueeze(0).unsqueeze(0)
                if attention_mask is not None:
                    pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    causal_mask = causal_mask * pad_mask

                ucf_metrics = self._get_batch_ucf_metrics(batch)
                system_state = self._get_batch_system_state(batch)
                coordination_state = self._get_batch_coordination_state(batch)

                logits, _kv = self.model.model(
                    tokens=input_ids,
                    mask=causal_mask,
                    ucf_metrics=ucf_metrics,
                    system_state=system_state,
                    coordination_state=coordination_state,
                )

                # Calculate loss
                loss = self._calculate_base_loss(logits, labels)

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {"eval_loss": avg_loss, "perplexity": perplexity}

        self.model.train()
        return metrics

    def _log_evaluation(self, metrics: dict[str, float]):
        """Log evaluation metrics"""

        logger.info(
            "Evaluation - loss=%.4f, perplexity=%.2f",
            metrics["eval_loss"],
            metrics["perplexity"],
        )

        # Update best loss
        if metrics["eval_loss"] < self.best_loss:
            self.best_loss = metrics["eval_loss"]
            self._save_checkpoint(best=True)

        # Log to WandB
        if WANDB_AVAILABLE and self.config.log_wandb and self.config.rank == 0:
            wandb.log(metrics, step=self.global_step)

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint"""

        weights = self.model.model.module.state_dict() if self.config.distributed else self.model.model.state_dict()
        checkpoint = {
            "global_step": self.global_step,
            # "state_dict" is the key expected by CoordinationAwareModel.load() and
            # CoordinationInference._load_model() so trained checkpoints can be used
            # for inference without any conversion step.
            "state_dict": weights,
            # Keep "model_state_dict" so load_checkpoint() (trainer resume) still works.
            "model_state_dict": weights,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            # Save the ModelConfig instance (not the full TrainingConfig) under the
            # "config" key so inference can reconstruct the exact model architecture.
            "config": self.config.model_config,
            "training_config": asdict(self.config),
            "coordination_metrics": self.coordination_metrics,
            "training_history": self.training_history,
        }

        # Save checkpoint
        if final:
            checkpoint_path = Path(self.config.checkpoint_dir) / "final_model.pt"
        elif best:
            checkpoint_path = Path(self.config.checkpoint_dir) / f"best_model_step_{self.global_step}.pt"
        else:
            checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"

        torch.save(checkpoint, checkpoint_path)
        logger.info("✅ Checkpoint saved: %s", checkpoint_path)

        # Save coordination state separately
        coordination_state_path = checkpoint_path.with_suffix(".coordination.pt")
        torch.save(
            {
                "coordination_state": self.model.get_coordination_state(),
                "system_state": self.model.get_system_state(),
            },
            coordination_state_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)  # nosec B614

        # Load model state
        if self.config.distributed:
            self.model.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.coordination_metrics = checkpoint.get("coordination_metrics", {})
        self.training_history = checkpoint.get("training_history", [])

        logger.info("✅ Checkpoint loaded from step %d", self.global_step)


__all__ = ["CoordinationTrainer", "TextFileDataset", "TrainingConfig"]


if __name__ == "__main__":
    import argparse

    from apps.backend.proprietary_llm.models import _MODEL_CONFIGS

    parser = argparse.ArgumentParser(description="Train the Helix proprietary LLM")
    parser.add_argument(
        "--model",
        default="awakening",
        choices=list(_MODEL_CONFIGS.keys()),
        help="Model size to train (default: awakening)",
    )
    parser.add_argument(
        "--data-dir",
        default="./docs",
        help="Directory containing .txt/.md/.py training files (default: ./docs)",
    )
    parser.add_argument(
        "--output",
        default="./checkpoints",
        help="Directory to write checkpoints (default: ./checkpoints)",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Training steps (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if not HAS_TORCH:
        raise SystemExit("PyTorch is not installed. Run: pip install torch")

    model_cfg = _MODEL_CONFIGS[args.model]
    dataset = TextFileDataset(args.data_dir, seq_len=args.seq_len)
    if len(dataset) == 0:
        raise SystemExit(
            f"No training files found in {args.data_dir!r}. Add .txt, .md, .py, or .rst files and try again."
        )

    cfg = TrainingConfig(
        model_config=model_cfg,
        checkpoint_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.lr,
        log_wandb=args.wandb,
    )

    logger.info("🚀 Starting Helix LLM training")
    logger.info("   Model  : %s (d_model=%d, layers=%d)", args.model, model_cfg.d_model, model_cfg.n_layers)
    logger.info("   Data   : %s (%d chunks)", args.data_dir, len(dataset))
    logger.info("   Steps  : %d", args.steps)
    logger.info("   Output : %s", args.output)

    trainer = CoordinationTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(dataset)
