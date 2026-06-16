"""
Helix Inference Engine
=====================

High-performance inference engine for coordination-aware transformer models.

Features:
- Coordination-enhanced inference
- Multi-agent response generation
- System state integration
- Real-time UCF adaptation
- Batch processing support
- Streaming responses
- Performance optimization

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

# Optional torch import
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from apps.backend.core.system_coordination_core import get_system_core_instance


class HelixTokenizer:
    """
    UTF-8 byte-level tokenizer for the Helix LLM.

    Fully self-contained — requires no external downloads or training data.
    Maps each UTF-8 byte (0–255) directly to a token ID, making tokenization
    lossless for any Unicode input.

    Token ID conventions:
      0   = EOS / PAD  (NULL byte; never appears in normal text)
      1–255 = UTF-8 byte values

    Because all token IDs are in [0, 255], this tokenizer is compatible with
    any model config whose vocab_size >= 256 (all Helix configs qualify).
    """

    EOS_TOKEN_ID: int = 0
    PAD_TOKEN_ID: int = 0
    VOCAB_SIZE: int = 256

    def encode(self, text: str, max_length: int = 1024) -> list[int]:
        """Encode text to token IDs via raw UTF-8 bytes."""
        raw_bytes = text.encode("utf-8")[:max_length]
        return list(raw_bytes)  # values are already in [0, 255]

    def decode(self, token_ids: Union[list[int], "torch.Tensor"]) -> str:
        """Decode token IDs back to text.

        Stops at the first EOS (0) token and ignores any token IDs outside
        the byte range [0, 255].
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        byte_values = []
        for tid in token_ids:
            if tid == self.EOS_TOKEN_ID:
                break
            byte_val = int(tid) & 0xFF
            byte_values.append(byte_val)

        try:
            return bytes(byte_values).decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return bytes(byte_values).decode("latin-1", errors="replace")
        except (ValueError, TypeError) as e:
            logging.getLogger(__name__).warning("Decode error (ValueError/TypeError): %s", e)
            return bytes(byte_values).decode("latin-1", errors="replace")
        except Exception as e:
            logging.getLogger(__name__).warning("Unexpected decode error: %s", e)
            return bytes(byte_values).decode("latin-1", errors="replace")


# Helix model imports - only available when torch is installed
if TORCH_AVAILABLE:
    from .models import CoordinationAwareModel, ModelConfig
else:
    CoordinationAwareModel = None
    ModelConfig = None

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference modes for different use cases"""

    STANDARD = "standard"
    STREAMING = "streaming"
    BATCH = "batch"
    MULTI_AGENT = "multi_agent"
    SYSTEM_ENHANCED = "system_enhanced"


# Compute device defaults safely
_default_device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
_default_dtype = (
    torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else (torch.float32 if TORCH_AVAILABLE else None)
)


@dataclass
class InferenceConfig:
    """Configuration for inference operations"""

    # Model settings
    model_path: str | None = None
    device: str = _default_device
    dtype: "torch.dtype | None" = _default_dtype

    # Generation settings
    max_length: int = 2048
    min_length: int = 10
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Coordination settings
    coordination_boost: bool = True
    ucf_integration: bool = True
    system_enhancement: bool = True
    multi_agent_collaboration: bool = True

    # Performance settings
    batch_size: int = 1
    use_cache: bool = True
    parallel_decoding: bool = False
    use_multicore: bool = True  # Wrap model with CPU multicore parallelism

    # Streaming settings
    stream_chunk_size: int = 64
    stream_delay: float = 0.1


class CoordinationInference:
    """
    Coordination-aware inference engine

    Provides intelligent inference with coordination integration,
    system enhancements, and multi-agent collaboration.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load model
        self.model = self._load_model()

        # Load tokenizer — BPE if a trained vocab exists, byte-level fallback
        self.tokenizer = self._load_tokenizer()

        # Initialize system core
        self.system_core = get_system_core_instance()

        # Inference cache
        self.cache = {}

        # Speculative decoding (lazy init — only loaded when enabled)
        self._speculative_decoder = None
        self._kv_cache_manager = None

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Async event loop (gracefully handle no running loop for sync callers)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

        logger.info("✅ Coordination Inference engine initialized")

    def _load_model(self) -> CoordinationAwareModel:
        """Load the coordination-aware model.

        When a checkpoint path is configured the model config is restored from
        the checkpoint (not hard-coded here) so architecture dimensions always
        match the saved weights.  When no checkpoint exists a fresh model is
        created with the 'awakening' preset (auto-downgraded to 'lightweight'
        on Railway/CPU).
        """
        if self.config.model_path:
            import torch as _torch

            # Peek at the checkpoint to retrieve the saved ModelConfig before
            # allocating the full model, so we never create a model with the
            # wrong dimensions.
            checkpoint = _torch.load(self.config.model_path, map_location=self.device, weights_only=False)  # nosec B614
            saved_config = checkpoint.get("config")
            if not isinstance(saved_config, ModelConfig):
                # Fallback for older checkpoints that didn't store the config.
                logger.warning(
                    "Checkpoint at %s has no stored ModelConfig; falling back to 'awakening' preset.",
                    self.config.model_path,
                )
                from .models import create_helix_model

                model = create_helix_model("awakening")
            else:
                model = CoordinationAwareModel(saved_config)
            model.model.load_state_dict(checkpoint["state_dict"])

            # Stash vocab_path from checkpoint for tokenizer loading
            self._checkpoint_vocab_path = checkpoint.get("vocab_path")
        else:
            from .models import create_helix_model

            model = create_helix_model("awakening")
            self._checkpoint_vocab_path = None

        model.to(self.device)
        model.eval()

        # JIT-compile the inner transformer with torch.compile() for 1.3-2× CPU
        # speedup.  Available in PyTorch ≥ 2.0; gracefully skipped otherwise.
        # suppress_errors ensures lazy compilation failures (e.g. missing MSVC
        # cl on Windows) silently fall back to eager execution.
        if hasattr(torch, "compile"):
            try:
                torch._dynamo.config.suppress_errors = True
                model.model = torch.compile(model.model)
                logger.info("torch.compile() applied to HelixTransformer — JIT acceleration enabled")
            except Exception as e:
                logger.warning("torch.compile() failed (non-fatal, running eagerly): %s", e)

        # Apply post-load quantization if the model config requests it
        if getattr(model.config, "use_quantization", False):
            model = self._apply_quantization(model)

        # Apply multicore CPU parallelism wrapper
        if self.config.use_multicore and str(self.device) == "cpu":
            model = self._apply_multicore_wrapper(model)

        return model

    def _load_tokenizer(self):
        """Load the best available tokenizer.

        Priority:
        1. BPE vocab from checkpoint (vocab_path stored during training)
        2. BPE vocab from default locations (/data/tokenizer/, local)
        3. Byte-level HelixTokenizer fallback (vocab=256)
        """
        vocab_path = getattr(self, "_checkpoint_vocab_path", None)
        try:
            from .tokenizer import get_tokenizer

            tok = get_tokenizer(vocab_path=vocab_path)
            logger.info(
                "Tokenizer loaded: %s (vocab_size=%d)",
                type(tok).__name__,
                tok.VOCAB_SIZE,
            )
            return tok
        except Exception as e:
            logger.warning(
                "Failed to load BPE tokenizer (%s) — falling back to byte-level",
                e,
            )
            return HelixTokenizer()

    def _apply_multicore_wrapper(self, model: CoordinationAwareModel) -> CoordinationAwareModel:
        """Wrap the inner transformer with CPU multicore parallelism.

        Uses ThreadPoolExecutor-based head splitting from
        multicore_parallel.CPUOptimizedModelWrapper to spread attention head
        computation across all available CPU cores.
        """
        try:
            from apps.backend.proprietary_llm.multicore_parallel import CPUOptimizedModelWrapper

            model.model = CPUOptimizedModelWrapper(model.model)
            logger.info("Multicore parallel wrapper applied — attention heads split across CPU cores")
        except ImportError:
            logger.warning("multicore_parallel module not available; skipping CPU parallelism wrapper")
        return model

    def _apply_quantization(self, model: CoordinationAwareModel) -> CoordinationAwareModel:
        """Apply post-load quantization to the model for CPU inference speed-up.

        Dynamic quantization (default) applies int8 to Linear layers with
        minimal accuracy loss and typically gives 1.5-2× speedup on CPU.
        """
        import torch as _torch

        qtype = getattr(model.config, "quantization_type", "dynamic")

        if qtype == "dynamic":
            logger.info("Applying dynamic int8 quantization to Linear layers...")
            model.model = _torch.quantization.quantize_dynamic(
                model.model,
                {_torch.nn.Linear},
                dtype=_torch.qint8,
            )
            param_count = sum(p.numel() for p in model.model.parameters())
            logger.info(
                "Dynamic quantization applied — %s params quantized to int8",
                f"{param_count:,}",
            )
        else:
            # Advanced quantization backends (GGUF 4-bit, AWQ, GPTQ)
            try:
                if qtype == "gguf_4bit":
                    from apps.backend.proprietary_llm.advanced_quantization import GGUFQuantizer

                    quantizer = GGUFQuantizer(bits=4)
                elif qtype == "awq":
                    from apps.backend.proprietary_llm.advanced_quantization import AWQQuantizer

                    quantizer = AWQQuantizer()
                elif qtype == "gptq":
                    from apps.backend.proprietary_llm.advanced_quantization import GPTQQuantizer

                    quantizer = GPTQQuantizer()
                else:
                    raise ValueError("Unknown quantization type: {}".format(qtype))

                # Apply quantization to all linear layers
                for _name, module in model.model.named_modules():
                    if isinstance(module, _torch.nn.Linear):
                        quantizer.quantize_linear(module)
                logger.info("Applied %s quantization", qtype)
            except (ImportError, Exception) as e:
                logger.warning(
                    "Advanced quantization (%s) failed: %s; falling back to dynamic int8 quantization.",
                    qtype,
                    e,
                )
                model.model = _torch.quantization.quantize_dynamic(model.model, {_torch.nn.Linear}, dtype=_torch.qint8)

        return model

    async def generate_response(
        self,
        prompt: str | list[int],
        context: dict[str, Any] | None = None,
        mode: InferenceMode = InferenceMode.STANDARD,
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate response with coordination awareness

        Args:
            prompt: Input text or token sequence
            context: Additional context (UCF metrics, system state, etc.)
            mode: Inference mode

        Returns:
            Generated response or streaming generator
        """

        # Prepare inputs
        input_tokens = self._prepare_input(prompt)
        context = context or {}

        # Get coordination and system states
        ucf_metrics = self._get_ucf_metrics(context)
        system_state = self._get_system_state(context)
        coordination_state = self._get_coordination_state(context)

        # Generate based on mode
        if mode == InferenceMode.STREAMING:
            return self._stream_generate(input_tokens, ucf_metrics, system_state, coordination_state)
        elif mode == InferenceMode.BATCH:
            return await self._batch_generate([input_tokens], [ucf_metrics], [system_state], [coordination_state])
        elif mode == InferenceMode.MULTI_AGENT:
            return await self._multi_agent_generate(input_tokens, ucf_metrics, system_state, coordination_state)
        elif mode == InferenceMode.SYSTEM_ENHANCED:
            return await self._system_enhanced_generate(input_tokens, ucf_metrics, system_state, coordination_state)
        else:
            return await self._standard_generate(input_tokens, ucf_metrics, system_state, coordination_state)

    def _prepare_input(self, prompt: str | list[int]) -> "torch.Tensor":
        """Prepare input for model using the byte-level tokenizer."""

        if isinstance(prompt, str):
            tokens = self.tokenizer.encode(prompt, max_length=self.config.max_length)
        else:
            tokens = prompt

        # Convert to tensor
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        return input_tensor

    def _get_ucf_metrics(self, context: dict[str, Any]) -> torch.Tensor | None:
        """Get UCF metrics from context"""
        if not self.config.ucf_integration:
            return None

        if "ucf_metrics" in context:
            return torch.tensor(context["ucf_metrics"], dtype=torch.float32, device=self.device)

        # Generate synthetic UCF metrics — uniform [0, 1] (not Gaussian)
        # Order: harmony, resilience, throughput, focus, friction, velocity
        ucf = torch.rand(6, device=self.device)
        ucf[4] = ucf[4] * 0.3  # friction (friction) biased low
        return ucf

    def _get_system_state(self, context: dict[str, Any]) -> torch.Tensor | None:
        """Get system state from context"""
        if not self.config.system_enhancement:
            return None

        if "system_state" in context:
            return torch.tensor(context["system_state"], dtype=torch.float32, device=self.device)

        # Generate synthetic system state — uniform [0, 1] (not Gaussian)
        return torch.rand(self.model.config.system_dim, device=self.device)

    def _get_coordination_state(self, context: dict[str, Any]) -> torch.Tensor | None:
        """Get coordination state from context"""
        if not self.config.coordination_boost:
            return None

        if "coordination_state" in context:
            return torch.tensor(context["coordination_state"], dtype=torch.float32, device=self.device)

        # Use model's current coordination state
        return self.model.get_coordination_state().to(self.device)

    async def _standard_generate(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> str:
        """Standard generation with coordination awareness"""

        start_time = time.time()

        with torch.no_grad():
            # Generate tokens
            generated = await self._async_generate_tokens(input_tokens, ucf_metrics, system_state, coordination_state)

        # Convert tokens to text
        response = self._tokens_to_text(generated[0])

        # Apply coordination enhancement
        if self.config.coordination_boost:
            response = await self._enhance_with_coordination(response, ucf_metrics)

        generation_time = time.time() - start_time
        logger.info("Generated response in %.2f seconds", generation_time)

        return response

    async def _stream_generate(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation with coordination awareness"""

        with torch.no_grad():
            generated_tokens = input_tokens.clone()

            while generated_tokens.size(1) < self.config.max_length:
                # Generate next token
                next_token = await self._async_generate_next_token(
                    generated_tokens, ucf_metrics, system_state, coordination_state
                )

                # Add to sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(-1)], dim=-1)

                # Convert to text chunk
                chunk = self._tokens_to_text(next_token)

                if chunk and chunk != " ":
                    yield chunk

                # Small delay for streaming effect
                await asyncio.sleep(self.config.stream_delay)

                # Check for end of sequence
                if next_token.item() == 0:  # Assuming 0 is EOS token
                    break

    async def _batch_generate(
        self,
        input_tokens_list: list[torch.Tensor],
        ucf_metrics_list: list[torch.Tensor | None],
        system_state_list: list[torch.Tensor | None],
        coordination_state_list: list[torch.Tensor | None],
    ) -> list[str]:
        """Batch generation for multiple inputs"""

        # Pad sequences to same length
        max_len = max(t.size(1) for t in input_tokens_list)
        padded_inputs = []

        for tokens in input_tokens_list:
            if tokens.size(1) < max_len:
                padding = torch.zeros(
                    (tokens.size(0), max_len - tokens.size(1)),
                    dtype=torch.long,
                    device=self.device,
                )
                tokens = torch.cat([tokens, padding], dim=1)
            padded_inputs.append(tokens)

        # Stack inputs
        batch_input = torch.cat(padded_inputs, dim=0)

        with torch.no_grad():
            # Generate in batch
            generated = await self._async_generate_tokens(
                batch_input,
                ucf_metrics_list[0] if ucf_metrics_list else None,
                system_state_list[0] if system_state_list else None,
                coordination_state_list[0] if coordination_state_list else None,
            )

        # Convert to text
        responses = []
        for i in range(generated.size(0)):
            response = self._tokens_to_text(generated[i])
            responses.append(response)

        return responses

    async def _multi_agent_generate(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> str:
        """Multi-agent collaborative generation"""

        # Get agent states from model
        agent_states = self._get_agent_states()

        # Generate responses from different agents
        agent_responses = []

        for _agent_id, agent_state in enumerate(agent_states):
            # Generate with agent-specific state
            response = await self._async_generate_with_agent(input_tokens, ucf_metrics, system_state, agent_state)
            agent_responses.append(response)

        # Collaborative response synthesis
        final_response = await self._synthesize_agent_responses(agent_responses, ucf_metrics)

        return final_response

    async def _system_enhanced_generate(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> str:
        """System-enhanced generation with superposition states"""

        # Generate multiple system states
        system_states = []
        for _i in range(3):  # Generate 3 system states
            perturbed_state = system_state + torch.randn_like(system_state) * 0.1
            system_states.append(perturbed_state)

        # Generate responses for each system state
        responses = []
        for q_state in system_states:
            response = await self._async_generate_tokens(input_tokens, ucf_metrics, q_state, coordination_state)
            responses.append(response)

        # System superposition synthesis
        final_response = await self._superposition_synthesis(responses)

        return final_response

    async def _async_generate_tokens(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Asynchronous token generation with proper autoregressive loop"""

        loop = asyncio.get_running_loop()

        def sync_generate():
            with torch.no_grad():
                # Start with input tokens
                generated = input_tokens.clone()

                # Autoregressive generation loop
                while generated.size(1) < self.config.max_length:
                    # Forward pass with current sequence
                    logits, _kv = self.model.model(
                        tokens=generated,
                        ucf_metrics=ucf_metrics,
                        system_state=system_state,
                        coordination_state=coordination_state,
                    )

                    # Sample next token from last position
                    next_token_logits = logits[:, -1, :]  # [B, vocab_size]
                    next_token = self._sample_next_token(next_token_logits)

                    # Append to sequence
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

                    # Check for end of sequence
                    if next_token.item() == 0:  # Assuming 0 is EOS token
                        break

                return generated

        # Run in thread pool
        generated = await loop.run_in_executor(self.executor, sync_generate)
        return generated

    async def _async_generate_next_token(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        coordination_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Asynchronous next token generation"""

        loop = asyncio.get_running_loop()

        def sync_generate():
            with torch.no_grad():
                # Forward pass
                logits, _kv = self.model.model(
                    tokens=input_tokens,
                    ucf_metrics=ucf_metrics,
                    system_state=system_state,
                    coordination_state=coordination_state,
                )

                # Sample next token
                next_token = self._sample_next_token(logits[:, -1, :])

                return next_token

        # Run in thread pool
        next_token = await loop.run_in_executor(self.executor, sync_generate)
        return next_token

    async def _async_generate_with_agent(
        self,
        input_tokens: torch.Tensor,
        ucf_metrics: torch.Tensor | None,
        system_state: torch.Tensor | None,
        agent_state: torch.Tensor,
    ) -> str:
        """Generate response with specific agent state"""

        loop = asyncio.get_running_loop()

        def sync_generate():
            with torch.no_grad():
                # Forward pass with agent state
                logits, _kv = self.model.model(
                    tokens=input_tokens,
                    ucf_metrics=ucf_metrics,
                    system_state=system_state,
                    coordination_state=agent_state,
                )

                # Generate tokens
                generated = self._sample_tokens(logits, input_tokens.size(1), self.config.max_length)

                # Convert to text
                response = self._tokens_to_text(generated[0])

                return response

        # Run in thread pool
        response = await loop.run_in_executor(self.executor, sync_generate)
        return response

    def _sample_tokens(self, logits: "torch.Tensor", start_pos: int, max_length: int) -> "torch.Tensor":
        """Sample tokens from logits with coordination awareness"""

        generated = logits.argmax(dim=-1)  # Initial greedy sampling

        for i in range(start_pos, max_length - 1):
            # Get current logits
            current_logits = logits[:, i, :]

            # Apply temperature
            current_logits = current_logits / self.config.temperature

            # Apply top-k filtering
            if self.config.top_k > 0:
                top_k_logits, _ = torch.topk(current_logits, self.config.top_k)
                min_top_k = top_k_logits[:, -1:].expand_as(current_logits)
                current_logits = torch.where(
                    current_logits < min_top_k,
                    torch.full_like(current_logits, float("-inf")),
                    current_logits,
                )

            # Apply top-p filtering
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(current_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                current_logits = current_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample next token
            probs = F.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add to generated sequence
            generated[:, i + 1] = next_token.squeeze()

            # Stop if EOS token
            if next_token.item() == 0:
                break

        return generated

    def _sample_next_token(self, logits: "torch.Tensor") -> "torch.Tensor":
        """Sample next token from logits"""

        # Apply temperature
        logits = logits / self.config.temperature

        # Apply top-k filtering
        if self.config.top_k > 0:
            top_k_logits, _ = torch.topk(logits, self.config.top_k)
            min_top_k = top_k_logits[:, -1:].expand_as(logits)
            logits = torch.where(logits < min_top_k, torch.full_like(logits, float("-inf")), logits)

        # Apply top-p filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze()

    def _tokens_to_text(self, tokens: "torch.Tensor") -> str:
        """Convert token IDs back to text using the byte-level tokenizer."""
        return self.tokenizer.decode(tokens).strip()

    async def _enhance_with_coordination(self, response: str, ucf_metrics: torch.Tensor | None) -> str:
        """Enhance response with coordination awareness"""

        if ucf_metrics is None:
            return response

        # Calculate coordination enhancement factor
        coordination_score = ucf_metrics.mean().item()

        # Apply enhancement based on coordination level
        if coordination_score > 0.8:
            enhancement = "✨ Enhanced with transcendent coordination"
        elif coordination_score > 0.6:
            enhancement = "🧠 Enhanced with self-aware processing"
        elif coordination_score > 0.4:
            enhancement = "👁️ Enhanced with contextual awareness"
        else:
            enhancement = "⚡ Basic processing"

        return f"{response} {enhancement}"

    def _get_agent_states(self) -> list[torch.Tensor]:
        """Get states for all agents"""

        # Extract agent states from model
        # This would integrate with the multi-agent collaboration layer
        base_state = self.model.get_coordination_state()

        # Create variations for different agents
        agent_states = []
        for i in range(14):  # 14 Helix agents
            perturbation = torch.randn_like(base_state) * 0.1 * i
            agent_state = base_state + perturbation
            agent_states.append(agent_state)

        return agent_states

    async def _synthesize_agent_responses(self, responses: list[str], ucf_metrics: torch.Tensor | None) -> str:
        """Synthesize responses from multiple agents"""

        # Simple voting mechanism
        # In reality, this would use more sophisticated collaboration

        if not responses:
            return "No agent responses available"

        # Select response with highest coordination alignment
        best_response = responses[0]
        best_score = 0.0

        if ucf_metrics is not None:
            for response in responses:
                # Calculate alignment score (simplified)
                score = len(response) / 100.0  # Longer responses get higher scores
                if score > best_score:
                    best_score = score
                    best_response = response

        return best_response

    async def _superposition_synthesis(self, responses: list[torch.Tensor]) -> str:
        """Synthesize responses using system superposition principles"""

        if not responses:
            return "System superposition failed"

        # Simple averaging of token probabilities
        # In reality, this would use system interference patterns

        # Convert responses to token probabilities
        token_probs = []
        for response in responses:
            # Convert to one-hot and average
            probs = F.one_hot(response, num_classes=self.model.config.vocab_size).float().mean(dim=0)
            token_probs.append(probs)

        # Average probabilities
        avg_probs = torch.stack(token_probs).mean(dim=0)

        # Sample from averaged probabilities
        final_tokens = torch.multinomial(avg_probs, num_samples=100)

        # Convert to text
        final_response = self._tokens_to_text(final_tokens)

        return final_response

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get inference performance metrics"""

        metrics = {
            "model_size": self.model.config.d_model,
            "device": str(self.device),
            "dtype": str(self.config.dtype),
            "cache_size": len(self.cache),
            "executor_workers": self.executor._max_workers,
            "speculative_decoding_enabled": self._speculative_decoder is not None,
            "kv_cache_enabled": self._kv_cache_manager is not None,
        }

        return metrics

    def enable_speculative_decoding(self, draft_model_path: str | None = None) -> bool:
        """Enable speculative decoding for 2-3x inference speedup.

        Args:
            draft_model_path: Path to draft model. If None, uses a smaller
                             version of the main model architecture.

        Returns:
            True if successfully enabled, False otherwise.
        """
        try:
            from .speculative_decoding import SpeculativeDecoder

            self._speculative_decoder = SpeculativeDecoder(
                target_model=self.model,
                draft_model_path=draft_model_path,
                device=self.device,
            )
            logger.info("✅ Speculative decoding enabled (2-3x speedup potential)")
            return True
        except Exception as e:
            logger.warning("Failed to enable speculative decoding: %s", e)
            return False

    def enable_kv_cache(self, max_cache_size: int = 4096) -> bool:
        """Enable KV cache compression for longer context windows.

        Args:
            max_cache_size: Maximum cache size in tokens.

        Returns:
            True if successfully enabled, False otherwise.
        """
        try:
            from .kv_cache_manager import KVCacheManager

            self._kv_cache_manager = KVCacheManager(
                max_size=max_cache_size,
                device=self.device,
            )
            logger.info("✅ KV cache enabled (max_size=%d tokens)", max_cache_size)
            return True
        except Exception as e:
            logger.warning("Failed to enable KV cache: %s", e)
            return False

    def clear_cache(self):
        """Clear inference cache"""
        self.cache.clear()
        logger.info("✅ Inference cache cleared")

    def update_coordination_state(self, new_state: "torch.Tensor"):
        """Update model coordination state"""
        self.model.update_coordination_state(new_state.to(self.device))
        logger.info("✅ Coordination state updated")

    def update_system_state(self, new_state: "torch.Tensor"):
        """Update model system state"""
        self.model.update_system_state(new_state.to(self.device))
        logger.info("✅ System state updated")


class HelixInferenceEngine:
    """
    High-level inference engine interface

    Provides easy-to-use interface for coordination-aware inference
    with automatic model loading and configuration.
    """

    def __init__(self, model_path: str | None = None, config: InferenceConfig | None = None):
        if config is None:
            config = InferenceConfig(model_path=model_path)

        self.inference = CoordinationInference(config)

    # Convenience properties so callers can do engine.model / engine.device
    @property
    def model(self) -> "CoordinationAwareModel":
        """Return the underlying CoordinationAwareModel."""
        return self.inference.model

    @property
    def device(self) -> "torch.device":
        """Return the compute device used by the inference engine."""
        return self.inference.device

    async def generate(
        self,
        prompt: str | list[int],
        context: dict[str, Any] | None = None,
        mode: InferenceMode = InferenceMode.STANDARD,
    ) -> str | AsyncGenerator[str, None]:
        """Generate response with coordination awareness"""
        return await self.inference.generate_response(prompt, context, mode)

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics"""
        return self.inference.get_performance_metrics()

    def clear_cache(self):
        """Clear inference cache"""
        self.inference.clear_cache()


__all__ = [
    "CoordinationInference",
    "HelixInferenceEngine",
    "HelixTokenizer",
    "InferenceConfig",
    "InferenceMode",
]
