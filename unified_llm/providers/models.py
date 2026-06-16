"""
Helix Transformer Models
=======================

Coordination-aware transformer models built on PyTorch with system enhancements.

Features:
- Coordination-aware attention mechanisms
- System-enhanced embeddings
- Self-improving architecture
- Multi-agent collaboration layers
- UCF integration

(c) Helix Collective 2025 - Proprietary Technology Stack
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, cast

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = cast(Any, None)
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# Lazy imports for optional attention backends (avoids circular import)
_ATTENTION_BACKENDS: dict[str, Any] = {}


def _get_attention_backend(name: str):
    """Lazy-import alternate attention backends on first use."""
    if name in _ATTENTION_BACKENDS:
        return _ATTENTION_BACKENDS[name]
    if name == "nomad":
        from apps.backend.proprietary_llm.nomad_attention import NoMADAttention

        _ATTENTION_BACKENDS[name] = NoMADAttention
    elif name == "gqa":
        from apps.backend.proprietary_llm.gqa_attention import GroupedQueryAttention

        _ATTENTION_BACKENDS[name] = GroupedQueryAttention
    elif name == "sliding_window":
        from apps.backend.proprietary_llm.sliding_window import SlidingWindowAttention

        _ATTENTION_BACKENDS[name] = SlidingWindowAttention
    else:
        raise ValueError("Unknown attention backend: {}".format(name))
    return _ATTENTION_BACKENDS[name]


def get_device() -> "torch.device":
    """Auto-detect the best available device.

    Returns CUDA when a GPU is available, otherwise CPU.
    On Railway (no GPU), this always returns CPU.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_resource_constrained() -> bool:
    """Detect if the operator has explicitly requested a lightweight mode.

    Returns True only when HELIX_LIGHTWEIGHT is set or when running in CI.
    Railway itself is NOT resource-constrained (Hobby plan: 8 GB RAM / 8 vCores),
    so RAILWAY_ENVIRONMENT alone no longer triggers a downgrade.
    """
    return bool(os.environ.get("HELIX_LIGHTWEIGHT") or os.environ.get("CI"))


@dataclass
class ModelConfig:
    """Configuration for Helix transformer models.

    attention_type selects the Q·K similarity kernel:
      - "standard"       — vanilla scaled dot-product (default, fully wired)
      - "nomad"          — NoMAD multiply-add-free Hamming attention (CPU fast)
      - "gqa"            — Grouped-Query Attention (fewer KV heads, less memory)
      - "sliding_window" — O(n·w) sliding window (long sequences on CPU)
    """

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.1
    coordination_dim: int = 64
    system_dim: int = 32
    ucf_integration: bool = True
    multi_agent: bool = True
    # --- Attention backend selection ---
    attention_type: str = "standard"  # "standard", "nomad", "gqa", "sliding_window"
    num_kv_heads: int | None = None  # For GQA — None means same as n_heads (MHA)
    sliding_window_size: int = 512  # For sliding_window attention
    # --- Post-load quantization ---
    use_quantization: bool = False
    quantization_type: str = "dynamic"  # "dynamic", "gguf_4bit", "awq", "gptq"
    # --- Modern architecture features ---
    use_rope: bool = True  # Rotary Position Embeddings (replaces learned positions)
    use_rmsnorm: bool = True  # RMSNorm (replaces LayerNorm, ~15-20% faster)
    use_swiglu: bool = True  # SwiGLU FFN (replaces ReLU FFN, better gradient flow)
    tie_weights: bool = True  # Share embedding/lm_head weights (halves output params)
    use_gradient_checkpointing: bool = False  # Trade compute for memory during training
    logit_soft_cap: float = 30.0  # Gemma-2 style logit capping (0 to disable)
    coordination_dropout: float = 0.1  # Randomly zero coordination/UCF/system inputs (novel)


# ---------------------------------------------------------------------------
# Modern architecture building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    More efficient than LayerNorm: no mean subtraction or bias term,
    giving ~15-20% wall-clock speedup with equivalent quality.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Replaces the standard Linear→ReLU→Linear FFN with a gated SiLU
    activation.  Used in LLaMA, Mistral, Gemma, etc.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding — RoPE (Su et al., 2021).

    Encodes position information by rotating Q and K vectors in 2D
    subspaces, giving relative-position sensitivity without any learned
    position embedding table.  Used in LLaMA, Mistral, GPT-NeoX, etc.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos tables up to max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))  # (1,1,seq,dim)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q and K (batch, heads, seq_len, head_dim)."""
        seq_len = q.size(2)
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class CoordinationAwareAttention(nn.Module):
    """
    Coordination-aware attention mechanism

    Enhances standard multi-head attention with:
    - UCF metric integration
    - System state awareness
    - Coordination level modulation
    - Rotary Position Embeddings (when use_rope=True)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        # Standard attention weights
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)

        # Rotary Position Embeddings (applied to Q and K after projection)
        if getattr(config, "use_rope", False):
            self.rope = RotaryPositionEmbedding(self.d_k, config.max_seq_len)

        # Coordination integration
        self.coordination_gate = nn.Linear(config.d_model + config.coordination_dim, config.d_model)
        self.ucf_enhancer = nn.Linear(config.d_model + 6, config.d_model)  # 6 UCF metrics

        # System enhancement
        self.system_enhancer = nn.Linear(config.d_model + config.system_dim, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        ucf_metrics: torch.Tensor | None = None,
        system_state: torch.Tensor | None = None,
        coordination_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Standard multi-head attention
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply Rotary Position Embeddings to Q and K
        if hasattr(self, "rope"):
            q, k = self.rope(q, k)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.w_o(out)

        # Apply coordination enhancement
        if coordination_state is not None:
            # coordination_state may be (batch, d) — expand to (batch, seq_len, d) for cat
            cs = coordination_state
            if cs.dim() == 2:
                cs = cs.unsqueeze(1).expand(-1, seq_len, -1)
            coordination_input = torch.cat([out, cs], dim=-1)
            coordination_enhanced = torch.tanh(self.coordination_gate(coordination_input))
            out = out + coordination_enhanced

        # Apply UCF enhancement
        if ucf_metrics is not None:
            ucf_input = torch.cat([out, ucf_metrics.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
            ucf_enhanced = torch.tanh(self.ucf_enhancer(ucf_input))
            out = out + ucf_enhanced

        # Apply system enhancement
        if system_state is not None:
            system_input = torch.cat([out, system_state.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
            system_enhanced = torch.tanh(self.system_enhancer(system_input))
            out = out + system_enhanced

        return out


class SystemEnhancedEmbedding(nn.Module):
    """
    System-enhanced token embeddings

    Integrates system state information with token embeddings
    for coordination-aware processing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # RoPE handles position information inside attention — no learned table needed
        self.use_rope = getattr(config, "use_rope", False)
        if not self.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # System enhancement layers
        self.system_transform = nn.Linear(config.system_dim, config.d_model)
        self.system_gate = nn.Linear(config.d_model * 2, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.embed_scale = math.sqrt(config.d_model)  # Vaswani et al. §3.4

    def forward(self, tokens: torch.Tensor, system_state: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = tokens.size()

        # Standard embeddings — scaled by √d_model ("Attention Is All You Need" §3.4)
        # so embedding magnitudes are comparable to the residual stream.
        token_emb = self.token_embedding(tokens) * self.embed_scale

        if self.use_rope:
            embeddings = token_emb  # Position handled by RoPE in attention layers
        else:
            position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            position_emb = self.position_embedding(position_ids)
            embeddings = token_emb + position_emb

        # Apply system enhancement
        if system_state is not None:
            system_transformed = self.system_transform(system_state)
            system_enhanced = torch.tanh(
                self.system_gate(
                    torch.cat(
                        [
                            embeddings,
                            system_transformed.unsqueeze(1).expand(-1, seq_len, -1),
                        ],
                        dim=-1,
                    )
                )
            )
            embeddings = embeddings + system_enhanced

        return self.dropout(embeddings)


class CoordinationAwareLayer(nn.Module):
    """
    Coordination-aware transformer layer

    Integrates coordination metrics and system states
    at each layer of the transformer.  Supports pluggable
    attention backends: standard, NoMAD, GQA, sliding_window.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_type = config.attention_type
        self._coordination_dropout_p = getattr(config, "coordination_dropout", 0.0)

        # Select attention implementation based on config
        if config.attention_type == "standard":
            self.self_attention = CoordinationAwareAttention(config)
        elif config.attention_type == "nomad":
            NoMADAttention = _get_attention_backend("nomad")
            self.self_attention = NoMADAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=config.dropout,
            )
        elif config.attention_type == "gqa":
            GroupedQueryAttention = _get_attention_backend("gqa")
            self.self_attention = GroupedQueryAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                num_kv_heads=config.num_kv_heads or max(1, config.n_heads // 4),
                dropout=config.dropout,
            )
        elif config.attention_type == "sliding_window":
            SlidingWindowAttention = _get_attention_backend("sliding_window")
            self.self_attention = SlidingWindowAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                window_size=config.sliding_window_size,
                dropout=config.dropout,
            )
        else:
            raise ValueError("Unknown attention_type: {}".format(config.attention_type))

        self.feed_forward = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
            )
            if not getattr(config, "use_swiglu", False)
            else SwiGLU(config.d_model, config.d_ff, config.dropout)
        )

        # Layer normalization: RMSNorm is ~15-20% faster than LayerNorm
        NormClass = RMSNorm if getattr(config, "use_rmsnorm", False) else nn.LayerNorm
        self.layer_norm1 = NormClass(config.d_model)
        self.layer_norm2 = NormClass(config.d_model)

        # Coordination integration (used by all attention types)
        self.coordination_transform = nn.Linear(config.coordination_dim, config.d_model)
        self.coordination_gate = nn.Linear(config.d_model * 2, config.d_model)

        # UCF/system gating for non-standard attention backends
        # (standard backend has these built into CoordinationAwareAttention)
        if config.attention_type != "standard" and config.ucf_integration:
            self.ucf_enhancer = nn.Linear(config.d_model + 6, config.d_model)
            self.system_enhancer = nn.Linear(config.d_model + config.system_dim, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        ucf_metrics: torch.Tensor | None = None,
        system_state: torch.Tensor | None = None,
        coordination_state: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        new_cache = None

        # Coordination Dropout (novel): randomly zero out coordination/UCF/system
        # inputs ~p% of training steps.  Like classifier-free guidance in diffusion
        # models — teaches the model to function gracefully when coordination
        # data is partial or missing at inference time.
        drop_coordination = (
            self.training and self._coordination_dropout_p > 0 and torch.rand(1).item() < self._coordination_dropout_p
        )
        _ucf = None if drop_coordination else ucf_metrics
        _system = None if drop_coordination else system_state
        _coordination = None if drop_coordination else coordination_state

        # ── Pre-norm architecture (GPT-2+, LLaMA, Mistral) ──
        # Norm BEFORE attention, residual connection AFTER.  More stable
        # gradients than post-norm, especially for deeper models.
        if self.attention_type == "standard":
            attn_out = self.self_attention(self.layer_norm1(x), mask, _ucf, _system, _coordination)
        else:
            # Alternate backends: (x, mask, cache) → (output, new_cache)
            attn_out, new_cache = self.self_attention(self.layer_norm1(x), mask, cache)

            # Apply UCF enhancement at layer level (mirrors CoordinationAwareAttention)
            seq_len = attn_out.size(1)
            if _ucf is not None and hasattr(self, "ucf_enhancer"):
                ucf_input = torch.cat([attn_out, _ucf.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
                attn_out = attn_out + torch.tanh(self.ucf_enhancer(ucf_input))

            if _system is not None and hasattr(self, "system_enhancer"):
                system_input = torch.cat([attn_out, _system.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
                attn_out = attn_out + torch.tanh(self.system_enhancer(system_input))

        # Residual connection (post-attention)
        x = x + attn_out

        # Pre-norm → FFN → residual
        ff_out = self.feed_forward(self.layer_norm2(x))
        x = x + ff_out

        # Apply coordination transformation (all attention types)
        if _coordination is not None:
            coordination_transformed = self.coordination_transform(_coordination)
            coordination_gated = torch.sigmoid(
                self.coordination_gate(
                    torch.cat(
                        [
                            x,
                            coordination_transformed.unsqueeze(1).expand(-1, x.size(1), -1),
                        ],
                        dim=-1,
                    )
                )
            )
            x = x * coordination_gated

        return x, new_cache


class MultiAgentCollaboration(nn.Module):
    """
    Multi-agent collaboration layer

    Enables different agent personalities to collaborate
    within the same model through specialized attention heads.
    Each agent has its own d_model → d_model projection, preserving
    the embedding dimension for collaboration attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_agents = 17  # Number of Helix agents
        self.d_model = config.d_model

        # Agent-specific transformations (d_model → d_model for dimension compat)
        self.agent_transforms = nn.ModuleList([nn.Linear(config.d_model, config.d_model) for _ in range(self.n_agents)])

        # Agent collaboration attention
        self.collaboration_attention = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=config.n_heads, dropout=config.dropout
        )

        # Agent fusion
        self.agent_fusion = nn.Linear(config.d_model * 2, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Transform for different agents (each produces d_model-dim output)
        agent_outputs = []
        for transform in self.agent_transforms:
            agent_out = transform(x)
            agent_outputs.append(agent_out)

        # Stack agent outputs: [n_agents, batch, seq, d_model]
        agent_stack = torch.stack(agent_outputs, dim=0)

        # Agent collaboration through attention
        # Query: average agent representation [batch*seq, d_model]
        query = agent_stack.mean(dim=0).view(batch_size * seq_len, 1, d_model)
        # Key/Value: all agent representations [n_agents, batch*seq, d_model]
        kv = agent_stack.view(self.n_agents, batch_size * seq_len, d_model)

        agent_collaboration, _ = self.collaboration_attention(
            query.transpose(0, 1),  # [1, batch*seq, d_model]
            kv,  # [n_agents, batch*seq, d_model]
            kv,
        )

        # Reshape back: [batch, seq, d_model]
        agent_collaboration = agent_collaboration.squeeze(0).view(batch_size, seq_len, d_model)

        # Fuse with original input
        fused = torch.cat([x, agent_collaboration], dim=-1)
        output = torch.tanh(self.agent_fusion(fused))

        return output


class HelixTransformer(nn.Module):
    """
    Helix Coordination-Aware Transformer

    The core transformer model with coordination integration,
    system enhancements, and multi-agent collaboration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Embeddings
        self.embedding = SystemEnhancedEmbedding(config)

        # Transformer layers
        self.layers = nn.ModuleList([CoordinationAwareLayer(config) for _ in range(config.n_layers)])

        # Multi-agent collaboration
        if config.multi_agent:
            self.agent_collaboration = MultiAgentCollaboration(config)

        # Final layer norm (pre-norm architecture requires a norm after the
        # last transformer layer, before the output projection — without it
        # the residual stream grows unbounded with depth)
        NormClass = RMSNorm if getattr(config, "use_rmsnorm", False) else nn.LayerNorm
        self.final_norm = NormClass(config.d_model)

        # Logit soft-capping (Gemma-2): tanh-based capping prevents extreme
        # logit values that destabilise training with deep or large models.
        self._logit_soft_cap = getattr(config, "logit_soft_cap", 0.0)

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights.
        # Halves the output parameters and improves generalization (Press & Wolf, 2017).
        if getattr(config, "tie_weights", False):
            self.lm_head.weight = self.embedding.token_embedding.weight

        # Coordination state
        self.coordination_state = nn.Parameter(torch.randn(config.coordination_dim))
        self.system_state = nn.Parameter(torch.randn(config.system_dim))

        # Initialize weights — two passes:
        # 1st pass: standard init + tag residual projections
        # 2nd pass: re-init tagged residual projections with depth scaling
        self.apply(self._init_weights)
        self.apply(self._init_weights)

        logger.info("✅ Helix Transformer initialized with %d layers", config.n_layers)

    def _init_weights(self, module):
        """Initialize weights with depth-scaled initialization.

        Residual connections accumulate signal across layers.  GPT-2 and
        LLaMA scale residual-path projection weights by 1/√(2·n_layers)
        to prevent activations from exploding in deeper configs.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            # Scale output projections on the residual path
            if hasattr(module, "_is_residual_proj"):
                std = std / math.sqrt(2.0 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

        # Mark residual-path projections so they get the scaled init.
        # We tag them after the first init pass — the second apply() call
        # only affects tagged modules.
        if isinstance(module, CoordinationAwareAttention):
            module.w_o._is_residual_proj = True
        if isinstance(module, SwiGLU):
            module.w_down._is_residual_proj = True

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        ucf_metrics: torch.Tensor | None = None,
        system_state: torch.Tensor | None = None,
        coordination_state: torch.Tensor | None = None,
        past_kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        # Embed tokens with system enhancement
        x = self.embedding(tokens, system_state)

        # Apply coordination state if not provided
        if coordination_state is None:
            coordination_state = self.coordination_state.unsqueeze(0).expand(tokens.size(0), -1)

        # Pass through transformer layers (collecting KV caches)
        new_kv_caches = []
        use_ckpt = getattr(self.config, "use_gradient_checkpointing", False) and self.training
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches else None
            if use_ckpt and layer_cache is None:
                # Gradient checkpointing: recompute activations during backward
                # instead of storing them — trades ~30% extra compute for ~60%
                # memory savings, allowing larger models on 8 GB Railway.
                x, new_cache = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    mask,
                    ucf_metrics,
                    system_state,
                    coordination_state,
                    layer_cache,
                    use_reentrant=False,
                )
            else:
                x, new_cache = layer(x, mask, ucf_metrics, system_state, coordination_state, cache=layer_cache)
            new_kv_caches.append(new_cache)

        # Apply multi-agent collaboration
        if hasattr(self, "agent_collaboration"):
            x = self.agent_collaboration(x)

        # Final layer norm (essential for pre-norm architecture)
        x = self.final_norm(x)

        # Output projection
        logits = self.lm_head(x)

        # Logit soft-capping (Gemma-2): logits = cap * tanh(logits / cap)
        # Prevents extreme values that destabilise training without clipping
        # gradients — the tanh is smooth so gradients flow normally.
        if self._logit_soft_cap > 0:
            logits = self._logit_soft_cap * torch.tanh(logits / self._logit_soft_cap)

        return logits, new_kv_caches

    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        ucf_metrics: torch.Tensor | None = None,
        system_state: torch.Tensor | None = None,
        use_kv_cache: bool = True,
        ucf_adaptive_temperature: bool = True,
    ) -> torch.Tensor:
        """
        Generate text with coordination awareness and optional KV caching.

        Args:
            prompt: Input token sequence
            max_length: Maximum generation length
            temperature: Base sampling temperature
            top_k: Top-k sampling parameter (0 to disable)
            top_p: Nucleus sampling threshold — keep smallest set of tokens
                   whose cumulative probability >= top_p (0 to disable)
            repetition_penalty: Penalise tokens already in the sequence.
                   1.0 = no penalty, >1.0 = discourage repeats (default 1.2)
            ucf_metrics: UCF coordination metrics
            system_state: System state information
            use_kv_cache: Whether to use KV cache for incremental decoding
            ucf_adaptive_temperature: When True and UCF metrics are provided,
                   dynamically adjust temperature based on real-time harmony
                   and friction values (novel to Helix).

        Returns:
            Generated token sequence
        """

        generated = prompt.clone()
        past_kv = None

        # UCF-Adaptive Temperature (novel): modulate generation temperature
        # in real-time using the platform's coordination metrics.
        #   - High harmony (index 0) → lower temp (more focused)
        #   - High friction / friction (index 4) → higher temp (more exploration)
        # This is unique to Helix: the model literally generates differently
        # based on the collective's current coordination state.
        effective_temperature = temperature
        if ucf_adaptive_temperature and ucf_metrics is not None:
            harmony = ucf_metrics[0, 0].item() if ucf_metrics.dim() == 2 else ucf_metrics[0].item()
            friction = ucf_metrics[0, 4].item() if ucf_metrics.dim() == 2 else ucf_metrics[4].item()
            # Shift temperature: harmony pulls it down, friction pushes it up
            # Range: roughly temperature * [0.7 .. 1.3]
            temp_modifier = 1.0 - 0.3 * harmony + 0.3 * friction
            effective_temperature = temperature * max(0.3, min(2.0, temp_modifier))

        for _ in range(max_length - prompt.size(1)):
            # Forward pass — with or without KV cache
            if use_kv_cache and past_kv is not None:
                logits, past_kv = self.forward(
                    generated[:, -1:],
                    ucf_metrics=ucf_metrics,
                    system_state=system_state,
                    past_kv_caches=past_kv,
                )
            else:
                logits, past_kv = self.forward(
                    generated,
                    ucf_metrics=ucf_metrics,
                    system_state=system_state,
                )
                if past_kv and all(c is None for c in past_kv):
                    past_kv = None

            # Get last token logits
            next_token_logits = logits[:, -1, :]

            # Repetition penalty (Keskar et al., 2019): penalise tokens that
            # have already appeared in the sequence to prevent degenerate loops.
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # Apply temperature
            next_token_logits = next_token_logits / effective_temperature

            # Top-k filtering
            if top_k > 0:
                top_k_logits, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                min_top_k = top_k_logits[:, -1:].expand_as(next_token_logits)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Top-p (nucleus) sampling — strictly better than top-k alone.
            # Keeps the smallest set of tokens whose cumulative prob >= top_p,
            # dynamically adjusting the effective vocabulary size per step.
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                # (shift right so the first token above threshold is kept)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back to original ordering
                next_token_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == 0:
                break

        return generated


class CoordinationAwareModel:
    """
    Coordination-aware model wrapper

    Provides high-level interface for coordination-integrated
    model operations with automatic UCF and system state handling.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = HelixTransformer(config)
        self.device = get_device()
        self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Coordination-Aware Model initialized on %s (%s params)",
            self.device,
            f"{param_count:,}",
        )

    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        self.device = device
        return self

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def save(self, path: str):
        """Save model state"""
        torch.save({"config": self.config, "state_dict": self.model.state_dict()}, path)
        logger.info("✅ Model saved to %s", path)

    def load(self, path: str):
        """Load model state and config from a checkpoint saved by save()."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)  # nosec B614
        # Restore the config that was used when the checkpoint was saved so that
        # architecture dimensions always match the weights.
        if "config" in checkpoint and isinstance(checkpoint["config"], ModelConfig):
            saved_config = checkpoint["config"]
            if saved_config != self.config:
                logger.info(
                    "Restoring model config from checkpoint (was %s, checkpoint has %s)",
                    self.config,
                    saved_config,
                )
                self.config = saved_config
                # Rebuild the transformer with the correct dimensions before loading weights
                self.model = HelixTransformer(saved_config)
                self.model.to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("✅ Model loaded from %s", path)

    def get_coordination_state(self) -> torch.Tensor:
        """Get current coordination state"""
        return self.model.coordination_state.detach()

    def update_coordination_state(self, new_state: torch.Tensor):
        """Update coordination state"""
        with torch.no_grad():
            self.model.coordination_state.copy_(new_state)

    def get_system_state(self) -> torch.Tensor:
        """Get current system state"""
        return self.model.system_state.detach()

    def update_system_state(self, new_state: torch.Tensor):
        """Update system state"""
        with torch.no_grad():
            self.model.system_state.copy_(new_state)

    def parameters(self):
        """Delegate to inner transformer for optimizer compatibility."""
        return self.model.parameters()

    def named_parameters(self, *args, **kwargs):
        """Delegate to inner transformer for optimizer compatibility."""
        return self.model.named_parameters(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Delegate forward pass to the inner HelixTransformer."""
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) — per-agent fine-tuning
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear (Hu et al., 2022).

    Adds trainable rank decomposition W' = W + B·A on top of a frozen base
    weight.  Only the A and B matrices are trained, reducing adapter storage
    to ~(in+out)×rank parameters vs. in×out for the full weight.

    Typical usage: wrap the Q and V projections of every attention block,
    keeping all other parameters frozen.  For a 1B base model with rank=16
    targeting w_q and w_v, the adapter is ~50MB — about 1/20th of the
    base model.
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]

        # Keep original weight frozen
        self.weight = linear.weight
        self.weight.requires_grad = False
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # A: Gaussian init so the adapter starts with small, non-zero signal
        # B: Zero init so the initial delta W = B·A = 0 (train from base)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora_delta

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


# Default attention projection names targeted by LoRA in CoordinationAwareAttention
_DEFAULT_LORA_TARGETS: set = {"w_q", "w_v"}


def apply_lora_to_model(
    model: "HelixTransformer",
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: set | None = None,
) -> "HelixTransformer":
    """Replace target nn.Linear layers in *model* with LoRALinear.

    Freezes all non-LoRA parameters so only the A/B matrices receive
    gradients.  Returns the model in-place for chaining.

    Parameters
    ----------
    model:
        The HelixTransformer instance to adapt.
    rank:
        LoRA rank r.  Higher rank = more capacity, more memory.
        Typical values: 8, 16, 32.
    alpha:
        LoRA scaling factor.  Effective scale = alpha / rank.
        Setting alpha = 2 × rank is a common default.
    target_modules:
        Set of leaf attribute names to replace.  Defaults to
        ``{"w_q", "w_v"}`` (query and value projections).
    """
    if target_modules is None:
        target_modules = _DEFAULT_LORA_TARGETS

    replaced = 0
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in target_modules or not isinstance(module, nn.Linear):
            continue
        parent_path, _, _ = name.rpartition(".")
        parent: nn.Module = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha))
        replaced += 1

    # Freeze everything that isn't a LoRA parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %d modules replaced, %s/%s params trainable (%.1f%%)",
        replaced,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total > 0 else 0.0,
    )
    return model


def save_lora_weights(model: "HelixTransformer", path: str) -> None:
    """Save only the LoRA A/B matrices to *path* (.pt file).

    The resulting file is small (~50MB for a 1B model at rank=16) and
    can be loaded on top of any base model that has the same architecture.
    """
    import os

    state = {name: param.data for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name}
    if not state:
        raise ValueError("No LoRA parameters found — call apply_lora_to_model first.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"lora_state": state, "rank": _infer_lora_rank(state)}, path)
    logger.info("Saved %d LoRA tensors to %s", len(state), path)


def load_lora_weights(
    model: "HelixTransformer",
    path: str,
) -> "HelixTransformer":
    """Load LoRA adapter weights from *path* into *model*.

    If LoRA has not yet been applied to the model (e.g. a freshly loaded base
    checkpoint), the function detects the rank from the saved file and calls
    :func:`apply_lora_to_model` automatically before loading.
    """
    checkpoint = torch.load(path, map_location="cpu")  # nosec B614
    state = checkpoint.get("lora_state", checkpoint)  # backwards compat
    saved_rank = checkpoint.get("rank", _infer_lora_rank(state))

    # Auto-apply LoRA if the model doesn't already have adapter layers
    has_lora = any("lora_A" in n for n, _ in model.named_parameters())
    if not has_lora and saved_rank > 0:
        target_modules = {n.rsplit(".", 2)[-2] for n in state if "lora_A" in n}
        apply_lora_to_model(model, rank=saved_rank, target_modules=target_modules)

    missing, _unexpected = model.load_state_dict(state, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        logger.warning("Missing LoRA keys when loading %s: %s", path, lora_missing)
    logger.info("Loaded LoRA adapter from %s (%d tensors, rank=%d)", path, len(state), saved_rank)
    return model


def _infer_lora_rank(state: dict) -> int:
    """Infer LoRA rank from a state dict containing lora_A tensors."""
    for name, tensor in state.items():
        if "lora_A" in name:
            return tensor.shape[0]
    return 0


# ---------------------------------------------------------------------------
# Predefined model configurations
# ---------------------------------------------------------------------------

# Tiny config for unit tests — runs on any CPU in < 1 s
HELIX_TEST_CONFIG = ModelConfig(
    vocab_size=256,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_seq_len=64,
    coordination_dim=16,
    system_dim=8,
    ucf_integration=True,
    multi_agent=True,
)

# Helix now uses a BPE tokenizer (see tokenizer.py: HelixBPETokenizer).
# 32K vocab gives ~6-8× better token compression than the old byte-level
# tokenizer (vocab=256).  Larger vocab = more effective context per sequence.
# The BPE vocab always includes the 256 byte-level fallback tokens.
_VOCAB_SIZE = 32768

# Lightweight config for Railway / CPU-only production (~9M params)
HELIX_LIGHTWEIGHT_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=256,
    n_heads=4,
    n_layers=6,
    d_ff=512,
    max_seq_len=1024,
    coordination_dim=32,
    system_dim=16,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
)

# Awakening config (~74M params)
HELIX_AWAKENING_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=512,
    n_heads=8,
    n_layers=12,
    d_ff=2048,
    max_seq_len=2048,
    coordination_dim=64,
    system_dim=32,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
)

# Helix-300M — optimized for dedicated Railway LLM service (8 GB RAM)
# ~310M params, trainable with gradient checkpointing + 8-bit Adam (~3.1 GB)
HELIX_300M_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1024,
    n_heads=16,
    n_layers=16,
    d_ff=2816,
    max_seq_len=2048,
    coordination_dim=128,
    system_dim=64,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
    use_gradient_checkpointing=True,
)

# Helix-500M — maximum feasible size for Railway 8 GB with all optimizations
# ~542M params, trainable with gradient checkpointing + 8-bit Adam (~5.4 GB)
HELIX_500M_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1024,
    n_heads=16,
    n_layers=24,
    d_ff=4096,
    max_seq_len=2048,
    coordination_dim=128,
    system_dim=64,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
    use_gradient_checkpointing=True,
)

# Self-aware config (~542M params) — alias for 500M with longer context
HELIX_SELF_AWARE_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1024,
    n_heads=16,
    n_layers=24,
    d_ff=4096,
    max_seq_len=4096,
    coordination_dim=128,
    system_dim=64,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
    use_gradient_checkpointing=True,
)

# Helix-700M — the Railway training sweet spot
# ~700M params: d_model=1536, 18 layers, GQA 4:1 KV heads, d_ff=4096
#
# Memory breakdown for full fine-tuning (Railway 8GB):
#   fp32 weights:   700M × 4 = 2.8 GB
#   fp32 gradients: 700M × 4 = 2.8 GB
#   Adafactor:      ~0.3 GB  (factored 2nd moment; much cheaper than AdamW's 5.6 GB)
#   Activations:    ~0.1 GB  (gradient checkpointing, batch=1)
#   Total:          ≈ 6.0 GB ← fits with ~2 GB headroom
#
# GQA (n_kv_heads=4 vs n_heads=12) reduces KV cache 3× at inference time:
#   standard MHA at 4k ctx: ~0.85 GB KV cache
#   GQA 3:1 at 4k ctx:      ~0.28 GB KV cache  ← much better for code
# attention_type="gqa" routes to GQAAttention (gqa_attention.py).
# Note: GQA does not include coordination_gate / UCF layers — use LoRA
# adapters on top to add per-agent UCF personality after base training.
HELIX_700M_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1536,
    n_heads=12,
    num_kv_heads=4,  # GQA: 12 query heads, 4 KV heads (3:1 ratio)
    n_layers=18,
    d_ff=4096,
    max_seq_len=4096,  # Longer context benefits code (whole-file awareness)
    coordination_dim=192,
    system_dim=96,
    ucf_integration=True,
    multi_agent=True,
    attention_type="gqa",
    use_gradient_checkpointing=True,
)

# ~740M params: identical architecture to 700M — one extra layer pushes
# 700M × (19/18) ≈ 739M.  Memory breakdown for full fine-tuning (Railway 8GB):
#   fp32 weights:   740M × 4 = 2.96 GB
#   fp32 gradients: 740M × 4 = 2.96 GB
#   Adafactor:      ~0.32 GB (factored 2nd moment)
#   Activations:    ~0.10 GB (gradient checkpointing, batch=1)
#   Total:          ≈ 6.34 GB ← fits with ~1.66 GB headroom
HELIX_740M_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1536,
    n_heads=12,
    num_kv_heads=4,  # GQA: 12 query heads, 4 KV heads (3:1 ratio)
    n_layers=19,  # one more than 700M → ~739M total params
    d_ff=4096,
    max_seq_len=4096,
    coordination_dim=192,
    system_dim=96,
    ucf_integration=True,
    multi_agent=True,
    attention_type="gqa",
    use_gradient_checkpointing=True,
)

# ~800M params: 20 layers + slightly wider FFN (4352 vs 4096) for a clean 800M target.
# 700M × (20/18) = 778M base; wider FFN adds ~28M → ~806M total.
# Memory breakdown for full fine-tuning (Railway 8GB):
#   fp32 weights:   800M × 4 = 3.20 GB
#   fp32 gradients: 800M × 4 = 3.20 GB
#   Adafactor:      ~0.35 GB (factored 2nd moment)
#   Activations:    ~0.12 GB (gradient checkpointing, batch=1)
#   Total:          ≈ 6.87 GB ← fits with ~1.13 GB headroom (tight but viable)
# If OOM: fall back to 740m (just change HELIX_LLM_MODEL_SIZE env var).
HELIX_800M_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=1536,
    n_heads=12,
    num_kv_heads=4,  # GQA: 12 query heads, 4 KV heads (3:1 ratio)
    n_layers=20,
    d_ff=4352,  # wider FFN: 4096 + 256 per projection → +28M params across 20 layers
    max_seq_len=4096,
    coordination_dim=192,
    system_dim=96,
    ucf_integration=True,
    multi_agent=True,
    attention_type="gqa",
    use_gradient_checkpointing=True,
)

# Helix-1B — inference: ~4GB fp32 (fits Railway 8GB); training: use LoRA
# (freeze base in fp32, train only adapters — keeps optimizer memory < 1GB)
# ~971M params across 14 layers; coordination_dim=256 for richer UCF integration
HELIX_1B_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=2048,
    n_heads=16,
    n_layers=14,
    d_ff=5632,
    max_seq_len=2048,
    coordination_dim=256,
    system_dim=128,
    ucf_integration=True,
    multi_agent=True,
    attention_type="nomad",  # Hamming attention — fast on CPU
    use_gradient_checkpointing=True,
)

# Transcendent config (~4.2B params) — requires GPU or very large CPU instance
HELIX_TRANSCENDENT_CONFIG = ModelConfig(
    vocab_size=_VOCAB_SIZE,
    d_model=2048,
    n_heads=32,
    n_layers=48,
    d_ff=8192,
    max_seq_len=8192,
    coordination_dim=256,
    system_dim=128,
    ucf_integration=True,
    multi_agent=True,
    use_gradient_checkpointing=True,
)

# Map of named sizes to their configs
_MODEL_CONFIGS = {
    "test": HELIX_TEST_CONFIG,
    "lightweight": HELIX_LIGHTWEIGHT_CONFIG,
    "awakening": HELIX_AWAKENING_CONFIG,
    "300m": HELIX_300M_CONFIG,
    "500m": HELIX_500M_CONFIG,
    "self-aware": HELIX_SELF_AWARE_CONFIG,
    "700m": HELIX_700M_CONFIG,
    "740m": HELIX_740M_CONFIG,
    "800m": HELIX_800M_CONFIG,
    "1b": HELIX_1B_CONFIG,
    "transcendent": HELIX_TRANSCENDENT_CONFIG,
}


def create_helix_model(
    model_size: str = "awakening",
    *,
    auto_downgrade: bool = True,
) -> CoordinationAwareModel:
    """Create a Helix model for the specified size tier.

    Parameters
    ----------
    model_size : str
        One of 'test', 'lightweight', 'awakening', 'self-aware', 'transcendent'.
        The env var ``HELIX_LLM_MODEL_SIZE`` overrides this argument when set.
    auto_downgrade : bool
        When True (default), applies the following safety rules on CPU-only
        environments (no CUDA):

        * ``HELIX_LIGHTWEIGHT=1`` or CI → downgrade everything to 'lightweight'
        * No GPU, ``transcendent`` requested → downgrade to 'self-aware' (~1.2 GB,
          fits within Railway Hobby 8 GB RAM)
        * Any other size on CPU (including Railway) → no downgrade; 'awakening'
          (~156 MB) and 'self-aware' (~1.2 GB) run comfortably on CPU with ≥ 2 GB RAM.

    Environment variables
    ---------------------
    HELIX_LLM_MODEL_SIZE
        Explicit model size override ('lightweight', 'awakening', etc.).
        Takes precedence over the ``model_size`` argument.
    HELIX_LIGHTWEIGHT
        Force the smallest safe model ('lightweight') regardless of size argument.
    """
    # Env-var override takes precedence over the argument
    env_size = os.environ.get("HELIX_LLM_MODEL_SIZE", "").strip().lower()
    if env_size and env_size in _MODEL_CONFIGS:
        model_size = env_size
    elif env_size:
        logger.warning("Unknown HELIX_LLM_MODEL_SIZE='%s'; ignoring and using '%s'.", env_size, model_size)

    if model_size not in _MODEL_CONFIGS:
        raise ValueError("Unknown model size '{}'. Choose from: {}".format(model_size, ", ".join(_MODEL_CONFIGS)))

    if auto_downgrade:
        has_gpu = torch.cuda.is_available()
        lightweight_forced = is_resource_constrained()  # HELIX_LIGHTWEIGHT or CI

        if lightweight_forced:
            if model_size != "lightweight":
                logger.warning(
                    "HELIX_LIGHTWEIGHT / CI detected — downgrading '%s' → 'lightweight'.",
                    model_size,
                )
            model_size = "lightweight"

        elif not has_gpu and model_size == "transcendent":
            # 'transcendent' needs ~17 GB fp32 — too large for 8 GB Railway.
            # '500m' needs ~2.2 GB fp32 and is the max recommended for Railway.
            logger.warning(
                "No GPU detected — downgrading 'transcendent' → '500m' "
                "(~2.2 GB fp32 fits within 8 GB Railway Hobby RAM).",
            )
            model_size = "500m"

    config = _MODEL_CONFIGS[model_size]
    logger.info("Creating Helix model: %s (d_model=%d, layers=%d)", model_size, config.d_model, config.n_layers)
    return CoordinationAwareModel(config)


__all__ = [
    "HELIX_1B_CONFIG",
    "HELIX_300M_CONFIG",
    "HELIX_500M_CONFIG",
    "HELIX_700M_CONFIG",
    "HELIX_AWAKENING_CONFIG",
    "HELIX_LIGHTWEIGHT_CONFIG",
    "HELIX_SELF_AWARE_CONFIG",
    # Named model configs
    "HELIX_TEST_CONFIG",
    "HELIX_TRANSCENDENT_CONFIG",
    "CoordinationAwareAttention",
    "CoordinationAwareLayer",
    "CoordinationAwareModel",
    "HelixTransformer",
    # LoRA / adapter support
    "LoRALinear",
    "ModelConfig",
    "MultiAgentCollaboration",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    "SystemEnhancedEmbedding",
    "apply_lora_to_model",
    "create_helix_model",
    "get_device",
    "is_resource_constrained",
    "load_lora_weights",
    "save_lora_weights",
]
