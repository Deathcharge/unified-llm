"""
Grouped-Query Attention (GQA) Implementation for CPU-Optimized LLM

GQA reduces the memory footprint of the KV cache by sharing key and value projections
across multiple query heads. This is especially beneficial for CPU inference where
memory bandwidth is often the bottleneck.
"""

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) Implementation

    Instead of having separate K and V for each query head, GQA groups multiple
    query heads to share the same K and V. This reduces KV cache memory by a
    factor of num_heads // num_kv_heads.

    Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    """

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_query_groups = num_heads // num_kv_heads
        self.dropout = dropout

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert num_kv_heads <= num_heads, "num_kv_heads must be <= num_heads"

        # Q projection (full num_heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # K and V projections (only num_kv_heads)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat keys/values to match number of query heads.

        Args:
            x: Tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass with optional KV caching.
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q (full num_heads)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Reshape K and V (num_kv_heads)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            seq_len = k.size(2)

        # Repeat K and V to match number of query heads
        k_repeated = self._repeat_kv(k, self.num_query_groups)
        v_repeated = self._repeat_kv(v, self.num_query_groups)

        # Compute attention scores
        scores = torch.matmul(q, k_repeated.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v_repeated)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # Cache the original K and V (before repetition)
        new_cache = (k, v)

        return output, new_cache


class MQAttention(GroupedQueryAttention):
    """
    Multi-Query Attention (MQA) - a special case of GQA with num_kv_heads = 1

    This is the most memory-efficient variant where all query heads share a single
    K and V projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, num_kv_heads=1, dropout=dropout)


class GQATransformerLayer(nn.Module):
    """
    Transformer layer using Grouped-Query Attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_output, new_cache = self.attention(self.norm1(x), mask, cache)
        x = x + attn_output

        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x, new_cache


def calculate_kv_cache_memory_reduction(
    num_heads: int, num_kv_heads: int, seq_len: int, head_dim: int, dtype_size: int = 2
) -> dict:
    """
    Calculate memory savings from using GQA.

    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        seq_len: Sequence length
        head_dim: Head dimension
        dtype_size: Size of data type in bytes (2 for FP16/BF16, 4 for FP32)

    Returns:
        Dictionary with memory calculations
    """
    # Standard MHA memory
    mha_memory = 2 * num_heads * seq_len * head_dim * dtype_size  # K and V

    # GQA memory
    gqa_memory = 2 * num_kv_heads * seq_len * head_dim * dtype_size

    # Savings
    savings = mha_memory - gqa_memory
    reduction_percent = (savings / mha_memory) * 100

    return {
        "mha_memory_bytes": mha_memory,
        "gqa_memory_bytes": gqa_memory,
        "savings_bytes": savings,
        "reduction_percent": reduction_percent,
        "mha_memory_mb": mha_memory / (1024 * 1024),
        "gqa_memory_mb": gqa_memory / (1024 * 1024),
        "savings_mb": savings / (1024 * 1024),
    }


# Export classes and functions
__all__ = ["GQATransformerLayer", "GroupedQueryAttention", "MQAttention", "calculate_kv_cache_memory_reduction"]
