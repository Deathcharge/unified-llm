"""
Sliding Window Attention Implementation for CPU-Optimized LLM

Sliding Window Attention limits the attention span to a fixed window around each token,
enabling linear complexity instead of quadratic. This is crucial for CPU inference
with long sequences.
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


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention Implementation

    Each token only attends to a local window of surrounding tokens instead of the
    entire sequence. This reduces computation from O(n²) to O(n*w) where w is window size.

    Reference: "Longformer: The Long-Document Transformer"
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Mask tensor of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1.0

        return mask

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

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            seq_len = k.size(2)

        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, x.device)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply sliding window mask
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # Cache keys and values
        new_cache = (k, v)

        return output, new_cache


class GlobalTokenAttention(nn.Module):
    """
    Global Token Attention (for special tokens like [CLS], [SEP])

    Allows certain global tokens to attend to all positions in the sequence.
    This can be combined with sliding window for hybrid attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        global_indices: list | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.global_indices = global_indices or [0]  # Default: first token is global
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass with global token handling.
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            seq_len = k.size(2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create mask for sliding window + global tokens
        mask_tensor = torch.zeros(seq_len, seq_len, device=x.device)

        for i in range(seq_len):
            if i in self.global_indices:
                # Global token attends to all positions
                mask_tensor[i, :] = 1.0
                # All positions attend to global token
                mask_tensor[:, i] = 1.0
            else:
                # Local sliding window
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask_tensor[i, start:end] = 1.0

        # Apply mask
        scores = scores.masked_fill(mask_tensor.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # Cache keys and values
        new_cache = (k, v) if cache is not None or self.training else None

        return output, new_cache


class SlidingWindowTransformerLayer(nn.Module):
    """
    Transformer layer using Sliding Window Attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_global_tokens: bool = False,
    ):
        super().__init__()

        if use_global_tokens:
            self.attention: SlidingWindowAttention | GlobalTokenAttention = GlobalTokenAttention(
                embed_dim, num_heads, window_size, dropout=dropout
            )
        else:
            self.attention = SlidingWindowAttention(embed_dim, num_heads, window_size, dropout)

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


def calculate_attention_complexity(seq_len: int, window_size: int, num_heads: int, head_dim: int) -> dict:
    """
    Calculate computational complexity comparison.

    Args:
        seq_len: Sequence length
        window_size: Sliding window size
        num_heads: Number of attention heads
        head_dim: Head dimension

    Returns:
        Dictionary with complexity calculations
    """
    # Standard attention: O(n²)
    standard_flops = 2 * num_heads * seq_len * seq_len * head_dim

    # Sliding window: O(n*w)
    sliding_flops = 2 * num_heads * seq_len * window_size * head_dim

    # Speedup
    speedup = standard_flops / sliding_flops if sliding_flops > 0 else 0

    return {
        "seq_len": seq_len,
        "window_size": window_size,
        "standard_flops": standard_flops,
        "sliding_flops": sliding_flops,
        "speedup": speedup,
        "complexity_reduction_percent": (1 - 1 / speedup) * 100 if speedup > 0 else 0,
    }


# Export classes and functions
__all__ = [
    "GlobalTokenAttention",
    "SlidingWindowAttention",
    "SlidingWindowTransformerLayer",
    "calculate_attention_complexity",
]
