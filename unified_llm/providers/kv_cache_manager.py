"""
KV Cache Manager for CPU-Optimized LLM

Implements various KV cache management strategies to optimize memory usage and
inference speed for CPU-based LLMs.
"""

import heapq
from collections import deque
from enum import Enum
from typing import Any

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class EvictionStrategy(Enum):
    """KV cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    FIFO = "fifo"  # First In First Out
    LFU = "lfu"  # Least Frequently Used
    HYBRID = "hybrid"  # Combination of strategies


class KVCacheEntry:
    """
    Represents a single entry in the KV cache.
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor, position: int, access_count: int = 1):
        self.key = key
        self.value = value
        self.position = position
        self.access_count = access_count
        self.last_access = torch.time_ns()

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = torch.time_ns()

    def memory_usage(self) -> int:
        """Calculate memory usage in bytes."""
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


class KVCacheManager:
    """
    Manages KV cache with configurable eviction strategies.
    """

    def __init__(
        self,
        max_cache_size_mb: float = 512,
        eviction_strategy: EvictionStrategy = EvictionStrategy.LRU,
        max_sequence_length: int = 4096,
    ):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.eviction_strategy = eviction_strategy
        self.max_sequence_length = max_sequence_length

        # Cache storage: {layer_idx: List[KVCacheEntry]}
        self.cache: dict[int, list[KVCacheEntry]] = {}

        # Access tracking
        self.total_accesses = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # For FIFO strategy
        self.fifo_queue: deque[tuple[int, KVCacheEntry]] = deque()

        # For LFU strategy
        self.lfu_heap: list[tuple[int, int, KVCacheEntry]] = []

        # Current cache size
        self.current_cache_size = 0

    def _evict_needed(self) -> bool:
        """Check if eviction is needed."""
        return self.current_cache_size > self.max_cache_size_bytes * 0.9

    def _evict_lru(self, layer_idx: int) -> bool:
        """Evict least recently used entry."""
        if layer_idx not in self.cache or not self.cache[layer_idx]:
            return False

        # Find LRU entry
        entries = self.cache[layer_idx]
        lru_entry = min(entries, key=lambda e: e.last_access)

        # Remove entry
        self.current_cache_size -= lru_entry.memory_usage()
        entries.remove(lru_entry)

        return True

    def _evict_fifo(self, layer_idx: int) -> bool:
        """Evict oldest entry (FIFO)."""
        if not self.fifo_queue:
            return False

        # Get oldest entry
        layer, entry = self.fifo_queue.popleft()

        if layer in self.cache and entry in self.cache[layer]:
            self.current_cache_size -= entry.memory_usage()
            self.cache[layer].remove(entry)

        return True

    def _evict_lfu(self, layer_idx: int) -> bool:
        """Evict least frequently used entry."""
        if not self.lfu_heap:
            return False

        # Get LFU entry
        _, layer, entry = heapq.heappop(self.lfu_heap)

        if layer in self.cache and entry in self.cache[layer]:
            self.current_cache_size -= entry.memory_usage()
            self.cache[layer].remove(entry)

        return True

    def _evict_hybrid(self, layer_idx: int) -> bool:
        """Hybrid eviction strategy."""
        # Use LFU for critical layers, LRU for others
        if layer_idx % 4 == 0:  # Critical layers
            return self._evict_lfu(layer_idx)
        else:
            return self._evict_lru(layer_idx)

    def _evict(self, layer_idx: int):
        """Perform eviction based on strategy."""
        evicted = False

        if self.eviction_strategy == EvictionStrategy.LRU:
            evicted = self._evict_lru(layer_idx)
        elif self.eviction_strategy == EvictionStrategy.FIFO:
            evicted = self._evict_fifo(layer_idx)
        elif self.eviction_strategy == EvictionStrategy.LFU:
            evicted = self._evict_lfu(layer_idx)
        elif self.eviction_strategy == EvictionStrategy.HYBRID:
            evicted = self._evict_hybrid(layer_idx)

        return evicted

    def get(self, layer_idx: int, position: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Retrieve KV cache entry.

        Returns:
            Tuple of (key, value) or None if not in cache
        """
        self.total_accesses += 1

        if layer_idx not in self.cache:
            self.cache_misses += 1
            return None

        # Find entry at position
        for entry in self.cache[layer_idx]:
            if entry.position == position:
                entry.update_access()
                self.cache_hits += 1
                return entry.key, entry.value

        self.cache_misses += 1
        return None

    def put(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, position: int):
        """
        Store KV cache entry.
        """
        # Create new entry
        entry = KVCacheEntry(key, value, position)

        # Initialize layer cache if needed
        if layer_idx not in self.cache:
            self.cache[layer_idx] = []

        # Check for duplicate and update
        for i, existing in enumerate(self.cache[layer_idx]):
            if existing.position == position:
                self.current_cache_size -= existing.memory_usage()
                self.cache[layer_idx][i] = entry
                break
        else:
            self.cache[layer_idx].append(entry)

            # Add to tracking structures
            self.fifo_queue.append((layer_idx, entry))
            heapq.heappush(self.lfu_heap, (entry.access_count, layer_idx, entry))

        # Update cache size
        self.current_cache_size += entry.memory_usage()

        # Evict if needed
        while self._evict_needed():
            self._evict(layer_idx)

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.fifo_queue.clear()
        self.lfu_heap = []
        self.current_cache_size = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.cache_hits / self.total_accesses * 100) if self.total_accesses > 0 else 0

        return {
            "total_accesses": self.total_accesses,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "current_cache_size_mb": self.current_cache_size / (1024 * 1024),
            "max_cache_size_mb": self.max_cache_size_bytes / (1024 * 1024),
            "cache_utilization_percent": (self.current_cache_size / self.max_cache_size_bytes * 100),
            "eviction_strategy": self.eviction_strategy.value,
        }


class RollingWindowKVCache:
    """
    KV cache with a rolling window policy.

    Only keeps the most recent N tokens in the cache, ensuring bounded memory usage.
    """

    def __init__(self, window_size: int = 2048, num_layers: int = 12):
        self.window_size = window_size
        self.num_layers = num_layers

        # Cache: list of (key, value) tuples per layer
        self.cache: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(num_layers)]

    def get(self, layer_idx: int, position: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Get cache entry at position.
        """
        cache_layer = self.cache[layer_idx]

        # Check if position is within rolling window
        if not cache_layer:
            return None

        # Position relative to window start
        if position >= len(cache_layer):
            return None

        return cache_layer[position]

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        Append new KV entry to cache.
        """
        cache_layer = self.cache[layer_idx]

        # Add new entry
        cache_layer.append((key, value))

        # Enforce window size
        if len(cache_layer) > self.window_size:
            cache_layer.pop(0)

    def get_cached_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached keys and values for a layer.
        """
        cache_layer = self.cache[layer_idx]

        if not cache_layer:
            return None, None

        # Concatenate all cached entries
        keys = torch.cat([k for k, v in cache_layer], dim=1)
        values = torch.cat([v for k, v in cache_layer], dim=1)

        return keys, values

    def clear(self):
        """Clear all cache entries."""
        self.cache = [[] for _ in range(self.num_layers)]


class CompressedKVCache:
    """
    KV cache with compression to reduce memory usage.

    Uses quantization and pruning to compress cached keys and values.
    """

    def __init__(self, compression_ratio: float = 0.5, quantization_bits: int = 8):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits

        self.cache: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor using quantization."""
        # Calculate scale
        scale = tensor.abs().max() / (2 ** (self.quantization_bits - 1) - 1)

        # Quantize
        qmin = -(2 ** (self.quantization_bits - 1))
        qmax = 2 ** (self.quantization_bits - 1) - 1

        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)

        return quantized

    def _decompress_tensor(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Decompress tensor."""
        return quantized.to(torch.float32) * scale

    def put(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        Store compressed KV entry.
        """
        if layer_idx not in self.cache:
            self.cache[layer_idx] = []

        # Compress tensors
        compressed_key = self._compress_tensor(key)
        compressed_value = self._compress_tensor(value)

        self.cache[layer_idx].append((compressed_key, compressed_value))

    def get(self, layer_idx: int, position: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Retrieve and decompress KV entry.
        """
        if layer_idx not in self.cache or position >= len(self.cache[layer_idx]):
            return None

        compressed_key, compressed_value = self.cache[layer_idx][position]

        # Decompress (approximate)
        key = compressed_key.to(torch.float32)
        value = compressed_value.to(torch.float32)

        return key, value


def calculate_kv_cache_memory(
    num_layers: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_size: int = 2,
    use_gqa: bool = False,
    num_kv_heads: int | None = None,
) -> dict[str, Any]:
    """
    Calculate KV cache memory requirements.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        dtype_size: Size of data type in bytes
        use_gqa: Whether using Grouped-Query Attention
        num_kv_heads: Number of KV heads (for GQA)

    Returns:
        Dictionary with memory calculations
    """
    if use_gqa and num_kv_heads:
        kv_heads = num_kv_heads
    else:
        kv_heads = num_heads

    # KV cache per layer: 2 * num_heads * seq_len * head_dim * dtype_size
    kv_per_layer = 2 * kv_heads * seq_len * head_dim * dtype_size

    # Total KV cache
    total_kv = kv_per_layer * num_layers

    # With GQA
    if use_gqa and num_kv_heads:
        reduction_factor = num_heads / num_kv_heads
        gqa_total = total_kv / reduction_factor
    else:
        gqa_total = total_kv

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "kv_per_layer_mb": kv_per_layer / (1024 * 1024),
        "total_kv_mb": total_kv / (1024 * 1024),
        "gqa_total_mb": gqa_total / (1024 * 1024),
        "gqa_savings_mb": total_kv / (1024 * 1024) - gqa_total / (1024 * 1024) if use_gqa else 0,
    }


# Export classes and functions
__all__ = [
    "CompressedKVCache",
    "EvictionStrategy",
    "KVCacheEntry",
    "KVCacheManager",
    "RollingWindowKVCache",
    "calculate_kv_cache_memory",
]
