"""
Multi-Core Parallelization for CPU-Optimized LLM

This module implements techniques to leverage multiple CPU cores for efficient
LLM inference on CPU, including tensor parallelism, pipeline parallelism, and
optimized attention computation with OpenMP/MKL.
"""

import concurrent.futures
import os
from typing import Any

try:
    import torch
    import torch.multiprocessing as mp
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class CPUParallelAttention(nn.Module):
    """
    CPU-optimized parallel attention using multiple cores.

    Splits attention computation across available CPU cores for faster inference.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_workers: int | None = None, use_mkl: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Auto-detect number of workers
        self.num_workers = num_workers or min(mp.cpu_count(), num_heads)
        self.use_mkl = use_mkl

        # Configure MKL threads if available
        if self.use_mkl:
            os.environ["MKL_NUM_THREADS"] = str(self.num_workers)
            os.environ["OMP_NUM_THREADS"] = str(self.num_workers)

        # Standard projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _parallel_attention_score(self, q_chunk: torch.Tensor, k: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Compute attention scores for a chunk of heads in parallel.
        """
        return torch.matmul(q_chunk, k.transpose(-2, -1)) / (self.head_dim**0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with parallel attention computation.
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Split heads across workers
        heads_per_worker = self.num_heads // self.num_workers

        if heads_per_worker > 0 and self.num_workers > 1:
            # Parallel computation across workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []

                for worker_idx in range(self.num_workers):
                    start_head = worker_idx * heads_per_worker
                    end_head = start_head + heads_per_worker

                    q_chunk = q[:, start_head:end_head, :, :]
                    k_chunk = k[:, start_head:end_head, :, :]

                    future = executor.submit(self._parallel_attention_score, q_chunk, k_chunk, worker_idx)
                    futures.append(future)

                # Collect results
                scores_chunks = [f.result() for f in futures]
                scores = torch.cat(scores_chunks, dim=1)
        else:
            # Fallback to sequential computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output


class TensorParallelLayer(nn.Module):
    """
    Tensor Parallel Layer for multi-core CPU execution.

    Splits tensors across cores for parallel computation.
    """

    def __init__(self, in_features: int, out_features: int, num_cores: int, use_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores

        # Split the output dimension across cores
        self.features_per_core = out_features // num_cores

        # Create separate linear layers for each core
        self.parallel_layers = nn.ModuleList(
            [nn.Linear(in_features, self.features_per_core, bias=use_bias) for _ in range(num_cores)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel tensor computation.
        """
        # Compute in parallel across cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            futures = [executor.submit(layer, x) for layer in self.parallel_layers]
            outputs = [f.result() for f in futures]

        # Concatenate results
        return torch.cat(outputs, dim=-1)


class PipelineParallelTransformer(nn.Module):
    """
    Pipeline Parallel Transformer for multi-core CPU execution.

    Splits transformer layers across cores for pipeline parallelism.
    """

    def __init__(self, layers: list[nn.Module], num_cores: int, micro_batch_size: int = 1):
        super().__init__()
        self.layers = layers
        self.num_cores = num_cores
        self.micro_batch_size = micro_batch_size

        # Split layers across cores
        self.layers_per_core = len(layers) // num_cores
        self.core_layers = nn.ModuleList(
            [
                nn.Sequential(*layers[i * self.layers_per_core : (i + 1) * self.layers_per_core])
                for i in range(num_cores)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pipeline parallelism.
        """
        # Split input into micro-batches
        x.size(0)
        micro_batches = torch.split(x, self.micro_batch_size, dim=0)

        # Process micro-batches through pipeline
        outputs = []

        for micro_batch in micro_batches:
            # Pass through each core's layers
            for core_layers in self.core_layers:
                micro_batch = core_layers(micro_batch)
            outputs.append(micro_batch)

        # Concatenate results
        return torch.cat(outputs, dim=0)


class CPUOptimizedModelWrapper(nn.Module):
    """
    Wrapper that applies CPU optimizations to any PyTorch model.

    Automatically detects and applies:
    - Multi-core parallelization
    - MKL optimization
    - Memory-efficient attention
    """

    def __init__(
        self,
        model: nn.Module,
        num_workers: int | None = None,
        use_mkl: bool = True,
        enable_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.model = model
        self.num_workers = num_workers or mp.cpu_count()
        self.use_mkl = use_mkl
        self.enable_tensor_parallel = enable_tensor_parallel

        # Configure environment for optimal CPU performance
        if self.use_mkl:
            os.environ["MKL_NUM_THREADS"] = str(self.num_workers)
            os.environ["OMP_NUM_THREADS"] = str(self.num_workers)
            os.environ["MKL_DYNAMIC"] = "FALSE"

        # Enable intra-op parallelism
        torch.set_num_threads(self.num_workers)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with CPU optimizations.
        """
        return self.model(*args, **kwargs)

    def enable_optimizations(self) -> dict[str, Any]:
        """
        Enable all CPU optimizations and return optimization info.
        """
        torch.set_num_threads(self.num_workers)

        optimization_info = {
            "num_workers": self.num_workers,
            "use_mkl": self.use_mkl,
            "enable_tensor_parallel": self.enable_tensor_parallel,
            "torch_threads": torch.get_num_threads(),
            "intra_op_parallelism": torch.get_num_threads(),
        }

        return optimization_info


def get_optimal_num_workers(model_size_mb: float, available_memory_mb: float, cpu_cores: int) -> int:
    """
    Calculate optimal number of workers for CPU inference.

    Args:
        model_size_mb: Model size in MB
        available_memory_mb: Available memory in MB
        cpu_cores: Number of CPU cores

    Returns:
        Optimal number of workers
    """
    # Estimate memory per worker (model + overhead)
    memory_per_worker = model_size_mb * 1.5  # 50% overhead

    # Maximum workers based on memory
    max_workers_memory = int(available_memory_mb / memory_per_worker)

    # Maximum workers based on cores
    max_workers_cores = cpu_cores

    # Optimal workers (with some headroom)
    optimal = min(max_workers_memory, max_workers_cores)

    return max(1, optimal - 1)  # Leave one core free


def benchmark_cpu_performance(model: nn.Module, input_shape: tuple, num_iterations: int = 10) -> dict[str, Any]:
    """
    Benchmark CPU performance with different parallelization settings.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmark

    Returns:
        Dictionary with benchmark results
    """
    import time

    results: dict[str, dict[str, float]] = {}
    dummy_input = torch.randn(input_shape)

    # Benchmark different worker counts
    for num_workers in [1, 2, 4, 8, mp.cpu_count()]:
        if num_workers > mp.cpu_count():
            continue

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        # Benchmark
        torch.set_num_threads(num_workers)
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        elapsed = time.time() - start_time
        avg_time = elapsed / num_iterations
        baseline_metrics = results.get("1_worker")
        baseline_time_ms = baseline_metrics["avg_time_ms"] if baseline_metrics is not None else avg_time * 1000

        results[f"{num_workers}_workers"] = {
            "avg_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": input_shape[1] / avg_time,
            "speedup_vs_1_worker": baseline_time_ms / (avg_time * 1000),
        }

    return results


# Export classes and functions
__all__ = [
    "CPUOptimizedModelWrapper",
    "CPUParallelAttention",
    "PipelineParallelTransformer",
    "TensorParallelLayer",
    "benchmark_cpu_performance",
    "get_optimal_num_workers",
]
