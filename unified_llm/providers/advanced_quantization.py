"""
Advanced Quantization Techniques for CPU-Optimized LLM

Implements state-of-the-art quantization methods including GGUF, AWQ, GPTQ, and EXL2
for efficient CPU inference with minimal accuracy loss.
"""

from enum import Enum
from typing import Any, cast

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class QuantizationMethod(Enum):
    """Supported quantization methods."""

    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    EXL2 = "exl2"
    DYNAMIC = "dynamic"


class GGUFQuantizer(nn.Module):
    """
    GGUF Quantization Implementation

    GGUF is a quantization format that uses per-tensor scale factors and zero-points
    with mixed precision (e.g., 4-bit weights with 8-bit activations).
    """

    def __init__(self, weight_bitwidth: int = 4, activation_bitwidth: int = 16, group_size: int = 128):
        super().__init__()
        self.weight_bitwidth = weight_bitwidth
        self.activation_bitwidth = activation_bitwidth
        self.group_size = group_size

        # Quantization range for weights
        self.weight_scale = None
        self.weight_zero_point = None

        # Quantization range for activations
        self.activation_scale = None
        self.activation_zero_point = None

    def quantize_weights(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weights using GGUF format.

        Returns:
            Tuple of (quantized_weights, scale, zero_point)
        """
        # Reshape for grouped quantization
        original_shape = weights.shape
        weights_reshaped = weights.view(-1, self.group_size)

        # Find min and max per group
        min_vals = weights_reshaped.amin(dim=1, keepdim=True)
        max_vals = weights_reshaped.amax(dim=1, keepdim=True)

        # Calculate scale and zero point
        self.weight_scale = (max_vals - min_vals) / (2**self.weight_bitwidth - 1)
        self.weight_zero_point = (-min_vals / self.weight_scale).round()

        # Quantize
        quantized = ((weights_reshaped / self.weight_scale) + self.weight_zero_point).clamp(
            0, 2**self.weight_bitwidth - 1
        )

        # Store as appropriate dtype
        if self.weight_bitwidth == 4:
            quantized = quantized.to(torch.uint8)
            # Pack two 4-bit values per byte
            packed = (quantized[:, ::2] << 4) | quantized[:, 1::2]
            quantized = packed
        else:
            quantized = quantized.to(torch.uint8)

        # Reshape back
        quantized = quantized.view(original_shape[0], -1)

        return (
            quantized,
            cast(torch.Tensor, self.weight_scale).squeeze(),
            cast(torch.Tensor, self.weight_zero_point).squeeze(),
        )

    def dequantize_weights(
        self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, original_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """
        Dequantize weights from GGUF format.
        """
        # Unpack if 4-bit
        if self.weight_bitwidth == 4:
            unpacked_low = quantized & 0x0F
            unpacked_high = quantized >> 4
            quantized = torch.stack([unpacked_low, unpacked_high], dim=-1).flatten(-2)

        # Dequantize
        dequantized = (quantized.to(torch.float32) - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)

        # Reshape to original
        dequantized = dequantized.view(original_shape)

        return dequantized


class AWQQuantizer(nn.Module):
    """
    Activation-aware Weight Quantization (AWQ)

    AWQ preserves outliers in weights by using activation statistics to guide
    quantization. This leads to better accuracy at 4-bit precision.
    """

    def __init__(self, weight_bitwidth: int = 4, clip_ratio: float = 0.99):
        super().__init__()
        self.weight_bitwidth = weight_bitwidth
        self.clip_ratio = clip_ratio

        # Learned scaling factors per channel
        self.scales = nn.Parameter(torch.ones(1))

    def collect_activation_stats(self, module: nn.Module, dataloader, num_batches: int = 10) -> torch.Tensor:
        """
        Collect activation statistics for calibration.
        """
        activation_magnitudes = []

        module.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break

                # Hook to collect activations
                activations: list[torch.Tensor] = []

                def hook_fn(module, input, output, _activations=activations):
                    _activations.append(output.abs().mean())

                handle = module.register_forward_hook(hook_fn)

                try:
                    _ = module(inputs)
                    activation_magnitudes.append(activations[0])
                finally:
                    handle.remove()

        # Average activation magnitudes
        avg_magnitudes = torch.stack(activation_magnitudes).mean(dim=0)

        return avg_magnitudes

    def quantize_weights(
        self, weights: torch.Tensor, activation_magnitudes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights using AWQ method.
        """
        # Calculate channel-wise scaling factors based on activation statistics
        scales = activation_magnitudes / weights.abs().mean(dim=1, keepdim=True)
        scales = scales.clamp(min=0.5, max=2.0)

        # Scale weights before quantization
        scaled_weights = weights * scales

        # Clip outliers
        clip_val = torch.quantile(scaled_weights.abs(), self.clip_ratio)
        scaled_weights = scaled_weights.clamp(-clip_val, clip_val)

        # Quantize to specified bitwidth
        qmin = -(2 ** (self.weight_bitwidth - 1))
        qmax = 2 ** (self.weight_bitwidth - 1) - 1

        scale = clip_val / qmax
        zero_point = 0  # Symmetric quantization

        quantized = torch.clamp(torch.round(scaled_weights / scale) + zero_point, qmin, qmax)

        return quantized.to(torch.int8), scale


class GPTQQuantizer(nn.Module):
    """
    GPTQ Quantization

    Uses Hessian information to minimize quantization error in a single pass.
    Particularly effective for transformer models.
    """

    def __init__(self, weight_bitwidth: int = 4, damp_percent: float = 0.01):
        super().__init__()
        self.weight_bitwidth = weight_bitwidth
        self.damp_percent = damp_percent

    def quantize_layer(
        self, weight: torch.Tensor, hessian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a layer using GPTQ algorithm.
        """
        assert weight.shape[1] == hessian.shape[0], "Shape mismatch"

        n = weight.shape[1]

        # Initialize quantization parameters
        qmin = -(2 ** (self.weight_bitwidth - 1))
        qmax = 2 ** (self.weight_bitwidth - 1) - 1

        quantized_weight = weight.clone()
        torch.zeros_like(weight)

        for i in range(n):
            # Calculate optimal scale for this column
            column = weight[:, i]
            scale = column.abs().max() / qmax

            # Quantize
            quantized = torch.clamp(torch.round(column / scale), qmin, qmax)
            quantized_weight[:, i] = quantized * scale

            # Calculate quantization error
            error = column - quantized_weight[:, i]

            # Update error for next columns using Hessian
            if i < n - 1:
                hess_row = hessian[i, i + 1 :]
                weight[:, i + 1 :] += torch.outer(error, hess_row / hessian[i, i])

        return quantized_weight, scale, 0


class DynamicQuantization(nn.Module):
    """
    Dynamic Quantization

    Quantizes weights statically but activations dynamically during inference.
    Good for CPUs with mixed precision support.
    """

    def __init__(self, weight_bitwidth: int = 8, activation_bitwidth: int = 8):
        super().__init__()
        self.weight_bitwidth = weight_bitwidth
        self.activation_bitwidth = activation_bitwidth

        self.quantized_modules: dict[nn.Module, dict[str, Any]] = {}

    def quantize_linear(self, module: nn.Linear) -> nn.Module:
        """
        Quantize a linear layer.
        """
        # Get weight statistics
        weight = module.weight.data
        scale = weight.abs().max() / (2 ** (self.weight_bitwidth - 1) - 1)
        zero_point = 0  # Symmetric quantization

        # Quantize weights
        qmin = -(2 ** (self.weight_bitwidth - 1))
        qmax = 2 ** (self.weight_bitwidth - 1) - 1

        quantized_weight = torch.clamp(torch.round(weight / scale) + zero_point, qmin, qmax).to(torch.int8)

        # Store quantization parameters
        self.quantized_modules[module] = {
            "quantized_weight": quantized_weight,
            "scale": scale,
            "zero_point": zero_point,
            "bias": module.bias.data if module.bias is not None else None,
        }

        return module

    def forward(self, x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        """
        Forward pass with dynamic quantization.
        """
        if module not in self.quantized_modules:
            self.quantize_linear(module)

        params = self.quantized_modules[module]

        # Quantize input (dynamic)
        qmin_act = -(2 ** (self.activation_bitwidth - 1))
        qmax_act = 2 ** (self.activation_bitwidth - 1) - 1

        x_scale = x.abs().max() / qmax_act
        x_quantized = torch.clamp(torch.round(x / x_scale), qmin_act, qmax_act).to(torch.int8)

        # Dequantize and compute
        w_dequantized = params["quantized_weight"].to(torch.float32) * params["scale"]
        x_dequantized = x_quantized.to(torch.float32) * x_scale

        output = torch.matmul(x_dequantized, w_dequantized.t())

        if params["bias"] is not None:
            output = output + params["bias"]

        return output


def calculate_memory_savings(
    original_size_bytes: int, quantization_method: QuantizationMethod, bitwidth: int
) -> dict[str, Any]:
    """
    Calculate memory savings from quantization.
    """
    # Original memory (assuming FP16)
    original_memory = original_size_bytes

    # Quantized memory
    quantized_memory = (original_size_bytes * bitwidth) / 16

    # Savings
    savings = original_memory - quantized_memory
    savings_percent = (savings / original_memory) * 100

    return {
        "method": quantization_method.value,
        "bitwidth": bitwidth,
        "original_mb": original_memory / (1024 * 1024),
        "quantized_mb": quantized_memory / (1024 * 1024),
        "savings_mb": savings / (1024 * 1024),
        "savings_percent": savings_percent,
        "compression_ratio": original_memory / quantized_memory,
    }


# Export classes and functions
__all__ = [
    "AWQQuantizer",
    "DynamicQuantization",
    "GGUFQuantizer",
    "GPTQQuantizer",
    "QuantizationMethod",
    "calculate_memory_savings",
]
