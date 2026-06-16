"""
Speculative Decoding Implementation for CPU-Optimized LLM

Speculative decoding uses a smaller, faster draft model to predict tokens that are
then verified by a larger target model. This can significantly speed up inference
while maintaining the quality of the larger model.
"""

import time
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Helix proprietary LLM modules. "
        "Install CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class SpeculativeDecoder(nn.Module):
    """
    Speculative Decoding with Draft and Target Models.

    The draft model is smaller and faster, generating candidate tokens.
    The target model is larger and more accurate, verifying the candidates.
    """

    def __init__(
        self, draft_model: nn.Module, target_model: nn.Module, speculation_steps: int = 5, use_parallel: bool = True
    ):
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.speculation_steps = speculation_steps
        self.use_parallel = use_parallel

        # Statistics
        self.total_tokens_generated = 0
        self.accepted_tokens = 0
        self.total_draft_steps = 0
        self.total_target_steps = 0

    def _generate_draft_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_tokens: int
    ) -> list[int]:
        """
        Generate candidate tokens using the draft model.
        """
        self.draft_model.eval()
        draft_tokens = []

        current_input = input_ids.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.draft_model(current_input, attention_mask=attention_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # Sample next token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                draft_tokens.append(next_token.item())

                # Append token for next iteration
                current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1
                    )

        return draft_tokens

    def _verify_with_target(
        self, input_ids: torch.Tensor, draft_tokens: list[int], attention_mask: torch.Tensor | None = None
    ) -> tuple[list[int], int]:
        """
        Verify draft tokens using the target model.

        Returns:
            Tuple of (accepted_tokens, accepted_position)
        """
        self.target_model.eval()

        accepted_tokens = []
        accepted_position = 0

        current_input = input_ids.clone()

        with torch.no_grad():
            for i, draft_token in enumerate(draft_tokens):
                self.total_target_steps += 1

                # Get target model's prediction
                outputs = self.target_model(current_input, attention_mask=attention_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                target_logits = logits[:, -1, :]

                # Compare with draft token
                target_token = torch.argmax(target_logits, dim=-1)

                # Calculate acceptance probability
                target_probs = F.softmax(target_logits, dim=-1)
                acceptance_prob = target_probs[0, draft_token].item()

                # Accept or reject based on probability comparison
                if torch.rand(1).item() < acceptance_prob:
                    accepted_tokens.append(draft_token)
                    accepted_position = i + 1
                    self.accepted_tokens += 1

                    # Append accepted token
                    current_input = torch.cat(
                        [current_input, torch.tensor([[draft_token]], device=current_input.device)], dim=1
                    )

                    # Update attention mask
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
                            dim=1,
                        )
                else:
                    # Reject and use target's prediction
                    accepted_tokens.append(target_token.item())
                    break

        return accepted_tokens, accepted_position

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_ids, statistics)
        """
        self.draft_model.eval()
        self.target_model.eval()

        generated_ids = input_ids.clone()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Generate draft tokens
                self.total_draft_steps += self.speculation_steps
                draft_tokens = self._generate_draft_tokens(generated_ids, attention_mask, self.speculation_steps)

                # Verify with target model
                accepted_tokens, accepted_pos = self._verify_with_target(generated_ids, draft_tokens, attention_mask)

                # Append accepted tokens
                if accepted_tokens:
                    new_tokens = torch.tensor([accepted_tokens], device=input_ids.device)
                    generated_ids = torch.cat([generated_ids, new_tokens], dim=1)
                    self.total_tokens_generated += len(accepted_tokens)

                    # Update attention mask
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    (attention_mask.size(0), len(accepted_tokens)), device=attention_mask.device
                                ),
                            ],
                            dim=1,
                        )

                # Stop if we rejected early
                if accepted_pos < self.speculation_steps:
                    continue

                # Stop if we've generated enough tokens
                if generated_ids.size(1) - input_ids.size(1) >= max_new_tokens:
                    break

        elapsed_time = time.time() - start_time

        # Calculate statistics
        stats = {
            "total_tokens_generated": self.total_tokens_generated,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": (
                (self.accepted_tokens / self.total_tokens_generated * 100) if self.total_tokens_generated > 0 else 0
            ),
            "total_draft_steps": self.total_draft_steps,
            "total_target_steps": self.total_target_steps,
            "speedup": self.total_draft_steps / self.total_target_steps if self.total_target_steps > 0 else 1,
            "tokens_per_second": self.total_tokens_generated / elapsed_time if elapsed_time > 0 else 0,
            "elapsed_time_seconds": elapsed_time,
        }

        return generated_ids, stats

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_tokens_generated = 0
        self.accepted_tokens = 0
        self.total_draft_steps = 0
        self.total_target_steps = 0


class LookaheadDecoding(nn.Module):
    """
    Lookahead Decoding - a simpler variant of speculative decoding.

    Instead of using a separate draft model, lookahead decoding uses the
    same model but speculates multiple tokens ahead with n-gram lookahead.
    """

    def __init__(self, model: nn.Module, ngram_size: int = 2, speculation_steps: int = 3):
        super().__init__()
        self.model = model
        self.ngram_size = ngram_size
        self.speculation_steps = speculation_steps

        # N-gram cache
        self.ngram_cache: dict[tuple[int, ...], list[tuple[int, int]]] = {}

    def _build_ngram_cache(self, text: list[int]):
        """Build n-gram frequency cache from text."""
        for i in range(len(text) - self.ngram_size):
            ngram = tuple(text[i : i + self.ngram_size])
            next_token = text[i + self.ngram_size]

            if ngram not in self.ngram_cache:
                self.ngram_cache[ngram] = []

            self.ngram_cache[ngram].append((next_token, 1))

        # Consolidate counts
        for ngram in self.ngram_cache:
            token_counts: dict[int, int] = {}
            for token, count in self.ngram_cache[ngram]:
                if token not in token_counts:
                    token_counts[token] = 0
                token_counts[token] += count

            # Sort by frequency
            self.ngram_cache[ngram] = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    def _get_lookahead_tokens(self, context: list[int]) -> list[int]:
        """Get candidate tokens using n-gram lookahead."""
        if len(context) < self.ngram_size:
            return []

        ngram = tuple(context[-self.ngram_size :])

        if ngram in self.ngram_cache:
            # Return top-k candidates
            return [token for token, _ in self.ngram_cache[ngram][: self.speculation_steps]]

        return []

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 100
    ) -> torch.Tensor:
        """
        Generate tokens using lookahead decoding.
        """
        self.model.eval()

        generated_ids = input_ids.clone()
        context = input_ids[0].tolist()

        # Build n-gram cache from context
        self._build_ngram_cache(context)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get lookahead tokens
                lookahead_tokens = self._get_lookahead_tokens(context)

                if lookahead_tokens:
                    # Verify lookahead tokens
                    outputs = self.model(generated_ids, attention_mask=attention_mask)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

                    # Check each lookahead token
                    accepted_token = None
                    for token in lookahead_tokens:
                        token_prob = F.softmax(logits[:, -1, :], dim=-1)[0, token].item()

                        if torch.rand(1).item() < token_prob:
                            accepted_token = token
                            break

                    if accepted_token is not None:
                        # Accept lookahead token
                        new_token = torch.tensor([[accepted_token]], device=input_ids.device)
                        generated_ids = torch.cat([generated_ids, new_token], dim=1)
                        context.append(accepted_token)
                        continue

                # Fallback to standard decoding
                outputs = self.model(generated_ids, attention_mask=attention_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
                context.append(next_token.item())

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1
                    )

        return generated_ids


def calculate_speculative_decoding_benefit(
    draft_tokens_per_second: float, target_tokens_per_second: float, acceptance_rate: float, speculation_steps: int
) -> dict[str, Any]:
    """
    Calculate the theoretical speedup from speculative decoding.

    Args:
        draft_tokens_per_second: Draft model throughput
        target_tokens_per_second: Target model throughput
        acceptance_rate: Token acceptance rate (0-1)
        speculation_steps: Number of speculation steps

    Returns:
        Dictionary with speedup calculations
    """
    # Standard decoding time (target only)
    standard_time_per_token = 1.0 / target_tokens_per_second

    # Speculative decoding time
    # Draft model generates speculation_steps tokens
    draft_time = speculation_steps / draft_tokens_per_second

    # Target model verifies accepted_tokens = acceptance_rate * speculation_steps
    accepted_tokens = acceptance_rate * speculation_steps
    target_time = accepted_tokens / target_tokens_per_second

    # Total speculative time
    speculative_time = draft_time + target_time

    # Effective tokens per second
    speculative_tokens_per_second = accepted_tokens / speculative_time

    # Speedup
    speedup = speculative_tokens_per_second / target_tokens_per_second

    return {
        "draft_tokens_per_second": draft_tokens_per_second,
        "target_tokens_per_second": target_tokens_per_second,
        "acceptance_rate": acceptance_rate,
        "speculation_steps": speculation_steps,
        "standard_time_per_token_ms": standard_time_per_token * 1000,
        "draft_time_ms": draft_time * 1000,
        "target_time_ms": target_time * 1000,
        "speculative_time_ms": speculative_time * 1000,
        "speculative_tokens_per_second": speculative_tokens_per_second,
        "speedup": speedup,
        "speedup_percent": (speedup - 1) * 100,
    }


# Export classes and functions
__all__ = ["LookaheadDecoding", "SpeculativeDecoder", "calculate_speculative_decoding_benefit"]
