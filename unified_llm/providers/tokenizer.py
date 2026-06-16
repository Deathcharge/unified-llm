"""
Helix BPE Tokenizer
====================

Byte Pair Encoding tokenizer for the Helix LLM.  Replaces the raw byte-level
tokenizer with a learned subword vocabulary that compresses text ~4× better:

  byte-level:  "coordination" = 13 tokens  (one per byte)
  BPE (8k):    "coordination" ≈ 1–2 tokens

How it works
------------
1.  Start from the 256 raw byte tokens (same base as the old tokenizer).
2.  Iteratively merge the most-frequent adjacent token pair into a new
    token until a target vocab size is reached.
3.  Store the merge list + vocab as a compact JSON file.
4.  At encode time, apply the merge rules greedily to compress input.

The tokenizer is **pure-Python, zero external dependencies** (no sentencepiece,
no tiktoken, no HuggingFace tokenizers).  It can be trained on any UTF-8 text
corpus in seconds (CPU).

Compatibility
-------------
- ``vocab_size`` must match the model's ``ModelConfig.vocab_size``.
- The old ``HelixTokenizer`` (byte-level) is still available for backwards
  compatibility but is no longer the default for production configs.
- Trained vocabs are saved alongside checkpoints; inference automatically
  loads the right vocab when restoring from a checkpoint.

(c) Helix Collective 2025–2026 — Proprietary Technology Stack
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default vocab path — prefers Railway Volume, falls back to local
_DEFAULT_VOCAB_DIR = os.environ.get(
    "HELIX_BPE_VOCAB_DIR",
    "/data/tokenizer" if Path("/data").exists() else "models/tokenizer",
)

# Special token IDs (reserved at the front of the vocab)
PAD_TOKEN_ID = 0  # Also doubles as EOS for backwards compat with byte tokenizer
EOS_TOKEN_ID = 0
BOS_TOKEN_ID = 1  # New: beginning-of-sequence
UNK_TOKEN_ID = 2  # New: unknown (shouldn't appear after training)

# Reserve IDs 0–2 for special tokens; byte tokens start at offset 3
_BYTE_OFFSET = 3
_NUM_SPECIAL_TOKENS = 3

# Fill-in-the-Middle (FIM) sentinel token IDs — reserved at the END of the vocab.
# These are injected directly into token sequences during training (never produced
# by the BPE merge rules).  Based on the approach used by StarCoder and DeepSeek-Coder.
# They occupy the last 3 positions of the 32768-token vocabulary:
#   32765 → <PRE>  (prefix section marker)
#   32766 → <SUF>  (suffix section marker)
#   32767 → <MID>  (middle / fill-target marker)
FIM_PRE_TOKEN_ID = 32765  # <PRE>
FIM_SUF_TOKEN_ID = 32766  # <SUF>
FIM_MID_TOKEN_ID = 32767  # <MID>

# Default target vocab size — good balance for a 73M model on CPU
DEFAULT_VOCAB_SIZE = 8192


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class HelixBPETokenizer:
    """Byte Pair Encoding tokenizer for the Helix LLM.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (including 256 byte tokens + 3 special tokens).
        Must be ≥ 259 (256 bytes + 3 specials).  Default 8192.
    vocab_path : str | Path | None
        Path to a saved vocab JSON file.  If provided, the tokenizer loads
        the pre-trained merges immediately.  If None, call ``train()`` first.

    Example
    -------
    >>> tok = HelixBPETokenizer()
    >>> tok.train("The quick brown fox jumps over the lazy dog " * 1000)
    >>> ids = tok.encode("The quick brown fox")
    >>> tok.decode(ids)
    'The quick brown fox'
    >>> tok.save("models/tokenizer/vocab.json")
    """

    # Class-level constants for external access
    PAD_TOKEN_ID = PAD_TOKEN_ID
    EOS_TOKEN_ID = EOS_TOKEN_ID
    BOS_TOKEN_ID = BOS_TOKEN_ID
    UNK_TOKEN_ID = UNK_TOKEN_ID
    VOCAB_SIZE: int  # Set dynamically in __init__

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        vocab_path: str | Path | None = None,
    ) -> None:
        if vocab_size < _NUM_SPECIAL_TOKENS + 256:
            raise ValueError(
                f"vocab_size must be >= {_NUM_SPECIAL_TOKENS + 256} (256 bytes + {_NUM_SPECIAL_TOKENS} specials), got {vocab_size}"
            )

        self.target_vocab_size = vocab_size
        self.VOCAB_SIZE = vocab_size

        # Merge rules: list of (token_a, token_b) pairs in merge priority order.
        # The i-th merge creates token ID = _BYTE_OFFSET + 256 + i.
        self.merges: list[tuple[int, int]] = []

        # Vocab: token_id → bytes representation
        # IDs 0–2: special tokens (PAD/EOS, BOS, UNK)
        # IDs 3–258: raw bytes (0x00–0xFF)
        # IDs 259+: BPE merges
        self.vocab: dict[int, bytes] = {}
        self._token_to_id: dict[bytes, int] = {}

        # Always initialize the base vocab (special + 256 bytes)
        self._build_base_vocab()

        # Load pre-trained vocab if path provided
        if vocab_path is not None:
            self.load(vocab_path)

    # ------------------------------------------------------------------
    # Base vocab construction
    # ------------------------------------------------------------------

    def _build_base_vocab(self) -> None:
        """Build the base vocabulary: 3 specials + 256 byte tokens."""
        self.vocab = {}
        self._token_to_id = {}

        # Special tokens (use single-byte markers that won't collide)
        special_bytes = [
            b"\x00",  # PAD / EOS
            b"\x01",  # BOS
            b"\x02",  # UNK
        ]
        for i, sb in enumerate(special_bytes):
            self.vocab[i] = sb
            # Don't add specials to _token_to_id — they're accessed by ID only

        # Byte tokens: ID = byte_value + _BYTE_OFFSET
        for byte_val in range(256):
            token_id = byte_val + _BYTE_OFFSET
            token_bytes = bytes([byte_val])
            self.vocab[token_id] = token_bytes
            self._token_to_id[token_bytes] = token_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        text: str,
        *,
        verbose: bool = True,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE merges from a text corpus.

        Parameters
        ----------
        text : str
            The full training corpus as a single string.
        verbose : bool
            Log progress every 500 merges.
        min_frequency : int
            Minimum pair frequency to consider a merge (default 2).
        """
        t0 = time.time()
        num_merges = self.target_vocab_size - _NUM_SPECIAL_TOKENS - 256

        if num_merges <= 0:
            logger.info("Target vocab_size <= 259 — no BPE merges to learn.")
            return

        # Encode the corpus as a list of byte-token IDs
        raw_bytes = text.encode("utf-8")
        token_ids = [b + _BYTE_OFFSET for b in raw_bytes]

        if verbose:
            logger.info(
                "🔤 BPE training: %d bytes → learning %d merges (target vocab=%d)",
                len(raw_bytes),
                num_merges,
                self.target_vocab_size,
            )

        # Pre-split into chunks at whitespace boundaries for efficiency.
        # BPE merges should not cross word boundaries (GPT-2 style).
        # Split on whitespace-preceded tokens: each chunk starts with the
        # whitespace bytes that precede it.
        chunks = self._split_into_words(token_ids)

        self.merges = []

        for merge_idx in range(num_merges):
            # Count all adjacent pairs across chunks
            pair_counts: Counter = Counter()
            for chunk in chunks:
                for i in range(len(chunk) - 1):
                    pair_counts[(chunk[i], chunk[i + 1])] += 1

            if not pair_counts:
                if verbose:
                    logger.info("🔤 BPE: exhausted all pairs after %d merges", merge_idx)
                break

            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0]
            (pair_a, pair_b), freq = best_pair

            if freq < min_frequency:
                if verbose:
                    logger.info(
                        "🔤 BPE: stopped at merge %d — top pair freq=%d < min=%d",
                        merge_idx,
                        freq,
                        min_frequency,
                    )
                break

            # New token ID for this merge
            new_id = _BYTE_OFFSET + 256 + merge_idx

            # Register the merge
            self.merges.append((pair_a, pair_b))
            new_bytes = self.vocab[pair_a] + self.vocab[pair_b]
            self.vocab[new_id] = new_bytes
            self._token_to_id[new_bytes] = new_id

            # Apply the merge to all chunks
            chunks = [self._apply_merge(chunk, pair_a, pair_b, new_id) for chunk in chunks]

            if verbose and (merge_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "🔤 BPE merge %d/%d  (freq=%d, elapsed=%.1fs)",
                    merge_idx + 1,
                    num_merges,
                    freq,
                    elapsed,
                )

        # Update actual vocab size
        self.VOCAB_SIZE = _NUM_SPECIAL_TOKENS + 256 + len(self.merges)

        elapsed = time.time() - t0
        if verbose:
            logger.info(
                "✅ BPE training complete: %d merges learned, vocab_size=%d (%.1fs)",
                len(self.merges),
                self.VOCAB_SIZE,
                elapsed,
            )

    @staticmethod
    def _split_into_words(token_ids: list[int]) -> list[list[int]]:
        """Split token IDs into word-level chunks at whitespace boundaries.

        Each chunk includes the leading whitespace (if any), mirroring GPT-2's
        pre-tokenization strategy.  This prevents BPE from merging across word
        boundaries, which produces cleaner subwords.
        """
        if not token_ids:
            return []

        # Whitespace byte token IDs: space(32), tab(9), newline(10), cr(13)
        ws_ids = {
            32 + _BYTE_OFFSET,
            9 + _BYTE_OFFSET,
            10 + _BYTE_OFFSET,
            13 + _BYTE_OFFSET,
        }

        chunks: list[list[int]] = []
        current: list[int] = []

        for tid in token_ids:
            if tid in ws_ids and current:
                chunks.append(current)
                current = [tid]
            else:
                current.append(tid)

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _apply_merge(
        chunk: list[int],
        pair_a: int,
        pair_b: int,
        new_id: int,
    ) -> list[int]:
        """Replace all occurrences of (pair_a, pair_b) with new_id in a chunk."""
        if len(chunk) < 2:
            return chunk

        result: list[int] = []
        i = 0
        while i < len(chunk):
            if i < len(chunk) - 1 and chunk[i] == pair_a and chunk[i + 1] == pair_b:
                result.append(new_id)
                i += 2
            else:
                result.append(chunk[i])
                i += 1
        return result

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, max_length: int = 2048) -> list[int]:
        """Encode text to token IDs using the trained BPE merges.

        Parameters
        ----------
        text : str
            Input text to encode.
        max_length : int
            Maximum number of tokens to return (truncates if longer).

        Returns
        -------
        list[int]
            Token IDs.  If no merges are trained, falls back to raw bytes
            (equivalent to the old HelixTokenizer).
        """
        if not text:
            return []

        # Start with raw byte IDs
        raw_bytes = text.encode("utf-8")
        token_ids = [b + _BYTE_OFFSET for b in raw_bytes]

        # Apply each merge in priority order
        for pair_a, pair_b in self.merges:
            new_id = self._merge_pair_to_id(pair_a, pair_b)
            token_ids = self._apply_merge(token_ids, pair_a, pair_b, new_id)

        return token_ids[:max_length]

    def _merge_pair_to_id(self, pair_a: int, pair_b: int) -> int:
        """Look up the token ID for a merged pair."""
        merged_bytes = self.vocab[pair_a] + self.vocab[pair_b]
        return self._token_to_id[merged_bytes]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int] | list) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        token_ids : list[int] or torch.Tensor
            Token IDs to decode.

        Returns
        -------
        str
            Decoded UTF-8 text.  Stops at the first EOS/PAD token (ID 0).
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        byte_chunks: list[bytes] = []
        for tid in token_ids:
            tid = int(tid)
            if tid == EOS_TOKEN_ID:
                break
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            # Skip unknown IDs silently

        raw = b"".join(byte_chunks)
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("UTF-8 decode failed, falling back to latin-1: %s", e)
            return raw.decode("latin-1", errors="replace")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained vocabulary to a JSON file.

        File format::

            {
              "version": 1,
              "vocab_size": 8192,
              "num_merges": 7933,
              "merges": [[3, 4], [259, 260], ...],
            }

        Only the merge list is stored.  The base vocab (specials + 256 bytes)
        is reconstructed deterministically at load time.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "vocab_size": self.VOCAB_SIZE,
            "num_merges": len(self.merges),
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        logger.info("💾 Saved BPE vocab to %s (%d merges, vocab_size=%d)", path, len(self.merges), self.VOCAB_SIZE)

    def load(self, path: str | Path) -> None:
        """Load a trained vocabulary from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("BPE vocab file not found: {}".format(path))

        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("version", 0) != 1:
            raise ValueError("Unsupported BPE vocab version: {}".format(data.get("version")))

        # Rebuild base vocab
        self._build_base_vocab()

        # Replay merges to reconstruct the full vocab
        self.merges = []
        for i, (pair_a, pair_b) in enumerate(data["merges"]):
            new_id = _BYTE_OFFSET + 256 + i
            new_bytes = self.vocab[pair_a] + self.vocab[pair_b]
            self.vocab[new_id] = new_bytes
            self._token_to_id[new_bytes] = new_id
            self.merges.append((pair_a, pair_b))

        self.VOCAB_SIZE = _NUM_SPECIAL_TOKENS + 256 + len(self.merges)
        logger.info("✅ Loaded BPE vocab from %s (%d merges, vocab_size=%d)", path, len(self.merges), self.VOCAB_SIZE)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compression_ratio(self, text: str) -> float:
        """Compute the compression ratio vs raw bytes.

        Returns the ratio ``len(raw_bytes) / len(bpe_tokens)``.
        Higher is better — e.g. 4.0 means BPE is 4× more compact.
        """
        raw_len = len(text.encode("utf-8"))
        bpe_len = len(self.encode(text))
        return raw_len / max(bpe_len, 1)

    @property
    def trained(self) -> bool:
        """Whether the tokenizer has learned BPE merges."""
        return len(self.merges) > 0

    def __repr__(self) -> str:
        return f"HelixBPETokenizer(vocab_size={self.VOCAB_SIZE}, merges={len(self.merges)}, trained={self.trained})"


# ---------------------------------------------------------------------------
# Factory: auto-detect best available tokenizer
# ---------------------------------------------------------------------------


def get_tokenizer(
    vocab_path: str | Path | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> HelixBPETokenizer:
    """Get the best available tokenizer.

    Resolution order:
    1.  If ``vocab_path`` is given and exists → load that BPE vocab.
    2.  Check the default vocab directory for ``vocab.json``.
    3.  Return an untrained BPE tokenizer (behaves identically to the old
        byte-level HelixTokenizer until ``train()`` is called).

    Parameters
    ----------
    vocab_path : str | Path | None
        Explicit path to a vocab JSON file.
    vocab_size : int
        Target vocab size for a fresh (untrained) tokenizer.
    """
    # 1. Explicit path
    if vocab_path is not None:
        p = Path(vocab_path)
        if p.exists():
            return HelixBPETokenizer(vocab_size=vocab_size, vocab_path=p)
        logger.warning("Vocab path %s not found — creating untrained tokenizer", p)
        return HelixBPETokenizer(vocab_size=vocab_size)

    # 2. Default location
    default_path = Path(_DEFAULT_VOCAB_DIR) / "vocab.json"
    if default_path.exists():
        tok = HelixBPETokenizer(vocab_size=vocab_size, vocab_path=default_path)
        return tok

    # 3. Fresh untrained (operates as byte-level until trained)
    return HelixBPETokenizer(vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# CLI: train vocab from command line
# ---------------------------------------------------------------------------


def train_bpe_vocab(
    data_dirs: list[str] | None = None,
    output_path: str | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
) -> HelixBPETokenizer:
    """Train a BPE vocab from data directories (convenience function).

    Parameters
    ----------
    data_dirs : list[str] | None
        Directories to scan for .txt/.md/.py/.rst/.jsonl files.
        Defaults to ["docs"].
    output_path : str | None
        Where to save the vocab JSON.  Defaults to the standard location.
    vocab_size : int
        Target vocabulary size.
    min_frequency : int
        Minimum merge pair frequency.

    Returns
    -------
    HelixBPETokenizer
        The trained tokenizer.
    """
    if data_dirs is None:
        data_dirs = ["docs"]

    if output_path is None:
        output_path = str(Path(_DEFAULT_VOCAB_DIR) / "vocab.json")

    # Collect text from all data directories
    extensions = [".txt", ".md", ".py", ".rst", ".jsonl"]
    all_text: list[str] = []

    for data_dir in data_dirs:
        dp = Path(data_dir)
        if not dp.exists():
            logger.warning("Data directory %s not found, skipping", data_dir)
            continue

        for ext in extensions:
            for fp in sorted(dp.rglob("*{}".format(ext))):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    if len(text) > 50:
                        all_text.append(text)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", fp, e)

    if not all_text:
        raise ValueError("No training text found in directories: {}".format(data_dirs))

    corpus = "\n\n".join(all_text)
    logger.info("📚 Collected %d files, %d chars for BPE training", len(all_text), len(corpus))

    # Train
    tok = HelixBPETokenizer(vocab_size=vocab_size)
    tok.train(corpus, min_frequency=min_frequency)

    # Demo compression
    sample = corpus[:500] if len(corpus) > 500 else corpus
    ratio = tok.compression_ratio(sample)
    logger.info(
        "📊 Compression ratio on sample: %.2f× (%.0f%% fewer tokens than byte-level)", ratio, (1 - 1 / ratio) * 100
    )

    # Save
    tok.save(output_path)
    return tok


# ---------------------------------------------------------------------------
# Module entry point for CLI: python -m apps.backend.proprietary_llm.tokenizer
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["docs"]
    train_bpe_vocab(data_dirs=dirs)
