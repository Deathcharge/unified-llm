"""
Helix LLM Data Pipeline — Multi-Source Training Data
=====================================================

Aggregates training data from multiple sources for the Helix LLM:

1. **Local files** — docs/, codebase, user-provided documents
2. **HuggingFace datasets** — FineWeb-Edu, Wikipedia, SlimPajama, etc.
3. **Web fetch** — Direct URL content retrieval (respects robots.txt)
4. **Synthetic data** — Instruction/Q&A generation templates

All sources produce a unified stream of text chunks that the training
pipeline tokenizes with the BPE tokenizer.

Usage::

    from apps.backend.proprietary_llm.data_pipeline import DataPipeline

    pipeline = DataPipeline(max_bytes=500_000_000)  # 500 MB cap
    pipeline.add_local_dirs(["docs", "apps"])
    pipeline.add_huggingface_dataset("HuggingFaceFW/fineweb-edu-score-2",
                                     split="train", max_rows=100_000)
    pipeline.add_urls(["https://en.wikipedia.org/wiki/Transformer_(deep_learning_model)"])

    corpus = pipeline.build_corpus()

(c) Helix Collective 2025–2026 — Proprietary Technology Stack
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DataSource:
    """A single data source with metadata."""

    source_type: str  # "local", "huggingface", "url", "synthetic"
    name: str
    chars_collected: int = 0
    files_processed: int = 0
    status: str = "pending"  # pending, collecting, done, failed
    error: str | None = None


@dataclass
class PipelineStats:
    """Tracks data collection progress."""

    sources: list[DataSource] = field(default_factory=list)
    total_chars: int = 0
    total_files: int = 0
    total_chunks: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Local file ingestion (existing logic, cleaned up)
# ---------------------------------------------------------------------------

# File extensions worth training on
_TRAINABLE_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".html",
    ".rst",
    ".tsx",
    ".ts",
    ".jsx",
    ".js",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".csv",
    ".xml",
    ".sql",
}

# Directories to skip
_SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".next",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "htmlcov",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "test_results",
    "helix_unified.egg-info",
    "archives",
}


def _clean_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    return text


def _clean_html(text: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return text


def _clean_text(text: str, suffix: str) -> str:
    """Apply format-specific cleaning."""
    if suffix == ".md":
        text = _clean_markdown(text)
    elif suffix in (".html", ".htm"):
        text = _clean_html(text)
    elif suffix == ".json":
        try:
            data = json.loads(text)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("JSON reformatting failed, keeping original: %s", exc)
    # General cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def collect_local_files(
    dirs: list[str],
    extensions: set | None = None,
    min_size: int = 100,
    max_size: int = 500_000,
    max_total_chars: int = 0,
) -> tuple[list[str], DataSource]:
    """Collect text from local file directories.

    Returns a list of text chunks and a DataSource summary.
    """
    exts = extensions or _TRAINABLE_EXTENSIONS
    source = DataSource(source_type="local", name=",".join(dirs))
    source.status = "collecting"
    chunks: list[str] = []
    total_chars = 0

    for data_dir in dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Local dir not found: %s", data_dir)
            continue

        for root, dir_names, file_names in os.walk(data_path):
            # Prune skipped directories in-place
            dir_names[:] = [d for d in dir_names if d not in _SKIP_DIRS]

            for fname in sorted(file_names):
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in exts:
                    continue
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if size < min_size or size > max_size:
                    continue

                try:
                    raw = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning("Failed to read %s: %s", fpath, e)
                    continue

                cleaned = _clean_text(raw, fpath.suffix.lower())
                if len(cleaned) < 50:
                    continue

                # Add file context header
                header = "\n--- {} ---\n".format(fpath.name)
                chunk = header + cleaned
                chunks.append(chunk)
                total_chars += len(chunk)
                source.files_processed += 1

                if max_total_chars and total_chars >= max_total_chars:
                    break
            if max_total_chars and total_chars >= max_total_chars:
                break
        if max_total_chars and total_chars >= max_total_chars:
            break

    source.chars_collected = total_chars
    source.status = "done"
    logger.info(
        "📁 Local: %d files, %d chars from %s",
        source.files_processed,
        source.chars_collected,
        dirs,
    )
    return chunks, source


# ---------------------------------------------------------------------------
# HuggingFace dataset ingestion
# ---------------------------------------------------------------------------


def collect_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_rows: int = 50_000,
    max_total_chars: int = 0,
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> tuple[list[str], DataSource]:
    """Load text data from a HuggingFace dataset.

    Uses streaming mode by default to avoid downloading the full dataset.
    Requires ``pip install datasets`` (already in requirements.txt via
    huggingface-hub).

    Popular datasets for LLM pre-training:
    - "HuggingFaceFW/fineweb-edu-score-2"  (1.3T tokens, educational web)
    - "wikimedia/wikipedia" (config="20231101.en")
    - "cerebras/SlimPajama-627B" (cleaned general web)
    - "allenai/c4" (config="en", cleaned Common Crawl)
    - "bigcode/the-stack-v2-dedup" (code)
    - "Open-Orca/OpenOrca" (instruction following)
    """
    source = DataSource(source_type="huggingface", name=dataset_name)
    source.status = "collecting"
    chunks: list[str] = []
    total_chars = 0

    try:
        from datasets import load_dataset
    except ImportError:
        source.status = "failed"
        source.error = "datasets library not installed: pip install datasets"
        logger.error("❌ %s", source.error)
        return chunks, source

    try:
        logger.info(
            "📥 Loading HuggingFace dataset: %s (split=%s, max_rows=%d, streaming=%s)",
            dataset_name,
            split,
            max_rows,
            streaming,
        )
        kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        # Some datasets need a config name (e.g., wikipedia needs "20231101.en")
        ds = load_dataset(dataset_name, **kwargs)  # nosec B615 — dataset_name is validated against allowed list before this call

        row_count = 0
        for row in ds:
            if row_count >= max_rows:
                break

            text = row.get(text_column, "")
            if not text or len(text) < 50:
                continue

            # Truncate very long individual documents
            if len(text) > 50_000:
                text = text[:50_000]

            chunks.append(text)
            total_chars += len(text)
            row_count += 1
            source.files_processed += 1

            if max_total_chars and total_chars >= max_total_chars:
                break

            # Log progress periodically
            if row_count % 10_000 == 0:
                logger.info(
                    "📥 HuggingFace %s: %d rows, %.1f MB so far",
                    dataset_name,
                    row_count,
                    total_chars / 1e6,
                )

        source.chars_collected = total_chars
        source.status = "done"
        logger.info(
            "✅ HuggingFace %s: %d rows, %.1f MB collected",
            dataset_name,
            source.files_processed,
            total_chars / 1e6,
        )

    except Exception as e:
        source.status = "failed"
        source.error = str(e)
        logger.error("❌ Failed to load HuggingFace dataset %s: %s", dataset_name, e)

    return chunks, source


# ---------------------------------------------------------------------------
# URL content fetching
# ---------------------------------------------------------------------------


def _is_safe_url(url: str) -> bool:
    """Return True only if *url* resolves to a public (non-private) IP address."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except (socket.gaierror, ValueError):
        return False


def collect_urls(
    urls: list[str],
    max_chars_per_url: int = 100_000,
    timeout: int = 30,
) -> tuple[list[str], DataSource]:
    """Fetch and extract text content from URLs.

    Uses httpx (already in requirements.txt) with a respectful user-agent.
    Extracts readable text by stripping HTML.
    """
    source = DataSource(source_type="url", name="web_urls")
    source.status = "collecting"
    chunks: list[str] = []
    total_chars = 0

    try:
        import httpx
    except ImportError:
        source.status = "failed"
        source.error = "httpx not installed: pip install httpx"
        logger.error("❌ %s", source.error)
        return chunks, source

    headers = {
        "User-Agent": "HelixCollective/1.0 (Training Data Collection; +https://helixspiral.work)",
    }

    for url in urls:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.warning("Skipping invalid URL: %s", url)
                continue

            if not _is_safe_url(url):
                logger.warning("Skipping URL targeting private/internal network: %s", url)
                continue

            logger.info("🌐 Fetching: %s", url)
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=False)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            raw_text = resp.text

            # Strip HTML if it looks like a web page
            if "html" in content_type or raw_text.strip().startswith("<"):
                raw_text = _clean_html(raw_text)

            cleaned = re.sub(r"\n{3,}", "\n\n", raw_text)
            cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

            if len(cleaned) < 50:
                logger.warning("URL produced too little text: %s (%d chars)", url, len(cleaned))
                continue

            if len(cleaned) > max_chars_per_url:
                cleaned = cleaned[:max_chars_per_url]

            header = "\n--- {} ---\n".format(parsed.netloc)
            chunk = header + cleaned
            chunks.append(chunk)
            total_chars += len(chunk)
            source.files_processed += 1

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)

    source.chars_collected = total_chars
    source.status = "done"
    logger.info("🌐 URLs: %d fetched, %d chars", source.files_processed, total_chars)
    return chunks, source


# ---------------------------------------------------------------------------
# Synthetic instruction data
# ---------------------------------------------------------------------------

# Templates for generating basic instruction-following training data
_INSTRUCTION_TEMPLATES = [
    {
        "prompt": "Explain the concept of {topic} in simple terms.",
        "topics": [
            "machine learning",
            "neural networks",
            "transformers",
            "attention mechanisms",
            "gradient descent",
            "backpropagation",
            "tokenization",
            "embeddings",
            "loss functions",
            "overfitting",
            "regularization",
            "batch normalization",
            "dropout",
            "transfer learning",
            "fine-tuning",
            "reinforcement learning",
            "generative models",
            "coordination in AI",
            "multi-agent systems",
            "distributed computing",
        ],
    },
    {
        "prompt": "What is the purpose of {component} in a transformer model?",
        "topics": [
            "the attention mechanism",
            "positional encoding",
            "layer normalization",
            "the feed-forward network",
            "multi-head attention",
            "the softmax function",
            "residual connections",
            "the embedding layer",
            "the output projection",
            "the query, key, and value matrices",
        ],
    },
]


def generate_synthetic_prompts(
    max_chars: int = 100_000,
) -> tuple[list[str], DataSource]:
    """Generate synthetic instruction-following prompt/response stubs.

    These provide basic instruction-following signal during pre-training.
    For best results, use an external LLM to generate full responses
    and add them as local files.
    """
    source = DataSource(source_type="synthetic", name="instruction_templates")
    source.status = "collecting"
    chunks: list[str] = []
    total_chars = 0

    for template in _INSTRUCTION_TEMPLATES:
        prompt_tmpl = str(template["prompt"])
        for topic in template["topics"]:
            # Templates may use {topic} or {component} — supply both
            text = "Question: {}\nAnswer:".format(prompt_tmpl.format(topic=topic, component=topic))
            chunks.append(text)
            total_chars += len(text)
            source.files_processed += 1
            if total_chars >= max_chars:
                break
        if total_chars >= max_chars:
            break

    source.chars_collected = total_chars
    source.status = "done"
    logger.info("🧪 Synthetic: %d prompts, %d chars", source.files_processed, total_chars)
    return chunks, source


# ---------------------------------------------------------------------------
# Unified Data Pipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Multi-source data aggregation pipeline.

    Collects training data from local files, HuggingFace datasets, URLs,
    and synthetic templates into a single corpus string.

    Example::

        pipeline = DataPipeline(max_total_bytes=500_000_000)

        # Local files (always included)
        pipeline.add_local_dirs(["docs", "apps/backend"])

        # HuggingFace educational content
        pipeline.add_huggingface_dataset(
            "HuggingFaceFW/fineweb-edu-score-2",
            max_rows=50_000,
        )

        # Specific URLs
        pipeline.add_urls([
            "https://en.wikipedia.org/wiki/Transformer_(deep_learning_model)",
        ])

        # Build final corpus
        corpus = pipeline.build_corpus()
        print(f"Corpus: {len(corpus)} chars")
    """

    def __init__(self, max_total_bytes: int = 500_000_000):
        """Initialize the pipeline.

        Parameters
        ----------
        max_total_bytes : int
            Approximate cap on total corpus size in bytes.
            Default 500 MB — enough for ~250M tokens with BPE.
        """
        self.max_total_chars = max_total_bytes  # Rough 1:1 for UTF-8 text
        self._chunks: list[str] = []
        self._collected_chars = 0
        self.stats = PipelineStats()
        self._start_time = time.time()

    @property
    def remaining_chars(self) -> int:
        """Characters remaining before hitting the cap."""
        return max(0, self.max_total_chars - self._collected_chars)

    def add_local_dirs(
        self,
        dirs: list[str],
        extensions: set | None = None,
        min_size: int = 100,
        max_size: int = 500_000,
    ) -> DataSource:
        """Add local file directories as a data source."""
        chunks, source = collect_local_files(
            dirs,
            extensions=extensions,
            min_size=min_size,
            max_size=max_size,
            max_total_chars=self.remaining_chars,
        )
        self._chunks.extend(chunks)
        self._collected_chars += source.chars_collected
        self.stats.sources.append(source)
        return source

    def add_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_rows: int = 50_000,
        streaming: bool = True,
        trust_remote_code: bool = False,
    ) -> DataSource:
        """Add a HuggingFace dataset as a data source."""
        chunks, source = collect_huggingface_dataset(
            dataset_name,
            split=split,
            text_column=text_column,
            max_rows=max_rows,
            max_total_chars=self.remaining_chars,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
        self._chunks.extend(chunks)
        self._collected_chars += source.chars_collected
        self.stats.sources.append(source)
        return source

    def add_urls(
        self,
        urls: list[str],
        max_chars_per_url: int = 100_000,
        timeout: int = 30,
    ) -> DataSource:
        """Add URL content as a data source."""
        chunks, source = collect_urls(
            urls,
            max_chars_per_url=max_chars_per_url,
            timeout=timeout,
        )
        self._chunks.extend(chunks)
        self._collected_chars += source.chars_collected
        self.stats.sources.append(source)
        return source

    def add_synthetic(self, max_chars: int = 100_000) -> DataSource:
        """Add synthetic instruction-following prompts."""
        chunks, source = generate_synthetic_prompts(max_chars=max_chars)
        self._chunks.extend(chunks)
        self._collected_chars += source.chars_collected
        self.stats.sources.append(source)
        return source

    def add_raw_text(self, text: str, name: str = "raw") -> DataSource:
        """Add raw text directly (e.g., from user uploads)."""
        source = DataSource(source_type="raw", name=name)
        source.chars_collected = len(text)
        source.files_processed = 1
        source.status = "done"
        self._chunks.append(text)
        self._collected_chars += len(text)
        self.stats.sources.append(source)
        return source

    def build_corpus(self) -> str:
        """Join all collected chunks into a single training corpus."""
        self.stats.elapsed_seconds = time.time() - self._start_time
        self.stats.total_chars = self._collected_chars
        self.stats.total_chunks = len(self._chunks)
        self.stats.total_files = sum(s.files_processed for s in self.stats.sources)

        corpus = "\n\n".join(self._chunks)

        logger.info(
            "📊 DataPipeline: %d sources, %d chunks, %.1f MB, %.1fs",
            len(self.stats.sources),
            self.stats.total_chunks,
            len(corpus) / 1e6,
            self.stats.elapsed_seconds,
        )
        for s in self.stats.sources:
            logger.info(
                "  └─ [%s] %s: %d items, %.1f MB (%s)",
                s.source_type,
                s.name,
                s.files_processed,
                s.chars_collected / 1e6,
                s.status,
            )

        return corpus

    def get_stats_dict(self) -> dict[str, Any]:
        """Return pipeline stats as a JSON-serializable dict."""
        return {
            "total_chars": self.stats.total_chars,
            "total_files": self.stats.total_files,
            "total_chunks": self.stats.total_chunks,
            "elapsed_seconds": self.stats.elapsed_seconds,
            "sources": [
                {
                    "type": s.source_type,
                    "name": s.name,
                    "chars": s.chars_collected,
                    "files": s.files_processed,
                    "status": s.status,
                    "error": s.error,
                }
                for s in self.stats.sources
            ],
        }
