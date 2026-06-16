"""
Unified LLM - Multi-provider LLM abstraction layer for Helix Collective

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google Gemini
- Custom providers
"""

__version__ = "0.1.0"
__author__ = "Helix Collective"

from .client import InferenceClient
from .engine import LLMAgentEngine

__all__ = [
    "InferenceClient",
    "LLMAgentEngine",
]
