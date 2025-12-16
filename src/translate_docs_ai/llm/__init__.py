"""
LLM provider abstraction layer.

Supports multiple LLM backends:
- OpenRouter (default): Pay-per-token via OpenRouter API
- Claude Code: Use your existing Claude Pro/Max subscription via Claude Agent SDK
"""

from translate_docs_ai.llm.base import LLMProvider, LLMResponse
from translate_docs_ai.llm.factory import LLMProviderType, create_llm_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMProviderType",
    "create_llm_provider",
]
