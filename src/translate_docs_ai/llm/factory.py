"""
LLM provider factory.

Creates the appropriate LLM provider based on configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from translate_docs_ai.llm.base import LLMProvider

if TYPE_CHECKING:
    pass


class LLMProviderType(str, Enum):
    """Available LLM provider types."""

    OPENROUTER = "openrouter"
    CLAUDE_CODE = "claude-code"


def create_llm_provider(
    provider_type: LLMProviderType | str,
    *,
    api_key: str | None = None,
    model: str = "default",
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider_type: Type of provider to create (openrouter or claude-code).
        api_key: API key (required for openrouter, ignored for claude-code).
        model: Model name or alias.
        **kwargs: Additional provider-specific options.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider_type is invalid or required config is missing.

    Examples:
        # OpenRouter (pay-per-token)
        provider = create_llm_provider(
            "openrouter",
            api_key="sk-or-...",
            model="anthropic/claude-3.5-sonnet"
        )

        # Claude Code (uses your subscription)
        provider = create_llm_provider(
            "claude-code",
            model="sonnet"  # or opus, haiku
        )
    """
    # Normalize provider type
    if isinstance(provider_type, str):
        provider_type = provider_type.lower().replace("_", "-")
        try:
            provider_type = LLMProviderType(provider_type)
        except ValueError:
            valid = [p.value for p in LLMProviderType]
            raise ValueError(
                f"Invalid provider type: {provider_type}. Valid options: {valid}"
            ) from None

    if provider_type == LLMProviderType.OPENROUTER:
        if not api_key:
            raise ValueError("OpenRouter provider requires an API key")

        from translate_docs_ai.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(
            api_key=api_key,
            model=model,
            **kwargs,
        )

    elif provider_type == LLMProviderType.CLAUDE_CODE:
        from translate_docs_ai.llm.claude_code import ClaudeCodeProvider

        return ClaudeCodeProvider(
            model=model,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def get_default_model_for_provider(provider_type: LLMProviderType | str) -> str:
    """
    Get the default model for a provider type.

    Args:
        provider_type: Provider type.

    Returns:
        Default model name for that provider.
    """
    if isinstance(provider_type, str):
        provider_type = LLMProviderType(provider_type.lower().replace("_", "-"))

    defaults = {
        LLMProviderType.OPENROUTER: "anthropic/claude-3.5-sonnet",
        LLMProviderType.CLAUDE_CODE: "sonnet",
    }
    return defaults.get(provider_type, "default")
