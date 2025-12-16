"""
LLM provider factory.

Creates the appropriate LLM provider based on configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

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
        LLMProviderType.OPENROUTER: "anthropic/claude-sonnet-4.5",
        LLMProviderType.CLAUDE_CODE: "sonnet",
    }
    return defaults.get(provider_type, "default")


def create_llm_provider_with_fallback(
    primary_provider: LLMProviderType | str,
    fallback_provider: LLMProviderType | str,
    *,
    primary_api_key: str | None = None,
    fallback_api_key: str | None = None,
    primary_model: str = "default",
    fallback_model: str = "default",
    log_callback: Any = None,
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM provider with automatic fallback support.

    Args:
        primary_provider: Primary provider type to try first.
        fallback_provider: Fallback provider to use if primary fails.
        primary_api_key: API key for primary provider (if needed).
        fallback_api_key: API key for fallback provider (if needed).
        primary_model: Model for primary provider.
        fallback_model: Model for fallback provider.
        log_callback: Optional callback for logging provider switches.
        **kwargs: Additional options passed to both providers.

    Returns:
        FallbackLLMProvider wrapping both providers.

    Example:
        # Claude Code primary with OpenRouter fallback
        provider = create_llm_provider_with_fallback(
            primary_provider="claude-code",
            fallback_provider="openrouter",
            fallback_api_key="sk-or-...",
            primary_model="sonnet",
            fallback_model="anthropic/claude-sonnet-4.5"
        )
    """
    from translate_docs_ai.llm.fallback import FallbackLLMProvider

    # Create primary provider
    primary = create_llm_provider(
        primary_provider,
        api_key=primary_api_key,
        model=primary_model,
        **kwargs,
    )

    # Create fallback provider
    fallback = create_llm_provider(
        fallback_provider,
        api_key=fallback_api_key,
        model=fallback_model,
        **kwargs,
    )

    # Wrap in fallback provider
    return FallbackLLMProvider(
        primary=primary,
        fallback=fallback,
        log_callback=log_callback,
    )
