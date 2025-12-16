"""
Base classes for LLM providers.

Defines the abstract interface that all LLM providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (OpenRouter, Claude Code, etc.) must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Current model name."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        ...

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Convenience method for simple system + user prompt interactions.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional options.

        Returns:
            LLMResponse with the generated content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
