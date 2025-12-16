"""
Fallback LLM provider wrapper.

Automatically retries failed requests with a fallback provider.
"""

from __future__ import annotations

import time
from typing import Any

from translate_docs_ai.llm.base import LLMProvider, LLMResponse


class FallbackLLMProvider(LLMProvider):
    """
    LLM provider wrapper with automatic fallback.

    Attempts requests with the primary provider first. If the primary fails,
    automatically retries with the fallback provider.

    This is useful for handling rate limits, service outages, or other
    transient errors by seamlessly switching to an alternative provider.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        log_callback: Any = None,
    ):
        """
        Initialize fallback provider wrapper.

        Args:
            primary: Primary LLM provider to try first.
            fallback: Fallback LLM provider to use if primary fails.
            log_callback: Optional callback for logging provider switches.
                         Should accept (level: str, message: str, context: dict).
        """
        self._primary = primary
        self._fallback = fallback
        self._log_callback = log_callback

        # Track usage statistics
        self._primary_requests = 0
        self._fallback_requests = 0
        self._primary_failures = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return f"{self._primary.name}+{self._fallback.name}"

    @property
    def model(self) -> str:
        """Current model name (from primary provider)."""
        return self._primary.model

    @property
    def primary_provider(self) -> LLMProvider:
        """Get the primary provider."""
        return self._primary

    @property
    def fallback_provider(self) -> LLMProvider:
        """Get the fallback provider."""
        return self._fallback

    def get_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with request counts and failure rates.
        """
        total_requests = self._primary_requests + self._fallback_requests
        return {
            "total_requests": total_requests,
            "primary_requests": self._primary_requests,
            "fallback_requests": self._fallback_requests,
            "primary_failures": self._primary_failures,
            "fallback_rate": (
                self._fallback_requests / total_requests if total_requests > 0 else 0.0
            ),
        }

    def _log(self, level: str, message: str, context: dict[str, Any] | None = None) -> None:
        """Log a message via callback if available."""
        if self._log_callback:
            self._log_callback(level, message, context or {})

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with automatic fallback.

        Tries the primary provider first. If it fails, automatically retries
        with the fallback provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            Exception: If both primary and fallback providers fail.
        """
        start_time = time.perf_counter()
        primary_error: Exception | None = None

        # Try primary provider first
        try:
            self._log(
                "DEBUG",
                f"Attempting request with primary provider: {self._primary.name}",
                {"model": self._primary.model},
            )

            response = await self._primary.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            self._primary_requests += 1

            # Add metadata about provider used
            response.metadata = response.metadata or {}
            response.metadata["provider_used"] = "primary"
            response.metadata["primary_provider"] = self._primary.name
            response.metadata["fallback_available"] = True

            return response

        except Exception as e:
            primary_error = e
            self._primary_failures += 1

            # Log the failure and fallback attempt
            self._log(
                "WARNING",
                f"Primary provider ({self._primary.name}) failed, switching to fallback ({self._fallback.name})",
                {
                    "primary_provider": self._primary.name,
                    "primary_model": self._primary.model,
                    "fallback_provider": self._fallback.name,
                    "fallback_model": self._fallback.model,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

        # Try fallback provider
        try:
            self._log(
                "INFO",
                f"Retrying with fallback provider: {self._fallback.name}",
                {"model": self._fallback.model},
            )

            response = await self._fallback.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            self._fallback_requests += 1

            # Add metadata about provider used and primary failure
            response.metadata = response.metadata or {}
            response.metadata["provider_used"] = "fallback"
            response.metadata["primary_provider"] = self._primary.name
            response.metadata["fallback_provider"] = self._fallback.name
            response.metadata["primary_error"] = str(primary_error)
            response.metadata["primary_error_type"] = type(primary_error).__name__

            # Log successful fallback
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._log(
                "INFO",
                f"✓ Fallback provider {self._fallback.name} ({self._fallback.model}) succeeded after {elapsed_ms:.0f}ms",
                {
                    "fallback_provider": self._fallback.name,
                    "fallback_model": self._fallback.model,
                    "latency_ms": elapsed_ms,
                },
            )

            return response

        except Exception as fallback_error:
            # Both providers failed
            self._log(
                "ERROR",
                "Both primary and fallback providers failed",
                {
                    "primary_provider": self._primary.name,
                    "primary_error": str(primary_error),
                    "fallback_provider": self._fallback.name,
                    "fallback_error": str(fallback_error),
                },
            )

            # Raise the fallback error (more recent)
            raise fallback_error from primary_error
