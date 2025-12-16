"""
OpenRouter LLM provider.

Uses the OpenAI-compatible API via OpenRouter to access multiple LLM providers.
This is the traditional pay-per-token approach.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI

from translate_docs_ai.llm.base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM provider.

    Uses OpenRouter's unified API to access Claude, GPT, Gemini, DeepSeek, etc.
    Requires an OpenRouter API key and charges per token.
    """

    # Model aliases for convenience
    MODELS = {
        "default": "anthropic/claude-sonnet-4.5",
        "fast": "anthropic/claude-3-haiku",
        "quality": "anthropic/claude-3-opus",
        "deepseek": "deepseek/deepseek-chat",
        "gemini": "google/gemini-pro-1.5",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "default",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key.
            model: Model key (from MODELS) or full model name.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts on failure.
        """
        self._api_key = api_key
        self._model_name = self.MODELS.get(model, model)
        self._max_retries = max_retries

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "openrouter"

    @property
    def model(self) -> str:
        """Current model name."""
        return self._model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion via OpenRouter.

        Args:
            messages: List of message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            **kwargs: Additional options passed to the API.

        Returns:
            LLMResponse with content and usage stats.
        """
        start_time = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                content = response.choices[0].message.content or ""
                usage = response.usage

                return LLMResponse(
                    content=content.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model=self._model_name,
                    latency_ms=latency_ms,
                    metadata={
                        "provider": "openrouter",
                        "finish_reason": response.choices[0].finish_reason,
                        "attempt": attempt + 1,
                    },
                )

            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        raise last_error or Exception("OpenRouter request failed after retries")
