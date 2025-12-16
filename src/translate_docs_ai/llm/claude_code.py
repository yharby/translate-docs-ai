"""
Claude Code LLM provider.

Uses the Claude Agent SDK to leverage your existing Claude Pro/Max subscription.
No additional API costs - uses your subscription's included usage.
"""

from __future__ import annotations

import time
from typing import Any

from translate_docs_ai.llm.base import LLMProvider, LLMResponse


class ClaudeCodeProvider(LLMProvider):
    """
    Claude Code LLM provider using Claude Agent SDK.

    This provider uses your existing Claude Pro/Max subscription via the
    Claude Agent SDK. No separate API key is required - it uses your
    authenticated Claude Code CLI session.

    Benefits:
    - No per-token charges (uses your subscription)
    - Access to Claude's full context window (up to 1M tokens)
    - Built-in tool support and agent capabilities

    Requirements:
    - Claude Code CLI must be installed and authenticated
    - Run `claude login` if not already authenticated
    """

    # Model aliases mapping to Claude models
    MODELS = {
        "default": "sonnet",
        "fast": "haiku",
        "quality": "opus",
        "sonnet": "sonnet",
        "opus": "opus",
        "haiku": "haiku",
    }

    def __init__(
        self,
        model: str = "default",
        max_turns: int = 1,
        timeout: float = 120.0,
    ):
        """
        Initialize Claude Code provider.

        Args:
            model: Model key (sonnet, opus, haiku) or alias (default, fast, quality).
            max_turns: Maximum conversation turns (1 for simple completions).
            timeout: Request timeout in seconds.
        """
        self._model_name = self.MODELS.get(model, model)
        self._max_turns = max_turns
        self._timeout = timeout

        # Lazy import to avoid import errors if SDK not installed
        self._sdk_available: bool | None = None

    def _check_sdk(self) -> bool:
        """Check if Claude Agent SDK is available."""
        if self._sdk_available is None:
            try:
                from claude_agent_sdk import query  # noqa: F401

                self._sdk_available = True
            except ImportError:
                self._sdk_available = False
        return self._sdk_available

    @property
    def name(self) -> str:
        """Provider name."""
        return "claude-code"

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
        Generate a completion via Claude Agent SDK.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (note: may not be fully supported).
            max_tokens: Maximum output tokens.
            **kwargs: Additional options.

        Returns:
            LLMResponse with content and metadata.
        """
        if not self._check_sdk():
            raise ImportError(
                "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
            )

        from claude_agent_sdk import ClaudeAgentOptions, query

        start_time = time.perf_counter()

        # Build prompt from messages
        # Claude Agent SDK uses a single prompt, so we combine system + user messages
        prompt_parts = []
        system_prompt = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                # Include assistant messages as context
                prompt_parts.append(f"[Previous response]: {content}")

        full_prompt = "\n\n".join(prompt_parts)

        # Configure options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=self._max_turns,
            # Disable tools for pure text generation
            allowed_tools=[],
        )

        # Collect response from async iterator
        content_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            async for message in query(prompt=full_prompt, options=options):
                # Handle different message types from the SDK
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            content_parts.append(block.text)

                # Try to extract token usage from result messages
                if hasattr(message, "usage"):
                    usage = message.usage
                    if hasattr(usage, "input_tokens"):
                        input_tokens = usage.input_tokens
                    if hasattr(usage, "output_tokens"):
                        output_tokens = usage.output_tokens

        except Exception as e:
            # Check for common errors
            error_str = str(e).lower()
            if "not authenticated" in error_str or "login" in error_str:
                raise RuntimeError(
                    "Claude Code not authenticated. Run 'claude login' to authenticate."
                ) from e
            if "not found" in error_str or "cli" in error_str:
                raise RuntimeError(
                    "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
                ) from e
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        content = "".join(content_parts).strip()

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=f"claude-code/{self._model_name}",
            latency_ms=latency_ms,
            metadata={
                "provider": "claude-code",
                "subscription_based": True,
            },
        )
