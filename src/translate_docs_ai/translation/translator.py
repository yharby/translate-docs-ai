"""
Page translator using OpenRouter API.

Provides LLM-based translation with context awareness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from translate_docs_ai.database import Database, Page
from translate_docs_ai.translation.context import (
    ContextBuilder,
    TranslationContext,
    build_translation_prompt,
)


@dataclass
class TranslationResult:
    """Result of a page translation."""

    translated_content: str
    source_tokens: int
    target_tokens: int
    model_used: str
    latency_ms: float
    metadata: dict[str, Any] | None = None


class PageTranslator:
    """
    Translates document pages using OpenRouter API.

    Supports multiple LLM providers through OpenRouter's unified API.
    """

    # Recommended models for translation
    MODELS = {
        "default": "anthropic/claude-3.5-sonnet",
        "fast": "anthropic/claude-3-haiku",
        "quality": "anthropic/claude-3-opus",
        "deepseek": "deepseek/deepseek-chat",
        "gemini": "google/gemini-pro-1.5",
    }

    def __init__(
        self,
        api_key: str,
        db: Database,
        model: str = "default",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize page translator.

        Args:
            api_key: OpenRouter API key.
            db: Database instance.
            model: Model key or full model name.
            base_url: API base URL.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self.db = db
        self._model = self.MODELS.get(model, model)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        self._context_builder = ContextBuilder(db)

    @property
    def model(self) -> str:
        """Get current model name."""
        return self._model

    async def translate_page(
        self,
        page: Page,
        context: TranslationContext,
        source_lang: str = "en",
        target_lang: str = "ar",
        **kwargs: Any,
    ) -> TranslationResult:
        """
        Translate a single page.

        Args:
            page: Page to translate.
            context: Translation context.
            source_lang: Source language code.
            target_lang: Target language code.
            **kwargs: Additional options.

        Returns:
            TranslationResult with translated content.
        """
        import time

        if not page.original_content:
            return TranslationResult(
                translated_content="",
                source_tokens=0,
                target_tokens=0,
                model_used=self._model,
                latency_ms=0,
            )

        # Build prompt
        prompt = build_translation_prompt(
            source_content=page.original_content,
            context=context,
            source_lang=source_lang,
            target_lang=target_lang,
            preserve_formatting=kwargs.get("preserve_formatting", True),
        )

        # Call API with retry
        start_time = time.perf_counter()
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(source_lang, target_lang),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage

                return TranslationResult(
                    translated_content=content.strip(),
                    source_tokens=usage.prompt_tokens if usage else 0,
                    target_tokens=usage.completion_tokens if usage else 0,
                    model_used=self._model,
                    latency_ms=latency_ms,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "attempt": attempt + 1,
                    },
                )

            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

                # Log retry
                self.db.log(
                    level="WARNING",
                    stage="translation",
                    message=f"Translation retry {attempt + 1}: {e}",
                    page_id=page.id,
                    context={"error": str(e)},
                )

        # All retries failed
        raise last_error or Exception("Translation failed")

    def _get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Get system prompt for translation with RTL/LTR table handling."""
        lang_names = {
            "en": "English",
            "ar": "Arabic",
            "fr": "French",
            "he": "Hebrew",
            "fa": "Persian",
        }

        rtl_languages = {"ar", "he", "fa", "ur"}

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        source_is_rtl = source_lang in rtl_languages
        target_is_rtl = target_lang in rtl_languages

        # Determine if direction change is needed
        direction_change = source_is_rtl != target_is_rtl

        # Build table handling instructions
        table_instructions = ""
        if direction_change:
            if source_is_rtl and not target_is_rtl:
                # RTL to LTR (e.g., Arabic to English)
                table_instructions = """
7. **TABLE COLUMN ORDER**: Since you are translating from a right-to-left (RTL) language to a left-to-right (LTR) language:
   - REVERSE the column order in all tables
   - The rightmost column in the source becomes the leftmost column in the translation
   - Example: If source table has columns [Description | Name | No.] (RTL order), output should be [No. | Name | Description] (LTR order)
   - This ensures proper reading flow in the target language"""
            else:
                # LTR to RTL (e.g., English to Arabic)
                table_instructions = """
7. **TABLE COLUMN ORDER**: Since you are translating from a left-to-right (LTR) language to a right-to-left (RTL) language:
   - REVERSE the column order in all tables
   - The leftmost column in the source becomes the rightmost column in the translation
   - Example: If source table has columns [No. | Name | Description] (LTR order), output should be [Description | Name | No.] (RTL order)
   - This ensures proper reading flow in the target language"""

        # Build HTML to Markdown instruction
        html_table_instruction = """
8. **HTML TABLES**: If the source contains HTML tables (<table>, <tr>, <td>, <th>):
   - Convert them to proper Markdown table format
   - Use pipe (|) separators and dashes for headers
   - Example:
     | Header 1 | Header 2 | Header 3 |
     |----------|----------|----------|
     | Cell 1   | Cell 2   | Cell 3   |"""

        return f"""You are an expert translator specializing in technical documentation.
Your task is to translate text from {source_name} to {target_name} while:

1. Preserving all markdown formatting exactly (headers, lists, code blocks, links)
2. Using terminology from the provided glossary consistently
3. Maintaining the technical accuracy and meaning of the original
4. Ensuring the translation flows naturally in {target_name}
5. Keeping code snippets, URLs, and file paths unchanged
6. Translating UI element names and button text appropriately
{table_instructions}
{html_table_instruction}

Provide only the translation without any explanations or notes."""

    async def translate_with_context(
        self,
        page: Page,
        document_id: int,
        source_lang: str = "en",
        target_lang: str = "ar",
        **kwargs: Any,
    ) -> TranslationResult:
        """
        Translate a page with automatic context building.

        Args:
            page: Page to translate.
            document_id: Document ID.
            source_lang: Source language code.
            target_lang: Target language code.
            **kwargs: Additional options.

        Returns:
            TranslationResult with translated content.
        """
        # Get document
        doc = self.db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        # Build context
        context = self._context_builder.build_context(doc, page, target_lang)

        # Translate
        return await self.translate_page(page, context, source_lang, target_lang, **kwargs)

    async def batch_translate(
        self,
        pages: list[Page],
        document_id: int,
        source_lang: str = "en",
        target_lang: str = "ar",
        concurrent: int = 2,
        **kwargs: Any,
    ) -> list[TranslationResult]:
        """
        Translate multiple pages with concurrency control.

        Note: Pages are translated sequentially to maintain context continuity,
        but multiple documents can be processed in parallel.

        Args:
            pages: Pages to translate.
            document_id: Document ID.
            source_lang: Source language code.
            target_lang: Target language code.
            concurrent: Max concurrent translations (for different documents).
            **kwargs: Additional options.

        Returns:
            List of TranslationResults.
        """
        results = []

        # Sort pages by page number
        sorted_pages = sorted(pages, key=lambda p: p.page_number)

        for page in sorted_pages:
            result = await self.translate_with_context(
                page, document_id, source_lang, target_lang, **kwargs
            )
            results.append(result)

            # Save translation to database
            if page.id:
                self.db.update_page(
                    page.id,
                    translated_content=result.translated_content,
                    language=target_lang,
                )

                # Log progress
                self.db.log(
                    level="INFO",
                    stage="translation",
                    message=f"Translated page {page.page_number + 1}",
                    document_id=document_id,
                    page_id=page.id,
                    context={
                        "tokens_in": result.source_tokens,
                        "tokens_out": result.target_tokens,
                        "latency_ms": result.latency_ms,
                    },
                )

        return results

    def estimate_cost(
        self,
        pages: list[Page],
        target_lang: str = "ar",
    ) -> dict[str, Any]:
        """
        Estimate translation cost for pages.

        Args:
            pages: Pages to estimate.
            target_lang: Target language.

        Returns:
            Dictionary with cost estimates.
        """
        # Rough token estimation (4 chars per token for English)
        total_chars = sum(len(p.original_content or "") for p in pages)
        estimated_input_tokens = total_chars // 4

        # Output typically 1.2x input for Arabic
        output_multiplier = 1.2 if target_lang == "ar" else 1.0
        estimated_output_tokens = int(estimated_input_tokens * output_multiplier)

        # Pricing per million tokens (approximate)
        pricing = {
            "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
            "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
            "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
            "google/gemini-pro-1.5": {"input": 1.25, "output": 5.0},
        }

        model_pricing = pricing.get(
            self._model,
            {"input": 3.0, "output": 15.0},  # Default to Claude pricing
        )

        estimated_cost = (estimated_input_tokens / 1_000_000) * model_pricing["input"] + (
            estimated_output_tokens / 1_000_000
        ) * model_pricing["output"]

        return {
            "pages": len(pages),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "model": self._model,
            "pricing": model_pricing,
        }
