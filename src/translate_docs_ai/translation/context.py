"""
Context building for translation.

Provides context from previous pages, terminology glossary, and document metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from translate_docs_ai.database import Database, Document, Page


@dataclass
class TranslationContext:
    """Context provided to the translator for a single page."""

    # Document info
    document_title: str
    document_type: str
    total_pages: int
    current_page: int

    # Previous content for continuity
    previous_summary: str | None = None
    previous_page_ending: str | None = None

    # Terminology glossary
    glossary: list[dict[str, str]] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self, target_lang: str = "ar") -> str:
        """Format context for inclusion in translation prompt."""
        parts = []

        # Document context
        parts.append(f"Document: {self.document_title}")
        parts.append(f"Page {self.current_page + 1} of {self.total_pages}")

        # Previous context
        if self.previous_summary:
            parts.append(f"\nPrevious content summary:\n{self.previous_summary}")

        if self.previous_page_ending:
            parts.append(f"\nPrevious page ending:\n...{self.previous_page_ending}")

        # Glossary
        if self.glossary:
            parts.append("\nTerminology glossary (use these translations):")
            for entry in self.glossary[:30]:  # Limit to avoid token overflow
                term = entry.get("term", "")
                translation = entry.get(target_lang, "")
                if term and translation:
                    parts.append(f"  - {term} → {translation}")

        return "\n".join(parts)


class ContextBuilder:
    """
    Builds translation context from database.

    Gathers previous page content, terminology, and document metadata
    to provide the translator with necessary context.
    """

    def __init__(
        self,
        db: Database,
        context_pages: int = 2,
        max_glossary_terms: int = 50,
        summary_length: int = 500,
    ):
        """
        Initialize context builder.

        Args:
            db: Database instance.
            context_pages: Number of previous pages to consider.
            max_glossary_terms: Maximum glossary entries to include.
            summary_length: Maximum length for previous page summary.
        """
        self.db = db
        self.context_pages = context_pages
        self.max_glossary_terms = max_glossary_terms
        self.summary_length = summary_length

    def build_context(
        self,
        document: Document,
        page: Page,
        target_lang: str = "ar",
    ) -> TranslationContext:
        """
        Build translation context for a page.

        Args:
            document: Document being translated.
            page: Current page to translate.
            target_lang: Target language code.

        Returns:
            TranslationContext with all relevant context.
        """
        # Get previous pages for context
        previous_summary, previous_ending = self._get_previous_context(
            document.id, page.page_number
        )

        # Get glossary
        glossary = self._get_glossary(document.id, target_lang)

        return TranslationContext(
            document_title=document.file_name,
            document_type=document.file_type,
            total_pages=document.total_pages,
            current_page=page.page_number,
            previous_summary=previous_summary,
            previous_page_ending=previous_ending,
            glossary=glossary,
            metadata={
                "document_id": document.id,
                "page_id": page.id,
                "file_path": document.full_path,
            },
        )

    def _get_previous_context(
        self,
        document_id: int | None,
        current_page: int,
    ) -> tuple[str | None, str | None]:
        """Get context from previous pages."""
        if document_id is None or current_page == 0:
            return None, None

        # Get previous translated pages
        pages = self.db.get_document_pages(document_id)
        previous_pages = [p for p in pages if p.page_number < current_page and p.translated_content]

        if not previous_pages:
            return None, None

        # Sort by page number (most recent first for ending)
        previous_pages.sort(key=lambda p: p.page_number, reverse=True)

        # Get ending of most recent page
        most_recent = previous_pages[0]
        previous_ending = None
        if most_recent.translated_content:
            # Get last ~200 chars
            content = most_recent.translated_content.strip()
            if len(content) > 200:
                previous_ending = content[-200:]
            else:
                previous_ending = content

        # Build summary from recent pages
        summary_parts = []
        chars_remaining = self.summary_length

        for page in previous_pages[: self.context_pages]:
            if page.translated_content and chars_remaining > 0:
                # Get first part of each page
                content = page.translated_content.strip()
                snippet = content[: min(chars_remaining, 250)]
                if len(content) > 250:
                    snippet += "..."
                summary_parts.append(f"[Page {page.page_number + 1}]: {snippet}")
                chars_remaining -= len(snippet)

        previous_summary = "\n".join(summary_parts) if summary_parts else None

        return previous_summary, previous_ending

    def _get_glossary(
        self,
        document_id: int | None,
        target_lang: str,
    ) -> list[dict[str, str]]:
        """Get terminology glossary for document."""
        if document_id is None:
            return []

        terms = self.db.get_document_terms(document_id)

        # Filter to terms with translations and sort by frequency
        glossary = []
        for term in terms:
            translation = getattr(term, f"translation_{target_lang}", None)
            if translation:
                glossary.append(
                    {
                        "term": term.term,
                        target_lang: translation,
                        "frequency": term.frequency,
                        "approved": term.approved,
                    }
                )

        # Sort by approved first, then by frequency
        glossary.sort(key=lambda x: (-x.get("approved", False), -x.get("frequency", 0)))

        return glossary[: self.max_glossary_terms]

    def update_page_summary(
        self,
        page: Page,
        summary: str,
    ) -> None:
        """
        Store a summary of a translated page for future context.

        Args:
            page: Page that was translated.
            summary: Summary of the page content.
        """
        if page.id:
            self.db.conn.execute(
                """
                UPDATE pages
                SET translation_notes = ?
                WHERE id = ?
                """,
                [summary, page.id],
            )


def build_translation_prompt(
    source_content: str,
    context: TranslationContext,
    source_lang: str = "en",
    target_lang: str = "ar",
    preserve_formatting: bool = True,
) -> str:
    """
    Build a complete translation prompt.

    Args:
        source_content: Content to translate.
        context: Translation context.
        source_lang: Source language code.
        target_lang: Target language code.
        preserve_formatting: Whether to preserve markdown formatting.

    Returns:
        Complete prompt string for the LLM.
    """
    lang_names = {
        "en": "English",
        "ar": "Arabic",
        "fr": "French",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    prompt_parts = [
        f"Translate the following {source_name} text to {target_name}.",
    ]

    if preserve_formatting:
        prompt_parts.append(
            "Preserve all markdown formatting, including headers, lists, "
            "code blocks, and links. Keep technical terms in their original "
            "language if no translation is provided in the glossary."
        )

    # Add context
    context_str = context.to_prompt_context(target_lang)
    if context_str:
        prompt_parts.append(f"\n## Context\n{context_str}")

    # Add source content
    prompt_parts.append(f"\n## Source Text\n{source_content}")

    # Add instruction
    prompt_parts.append(f"\n## Translation\nProvide only the {target_name} translation below:")

    return "\n".join(prompt_parts)
