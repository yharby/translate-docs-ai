"""
LLM-based terminology extraction using LLM providers.

Uses language models to intelligently identify technical terms,
domain-specific vocabulary, and generate translations.
Supports multiple providers: OpenRouter (pay-per-token) and Claude Code (subscription).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from translate_docs_ai.database import Database, Term
from translate_docs_ai.llm import LLMProviderType, create_llm_provider


@dataclass
class ExtractedTermWithTranslation:
    """A term extracted by LLM with translation."""

    term: str
    translation: str
    category: str  # e.g., "technical", "domain-specific", "acronym", "proper noun"
    context: str  # Example usage from the document
    confidence: float  # LLM's confidence in the extraction


class LLMTerminologyExtractor:
    """
    Extract terminology using LLM for intelligent term identification.

    Supports multiple providers:
    - OpenRouter: Pay-per-token access to Claude, GPT, Gemini, etc.
    - Claude Code: Use your Claude Pro/Max subscription (no extra cost)

    Extracts:
    - Technical terms and jargon
    - Domain-specific vocabulary
    - Acronyms and abbreviations
    - Proper nouns that need special handling
    - Auto-generated translations
    """

    # Model aliases for different providers
    OPENROUTER_MODELS = {
        "default": "anthropic/claude-3.5-sonnet",
        "fast": "anthropic/claude-3-haiku",
        "quality": "anthropic/claude-3-opus",
    }

    CLAUDE_CODE_MODELS = {
        "default": "sonnet",
        "fast": "haiku",
        "quality": "opus",
    }

    def __init__(
        self,
        db: Database,
        *,
        provider: LLMProviderType | str = LLMProviderType.OPENROUTER,
        api_key: str | None = None,
        model: str = "default",
        max_terms_per_chunk: int = 50,
        chunk_size: int = 4000,  # Characters per chunk to avoid context limits
    ):
        """
        Initialize LLM terminology extractor.

        Args:
            db: Database instance.
            provider: LLM provider type ("openrouter" or "claude-code").
            api_key: API key (required for openrouter, ignored for claude-code).
            model: LLM model to use.
            max_terms_per_chunk: Maximum terms to extract per chunk.
            chunk_size: Maximum characters per text chunk.
        """
        self.db = db
        self._max_terms_per_chunk = max_terms_per_chunk
        self._chunk_size = chunk_size

        # Normalize provider type
        if isinstance(provider, str):
            provider = LLMProviderType(provider.lower().replace("_", "-"))

        # Resolve model name based on provider
        if provider == LLMProviderType.CLAUDE_CODE:
            resolved_model = self.CLAUDE_CODE_MODELS.get(model, model)
        else:
            resolved_model = self.OPENROUTER_MODELS.get(model, model)

        # Create LLM provider
        self._provider = create_llm_provider(
            provider,
            api_key=api_key,
            model=resolved_model,
            timeout=120.0,
        )

    async def extract_from_document(
        self,
        document_id: int,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> list[Term]:
        """
        Extract terminology from a document using LLM.

        Args:
            document_id: Document ID in database.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of Term objects extracted and saved.
        """
        # Get all pages for the document
        pages = self.db.get_document_pages(document_id)

        if not pages:
            return []

        # Combine all page content
        all_text = "\n\n".join(p.original_content or "" for p in pages)

        if not all_text.strip():
            return []

        # Split into chunks for processing
        chunks = self._split_into_chunks(all_text)

        # Extract terms from each chunk
        all_extracted: list[ExtractedTermWithTranslation] = []
        seen_terms: set[str] = set()

        for chunk in chunks:
            extracted = await self._extract_from_chunk(chunk, source_lang, target_lang)
            for term in extracted:
                # Deduplicate
                term_lower = term.term.lower()
                if term_lower not in seen_terms:
                    seen_terms.add(term_lower)
                    all_extracted.append(term)

        # Save terms to database
        terms: list[Term] = []
        for ext_term in all_extracted:
            term = Term(
                document_id=document_id,
                term=ext_term.term,
                frequency=1,  # LLM extraction doesn't count frequency
                context=ext_term.context,
            )

            # Set translation based on target language
            if target_lang == "en":
                term.translation_en = ext_term.translation
            elif target_lang == "ar":
                term.translation_ar = ext_term.translation
            elif target_lang == "fr":
                term.translation_fr = ext_term.translation

            term_id = self.db.add_term(term)
            term.id = term_id
            terms.append(term)

        # Log extraction
        self.db.log(
            level="INFO",
            stage="terminology_extract_llm",
            message=f"LLM extracted {len(terms)} terms from document",
            document_id=document_id,
            context={
                "model": self._provider.model,
                "provider": self._provider.name,
                "chunks_processed": len(chunks),
                "terms_extracted": len(terms),
            },
        )

        return terms

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks for processing."""
        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self._chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[: self._chunk_size]]

    async def _extract_from_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> list[ExtractedTermWithTranslation]:
        """Extract terms from a text chunk using LLM."""
        lang_names = {
            "en": "English",
            "ar": "Arabic",
            "fr": "French",
            "he": "Hebrew",
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        system_prompt = f"""You are an expert terminology extractor and translator.
Your task is to identify important terms from {source_name} text that should be consistently translated.

Focus on extracting:
1. **Technical terms**: Domain-specific terminology, scientific terms, technical jargon
2. **Acronyms/Abbreviations**: With their full forms if apparent
3. **Proper nouns**: Organization names, product names, place names that need careful translation
4. **Domain vocabulary**: Industry-specific terms that have standard translations
5. **Compound terms**: Multi-word expressions that function as single concepts

DO NOT extract:
- Common words that have obvious translations
- Generic verbs, adjectives, or articles
- Numbers or dates (unless part of a named entity)

For each term, provide:
- The original term exactly as it appears
- The best {target_name} translation
- A category (technical, acronym, proper_noun, domain_specific, compound)
- A short context snippet showing usage

Respond ONLY with a JSON array. Example format:
[
  {{
    "term": "machine learning",
    "translation": "التعلم الآلي",
    "category": "technical",
    "context": "...using machine learning algorithms...",
    "confidence": 0.95
  }}
]

If no relevant terms are found, return an empty array: []"""

        user_prompt = f"""Extract terminology from this {source_name} text and provide {target_name} translations.
Return ONLY a valid JSON array, no other text.

Text to analyze:
{text}"""

        try:
            response = await self._provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.content or "[]"

            # Parse JSON response
            # Handle markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code block
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else "[]"

            try:
                terms_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
                if json_match:
                    terms_data = json.loads(json_match.group())
                else:
                    terms_data = []

            # Convert to ExtractedTermWithTranslation objects
            extracted = []
            for item in terms_data:
                if isinstance(item, dict) and "term" in item:
                    extracted.append(
                        ExtractedTermWithTranslation(
                            term=item.get("term", ""),
                            translation=item.get("translation", ""),
                            category=item.get("category", "unknown"),
                            context=item.get("context", ""),
                            confidence=item.get("confidence", 0.5),
                        )
                    )

            return extracted[: self._max_terms_per_chunk]

        except Exception as e:
            self.db.log(
                level="WARNING",
                stage="terminology_extract_llm",
                message=f"LLM extraction failed for chunk: {e}",
                context={"error": str(e)},
            )
            return []

    async def translate_terms(
        self,
        document_id: int,
        target_lang: str = "en",
        batch_size: int = 50,
    ) -> int:
        """
        Generate context-aware translations for terms.

        Uses the sentence/context where each term appears to provide
        more accurate translations based on how the term is used.
        Processes terms in batches to handle large term lists.

        Args:
            document_id: Document ID.
            target_lang: Target language for translation.
            batch_size: Number of terms to translate per LLM call.

        Returns:
            Number of terms translated.
        """
        terms = self.db.get_document_terms(document_id)

        # Filter terms that need translation
        trans_column = f"translation_{target_lang}"
        terms_to_translate = [t for t in terms if not getattr(t, trans_column, None)]

        if not terms_to_translate:
            return 0

        lang_names = {
            "en": "English",
            "ar": "Arabic",
            "fr": "French",
        }
        target_name = lang_names.get(target_lang, target_lang)

        total_translated = 0

        # Process terms in batches
        for i in range(0, len(terms_to_translate), batch_size):
            batch = terms_to_translate[i : i + batch_size]

            # Build terms with context for better translation
            terms_with_context = []
            for t in batch:
                if t.context:
                    terms_with_context.append(f'- Term: "{t.term}"\n  Context: "{t.context}"')
                else:
                    terms_with_context.append(f'- Term: "{t.term}"')

            terms_text = "\n".join(terms_with_context)

            system_prompt = f"""You are an expert translator specializing in technical and domain-specific terminology.

Your task is to translate terms from their source language to {target_name}.

IMPORTANT: Use the provided context to understand how each term is used in the document.
The context shows the sentence or phrase where the term appears, which helps determine:
- The correct meaning when a term has multiple translations
- The appropriate register (formal/informal)
- Domain-specific usage (legal, technical, scientific, etc.)

For example:
- "bank" with context "river bank" → translate as riverbank, not financial institution
- "العامة" with context "الهيئة العامة للمساحة" → "General" (as in General Authority)

Return ONLY a valid JSON object mapping each original term to its best translation.
Do not include explanations, just the JSON."""

            user_prompt = f"""Translate these terms to {target_name}, using the context to determine the correct translation:

{terms_text}

Return format:
{{"term1": "translation1", "term2": "translation2"}}"""

            try:
                response = await self._provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=4000,
                )

                content = response.content or "{}"

                # Parse response
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1]) if len(lines) > 2 else "{}"

                try:
                    translations = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON object from response
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        translations = json.loads(json_match.group())
                    else:
                        translations = {}

                # Update terms in database
                for term in batch:
                    if term.term in translations:
                        translation = translations[term.term]
                        self.db.conn.execute(
                            f"UPDATE terminology SET {trans_column} = ? WHERE id = ?",
                            [translation, term.id],
                        )
                        total_translated += 1

            except Exception as e:
                self.db.log(
                    level="WARNING",
                    stage="terminology_translate",
                    message=f"Failed to translate batch {i // batch_size + 1}: {e}",
                    document_id=document_id,
                    context={"batch_start": i, "batch_size": len(batch)},
                )
                # Continue with next batch instead of failing completely

        return total_translated
