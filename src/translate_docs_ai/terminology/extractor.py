"""
Terminology extraction using DuckDB FTS and frequency analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from translate_docs_ai.database import Database, Term


@dataclass
class ExtractedTerm:
    """A term extracted from document text."""

    term: str
    frequency: int
    contexts: list[str]
    document_frequency: int = 1  # Number of documents containing this term


class TerminologyExtractor:
    """
    Extract terminology from document pages using frequency analysis.

    Uses DuckDB for efficient text processing and storage.
    """

    # Common stop words to exclude
    STOP_WORDS = {
        # English
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        # Arabic common words
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "التي",
        "الذي",
        "كان",
        "كانت",
        "يكون",
        "أن",
        "إن",
        "لا",
        "ما",
        "هو",
        "هي",
        "هم",
        "نحن",
        "أنا",
        "أنت",
        "كل",
        "بعض",
        "أي",
    }

    # Minimum word length
    MIN_WORD_LENGTH = 3

    # Maximum words in a term (for multi-word terms)
    MAX_TERM_WORDS = 4

    def __init__(
        self,
        db: Database,
        min_frequency: int = 3,
        max_terms: int = 500,
    ):
        """
        Initialize terminology extractor.

        Args:
            db: Database instance.
            min_frequency: Minimum frequency for a term to be extracted.
            max_terms: Maximum number of terms to extract per document.
        """
        self.db = db
        self.min_frequency = min_frequency
        self.max_terms = max_terms

    def extract_from_document(self, document_id: int) -> list[Term]:
        """
        Extract terminology from all pages of a document.

        Args:
            document_id: Document ID in database.

        Returns:
            List of Term objects that were extracted and saved.
        """
        # Get all pages for the document
        pages = self.db.get_document_pages(document_id)

        if not pages:
            return []

        # Combine all page content
        all_text = "\n".join(p.original_content or "" for p in pages)

        # Extract terms
        extracted = self._extract_terms(all_text)

        # Save terms to database
        terms: list[Term] = []
        for ext_term in extracted[: self.max_terms]:
            term = Term(
                document_id=document_id,
                term=ext_term.term,
                frequency=ext_term.frequency,
                context=ext_term.contexts[0] if ext_term.contexts else None,
            )
            term_id = self.db.add_term(term)
            term.id = term_id
            terms.append(term)

        # Log extraction
        self.db.log(
            level="INFO",
            stage="terminology_extract",
            message=f"Extracted {len(terms)} terms from document",
            document_id=document_id,
            context={"total_terms": len(extracted), "saved_terms": len(terms)},
        )

        return terms

    def _extract_terms(self, text: str) -> list[ExtractedTerm]:
        """Extract terms from text using frequency analysis."""
        if not text:
            return []

        # Normalize text
        text = self._normalize_text(text)

        # Extract single words
        words = self._tokenize(text)
        word_freq = Counter(words)

        # Extract multi-word terms (n-grams)
        ngram_freq: Counter[str] = Counter()
        for n in range(2, self.MAX_TERM_WORDS + 1):
            ngrams = self._extract_ngrams(words, n)
            ngram_freq.update(ngrams)

        # Combine and filter
        all_terms: dict[str, ExtractedTerm] = {}

        # Add single words
        for word, freq in word_freq.items():
            if freq >= self.min_frequency and self._is_valid_term(word):
                contexts = self._find_contexts(text, word)
                all_terms[word] = ExtractedTerm(
                    term=word,
                    frequency=freq,
                    contexts=contexts[:3],  # Keep up to 3 contexts
                )

        # Add n-grams (prefer longer terms if they're frequent enough)
        for ngram, freq in ngram_freq.items():
            if freq >= self.min_frequency and self._is_valid_ngram(ngram):
                # Only add if it's more informative than its parts
                if self._is_informative_ngram(ngram, word_freq):
                    contexts = self._find_contexts(text, ngram)
                    all_terms[ngram] = ExtractedTerm(
                        term=ngram,
                        frequency=freq,
                        contexts=contexts[:3],
                    )

        # Sort by frequency
        sorted_terms = sorted(all_terms.values(), key=lambda t: t.frequency, reverse=True)

        return sorted_terms

    def _normalize_text(self, text: str) -> str:
        """Normalize text for term extraction."""
        # Convert to lowercase
        text = text.lower()

        # Remove markdown formatting
        text = re.sub(r"[#*_`\[\](){}|]", " ", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        # Split on whitespace and punctuation
        words = re.findall(r"\b[\w\u0600-\u06FF]+\b", text, re.UNICODE)

        # Filter
        filtered = []
        for word in words:
            if (
                len(word) >= self.MIN_WORD_LENGTH
                and word not in self.STOP_WORDS
                and not word.isdigit()
            ):
                filtered.append(word)

        return filtered

    def _extract_ngrams(self, words: list[str], n: int) -> list[str]:
        """Extract n-grams from word list."""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram_words = words[i : i + n]
            # Skip if any word is a stop word
            if not any(w in self.STOP_WORDS for w in ngram_words):
                ngrams.append(" ".join(ngram_words))
        return ngrams

    def _is_valid_term(self, term: str) -> bool:
        """Check if a term is valid."""
        if len(term) < self.MIN_WORD_LENGTH:
            return False
        if term in self.STOP_WORDS:
            return False
        if term.isdigit():
            return False
        # Check for too many special characters
        alpha_ratio = sum(c.isalpha() for c in term) / len(term)
        return alpha_ratio >= 0.7

    def _is_valid_ngram(self, ngram: str) -> bool:
        """Check if an n-gram is valid."""
        words = ngram.split()
        return all(self._is_valid_term(w) for w in words)

    def _is_informative_ngram(
        self,
        ngram: str,
        word_freq: Counter[str],  # noqa: ARG002
    ) -> bool:
        """Check if n-gram is more informative than its parts."""
        words = ngram.split()
        if len(words) < 2:
            return True

        # TODO: Implement smarter n-gram filtering based on word_freq
        # The n-gram should appear in a significant portion of
        # occurrences where its words co-occur
        return True  # Simplified for now

    def _find_contexts(self, text: str, term: str, window: int = 50) -> list[str]:
        """Find context snippets around term occurrences."""
        contexts = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        for match in pattern.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end].strip()

            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."

            contexts.append(context)

            if len(contexts) >= 5:
                break

        return contexts

    def get_document_glossary(
        self, document_id: int, include_untranslated: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get glossary for a document.

        Args:
            document_id: Document ID.
            include_untranslated: Include terms without translations.

        Returns:
            List of term dictionaries with translations.
        """
        terms = self.db.get_document_terms(document_id)

        glossary = []
        for term in terms:
            if not include_untranslated:
                if not (term.translation_ar or term.translation_en or term.translation_fr):
                    continue

            glossary.append(
                {
                    "term": term.term,
                    "frequency": term.frequency,
                    "ar": term.translation_ar,
                    "en": term.translation_en,
                    "fr": term.translation_fr,
                    "approved": term.approved,
                }
            )

        return glossary
