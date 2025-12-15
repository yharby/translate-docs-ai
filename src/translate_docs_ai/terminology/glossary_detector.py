"""
Glossary/terminology page detection and extraction.

Detects pages that contain glossary/terminology sections in documents
and extracts pre-defined term-translation pairs from tables or lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from translate_docs_ai.database import Database, Page, Term


@dataclass
class GlossaryEntry:
    """A term-definition pair from a glossary page."""

    term: str
    definition: str
    source_page: int


# Multilingual glossary section headers
# Format: (pattern, language_code)
GLOSSARY_HEADERS: list[tuple[str, str]] = [
    # Arabic
    (r"المصطلحات", "ar"),
    (r"قائمة المصطلحات", "ar"),
    (r"مسرد المصطلحات", "ar"),
    (r"المصطلحات والتعريفات", "ar"),
    (r"التعريفات", "ar"),
    (r"المفردات", "ar"),
    (r"الكلمات المفتاحية", "ar"),
    # English
    (r"glossary", "en"),
    (r"terminology", "en"),
    (r"terms and definitions", "en"),
    (r"definitions", "en"),
    (r"key terms", "en"),
    (r"vocabulary", "en"),
    (r"list of terms", "en"),
    (r"abbreviations", "en"),
    (r"acronyms", "en"),
    # French
    (r"glossaire", "fr"),
    (r"terminologie", "fr"),
    (r"termes et définitions", "fr"),
    (r"définitions", "fr"),
    (r"lexique", "fr"),
    (r"vocabulaire", "fr"),
    # German
    (r"glossar", "de"),
    (r"terminologie", "de"),
    (r"begriffe", "de"),
    (r"definitionen", "de"),
    # Spanish
    (r"glosario", "es"),
    (r"terminología", "es"),
    (r"definiciones", "es"),
    # Hebrew
    (r"מילון מונחים", "he"),
    (r"מונחים", "he"),
    (r"הגדרות", "he"),
    # Persian/Farsi
    (r"واژه‌نامه", "fa"),
    (r"اصطلاحات", "fa"),
]


class GlossaryDetector:
    """
    Detects and extracts terminology from glossary pages in documents.

    Supports multiple languages and formats:
    - Tables with term/definition columns
    - Colon-separated lists (term: definition)
    - Dash-separated lists (term - definition)
    - Numbered lists with terms
    """

    def __init__(
        self,
        db: Database,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize glossary detector.

        Args:
            db: Database instance.
            confidence_threshold: Minimum confidence for glossary detection.
        """
        self.db = db
        self.confidence_threshold = confidence_threshold

        # Compile header patterns
        self._header_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.UNICODE), lang)
            for pattern, lang in GLOSSARY_HEADERS
        ]

    def detect_glossary_pages(self, document_id: int) -> list[Page]:
        """
        Detect pages that contain glossary sections.

        Args:
            document_id: Document ID in database.

        Returns:
            List of pages identified as glossary pages.
        """
        pages = self.db.get_document_pages(document_id)
        glossary_pages = []

        for page in pages:
            if not page.original_content:
                continue

            confidence, detected_lang = self._calculate_glossary_confidence(page.original_content)

            if confidence >= self.confidence_threshold:
                glossary_pages.append(page)
                self.db.log(
                    level="INFO",
                    stage="glossary_detection",
                    message=f"Detected glossary page {page.page_number + 1} "
                    f"(confidence: {confidence:.2f}, lang: {detected_lang})",
                    document_id=document_id,
                    context={"page_id": page.id, "page_number": page.page_number + 1},
                )

        return glossary_pages

    def _calculate_glossary_confidence(self, content: str) -> tuple[float, str | None]:
        """
        Calculate confidence that content is a glossary page.

        Returns:
            Tuple of (confidence score 0-1, detected language or None).
        """
        confidence = 0.0
        detected_lang = None

        # Check for glossary headers
        for pattern, lang in self._header_patterns:
            if pattern.search(content):
                confidence += 0.5
                detected_lang = lang
                break

        # Check for table-like structures (pipes for markdown tables)
        table_pattern = r"\|.*\|.*\|"
        if re.search(table_pattern, content):
            confidence += 0.2

        # Check for colon-separated pairs (term: definition)
        colon_pairs = len(re.findall(r"^[^\n:]+:\s*[^\n]+$", content, re.MULTILINE))
        if colon_pairs >= 5:
            confidence += 0.2

        # Check for dash-separated pairs (term - definition)
        dash_pairs = len(re.findall(r"^[^\n\-]+\s*[-–—]\s*[^\n]+$", content, re.MULTILINE))
        if dash_pairs >= 5:
            confidence += 0.2

        # Check for numbered list with consistent structure
        numbered_items = len(re.findall(r"^\d+[.)]\s*[^\n]+[:–-]\s*[^\n]+", content, re.MULTILINE))
        if numbered_items >= 5:
            confidence += 0.15

        # Cap at 1.0
        return min(confidence, 1.0), detected_lang

    def extract_glossary_entries(
        self,
        page: Page,
        source_lang: str | None = None,
    ) -> list[GlossaryEntry]:
        """
        Extract term-definition pairs from a glossary page.

        Args:
            page: Page to extract from.
            source_lang: Source language hint.

        Returns:
            List of GlossaryEntry objects.
        """
        if not page.original_content:
            return []

        content = page.original_content
        entries: list[GlossaryEntry] = []

        # Try different extraction methods
        # 1. Table extraction (highest priority)
        table_entries = self._extract_from_table(content, page.page_number)
        if table_entries:
            entries.extend(table_entries)

        # 2. Colon-separated pairs
        if not entries:
            colon_entries = self._extract_colon_pairs(content, page.page_number)
            entries.extend(colon_entries)

        # 3. Dash-separated pairs
        if not entries:
            dash_entries = self._extract_dash_pairs(content, page.page_number)
            entries.extend(dash_entries)

        return entries

    def _extract_from_table(self, content: str, page_number: int) -> list[GlossaryEntry]:
        """Extract entries from markdown table format."""
        entries = []

        # Match markdown tables
        # Pattern: | term | definition |
        lines = content.split("\n")
        in_table = False
        header_found = False

        for line in lines:
            line = line.strip()

            # Check if line is a table row
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]

                # Skip separator rows (| --- | --- |)
                if cells and all(re.match(r"^[-:]+$", c) for c in cells):
                    header_found = True
                    continue

                # Skip header row (first row before separator)
                if not header_found:
                    in_table = True
                    continue

                # Data row
                if len(cells) >= 2 and in_table:
                    term = cells[0].strip()
                    definition = cells[1].strip()

                    if term and definition and len(term) < 200:
                        entries.append(
                            GlossaryEntry(
                                term=term,
                                definition=definition,
                                source_page=page_number,
                            )
                        )
            else:
                # Reset table state if we hit a non-table line
                if in_table and header_found:
                    in_table = False
                    header_found = False

        return entries

    def _extract_colon_pairs(self, content: str, page_number: int) -> list[GlossaryEntry]:
        """Extract term: definition pairs."""
        entries = []

        # Pattern: term: definition (allowing Arabic and other Unicode)
        pattern = r"^([^\n:]{2,100}):\s*(.{10,})$"
        matches = re.findall(pattern, content, re.MULTILINE | re.UNICODE)

        for term, definition in matches:
            term = term.strip()
            definition = definition.strip()

            # Basic validation
            if (
                term
                and definition
                and len(term) < 150
                and not term.startswith("#")  # Skip markdown headers
            ):
                entries.append(
                    GlossaryEntry(
                        term=term,
                        definition=definition,
                        source_page=page_number,
                    )
                )

        return entries

    def _extract_dash_pairs(self, content: str, page_number: int) -> list[GlossaryEntry]:
        """Extract term - definition pairs."""
        entries = []

        # Pattern: term - definition or term – definition
        pattern = r"^([^\n\-–—]{2,100})\s*[-–—]\s*(.{10,})$"
        matches = re.findall(pattern, content, re.MULTILINE | re.UNICODE)

        for term, definition in matches:
            term = term.strip()
            definition = definition.strip()

            if term and definition and len(term) < 150:
                entries.append(
                    GlossaryEntry(
                        term=term,
                        definition=definition,
                        source_page=page_number,
                    )
                )

        return entries

    def extract_and_save_glossary(
        self,
        document_id: int,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> list[Term]:
        """
        Detect glossary pages and save extracted terms to database.

        This method:
        1. Finds glossary pages in the document
        2. Extracts term-definition pairs
        3. Saves them as Terms with high priority (approved=True)

        Args:
            document_id: Document ID.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of saved Term objects.
        """
        # Detect glossary pages
        glossary_pages = self.detect_glossary_pages(document_id)

        if not glossary_pages:
            self.db.log(
                level="INFO",
                stage="glossary_detection",
                message="No glossary pages detected in document",
                document_id=document_id,
            )
            return []

        # Extract entries from all glossary pages
        all_entries: list[GlossaryEntry] = []
        seen_terms: set[str] = set()

        for page in glossary_pages:
            entries = self.extract_glossary_entries(page, source_lang)
            for entry in entries:
                # Deduplicate by normalized term
                term_key = entry.term.lower().strip()
                if term_key not in seen_terms:
                    seen_terms.add(term_key)
                    all_entries.append(entry)

        # Save terms to database
        terms: list[Term] = []
        for entry in all_entries:
            term = Term(
                document_id=document_id,
                term=entry.term,
                frequency=100,  # High frequency = high priority
                context=f"From glossary page {entry.source_page + 1}",
                approved=True,  # Pre-approved since from official glossary
            )

            # Set definition as translation based on document direction
            # If source is Arabic and target is English:
            #   - term is Arabic, definition might be English
            # We store the definition in the target language column
            if target_lang == "en":
                term.translation_en = entry.definition
            elif target_lang == "ar":
                term.translation_ar = entry.definition
            elif target_lang == "fr":
                term.translation_fr = entry.definition

            term_id = self.db.add_term(term)
            term.id = term_id
            terms.append(term)

        # Log results
        self.db.log(
            level="INFO",
            stage="glossary_extraction",
            message=f"Extracted {len(terms)} terms from {len(glossary_pages)} glossary pages",
            document_id=document_id,
            context={
                "glossary_pages": [p.page_number + 1 for p in glossary_pages],
                "terms_extracted": len(terms),
            },
        )

        return terms


def detect_glossary_language(content: str) -> str | None:
    """
    Detect the language of glossary content.

    Args:
        content: Text content to analyze.

    Returns:
        Language code or None if not detected.
    """
    for pattern, lang in GLOSSARY_HEADERS:
        if re.search(pattern, content, re.IGNORECASE | re.UNICODE):
            return lang

    # Check character ranges as fallback
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", content))
    latin_chars = len(re.findall(r"[a-zA-Z]", content))
    hebrew_chars = len(re.findall(r"[\u0590-\u05FF]", content))

    total = arabic_chars + latin_chars + hebrew_chars
    if total == 0:
        return None

    if arabic_chars / total > 0.5:
        return "ar"
    if hebrew_chars / total > 0.5:
        return "he"
    if latin_chars / total > 0.5:
        return "en"

    return None
