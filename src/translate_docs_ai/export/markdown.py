"""
Markdown exporter for translated documents.

Exports translated content to markdown files with various organization options.
Handles RTL to LTR table conversion and HTML to Markdown table conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from translate_docs_ai.export.table_utils import process_content_tables

if TYPE_CHECKING:
    from translate_docs_ai.database import Database, Document


@dataclass
class ExportResult:
    """Result of an export operation."""

    document_id: int
    document_name: str
    output_path: Path
    pages_exported: int
    language: str
    success: bool
    error: str | None = None


class MarkdownExporter:
    """
    Exports translated documents to markdown files.

    Supports various output formats:
    - Combined: Single file per document with all pages
    - Separate: Individual files per page in a directory
    """

    # Supported language codes and their content column names
    LANGUAGE_COLUMNS = {
        "en": "en_content",
        "ar": "ar_content",
        "fr": "fr_content",
    }

    def __init__(self, db: Database, output_dir: Path) -> None:
        """
        Initialize the markdown exporter.

        Args:
            db: Database instance for retrieving documents and pages.
            output_dir: Base output directory for exported files.
        """
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_document(
        self,
        document: Document,
        language: str = "en",
        combined: bool = True,
        source_lang: str | None = None,
        clean: bool = False,
    ) -> ExportResult:
        """
        Export a single document to markdown.

        Args:
            document: Document to export.
            language: Target language code (en, ar, fr).
            combined: If True, export as single file; if False, separate files per page.
            source_lang: Source language code for RTL/LTR table conversion.
            clean: If True, export without metadata headers, page numbers, or separators.

        Returns:
            ExportResult with export status and details.
        """
        # Validate language
        content_column = self.LANGUAGE_COLUMNS.get(language)
        if not content_column:
            return ExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=f"Unsupported language: {language}. Use: {', '.join(self.LANGUAGE_COLUMNS.keys())}",
            )

        # Get pages
        pages = self.db.get_document_pages(document.id)

        # Extract translated content and process tables
        translated_pages: list[tuple[int, str]] = []
        for page in pages:
            content = getattr(page, content_column, None)
            if content:
                # Process tables: convert HTML to MD and reverse columns for RTL→LTR
                if source_lang:
                    content = process_content_tables(content, source_lang, language)
                translated_pages.append((page.page_number, content))

        if not translated_pages:
            return ExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=f"No {language} translations found",
            )

        # Sort by page number
        translated_pages.sort(key=lambda x: x[0])

        # Create safe filename
        doc_stem = Path(document.file_name).stem
        safe_name = self._sanitize_filename(doc_stem)

        if combined:
            output_path = self._export_combined(
                document, safe_name, translated_pages, language, clean
            )
        else:
            output_path = self._export_separate(safe_name, translated_pages, language)

        return ExportResult(
            document_id=document.id,
            document_name=document.file_name,
            output_path=output_path,
            pages_exported=len(translated_pages),
            language=language,
            success=True,
        )

    def _export_combined(
        self,
        document: Document,
        safe_name: str,
        pages: list[tuple[int, str]],
        language: str,
        clean: bool = False,
    ) -> Path:
        """Export document as a single combined markdown file."""
        output_file = self.output_dir / f"{safe_name}_{language}.md"

        content_parts = []

        if not clean:
            # Add header with metadata
            content_parts.append(f"# {document.file_name}")
            content_parts.append("")
            content_parts.append(f"**Source:** {document.file_name}")
            content_parts.append(f"**Language:** {language.upper()}")
            content_parts.append(f"**Total Pages:** {len(pages)}")
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")

        # Add pages
        for page_num, content in pages:
            if not clean:
                content_parts.append(f"## Page {page_num + 1}")
                content_parts.append("")
            content_parts.append(content)
            if not clean:
                content_parts.append("")
                content_parts.append("---")
            content_parts.append("")

        output_file.write_text("\n".join(content_parts), encoding="utf-8")
        return output_file

    def _export_separate(
        self,
        safe_name: str,
        pages: list[tuple[int, str]],
        language: str,
    ) -> Path:
        """Export document as separate markdown files per page."""
        doc_output_dir = self.output_dir / safe_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        for page_num, content in pages:
            page_file = doc_output_dir / f"page_{page_num + 1:03d}_{language}.md"
            page_file.write_text(content, encoding="utf-8")

        return doc_output_dir

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

    def export_all(
        self,
        documents: list[Document],
        language: str = "en",
        combined: bool = True,
        source_lang: str | None = None,
        clean: bool = False,
    ) -> list[ExportResult]:
        """
        Export multiple documents.

        Args:
            documents: List of documents to export.
            language: Target language code (en, ar, fr).
            combined: If True, export as single file; if False, separate files per page.
            source_lang: Source language code for RTL/LTR table conversion.
            clean: If True, export without metadata headers, page numbers, or separators.

        Returns:
            List of ExportResults for each document.
        """
        results = []
        for doc in documents:
            result = self.export_document(doc, language, combined, source_lang, clean)
            results.append(result)
        return results
