"""
PDF exporter for translated documents.

Uses markdown-pdf library for beautiful PDF generation with RTL support.
Handles RTL to LTR table column reversal for proper reading direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from markdown_pdf import MarkdownPdf, Section

from translate_docs_ai.export.table_utils import process_content_tables


def _sanitize_markdown_for_pdf(content: str) -> str:
    """
    Sanitize markdown content to avoid PDF generation issues.

    Removes or fixes problematic patterns like:
    - Anchor links that reference non-existent sections
    - Malformed internal links
    """
    # Replace anchor links [text](#anchor) with just the text
    # These cause "No destination with id=X" errors in markdown-pdf
    content = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", content)

    # Also handle empty links [text]()
    content = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", content)

    return content


if TYPE_CHECKING:
    from translate_docs_ai.database import Database, Document


# CSS styles for PDF output - optimized to fit content on pages
DEFAULT_CSS = """
@page {
    size: A4;
    margin: 1.5cm 1.5cm 1.5cm 1.5cm;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 9pt;
    line-height: 1.3;
    color: #333;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 14pt;
    color: #1a1a2e;
    border-bottom: 1px solid #4a90d9;
    padding-bottom: 4px;
    margin-top: 8px;
    margin-bottom: 8px;
}

h2 {
    font-size: 12pt;
    color: #16213e;
    border-bottom: 1px solid #ddd;
    padding-bottom: 3px;
    margin-top: 6px;
    margin-bottom: 4px;
}

h3 {
    font-size: 10pt;
    color: #1f4068;
    margin-top: 5px;
    margin-bottom: 3px;
}

h4, h5, h6 {
    font-size: 9pt;
    color: #1f4068;
    margin-top: 4px;
    margin-bottom: 2px;
}

p {
    text-align: justify;
    margin-bottom: 4px;
    margin-top: 2px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 6px 0;
    font-size: 8pt;
}

th, td {
    border: 1px solid #ddd;
    padding: 4px 6px;
    text-align: left;
}

th {
    background-color: #4a90d9;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

code {
    background-color: #f4f4f4;
    padding: 1px 3px;
    border-radius: 2px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 8pt;
}

pre {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 8px;
    border-radius: 3px;
    overflow-x: auto;
    font-size: 7pt;
    margin: 4px 0;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: inherit;
}

blockquote {
    border-left: 3px solid #4a90d9;
    margin: 6px 0;
    padding: 4px 10px;
    background-color: #f8f9fa;
    font-style: italic;
    font-size: 8pt;
}

ul, ol {
    margin-bottom: 4px;
    margin-top: 2px;
    padding-left: 20px;
}

li {
    margin-bottom: 2px;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 30px 0;
}

.page-header {
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #4a90d9;
}

.metadata {
    font-size: 10pt;
    color: #666;
    margin-bottom: 20px;
}
"""

# RTL CSS for Arabic
RTL_CSS = """
body {
    direction: rtl;
    text-align: right;
    font-family: 'Traditional Arabic', 'Simplified Arabic', 'Tahoma', sans-serif;
}

th, td {
    text-align: right;
}

blockquote {
    border-left: none;
    border-right: 4px solid #4a90d9;
}

ul, ol {
    padding-left: 0;
    padding-right: 30px;
}
"""


@dataclass
class PDFExportResult:
    """Result of a PDF export operation."""

    document_id: int
    document_name: str
    output_path: Path
    pages_exported: int
    language: str
    success: bool
    error: str | None = None


class PDFExporter:
    """
    Exports translated documents to PDF files.

    Uses markdown-pdf for high-quality PDF generation with:
    - Professional styling
    - RTL support for Arabic
    - Table of contents
    - Page headers/footers
    """

    LANGUAGE_COLUMNS = {
        "en": "en_content",
        "ar": "ar_content",
        "fr": "fr_content",
    }

    RTL_LANGUAGES = {"ar", "he", "fa", "ur"}

    def __init__(self, db: Database, output_dir: Path) -> None:
        """
        Initialize the PDF exporter.

        Args:
            db: Database instance.
            output_dir: Base output directory.
        """
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_document(
        self,
        document: Document,
        language: str = "en",
        include_toc: bool = True,
        source_lang: str | None = None,
    ) -> PDFExportResult:
        """
        Export a single document to PDF.

        Args:
            document: Document to export.
            language: Target language code (en, ar, fr).
            include_toc: Include table of contents.
            source_lang: Source language code for RTL/LTR table conversion.

        Returns:
            PDFExportResult with export status.
        """
        content_column = self.LANGUAGE_COLUMNS.get(language)
        if not content_column:
            return PDFExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=f"Unsupported language: {language}",
            )

        pages = self.db.get_document_pages(document.id)

        translated_pages: list[tuple[int, str]] = []
        for page in pages:
            content = getattr(page, content_column, None)
            if content:
                # Process tables: convert HTML to MD and reverse columns for RTL→LTR
                if source_lang:
                    content = process_content_tables(content, source_lang, language)
                # Sanitize markdown to avoid PDF generation issues (anchor links, etc.)
                content = _sanitize_markdown_for_pdf(content)
                translated_pages.append((page.page_number, content))

        if not translated_pages:
            return PDFExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=f"No {language} translations found",
            )

        translated_pages.sort(key=lambda x: x[0])

        # Create safe filename
        doc_stem = Path(document.file_name).stem
        safe_name = self._sanitize_filename(doc_stem)
        output_file = self.output_dir / f"{safe_name}_{language}.pdf"

        # Build CSS
        css = DEFAULT_CSS
        if language in self.RTL_LANGUAGES:
            css += RTL_CSS

        try:
            # Create PDF with separate sections for each page (creates page breaks)
            # Use toc_level=1 to include h1 headings in TOC
            pdf = MarkdownPdf(toc_level=1 if include_toc else 0)

            # Title page section
            title_content = f"""# {document.file_name}

**Language:** {language.upper()}

**Total Pages:** {len(translated_pages)}

---
"""
            pdf.add_section(Section(title_content, toc=False))

            # Each translated page as a separate section (creates page breaks)
            for page_num, content in translated_pages:
                # Use h1 for page headers so they appear in TOC
                page_content = f"# Page {page_num + 1}\n\n{content}"
                pdf.add_section(Section(page_content, toc=include_toc))

            pdf.meta["title"] = document.file_name
            pdf.meta["author"] = "translate-docs-ai"
            pdf.save(str(output_file))

            return PDFExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=output_file,
                pages_exported=len(translated_pages),
                language=language,
                success=True,
            )

        except Exception as e:
            return PDFExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=str(e),
            )

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

    def export_all(
        self,
        documents: list[Document],
        language: str = "en",
        include_toc: bool = True,
        source_lang: str | None = None,
    ) -> list[PDFExportResult]:
        """Export multiple documents to PDF."""
        results = []
        for doc in documents:
            result = self.export_document(doc, language, include_toc, source_lang)
            results.append(result)
        return results
