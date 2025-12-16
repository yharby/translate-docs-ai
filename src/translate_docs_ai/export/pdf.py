"""
PDF exporter for translated documents.

Uses markdown-pdf library for beautiful PDF generation with RTL support.
Handles RTL to LTR table column reversal for proper reading direction.
Supports applying extracted styling from source PDF documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markdown_pdf import MarkdownPdf, Section

from translate_docs_ai.export.table_utils import process_content_tables
from translate_docs_ai.styling import get_fallback_font


def _normalize_bullets_to_markdown(content: str) -> str:
    """
    Convert Unicode bullet characters to proper markdown list syntax.

    The OCR/translation may produce Unicode bullets (•, ○, ▪) as plain text
    instead of proper markdown list markers (-, *). This converts them.
    """
    lines = content.split("\n")
    result = []

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Convert various Unicode bullet styles to markdown
        # • (U+2022), ● (U+25CF), ○ (U+25CB), ▪ (U+25AA), ▫ (U+25AB), ◦ (U+25E6)
        if stripped.startswith(("• ", "● ", "○ ", "▪ ", "▫ ", "◦ ", "- ", "* ")):
            # Calculate indent level (each 2-4 spaces = one level)
            indent_level = indent // 2
            # Use proper markdown with indentation
            bullet_text = stripped[2:].strip() if len(stripped) > 2 else ""
            new_line = "  " * indent_level + "- " + bullet_text
            result.append(new_line)
        elif stripped.startswith(("•", "●", "○", "▪", "▫", "◦")) and len(stripped) > 1:
            # Handle bullets without space after (•text)
            indent_level = indent // 2
            bullet_text = stripped[1:].strip()
            new_line = "  " * indent_level + "- " + bullet_text
            result.append(new_line)
        else:
            result.append(line)

    return "\n".join(result)


def _sanitize_markdown_for_pdf(content: str) -> str:
    """
    Sanitize markdown content to avoid PDF generation issues.

    Removes or fixes problematic patterns like:
    - Anchor links that reference non-existent sections
    - Malformed internal links
    - Unicode bullets that should be markdown lists
    """
    # First, normalize Unicode bullets to markdown lists
    content = _normalize_bullets_to_markdown(content)

    # Replace anchor links [text](#anchor) with just the text
    # These cause "No destination with id=X" errors in markdown-pdf
    content = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", content)

    # Also handle empty links [text]()
    content = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", content)

    return content


if TYPE_CHECKING:
    from translate_docs_ai.database import Database, Document


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
        self._doc_style: dict[str, Any] | None = None

    def _load_document_styling(self, document_id: int) -> dict[str, Any] | None:
        """Load extracted styling metadata for a document."""
        return self.db.get_document_styling(document_id)

    def _get_font_for_language(self, language: str, original_font: str | None = None) -> str:
        """Get appropriate font for language with fallback support."""
        if original_font:
            return get_fallback_font(original_font, language)
        return get_fallback_font("Arial", language)

    def _generate_dynamic_css(self, language: str) -> str:
        """Generate CSS based on extracted document styling."""
        # Default values (numeric for computation)
        body_font = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
        body_size = 9.0  # pt
        body_color = "#333"
        heading_size = 14.0  # pt

        if self._doc_style:
            # Extract styling info
            original_font = self._doc_style.get("dominant_font")
            if original_font:
                fallback_font = self._get_font_for_language(language, original_font)
                body_font = f"'{fallback_font}', 'Segoe UI', Tahoma, sans-serif"

            if self._doc_style.get("dominant_size"):
                body_size = min(self._doc_style["dominant_size"], 12.0)
                heading_size = body_size + 5

            if self._doc_style.get("dominant_color"):
                body_color = self._doc_style["dominant_color"]

        # Pre-compute derived sizes (MuPDF doesn't support calc())
        h2_size = heading_size - 2
        h3_size = heading_size - 4
        table_size = body_size - 1
        code_size = body_size - 1
        pre_size = body_size - 2
        blockquote_size = body_size - 1
        metadata_size = body_size + 1

        # Generate CSS with pre-computed values
        css = f"""
@page {{
    size: A4;
    margin: 1.5cm 1.5cm 1.5cm 1.5cm;
}}

body {{
    font-family: {body_font};
    font-size: {body_size}pt;
    line-height: 1.3;
    color: {body_color};
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: {heading_size}pt;
    color: #1a1a2e;
    border-bottom: 1px solid #4a90d9;
    padding-bottom: 4px;
    margin-top: 8px;
    margin-bottom: 8px;
}}

h2 {{
    font-size: {h2_size}pt;
    color: #16213e;
    border-bottom: 1px solid #ddd;
    padding-bottom: 3px;
    margin-top: 6px;
    margin-bottom: 4px;
}}

h3 {{
    font-size: {h3_size}pt;
    color: #1f4068;
    margin-top: 5px;
    margin-bottom: 3px;
}}

h4, h5, h6 {{
    font-size: {body_size}pt;
    color: #1f4068;
    margin-top: 4px;
    margin-bottom: 2px;
}}

p {{
    text-align: justify;
    margin-bottom: 4px;
    margin-top: 2px;
}}

table {{
    border-spacing: 0;
    width: 100%;
    margin: 4px 0;
    font-size: {table_size}pt;
    border: 1px solid #000;
}}

th, td {{
    border: 0.5px solid #000;
    text-align: left;
    padding: 2px 4px;
    min-width: 40px;
}}

th {{
    background-color: transparent;
    font-weight: bold;
    border-bottom: 2px solid #666;
}}

code {{
    background-color: #f4f4f4;
    padding: 1px 3px;
    border-radius: 2px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: {code_size}pt;
}}

pre {{
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 8px;
    border-radius: 3px;
    overflow-x: auto;
    font-size: {pre_size}pt;
    margin: 4px 0;
}}

pre code {{
    background-color: transparent;
    padding: 0;
    color: inherit;
}}

blockquote {{
    border-left: 3px solid #4a90d9;
    margin: 6px 0;
    padding: 4px 10px;
    background-color: #f8f9fa;
    font-style: italic;
    font-size: {blockquote_size}pt;
}}

ul {{
    list-style-type: disc;
    list-style-position: outside;
    margin-bottom: 6px;
    margin-top: 4px;
    padding-left: 20px;
    margin-left: 0;
}}

ol {{
    list-style-type: decimal;
    list-style-position: outside;
    margin-bottom: 6px;
    margin-top: 4px;
    padding-left: 20px;
    margin-left: 0;
}}

li {{
    margin-bottom: 3px;
    line-height: 1.3;
    padding-left: 2px;
}}

li p {{
    margin: 0;
    display: inline;
}}

ul ul, ol ul {{
    list-style-type: circle;
    margin-top: 2px;
    margin-bottom: 2px;
    padding-left: 18px;
}}

ul ul ul, ol ul ul {{
    list-style-type: square;
    padding-left: 16px;
}}

ol ol {{
    list-style-type: lower-alpha;
    padding-left: 18px;
}}

hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 30px 0;
}}

.page-header {{
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #4a90d9;
}}

.metadata {{
    font-size: {metadata_size}pt;
    color: #666;
    margin-bottom: 20px;
}}
"""
        return css

    def _get_rtl_css(self, language: str) -> str:
        """Get RTL CSS with appropriate font for the language."""
        rtl_font = self._get_font_for_language(language)
        return f"""
body {{
    direction: rtl;
    text-align: right;
    font-family: '{rtl_font}', 'Traditional Arabic', 'Simplified Arabic', 'Tahoma', sans-serif;
}}

th, td {{
    text-align: right;
}}

blockquote {{
    border-left: none;
    border-right: 4px solid #4a90d9;
}}

ul, ol {{
    padding-left: 0;
    padding-right: 30px;
}}
"""

    def export_document(
        self,
        document: Document,
        language: str = "en",
        include_toc: bool = True,
        source_lang: str | None = None,
        clean: bool = False,
    ) -> PDFExportResult:
        """
        Export a single document to PDF.

        Args:
            document: Document to export.
            language: Target language code (en, ar, fr).
            include_toc: Include table of contents.
            source_lang: Source language code for RTL/LTR table conversion.
            clean: If True, export without title page, page headers, or metadata.

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

        # Load extracted styling from source document
        self._doc_style = self._load_document_styling(document.id)

        # Build CSS with dynamic styling based on source document
        css = self._generate_dynamic_css(language)
        if language in self.RTL_LANGUAGES:
            css += self._get_rtl_css(language)

        try:
            # Create PDF with content flowing naturally (no forced page breaks)
            # Use toc_level=1 to include h1 headings in TOC (disabled for clean mode)
            use_toc = include_toc and not clean
            pdf = MarkdownPdf(toc_level=1 if use_toc else 0)

            # Combine all content into a single section for natural flow
            all_content_parts: list[str] = []

            if not clean:
                # Title header (not a separate section to avoid page break)
                title_content = f"""# {document.file_name}

**Language:** {language.upper()}

**Total Pages:** {len(translated_pages)}

---
"""
                all_content_parts.append(title_content)

            # Combine all translated pages into continuous content
            for _page_num, content in translated_pages:
                all_content_parts.append(content)

            # Join content with double newlines (no page markers, just continuous flow)
            combined_content = "\n\n".join(all_content_parts)

            # Single section = content flows naturally across pages
            pdf.add_section(Section(combined_content, toc=use_toc), user_css=css)

            if not clean:
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
        clean: bool = False,
    ) -> list[PDFExportResult]:
        """Export multiple documents to PDF."""
        results = []
        for doc in documents:
            result = self.export_document(doc, language, include_toc, source_lang, clean)
            results.append(result)
        return results
