"""
DOCX exporter for translated documents.

Uses python-docx for professional Word document generation.
Handles RTL to LTR table column reversal for proper reading direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from docx import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

from translate_docs_ai.export.table_utils import process_content_tables

if TYPE_CHECKING:
    from translate_docs_ai.database import Database, Document


@dataclass
class DOCXExportResult:
    """Result of a DOCX export operation."""

    document_id: int
    document_name: str
    output_path: Path
    pages_exported: int
    language: str
    success: bool
    error: str | None = None


class DOCXExporter:
    """
    Exports translated documents to DOCX files.

    Creates professionally formatted Word documents with:
    - Custom styling
    - RTL support for Arabic
    - Tables with borders
    - Headers and footers
    - Page numbers
    """

    LANGUAGE_COLUMNS = {
        "en": "en_content",
        "ar": "ar_content",
        "fr": "fr_content",
    }

    RTL_LANGUAGES = {"ar", "he", "fa", "ur"}

    # Color scheme
    COLORS = {
        "primary": RGBColor(0x1A, 0x1A, 0x2E),  # Dark blue
        "secondary": RGBColor(0x4A, 0x90, 0xD9),  # Light blue
        "text": RGBColor(0x33, 0x33, 0x33),  # Dark gray
        "light": RGBColor(0x66, 0x66, 0x66),  # Medium gray
    }

    def __init__(self, db: Database, output_dir: Path) -> None:
        """
        Initialize the DOCX exporter.

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
        source_lang: str | None = None,
    ) -> DOCXExportResult:
        """
        Export a single document to DOCX.

        Args:
            document: Document to export.
            language: Target language code (en, ar, fr).
            source_lang: Source language code for RTL/LTR table conversion.

        Returns:
            DOCXExportResult with export status.
        """
        content_column = self.LANGUAGE_COLUMNS.get(language)
        if not content_column:
            return DOCXExportResult(
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
                translated_pages.append((page.page_number, content))

        if not translated_pages:
            return DOCXExportResult(
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
        output_file = self.output_dir / f"{safe_name}_{language}.docx"

        try:
            # Create document
            docx = DocxDocument()
            is_rtl = language in self.RTL_LANGUAGES

            # Setup styles
            self._setup_styles(docx, is_rtl)

            # Add title page
            self._add_title_page(docx, document.file_name, language, len(translated_pages), is_rtl)

            # Add content with page breaks between each page
            for idx, (page_num, content) in enumerate(translated_pages):
                is_last = idx == len(translated_pages) - 1
                self._add_page_content(docx, page_num, content, is_rtl, is_last_page=is_last)

            # Add footer with page numbers
            self._add_footer(docx)

            # Save
            docx.save(str(output_file))

            return DOCXExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=output_file,
                pages_exported=len(translated_pages),
                language=language,
                success=True,
            )

        except Exception as e:
            return DOCXExportResult(
                document_id=document.id,
                document_name=document.file_name,
                output_path=self.output_dir,
                pages_exported=0,
                language=language,
                success=False,
                error=str(e),
            )

    def _setup_styles(self, docx: DocxDocument, is_rtl: bool) -> None:
        """Setup custom styles for the document - optimized for page fit."""
        styles = docx.styles

        # Title style
        if "CustomTitle" not in [s.name for s in styles]:
            title_style = styles.add_style("CustomTitle", WD_STYLE_TYPE.PARAGRAPH)
            title_style.font.size = Pt(16)
            title_style.font.bold = True
            title_style.font.color.rgb = self.COLORS["primary"]
            title_style.paragraph_format.space_after = Pt(6)
            title_style.paragraph_format.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if is_rtl else WD_ALIGN_PARAGRAPH.CENTER
            )

        # Heading 1 style - page headers
        h1_style = styles["Heading 1"]
        h1_style.font.size = Pt(12)
        h1_style.font.bold = True
        h1_style.font.color.rgb = self.COLORS["primary"]
        h1_style.paragraph_format.space_before = Pt(6)
        h1_style.paragraph_format.space_after = Pt(4)

        # Heading 2 style
        h2_style = styles["Heading 2"]
        h2_style.font.size = Pt(11)
        h2_style.font.bold = True
        h2_style.font.color.rgb = self.COLORS["secondary"]
        h2_style.paragraph_format.space_before = Pt(4)
        h2_style.paragraph_format.space_after = Pt(2)

        # Heading 3 style
        h3_style = styles["Heading 3"]
        h3_style.font.size = Pt(10)
        h3_style.font.bold = True
        h3_style.font.color.rgb = self.COLORS["text"]
        h3_style.paragraph_format.space_before = Pt(3)
        h3_style.paragraph_format.space_after = Pt(2)

        # Normal text - smaller for page fit
        normal_style = styles["Normal"]
        normal_style.font.size = Pt(9)
        normal_style.font.color.rgb = self.COLORS["text"]
        normal_style.paragraph_format.space_after = Pt(3)
        normal_style.paragraph_format.space_before = Pt(1)
        normal_style.paragraph_format.line_spacing = 1.15

        # List styles
        if "List Bullet" in [s.name for s in styles]:
            list_style = styles["List Bullet"]
            list_style.font.size = Pt(9)
            list_style.paragraph_format.space_after = Pt(2)

        if "List Number" in [s.name for s in styles]:
            list_num_style = styles["List Number"]
            list_num_style.font.size = Pt(9)
            list_num_style.paragraph_format.space_after = Pt(2)

        if is_rtl:
            normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            normal_style.font.name = "Traditional Arabic"

    def _add_title_page(
        self,
        docx: DocxDocument,
        title: str,
        language: str,
        page_count: int,
        is_rtl: bool,
    ) -> None:
        """Add a title page to the document."""
        # Main title - smaller for page fit
        title_para = docx.add_paragraph()
        title_run = title_para.add_run(title)
        title_run.font.size = Pt(16)
        title_run.font.bold = True
        title_run.font.color.rgb = self.COLORS["primary"]
        title_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_rtl else WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_after = Pt(12)

        # Separator line
        sep_para = docx.add_paragraph("_" * 40)
        sep_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_rtl else WD_ALIGN_PARAGRAPH.CENTER

        # Metadata
        meta_para = docx.add_paragraph()
        meta_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_rtl else WD_ALIGN_PARAGRAPH.CENTER

        lang_label = "اللغة" if is_rtl else "Language"
        pages_label = "عدد الصفحات" if is_rtl else "Total Pages"

        meta_run = meta_para.add_run(
            f"{lang_label}: {language.upper()}\n{pages_label}: {page_count}"
        )
        meta_run.font.size = Pt(10)
        meta_run.font.color.rgb = self.COLORS["light"]

        # Page break after title
        docx.add_page_break()

    def _add_page_content(
        self,
        docx: DocxDocument,
        page_num: int,
        content: str,
        is_rtl: bool,
        is_last_page: bool = False,
    ) -> None:
        """Add page content with markdown parsing."""
        # Page header with page number in footer style
        page_label = f"صفحة {page_num + 1}" if is_rtl else f"Page {page_num + 1}"
        heading = docx.add_heading(page_label, level=1)
        if is_rtl:
            heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        # Parse and add content
        self._parse_markdown_to_docx(docx, content, is_rtl)

        # Add page break after each page (except the last one)
        if not is_last_page:
            docx.add_page_break()

    def _parse_markdown_to_docx(
        self,
        docx: DocxDocument,
        content: str,
        is_rtl: bool,
    ) -> None:
        """Parse markdown content and add to document."""
        lines = content.split("\n")
        in_code_block = False
        code_lines: list[str] = []
        in_table = False
        table_rows: list[list[str]] = []

        for line in lines:
            # Code block handling
            if line.strip().startswith("```"):
                if in_code_block:
                    # End code block
                    self._add_code_block(docx, "\n".join(code_lines))
                    code_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # Table handling
            if "|" in line and not line.strip().startswith("|--"):
                if line.strip().startswith("|"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if cells:
                        table_rows.append(cells)
                        in_table = True
                continue
            elif in_table and table_rows:
                self._add_table(docx, table_rows, is_rtl)
                table_rows = []
                in_table = False

            # Skip table separator lines
            if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                continue

            # Empty line
            if not line.strip():
                docx.add_paragraph()
                continue

            # Headings
            if line.startswith("# "):
                h = docx.add_heading(line[2:], level=1)
                if is_rtl:
                    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif line.startswith("## "):
                h = docx.add_heading(line[3:], level=2)
                if is_rtl:
                    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif line.startswith("### "):
                h = docx.add_heading(line[4:], level=3)
                if is_rtl:
                    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Horizontal rule
            elif line.strip() in ("---", "***", "___"):
                para = docx.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = para.add_run("─" * 40)
                run.font.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
            # Bullet list
            elif line.strip().startswith(("- ", "* ", "+ ")):
                para = docx.add_paragraph(line.strip()[2:], style="List Bullet")
                if is_rtl:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Numbered list
            elif re.match(r"^\d+\.\s", line.strip()):
                text = re.sub(r"^\d+\.\s", "", line.strip())
                para = docx.add_paragraph(text, style="List Number")
                if is_rtl:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Blockquote
            elif line.strip().startswith(">"):
                para = docx.add_paragraph()
                para.paragraph_format.left_indent = Inches(0.5)
                run = para.add_run(line.strip()[1:].strip())
                run.font.italic = True
                run.font.color.rgb = self.COLORS["light"]
                if is_rtl:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Regular paragraph
            else:
                para = docx.add_paragraph()
                self._add_formatted_text(para, line, is_rtl)
                if is_rtl:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        # Handle any remaining table
        if table_rows:
            self._add_table(docx, table_rows, is_rtl)

    def _add_formatted_text(self, para, text: str, is_rtl: bool) -> None:
        """Add text with inline formatting (bold, italic, code)."""
        # Simple pattern matching for bold, italic, code
        patterns = [
            (r"\*\*(.+?)\*\*", "bold"),
            (r"__(.+?)__", "bold"),
            (r"\*(.+?)\*", "italic"),
            (r"_(.+?)_", "italic"),
            (r"`(.+?)`", "code"),
        ]

        remaining = text
        while remaining:
            earliest_match = None
            earliest_pos = len(remaining)
            match_type = None

            for pattern, fmt_type in patterns:
                match = re.search(pattern, remaining)
                if match and match.start() < earliest_pos:
                    earliest_match = match
                    earliest_pos = match.start()
                    match_type = fmt_type

            if earliest_match:
                # Add text before match
                if earliest_pos > 0:
                    para.add_run(remaining[:earliest_pos])

                # Add formatted text
                run = para.add_run(earliest_match.group(1))
                if match_type == "bold":
                    run.font.bold = True
                elif match_type == "italic":
                    run.font.italic = True
                elif match_type == "code":
                    run.font.name = "Consolas"
                    run.font.size = Pt(10)

                remaining = remaining[earliest_match.end() :]
            else:
                para.add_run(remaining)
                break

    def _add_code_block(self, docx: DocxDocument, code: str) -> None:
        """Add a formatted code block with smaller font for page fit."""
        para = docx.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.15)
        para.paragraph_format.right_indent = Inches(0.15)
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(4)

        # Add shading
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="2D2D2D"/>')
        para._p.get_or_add_pPr().append(shading)

        run = para.add_run(code)
        run.font.name = "Consolas"
        run.font.size = Pt(7)  # Smaller for code to fit
        run.font.color.rgb = RGBColor(0xF8, 0xF8, 0xF2)

    def _add_table(self, docx: DocxDocument, rows: list[list[str]], is_rtl: bool) -> None:
        """Add a formatted table with proper styling for page fit."""
        if not rows:
            return

        num_cols = max(len(row) for row in rows)
        table = docx.add_table(rows=len(rows), cols=num_cols)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.RIGHT if is_rtl else WD_TABLE_ALIGNMENT.LEFT

        for i, row_data in enumerate(rows):
            row = table.rows[i]
            for j, cell_text in enumerate(row_data):
                if j < num_cols:
                    cell = row.cells[j]

                    # Clear default paragraph and add formatted one
                    cell.text = ""
                    para = cell.paragraphs[0]
                    run = para.add_run(cell_text)

                    # Set smaller font for tables to fit content
                    run.font.size = Pt(8)
                    run.font.name = "Segoe UI"

                    # Reduce cell padding via paragraph spacing
                    para.paragraph_format.space_before = Pt(2)
                    para.paragraph_format.space_after = Pt(2)

                    # Header row styling
                    if i == 0:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        # Add blue background to header
                        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="4A90D9"/>')
                        cell._tc.get_or_add_tcPr().append(shading)
                    # Alternate row shading for readability
                    elif i % 2 == 0:
                        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F5"/>')
                        cell._tc.get_or_add_tcPr().append(shading)

        docx.add_paragraph()  # Space after table

    def _add_footer(self, docx: DocxDocument) -> None:
        """Add footer with page numbers."""
        section = docx.sections[0]
        footer = section.footer
        footer.is_linked_to_previous = False

        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run = para.add_run("Generated by translate-docs-ai")
        run.font.size = Pt(9)
        run.font.color.rgb = self.COLORS["light"]

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

    def export_all(
        self,
        documents: list[Document],
        language: str = "en",
        source_lang: str | None = None,
    ) -> list[DOCXExportResult]:
        """Export multiple documents to DOCX."""
        results = []
        for doc in documents:
            result = self.export_document(doc, language, source_lang)
            results.append(result)
        return results
