"""
Style extraction from PDF and DOCX documents.

Extracts font, color, and formatting information for preservation
during translation and export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from translate_docs_ai.styling.fonts import FontCategory, categorize_font


@dataclass
class TextStyle:
    """Style information for a text span."""

    font_name: str | None = None
    font_size: float | None = None
    font_color: str | None = None  # Hex color like "#000000"
    is_bold: bool = False
    is_italic: bool = False
    is_underline: bool = False
    is_strikethrough: bool = False
    background_color: str | None = None
    category: FontCategory = FontCategory.SANS_SERIF

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "font_name": self.font_name,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "is_underline": self.is_underline,
            "is_strikethrough": self.is_strikethrough,
            "background_color": self.background_color,
            "category": self.category.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextStyle:
        """Create from dictionary."""
        return cls(
            font_name=data.get("font_name"),
            font_size=data.get("font_size"),
            font_color=data.get("font_color"),
            is_bold=data.get("is_bold", False),
            is_italic=data.get("is_italic", False),
            is_underline=data.get("is_underline", False),
            is_strikethrough=data.get("is_strikethrough", False),
            background_color=data.get("background_color"),
            category=FontCategory(data.get("category", "sans-serif")),
        )


@dataclass
class PageStyle:
    """Style information for a page."""

    page_number: int
    dominant_style: TextStyle | None = None
    heading_styles: list[TextStyle] = field(default_factory=list)
    body_style: TextStyle | None = None
    table_style: TextStyle | None = None
    has_headers: bool = False
    has_footers: bool = False
    page_width: float | None = None
    page_height: float | None = None
    margins: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "page_number": self.page_number,
            "dominant_style": self.dominant_style.to_dict() if self.dominant_style else None,
            "heading_styles": [s.to_dict() for s in self.heading_styles],
            "body_style": self.body_style.to_dict() if self.body_style else None,
            "table_style": self.table_style.to_dict() if self.table_style else None,
            "has_headers": self.has_headers,
            "has_footers": self.has_footers,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "margins": self.margins,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageStyle:
        """Create from dictionary."""
        return cls(
            page_number=data.get("page_number", 0),
            dominant_style=TextStyle.from_dict(data["dominant_style"])
            if data.get("dominant_style")
            else None,
            heading_styles=[TextStyle.from_dict(s) for s in data.get("heading_styles", [])],
            body_style=TextStyle.from_dict(data["body_style"]) if data.get("body_style") else None,
            table_style=TextStyle.from_dict(data["table_style"])
            if data.get("table_style")
            else None,
            has_headers=data.get("has_headers", False),
            has_footers=data.get("has_footers", False),
            page_width=data.get("page_width"),
            page_height=data.get("page_height"),
            margins=data.get("margins", {}),
        )


@dataclass
class DocumentStyle:
    """Style information for an entire document."""

    file_path: str
    page_styles: list[PageStyle] = field(default_factory=list)
    dominant_font: str | None = None
    dominant_size: float | None = None
    dominant_color: str | None = None
    heading_font: str | None = None
    body_font: str | None = None
    is_rtl: bool = False
    color_palette: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert to JSON string for storage."""
        return json.dumps(
            {
                "file_path": self.file_path,
                "page_styles": [ps.to_dict() for ps in self.page_styles],
                "dominant_font": self.dominant_font,
                "dominant_size": self.dominant_size,
                "dominant_color": self.dominant_color,
                "heading_font": self.heading_font,
                "body_font": self.body_font,
                "is_rtl": self.is_rtl,
                "color_palette": self.color_palette,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> DocumentStyle:
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            file_path=data.get("file_path", ""),
            page_styles=[PageStyle.from_dict(ps) for ps in data.get("page_styles", [])],
            dominant_font=data.get("dominant_font"),
            dominant_size=data.get("dominant_size"),
            dominant_color=data.get("dominant_color"),
            heading_font=data.get("heading_font"),
            body_font=data.get("body_font"),
            is_rtl=data.get("is_rtl", False),
            color_palette=data.get("color_palette", []),
        )

    def get_page_style(self, page_number: int) -> PageStyle | None:
        """Get style for a specific page."""
        for ps in self.page_styles:
            if ps.page_number == page_number:
                return ps
        return None


class StyleExtractor:
    """
    Extracts styling information from PDF documents.

    Uses PyMuPDF to analyze text spans and extract:
    - Font names, sizes, and colors
    - Bold, italic, underline flags
    - Page dimensions and margins
    - Heading vs body text detection
    """

    def __init__(self):
        """Initialize style extractor."""
        self._font_stats: dict[str, int] = {}
        self._size_stats: dict[float, int] = {}
        self._color_stats: dict[str, int] = {}

    def extract_from_pdf(self, file_path: Path) -> DocumentStyle:
        """
        Extract styling information from a PDF file.

        Args:
            file_path: Path to PDF file.

        Returns:
            DocumentStyle with extracted style information.
        """
        doc_style = DocumentStyle(file_path=str(file_path))
        self._font_stats.clear()
        self._size_stats.clear()
        self._color_stats.clear()

        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_style = self._extract_page_style(page, page_num)
                    doc_style.page_styles.append(page_style)

                # Calculate document-level dominant styles
                self._calculate_dominant_styles(doc_style)

                # Detect RTL based on content analysis
                doc_style.is_rtl = self._detect_rtl(doc)

        except Exception:
            # Return empty style on error
            doc_style.page_styles = []

        return doc_style

    def _extract_page_style(self, page: fitz.Page, page_num: int) -> PageStyle:
        """Extract style information from a single page."""
        page_style = PageStyle(
            page_number=page_num,
            page_width=page.rect.width,
            page_height=page.rect.height,
        )

        # Get text with detailed formatting info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        span_styles: list[tuple[TextStyle, int]] = []  # (style, char_count)
        heading_candidates: list[TextStyle] = []

        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    style = self._extract_span_style(span)
                    char_count = len(span.get("text", ""))

                    if char_count > 0:
                        span_styles.append((style, char_count))

                        # Track statistics
                        if style.font_name:
                            self._font_stats[style.font_name] = (
                                self._font_stats.get(style.font_name, 0) + char_count
                            )
                        if style.font_size:
                            self._size_stats[style.font_size] = (
                                self._size_stats.get(style.font_size, 0) + char_count
                            )
                        if style.font_color:
                            self._color_stats[style.font_color] = (
                                self._color_stats.get(style.font_color, 0) + char_count
                            )

                        # Detect headings (larger font or bold)
                        if (
                            style.font_size
                            and style.font_size > 14
                            or style.is_bold
                            and char_count < 200
                        ):
                            heading_candidates.append(style)

        # Determine dominant style for page
        if span_styles:
            # Find most common style by character count
            style_weights: dict[str, tuple[TextStyle, int]] = {}

            for style, count in span_styles:
                key = f"{style.font_name}:{style.font_size}:{style.font_color}"
                if key in style_weights:
                    style_weights[key] = (style, style_weights[key][1] + count)
                else:
                    style_weights[key] = (style, count)

            # Get the style with most characters
            dominant = max(style_weights.values(), key=lambda x: x[1])
            page_style.dominant_style = dominant[0]
            page_style.body_style = dominant[0]

        # Set heading styles (deduplicated)
        seen_headings: set[str] = set()
        for style in heading_candidates:
            key = f"{style.font_name}:{style.font_size}:{style.is_bold}"
            if key not in seen_headings:
                seen_headings.add(key)
                page_style.heading_styles.append(style)

        # Detect headers/footers by checking top/bottom of page
        page_style.has_headers = self._detect_header_footer(page, is_header=True)
        page_style.has_footers = self._detect_header_footer(page, is_header=False)

        return page_style

    def _extract_span_style(self, span: dict[str, Any]) -> TextStyle:
        """Extract style from a text span."""
        font_name = span.get("font", "")
        font_size = span.get("size", 12.0)
        flags = span.get("flags", 0)

        # Extract color (int to hex)
        color_int = span.get("color", 0)
        if isinstance(color_int, int):
            # Convert to hex RGB
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            font_color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            font_color = "#000000"

        # Parse flags
        # flags: 1=superscript, 2=italic, 4=serifed, 8=monospaced, 16=bold
        is_bold = bool(flags & 16)
        is_italic = bool(flags & 2)

        # Categorize font
        category = categorize_font(font_name)

        return TextStyle(
            font_name=font_name,
            font_size=font_size,
            font_color=font_color,
            is_bold=is_bold,
            is_italic=is_italic,
            category=category,
        )

    def _calculate_dominant_styles(self, doc_style: DocumentStyle) -> None:
        """Calculate document-level dominant styles."""
        # Dominant font (most used)
        if self._font_stats:
            doc_style.dominant_font = max(self._font_stats, key=self._font_stats.get)

        # Dominant size (most used)
        if self._size_stats:
            doc_style.dominant_size = max(self._size_stats, key=self._size_stats.get)

        # Dominant color (most used, excluding black)
        if self._color_stats:
            doc_style.dominant_color = max(self._color_stats, key=self._color_stats.get)

        # Color palette (top 5 colors)
        sorted_colors = sorted(self._color_stats.items(), key=lambda x: x[1], reverse=True)
        doc_style.color_palette = [c[0] for c in sorted_colors[:5]]

        # Detect heading font (largest font that's not body)
        if self._size_stats and doc_style.dominant_size:
            heading_sizes = [s for s in self._size_stats if s > doc_style.dominant_size]
            if heading_sizes:
                # Find font most commonly used with larger sizes
                for page_style in doc_style.page_styles:
                    for heading in page_style.heading_styles:
                        if heading.font_name:
                            doc_style.heading_font = heading.font_name
                            break
                    if doc_style.heading_font:
                        break

        # Body font is the dominant font
        doc_style.body_font = doc_style.dominant_font

    def _detect_header_footer(self, page: fitz.Page, is_header: bool) -> bool:
        """Detect if page has header or footer."""
        height = page.rect.height
        margin = height * 0.1  # Top/bottom 10%

        if is_header:
            rect = fitz.Rect(0, 0, page.rect.width, margin)
        else:
            rect = fitz.Rect(0, height - margin, page.rect.width, height)

        text = page.get_text("text", clip=rect).strip()
        return len(text) > 0

    def _detect_rtl(self, doc: fitz.Document) -> bool:
        """Detect if document is primarily RTL."""
        # Sample first few pages
        rtl_chars = 0
        total_chars = 0

        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text()

            for char in text:
                code = ord(char)
                total_chars += 1

                # Arabic Unicode range: 0x0600-0x06FF
                # Hebrew Unicode range: 0x0590-0x05FF
                if 0x0600 <= code <= 0x06FF or 0x0590 <= code <= 0x05FF:
                    rtl_chars += 1

        if total_chars == 0:
            return False

        # Consider RTL if >30% of characters are RTL
        return (rtl_chars / total_chars) > 0.3


def extract_document_style(file_path: Path) -> DocumentStyle | None:
    """
    Extract styling from a document file.

    Args:
        file_path: Path to document (PDF).

    Returns:
        DocumentStyle or None if extraction fails.
    """
    if file_path.suffix.lower() != ".pdf":
        return None

    extractor = StyleExtractor()
    return extractor.extract_from_pdf(file_path)
