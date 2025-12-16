"""
Styling extraction and application for document translation.

Extracts font, color, and formatting information from source documents
and applies them to exported translations.
"""

from translate_docs_ai.styling.extractor import DocumentStyle, PageStyle, StyleExtractor, TextStyle
from translate_docs_ai.styling.fonts import FontManager, get_fallback_font

__all__ = [
    "StyleExtractor",
    "DocumentStyle",
    "PageStyle",
    "TextStyle",
    "FontManager",
    "get_fallback_font",
]
