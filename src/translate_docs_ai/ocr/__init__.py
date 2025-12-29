"""
OCR and text extraction pipeline for translate-docs-ai.

Provides multiple extraction backends:
- PyMuPDF4LLM for native PDF text extraction
- olmOCR-2 via DeepInfra for scanned documents
- DeepSeek-OCR via DeepInfra for complex layouts
- TextExtractor for markdown and plain text files (no OCR needed)
- DocxExtractor for DOCX files (no OCR needed)
"""

from translate_docs_ai.ocr.base import OCRProvider, OCRResult
from translate_docs_ai.ocr.deepinfra import DeepInfraOCR
from translate_docs_ai.ocr.docx import DocxExtractor
from translate_docs_ai.ocr.pymupdf import PyMuPDFExtractor
from translate_docs_ai.ocr.text import TextExtractor

__all__ = [
    "OCRResult",
    "OCRProvider",
    "PyMuPDFExtractor",
    "DeepInfraOCR",
    "TextExtractor",
    "DocxExtractor",
]
