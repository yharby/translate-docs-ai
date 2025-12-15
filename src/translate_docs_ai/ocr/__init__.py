"""
OCR pipeline for translate-docs-ai.

Provides multiple OCR backends:
- PyMuPDF4LLM for native PDF text extraction
- olmOCR-2 via DeepInfra for scanned documents
- DeepSeek-OCR via DeepInfra for complex layouts
"""

from translate_docs_ai.ocr.base import OCRProvider, OCRResult
from translate_docs_ai.ocr.deepinfra import DeepInfraOCR
from translate_docs_ai.ocr.pymupdf import PyMuPDFExtractor

__all__ = ["OCRResult", "OCRProvider", "PyMuPDFExtractor", "DeepInfraOCR"]
