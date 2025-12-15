"""
translate-docs-ai: AI-powered document translation pipeline.

This package provides tools for:
- Document digitization using modern OCR models (DeepSeek-OCR, olmOCR-2)
- Terminology extraction and management with DuckDB
- Context-aware page-by-page translation
- Both automatic and semi-automatic processing modes
"""

__version__ = "0.1.0"
__author__ = "yharby"

from translate_docs_ai.config import ProcessingMode, Settings, load_config
from translate_docs_ai.database import Database, Document, Page, Stage, Status, Term
from translate_docs_ai.ocr import DeepInfraOCR, OCRResult, PyMuPDFExtractor
from translate_docs_ai.scanner import DocumentScanner
from translate_docs_ai.terminology import EmbeddingGenerator, TerminologyExtractor
from translate_docs_ai.translation import PageTranslator, TranslationPipeline

__all__ = [
    # Config
    "Settings",
    "load_config",
    "ProcessingMode",
    # Database
    "Database",
    "Document",
    "Page",
    "Term",
    "Status",
    "Stage",
    # Scanner
    "DocumentScanner",
    # OCR
    "OCRResult",
    "PyMuPDFExtractor",
    "DeepInfraOCR",
    # Terminology
    "TerminologyExtractor",
    "EmbeddingGenerator",
    # Translation
    "PageTranslator",
    "TranslationPipeline",
]
