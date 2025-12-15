"""
Terminology extraction and management for translate-docs-ai.

Provides:
- LLM-based terminology extraction (primary)
- Frequency-based keyword extraction (fallback)
- Glossary page detection and extraction (highest priority)
- Semantic similarity using embeddings and VSS
- Term translation management
"""

from translate_docs_ai.terminology.embeddings import EmbeddingGenerator
from translate_docs_ai.terminology.extractor import TerminologyExtractor
from translate_docs_ai.terminology.glossary_detector import GlossaryDetector
from translate_docs_ai.terminology.llm_extractor import LLMTerminologyExtractor

__all__ = [
    "TerminologyExtractor",
    "LLMTerminologyExtractor",
    "EmbeddingGenerator",
    "GlossaryDetector",
]
