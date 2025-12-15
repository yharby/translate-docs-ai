"""
Terminology extraction and management for translate-docs-ai.

Provides:
- Keyword extraction using DuckDB FTS
- Semantic similarity using embeddings and VSS
- Term translation management
"""

from translate_docs_ai.terminology.embeddings import EmbeddingGenerator
from translate_docs_ai.terminology.extractor import TerminologyExtractor

__all__ = ["TerminologyExtractor", "EmbeddingGenerator"]
