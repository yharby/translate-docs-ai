"""
LangGraph-compatible search tools for terminology lookup.

Provides FTS (Full-Text Search) and VSS (Vector Similarity Search) tools
that can be used by LLMs during translation for context-aware terminology lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from translate_docs_ai.database import Database
    from translate_docs_ai.terminology.embeddings import EmbeddingGenerator


@dataclass
class SearchResult:
    """Result from a terminology search."""

    term: str
    translation: str | None
    frequency: int
    context: str | None
    score: float
    source: str  # 'fts', 'semantic', or 'hybrid'


class TerminologySearchTools:
    """
    Search tools for terminology lookup during translation.

    Provides both FTS and semantic search capabilities that can be
    exposed to LLMs via tool calling or used directly in the pipeline.
    """

    def __init__(
        self,
        db: Database,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Initialize search tools.

        Args:
            db: Database instance with search methods.
            embedding_generator: Optional embedding generator for semantic search.
        """
        self.db = db
        self.embedding_generator = embedding_generator
        self._search_indexes_initialized = False

    def initialize_search(self) -> None:
        """Initialize search indexes if not already done."""
        if not self._search_indexes_initialized:
            try:
                self.db.setup_search_indexes()
                self._search_indexes_initialized = True
            except Exception:
                # FTS/VSS may not be available
                pass

    def search_terms(
        self,
        query: str,
        document_id: int | None = None,
        target_lang: str = "en",
        limit: int = 10,
        search_type: str = "hybrid",
    ) -> list[SearchResult]:
        """
        Search for terms matching a query.

        Args:
            query: Search query (term or phrase).
            document_id: Optional document ID to scope search.
            target_lang: Target language for translations.
            limit: Maximum results to return.
            search_type: Type of search ('fts', 'semantic', or 'hybrid').

        Returns:
            List of SearchResult objects.
        """
        self.initialize_search()

        # Get translation column
        translation_key = f"translation_{target_lang}"

        results: list[SearchResult] = []

        if search_type == "fts":
            fts_results = self.db.search_terms_fts(query, document_id, limit)
            for r in fts_results:
                results.append(
                    SearchResult(
                        term=r["term"],
                        translation=r.get(translation_key),
                        frequency=r["frequency"],
                        context=r.get("context"),
                        score=r.get("score", 0.0),
                        source="fts",
                    )
                )

        elif search_type == "semantic" and self.embedding_generator:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_single(query).tolist()
            semantic_results = self.db.search_terms_semantic(query_embedding, document_id, limit)
            for r in semantic_results:
                results.append(
                    SearchResult(
                        term=r["term"],
                        translation=r.get(translation_key),
                        frequency=r["frequency"],
                        context=r.get("context"),
                        score=r.get("similarity", 0.0),
                        source="semantic",
                    )
                )

        elif search_type == "hybrid":
            # Generate query embedding if available
            query_embedding = None
            if self.embedding_generator:
                query_embedding = self.embedding_generator.encode_single(query).tolist()

            hybrid_results = self.db.search_terms_hybrid(query, query_embedding, document_id, limit)
            for r in hybrid_results:
                results.append(
                    SearchResult(
                        term=r["term"],
                        translation=r.get(translation_key),
                        frequency=r["frequency"],
                        context=r.get("context"),
                        score=r.get("combined_score", 0.0),
                        source="hybrid",
                    )
                )

        return results

    def get_glossary(
        self,
        document_id: int,
        target_lang: str = "en",
    ) -> dict[str, str]:
        """
        Get the complete term glossary for a document.

        Args:
            document_id: Document ID.
            target_lang: Target language for translations.

        Returns:
            Dictionary mapping terms to translations.
        """
        return self.db.get_term_glossary(document_id, target_lang)

    def lookup_term(
        self,
        term: str,
        document_id: int,
        target_lang: str = "en",
    ) -> str | None:
        """
        Look up a specific term's translation.

        Args:
            term: Term to look up.
            document_id: Document ID.
            target_lang: Target language.

        Returns:
            Translation if found, None otherwise.
        """
        glossary = self.get_glossary(document_id, target_lang)
        return glossary.get(term)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for LLM tool calling.

        Returns:
            List of tool definitions that can be passed to LLM APIs.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_terminology",
                    "description": "Search the terminology database for terms related to a query. Use this to find translations of technical terms, domain-specific vocabulary, or to check if a term has been translated before.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The term or phrase to search for",
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["fts", "semantic", "hybrid"],
                                "description": "Type of search: 'fts' for exact/keyword match, 'semantic' for meaning-based, 'hybrid' for both",
                                "default": "hybrid",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_term",
                    "description": "Look up a specific term's translation from the glossary. Use this when you know the exact term and want its translation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The exact term to look up",
                            },
                        },
                        "required": ["term"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_glossary",
                    "description": "Get all terms and their translations for the current document. Use this to see the complete terminology list.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        document_id: int,
        target_lang: str = "en",
    ) -> dict[str, Any]:
        """
        Execute a tool call from an LLM.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            document_id: Current document ID.
            target_lang: Target language.

        Returns:
            Tool execution result.
        """
        if tool_name == "search_terminology":
            results = self.search_terms(
                query=arguments["query"],
                document_id=document_id,
                target_lang=target_lang,
                limit=arguments.get("limit", 5),
                search_type=arguments.get("search_type", "hybrid"),
            )
            return {
                "results": [
                    {
                        "term": r.term,
                        "translation": r.translation,
                        "frequency": r.frequency,
                        "context": r.context,
                        "score": r.score,
                    }
                    for r in results
                ]
            }

        elif tool_name == "lookup_term":
            translation = self.lookup_term(
                term=arguments["term"],
                document_id=document_id,
                target_lang=target_lang,
            )
            return {
                "term": arguments["term"],
                "translation": translation,
                "found": translation is not None,
            }

        elif tool_name == "get_document_glossary":
            glossary = self.get_glossary(document_id, target_lang)
            return {
                "glossary": glossary,
                "total_terms": len(glossary),
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}


def create_search_tools(
    db: Database,
    embedding_generator: EmbeddingGenerator | None = None,
) -> TerminologySearchTools:
    """
    Factory function to create search tools instance.

    Args:
        db: Database instance.
        embedding_generator: Optional embedding generator.

    Returns:
        Configured TerminologySearchTools instance.
    """
    return TerminologySearchTools(db, embedding_generator)
