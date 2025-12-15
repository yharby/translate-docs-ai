"""
DuckDB-backed long-term memory store for LangGraph.

Implements the LangGraph BaseStore interface using DuckDB for persistence,
with optional semantic search capabilities via embeddings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from translate_docs_ai.database import Database
    from translate_docs_ai.terminology.embeddings import EmbeddingGenerator


@dataclass
class Item:
    """A memory item stored in DuckDB."""

    value: dict[str, Any]
    key: str
    namespace: tuple[str, ...]
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DuckDBStore:
    """
    DuckDB-backed store implementing LangGraph's memory store interface.

    Provides persistent storage for long-term memory across conversation threads,
    with support for:
    - Hierarchical namespace organization
    - JSON value storage
    - Optional semantic search via embeddings
    - Efficient prefix-based key iteration

    This is compatible with LangGraph's BaseStore interface but uses DuckDB
    for persistence instead of in-memory or PostgreSQL.
    """

    def __init__(
        self,
        db: Database,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Initialize DuckDB store.

        Args:
            db: Database instance with memory_store table.
            embedding_generator: Optional embedding generator for semantic search.
        """
        self.db = db
        self.embedding_generator = embedding_generator

    def _namespace_to_str(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to string for storage."""
        return "/".join(namespace)

    def _str_to_namespace(self, namespace_str: str) -> tuple[str, ...]:
        """Convert stored string back to namespace tuple."""
        return tuple(namespace_str.split("/"))

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        index: bool = True,
    ) -> None:
        """
        Store a memory item.

        Args:
            namespace: Hierarchical namespace (e.g., ("memories", "user_123")).
            key: Unique key within namespace.
            value: JSON-serializable dictionary to store.
            index: Whether to generate embedding for semantic search.
        """
        namespace_str = self._namespace_to_str(namespace)
        value_json = json.dumps(value)

        # Generate embedding if enabled
        embedding_sql = "NULL"
        if index and self.embedding_generator:
            # Create text representation for embedding
            text_for_embedding = self._value_to_text(value)
            if text_for_embedding:
                embedding = self.embedding_generator.encode_single(text_for_embedding)
                embedding_sql = f"[{','.join(map(str, embedding.tolist()))}]::FLOAT[]"

        self.db.conn.execute(
            f"""
            INSERT INTO memory_store (id, namespace, key, value, embedding)
            VALUES (nextval('memory_store_id_seq'), ?, ?, ?::JSON, {embedding_sql})
            ON CONFLICT (namespace, key) DO UPDATE
            SET value = EXCLUDED.value,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            """,
            [namespace_str, key, value_json],
        )

    def get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        """
        Retrieve a memory item by namespace and key.

        Args:
            namespace: Hierarchical namespace.
            key: Unique key within namespace.

        Returns:
            Item if found, None otherwise.
        """
        namespace_str = self._namespace_to_str(namespace)

        row = self.db.conn.execute(
            """
            SELECT namespace, key, value, created_at, updated_at
            FROM memory_store
            WHERE namespace = ? AND key = ?
            """,
            [namespace_str, key],
        ).fetchone()

        if row:
            return Item(
                namespace=self._str_to_namespace(row[0]),
                key=row[1],
                value=json.loads(row[2]) if isinstance(row[2], str) else row[2],
                created_at=row[3],
                updated_at=row[4],
            )
        return None

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """
        Delete a memory item.

        Args:
            namespace: Hierarchical namespace.
            key: Unique key within namespace.
        """
        namespace_str = self._namespace_to_str(namespace)
        self.db.conn.execute(
            "DELETE FROM memory_store WHERE namespace = ? AND key = ?",
            [namespace_str, key],
        )

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        """
        Search for memory items in a namespace.

        Args:
            namespace: Hierarchical namespace to search within.
            query: Optional semantic search query (requires embedding_generator).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching Items, ordered by relevance if query provided.
        """
        namespace_str = self._namespace_to_str(namespace)

        if query and self.embedding_generator:
            # Semantic search
            query_embedding = self.embedding_generator.encode_single(query)
            embedding_str = f"[{','.join(map(str, query_embedding.tolist()))}]::FLOAT[]"

            rows = self.db.conn.execute(
                f"""
                SELECT namespace, key, value, created_at, updated_at,
                       array_cosine_similarity(embedding, {embedding_str}) AS similarity
                FROM memory_store
                WHERE namespace = ? AND embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ? OFFSET ?
                """,
                [namespace_str, limit, offset],
            ).fetchall()
        else:
            # Simple retrieval by namespace
            rows = self.db.conn.execute(
                """
                SELECT namespace, key, value, created_at, updated_at
                FROM memory_store
                WHERE namespace = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                [namespace_str, limit, offset],
            ).fetchall()

        return [
            Item(
                namespace=self._str_to_namespace(row[0]),
                key=row[1],
                value=json.loads(row[2]) if isinstance(row[2], str) else row[2],
                created_at=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    def list_namespaces(self, prefix: tuple[str, ...] | None = None) -> list[tuple[str, ...]]:
        """
        List all unique namespaces, optionally filtered by prefix.

        Args:
            prefix: Optional namespace prefix to filter by.

        Returns:
            List of unique namespace tuples.
        """
        if prefix:
            prefix_str = self._namespace_to_str(prefix)
            rows = self.db.conn.execute(
                "SELECT DISTINCT namespace FROM memory_store WHERE namespace LIKE ?",
                [f"{prefix_str}%"],
            ).fetchall()
        else:
            rows = self.db.conn.execute("SELECT DISTINCT namespace FROM memory_store").fetchall()

        return [self._str_to_namespace(row[0]) for row in rows]

    def yield_keys(
        self,
        namespace: tuple[str, ...],
        *,
        prefix: str | None = None,
    ) -> Iterator[str]:
        """
        Yield all keys in a namespace.

        Args:
            namespace: Namespace to list keys from.
            prefix: Optional key prefix filter.

        Yields:
            Keys matching the criteria.
        """
        namespace_str = self._namespace_to_str(namespace)

        if prefix:
            rows = self.db.conn.execute(
                "SELECT key FROM memory_store WHERE namespace = ? AND key LIKE ?",
                [namespace_str, f"{prefix}%"],
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT key FROM memory_store WHERE namespace = ?",
                [namespace_str],
            ).fetchall()

        for row in rows:
            yield row[0]

    def _value_to_text(self, value: dict[str, Any]) -> str:
        """
        Convert a value dict to text for embedding generation.

        Prioritizes 'content' and 'text' fields, falls back to JSON serialization.
        """
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
        if "summary" in value:
            return str(value["summary"])
        # Fall back to JSON representation (limited)
        return json.dumps(value)[:1000]

    # ==================== Batch Operations (LangChain BaseStore compatible) ====================

    def mget(self, keys: list[tuple[tuple[str, ...], str]]) -> list[Item | None]:
        """
        Get multiple items by their (namespace, key) pairs.

        Args:
            keys: List of (namespace, key) tuples.

        Returns:
            List of Items or None for each key.
        """
        return [self.get(namespace, key) for namespace, key in keys]

    def mset(self, items: list[tuple[tuple[str, ...], str, dict[str, Any]]]) -> None:
        """
        Set multiple items.

        Args:
            items: List of (namespace, key, value) tuples.
        """
        for namespace, key, value in items:
            self.put(namespace, key, value)

    def mdelete(self, keys: list[tuple[tuple[str, ...], str]]) -> None:
        """
        Delete multiple items.

        Args:
            keys: List of (namespace, key) tuples.
        """
        for namespace, key in keys:
            self.delete(namespace, key)

    # ==================== Document Context Methods ====================

    def store_page_summary(
        self,
        document_id: int,
        page_number: int,
        summary: str,
        key_entities: list[str] | None = None,
    ) -> None:
        """
        Store a page summary for context building.

        Args:
            document_id: Document ID.
            page_number: Page number.
            summary: Brief summary of the page content.
            key_entities: Optional list of key entities/terms found.
        """
        namespace = ("documents", str(document_id), "pages")
        key = f"page_{page_number}"
        value = {
            "summary": summary,
            "page_number": page_number,
            "key_entities": key_entities or [],
        }
        self.put(namespace, key, value)

    def get_document_context(
        self,
        document_id: int,
        current_page: int,
        context_window: int = 3,
    ) -> dict[str, Any]:
        """
        Get context for translation including nearby page summaries.

        Args:
            document_id: Document ID.
            current_page: Current page being translated.
            context_window: Number of pages before/after to include.

        Returns:
            Dictionary with page summaries and context.
        """
        namespace = ("documents", str(document_id), "pages")

        # Get summaries for pages in the context window
        context_pages = []
        for page_num in range(
            max(0, current_page - context_window),
            current_page + context_window + 1,
        ):
            if page_num == current_page:
                continue
            item = self.get(namespace, f"page_{page_num}")
            if item:
                context_pages.append(item.value)

        # Get document-level glossary
        glossary_namespace = ("documents", str(document_id), "glossary")
        glossary_items = self.search(glossary_namespace, limit=100)
        glossary = {item.key: item.value.get("translation", "") for item in glossary_items}

        return {
            "nearby_pages": context_pages,
            "glossary": glossary,
            "current_page": current_page,
        }

    def store_translation_decision(
        self,
        document_id: int,
        term: str,
        translation: str,
        context: str | None = None,
    ) -> None:
        """
        Store a translation decision for consistency.

        Args:
            document_id: Document ID.
            term: Original term.
            translation: Chosen translation.
            context: Optional context for the decision.
        """
        namespace = ("documents", str(document_id), "glossary")
        value = {
            "term": term,
            "translation": translation,
            "context": context,
        }
        self.put(namespace, term, value)

    def get_translation_memory(
        self,
        document_id: int,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, str]:
        """
        Get translation memory (term -> translation mapping).

        Args:
            document_id: Document ID.
            query: Optional semantic search query.
            limit: Maximum number of entries.

        Returns:
            Dictionary mapping terms to translations.
        """
        namespace = ("documents", str(document_id), "glossary")
        items = self.search(namespace, query=query, limit=limit)
        return {
            item.value.get("term", item.key): item.value.get("translation", "")
            for item in items
            if item.value.get("translation")
        }

    def clear_document_memory(self, document_id: int) -> None:
        """
        Clear all memory for a specific document.

        Args:
            document_id: Document ID.
        """
        doc_prefix = f"documents/{document_id}"
        self.db.conn.execute(
            "DELETE FROM memory_store WHERE namespace LIKE ?",
            [f"{doc_prefix}%"],
        )

    def get_memory_stats(self, document_id: int | None = None) -> dict[str, Any]:
        """
        Get memory usage statistics.

        Args:
            document_id: Optional document ID to filter stats.

        Returns:
            Dictionary with memory statistics.
        """
        if document_id:
            prefix = f"documents/{document_id}"
            total_count = self.db.conn.execute(
                "SELECT COUNT(*) FROM memory_store WHERE namespace LIKE ?",
                [f"{prefix}%"],
            ).fetchone()[0]
            with_embeddings = self.db.conn.execute(
                "SELECT COUNT(*) FROM memory_store WHERE namespace LIKE ? AND embedding IS NOT NULL",
                [f"{prefix}%"],
            ).fetchone()[0]
        else:
            total_count = self.db.conn.execute("SELECT COUNT(*) FROM memory_store").fetchone()[0]
            with_embeddings = self.db.conn.execute(
                "SELECT COUNT(*) FROM memory_store WHERE embedding IS NOT NULL"
            ).fetchone()[0]

        namespaces = self.list_namespaces()

        return {
            "total_items": total_count,
            "items_with_embeddings": with_embeddings,
            "unique_namespaces": len(namespaces),
            "document_id": document_id,
        }
