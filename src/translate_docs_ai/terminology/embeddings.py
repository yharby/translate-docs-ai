"""
Embedding generation for terminology using sentence-transformers.

Provides semantic similarity capabilities for term clustering and matching.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from translate_docs_ai.database import Database, Term


class EmbeddingGenerator:
    """
    Generate embeddings for terminology using sentence-transformers.

    Uses multilingual models to support Arabic, English, and French.
    """

    # Recommended models for multilingual support
    MODELS = {
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "arabic": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "fast": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(
        self,
        db: Database,
        model_name: str = "multilingual",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize embedding generator.

        Args:
            db: Database instance.
            model_name: Model key or full model name.
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto).
            batch_size: Batch size for encoding.
        """
        self.db = db
        self.batch_size = batch_size

        # Resolve model name
        model = self.MODELS.get(model_name, model_name)
        self._model = SentenceTransformer(model, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text], normalize)[0]

    async def generate_term_embeddings(
        self,
        document_id: int,
        include_context: bool = True,
    ) -> int:
        """
        Generate embeddings for all terms in a document.

        Args:
            document_id: Document ID.
            include_context: Include context in embedding text.

        Returns:
            Number of terms with embeddings generated.
        """
        terms = self.db.get_document_terms(document_id)
        if not terms:
            return 0

        # Prepare texts for encoding
        texts = []
        for term in terms:
            if include_context and term.context:
                # Combine term with context for richer embedding
                text = f"{term.term}: {term.context}"
            else:
                text = term.term
            texts.append(text)

        # Generate embeddings (run in thread pool for async)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.encode(texts))

        # Store embeddings in database
        count = 0
        for term, embedding in zip(terms, embeddings, strict=True):
            if term.id:
                self._store_term_embedding(term.id, embedding)
                count += 1

        # Log generation
        self.db.log(
            level="INFO",
            stage="terminology_embed",
            message=f"Generated embeddings for {count} terms",
            document_id=document_id,
            context={"dimension": self._dimension},
        )

        return count

    def _store_term_embedding(self, term_id: int, embedding: np.ndarray) -> None:
        """Store embedding for a term in DuckDB."""
        # Convert to list for storage
        embedding_list = embedding.tolist()

        self.db.conn.execute(
            """
            UPDATE terminology
            SET embedding = ?
            WHERE id = ?
            """,
            [embedding_list, term_id],
        )

    def find_similar_terms(
        self,
        query: str,
        document_id: int | None = None,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[Term, float]]:
        """
        Find terms similar to a query.

        Args:
            query: Query text.
            document_id: Limit to specific document (optional).
            top_k: Maximum results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (Term, similarity_score) tuples.
        """
        # Encode query
        query_embedding = self.encode_single(query)

        # Build query based on whether we filter by document
        if document_id is not None:
            sql = """
                SELECT
                    id, document_id, term, frequency, context,
                    translation_ar, translation_en, translation_fr, approved,
                    list_cosine_similarity(embedding, ?) as similarity
                FROM terminology
                WHERE document_id = ?
                AND embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ?
            """
            params = [query_embedding.tolist(), document_id, top_k]
        else:
            sql = """
                SELECT
                    id, document_id, term, frequency, context,
                    translation_ar, translation_en, translation_fr, approved,
                    list_cosine_similarity(embedding, ?) as similarity
                FROM terminology
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ?
            """
            params = [query_embedding.tolist(), top_k]

        results = self.db.conn.execute(sql, params).fetchall()

        # Filter by threshold and convert to Term objects
        similar_terms = []
        for row in results:
            similarity = row[9]
            if similarity >= threshold:
                term = Term(
                    id=row[0],
                    document_id=row[1],
                    term=row[2],
                    frequency=row[3],
                    context=row[4],
                    translation_ar=row[5],
                    translation_en=row[6],
                    translation_fr=row[7],
                    approved=row[8],
                )
                similar_terms.append((term, similarity))

        return similar_terms

    def cluster_similar_terms(
        self,
        document_id: int,
        threshold: float = 0.8,
    ) -> list[list[Term]]:
        """
        Cluster similar terms together.

        Uses a simple greedy clustering based on cosine similarity.

        Args:
            document_id: Document ID.
            threshold: Similarity threshold for clustering.

        Returns:
            List of term clusters (each cluster is a list of similar Terms).
        """
        terms = self.db.get_document_terms(document_id)
        if not terms:
            return []

        # Get embeddings
        embeddings_data = self.db.conn.execute(
            """
            SELECT id, embedding
            FROM terminology
            WHERE document_id = ?
            AND embedding IS NOT NULL
            """,
            [document_id],
        ).fetchall()

        if not embeddings_data:
            return []

        # Build id to embedding mapping
        id_to_embedding: dict[int, np.ndarray] = {}
        for term_id, embedding in embeddings_data:
            if embedding:
                id_to_embedding[term_id] = np.array(embedding)

        # Build id to term mapping
        id_to_term = {t.id: t for t in terms if t.id}

        # Greedy clustering
        clustered: set[int] = set()
        clusters: list[list[Term]] = []

        for term_id, embedding in id_to_embedding.items():
            if term_id in clustered:
                continue

            # Start new cluster
            cluster = [id_to_term[term_id]]
            clustered.add(term_id)

            # Find similar terms
            for other_id, other_embedding in id_to_embedding.items():
                if other_id in clustered:
                    continue

                # Compute cosine similarity
                similarity = np.dot(embedding, other_embedding)

                if similarity >= threshold:
                    cluster.append(id_to_term[other_id])
                    clustered.add(other_id)

            clusters.append(cluster)

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        return clusters

    def suggest_translations(
        self,
        term: Term,
        source_lang: str = "en",
        target_lang: str = "ar",
    ) -> list[tuple[str, float]]:
        """
        Suggest translations based on similar terms that have translations.

        Args:
            term: Term to find translations for.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of (translation, confidence) tuples.
        """
        # Find similar terms across all documents
        similar = self.find_similar_terms(
            term.term,
            document_id=None,
            top_k=20,
            threshold=0.7,
        )

        # Get translation field based on target language
        translation_field = f"translation_{target_lang}"

        suggestions: dict[str, float] = {}
        for similar_term, similarity in similar:
            translation = getattr(similar_term, translation_field, None)
            if translation and similar_term.approved:
                # Weight by similarity and approval status
                if translation in suggestions:
                    suggestions[translation] = max(suggestions[translation], similarity)
                else:
                    suggestions[translation] = similarity

        # Sort by confidence
        sorted_suggestions = sorted(
            suggestions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_suggestions[:5]

    def get_embedding_stats(self, document_id: int | None = None) -> dict[str, Any]:
        """
        Get statistics about embeddings.

        Args:
            document_id: Limit to specific document (optional).

        Returns:
            Dictionary with embedding statistics.
        """
        if document_id is not None:
            where_clause = "WHERE document_id = ?"
            params = [document_id]
        else:
            where_clause = ""
            params = []

        # Count total and embedded terms
        total = self.db.conn.execute(
            f"SELECT COUNT(*) FROM terminology {where_clause}",
            params,
        ).fetchone()[0]

        embedded = self.db.conn.execute(
            f"""
            SELECT COUNT(*) FROM terminology
            {where_clause}
            {"AND" if where_clause else "WHERE"} embedding IS NOT NULL
            """,
            params,
        ).fetchone()[0]

        return {
            "total_terms": total,
            "embedded_terms": embedded,
            "embedding_coverage": embedded / total if total > 0 else 0.0,
            "embedding_dimension": self._dimension,
            "model": self._model.get_config_dict().get("model_name_or_path", "unknown"),
        }
