"""
DuckDB database operations for translate-docs-ai.

Handles document catalog, page content, terminology, checkpoints, and logging.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb


class Stage(str, Enum):
    """Processing stages."""

    SCAN = "scan"
    OCR = "ocr"
    TERMINOLOGY_EXTRACT = "terminology_extract"
    TERMINOLOGY_TRANSLATE = "terminology_translate"
    PAGE_TRANSLATE = "page_translate"
    EXPORT = "export"


class Status(str, Enum):
    """Processing status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    PAUSED = "paused"


@dataclass
class Document:
    """Document record."""

    id: int | None = None
    file_name: str = ""
    full_path: str = ""
    relative_path: str = ""
    file_type: str = ""
    total_pages: int = 0
    status: Status = Status.PENDING
    styling_metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Page:
    """Page record."""

    id: int | None = None
    document_id: int = 0
    page_number: int = 0
    original_content: str = ""
    ar_content: str | None = None
    en_content: str | None = None
    fr_content: str | None = None
    embedding: list[float] | None = None
    ocr_model: str | None = None
    ocr_confidence: float | None = None
    created_at: datetime | None = None

    @property
    def translated_content(self) -> str | None:
        """Get the first available translated content."""
        return self.en_content or self.ar_content or self.fr_content


@dataclass
class Term:
    """Terminology record."""

    id: int | None = None
    document_id: int = 0
    term: str = ""
    frequency: int = 0
    translation_ar: str | None = None
    translation_en: str | None = None
    translation_fr: str | None = None
    context: str | None = None
    embedding: list[float] | None = None
    approved: bool = False


@dataclass
class Checkpoint:
    """Checkpoint record."""

    run_id: str = ""
    document_id: int = 0
    stage: Stage = Stage.SCAN
    sub_stage: str | None = None
    status: Status = Status.PENDING
    items_total: int = 0
    items_completed: int = 0
    last_successful_item: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> int | None:
        """Alias for last_successful_item as page number."""
        if self.last_successful_item is not None:
            try:
                return int(self.last_successful_item)
            except (ValueError, TypeError):
                return None
        return None

    @property
    def state_data(self) -> dict[str, Any]:
        """Alias for metadata."""
        return self.metadata


class Database:
    """DuckDB database wrapper for translate-docs-ai."""

    # SQL for creating tables
    _SCHEMA = """
    -- Main documents table
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        file_name VARCHAR NOT NULL,
        full_path VARCHAR NOT NULL UNIQUE,
        relative_path VARCHAR NOT NULL,
        file_type VARCHAR,
        total_pages INTEGER DEFAULT 0,
        status VARCHAR DEFAULT 'pending',
        styling_metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create sequence for documents if not exists
    CREATE SEQUENCE IF NOT EXISTS documents_id_seq START 1;

    -- Page-level content table
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL,
        page_number INTEGER NOT NULL,
        original_content TEXT,
        ar_content TEXT,
        en_content TEXT,
        fr_content TEXT,
        embedding FLOAT[],
        ocr_model VARCHAR,
        ocr_confidence FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(document_id, page_number)
    );

    -- Create sequence for pages if not exists
    CREATE SEQUENCE IF NOT EXISTS pages_id_seq START 1;

    -- Terminology extraction table
    CREATE TABLE IF NOT EXISTS terminology (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL,
        term VARCHAR NOT NULL,
        frequency INTEGER DEFAULT 1,
        translation_ar VARCHAR,
        translation_en VARCHAR,
        translation_fr VARCHAR,
        context TEXT,
        embedding FLOAT[],
        approved BOOLEAN DEFAULT FALSE,
        UNIQUE(document_id, term)
    );

    -- Create sequence for terminology if not exists
    CREATE SEQUENCE IF NOT EXISTS terminology_id_seq START 1;

    -- Processing state/checkpoints table
    CREATE TABLE IF NOT EXISTS processing_state (
        id INTEGER PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        document_id INTEGER NOT NULL,
        stage VARCHAR NOT NULL,
        sub_stage VARCHAR,
        status VARCHAR NOT NULL DEFAULT 'pending',
        progress_percent FLOAT,
        items_total INTEGER,
        items_completed INTEGER DEFAULT 0,
        last_successful_item VARCHAR,
        retry_count INTEGER DEFAULT 0,
        error_type VARCHAR,
        error_message TEXT,
        error_traceback TEXT,
        started_at TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        metadata JSON
    );

    -- Create sequence for processing_state if not exists
    CREATE SEQUENCE IF NOT EXISTS processing_state_id_seq START 1;

    -- Processing log for audit trail
    CREATE TABLE IF NOT EXISTS processing_log (
        id INTEGER PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        document_id INTEGER,
        stage VARCHAR NOT NULL,
        level VARCHAR NOT NULL,
        message TEXT NOT NULL,
        context JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create sequence for processing_log if not exists
    CREATE SEQUENCE IF NOT EXISTS processing_log_id_seq START 1;

    -- Long-term memory store for LangGraph
    CREATE TABLE IF NOT EXISTS memory_store (
        id INTEGER PRIMARY KEY,
        namespace VARCHAR NOT NULL,
        key VARCHAR NOT NULL,
        value JSON NOT NULL,
        embedding FLOAT[],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(namespace, key)
    );

    -- Create sequence for memory_store if not exists
    CREATE SEQUENCE IF NOT EXISTS memory_store_id_seq START 1;

    -- Create indexes for efficient queries
    CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
    CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
    CREATE INDEX IF NOT EXISTS idx_terminology_document ON terminology(document_id);
    CREATE INDEX IF NOT EXISTS idx_state_resume ON processing_state(document_id, stage, status);
    CREATE INDEX IF NOT EXISTS idx_log_lookup ON processing_log(run_id, stage, level);
    CREATE INDEX IF NOT EXISTS idx_memory_namespace ON memory_store(namespace);
    """

    def __init__(self, db_path: Path | str):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._run_id = str(uuid.uuid4())

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._init_schema()
        return self._conn

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self._run_id

    def new_run(self) -> str:
        """Start a new run and return its ID."""
        self._run_id = str(uuid.uuid4())
        return self._run_id

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute(self._SCHEMA)
        # Migrate existing databases: add styling_metadata column if missing
        self._migrate_add_styling_column()

    def _migrate_add_styling_column(self) -> None:
        """Add styling_metadata column to documents table if it doesn't exist."""
        try:
            # Check if column exists by querying table info
            result = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'documents' AND column_name = 'styling_metadata'"
            ).fetchone()
            if not result:
                # Column doesn't exist, add it
                self.conn.execute("ALTER TABLE documents ADD COLUMN styling_metadata JSON")
        except Exception:
            # If anything fails, the column either exists or table doesn't exist yet
            pass

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for transactions."""
        try:
            self.conn.begin()
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ==================== Documents ====================

    def add_document(self, doc: Document) -> int:
        """Add a document to the catalog."""
        result = self.conn.execute(
            """
            INSERT INTO documents (id, file_name, full_path, relative_path, file_type, total_pages, status)
            VALUES (nextval('documents_id_seq'), ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                doc.file_name,
                doc.full_path,
                doc.relative_path,
                doc.file_type,
                doc.total_pages,
                doc.status.value,
            ],
        ).fetchone()
        return result[0] if result else 0

    def get_document(self, doc_id: int) -> Document | None:
        """Get a document by ID."""
        row = self.conn.execute("SELECT * FROM documents WHERE id = ?", [doc_id]).fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_document_by_path(self, full_path: str) -> Document | None:
        """Get a document by its full path."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE full_path = ?", [full_path]
        ).fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_all_documents(self, status: Status | None = None) -> list[Document]:
        """Get all documents, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM documents WHERE status = ? ORDER BY id",
                [status.value],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [self._row_to_document(row) for row in rows]

    def update_document_status(self, doc_id: int, status: Status) -> None:
        """Update document status."""
        self.conn.execute(
            "UPDATE documents SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            [status.value, doc_id],
        )

    def update_document_pages(self, doc_id: int, total_pages: int) -> None:
        """Update document total pages."""
        self.conn.execute(
            "UPDATE documents SET total_pages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            [total_pages, doc_id],
        )

    def set_document_styling(self, doc_id: int, styling_metadata: dict[str, Any]) -> None:
        """Store extracted styling metadata for a document."""
        styling_json = json.dumps(styling_metadata)
        self.conn.execute(
            "UPDATE documents SET styling_metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            [styling_json, doc_id],
        )

    def get_document_styling(self, doc_id: int) -> dict[str, Any] | None:
        """Get styling metadata for a document."""
        row = self.conn.execute(
            "SELECT styling_metadata FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        if row and row[0]:
            styling_data = row[0]
            if isinstance(styling_data, str):
                return json.loads(styling_data)
            return styling_data
        return None

    def _row_to_document(self, row: tuple) -> Document:
        """Convert database row to Document."""
        # Handle both old schema (9 columns) and new schema (10 columns with styling_metadata)
        if len(row) >= 10:
            # New schema with styling_metadata at index 7
            styling_data = row[7]
            if isinstance(styling_data, str):
                styling_data = json.loads(styling_data)
            return Document(
                id=row[0],
                file_name=row[1],
                full_path=row[2],
                relative_path=row[3],
                file_type=row[4],
                total_pages=row[5],
                status=Status(row[6]),
                styling_metadata=styling_data,
                created_at=row[8],
                updated_at=row[9],
            )
        else:
            # Old schema without styling_metadata
            return Document(
                id=row[0],
                file_name=row[1],
                full_path=row[2],
                relative_path=row[3],
                file_type=row[4],
                total_pages=row[5],
                status=Status(row[6]),
                styling_metadata=None,
                created_at=row[7],
                updated_at=row[8],
            )

    # ==================== Pages ====================

    def add_page(self, page: Page) -> int:
        """Add a page to the database."""
        embedding_sql = (
            f"[{','.join(map(str, page.embedding))}]::FLOAT[]" if page.embedding else "NULL"
        )

        result = self.conn.execute(
            f"""
            INSERT INTO pages (id, document_id, page_number, original_content, ocr_model, ocr_confidence, embedding)
            VALUES (nextval('pages_id_seq'), ?, ?, ?, ?, ?, {embedding_sql})
            ON CONFLICT (document_id, page_number) DO UPDATE
            SET original_content = EXCLUDED.original_content,
                ocr_model = EXCLUDED.ocr_model,
                ocr_confidence = EXCLUDED.ocr_confidence
            RETURNING id
            """,
            [
                page.document_id,
                page.page_number,
                page.original_content,
                page.ocr_model,
                page.ocr_confidence,
            ],
        ).fetchone()
        return result[0] if result else 0

    def get_page(self, doc_id: int, page_number: int) -> Page | None:
        """Get a specific page."""
        row = self.conn.execute(
            "SELECT * FROM pages WHERE document_id = ? AND page_number = ?",
            [doc_id, page_number],
        ).fetchone()
        if row:
            return self._row_to_page(row)
        return None

    def get_document_pages(self, doc_id: int) -> list[Page]:
        """Get all pages for a document."""
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
            [doc_id],
        ).fetchall()
        return [self._row_to_page(row) for row in rows]

    def update_page_translation(
        self, doc_id: int, page_number: int, language: str, content: str
    ) -> None:
        """Update page translation for a specific language."""
        column = f"{language}_content"
        if column not in ("ar_content", "en_content", "fr_content"):
            raise ValueError(f"Invalid language: {language}")

        self.conn.execute(
            f"UPDATE pages SET {column} = ? WHERE document_id = ? AND page_number = ?",
            [content, doc_id, page_number],
        )

    def update_page(
        self,
        page_id: int,
        translated_content: str | None = None,
        language: str = "en",
    ) -> None:
        """Update page with translated content."""
        column = f"{language}_content"
        if column not in ("ar_content", "en_content", "fr_content"):
            column = "en_content"  # Default to English

        self.conn.execute(
            f"UPDATE pages SET {column} = ? WHERE id = ?",
            [translated_content, page_id],
        )

    def _row_to_page(self, row: tuple) -> Page:
        """Convert database row to Page."""
        return Page(
            id=row[0],
            document_id=row[1],
            page_number=row[2],
            original_content=row[3],
            ar_content=row[4],
            en_content=row[5],
            fr_content=row[6],
            embedding=list(row[7]) if row[7] else None,
            ocr_model=row[8],
            ocr_confidence=row[9],
            created_at=row[10],
        )

    # ==================== Terminology ====================

    def add_term(self, term: Term) -> int:
        """Add or update a term."""
        embedding_sql = (
            f"[{','.join(map(str, term.embedding))}]::FLOAT[]" if term.embedding else "NULL"
        )

        result = self.conn.execute(
            f"""
            INSERT INTO terminology (id, document_id, term, frequency, context, embedding)
            VALUES (nextval('terminology_id_seq'), ?, ?, ?, ?, {embedding_sql})
            ON CONFLICT (document_id, term) DO UPDATE
            SET frequency = terminology.frequency + EXCLUDED.frequency,
                context = COALESCE(EXCLUDED.context, terminology.context)
            RETURNING id
            """,
            [term.document_id, term.term, term.frequency, term.context],
        ).fetchone()
        return result[0] if result else 0

    def get_document_terms(self, doc_id: int, approved_only: bool = False) -> list[Term]:
        """Get all terms for a document."""
        if approved_only:
            rows = self.conn.execute(
                "SELECT * FROM terminology WHERE document_id = ? AND approved = TRUE ORDER BY frequency DESC",
                [doc_id],
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM terminology WHERE document_id = ? ORDER BY frequency DESC",
                [doc_id],
            ).fetchall()
        return [self._row_to_term(row) for row in rows]

    def update_term_translation(self, term_id: int, language: str, translation: str) -> None:
        """Update term translation."""
        column = f"translation_{language}"
        if column not in ("translation_ar", "translation_en", "translation_fr"):
            raise ValueError(f"Invalid language: {language}")

        self.conn.execute(
            f"UPDATE terminology SET {column} = ? WHERE id = ?",
            [translation, term_id],
        )

    def approve_term(self, term_id: int, approved: bool = True) -> None:
        """Approve or unapprove a term."""
        self.conn.execute(
            "UPDATE terminology SET approved = ? WHERE id = ?",
            [approved, term_id],
        )

    def _row_to_term(self, row: tuple) -> Term:
        """Convert database row to Term."""
        return Term(
            id=row[0],
            document_id=row[1],
            term=row[2],
            frequency=row[3],
            translation_ar=row[4],
            translation_en=row[5],
            translation_fr=row[6],
            context=row[7],
            embedding=list(row[8]) if row[8] else None,
            approved=row[9],
        )

    # ==================== Checkpoints ====================

    def save_checkpoint(
        self,
        document_id: int,
        stage: Stage,
        page_number: int | None = None,
        state_data: dict | None = None,
        status: Status = Status.IN_PROGRESS,
    ) -> None:
        """Save or update a checkpoint."""
        metadata_json = json.dumps(state_data) if state_data else None

        self.conn.execute(
            """
            INSERT INTO processing_state
            (id, run_id, document_id, stage, sub_stage, status,
             items_total, items_completed, last_successful_item,
             error_type, error_message, metadata, updated_at)
            VALUES (nextval('processing_state_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                self._run_id,
                document_id,
                stage.value,
                None,  # sub_stage
                status.value,
                None,  # items_total
                None,  # items_completed
                page_number,  # last_successful_item
                None,  # error_type
                None,  # error_message
                metadata_json,
            ],
        )

    def get_latest_checkpoint(self, document_id: int, stage: Stage) -> Checkpoint | None:
        """Get the latest checkpoint for a document at a specific stage."""
        row = self.conn.execute(
            """
            SELECT run_id, document_id, stage, sub_stage, status,
                   items_total, items_completed, last_successful_item,
                   error_type, error_message, metadata
            FROM processing_state
            WHERE document_id = ? AND stage = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            [document_id, stage.value],
        ).fetchone()

        if row:
            return Checkpoint(
                run_id=row[0],
                document_id=row[1],
                stage=Stage(row[2]),
                sub_stage=row[3],
                status=Status(row[4]),
                items_total=row[5] or 0,
                items_completed=row[6] or 0,
                last_successful_item=row[7],
                error_type=row[8],
                error_message=row[9],
                metadata=json.loads(row[10]) if row[10] else {},
            )
        return None

    def get_incomplete_documents(self) -> list[tuple]:
        """Get all documents with incomplete processing."""
        return self.conn.execute(
            """
            SELECT d.id, d.file_name, ps.stage, ps.items_completed, ps.items_total,
                   ps.last_successful_item, ps.error_message
            FROM documents d
            JOIN processing_state ps ON d.id = ps.document_id
            WHERE ps.status IN ('in_progress', 'failed', 'retrying', 'paused')
            AND ps.updated_at = (
                SELECT MAX(updated_at) FROM processing_state
                WHERE document_id = d.id
            )
            ORDER BY d.id, ps.stage
            """
        ).fetchall()

    def get_checkpoint(self, document_id: int) -> Checkpoint | None:
        """Get the latest checkpoint for a document (any stage)."""
        row = self.conn.execute(
            """
            SELECT run_id, document_id, stage, sub_stage, status,
                   items_total, items_completed, last_successful_item,
                   error_type, error_message, metadata
            FROM processing_state
            WHERE document_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            [document_id],
        ).fetchone()

        if row:
            return Checkpoint(
                run_id=row[0],
                document_id=row[1],
                stage=Stage(row[2]),
                sub_stage=row[3],
                status=Status(row[4]),
                items_total=row[5] or 0,
                items_completed=row[6] or 0,
                last_successful_item=row[7],
                error_type=row[8],
                error_message=row[9],
                metadata=json.loads(row[10]) if row[10] else {},
            )
        return None

    # ==================== Logging ====================

    def log(
        self,
        level: str,
        stage: str,
        message: str,
        document_id: int | None = None,
        context: dict | None = None,
    ) -> None:
        """Insert a log entry."""
        context_json = json.dumps(context) if context else None
        self.conn.execute(
            """
            INSERT INTO processing_log
            (id, run_id, document_id, stage, level, message, context)
            VALUES (nextval('processing_log_id_seq'), ?, ?, ?, ?, ?, ?)
            """,
            [self._run_id, document_id, stage, level, message, context_json],
        )

    def get_logs(
        self,
        run_id: str | None = None,
        level: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get log entries."""
        conditions = []
        params = []

        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if level:
            conditions.append("level = ?")
            params.append(level)
        if stage:
            conditions.append("stage = ?")
            params.append(stage)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = self.conn.execute(
            f"""
            SELECT run_id, document_id, stage, level, message, context, created_at
            FROM processing_log
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()

        return [
            {
                "run_id": row[0],
                "document_id": row[1],
                "stage": row[2],
                "level": row[3],
                "message": row[4],
                "context": json.loads(row[5]) if row[5] else None,
                "created_at": row[6],
            }
            for row in rows
        ]

    # ==================== Statistics ====================

    def get_processing_stats(self, run_id: str | None = None) -> dict:
        """Get processing statistics."""
        run_filter = f"WHERE run_id = '{run_id}'" if run_id else ""

        stats = {}

        # Document counts by status
        doc_counts = self.conn.execute(
            "SELECT status, COUNT(*) FROM documents GROUP BY status"
        ).fetchall()
        stats["documents"] = {row[0]: row[1] for row in doc_counts}

        # Total pages
        total_pages = self.conn.execute("SELECT SUM(total_pages) FROM documents").fetchone()
        stats["total_pages"] = total_pages[0] or 0

        # Processed pages (with original content)
        processed_pages = self.conn.execute(
            "SELECT COUNT(*) FROM pages WHERE original_content IS NOT NULL"
        ).fetchone()
        stats["processed_pages"] = processed_pages[0] or 0

        # Translated pages
        translated_pages = self.conn.execute(
            "SELECT COUNT(*) FROM pages WHERE ar_content IS NOT NULL OR en_content IS NOT NULL OR fr_content IS NOT NULL"
        ).fetchone()
        stats["translated_pages"] = translated_pages[0] or 0

        # Terms
        terms = self.conn.execute("SELECT COUNT(*) FROM terminology").fetchone()
        stats["total_terms"] = terms[0] or 0

        approved_terms = self.conn.execute(
            "SELECT COUNT(*) FROM terminology WHERE approved = TRUE"
        ).fetchone()
        stats["approved_terms"] = approved_terms[0] or 0

        # Errors
        if run_id:
            errors = self.conn.execute(
                f"SELECT COUNT(*) FROM processing_log {run_filter} AND level = 'ERROR'"
            ).fetchone()
        else:
            errors = self.conn.execute(
                "SELECT COUNT(*) FROM processing_log WHERE level = 'ERROR'"
            ).fetchone()
        stats["errors"] = errors[0] or 0

        return stats

    def get_statistics(self) -> dict:
        """Get statistics for the CLI (formatted for display)."""
        doc_counts = self.conn.execute(
            "SELECT status, COUNT(*) FROM documents GROUP BY status"
        ).fetchall()
        status_map = {row[0]: row[1] for row in doc_counts}

        total_pages = self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] or 0

        translated_pages = (
            self.conn.execute(
                "SELECT COUNT(*) FROM pages WHERE ar_content IS NOT NULL OR en_content IS NOT NULL OR fr_content IS NOT NULL"
            ).fetchone()[0]
            or 0
        )

        total_terms = self.conn.execute("SELECT COUNT(*) FROM terminology").fetchone()[0] or 0

        return {
            "total_documents": sum(status_map.values()),
            "completed_documents": status_map.get("completed", 0),
            "in_progress_documents": status_map.get("in_progress", 0),
            "pending_documents": status_map.get("pending", 0),
            "failed_documents": status_map.get("failed", 0),
            "total_pages": total_pages,
            "translated_pages": translated_pages,
            "total_terms": total_terms,
        }

    # ==================== Extensions ====================

    def install_extensions(self) -> None:
        """Install and load DuckDB extensions for FTS and VSS."""
        # Full-text search
        self.conn.execute("INSTALL fts")
        self.conn.execute("LOAD fts")

        # Vector similarity search
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

    def create_fts_index(self, table: str, id_column: str, text_column: str) -> None:
        """Create a full-text search index."""
        self.conn.execute(f"PRAGMA create_fts_index('{table}', '{id_column}', '{text_column}')")

    def create_vss_index(self, table: str, embedding_column: str, metric: str = "cosine") -> None:
        """Create a vector similarity search index."""
        index_name = f"idx_{table}_{embedding_column}_hnsw"
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table}
            USING HNSW ({embedding_column})
            WITH (metric = '{metric}')
            """
        )

    def setup_search_indexes(self) -> None:
        """Initialize FTS and VSS indexes for terminology search."""
        # Install and load extensions
        self.install_extensions()

        # Create FTS index on terminology table for term and context search
        with contextlib.suppress(duckdb.CatalogException):
            self.conn.execute(
                "PRAGMA create_fts_index('terminology', 'id', 'term', 'context', overwrite=1)"
            )

        # Note: VSS index on terminology.embedding is created dynamically
        # when embeddings are added (requires non-null embeddings)

    def search_terms_fts(
        self,
        query: str,
        document_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Full-text search for terms using BM25 scoring.

        Args:
            query: Search query string.
            document_id: Optional document ID to filter results.
            limit: Maximum number of results.

        Returns:
            List of matching terms with scores.
        """
        doc_filter = f"AND document_id = {document_id}" if document_id else ""

        try:
            rows = self.conn.execute(
                f"""
                SELECT
                    t.id,
                    t.term,
                    t.frequency,
                    t.translation_ar,
                    t.translation_en,
                    t.translation_fr,
                    t.context,
                    t.document_id,
                    fts.score
                FROM (
                    SELECT *, fts_main_terminology.match_bm25(id, ?) AS score
                    FROM terminology
                    WHERE score IS NOT NULL {doc_filter}
                    ORDER BY score DESC
                    LIMIT ?
                ) AS fts
                JOIN terminology t ON fts.id = t.id
                """,
                [query, limit],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "term": row[1],
                    "frequency": row[2],
                    "translation_ar": row[3],
                    "translation_en": row[4],
                    "translation_fr": row[5],
                    "context": row[6],
                    "document_id": row[7],
                    "score": row[8],
                }
                for row in rows
            ]
        except duckdb.CatalogException:
            # FTS index not yet created
            return []

    def search_terms_semantic(
        self,
        query_embedding: list[float],
        document_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Vector similarity search for terms using cosine distance.

        Args:
            query_embedding: Query embedding vector.
            document_id: Optional document ID to filter results.
            limit: Maximum number of results.

        Returns:
            List of matching terms with similarity scores.
        """
        doc_filter = f"WHERE document_id = {document_id}" if document_id else ""
        if doc_filter:
            doc_filter += " AND embedding IS NOT NULL"
        else:
            doc_filter = "WHERE embedding IS NOT NULL"

        embedding_str = f"[{','.join(map(str, query_embedding))}]::FLOAT[]"

        try:
            rows = self.conn.execute(
                f"""
                SELECT
                    id,
                    term,
                    frequency,
                    translation_ar,
                    translation_en,
                    translation_fr,
                    context,
                    document_id,
                    array_cosine_similarity(embedding, {embedding_str}) AS similarity
                FROM terminology
                {doc_filter}
                ORDER BY similarity DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "term": row[1],
                    "frequency": row[2],
                    "translation_ar": row[3],
                    "translation_en": row[4],
                    "translation_fr": row[5],
                    "context": row[6],
                    "document_id": row[7],
                    "similarity": row[8],
                }
                for row in rows
            ]
        except duckdb.CatalogException:
            return []

    def search_terms_hybrid(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        document_id: int | None = None,
        limit: int = 20,
        fts_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining FTS and semantic search.

        Args:
            query: Text query for FTS.
            query_embedding: Query embedding for semantic search.
            document_id: Optional document ID to filter.
            limit: Maximum results.
            fts_weight: Weight for FTS score (0-1), semantic gets 1-fts_weight.

        Returns:
            Combined and ranked results.
        """
        # Get FTS results
        fts_results = self.search_terms_fts(query, document_id, limit * 2)
        fts_scores = {r["id"]: r["score"] for r in fts_results}

        # Get semantic results if embedding provided
        semantic_scores: dict[int, float] = {}
        if query_embedding:
            semantic_results = self.search_terms_semantic(query_embedding, document_id, limit * 2)
            semantic_scores = {r["id"]: r["similarity"] for r in semantic_results}

        # Combine and normalize scores
        all_ids = set(fts_scores.keys()) | set(semantic_scores.keys())
        combined_results = []

        # Normalize scores
        max_fts = max(fts_scores.values()) if fts_scores else 1
        max_sem = max(semantic_scores.values()) if semantic_scores else 1

        for term_id in all_ids:
            fts_norm = fts_scores.get(term_id, 0) / max_fts if max_fts else 0
            sem_norm = semantic_scores.get(term_id, 0) / max_sem if max_sem else 0

            combined_score = (fts_weight * fts_norm) + ((1 - fts_weight) * sem_norm)

            # Get full term data
            term = self.get_term_by_id(term_id)
            if term:
                combined_results.append(
                    {
                        "id": term_id,
                        "term": term.term,
                        "frequency": term.frequency,
                        "translation_ar": term.translation_ar,
                        "translation_en": term.translation_en,
                        "translation_fr": term.translation_fr,
                        "context": term.context,
                        "document_id": term.document_id,
                        "combined_score": combined_score,
                        "fts_score": fts_scores.get(term_id),
                        "semantic_score": semantic_scores.get(term_id),
                    }
                )

        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:limit]

    def get_term_by_id(self, term_id: int) -> Term | None:
        """Get a term by its ID."""
        row = self.conn.execute("SELECT * FROM terminology WHERE id = ?", [term_id]).fetchone()
        if row:
            return self._row_to_term(row)
        return None

    def get_term_glossary(
        self,
        document_id: int,
        target_lang: str,
    ) -> dict[str, str]:
        """
        Get a term→translation glossary for a document.

        Args:
            document_id: Document to get glossary for.
            target_lang: Target language code (ar, en, fr).

        Returns:
            Dictionary mapping source terms to target translations.
        """
        target_col = f"translation_{target_lang}"
        if target_col not in ("translation_ar", "translation_en", "translation_fr"):
            target_col = "translation_en"

        rows = self.conn.execute(
            f"""
            SELECT term, {target_col}
            FROM terminology
            WHERE document_id = ?
            AND {target_col} IS NOT NULL
            ORDER BY frequency DESC
            """,
            [document_id],
        ).fetchall()

        return {row[0]: row[1] for row in rows if row[1]}

    def update_term_embedding(self, term_id: int, embedding: list[float]) -> None:
        """Update embedding for a single term."""
        embedding_str = f"[{','.join(map(str, embedding))}]::FLOAT[]"
        self.conn.execute(
            f"UPDATE terminology SET embedding = {embedding_str} WHERE id = ?",
            [term_id],
        )

    def get_terms_without_embeddings(self, document_id: int | None = None) -> list[Term]:
        """Get terms that don't have embeddings yet."""
        if document_id:
            rows = self.conn.execute(
                "SELECT * FROM terminology WHERE embedding IS NULL AND document_id = ?",
                [document_id],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM terminology WHERE embedding IS NULL").fetchall()
        return [self._row_to_term(row) for row in rows]

    def create_terminology_vss_index(self) -> None:
        """Create VSS index on terminology embeddings if embeddings exist."""
        # Check if any embeddings exist
        count = self.conn.execute(
            "SELECT COUNT(*) FROM terminology WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        if count > 0:
            with contextlib.suppress(duckdb.CatalogException):
                self.create_vss_index("terminology", "embedding", "cosine")
