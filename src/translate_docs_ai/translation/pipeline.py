"""
LangGraph-based translation pipeline.

Provides stateful workflow for document translation with:
- Checkpoint/resume capability
- Auto and semi-auto processing modes
- Error recovery and retry logic
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from translate_docs_ai.config import ProcessingMode
from translate_docs_ai.database import Database, Page, Stage, Status
from translate_docs_ai.memory import DuckDBStore
from translate_docs_ai.ocr import DeepInfraOCR, PyMuPDFExtractor
from translate_docs_ai.terminology import (
    GlossaryDetector,
    LLMTerminologyExtractor,
    TerminologyExtractor,
)
from translate_docs_ai.terminology.embeddings import EmbeddingGenerator
from translate_docs_ai.translation.search_tools import TerminologySearchTools
from translate_docs_ai.translation.translator import PageTranslator


class PipelineStage(str, Enum):
    """Pipeline processing stages."""

    INIT = "init"
    OCR = "ocr"
    TERMINOLOGY = "terminology"
    REVIEW = "review"  # For semi-auto mode
    TRANSLATION = "translation"
    EXPORT = "export"
    COMPLETE = "complete"
    ERROR = "error"


class PipelineState(TypedDict):
    """State for the translation pipeline."""

    # Document info
    document_id: int
    document_path: str
    total_pages: int

    # Progress tracking
    current_stage: PipelineStage
    current_page: int
    pages_completed: Annotated[list[int], operator.add]

    # Configuration
    source_lang: str
    target_lang: str
    processing_mode: str

    # Results
    ocr_results: dict[int, str]
    terms_extracted: int
    pages_translated: int

    # Error handling
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: int

    # Control flags
    needs_review: bool
    review_approved: bool


@dataclass
class ProgressInfo:
    """Progress information for callbacks."""

    stage: str  # Current stage name (ocr, terminology, translation, etc.)
    stage_display: str  # Human-readable stage description
    page_current: int | None = None  # Current page number (1-indexed for display)
    page_total: int | None = None  # Total pages
    detail: str | None = None  # Additional detail message (e.g., "42 terms extracted")


# Type alias for progress callback
ProgressCallback = Callable[[ProgressInfo], None] | None


@dataclass
class PipelineConfig:
    """Configuration for the translation pipeline."""

    # API keys
    openrouter_api_key: str
    deepinfra_api_key: str | None = None

    # Processing options
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    source_lang: str = "en"
    target_lang: str = "ar"

    # OCR options
    ocr_dpi: int = 150
    force_ocr: bool = False  # Skip pymupdf and use DeepInfra OCR directly
    ocr_primary_model: str = "olmocr"  # Primary OCR model (olmocr or deepseek)
    ocr_fallback_model: str = "deepseek"  # Fallback OCR model

    # Translation options
    translation_model: str = "default"
    max_concurrent_pages: int = 1

    # Terminology options
    min_term_frequency: int = 3
    max_terms: int = 200

    # Embedding options
    enable_embeddings: bool = True  # Generate embeddings for terminology
    embedding_model: str = "multilingual"  # multilingual, arabic, or fast

    # Memory options
    enable_memory: bool = True  # Enable DuckDB-backed long-term memory

    # Retry options
    max_retries: int = 3


class TranslationPipeline:
    """
    LangGraph-based pipeline for document translation.

    Supports:
    - Auto mode: Fully automated processing
    - Semi-auto mode: Pauses for terminology review before translation
    - Checkpoint/resume for interrupted processing
    """

    def __init__(
        self,
        db: Database,
        config: PipelineConfig,
    ):
        """
        Initialize translation pipeline.

        Args:
            db: Database instance.
            config: Pipeline configuration.
        """
        self.db = db
        self.config = config

        # Initialize components
        self._init_components()

        # Build workflow graph
        self._graph = self._build_graph()

        # Memory-based checkpointer for state persistence
        self._checkpointer = MemorySaver()

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # OCR - only init pymupdf if not forcing OCR
        if not self.config.force_ocr:
            self.pymupdf = PyMuPDFExtractor()
        else:
            self.pymupdf = None

        # DeepInfra OCR (primary and fallback)
        if self.config.deepinfra_api_key:
            self.deepinfra_ocr = DeepInfraOCR(
                api_key=self.config.deepinfra_api_key,
                model=self.config.ocr_primary_model,
            )
            self.deepinfra_fallback = DeepInfraOCR(
                api_key=self.config.deepinfra_api_key,
                model=self.config.ocr_fallback_model,
            )
        else:
            self.deepinfra_ocr = None
            self.deepinfra_fallback = None

        # LLM-based Terminology Extractor (primary)
        self.llm_term_extractor: LLMTerminologyExtractor | None = None
        if self.config.openrouter_api_key:
            self.llm_term_extractor = LLMTerminologyExtractor(
                api_key=self.config.openrouter_api_key,
                db=self.db,
                model=self.config.translation_model,
            )

        # Frequency-based Terminology Extractor (fallback)
        self.term_extractor = TerminologyExtractor(
            db=self.db,
            min_frequency=self.config.min_term_frequency,
            max_terms=self.config.max_terms,
        )

        # Glossary Detector (extracts from document's glossary/terminology pages)
        self.glossary_detector = GlossaryDetector(db=self.db)

        # Embeddings (optional, for semantic search)
        self.embedding_generator: EmbeddingGenerator | None = None
        if self.config.enable_embeddings:
            try:
                self.embedding_generator = EmbeddingGenerator(
                    db=self.db,
                    model_name=self.config.embedding_model,
                )
            except Exception:
                # sentence-transformers may not be installed
                self.embedding_generator = None

        # Search tools for terminology lookup
        self.search_tools = TerminologySearchTools(
            db=self.db,
            embedding_generator=self.embedding_generator,
        )

        # Long-term memory store (optional)
        self.memory_store: DuckDBStore | None = None
        if self.config.enable_memory:
            self.memory_store = DuckDBStore(
                db=self.db,
                embedding_generator=self.embedding_generator,
            )

        # Translation
        self.translator = PageTranslator(
            api_key=self.config.openrouter_api_key,
            db=self.db,
            model=self.config.translation_model,
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("init", self._node_init)
        workflow.add_node("ocr", self._node_ocr)
        workflow.add_node("terminology", self._node_terminology)
        workflow.add_node("review", self._node_review)
        workflow.add_node("translation", self._node_translation)
        workflow.add_node("export", self._node_export)
        workflow.add_node("error", self._node_error)

        # Set entry point
        workflow.set_entry_point("init")

        # Add edges
        workflow.add_edge("init", "ocr")
        workflow.add_edge("ocr", "terminology")

        # Conditional edge after terminology (depends on mode)
        workflow.add_conditional_edges(
            "terminology",
            self._route_after_terminology,
            {
                "review": "review",
                "translation": "translation",
            },
        )

        # Review can go to translation or END (waiting for approval)
        workflow.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "translation": "translation",
                "end": END,  # Pause and wait for user approval
            },
        )

        workflow.add_edge("translation", "export")
        workflow.add_edge("export", END)
        workflow.add_edge("error", END)

        return workflow

    def _report_progress(
        self,
        stage: str,
        stage_display: str,
        page_current: int | None = None,
        page_total: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Report progress via callback if set."""
        if hasattr(self, "_progress_callback") and self._progress_callback:
            self._progress_callback(
                ProgressInfo(
                    stage=stage,
                    stage_display=stage_display,
                    page_current=page_current,
                    page_total=page_total,
                    detail=detail,
                )
            )

    def _route_after_terminology(self, state: PipelineState) -> str:
        """Route after terminology extraction."""
        if state["processing_mode"] == ProcessingMode.SEMI_AUTO.value:
            return "review"
        return "translation"

    def _route_after_review(self, state: PipelineState) -> str:
        """Route after review stage."""
        if state["review_approved"]:
            return "translation"
        return "end"  # Pause pipeline until user approves

    async def _node_init(self, state: PipelineState) -> dict[str, Any]:
        """Initialize pipeline for a document."""
        doc = self.db.get_document(state["document_id"])
        if not doc:
            return {
                "current_stage": PipelineStage.ERROR,
                "errors": [{"stage": "init", "error": "Document not found"}],
            }

        # Update document status
        self.db.update_document_status(doc.id, Status.IN_PROGRESS)

        # Log start
        self.db.log(
            level="INFO",
            stage="pipeline_init",
            message=f"Starting pipeline for {doc.file_name}",
            document_id=doc.id,
        )

        return {
            "document_path": doc.full_path,
            "total_pages": doc.total_pages,
            "current_stage": PipelineStage.OCR,
        }

    async def _node_ocr(self, state: PipelineState) -> dict[str, Any]:
        """Extract text from document pages."""
        from pathlib import Path

        from translate_docs_ai.ocr.base import OCRQuality

        doc_path = Path(state["document_path"])
        document_id = state["document_id"]
        ocr_results: dict[int, str] = {}
        errors: list[dict[str, Any]] = []

        try:
            # Determine OCR method based on config
            # If force_ocr is True, skip pymupdf entirely and use DeepInfra
            use_native = False
            if not self.config.force_ocr and self.pymupdf:
                is_native = doc_path.suffix.lower() == ".pdf" and self.pymupdf.is_native_pdf(
                    doc_path
                )
                use_native = is_native

            ocr_method = "native" if use_native else "deepinfra"

            self.db.log(
                level="INFO",
                stage="ocr",
                message=f"Using OCR method: {ocr_method} (force_ocr={self.config.force_ocr})",
                document_id=document_id,
            )

            # Get existing pages for resume capability
            existing_pages = {p.page_number: p for p in self.db.get_document_pages(document_id)}
            pages_skipped = 0

            total_pages = state["total_pages"]
            for page_num in range(total_pages):
                # Report progress for each page
                self._report_progress(
                    stage="ocr",
                    stage_display="Extracting text (OCR)",
                    page_current=page_num + 1,
                    page_total=total_pages,
                )

                # Check if page already exists with content (for resume)
                existing_page = existing_pages.get(page_num)
                if existing_page and existing_page.original_content:
                    # Page already OCR'd, skip and use existing content
                    ocr_results[page_num] = existing_page.original_content
                    pages_skipped += 1
                    continue

                try:
                    if use_native and self.pymupdf:
                        # Use native PDF text extraction
                        result = await self.pymupdf.extract_page(doc_path, page_num)
                    elif self.deepinfra_ocr:
                        # Use DeepInfra OCR (primary model)
                        result = await self.deepinfra_ocr.extract_page(
                            doc_path, page_num, dpi=self.config.ocr_dpi
                        )
                        # If poor quality, try fallback model
                        if result.quality == OCRQuality.POOR and self.deepinfra_fallback:
                            fallback_result = await self.deepinfra_fallback.extract_page(
                                doc_path, page_num, dpi=self.config.ocr_dpi
                            )
                            if fallback_result.quality.value > result.quality.value:
                                result = fallback_result
                    elif self.pymupdf:
                        # Fallback to pymupdf if no DeepInfra API key
                        result = await self.pymupdf.extract_page(doc_path, page_num)
                    else:
                        raise ValueError(
                            "No OCR method available. Set DEEPINFRA_API_KEY or disable force_ocr."
                        )

                    ocr_results[page_num] = result.content

                    # Save page to database
                    page = Page(
                        document_id=document_id,
                        page_number=page_num,
                        original_content=result.content,
                        ocr_confidence=result.confidence,
                    )
                    self.db.add_page(page)

                except Exception as e:
                    errors.append(
                        {
                            "stage": "ocr",
                            "page": page_num,
                            "error": str(e),
                        }
                    )

            # Log completion
            ocr_msg = (
                f"OCR completed: {len(ocr_results)}/{state['total_pages']} pages using {ocr_method}"
            )
            if pages_skipped > 0:
                ocr_msg += f" ({pages_skipped} pages resumed from previous run)"
            self.db.log(
                level="INFO",
                stage="ocr",
                message=ocr_msg,
                document_id=document_id,
            )

            # Save checkpoint
            self.db.save_checkpoint(
                document_id=document_id,
                stage=Stage.OCR,
                page_number=state["total_pages"] - 1,
                state_data={"ocr_method": ocr_method},
            )

            return {
                "current_stage": PipelineStage.TERMINOLOGY,
                "ocr_results": ocr_results,
                "pages_completed": list(ocr_results.keys()),
                "errors": errors,
            }

        except Exception as e:
            return {
                "current_stage": PipelineStage.ERROR,
                "errors": [{"stage": "ocr", "error": str(e)}],
            }

    async def _node_terminology(self, state: PipelineState) -> dict[str, Any]:
        """Extract and process terminology using hybrid approach.

        Strategy (token-efficient):
        0. Detect glossary pages in document (highest priority, pre-approved terms)
        1. Use frequency-based extraction to identify candidate terms (no LLM cost)
        2. Generate embeddings for semantic deduplication (local model)
        3. Use LLM only for translating the top filtered terms (minimal tokens)
        """
        document_id = state["document_id"]
        source_lang = state["source_lang"]
        target_lang = state["target_lang"]

        # Report terminology extraction start
        self._report_progress(
            stage="terminology",
            stage_display="Extracting terminology",
        )

        try:
            # Step 0: Detect and extract from glossary pages (highest priority)
            # These terms are pre-approved since they come from the document's
            # official glossary/terminology section
            glossary_terms = self.glossary_detector.extract_and_save_glossary(
                document_id=document_id,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            if glossary_terms:
                self._report_progress(
                    stage="terminology",
                    stage_display="Found document glossary",
                    detail=f"{len(glossary_terms)} terms from glossary pages",
                )
                self.db.log(
                    level="INFO",
                    stage="terminology",
                    message=f"Extracted {len(glossary_terms)} terms from glossary pages",
                    document_id=document_id,
                )

            # Step 1: Frequency-based extraction (fast, no API cost)
            terms = self.term_extractor.extract_from_document(document_id)

            self.db.log(
                level="INFO",
                stage="terminology",
                message=f"Frequency extraction found {len(terms)} candidate terms",
                document_id=document_id,
            )

            # Combine counts (glossary terms are already in DB)
            total_terms = len(glossary_terms) + len(terms)

            # Step 2: Generate embeddings for semantic search/deduplication
            embeddings_generated = 0
            if self.embedding_generator and terms:
                try:
                    embeddings_generated = await self.embedding_generator.generate_term_embeddings(
                        document_id=document_id,
                        include_context=True,
                    )
                    # Setup search indexes after embeddings are generated
                    self.search_tools.initialize_search()
                    self.db.create_terminology_vss_index()
                except Exception as e:
                    self.db.log(
                        level="WARNING",
                        stage="terminology",
                        message=f"Failed to generate embeddings: {e}",
                        document_id=document_id,
                    )

            # Step 3: Use LLM to translate terms (only if we have an LLM extractor)
            # This is token-efficient: we only send term list, not full document
            terms_translated = 0
            if self.llm_term_extractor and terms:
                try:
                    self._report_progress(
                        stage="terminology",
                        stage_display="Translating terms with LLM",
                        detail=f"{len(terms)} terms to translate",
                    )
                    terms_translated = await self.llm_term_extractor.translate_terms(
                        document_id=document_id,
                        target_lang=target_lang,
                    )
                    self.db.log(
                        level="INFO",
                        stage="terminology",
                        message=f"LLM translated {terms_translated} terms",
                        document_id=document_id,
                    )
                except Exception as e:
                    self.db.log(
                        level="WARNING",
                        stage="terminology",
                        message=f"LLM term translation failed: {e}",
                        document_id=document_id,
                    )

            # Save checkpoint
            self.db.save_checkpoint(
                document_id=document_id,
                stage=Stage.TERMINOLOGY_EXTRACT,
                state_data={
                    "glossary_terms": len(glossary_terms),
                    "frequency_terms": len(terms),
                    "total_terms": total_terms,
                    "embeddings_generated": embeddings_generated,
                    "terms_translated": terms_translated,
                },
            )

            # Log completion
            glossary_info = f" ({len(glossary_terms)} from glossary)" if glossary_terms else ""
            self.db.log(
                level="INFO",
                stage="terminology",
                message=f"Extracted {total_terms} terms{glossary_info}, "
                f"{embeddings_generated} embeddings, {terms_translated} translations",
                document_id=document_id,
            )

            # Report completion with term count
            detail = f"{total_terms} terms"
            if glossary_terms:
                detail += f" ({len(glossary_terms)} from glossary)"
            detail += f", {terms_translated} translated"
            self._report_progress(
                stage="terminology",
                stage_display="Terminology extracted",
                detail=detail,
            )

            return {
                "current_stage": PipelineStage.REVIEW
                if state["processing_mode"] == ProcessingMode.SEMI_AUTO.value
                else PipelineStage.TRANSLATION,
                "terms_extracted": total_terms,
                "needs_review": state["processing_mode"] == ProcessingMode.SEMI_AUTO.value,
            }

        except Exception as e:
            return {
                "current_stage": PipelineStage.ERROR,
                "errors": [{"stage": "terminology", "error": str(e)}],
            }

    async def _node_review(self, state: PipelineState) -> dict[str, Any]:
        """
        Review checkpoint for semi-auto mode.

        This node pauses the pipeline until the user approves terminology.
        """
        document_id = state["document_id"]

        if state["review_approved"]:
            # User has approved, continue to translation
            self.db.log(
                level="INFO",
                stage="review",
                message="Terminology review approved, continuing to translation",
                document_id=document_id,
            )
            return {
                "current_stage": PipelineStage.TRANSLATION,
                "needs_review": False,
            }

        # Save review checkpoint
        self.db.save_checkpoint(
            document_id=document_id,
            stage=Stage.TERMINOLOGY_EXTRACT,
            state_data={
                "awaiting_review": True,
                "terms_extracted": state["terms_extracted"],
            },
        )

        self.db.log(
            level="INFO",
            stage="review",
            message="Awaiting terminology review - run 'translate-docs approve' to continue",
            document_id=document_id,
        )

        # Report progress that we're awaiting review
        self._report_progress(
            stage="review",
            stage_display="Awaiting terminology review",
            detail="Use 'translate-docs approve' to continue",
        )

        # Return with review stage - pipeline will END and wait for user approval
        return {
            "current_stage": PipelineStage.REVIEW,
            "needs_review": True,
        }

    async def _node_translation(self, state: PipelineState) -> dict[str, Any]:
        """Translate document pages."""
        document_id = state["document_id"]
        errors: list[dict[str, Any]] = []
        pages_translated = 0

        try:
            pages = self.db.get_document_pages(document_id)
            pages_to_translate = [
                p for p in pages if p.original_content and not p.translated_content
            ]

            # Sort by page number
            pages_to_translate.sort(key=lambda p: p.page_number)
            total_to_translate = len(pages_to_translate)
            total_pages = state["total_pages"]
            pages_skipped = total_pages - total_to_translate

            for page in pages_to_translate:
                # Report progress for each page (show actual page number, not index)
                # page.page_number is 0-indexed, so +1 for display
                self._report_progress(
                    stage="translation",
                    stage_display="Translating pages",
                    page_current=page.page_number + 1,
                    page_total=total_pages,
                )

                try:
                    result = await self.translator.translate_with_context(
                        page=page,
                        document_id=document_id,
                        source_lang=state["source_lang"],
                        target_lang=state["target_lang"],
                    )

                    # Update page
                    if page.id:
                        self.db.update_page(
                            page.id,
                            translated_content=result.translated_content,
                            language=state["target_lang"],
                        )

                    pages_translated += 1

                    # Save checkpoint after each page
                    self.db.save_checkpoint(
                        document_id=document_id,
                        stage=Stage.PAGE_TRANSLATE,
                        page_number=page.page_number,
                        state_data={
                            "pages_translated": pages_translated,
                            "total_pages": len(pages_to_translate),
                        },
                    )

                except Exception as e:
                    errors.append(
                        {
                            "stage": "translation",
                            "page": page.page_number,
                            "error": str(e),
                        }
                    )

                    # Log error but continue with other pages
                    self.db.log(
                        level="ERROR",
                        stage="translation",
                        message=f"Translation failed for page {page.page_number}: {e}",
                        document_id=document_id,
                        context={"page_id": page.id, "error": str(e)},
                    )

            # Log completion
            trans_msg = f"Translation completed: {pages_translated}/{total_to_translate} pages"
            if pages_skipped > 0:
                trans_msg += f" ({pages_skipped} pages resumed from previous run)"
            self.db.log(
                level="INFO",
                stage="translation",
                message=trans_msg,
                document_id=document_id,
            )

            return {
                "current_stage": PipelineStage.EXPORT,
                "pages_translated": pages_translated,
                "errors": errors,
            }

        except Exception as e:
            return {
                "current_stage": PipelineStage.ERROR,
                "errors": [{"stage": "translation", "error": str(e)}],
            }

    async def _node_export(self, state: PipelineState) -> dict[str, Any]:
        """Export translated document."""
        document_id = state["document_id"]

        try:
            # Mark document as completed
            self.db.update_document_status(document_id, Status.COMPLETED)

            # Save final checkpoint
            self.db.save_checkpoint(
                document_id=document_id,
                stage=Stage.EXPORT,
                state_data={"completed": True},
            )

            self.db.log(
                level="INFO",
                stage="export",
                message="Document processing completed",
                document_id=document_id,
            )

            return {"current_stage": PipelineStage.COMPLETE}

        except Exception as e:
            return {
                "current_stage": PipelineStage.ERROR,
                "errors": [{"stage": "export", "error": str(e)}],
            }

    async def _node_error(self, state: PipelineState) -> dict[str, Any]:
        """Handle pipeline errors."""
        document_id = state["document_id"]

        # Mark document as failed
        self.db.update_document_status(document_id, Status.FAILED)

        # Log all errors
        for error in state.get("errors", []):
            self.db.log(
                level="ERROR",
                stage=error.get("stage", "unknown"),
                message=error.get("error", "Unknown error"),
                document_id=document_id,
            )

        return {"current_stage": PipelineStage.ERROR}

    async def process_document(
        self,
        document_id: int,
        resume: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> PipelineState:
        """
        Process a document through the pipeline.

        Args:
            document_id: Document ID to process.
            resume: Whether to resume from checkpoint if available.
            progress_callback: Optional callback for progress updates.

        Returns:
            Final pipeline state.
        """
        # Store callback for use in nodes
        self._progress_callback = progress_callback
        # Check for existing checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.db.get_checkpoint(document_id)

        # Initialize state
        initial_state: PipelineState = {
            "document_id": document_id,
            "document_path": "",
            "total_pages": 0,
            "current_stage": PipelineStage.INIT,
            "current_page": 0,
            "pages_completed": [],
            "source_lang": self.config.source_lang,
            "target_lang": self.config.target_lang,
            "processing_mode": self.config.processing_mode.value,
            "ocr_results": {},
            "terms_extracted": 0,
            "pages_translated": 0,
            "errors": [],
            "retry_count": 0,
            "needs_review": False,
            "review_approved": self.config.processing_mode == ProcessingMode.AUTO,
        }

        # Resume from checkpoint if available
        if checkpoint and checkpoint.state_data:
            # Check if review was already approved (from approve_review() call)
            review_approved = checkpoint.state_data.get("review_approved", False)

            # Determine starting stage from checkpoint
            if checkpoint.stage == Stage.TERMINOLOGY_EXTRACT:
                # If review was approved, skip to translation; otherwise go to review
                if review_approved:
                    initial_state["current_stage"] = PipelineStage.TRANSLATION
                    initial_state["review_approved"] = True
                elif self.config.processing_mode == ProcessingMode.SEMI_AUTO:
                    initial_state["current_stage"] = PipelineStage.REVIEW
                else:
                    initial_state["current_stage"] = PipelineStage.TRANSLATION
            else:
                stage_map = {
                    Stage.OCR: PipelineStage.TERMINOLOGY,
                    Stage.PAGE_TRANSLATE: PipelineStage.EXPORT,
                    Stage.EXPORT: PipelineStage.COMPLETE,
                }
                initial_state["current_stage"] = stage_map.get(checkpoint.stage, PipelineStage.INIT)

            initial_state["current_page"] = checkpoint.page_number or 0

            self.db.log(
                level="INFO",
                stage="pipeline_resume",
                message=f"Resuming from {checkpoint.stage.value}, review_approved={review_approved}",
                document_id=document_id,
            )

        # Compile and run the graph
        app = self._graph.compile(checkpointer=self._checkpointer)

        # Run the pipeline
        config = {"configurable": {"thread_id": f"doc_{document_id}"}}

        final_state = None
        async for state in app.astream(initial_state, config):
            # Get the last state from each node
            for _node_name, node_state in state.items():
                if isinstance(node_state, dict):
                    final_state = {**initial_state, **node_state}

        return final_state or initial_state

    async def approve_review(self, document_id: int) -> None:
        """
        Approve terminology review and continue pipeline.

        Args:
            document_id: Document ID.
        """
        # Get current checkpoint
        checkpoint = self.db.get_checkpoint(document_id)

        if checkpoint and checkpoint.stage == Stage.TERMINOLOGY_EXTRACT:
            # Update checkpoint to mark review as approved
            self.db.save_checkpoint(
                document_id=document_id,
                stage=Stage.TERMINOLOGY_EXTRACT,
                state_data={
                    **checkpoint.state_data,
                    "awaiting_review": False,
                    "review_approved": True,
                },
            )

            self.db.log(
                level="INFO",
                stage="review",
                message="Review approved by user",
                document_id=document_id,
            )

    def get_pipeline_status(self, document_id: int) -> dict[str, Any]:
        """
        Get current pipeline status for a document.

        Args:
            document_id: Document ID.

        Returns:
            Status dictionary.
        """
        doc = self.db.get_document(document_id)
        checkpoint = self.db.get_checkpoint(document_id)
        pages = self.db.get_document_pages(document_id)
        terms = self.db.get_document_terms(document_id)

        pages_ocr = len([p for p in pages if p.original_content])
        pages_translated = len([p for p in pages if p.translated_content])

        return {
            "document_id": document_id,
            "document_name": doc.file_name if doc else None,
            "status": doc.status.value if doc else None,
            "total_pages": doc.total_pages if doc else 0,
            "pages_extracted": pages_ocr,
            "pages_translated": pages_translated,
            "terms_extracted": len(terms),
            "current_stage": checkpoint.stage.value if checkpoint else None,
            "awaiting_review": (
                checkpoint.state_data.get("awaiting_review", False)
                if checkpoint and checkpoint.state_data
                else False
            ),
        }
