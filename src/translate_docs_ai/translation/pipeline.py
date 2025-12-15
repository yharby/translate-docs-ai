"""
LangGraph-based translation pipeline.

Provides stateful workflow for document translation with:
- Checkpoint/resume capability
- Auto and semi-auto processing modes
- Error recovery and retry logic
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from translate_docs_ai.config import ProcessingMode
from translate_docs_ai.database import Database, Page, Stage, Status
from translate_docs_ai.ocr import DeepInfraOCR, PyMuPDFExtractor
from translate_docs_ai.terminology import TerminologyExtractor
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
    max_terms: int = 500

    # Embedding options
    enable_embeddings: bool = True  # Generate embeddings for terminology
    embedding_model: str = "multilingual"  # multilingual, arabic, or fast

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

        # Terminology
        self.term_extractor = TerminologyExtractor(
            db=self.db,
            min_frequency=self.config.min_term_frequency,
            max_terms=self.config.max_terms,
        )

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

        # Review can go to translation or back to review
        workflow.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "translation": "translation",
                "review": "review",  # Wait for approval
            },
        )

        workflow.add_edge("translation", "export")
        workflow.add_edge("export", END)
        workflow.add_edge("error", END)

        return workflow

    def _route_after_terminology(self, state: PipelineState) -> str:
        """Route after terminology extraction."""
        if state["processing_mode"] == ProcessingMode.SEMI_AUTO.value:
            return "review"
        return "translation"

    def _route_after_review(self, state: PipelineState) -> str:
        """Route after review stage."""
        if state["review_approved"]:
            return "translation"
        return "review"

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

            for page_num in range(state["total_pages"]):
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
            self.db.log(
                level="INFO",
                stage="ocr",
                message=f"OCR completed: {len(ocr_results)}/{state['total_pages']} pages using {ocr_method}",
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
        """Extract and process terminology."""
        document_id = state["document_id"]

        try:
            # Extract terms
            terms = self.term_extractor.extract_from_document(document_id)

            # Generate embeddings for semantic search (if enabled)
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
                    # Log but don't fail - embeddings are optional
                    self.db.log(
                        level="WARNING",
                        stage="terminology",
                        message=f"Failed to generate embeddings: {e}",
                        document_id=document_id,
                    )

            # Save checkpoint
            self.db.save_checkpoint(
                document_id=document_id,
                stage=Stage.TERMINOLOGY_EXTRACT,
                state_data={
                    "terms_count": len(terms),
                    "embeddings_generated": embeddings_generated,
                },
            )

            # Log completion
            self.db.log(
                level="INFO",
                stage="terminology",
                message=f"Extracted {len(terms)} terms, generated {embeddings_generated} embeddings",
                document_id=document_id,
            )

            return {
                "current_stage": PipelineStage.REVIEW
                if state["processing_mode"] == ProcessingMode.SEMI_AUTO.value
                else PipelineStage.TRANSLATION,
                "terms_extracted": len(terms),
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
            message="Awaiting terminology review",
            document_id=document_id,
        )

        # Return without changing stage - will be called again when approved
        return {"needs_review": True}

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

            for page in pages_to_translate:
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
            self.db.log(
                level="INFO",
                stage="translation",
                message=f"Translation completed: {pages_translated}/{len(pages_to_translate)} pages",
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
    ) -> PipelineState:
        """
        Process a document through the pipeline.

        Args:
            document_id: Document ID to process.
            resume: Whether to resume from checkpoint if available.

        Returns:
            Final pipeline state.
        """
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
            # Determine starting stage from checkpoint
            stage_map = {
                Stage.OCR: PipelineStage.TERMINOLOGY,
                Stage.TERMINOLOGY_EXTRACT: PipelineStage.REVIEW
                if self.config.processing_mode == ProcessingMode.SEMI_AUTO
                else PipelineStage.TRANSLATION,
                Stage.PAGE_TRANSLATE: PipelineStage.EXPORT,
                Stage.EXPORT: PipelineStage.COMPLETE,
            }
            initial_state["current_stage"] = stage_map.get(checkpoint.stage, PipelineStage.INIT)
            initial_state["current_page"] = checkpoint.page_number or 0

            self.db.log(
                level="INFO",
                stage="pipeline_resume",
                message=f"Resuming from {checkpoint.stage.value}",
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
