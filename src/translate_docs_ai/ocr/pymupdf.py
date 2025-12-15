"""
PyMuPDF-based text extraction for native PDFs.

Uses pymupdf4llm for markdown-formatted output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pymupdf4llm

from translate_docs_ai.ocr.base import OCRProvider, OCRQuality, OCRResult, assess_text_quality


class PyMuPDFExtractor(OCRProvider):
    """
    Extract text from native PDFs using PyMuPDF and pymupdf4llm.

    This is the fastest option for PDFs with embedded text.
    Returns markdown-formatted output.
    """

    @property
    def name(self) -> str:
        return "pymupdf4llm"

    async def extract_page(
        self,
        file_path: Path,
        page_number: int,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Extract text from a single page.

        Args:
            file_path: Path to PDF file.
            page_number: Page number (0-indexed).
            **kwargs: Additional options:
                - write_images: bool - Extract images (default: False)
                - embed_images: bool - Embed images as base64 (default: False)

        Returns:
            OCRResult with markdown-formatted text.
        """
        try:
            # Use pymupdf4llm for markdown output with page selection
            result = pymupdf4llm.to_markdown(
                str(file_path),
                pages=[page_number],
                page_chunks=True,
                write_images=kwargs.get("write_images", False),
                embed_images=kwargs.get("embed_images", False),
            )

            if result and len(result) > 0:
                page_data = result[0]
                content = (
                    page_data.get("text", "") if isinstance(page_data, dict) else str(page_data)
                )

                # Assess quality
                quality, confidence = assess_text_quality(content)

                return OCRResult(
                    content=content,
                    page_number=page_number,
                    confidence=confidence,
                    quality=quality,
                    model_used=self.name,
                    metadata=page_data.get("metadata", {}) if isinstance(page_data, dict) else {},
                )
            else:
                return OCRResult(
                    content="",
                    page_number=page_number,
                    confidence=0.0,
                    quality=OCRQuality.POOR,
                    model_used=self.name,
                )

        except Exception as e:
            return OCRResult(
                content="",
                page_number=page_number,
                confidence=0.0,
                quality=OCRQuality.POOR,
                model_used=self.name,
                metadata={"error": str(e)},
            )

    async def extract_document(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> list[OCRResult]:
        """
        Extract text from all pages in a PDF.

        Args:
            file_path: Path to PDF file.
            **kwargs: Additional options passed to extract_page.

        Returns:
            List of OCRResult, one per page.
        """
        results: list[OCRResult] = []

        try:
            # Get total page count
            with fitz.open(file_path) as doc:
                total_pages = len(doc)

            # Use pymupdf4llm for all pages with page_chunks=True
            all_pages = pymupdf4llm.to_markdown(
                str(file_path),
                page_chunks=True,
                write_images=kwargs.get("write_images", False),
                embed_images=kwargs.get("embed_images", False),
            )

            for page_num, page_data in enumerate(all_pages):
                content = (
                    page_data.get("text", "") if isinstance(page_data, dict) else str(page_data)
                )
                quality, confidence = assess_text_quality(content)

                results.append(
                    OCRResult(
                        content=content,
                        page_number=page_num,
                        confidence=confidence,
                        quality=quality,
                        model_used=self.name,
                        metadata=page_data.get("metadata", {})
                        if isinstance(page_data, dict)
                        else {},
                    )
                )

            # Fill in any missing pages
            while len(results) < total_pages:
                results.append(
                    OCRResult(
                        content="",
                        page_number=len(results),
                        confidence=0.0,
                        quality=OCRQuality.POOR,
                        model_used=self.name,
                    )
                )

        except Exception as e:
            # Return single error result if we can't process the document
            results.append(
                OCRResult(
                    content="",
                    page_number=0,
                    confidence=0.0,
                    quality=OCRQuality.POOR,
                    model_used=self.name,
                    metadata={"error": str(e)},
                )
            )

        return results

    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        return file_path.suffix.lower() == ".pdf"

    def is_native_pdf(self, file_path: Path) -> bool:
        """
        Check if PDF has native text (not scanned).

        Returns True if the PDF has extractable text on most pages.
        """
        try:
            with fitz.open(file_path) as doc:
                if len(doc) == 0:
                    return False

                # Sample pages to check
                pages_to_check = min(3, len(doc))
                text_pages = 0

                for i in range(pages_to_check):
                    page = doc[i]
                    text = page.get_text().strip()
                    if len(text) > 50:  # Minimum threshold for meaningful text
                        text_pages += 1

                # Consider native if majority of sampled pages have text
                return text_pages >= pages_to_check * 0.5

        except Exception:
            return False

    async def get_page_as_image(
        self,
        file_path: Path,
        page_number: int,
        dpi: int = 150,
    ) -> bytes:
        """
        Render a PDF page as PNG image.

        Useful for sending to OCR APIs.

        Args:
            file_path: Path to PDF file.
            page_number: Page number (0-indexed).
            dpi: Resolution for rendering.

        Returns:
            PNG image bytes.
        """
        with fitz.open(file_path) as doc:
            page = doc[page_number]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            return pix.tobytes("png")
