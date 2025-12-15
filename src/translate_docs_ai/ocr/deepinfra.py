"""
DeepInfra-based OCR using olmOCR-2 and DeepSeek-OCR models.

Provides OCR capabilities for scanned documents and complex layouts.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from openai import AsyncOpenAI

from translate_docs_ai.ocr.base import OCRProvider, OCRQuality, OCRResult, assess_text_quality


class DeepInfraOCR(OCRProvider):
    """
    OCR using DeepInfra API with olmOCR-2 or DeepSeek-OCR models.

    Supports:
    - allenai/olmOCR-2-7B-1025: Best for general documents, cost-effective
    - deepseek-ai/DeepSeek-OCR: Best for complex layouts and diagrams
    """

    MODELS = {
        "olmocr": "allenai/olmOCR-2-7B-1025",
        "deepseek": "deepseek-ai/DeepSeek-OCR",
    }

    # Default prompts for OCR
    PROMPTS = {
        "olmocr": "Convert this document page to markdown. Preserve all text, tables, and formatting.",
        "deepseek": "<image>\nConvert the document to markdown.",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "olmocr",
        base_url: str = "https://api.deepinfra.com/v1/openai",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize DeepInfra OCR client.

        Args:
            api_key: DeepInfra API key.
            model: Model to use ('olmocr' or 'deepseek').
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self._model_key = model
        self._model = self.MODELS.get(model, model)
        self._prompt = self.PROMPTS.get(model, self.PROMPTS["olmocr"])
        self._timeout = timeout
        self._max_retries = max_retries

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return self._model

    async def extract_page(
        self,
        file_path: Path,
        page_number: int,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Extract text from a single page using OCR API.

        Args:
            file_path: Path to document (PDF or image).
            page_number: Page number (0-indexed).
            **kwargs: Additional options:
                - dpi: Image rendering DPI (default: 150)
                - prompt: Custom prompt override

        Returns:
            OCRResult with extracted text.
        """
        dpi = kwargs.get("dpi", 150)
        prompt = kwargs.get("prompt", self._prompt)

        try:
            # Get image bytes
            if file_path.suffix.lower() == ".pdf":
                image_bytes = await self._render_pdf_page(file_path, page_number, dpi)
            else:
                with open(file_path, "rb") as f:
                    image_bytes = f.read()

            # Encode to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Call API
            content = await self._call_api(image_base64, prompt)

            # Assess quality
            quality, confidence = assess_text_quality(content)

            return OCRResult(
                content=content,
                page_number=page_number,
                confidence=confidence,
                quality=quality,
                model_used=self.name,
                metadata={"dpi": dpi},
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
        Extract text from all pages in a document.

        Args:
            file_path: Path to document.
            **kwargs: Additional options:
                - dpi: Image rendering DPI
                - concurrent: Max concurrent requests (default: 4)
                - prompt: Custom prompt override

        Returns:
            List of OCRResult, one per page.
        """
        concurrent = kwargs.get("concurrent", 4)
        results: list[OCRResult] = []

        # Get total pages
        if file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        else:
            # Single image
            total_pages = 1

        # Process pages with concurrency limit
        semaphore = asyncio.Semaphore(concurrent)

        async def process_page(page_num: int) -> OCRResult:
            async with semaphore:
                return await self.extract_page(file_path, page_num, **kwargs)

        tasks = [process_page(i) for i in range(total_pages)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def _render_pdf_page(
        self,
        file_path: Path,
        page_number: int,
        dpi: int,
    ) -> bytes:
        """Render PDF page to PNG image."""
        with fitz.open(file_path) as doc:
            page = doc[page_number]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            return pix.tobytes("png")

    async def _call_api(
        self,
        image_base64: str,
        prompt: str,
    ) -> str:
        """Call DeepInfra API with retry logic."""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=4096,  # Reduced to avoid context overflow with images
                    temperature=0.0,
                )

                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content

                return ""

            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        raise last_error or Exception("Failed to call OCR API")

    def can_handle(self, file_path: Path) -> bool:
        """Check if this provider can handle the file."""
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        return file_path.suffix.lower() in supported


class OCRPipeline:
    """
    Intelligent OCR pipeline that selects the best OCR method.

    Uses PyMuPDF for native PDFs and falls back to DeepInfra for scanned documents.
    """

    def __init__(
        self,
        deepinfra_api_key: str,
        primary_model: str = "olmocr",
        fallback_model: str = "deepseek",
        dpi: int = 150,
    ):
        """
        Initialize OCR pipeline.

        Args:
            deepinfra_api_key: DeepInfra API key.
            primary_model: Primary OCR model for scanned docs.
            fallback_model: Fallback model for complex layouts.
            dpi: Default DPI for image rendering.
        """
        from translate_docs_ai.ocr.pymupdf import PyMuPDFExtractor

        self.pymupdf = PyMuPDFExtractor()
        self.primary_ocr = DeepInfraOCR(deepinfra_api_key, model=primary_model)
        self.fallback_ocr = DeepInfraOCR(deepinfra_api_key, model=fallback_model)
        self.dpi = dpi

    async def extract_page(
        self,
        file_path: Path,
        page_number: int,
        force_ocr: bool = False,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Extract text from a page using the best available method.

        Args:
            file_path: Path to document.
            page_number: Page number (0-indexed).
            force_ocr: Force OCR even for native PDFs.
            **kwargs: Additional options.

        Returns:
            OCRResult with extracted text.
        """
        kwargs.setdefault("dpi", self.dpi)

        # Try native extraction first (unless forced OCR)
        if not force_ocr and file_path.suffix.lower() == ".pdf":
            if self.pymupdf.is_native_pdf(file_path):
                result = await self.pymupdf.extract_page(file_path, page_number, **kwargs)

                # If good quality, return it
                if result.quality in (OCRQuality.EXCELLENT, OCRQuality.GOOD):
                    return result

        # Use primary OCR
        result = await self.primary_ocr.extract_page(file_path, page_number, **kwargs)

        # If poor quality, try fallback
        if result.quality == OCRQuality.POOR:
            fallback_result = await self.fallback_ocr.extract_page(file_path, page_number, **kwargs)
            if fallback_result.quality.value > result.quality.value:
                return fallback_result

        return result

    async def extract_document(
        self,
        file_path: Path,
        force_ocr: bool = False,
        **kwargs: Any,
    ) -> list[OCRResult]:
        """
        Extract text from all pages in a document.

        Args:
            file_path: Path to document.
            force_ocr: Force OCR even for native PDFs.
            **kwargs: Additional options.

        Returns:
            List of OCRResult, one per page.
        """
        kwargs.setdefault("dpi", self.dpi)

        # For native PDFs, try pymupdf first
        if not force_ocr and file_path.suffix.lower() == ".pdf":
            if self.pymupdf.is_native_pdf(file_path):
                results = await self.pymupdf.extract_document(file_path, **kwargs)

                # Check if all pages are good quality
                all_good = all(
                    r.quality in (OCRQuality.EXCELLENT, OCRQuality.GOOD) for r in results
                )
                if all_good:
                    return results

        # Use primary OCR for all pages
        return await self.primary_ocr.extract_document(file_path, **kwargs)
