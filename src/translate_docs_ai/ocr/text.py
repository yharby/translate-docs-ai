"""
Direct text extraction for plain text and markdown files.

No OCR needed - just reads the file content directly.
"""

from pathlib import Path
from typing import Any

from translate_docs_ai.ocr.base import OCRProvider, OCRQuality, OCRResult


class TextExtractor(OCRProvider):
    """
    Extract text from plain text and markdown files.

    This is a pass-through extractor that simply reads the file content.
    No OCR is performed - the text is already in readable format.
    """

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}

    @property
    def name(self) -> str:
        return "text_direct"

    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def extract_page(
        self,
        file_path: Path,
        page_number: int,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Extract text from a text/markdown file.

        For text files, we treat the entire file as a single "page" (page 0).
        If page_number > 0, returns empty result.

        Args:
            file_path: Path to text/markdown file.
            page_number: Page number (0-indexed). Only page 0 has content.
            **kwargs: Ignored for text files.

        Returns:
            OCRResult with file content (for page 0) or empty (for other pages).
        """
        if page_number != 0:
            # Text files only have one "page"
            return OCRResult(
                content="",
                page_number=page_number,
                confidence=1.0,
                quality=OCRQuality.EXCELLENT,
                model_used=self.name,
                metadata={"note": "Text files have only one page"},
            )

        try:
            content = file_path.read_text(encoding="utf-8")

            return OCRResult(
                content=content,
                page_number=0,
                confidence=1.0,  # Direct read = perfect confidence
                quality=OCRQuality.EXCELLENT,
                model_used=self.name,
                metadata={
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size,
                },
            )

        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return OCRResult(
                        content=content,
                        page_number=0,
                        confidence=0.9,  # Slightly lower due to encoding detection
                        quality=OCRQuality.GOOD,
                        model_used=self.name,
                        metadata={"encoding": encoding},
                    )
                except UnicodeDecodeError:
                    continue

            return OCRResult(
                content="",
                page_number=0,
                confidence=0.0,
                quality=OCRQuality.POOR,
                model_used=self.name,
                metadata={"error": "Could not decode file with any supported encoding"},
            )

        except Exception as e:
            return OCRResult(
                content="",
                page_number=0,
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
        Extract text from a text/markdown file.

        Returns a single-element list since text files are treated as one page.

        Args:
            file_path: Path to text/markdown file.
            **kwargs: Ignored for text files.

        Returns:
            List with one OCRResult containing the file content.
        """
        result = await self.extract_page(file_path, 0, **kwargs)
        return [result]

    def get_page_count(self, file_path: Path) -> int:
        """
        Get the number of pages in a text file.

        Text files are always treated as a single page.

        Args:
            file_path: Path to text/markdown file.

        Returns:
            Always returns 1.
        """
        return 1
