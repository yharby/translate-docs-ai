"""
Base classes and interfaces for OCR providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OCRQuality(str, Enum):
    """Quality assessment of OCR result."""

    EXCELLENT = "excellent"  # 95%+ confidence
    GOOD = "good"  # 80-95% confidence
    FAIR = "fair"  # 60-80% confidence
    POOR = "poor"  # Below 60%
    UNKNOWN = "unknown"


@dataclass
class OCRResult:
    """Result from OCR processing."""

    content: str
    page_number: int
    confidence: float | None = None
    quality: OCRQuality = OCRQuality.UNKNOWN
    model_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check if content is empty or whitespace only."""
        return not self.content or not self.content.strip()

    @property
    def word_count(self) -> int:
        """Count words in content."""
        return len(self.content.split()) if self.content else 0

    def assess_quality(self) -> OCRQuality:
        """Assess quality based on confidence and content."""
        if self.confidence is not None:
            if self.confidence >= 0.95:
                return OCRQuality.EXCELLENT
            elif self.confidence >= 0.80:
                return OCRQuality.GOOD
            elif self.confidence >= 0.60:
                return OCRQuality.FAIR
            else:
                return OCRQuality.POOR

        # Heuristic assessment if no confidence score
        if self.is_empty:
            return OCRQuality.POOR

        # Check for common OCR artifacts
        content = self.content
        artifacts = ["???", "###", "...", "   "]
        artifact_count = sum(content.count(a) for a in artifacts)

        if artifact_count > len(content) * 0.1:
            return OCRQuality.POOR
        elif artifact_count > len(content) * 0.05:
            return OCRQuality.FAIR

        return OCRQuality.GOOD


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    async def extract_page(
        self,
        file_path: Path,
        page_number: int,
        **kwargs: Any,
    ) -> OCRResult:
        """
        Extract text from a single page.

        Args:
            file_path: Path to the document.
            page_number: Page number (0-indexed).
            **kwargs: Provider-specific options.

        Returns:
            OCRResult with extracted text.
        """
        ...

    @abstractmethod
    async def extract_document(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> list[OCRResult]:
        """
        Extract text from all pages in a document.

        Args:
            file_path: Path to the document.
            **kwargs: Provider-specific options.

        Returns:
            List of OCRResult, one per page.
        """
        ...

    def can_handle(self, file_path: Path) -> bool:
        """Check if this provider can handle the given file type."""
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        return file_path.suffix.lower() in supported


def assess_text_quality(text: str) -> tuple[OCRQuality, float]:
    """
    Assess the quality of extracted text using heuristics.

    Returns:
        Tuple of (quality enum, confidence score 0-1)
    """
    if not text or not text.strip():
        return OCRQuality.POOR, 0.0

    # Count various quality indicators
    total_chars = len(text)
    word_count = len(text.split())

    if word_count == 0:
        return OCRQuality.POOR, 0.0

    # Average word length (good text: 4-8 chars)
    avg_word_len = total_chars / word_count

    # Count problematic patterns
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / total_chars

    # Repeated character sequences (OCR artifacts)
    repeated = 0
    for i in range(len(text) - 2):
        if text[i] == text[i + 1] == text[i + 2]:
            repeated += 1

    repeated_ratio = repeated / total_chars if total_chars > 0 else 0

    # Calculate confidence score
    confidence = 1.0

    # Penalize unusual word lengths
    if avg_word_len < 2 or avg_word_len > 15:
        confidence -= 0.2

    # Penalize high special character ratio
    if special_char_ratio > 0.3:
        confidence -= 0.3
    elif special_char_ratio > 0.2:
        confidence -= 0.1

    # Penalize repeated characters
    if repeated_ratio > 0.1:
        confidence -= 0.3
    elif repeated_ratio > 0.05:
        confidence -= 0.15

    confidence = max(0.0, min(1.0, confidence))

    # Map to quality enum
    if confidence >= 0.85:
        quality = OCRQuality.EXCELLENT
    elif confidence >= 0.70:
        quality = OCRQuality.GOOD
    elif confidence >= 0.50:
        quality = OCRQuality.FAIR
    else:
        quality = OCRQuality.POOR

    return quality, confidence
