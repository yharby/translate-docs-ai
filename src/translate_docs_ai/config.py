"""
Configuration management for translate-docs-ai.

Handles loading configuration from YAML files and environment variables.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file if present (before Settings initialization)
load_dotenv()


class ProcessingMode(str, Enum):
    """Processing mode for the translation pipeline."""

    AUTO = "auto"
    SEMI_AUTO = "semi-auto"


class OCRModel(str, Enum):
    """Available OCR models."""

    PYMUPDF = "pymupdf4llm"
    OLMOCR = "allenai/olmOCR-2-7B-1025"
    DEEPSEEK = "deepseek-ai/DeepSeek-OCR"


class LLMProvider(str, Enum):
    """Available LLM providers."""

    OPENROUTER = "openrouter"
    CLAUDE_CODE = "claude-code"


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    input_dir: Path = Field(default=Path("./documents"))
    output_dir: Path = Field(default=Path("./translated"))
    database_path: Path = Field(default=Path("./translate_docs.db"))
    checkpoints: Path = Field(default=Path("./data/checkpoints"))
    logs: Path = Field(default=Path("./logs"))

    @field_validator("input_dir", "output_dir", "database_path", "checkpoints", "logs")
    @classmethod
    def expand_path(cls, v: Path) -> Path:
        """Expand user home directory and make path absolute."""
        return Path(v).expanduser().resolve()


class OCRConfig(BaseModel):
    """Configuration for OCR processing."""

    primary_model: OCRModel = Field(default=OCRModel.OLMOCR)
    fallback_model: OCRModel = Field(default=OCRModel.DEEPSEEK)
    native_pdf_extractor: str = Field(default="pymupdf4llm")
    # Force OCR even for native PDFs (skip pymupdf text extraction)
    force_ocr: bool = Field(default=False)
    image_dpi: int = Field(default=150, ge=72, le=600)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=60, ge=10, le=300)
    deepinfra_api_key: str = Field(default="")


class ContextConfig(BaseModel):
    """Configuration for translation context."""

    include_previous_page: bool = Field(default=True)
    include_next_page_preview: bool = Field(default=True)
    max_context_tokens: int = Field(default=500, ge=100, le=2000)


class SystemPrompts(BaseModel):
    """System prompts for different document types."""

    default: str = Field(
        default="""You are a professional translator specializing in technical documents.
Maintain the original formatting and markdown structure.
Use the provided terminology glossary for consistency."""
    )
    technical: str = Field(
        default="""You are translating technical documentation.
Preserve code blocks, commands, and technical terms.
Use the terminology glossary provided."""
    )
    legal: str = Field(
        default="""You are translating legal documents.
Maintain precise legal terminology.
Do not paraphrase legal terms."""
    )
    scientific: str = Field(
        default="""You are translating scientific papers.
Preserve mathematical notation, citations, and technical terminology.
Maintain academic writing style."""
    )


class TranslationConfig(BaseModel):
    """Configuration for translation."""

    # LLM Provider selection: "openrouter" (pay-per-token) or "claude-code" (subscription)
    provider: LLMProvider = Field(default=LLMProvider.OPENROUTER)
    default_model: str = Field(default="anthropic/claude-3.5-sonnet")
    target_language: str = Field(default="ar")
    source_language: str = Field(default="en")
    context: ContextConfig = Field(default_factory=ContextConfig)
    system_prompts: SystemPrompts = Field(default_factory=SystemPrompts)
    max_tokens_per_request: int = Field(default=4096, ge=256, le=32000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    # OpenRouter API key (only required if provider is "openrouter")
    openrouter_api_key: str = Field(default="")
    min_term_frequency: int = Field(default=3, ge=1, le=100)
    max_terms: int = Field(default=500, ge=10, le=5000)


class SemiAutoConfig(BaseModel):
    """Configuration for semi-automatic processing mode."""

    review_terminology: bool = Field(default=True)
    review_term_translations: bool = Field(default=True)
    review_page_translations: bool = Field(default=False)
    export_terminology_csv: bool = Field(default=True)
    terminology_csv_path: Path = Field(default=Path("./review/terminology.csv"))
    reviewed_terminology_path: Path = Field(default=Path("./review/terminology_reviewed.csv"))
    review_timeout: int = Field(default=0)  # 0 = no timeout


class AutoConfig(BaseModel):
    """Configuration for automatic processing mode."""

    continue_on_error: bool = Field(default=True)
    auto_approve_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class ProcessingConfig(BaseModel):
    """Configuration for processing pipeline."""

    mode: ProcessingMode = Field(default=ProcessingMode.AUTO)
    batch_size: int = Field(default=10, ge=1, le=100)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=5.0, ge=1.0, le=60.0)
    concurrent_pages: int = Field(default=4, ge=1, le=20)
    semi_auto: SemiAutoConfig = Field(default_factory=SemiAutoConfig)
    auto: AutoConfig = Field(default_factory=AutoConfig)


class TerminologyConfig(BaseModel):
    """Configuration for terminology extraction."""

    min_frequency: int = Field(default=3, ge=1, le=100)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_terms_per_document: int = Field(default=500, ge=10, le=5000)
    # Use LLM to translate extracted terms (token-efficient: only sends term list)
    use_llm_translation: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    file: Path = Field(default=Path("./logs/translation.log"))
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    backup_count: int = Field(default=5, ge=1, le=20)


class ExportConfig(BaseModel):
    """Configuration for export formats."""

    # Enable/disable export formats
    markdown: bool = Field(default=True)
    pdf: bool = Field(default=True)
    docx: bool = Field(default=True)

    # Markdown-specific options
    markdown_combined: bool = Field(default=True)  # Single file vs separate per page

    # Auto-export after translation completes
    auto_export: bool = Field(default=True)

    # Languages to export (if empty, exports target_language)
    languages: list[str] = Field(default_factory=list)

    # Clean export: no metadata headers, page numbers, footers, or separators
    # Produces output matching the source document's formatting
    clean: bool = Field(default=False)


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = Field(default="translation-project")
    description: str = Field(default="")


class Settings(BaseSettings):
    """Main settings class that combines all configurations."""

    model_config = SettingsConfigDict(
        env_prefix="",  # No prefix for simpler env vars
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Configuration sections
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    terminology: TerminologyConfig = Field(default_factory=TerminologyConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def __init__(self, **data: Any) -> None:
        """Initialize with environment variable fallbacks for API keys."""
        super().__init__(**data)
        # Override API keys from environment if not set in config
        if not self.ocr.deepinfra_api_key:
            self.ocr.deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY", "")
        if not self.translation.openrouter_api_key:
            self.translation.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

    @classmethod
    def from_yaml(cls, path: Path | str) -> Settings:
        """Load settings from a YAML file, with environment variable overrides."""
        path = Path(path)
        if not path.exists():
            # Return defaults if file doesn't exist
            return cls()

        with open(path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Process environment variable substitutions in YAML values
        yaml_config = _substitute_env_vars(yaml_config)

        return cls(**yaml_config)


def _substitute_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute ${ENV_VAR} patterns in config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _substitute_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            result[key] = os.getenv(env_var, "")
        elif isinstance(value, list):
            result[key] = [
                _substitute_env_vars(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


def load_config(path: Path | str | None = None) -> Settings:
    """
    Load configuration from YAML file or return defaults.

    Args:
        path: Path to YAML config file. If None, looks for config.yaml in current directory.

    Returns:
        Settings instance with merged YAML and environment configurations.
    """
    if path is None:
        # Look for config.yaml in current directory
        default_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(".config.yaml"),
            Path(".translate-docs.yaml"),
        ]
        for p in default_paths:
            if p.exists():
                path = p
                break

    if path is not None:
        return Settings.from_yaml(path)

    return Settings()


def create_default_config(path: Path | str = "config.yaml") -> None:
    """Create a default configuration file."""
    default_config = """# translate-docs-ai configuration
project:
  name: "my-translation-project"
  description: "Document translation project"

paths:
  input_directory: "./input_docs"
  output_directory: "./output_translated"
  database: "./data/translation.duckdb"
  checkpoints: "./data/checkpoints"
  logs: "./logs"

ocr:
  # Primary OCR model for scanned documents
  primary_model: "allenai/olmOCR-2-7B-1025"
  # Fallback for complex layouts
  fallback_model: "deepseek-ai/DeepSeek-OCR"
  # Native PDF text extractor
  native_pdf_extractor: "pymupdf4llm"
  # Image DPI for OCR processing
  image_dpi: 150

translation:
  # OpenRouter model (use "openrouter/auto" for automatic selection)
  model: "openrouter/auto"
  # Target language code (ar, en, fr, es, de, etc.)
  target_language: "ar"
  # Source language code
  source_language: "en"

  context:
    # Include previous page summary for context
    include_previous_page: true
    # Include next page preview for context
    include_next_page_preview: true
    # Maximum tokens for context
    max_context_tokens: 500

  system_prompts:
    default: |
      You are a professional translator specializing in technical documents.
      Maintain the original formatting and markdown structure.
      Use the provided terminology glossary for consistency.

terminology:
  # Minimum occurrences to extract a term
  min_frequency: 3
  # Embedding model for semantic similarity
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  # Similarity threshold for term clustering
  similarity_threshold: 0.85
  # Use LLM to auto-translate extracted terms (efficient: only sends term list)
  use_llm_translation: true

processing:
  # Mode: "auto" or "semi-auto"
  mode: "auto"
  # Number of pages to process in a batch
  batch_size: 10
  # Maximum retry attempts
  max_retries: 3
  # Delay between retries (seconds)
  retry_delay: 5
  # Concurrent page processing
  concurrent_pages: 4

  semi_auto:
    # Pause for terminology review
    review_terminology: true
    # Pause for term translation review
    review_term_translations: true
    # Pause for page translation review (slower)
    review_page_translations: false
    # Export terminology to CSV
    export_terminology_csv: true
    terminology_csv_path: "./review/terminology.csv"

  auto:
    # Continue processing on non-critical errors
    continue_on_error: true
    # Auto-approve confidence threshold
    auto_approve_confidence_threshold: 0.8

logging:
  level: "INFO"
  file: "./logs/translation.log"
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(default_config)
