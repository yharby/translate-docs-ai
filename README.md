# translate-docs-ai

AI-powered document translation pipeline with OCR, terminology extraction, and multi-format export.

## Features

- **Smart OCR**: Uses olmOCR-2 via DeepInfra for accurate text extraction from PDFs
- **Smart Terminology Extraction**: Hybrid approach combining frequency analysis with LLM-powered context-aware translation
- **Context-Aware Translation**: Page-by-page translation with previous/next page context via OpenRouter LLMs
- **Semi-Auto Mode**: Pause for terminology review with CSV export/import workflow
- **RTL/LTR Support**: Automatic table column reversal when translating between RTL (Arabic, Hebrew) and LTR (English) languages
- **Multi-Format Export**: Export to Markdown, PDF, and DOCX
- **Progress Tracking**: Rich CLI with progress bars and status displays
- **Checkpointing**: Resume interrupted processing with DuckDB-backed state management
- **LangGraph Pipeline**: Stateful workflow with terminology -> translation -> export stages

## Installation

### Quick Install (Recommended)

Install globally with uv tool - no cloning needed:

```bash
# Install permanently as a CLI tool
uv tool install translate-docs-ai --from git+https://github.com/yharby/translate-docs-ai.git

# Now use it from anywhere
translate-docs --help
translate-docs run config.yaml
```

### One-Time Run (No Install)

Run directly without installing:

```bash
uvx --from git+https://github.com/yharby/translate-docs-ai.git translate-docs --help
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yharby/translate-docs-ai.git
cd translate-docs-ai

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

1. Copy the example config:
```bash
cp config.example.yaml config.yaml
```

2. Set your API keys in `.env` or environment:
```bash
export DEEPINFRA_API_KEY="your-deepinfra-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

3. Edit `config.yaml` to customize:
```yaml
# Key settings
paths:
  input_dir: ./documents      # Directory with PDFs to translate
  output_dir: ./translated    # Output directory

translation:
  source_language: ar         # Source language (ar, en, fr)
  target_language: en         # Target language
  default_model: anthropic/claude-3.5-sonnet

ocr:
  force_ocr: true            # Use OCR even for native PDFs
  image_dpi: 300             # Higher = better quality, slower

export:
  markdown: true
  pdf: true
  docx: true
  auto_export: true
```

## Usage

### Simple One-Command Execution

```bash
# Run the full pipeline: scan -> translate -> export
uv run translate-docs run config.yaml
```

### Step-by-Step Commands

```bash
# 1. Scan documents
uv run translate-docs scan ./documents

# 2. View status
uv run translate-docs status

# 3. Translate (all pending documents)
uv run translate-docs translate --all

# 4. Export to different formats
uv run translate-docs export --format md --language en
uv run translate-docs export --format pdf --language en
uv run translate-docs export --format docx --language en
```

### Semi-Auto Mode (Terminology Review Workflow)

Semi-auto mode pauses after terminology extraction, exports terms to CSV for review, and waits for approval before continuing translation.

```bash
# 1. Start translation in semi-auto mode
uv run translate-docs translate --all --mode semi-auto --source ar --target en
# → Extracts terms, auto-translates them using LLM with context
# → Exports CSV to ./translated/terms_*.csv
# → Pauses and shows instructions

# 2. Review the CSV file
# - Open ./translated/terms_*.csv in a spreadsheet
# - Review: original_term_ar, auto_translation_en columns
# - Fill corrected_translation_en column where needed
# - Leave empty to approve auto-translation as-is

# 3. Import reviewed terms and continue
uv run translate-docs approve --doc 1 --import ./translated/terms_*.csv
# → Imports corrections, continues translation, exports results
```

**CSV Format:**
| term_id | original_term_ar | frequency | auto_translation_en | corrected_translation_en |
|---------|------------------|-----------|---------------------|--------------------------|
| 1 | الهيئة العامة | 17 | General Authority | |
| 2 | الجيومكانية | 15 | Geospatial | Geomatics |

## CLI Commands

| Command | Description |
|---------|-------------|
| `run <config.yaml>` | Run full pipeline with config file |
| `scan <directory>` | Scan and catalog documents |
| `status` | Show processing status |
| `translate` | Translate documents (supports `--mode semi-auto`) |
| `terms --doc N --export` | Export terminology to CSV for review |
| `approve --doc N --import <csv>` | Import reviewed terms and continue |
| `export` | Export translations to MD/PDF/DOCX |
| `logs` | View processing logs |
| `init` | Generate default config.yaml |

## Architecture

```
+------------------+     +-------------+     +--------------+
|   config.yaml    | --> |   Scanner   | --> | OCR Pipeline |
+------------------+     +-------------+     +--------------+
                               |                    |
                               v                    v
                         +----------+         +-----------+
                         |  DuckDB  | <------ | Terminology|
                         +----------+         +-----------+
                               |
                               v
                    +---------------------+
                    | Translation         |
                    | (LangGraph +        |
                    |  OpenRouter)        |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | Export              |
                    | (MD, PDF, DOCX)     |
                    +---------------------+
```

### Key Components

- **OCR**: olmOCR-2 via DeepInfra for document digitization
- **Database**: DuckDB with FTS and VSS extensions for terminology management
- **Translation**: OpenRouter API with Claude 3.5 Sonnet (configurable)
- **Pipeline**: LangGraph for stateful, checkpointable workflows
- **Export**: Markdown, PDF (markdown-pdf), DOCX (python-docx)

## Smart Terminology Extraction

The terminology extraction uses a **hybrid token-efficient approach**:

1. **Frequency-based extraction** (no API cost)
   - Identifies candidate terms using n-gram frequency analysis
   - Filters common words with stop word lists
   - Stores context snippets where each term appears

2. **LLM-powered context-aware translation** (minimal tokens)
   - Only sends the term list with context to the LLM (not full document)
   - Uses sentence context to determine correct translation
   - Example: "bank" with context "river bank" translates correctly

3. **Embeddings for semantic search** (local model)
   - Generates embeddings using sentence-transformers
   - Enables semantic similarity search for term lookup

**Configuration:**
```yaml
terminology:
  min_frequency: 3              # Minimum occurrences to extract
  use_llm_translation: true     # Enable LLM-powered translation
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

## RTL/LTR Table Handling

When translating between right-to-left and left-to-right languages, tables are automatically handled:

1. **At Translation Time**: The LLM is instructed to reverse table column order
2. **At Export Time**: Fallback processing converts HTML tables to Markdown and reverses columns

Example: Arabic table `[No. | Name | Description]` becomes English `[Description | Name | No.]`

## Project Structure

```
translate-docs-ai/
   src/translate_docs_ai/
      cli.py              # CLI entry point
      config.py           # YAML config management
      database.py         # DuckDB operations
      scanner.py          # Document scanner
      ocr/
         deepinfra.py    # olmOCR via DeepInfra
         pymupdf.py      # Native PDF extraction
      terminology/
         extractor.py    # Frequency-based term extraction
         llm_extractor.py # LLM-powered context-aware translation
         embeddings.py   # Vector embeddings
      translation/
         pipeline.py     # LangGraph workflow
         translator.py   # OpenRouter client
         context.py      # Context builder
      export/
          markdown.py     # MD exporter
          pdf.py          # PDF exporter
          docx.py         # DOCX exporter
          table_utils.py  # RTL/LTR table handling
   config.yaml             # Main configuration
   config.example.yaml     # Example configuration
   pyproject.toml          # Project dependencies
```

## Dependencies

- Python 3.12+
- DuckDB 1.2+
- PyMuPDF (fitz)
- LangGraph
- OpenAI SDK (for DeepInfra and OpenRouter)
- Rich (CLI)
- python-docx, markdown-pdf (export)

## API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| DeepInfra | OCR (olmOCR-2) | [deepinfra.com](https://deepinfra.com) |
| OpenRouter | Translation LLMs | [openrouter.ai](https://openrouter.ai) |

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Credits

- [olmOCR](https://github.com/allenai/olmocr) by Allen AI
- [LangGraph](https://langchain-ai.github.io/langgraph/) by LangChain
- [DuckDB](https://duckdb.org/) for data management
