"""
CLI for translate-docs-ai.

Provides commands for document scanning, OCR, terminology management,
translation, and export.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

# Suppress HuggingFace tokenizers parallelism warning when forking processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from translate_docs_ai.config import ProcessingMode, Settings, load_config
from translate_docs_ai.database import Database, Status

app = typer.Typer(
    name="translate-docs",
    help="AI-powered document translation with terminology management.",
    add_completion=False,
)

console = Console()


def _display_config(settings: Settings, config_path: Path | None) -> None:
    """Display the configuration being used."""
    config_source = str(config_path) if config_path else "default (config.yaml or built-in)"

    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Config file", config_source)
    config_table.add_row("Project", settings.project.name)
    config_table.add_row("", "")
    config_table.add_row("[bold]OCR Settings[/bold]", "")
    config_table.add_row("  Primary model", settings.ocr.primary_model.value)
    config_table.add_row("  Fallback model", settings.ocr.fallback_model.value)
    config_table.add_row("  Force OCR", str(settings.ocr.force_ocr))
    config_table.add_row("  Image DPI", str(settings.ocr.image_dpi))
    config_table.add_row(
        "  DeepInfra API", "configured" if settings.ocr.deepinfra_api_key else "[red]not set[/red]"
    )
    config_table.add_row("", "")
    config_table.add_row("Translation Settings", "", style="bold cyan")
    config_table.add_row("  Model", settings.translation.default_model)
    # Show fallback if configured
    if settings.translation.fallback_provider:
        config_table.add_row(
            "  Fallback",
            f"{settings.translation.fallback_provider.value} ({settings.translation.fallback_model or 'default'})",
            style="yellow",
        )
    config_table.add_row("  Source language", settings.translation.source_language)
    config_table.add_row("  Target language", settings.translation.target_language)
    config_table.add_row(
        "  OpenRouter API",
        "configured" if settings.translation.openrouter_api_key else "[red]not set[/red]",
    )

    console.print(
        Panel(config_table, title="[bold blue]translate-docs-ai[/bold blue]", border_style="blue")
    )


def get_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config file or defaults."""
    if config_path and config_path.exists():
        return load_config(config_path)
    return Settings()


def get_database(settings: Settings) -> Database:
    """Get database instance."""
    return Database(settings.paths.database_path)


@app.command()
def scan(
    input_dir: Path = typer.Argument(..., help="Directory to scan for documents"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r", help="Scan recursively"
    ),
) -> None:
    """Scan directory for documents and add to catalog."""
    settings = get_settings(config)
    db = get_database(settings)

    from translate_docs_ai.scanner import DocumentScanner

    scanner = DocumentScanner(db, console)

    try:
        documents = scanner.scan_directory(input_dir, recursive=recursive)
        console.print(f"\n[green]Scanned {len(documents)} documents[/green]")

        # Show summary table
        if documents:
            table = Table(title="Documents Found")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Pages", justify="right")
            table.add_column("Status", style="yellow")

            for doc in documents[:20]:  # Limit to 20
                table.add_row(
                    doc.file_name,
                    doc.file_type,
                    str(doc.total_pages),
                    doc.status.value,
                )

            if len(documents) > 20:
                table.add_row("...", "...", "...", "...")

            console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def status(
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
) -> None:
    """Show processing status for all documents."""
    settings = get_settings(config)
    db = get_database(settings)

    docs = db.get_all_documents()

    if not docs:
        console.print("[yellow]No documents in database[/yellow]")
        return

    # Status summary
    status_counts = {}
    for doc in docs:
        status_counts[doc.status.value] = status_counts.get(doc.status.value, 0) + 1

    console.print(
        Panel(
            "\n".join(f"{k}: {v}" for k, v in status_counts.items()),
            title="Document Status Summary",
        )
    )

    # Document table
    table = Table(title="Documents")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Pages", justify="right")
    table.add_column("Status", style="yellow")

    for doc in docs[:50]:
        status_style = {
            Status.PENDING: "yellow",
            Status.IN_PROGRESS: "blue",
            Status.COMPLETED: "green",
            Status.FAILED: "red",
        }.get(doc.status, "white")

        table.add_row(
            str(doc.id or ""),
            doc.file_name[:40],
            doc.file_type,
            str(doc.total_pages),
            f"[{status_style}]{doc.status.value}[/{status_style}]",
        )

    console.print(table)


@app.command()
def extract(
    document_id: int | None = typer.Option(None, "--doc", "-d", help="Document ID"),
    all_docs: bool = typer.Option(False, "--all", "-a", help="Process all pending"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Force OCR even for native PDFs"),
) -> None:
    """Extract text from documents using OCR."""
    settings = get_settings(config)
    db = get_database(settings)

    from translate_docs_ai.ocr import PyMuPDFExtractor

    pymupdf = PyMuPDFExtractor()

    # Get documents to process
    if document_id:
        doc = db.get_document(document_id)
        if not doc:
            console.print(f"[red]Document {document_id} not found[/red]")
            raise typer.Exit(1)
        documents = [doc]
    elif all_docs:
        documents = db.get_all_documents(Status.PENDING)
    else:
        console.print("[red]Specify --doc ID or --all[/red]")
        raise typer.Exit(1)

    if not documents:
        console.print("[yellow]No documents to process[/yellow]")
        return

    async def process_docs():
        for doc in documents:
            console.print(f"\n[cyan]Processing: {doc.file_name}[/cyan]")
            doc_path = Path(doc.full_path)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Extracting {doc.file_name}", total=doc.total_pages)

                for page_num in range(doc.total_pages):
                    result = await pymupdf.extract_page(doc_path, page_num)

                    from translate_docs_ai.database import Page

                    page = Page(
                        document_id=doc.id,
                        page_number=page_num,
                        original_content=result.content,
                        ocr_confidence=result.confidence,
                    )
                    db.add_page(page)

                    progress.advance(task)

            console.print(f"[green]Extracted {doc.total_pages} pages[/green]")

    asyncio.run(process_docs())


@app.command()
def terms(
    document_id: int = typer.Option(..., "--doc", "-d", help="Document ID"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    extract_new: bool = typer.Option(False, "--extract", "-e", help="Extract terminology"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all terms"),
    export_csv: Path | None = typer.Option(None, "--export", help="Export terms to CSV for review"),
    import_csv: Path | None = typer.Option(None, "--import", help="Import reviewed terms from CSV"),
) -> None:
    """Manage terminology for a document.

    Workflow for semi-auto terminology review:
    1. Run 'translate-docs translate --doc N --mode semi-auto' to extract terms
    2. Run 'translate-docs terms --doc N --export' to export CSV for review
    3. Edit the CSV file (fill corrected_translation column where needed)
    4. Run 'translate-docs approve --doc N --import terms_doc_N.csv' to continue
    """
    import csv

    settings = get_settings(config)
    db = get_database(settings)

    doc = db.get_document(document_id)
    if not doc:
        console.print(f"[red]Document {document_id} not found[/red]")
        raise typer.Exit(1)

    if extract_new:
        from translate_docs_ai.terminology import TerminologyExtractor

        extractor = TerminologyExtractor(
            db=db,
            min_frequency=settings.translation.min_term_frequency,
            max_terms=settings.translation.max_terms,
        )

        with console.status("Extracting terminology..."):
            terms_list = extractor.extract_from_document(document_id)

        console.print(f"[green]Extracted {len(terms_list)} terms[/green]")

    # Export to CSV for review
    if export_csv is not None:
        terms_list = db.get_document_terms(document_id)
        if not terms_list:
            console.print("[yellow]No terms to export[/yellow]")
            raise typer.Exit(1)

        # Determine source/target columns based on config
        source_lang = settings.translation.source_language
        target_lang = settings.translation.target_language

        # Determine export path - use output_dir if no path specified or if just --export flag
        if export_csv == Path("."):
            # --export was used without a path, use default in output directory
            output_dir = settings.paths.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(
                c if c.isalnum() or c in "._- " else "_" for c in Path(doc.file_name).stem
            )
            export_path = output_dir / f"terms_{safe_name}_doc{document_id}.csv"
        else:
            export_path = export_csv

        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header: term_id, original_term, frequency, auto_translation, corrected_translation
            writer.writerow(
                [
                    "term_id",
                    f"original_term_{source_lang}",
                    "frequency",
                    f"auto_translation_{target_lang}",
                    f"corrected_translation_{target_lang}",
                ]
            )

            for term in terms_list:
                # Get the auto-translation based on target language
                auto_trans = getattr(term, f"translation_{target_lang}", "") or ""
                writer.writerow(
                    [
                        term.id,
                        term.term,
                        term.frequency,
                        auto_trans,
                        "",  # Empty column for corrections
                    ]
                )

        # Show clear instructions
        console.print()
        console.print(
            Panel(
                f"[green]✓ Exported {len(terms_list)} terms[/green]\n\n"
                f"[bold]CSV file:[/bold] [cyan]{export_path.absolute()}[/cyan]\n\n"
                "[bold]Instructions:[/bold]\n"
                "1. Open the CSV file in a spreadsheet editor\n"
                f"2. Review the '{source_lang}' terms and their auto-translations\n"
                f"3. Fill '[bold]corrected_translation_{target_lang}[/bold]' column where needed\n"
                "4. Leave the correction column empty to approve auto-translation as-is\n"
                "5. Save the CSV file\n\n"
                f"[bold]Then run:[/bold]\n"
                f"  [cyan]translate-docs approve --doc {document_id} --import {export_path}[/cyan]",
                title="[bold blue]Terminology Review[/bold blue]",
                border_style="blue",
            )
        )
        return

    # Import from CSV
    if import_csv:
        if not import_csv.exists():
            console.print(f"[red]CSV file not found: {import_csv}[/red]")
            raise typer.Exit(1)

        target_lang = settings.translation.target_language
        updated = 0
        approved = 0

        with open(import_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term_id = int(row["term_id"])
                corrected = row.get(f"corrected_translation_{target_lang}", "").strip()

                # Get current term
                term = db.conn.execute(
                    "SELECT * FROM terminology WHERE id = ?", [term_id]
                ).fetchone()

                if not term:
                    continue

                # Update translation if corrected, otherwise mark as approved
                if corrected:
                    # User provided a correction
                    db.conn.execute(
                        f"UPDATE terminology SET translation_{target_lang} = ?, approved = TRUE WHERE id = ?",
                        [corrected, term_id],
                    )
                    updated += 1
                else:
                    # No correction = approve auto-translation as-is
                    db.conn.execute(
                        "UPDATE terminology SET approved = TRUE WHERE id = ?",
                        [term_id],
                    )
                approved += 1

        console.print(
            f"[green]Imported terminology: {approved} approved, {updated} corrected[/green]"
        )
        return

    # Display terms
    terms_list = db.get_document_terms(document_id)

    if not terms_list:
        console.print("[yellow]No terms found[/yellow]")
        return

    table = Table(title=f"Terminology - {doc.file_name}")
    table.add_column("Term", style="cyan")
    table.add_column("Freq", justify="right")
    table.add_column("AR Translation")
    table.add_column("Approved", justify="center")

    display_terms = terms_list if show_all else terms_list[:30]

    for term in display_terms:
        approved_mark = "[green]✓[/green]" if term.approved else ""
        table.add_row(
            term.term[:30],
            str(term.frequency),
            (term.translation_ar or "")[:30],
            approved_mark,
        )

    console.print(table)

    if len(terms_list) > 30 and not show_all:
        console.print(f"[dim]Showing 30 of {len(terms_list)} terms. Use --all to see all.[/dim]")

    console.print(
        f"\n[dim]To review terms: translate-docs terms --doc {document_id} --export terms.csv[/dim]"
    )


@app.command()
def translate(
    document_id: int | None = typer.Option(None, "--doc", "-d", help="Document ID"),
    all_docs: bool = typer.Option(False, "--all", "-a", help="Process all pending"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Processing mode: auto or semi-auto"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: openrouter (pay-per-token) or claude-code (subscription)",
    ),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Source language (default: from config)"
    ),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target language (default: from config)"
    ),
) -> None:
    """Translate documents."""
    from translate_docs_ai.config import LLMProvider

    settings = get_settings(config)
    db = get_database(settings)

    # Use config values if not specified on command line
    source_lang = source or settings.translation.source_language
    target_lang = target or settings.translation.target_language

    # Determine LLM provider (CLI flag overrides config)
    if provider:
        try:
            llm_provider = LLMProvider(provider.lower().replace("_", "-"))
        except ValueError:
            console.print(f"[red]Invalid provider: {provider}[/red]")
            console.print("Valid options: openrouter, claude-code")
            raise typer.Exit(1) from None
    else:
        llm_provider = settings.translation.provider

    # Display config being used
    _display_config(settings, config)

    # Validate API keys based on provider
    if llm_provider == LLMProvider.OPENROUTER and not settings.translation.openrouter_api_key:
        console.print("[red]OpenRouter API key not configured[/red]")
        console.print("Set OPENROUTER_API_KEY environment variable or add to config")
        console.print("[dim]Or use --provider claude-code to use your Claude subscription[/dim]")
        raise typer.Exit(1)
    elif llm_provider == LLMProvider.CLAUDE_CODE:
        console.print(
            "[green]Using Claude Code provider (subscription-based, no per-token cost)[/green]"
        )

    # Get documents
    if document_id:
        doc = db.get_document(document_id)
        if not doc:
            console.print(f"[red]Document {document_id} not found[/red]")
            raise typer.Exit(1)
        documents = [doc]
    elif all_docs:
        documents = db.get_all_documents(Status.PENDING) + db.get_all_documents(Status.IN_PROGRESS)
    else:
        console.print("[red]Specify --doc ID or --all[/red]")
        raise typer.Exit(1)

    if not documents:
        console.print("[yellow]No documents to translate[/yellow]")
        return

    # Show documents to process
    console.print(f"\n[bold]Processing {len(documents)} document(s)[/bold]\n")

    # Setup pipeline
    from translate_docs_ai.translation.pipeline import PipelineConfig, TranslationPipeline

    processing_mode = ProcessingMode.SEMI_AUTO if mode == "semi-auto" else ProcessingMode.AUTO

    # Map OCR model names from config to short names used by DeepInfraOCR
    def map_ocr_model(model_name: str) -> str:
        """Map full model name to short name for DeepInfraOCR."""
        model_map = {
            "deepseek-ai/DeepSeek-OCR": "deepseek",
            "allenai/olmOCR-2-7B-1025": "olmocr",
            "pymupdf4llm": "pymupdf",
        }
        return model_map.get(model_name, model_name)

    pipeline_config = PipelineConfig(
        llm_provider=llm_provider,
        openrouter_api_key=settings.translation.openrouter_api_key,
        deepinfra_api_key=settings.ocr.deepinfra_api_key,
        # Fallback provider configuration
        fallback_provider=settings.translation.fallback_provider,
        fallback_api_key=settings.translation.openrouter_api_key,
        fallback_model=settings.translation.fallback_model,
        enable_fallback=settings.translation.enable_fallback,
        processing_mode=processing_mode,
        source_lang=source_lang,
        target_lang=target_lang,
        translation_model=settings.translation.default_model,
        # OCR settings from config
        ocr_dpi=settings.ocr.image_dpi,
        force_ocr=settings.ocr.force_ocr,
        ocr_primary_model=map_ocr_model(settings.ocr.primary_model.value),
        ocr_fallback_model=map_ocr_model(settings.ocr.fallback_model.value),
    )

    async def run_pipeline():
        # Pipeline stages for progress tracking
        stages = [
            ("init", "Initializing"),
            ("ocr", "Extracting text (OCR)"),
            ("terminology", "Extracting terminology"),
            ("translation", "Translating pages"),
            ("export", "Finalizing"),
        ]

        for doc_idx, doc in enumerate(documents):
            # Document header
            doc_info = Table(show_header=False, box=None, padding=(0, 1))
            doc_info.add_column("", style="bold")
            doc_info.add_column("")
            doc_info.add_row("Document", f"[cyan]{doc.file_name}[/cyan]")
            doc_info.add_row("Pages", str(doc.total_pages))
            doc_info.add_row("Progress", f"{doc_idx + 1}/{len(documents)}")
            console.print(Panel(doc_info, title="[bold]Processing[/bold]", border_style="cyan"))

            # Create progress bar for stages
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                # Overall stage progress
                stage_task = progress.add_task("[cyan]Starting pipeline...", total=len(stages))

                try:
                    pipeline = TranslationPipeline(db, pipeline_config)

                    # Update progress for init
                    progress.update(stage_task, description="[cyan]Initializing...")

                    state = await pipeline.process_document(doc.id)

                    # Update based on final state
                    final_stage = state.get("current_stage")
                    if final_stage:
                        stage_name = (
                            final_stage.value if hasattr(final_stage, "value") else str(final_stage)
                        )

                        # Find completed stage index
                        stage_idx = 0
                        for i, (name, _desc) in enumerate(stages):
                            if name == stage_name or stage_name in ("complete", "error"):
                                stage_idx = i + 1
                                break

                        # Update to final state
                        if stage_name == "complete":
                            progress.update(
                                stage_task,
                                completed=len(stages),
                                description="[green]Complete!",
                            )
                        elif stage_name == "error":
                            progress.update(
                                stage_task,
                                description="[red]Error occurred",
                            )
                        else:
                            # Find the stage description
                            for name, desc in stages:
                                if name == stage_name:
                                    progress.update(
                                        stage_task,
                                        completed=stage_idx,
                                        description=f"[yellow]{desc}",
                                    )
                                    break

                    # Show result summary
                    console.print()
                    if state["current_stage"].value == "complete":
                        # Show stats
                        pages_ocr = len(state.get("ocr_results", {}))
                        terms = state.get("terms_extracted", 0)
                        pages_translated = state.get("pages_translated", 0)

                        result_table = Table(show_header=False, box=None)
                        result_table.add_column("", style="dim")
                        result_table.add_column("")
                        result_table.add_row("OCR", f"{pages_ocr} pages extracted")
                        result_table.add_row("Terms", f"{terms} terms found")
                        result_table.add_row("Translation", f"{pages_translated} pages translated")

                        console.print(
                            Panel(
                                result_table,
                                title="[green]✓ Completed[/green]",
                                border_style="green",
                            )
                        )
                    elif state["current_stage"].value == "review":
                        # Auto-export terms to CSV for review
                        terms_count = state.get("terms_extracted", 0)
                        output_dir = settings.paths.output_dir
                        # Store review CSVs in translated/review/ subfolder
                        review_dir = output_dir / "review"
                        review_dir.mkdir(parents=True, exist_ok=True)
                        safe_name = "".join(
                            c if c.isalnum() or c in "._- " else "_"
                            for c in Path(doc.file_name).stem
                        )
                        csv_path = review_dir / f"terms_{safe_name}_doc{doc.id}.csv"

                        # Export terms to CSV
                        import csv as csv_module

                        terms_list = db.get_document_terms(doc.id)
                        if terms_list:
                            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv_module.writer(f)
                                writer.writerow(
                                    [
                                        "term_id",
                                        f"original_term_{source_lang}",
                                        "frequency",
                                        f"auto_translation_{target_lang}",
                                        f"corrected_translation_{target_lang}",
                                    ]
                                )
                                for term in terms_list:
                                    auto_trans = (
                                        getattr(term, f"translation_{target_lang}", "") or ""
                                    )
                                    writer.writerow(
                                        [term.id, term.term, term.frequency, auto_trans, ""]
                                    )

                        console.print(
                            Panel(
                                f"[yellow]⏸ Awaiting terminology review[/yellow]\n\n"
                                f"[bold]Document:[/bold] {doc.file_name}\n"
                                f"[bold]Terms extracted:[/bold] {terms_count}\n\n"
                                f"[bold]CSV file:[/bold] [cyan]{csv_path.absolute()}[/cyan]\n\n"
                                "[bold]Instructions:[/bold]\n"
                                f"1. Open the CSV file and review the {source_lang} terms\n"
                                f"2. Fill 'corrected_translation_{target_lang}' column where needed\n"
                                "3. Leave empty to approve auto-translation as-is\n"
                                "4. Save the file\n\n"
                                f"[bold]Then run:[/bold]\n"
                                f"  [cyan]translate-docs approve --doc {doc.id} --import {csv_path}[/cyan]",
                                title="[bold blue]Terminology Review Required[/bold blue]",
                                border_style="yellow",
                            )
                        )
                    else:
                        console.print(f"[red]✗ Failed: {doc.file_name}[/red]")
                        for error in state.get("errors", []):
                            console.print(f"  [red]• {error}[/red]")

                except Exception as e:
                    progress.update(stage_task, description="[red]Error")
                    console.print(f"\n[red]Error processing {doc.file_name}: {e}[/red]")

            console.print()  # Spacing between documents

    asyncio.run(run_pipeline())


@app.command()
def approve(
    document_id: int = typer.Option(..., "--doc", "-d", help="Document ID"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    import_csv: Path | None = typer.Option(
        None, "--import", "-i", help="Import reviewed terms from CSV before approving"
    ),
) -> None:
    """Approve terminology review and continue translation.

    Workflow:
    1. Run 'translate-docs translate --doc N --mode semi-auto' to OCR and extract terms
    2. Run 'translate-docs terms --doc N --export terms.csv' to export for review
    3. Edit CSV: fill 'corrected_translation' column where needed (leave empty to approve as-is)
    4. Run 'translate-docs approve --doc N --import terms.csv' to import and continue
    """
    import csv

    settings = get_settings(config)
    db = get_database(settings)

    doc = db.get_document(document_id)
    if not doc:
        console.print(f"[red]Document {document_id} not found[/red]")
        raise typer.Exit(1)

    # Import CSV if provided
    if import_csv:
        if not import_csv.exists():
            console.print(f"[red]CSV file not found: {import_csv}[/red]")
            raise typer.Exit(1)

        target_lang = settings.translation.target_language
        updated = 0
        approved = 0

        with open(import_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term_id = int(row["term_id"])
                corrected = row.get(f"corrected_translation_{target_lang}", "").strip()

                # Get current term
                term = db.conn.execute(
                    "SELECT * FROM terminology WHERE id = ?", [term_id]
                ).fetchone()

                if not term:
                    continue

                # Update translation if corrected, otherwise mark as approved
                if corrected:
                    db.conn.execute(
                        f"UPDATE terminology SET translation_{target_lang} = ?, approved = TRUE WHERE id = ?",
                        [corrected, term_id],
                    )
                    updated += 1
                else:
                    db.conn.execute(
                        "UPDATE terminology SET approved = TRUE WHERE id = ?",
                        [term_id],
                    )
                approved += 1

        console.print(
            f"[green]Imported: {approved} terms approved, {updated} corrections applied[/green]"
        )

    from translate_docs_ai.translation.pipeline import PipelineConfig, TranslationPipeline

    # Map OCR model names
    def map_ocr_model(model_name: str) -> str:
        model_map = {
            "deepseek-ai/DeepSeek-OCR": "deepseek",
            "allenai/olmOCR-2-7B-1025": "olmocr",
            "pymupdf4llm": "pymupdf",
        }
        return model_map.get(model_name, model_name)

    pipeline_config = PipelineConfig(
        llm_provider=settings.translation.provider,
        openrouter_api_key=settings.translation.openrouter_api_key,
        deepinfra_api_key=settings.ocr.deepinfra_api_key,
        # Fallback provider configuration
        fallback_provider=settings.translation.fallback_provider,
        fallback_api_key=settings.translation.openrouter_api_key,
        fallback_model=settings.translation.fallback_model,
        enable_fallback=settings.translation.enable_fallback,
        source_lang=settings.translation.source_language,
        target_lang=settings.translation.target_language,
        translation_model=settings.translation.default_model,
        ocr_dpi=settings.ocr.image_dpi,
        force_ocr=settings.ocr.force_ocr,
        ocr_primary_model=map_ocr_model(settings.ocr.primary_model.value),
        ocr_fallback_model=map_ocr_model(settings.ocr.fallback_model.value),
    )

    pipeline = TranslationPipeline(db, pipeline_config)

    async def approve_and_continue():
        await pipeline.approve_review(document_id)
        console.print(f"[green]Review approved for document {document_id}[/green]")
        console.print("[cyan]Continuing translation...[/cyan]\n")

        # Continue processing
        state = await pipeline.process_document(document_id, resume=True)

        if state["current_stage"].value == "complete":
            pages_translated = state.get("pages_translated", 0)
            console.print(f"[green]✓ Translation completed: {pages_translated} pages[/green]")
        else:
            stage = state["current_stage"].value
            console.print(f"[yellow]Current stage: {stage}[/yellow]")

    asyncio.run(approve_and_continue())


@app.command()
def logs(
    document_id: int | None = typer.Option(None, "--doc", "-d", help="Filter by document"),
    level: str | None = typer.Option(None, "--level", "-l", help="Filter by level"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max entries to show"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
) -> None:
    """View processing logs."""
    settings = get_settings(config)
    db = get_database(settings)

    # Build query
    where_clauses = []
    params = []

    if document_id:
        where_clauses.append("document_id = ?")
        params.append(document_id)

    if level:
        where_clauses.append("level = ?")
        params.append(level.upper())

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    logs_data = db.conn.execute(
        f"""
        SELECT created_at, level, stage, message, document_id
        FROM processing_log
        {where_sql}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()

    if not logs_data:
        console.print("[yellow]No log entries found[/yellow]")
        return

    table = Table(title="Processing Logs")
    table.add_column("Time", style="dim")
    table.add_column("Level")
    table.add_column("Stage", style="cyan")
    table.add_column("Message")
    table.add_column("Doc", justify="right")

    for row in logs_data:
        timestamp, lvl, stage, message, doc_id = row

        level_style = {
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
        }.get(lvl, "white")

        time_str = (
            str(timestamp).split("T")[1][:8] if "T" in str(timestamp) else str(timestamp)[:19]
        )

        table.add_row(
            time_str,
            f"[{level_style}]{lvl}[/{level_style}]",
            stage or "",
            (message or "")[:60],
            str(doc_id or ""),
        )

    console.print(table)


@app.command()
def init(
    output_path: Path = typer.Option(
        Path("config.yaml"),
        "--output",
        "-o",
        help="Output path for config file",
    ),
) -> None:
    """Generate a default configuration file."""
    default_config = """# translate-docs-ai configuration

# Path settings
paths:
  input_dir: ./documents
  output_dir: ./translated
  database_path: ./translate_docs.db

# OCR settings
ocr:
  default_provider: pymupdf
  deepinfra_model: olmocr
  dpi: 150
  force_ocr: false
  # deepinfra_api_key: ${DEEPINFRA_API_KEY}

# Translation settings
translation:
  default_model: anthropic/claude-3.5-sonnet
  source_language: en
  target_languages:
    - ar
  temperature: 0.3
  max_tokens: 8192
  min_term_frequency: 3
  max_terms: 500
  # openrouter_api_key: ${OPENROUTER_API_KEY}

# Terminology extraction settings
terminology:
  min_frequency: 3                    # Minimum occurrences to extract a term
  use_llm_translation: true           # Use LLM to auto-translate terms

# Processing mode: auto or semi_auto
processing:
  mode: auto                          # 'auto' or 'semi-auto' (pauses for terminology review)
  concurrent_pages: 1
  max_retries: 3

# Export settings
export:
  markdown: true
  pdf: true
  docx: true
  auto_export: true                   # Auto-export after translation completes

# Logging
logging:
  level: INFO
  json_file: ./logs/translate_docs.json
"""

    if output_path.exists():
        overwrite = typer.confirm(f"{output_path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Abort()

    output_path.write_text(default_config)
    console.print(f"[green]Created config file: {output_path}[/green]")
    console.print("\nEdit the file and set your API keys, then run:")
    console.print("  translate-docs scan ./documents --config config.yaml")


@app.command()
def stats(
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
) -> None:
    """Show processing statistics."""
    settings = get_settings(config)
    db = get_database(settings)

    stats_data = db.get_statistics()

    console.print(
        Panel(
            f"""
Documents: {stats_data["total_documents"]}
  - Completed: {stats_data["completed_documents"]}
  - In Progress: {stats_data["in_progress_documents"]}
  - Pending: {stats_data["pending_documents"]}
  - Failed: {stats_data["failed_documents"]}

Pages: {stats_data["total_pages"]}
  - Translated: {stats_data["translated_pages"]}

Terms: {stats_data["total_terms"]}
        """.strip(),
            title="Processing Statistics",
        )
    )


@app.command()
def export(
    document_id: int | None = typer.Option(None, "--doc", "-d", help="Document ID to export"),
    all_docs: bool = typer.Option(False, "--all", "-a", help="Export all completed documents"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    language: str = typer.Option("en", "--lang", "-l", help="Language to export (en, ar, fr)"),
    fmt: str = typer.Option("md", "--format", "-f", help="Export format: md, pdf, docx"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    combined: bool = typer.Option(
        True, "--combined/--separate", help="For markdown: single file or separate per page"
    ),
    clean: bool | None = typer.Option(
        None, "--clean/--no-clean", help="Clean export without metadata/page headers/footers"
    ),
) -> None:
    """Export translated documents to markdown, PDF, or DOCX files."""
    from translate_docs_ai.export import DOCXExporter, MarkdownExporter, PDFExporter

    settings = get_settings(config)
    db = get_database(settings)

    # Determine output directory
    out_dir = output_dir or settings.paths.output_dir

    # Get documents to export
    if document_id:
        doc = db.get_document(document_id)
        if not doc:
            console.print(f"[red]Document {document_id} not found[/red]")
            raise typer.Exit(1)
        documents = [doc]
    elif all_docs:
        documents = db.get_all_documents(Status.COMPLETED)
    else:
        console.print("[red]Specify --doc ID or --all[/red]")
        raise typer.Exit(1)

    if not documents:
        console.print("[yellow]No completed documents to export[/yellow]")
        return

    # Validate format
    fmt = fmt.lower()
    if fmt not in ("md", "pdf", "docx"):
        console.print(f"[red]Unsupported format: {fmt}. Use md, pdf, or docx[/red]")
        raise typer.Exit(1)

    format_name = {"pdf": "PDF", "docx": "DOCX", "md": "Markdown"}[fmt]

    # Determine clean mode: CLI flag overrides config
    use_clean = clean if clean is not None else settings.export.clean

    console.print(f"[cyan]Exporting to {format_name}...[/cyan]")
    if use_clean:
        console.print("[dim](clean mode: no metadata/page headers)[/dim]")
    console.print()

    # Helper function to sanitize filename for directory name
    def sanitize_dirname(name: str) -> str:
        """Create safe directory name from document filename."""
        stem = Path(name).stem
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in stem)

    # Export documents with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting...", total=len(documents))

        results = []
        for doc in documents:
            progress.update(task, description=f"Exporting {doc.file_name}")

            # Create document-specific output directory
            doc_dir = out_dir / sanitize_dirname(doc.file_name)
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Create exporter for this document's directory
            if fmt == "pdf":
                exporter = PDFExporter(db, doc_dir)
            elif fmt == "docx":
                exporter = DOCXExporter(db, doc_dir)
            else:
                exporter = MarkdownExporter(db, doc_dir)

            if fmt == "md":
                result = exporter.export_document(
                    doc, language=language, combined=combined, clean=use_clean
                )
            else:
                result = exporter.export_document(doc, language=language, clean=use_clean)

            results.append(result)
            progress.advance(task)

    # Display results
    exported_count = 0
    for result in results:
        if result.success:
            exported_count += 1
            # Show path relative to out_dir
            rel_path = result.output_path.relative_to(out_dir)
            console.print(f"  [green]✓[/green] {rel_path} ({result.pages_exported} pages)")
        else:
            console.print(f"  [yellow]⚠[/yellow] {result.document_name}: {result.error}")

    console.print(f"\n[green]Exported {exported_count} document(s) to {out_dir}[/green]")


@app.command()
def run(
    config_file: Path = typer.Argument(..., help="Path to config.yaml file"),
) -> None:
    """
    Run the full pipeline: scan, translate, and export.

    This is the simplest way to process documents - just provide a config file:

        translate-docs run config.yaml

    The config file defines input directory, output formats, languages, and all other settings.
    API keys are read from environment variables or .env file.
    """
    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    settings = load_config(config_file)
    db = get_database(settings)

    # Display config
    _display_config(settings, config_file)

    # Import LLMProvider for validation
    from translate_docs_ai.config import LLMProvider

    # Validate API keys based on provider
    llm_provider = settings.translation.provider
    if llm_provider == LLMProvider.OPENROUTER and not settings.translation.openrouter_api_key:
        console.print("[red]OpenRouter API key not configured[/red]")
        console.print("Set OPENROUTER_API_KEY environment variable or in .env file")
        console.print(
            "[dim]Or set provider: claude-code in config to use your Claude subscription[/dim]"
        )
        raise typer.Exit(1)
    elif llm_provider == LLMProvider.CLAUDE_CODE:
        console.print(
            "[green]Using Claude Code provider (subscription-based, no per-token cost)[/green]"
        )

    if settings.ocr.force_ocr and not settings.ocr.deepinfra_api_key:
        console.print("[red]DeepInfra API key not configured (required for OCR)[/red]")
        console.print("Set DEEPINFRA_API_KEY environment variable or in .env file")
        raise typer.Exit(1)

    # Step 1: Scan input directory
    console.print(f"\n[bold cyan]Step 1: Scanning {settings.paths.input_dir}[/bold cyan]\n")

    from translate_docs_ai.scanner import DocumentScanner

    scanner = DocumentScanner(db, console)

    try:
        documents = scanner.scan_directory(settings.paths.input_dir, recursive=True)
        console.print(f"[green]Found {len(documents)} document(s)[/green]\n")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    if not documents:
        console.print("[yellow]No documents to process[/yellow]")
        return

    from translate_docs_ai.export import DOCXExporter, MarkdownExporter, PDFExporter
    from translate_docs_ai.translation.pipeline import (
        PipelineConfig,
        ProgressInfo,
        TranslationPipeline,
    )

    def map_ocr_model(model_name: str) -> str:
        model_map = {
            "deepseek-ai/DeepSeek-OCR": "deepseek",
            "allenai/olmOCR-2-7B-1025": "olmocr",
            "pymupdf4llm": "pymupdf",
        }
        return model_map.get(model_name, model_name)

    pipeline_config = PipelineConfig(
        llm_provider=llm_provider,
        openrouter_api_key=settings.translation.openrouter_api_key,
        deepinfra_api_key=settings.ocr.deepinfra_api_key,
        # Fallback provider configuration
        fallback_provider=settings.translation.fallback_provider,
        fallback_api_key=settings.translation.openrouter_api_key,  # Use same OpenRouter key for fallback
        fallback_model=settings.translation.fallback_model,
        enable_fallback=settings.translation.enable_fallback,
        processing_mode=settings.processing.mode,
        source_lang=settings.translation.source_language,
        target_lang=settings.translation.target_language,
        translation_model=settings.translation.default_model,
        ocr_dpi=settings.ocr.image_dpi,
        force_ocr=settings.ocr.force_ocr,
        ocr_primary_model=map_ocr_model(settings.ocr.primary_model.value),
        ocr_fallback_model=map_ocr_model(settings.ocr.fallback_model.value),
    )

    # Helper function to sanitize filename for directory name
    def sanitize_dirname(name: str) -> str:
        """Create safe directory name from document filename."""
        stem = Path(name).stem
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in stem)

    # Helper function to check if document is already exported
    def is_exported(doc, base_output_dir: Path) -> bool:
        """Check if document has already been exported."""
        doc_dir = base_output_dir / sanitize_dirname(doc.file_name)
        if not doc_dir.exists():
            return False
        # Check if any export files exist
        export_files = list(doc_dir.glob("*.*"))
        return len(export_files) > 0

    # Helper function to export a single document to its own directory
    def export_document_to_dir(doc, base_output_dir: Path) -> None:
        """Export document to a directory named after the source file."""
        export_lang = settings.translation.target_language
        source_lang = settings.translation.source_language
        use_clean = settings.export.clean

        # Create document-specific output directory
        doc_dir = base_output_dir / sanitize_dirname(doc.file_name)
        doc_dir.mkdir(parents=True, exist_ok=True)

        exported_formats = []
        export_errors = []

        if settings.export.markdown:
            exporter = MarkdownExporter(db, doc_dir)
            result = exporter.export_document(
                doc,
                language=export_lang,
                combined=settings.export.markdown_combined,
                source_lang=source_lang,
                clean=use_clean,
            )
            if result.success:
                exported_formats.append("MD")
            elif result.error:
                export_errors.append(f"MD: {result.error}")

        if settings.export.pdf:
            exporter = PDFExporter(db, doc_dir)
            result = exporter.export_document(
                doc, language=export_lang, source_lang=source_lang, clean=use_clean
            )
            if result.success:
                exported_formats.append("PDF")
            elif result.error:
                export_errors.append(f"PDF: {result.error}")

        if settings.export.docx:
            exporter = DOCXExporter(db, doc_dir)
            result = exporter.export_document(
                doc, language=export_lang, source_lang=source_lang, clean=use_clean
            )
            if result.success:
                exported_formats.append("DOCX")
            elif result.error:
                export_errors.append(f"DOCX: {result.error}")

        if exported_formats:
            console.print(f"  [dim]Exported: {', '.join(exported_formats)} → {doc_dir.name}/[/dim]")
        if export_errors:
            for err in export_errors:
                console.print(f"  [yellow]⚠ {err}[/yellow]")

    out_dir = settings.paths.output_dir

    # Categorize documents by status
    completed_docs = [d for d in documents if d.status == Status.COMPLETED]
    in_progress_docs = [d for d in documents if d.status == Status.IN_PROGRESS]
    pending_docs = [d for d in documents if d.status == Status.PENDING]
    failed_docs = [d for d in documents if d.status == Status.FAILED]

    # Show status summary
    console.print(
        f"[dim]Status: {len(completed_docs)} completed, {len(in_progress_docs)} in-progress, "
        f"{len(pending_docs)} pending, {len(failed_docs)} failed[/dim]\n"
    )

    # Step 2a: Export any completed documents that haven't been exported yet
    if completed_docs and settings.export.auto_export:
        unexported = [d for d in completed_docs if not is_exported(d, out_dir)]
        if unexported:
            console.print(
                f"[bold cyan]Exporting {len(unexported)} previously completed document(s)...[/bold cyan]\n"
            )
            for doc in unexported:
                console.print(f"  [green]✓[/green] {doc.file_name}")
                export_document_to_dir(doc, out_dir)
            console.print()

    # Determine which documents need processing (in-progress first to resume, then pending)
    docs_to_process = in_progress_docs + pending_docs

    if not docs_to_process:
        console.print("[green]All documents already completed![/green]")
        if settings.export.auto_export:
            console.print(f"[green]Output saved to: {out_dir}[/green]")
        console.print("\n[bold green]Done![/bold green]")
        return

    # Step 2b: Process pending and in-progress documents
    console.print(f"[bold cyan]Step 2: Processing {len(docs_to_process)} document(s)[/bold cyan]\n")

    async def process_all():
        total_to_process = len(docs_to_process)
        for doc_idx, doc in enumerate(docs_to_process):
            # Show resuming status for in-progress documents
            status_hint = " [yellow](resuming)[/yellow]" if doc.status == Status.IN_PROGRESS else ""

            # Document header with compact info
            console.print(
                Panel(
                    f"[cyan]{doc.file_name}[/cyan]{status_hint}\n"
                    f"[dim]Pages: {doc.total_pages} | Document {doc_idx + 1}/{total_to_process}[/dim]",
                    title="[bold]Processing[/bold]",
                    border_style="cyan",
                )
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                TaskProgressColumn(),
                TextColumn("[dim]{task.fields[detail]}[/dim]"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                # Create task with detail field
                stage_task = progress.add_task("[cyan]Initializing...", total=100, detail="")

                # Map stages to progress percentages
                stage_progress = {
                    "init": 5,
                    "ocr": 40,
                    "terminology": 60,
                    "translation": 90,
                    "export": 95,
                    "complete": 100,
                }

                def make_progress_callback(task_id, stage_pct_map):
                    """Create progress callback with bound variables."""

                    def callback(info: ProgressInfo) -> None:
                        """Update progress display based on pipeline callback."""
                        # Build detail string (page info or other detail)
                        if info.page_current and info.page_total:
                            detail = f"page {info.page_current}/{info.page_total}"
                        else:
                            detail = info.detail or ""

                        # Get progress percentage based on stage
                        pct = stage_pct_map.get(info.stage, 0)

                        # Update progress bar
                        progress.update(
                            task_id,
                            completed=pct,
                            description=f"[cyan]{info.stage_display}",
                            detail=detail,
                        )

                    return callback

                progress_callback = make_progress_callback(stage_task, stage_progress)

                try:
                    pipeline = TranslationPipeline(db, pipeline_config)

                    # Process with callback
                    state = await pipeline.process_document(
                        doc.id, progress_callback=progress_callback
                    )

                    # Final update based on result
                    final_stage = state.get("current_stage")
                    if final_stage and hasattr(final_stage, "value"):
                        if final_stage.value == "complete":
                            progress.update(
                                stage_task,
                                completed=100,
                                description="[green]Complete!",
                                detail="",
                            )
                        elif final_stage.value == "review":
                            progress.update(
                                stage_task,
                                completed=60,
                                description="[yellow]Awaiting review",
                                detail="",
                            )
                        elif final_stage.value == "error":
                            progress.update(
                                stage_task,
                                description="[red]Error occurred",
                                detail="",
                            )

                    console.print()

                    # Handle semi-auto mode: pause for terminology review
                    if state["current_stage"].value == "review":
                        # Export terms to CSV for review
                        import csv as csv_module

                        terms_count = state.get("terms_extracted", 0)
                        source_lang = settings.translation.source_language
                        target_lang = settings.translation.target_language

                        safe_name = "".join(
                            c if c.isalnum() or c in "._- " else "_"
                            for c in Path(doc.file_name).stem
                        )
                        # Store review CSVs in translated/review/ subfolder
                        review_dir = out_dir / "review"
                        review_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = review_dir / f"terms_{safe_name}_doc{doc.id}.csv"

                        terms_list = db.get_document_terms(doc.id)
                        if terms_list:
                            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv_module.writer(f)
                                writer.writerow(
                                    [
                                        "term_id",
                                        f"original_term_{source_lang}",
                                        "frequency",
                                        f"auto_translation_{target_lang}",
                                        f"corrected_translation_{target_lang}",
                                    ]
                                )
                                for term in terms_list:
                                    auto_trans = (
                                        getattr(term, f"translation_{target_lang}", "") or ""
                                    )
                                    writer.writerow(
                                        [term.id, term.term, term.frequency, auto_trans, ""]
                                    )

                        # Show review panel with instructions
                        corr_col = f"corrected_translation_{target_lang}"
                        console.print(
                            Panel(
                                f"[yellow]⏸ Terminology Review Required[/yellow]\n\n"
                                f"[bold]Document:[/bold] {doc.file_name}\n"
                                f"[bold]Terms extracted:[/bold] {terms_count}\n\n"
                                f"[bold]CSV file:[/bold] [cyan]{csv_path.absolute()}[/cyan]\n\n"
                                "[bold]Instructions:[/bold]\n"
                                f"1. Open the CSV file and review the {source_lang} terms\n"
                                f"2. Fill '{corr_col}' column where needed\n"
                                "3. Leave empty to approve auto-translation as-is\n"
                                "4. Save the file and return here\n",
                                title="[bold blue]Semi-Auto Mode[/bold blue]",
                                border_style="yellow",
                            )
                        )

                        # Interactive prompt - wait for user to press Enter
                        console.print(
                            "[bold cyan]Press Enter to continue after reviewing the CSV "
                            "(or type 'skip' to skip this document):[/bold cyan] ",
                            end="",
                        )
                        user_input = input().strip().lower()

                        if user_input == "skip":
                            console.print("[yellow]⏭ Skipping document...[/yellow]\n")
                            continue

                        # Import corrections from CSV if user edited it
                        if csv_path.exists():
                            with open(csv_path, encoding="utf-8") as f:
                                reader = csv_module.DictReader(f)
                                corrections_imported = 0
                                for row in reader:
                                    corrected_col = f"corrected_translation_{target_lang}"
                                    if row.get(corrected_col, "").strip():
                                        term_id = int(row["term_id"])
                                        correction = row[corrected_col].strip()
                                        sql = (
                                            f"UPDATE terminology "
                                            f"SET translation_{target_lang} = ? WHERE id = ?"
                                        )
                                        db.conn.execute(sql, [correction, term_id])
                                        corrections_imported += 1

                            if corrections_imported > 0:
                                console.print(
                                    f"[green]✓ Imported {corrections_imported} "
                                    "correction(s) from CSV[/green]\n"
                                )

                        # Approve review and continue pipeline
                        console.print("[cyan]Continuing translation...[/cyan]\n")
                        await pipeline.approve_review(doc.id)

                        # Re-run pipeline to continue from where it left off
                        state = await pipeline.process_document(
                            doc.id, progress_callback=progress_callback
                        )

                    # Show result and export immediately if successful
                    if state["current_stage"].value == "complete":
                        pages_ocr = len(state.get("ocr_results", {}))
                        terms = state.get("terms_extracted", 0)
                        pages_translated = state.get("pages_translated", 0)

                        console.print(
                            f"[green]✓ Completed:[/green] "
                            f"[dim]{pages_ocr} pages OCR'd, {terms} terms, "
                            f"{pages_translated} pages translated[/dim]"
                        )

                        # Export immediately after document completes
                        if settings.export.auto_export:
                            # Refresh document from DB to get updated status
                            completed_doc = db.get_document(doc.id)
                            if completed_doc:
                                export_document_to_dir(completed_doc, out_dir)
                    elif state["current_stage"].value == "error":
                        console.print(f"[red]✗ Failed: {doc.file_name}[/red]")
                        for error in state.get("errors", []):
                            console.print(f"  [red]• {error}[/red]")

                except Exception as e:
                    progress.update(stage_task, description="[red]Error", detail="")
                    console.print(f"\n[red]Error: {e}[/red]")

            console.print()

    asyncio.run(process_all())

    # Summary
    if settings.export.auto_export:
        console.print(f"[green]Output saved to: {settings.paths.output_dir}[/green]")

    console.print("\n[bold green]Done![/bold green]")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
