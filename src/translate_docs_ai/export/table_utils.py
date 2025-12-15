"""
Table utilities for export with RTL/LTR conversion.

Handles:
- HTML table to Markdown table conversion
- RTL to LTR table column reversal
- Dynamic language direction detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

# RTL language codes
RTL_LANGUAGES = {"ar", "he", "fa", "ur", "yi", "ps", "sd"}

# LTR language codes
LTR_LANGUAGES = {"en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko"}


@dataclass
class TableData:
    """Parsed table data with headers and rows."""

    headers: list[str]
    rows: list[list[str]]
    has_headers: bool = True


class TableAwareMarkdownConverter(MarkdownConverter):
    """Custom markdown converter that preserves tables better."""

    def convert_table(self, el, text, convert_as_inline):
        """Convert HTML table to markdown table."""
        return text

    def convert_tr(self, el, text, convert_as_inline):
        """Convert table row."""
        cells = el.find_all(["th", "td"])
        cell_texts = []
        for cell in cells:
            # Get text content, handling nested elements
            cell_text = cell.get_text(strip=True)
            # Escape pipe characters
            cell_text = cell_text.replace("|", "\\|")
            cell_texts.append(cell_text)

        return "| " + " | ".join(cell_texts) + " |\n"

    def convert_th(self, el, text, convert_as_inline):
        """Don't process th separately - handled by tr."""
        return ""

    def convert_td(self, el, text, convert_as_inline):
        """Don't process td separately - handled by tr."""
        return ""


def is_rtl_language(lang_code: str) -> bool:
    """Check if a language code is RTL."""
    return lang_code.lower() in RTL_LANGUAGES


def is_ltr_language(lang_code: str) -> bool:
    """Check if a language code is LTR."""
    return lang_code.lower() in LTR_LANGUAGES


def reverse_table_columns(table_data: TableData) -> TableData:
    """
    Reverse table column order for RTL to LTR conversion.

    When translating from RTL (Arabic) to LTR (English), tables need
    their columns reversed to maintain proper reading order.
    """
    reversed_headers = list(reversed(table_data.headers))
    reversed_rows = [list(reversed(row)) for row in table_data.rows]

    return TableData(
        headers=reversed_headers,
        rows=reversed_rows,
        has_headers=table_data.has_headers,
    )


def parse_html_table(html: str) -> TableData | None:
    """Parse an HTML table into structured data."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if not table:
        return None

    headers: list[str] = []
    rows: list[list[str]] = []
    has_headers = False

    # Try to find header row
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
            has_headers = True

    # Get all rows from tbody or directly from table
    tbody = table.find("tbody") or table

    for tr in tbody.find_all("tr"):
        # Check if this is a header row (first row with th elements)
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        # If first row has th elements and we don't have headers yet
        if not headers and tr.find("th"):
            headers = [cell.get_text(strip=True) for cell in cells]
            has_headers = True
            continue

        # Regular data row
        row_data = [cell.get_text(strip=True) for cell in cells]
        rows.append(row_data)

    # If still no headers, use first row as headers
    if not headers and rows:
        headers = rows[0]
        rows = rows[1:]
        has_headers = True

    return TableData(headers=headers, rows=rows, has_headers=has_headers)


def table_data_to_markdown(table_data: TableData) -> str:
    """Convert TableData to markdown table format."""
    if not table_data.headers and not table_data.rows:
        return ""

    lines = []

    # Header row
    if table_data.headers:
        lines.append("| " + " | ".join(table_data.headers) + " |")
        # Separator row
        lines.append("| " + " | ".join(["---"] * len(table_data.headers)) + " |")

    # Data rows
    for row in table_data.rows:
        # Pad row if needed
        while len(row) < len(table_data.headers):
            row.append("")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def html_table_to_markdown(html: str, reverse_for_ltr: bool = False) -> str:
    """
    Convert HTML table to markdown table.

    Args:
        html: HTML string containing a table
        reverse_for_ltr: If True, reverse columns (for RTL to LTR conversion)

    Returns:
        Markdown table string
    """
    table_data = parse_html_table(html)
    if not table_data:
        return html  # Return original if no table found

    if reverse_for_ltr:
        table_data = reverse_table_columns(table_data)

    return table_data_to_markdown(table_data)


def process_content_tables(
    content: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    Process all tables in content for language direction conversion.

    - Converts HTML tables to Markdown
    - Reverses column order when converting from RTL to LTR

    Args:
        content: The markdown/HTML content
        source_lang: Source language code (e.g., 'ar')
        target_lang: Target language code (e.g., 'en')

    Returns:
        Processed content with proper table formatting
    """
    # Determine if we need to reverse columns
    source_is_rtl = is_rtl_language(source_lang)
    target_is_ltr = is_ltr_language(target_lang)
    reverse_columns = source_is_rtl and target_is_ltr

    # Find all HTML tables in content
    table_pattern = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

    def replace_table(match: re.Match) -> str:
        html_table = match.group(0)
        md_table = html_table_to_markdown(html_table, reverse_for_ltr=reverse_columns)
        return md_table

    processed = table_pattern.sub(replace_table, content)
    return processed


def process_markdown_tables_for_rtl_to_ltr(content: str) -> str:
    """
    Process existing markdown tables and reverse columns for RTL to LTR.

    This handles markdown tables that are already in markdown format
    but need column reversal.
    """
    lines = content.split("\n")
    result_lines = []
    in_table = False
    table_lines: list[str] = []

    for line in lines:
        # Detect markdown table line
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table:
            # End of table
            if table_lines:
                reversed_table = _reverse_markdown_table(table_lines)
                result_lines.extend(reversed_table)
                table_lines = []
            in_table = False
            result_lines.append(line)
        else:
            result_lines.append(line)

    # Handle table at end of content
    if table_lines:
        reversed_table = _reverse_markdown_table(table_lines)
        result_lines.extend(reversed_table)

    return "\n".join(result_lines)


def _reverse_markdown_table(table_lines: list[str]) -> list[str]:
    """Reverse columns in a markdown table."""
    result = []

    for line in table_lines:
        # Skip separator line (|---|---|)
        if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
            # Reverse separator columns too
            parts = line.split("|")
            # parts[0] and parts[-1] are empty strings from split
            inner_parts = parts[1:-1]
            reversed_parts = list(reversed(inner_parts))
            result.append("|" + "|".join(reversed_parts) + "|")
        else:
            # Regular row
            parts = line.split("|")
            inner_parts = parts[1:-1]
            reversed_parts = list(reversed(inner_parts))
            result.append("|" + "|".join(reversed_parts) + "|")

    return result
