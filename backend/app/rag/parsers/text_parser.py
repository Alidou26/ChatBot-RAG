"""
Plain text parser for simple text-based formats.

This parser handles plain text files such as ``.txt`` and ``.md``,
and also lightweight structured data like JSON and CSV.  The goal
is to extract human-readable content from files that do not require
heavy parsing libraries.  For example, JSON objects are serialized
into pretty‑printed strings and CSV files are converted into a
tabular text representation.

Examples::

    >>> parser = TextParser()
    >>> parser.supports(Path("notes.txt"))
    True
    >>> content = parser.parse(Path("notes.txt"))
    >>> content.text.startswith("This is the first line")
    True

    >>> parser.supports(Path("data.json"))
    True
    >>> content = parser.parse(Path("data.json"))
    >>> '"name":' in content.text
    True

"""

from __future__ import annotations

import json
import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict

from .base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """
    Parser for plain text and lightweight structured formats.

    Supports the following extensions:

    - ``.txt``: Reads as UTF‑8 text.
    - ``.md``: Markdown files are treated as plain text.  Front‑matter
      or header decorations are preserved for context.
    - ``.json``: JSON files are loaded and dumped into a pretty‑printed
      string with indentation for readability.  Complex nested
      structures are preserved.
    - ``.csv``: CSV files are converted into a tabular string using
      Python's ``csv`` module.  The header row appears once followed
      by each row on a new line.  The delimiter is auto‑detected but
      defaults to comma.
    - ``.tsv``: Tab‑separated values are treated similarly to CSV but
      with a tab delimiter.

    Other text-like formats can be added by extending the
    ``_read_file`` method.
    """

    # A broader set of plain‑text extensions.  In addition to simple
    # .txt and Markdown files, we support lightweight markup and
    # configuration formats commonly used in projects.  YAML, TOML,
    # INI/CFG, XML and script files (.sh, .bat) are read as plain
    # text.  Wiki pages and reStructuredText (.wiki, .rst) are also
    # treated as text.  These files typically contain human‑readable
    # documentation or configuration that should be indexed.
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".wiki", ".rst",
        ".json", ".csv", ".tsv", ".yaml", ".yml",
        ".ini", ".cfg", ".conf", ".toml",
        ".xml", ".properties", ".cmake", ".gitignore",
        ".sh", ".bat"
    }

    def supports(self, file_path: Path) -> bool:
        """Return True if the file has a supported text-based extension.

        We normalise the suffix to lower case and check membership in
        ``SUPPORTED_EXTENSIONS``.  Unknown extensions are not
        supported.  Note that files without an extension are not
        considered text by default and must be handled by other parsers.
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse a text file and return its content.

        Args:
            file_path: Path to the file to parse.

        Returns:
            ParsedContent containing the extracted text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or unsupported.
            Exception: For unexpected read errors.
        """
        self.validate_file(file_path)

        if not self.supports(file_path):
            raise ValueError(f"Unsupported text file type: {file_path.suffix}")

        logger.info(f"Parsing text file: {file_path.name}")

        try:
            text = self._read_file(file_path)
            metadata: Dict[str, Any] = {
                "parser": "text_parser",
                "format": file_path.suffix.lower().lstrip("."),
            }
            return ParsedContent(
                text=text,
                content_type="text",
                language="unknown",
                file_path=file_path,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            raise

    def _read_file(self, file_path: Path) -> str:
        """
        Internal helper to read supported text formats.

        Returns a string containing the human‑readable content of the file.
        """
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".wiki", ".rst",
                      ".yaml", ".yml", ".ini", ".cfg", ".conf", ".toml",
                      ".xml", ".properties", ".cmake", ".gitignore",
                      ".sh", ".bat"}:
            # Read plain text / Markdown / wiki / config / script as UTF‑8
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            # Pretty‑print JSON
            return json.dumps(data, indent=2, ensure_ascii=False)
        if suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            rows = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for row in reader:
                    rows.append(row)
            # Convert to tabular text: join by delimiter but preserve structure
            output = io.StringIO()
            writer = csv.writer(output, delimiter=delimiter)
            writer.writerows(rows)
            return output.getvalue()
        # Fallback: read as binary and decode, ignoring errors
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")