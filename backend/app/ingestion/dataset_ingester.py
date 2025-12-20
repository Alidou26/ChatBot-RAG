from __future__ import annotations

import os
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..rag.indexer import get_indexer
from ..rag.parsers import ParserFactory

def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract an archive (.tar.gz or .zip) into the given directory.

    Args:
        archive_path: Path to the archive file.
        dest_dir: Directory where contents will be extracted.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If the file is not a supported archive format.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    suffixes = archive_path.suffixes
    # support .tar.gz
    if suffixes[-2:] == ['.tar', '.gz']:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    # support .zip
    elif suffixes[-1] == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(path=dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

def _collect_files(root_dir: Path, parser_factory: ParserFactory) -> List[Path]:
    supported_files: List[Path] = []
    IGNORED_DIRS = {
        "node_modules", "vendor", "deps", ".git", "__pycache__", "obj"
    }
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_parts = Path(dirpath).parts
        if any(part in IGNORED_DIRS for part in current_parts):
            continue
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() == ".js" and fpath.name.startswith("chunk-vendors"):
                continue
            if parser_factory.supports(fpath):
                supported_files.append(fpath)
    return supported_files

def ingest_archive(
    archive_path: Path,
    project_id: str,
    project_name: str,
    collection_name: str = "redmine_projects"
) -> int:
    """Ingest a .tar.gz or .zip archive into ChromaDB using DocumentIndexer."""

    metadata: Dict[str, Any] = {
        'project_id': project_id,
        'project_name': project_name,
    }

    parser_factory = ParserFactory()
    indexer = get_indexer(collection_name=collection_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Extract archive (tar.gz or zip)
        _extract_archive(archive_path, tmp_path)

        # Collect supported files
        supported_files = _collect_files(tmp_path, parser_factory)
        if not supported_files:
            return 0

        try:
            initial_count = indexer.chroma_collection.count()
        except Exception:
            initial_count = 0

        index = indexer.index_files(supported_files, metadata)

        try:
            new_count = indexer.chroma_collection.count()
        except Exception:
            new_count = initial_count

        return max(new_count - initial_count, 0)
