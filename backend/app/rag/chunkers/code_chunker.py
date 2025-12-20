"""
Code chunker for semantic chunking based on functions and classes.
"""
import logging
from typing import List

from ...config import CODE_CHUNK_SIZE
from ..parsers.base_parser import ParsedContent
from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class CodeChunker(BaseChunker):
    """
    Semantic chunker for source code.
    
    Chunks code by semantic units (functions, classes) extracted by Tree-sitter.
    Preserves:
    - Complete function/class definitions
    - Associated comments
    - Context (imports, class definitions for methods)
    
    If a unit is too large, splits it intelligently while keeping signatures.
    
    Examples:
        >>> from pathlib import Path
        >>> content = ParsedContent(
        ...     text="def foo():\\n    pass\\n\\ndef bar():\\n    pass",
        ...     content_type="code",
        ...     language="python",
        ...     file_path=Path("test.py"),
        ...     metadata={'units': [
        ...         {'type': 'function', 'name': 'foo', 'start_line': 0, 'end_line': 1,
        ...          'text': 'def foo():\\n    pass', 'comments': []},
        ...         {'type': 'function', 'name': 'bar', 'start_line': 3, 'end_line': 4,
        ...          'text': 'def bar():\\n    pass', 'comments': []}
        ...     ]}
        ... )
        >>> chunker = CodeChunker(chunk_size=100)
        >>> chunks = chunker.chunk(content)
        >>> len(chunks) >= 2
        True
    """
    
    def __init__(self, chunk_size: int = CODE_CHUNK_SIZE):
        """
        Initialize code chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
        """
        # Code chunks don't use overlap (semantic boundaries are clear)
        super().__init__(chunk_size, chunk_overlap=0)
        logger.debug(f"Initialized CodeChunker with chunk_size={chunk_size}")
    
    def chunk(self, content: ParsedContent) -> List[Chunk]:
        """
        Chunk source code by semantic units.
        
        Args:
            content: ParsedContent from code parser
            
        Returns:
            List of Chunk objects, one per function/class (or split if too large)
        """
        if content.content_type != "code":
            raise ValueError(
                f"CodeChunker requires content_type='code', got '{content.content_type}'"
            )
        
        # Get semantic units from metadata
        units = content.metadata.get('units', [])
        
        if not units:
            # Fallback: treat entire file as one chunk
            logger.warning(
                f"No semantic units found in {content.file_path.name}, "
                "using entire file as single chunk"
            )
            return self._chunk_full_file(content)
        
        logger.info(
            f"Chunking {content.file_path.name} by {len(units)} semantic units"
        )
        
        chunks = []
        
        for idx, unit in enumerate(units):
            unit_chunks = self._chunk_unit(unit, content, idx, len(units))
            chunks.extend(unit_chunks)
        
        logger.info(
            f"✓ Created {len(chunks)} chunks from {len(units)} units in "
            f"{content.file_path.name}"
        )
        
        return chunks
    
    def _chunk_unit(
        self,
        unit: dict,
        content: ParsedContent,
        unit_index: int,
        total_units: int
    ) -> List[Chunk]:
        """
        Create chunk(s) from a single semantic unit.
        
        Args:
            unit: Unit metadata from code parser
            content: Original ParsedContent
            unit_index: Index of this unit
            total_units: Total number of units
            
        Returns:
            List of chunks (usually 1, or multiple if unit is very large)
        """
        # Combine comments and code
        comments = unit.get('comments', [])
        code_text = unit.get('text', '')
        
        full_text = '\n'.join(comments) + '\n' + code_text if comments else code_text
        full_text = full_text.strip()
        
        # Build metadata
        base_metadata = content.metadata.copy()
        base_metadata.update({
            'unit_type': unit.get('type'),
            'unit_name': unit.get('name', 'unknown'),
            'unit_start_line': unit.get('start_line'),
            'unit_end_line': unit.get('end_line'),
            'signature': unit.get('signature', ''),
            'has_comments': len(comments) > 0
        })
        
        # If unit fits in chunk size, return as single chunk
        if len(full_text) <= self.chunk_size:
            metadata = self._create_chunk_metadata(
                base_metadata=base_metadata,
                chunk_index=unit_index,
                total_chunks=total_units,
                content_type=content.content_type,
                language=content.language,
                source=str(content.file_path)
            )
            
            return [Chunk(text=full_text, metadata=metadata)]
        
        # Unit is too large - split intelligently
        logger.info(
            f"Unit '{unit.get('name')}' is large ({len(full_text)} chars), "
            "splitting..."
        )
        return self._split_large_unit(full_text, base_metadata, content)
    
    def _split_large_unit(
        self,
        text: str,
        base_metadata: dict,
        content: ParsedContent
    ) -> List[Chunk]:
        """
        Split a large code unit into smaller chunks.
        
        Tries to split on logical boundaries (blank lines, statement ends).
        Always includes signature in first chunk.
        """
        lines = text.split('\n')
        
        # Extract signature (first line or first few lines)
        signature_lines = []
        body_lines = []
        in_signature = True
        
        for line in lines:
            if in_signature:
                signature_lines.append(line)
                # Signature typically ends with { or : or )
                if any(char in line for char in ['{', ':', ')']) and len(line.strip()) < 100:
                    in_signature = False
            else:
                body_lines.append(line)
        
        signature = '\n'.join(signature_lines)
        
        # Now chunk the body
        chunks = []
        current_chunk = [signature]  # Start with signature
        current_size = len(signature)
        
        for line in body_lines:
            line_len = len(line) + 1  # +1 for newline
            
            if current_size + line_len > self.chunk_size and len(current_chunk) > 1:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                metadata = self._create_chunk_metadata(
                    base_metadata=base_metadata,
                    chunk_index=len(chunks),
                    total_chunks=-1,  # Unknown until finished
                    content_type=content.content_type,
                    language=content.language,
                    source=str(content.file_path),
                    is_partial=True
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
                
                # Start new chunk with signature
                current_chunk = [signature, line]
                current_size = len(signature) + line_len
            else:
                current_chunk.append(line)
                current_size += line_len
        
        # Add final chunk
        if len(current_chunk) > 1:  # More than just signature
            chunk_text = '\n'.join(current_chunk)
            metadata = self._create_chunk_metadata(
                base_metadata=base_metadata,
                chunk_index=len(chunks),
                total_chunks=len(chunks) + 1,
                content_type=content.content_type,
                language=content.language,
                source=str(content.file_path),
                is_partial=True
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))
        
        # Update total_chunks in all metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_full_file(self, content: ParsedContent) -> List[Chunk]:
        """
        Fallback: chunk entire file as single unit.
        
        Used when no semantic units are detected.
        """
        metadata = self._create_chunk_metadata(
            base_metadata=content.metadata,
            chunk_index=0,
            total_chunks=1,
            content_type=content.content_type,
            language=content.language,
            source=str(content.file_path),
            full_file=True
        )
        
        return [Chunk(text=content.text, metadata=metadata)]
