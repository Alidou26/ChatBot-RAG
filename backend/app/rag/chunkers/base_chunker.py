"""
Base chunker module with abstract interface for text chunking strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

from ..parsers.base_parser import ParsedContent


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    
    Attributes:
        text: The chunk text content
        metadata: Metadata dictionary including chunk_index, source info, etc.
        
    Examples:
        >>> chunk = Chunk(
        ...     text="def hello(): pass",
        ...     metadata={"chunk_index": 0, "source": "test.py"}
        ... )
        >>> chunk.metadata["chunk_index"]
        0
    """
    text: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    
    Subclasses implement different chunking approaches:
    - Character-based chunking for text documents
    - Semantic chunking for code (function/class boundaries)
    - Custom chunking for specific file types
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            
        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, content: ParsedContent) -> List[Chunk]:
        """
        Split parsed content into chunks.
        
        Args:
            content: ParsedContent object to chunk
            
        Returns:
            List of Chunk objects
            
        Raises:
            ValueError: If content is invalid or empty
        """
        pass
    
    def _create_chunk_metadata(
        self,
        base_metadata: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        **extra
    ) -> Dict[str, Any]:
        """
        Create metadata for a chunk by combining base metadata with chunk-specific info.
        
        Args:
            base_metadata: Metadata from the parsed content
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks
            **extra: Additional chunk-specific metadata
            
        Returns:
            Complete metadata dictionary
            
        Examples:
            >>> chunker = TextChunker(1000, 100)
            >>> base = {"file_name": "test.py", "language": "python"}
            >>> meta = chunker._create_chunk_metadata(base, 0, 5)
            >>> meta["chunk_index"]
            0
            >>> meta["total_chunks"]
            5
        """
        metadata = base_metadata.copy()
        metadata.update({
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
        })
        metadata.update(extra)
        return metadata
