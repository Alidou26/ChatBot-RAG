"""
Text chunker using LangChain's RecursiveCharacterTextSplitter for document chunking.
"""
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...config import TEXT_CHUNK_SIZE, CHUNK_OVERLAP
from ..parsers.base_parser import ParsedContent
from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class TextChunker(BaseChunker):
    """
    Character-based text chunker using recursive splitting.
    
    Uses LangChain's RecursiveCharacterTextSplitter to intelligently split text
    on natural boundaries (paragraphs, sentences, words) while maintaining
    chunk size constraints.
    
    Best for: PDF, DOCX, PPTX, TXT, Markdown, and other text documents.
    
    Examples:
        >>> from pathlib import Path
        >>> content = ParsedContent(
        ...     text="Para 1.\\n\\nPara 2.\\n\\nPara 3.",
        ...     content_type="text",
        ...     language="fr",
        ...     file_path=Path("doc.txt"),
        ...     metadata={"source": "test"}
        ... )
        >>> chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        >>> chunks = chunker.chunk(content)
        >>> len(chunks) > 0
        True
    """
    
    def __init__(
        self,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        """
        Initialize text chunker with LangChain splitter.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to try (in order of preference)
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # Default separators: try splitting on larger boundaries first
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                "! ",
                "? ",
                "; ",
                ", ",
                " ",     # Words
                ""       # Characters
            ]
        
        self.separators = separators
        
        # Initialize LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.debug(
            f"Initialized TextChunker: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={len(separators)}"
        )
    
    def chunk(self, content: ParsedContent) -> List[Chunk]:
        """
        Split text content into overlapping chunks.
        
        Args:
            content: ParsedContent object with text to chunk
            
        Returns:
            List of Chunk objects with metadata
            
        Raises:
            ValueError: If content is empty or invalid
            
        Examples:
            >>> from pathlib import Path
            >>> content = ParsedContent(
            ...     text="Line 1\\nLine 2\\nLine 3",
            ...     content_type="text",
            ...     language="en",
            ...     file_path=Path("test.txt")
            ... )
            >>> chunker = TextChunker(chunk_size=10, chunk_overlap=2)
            >>> chunks = chunker.chunk(content)
            >>> all(isinstance(c, Chunk) for c in chunks)
            True
        """
        if not content.text or not content.text.strip():
            raise ValueError(f"Cannot chunk empty text from {content.file_path}")
        
        # Clean and normalize text
        text = content.text.strip()
        
        try:
            # Split text using LangChain
            text_chunks = self.splitter.split_text(text)
            
            logger.info(
                f"Split {content.file_path.name} into {len(text_chunks)} chunks "
                f"(original length: {len(text)} chars)"
            )
            
            # Create Chunk objects with metadata
            chunks = []
            total_chunks = len(text_chunks)
            
            for idx, chunk_text in enumerate(text_chunks):
                metadata = self._create_chunk_metadata(
                    base_metadata=content.metadata,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    content_type=content.content_type,
                    language=content.language,
                    source=str(content.file_path),
                    chunk_size_chars=len(chunk_text)
                )
                
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking {content.file_path}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Directly split text without ParsedContent wrapper.
        
        Utility method for quick text splitting without full metadata.
        
        Args:
            text: Text string to split
            
        Returns:
            List of text chunks
            
        Examples:
            >>> chunker = TextChunker(chunk_size=50, chunk_overlap=10)
            >>> chunks = chunker.chunk_text("Hello world! " * 100)
            >>> len(chunks) > 1
            True
        """
        if not text or not text.strip():
            return []
        
        return self.splitter.split_text(text.strip())
