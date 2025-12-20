"""
Chunker factory for selecting appropriate chunking strategy.
"""
import logging
from typing import Optional

from ..parsers.base_parser import ParsedContent
from .base_chunker import BaseChunker
from .text_chunker import TextChunker
from .code_chunker import CodeChunker

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """
    Factory for creating appropriate chunker based on content type.
    
    Examples:
        >>> from pathlib import Path
        >>> factory = ChunkerFactory()
        >>> content = ParsedContent(
        ...     text="Some text",
        ...     content_type="text",
        ...     language="en",
        ...     file_path=Path("doc.txt")
        ... )
        >>> chunker = factory.get_chunker(content)
        >>> isinstance(chunker, TextChunker)
        True
    """
    
    def __init__(self):
        """Initialize chunker factory."""
        try:
            self.text_chunker = TextChunker()
            logger.debug("✓ Initialized TextChunker")
        except Exception as e:
            logger.error(f"Failed to initialize TextChunker: {e}")
            self.text_chunker = None
        
        try:
            self.code_chunker = CodeChunker()
            logger.debug("✓ Initialized CodeChunker")
        except Exception as e:
            logger.error(f"Failed to initialize CodeChunker: {e}")
            self.code_chunker = None
        
        logger.info("✓ ChunkerFactory initialized")
    
    def get_chunker(self, content: ParsedContent) -> Optional[BaseChunker]:
        """
        Get appropriate chunker for parsed content.
        
        Args:
            content: ParsedContent object
            
        Returns:
            Chunker instance appropriate for the content type
            
        Examples:
            >>> from pathlib import Path
            >>> factory = ChunkerFactory()
            >>> code_content = ParsedContent(
            ...     text="def foo(): pass",
            ...     content_type="code",
            ...     language="python",
            ...     file_path=Path("code.py")
            ... )
            >>> chunker = factory.get_chunker(code_content)
            >>> isinstance(chunker, CodeChunker)
            True
        """
        if content.content_type == "code":
            if self.code_chunker is None:
                logger.warning("CodeChunker not available, falling back to TextChunker")
                return self.text_chunker
            logger.debug(f"Selected CodeChunker for {content.file_path.name}")
            return self.code_chunker
        
        elif content.content_type == "text":
            if self.text_chunker is None:
                logger.error("TextChunker not available!")
                return None
            logger.debug(f"Selected TextChunker for {content.file_path.name}")
            return self.text_chunker
        
        else:
            logger.warning(
                f"Unknown content_type '{content.content_type}', "
                "defaulting to TextChunker"
            )
            return self.text_chunker


# Singleton instance
_chunker_factory: Optional[ChunkerFactory] = None


def get_chunker_factory() -> ChunkerFactory:
    """
    Get or create singleton chunker factory.
    
    Returns:
        Shared ChunkerFactory instance
    """
    global _chunker_factory
    
    if _chunker_factory is None:
        _chunker_factory = ChunkerFactory()
    
    return _chunker_factory


def get_chunker(content: ParsedContent) -> Optional[BaseChunker]:
    """
    Convenience function to get chunker for content.
    
    Args:
        content: ParsedContent object
        
    Returns:
        Appropriate chunker
        
    Examples:
        >>> from pathlib import Path
        >>> content = ParsedContent(
        ...     text="Hello",
        ...     content_type="text",
        ...     language="en",
        ...     file_path=Path("test.txt")
        ... )
        >>> chunker = get_chunker(content)
        >>> chunker is not None
        True
    """
    factory = get_chunker_factory()
    return factory.get_chunker(content)
