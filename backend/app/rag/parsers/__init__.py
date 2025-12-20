"""
Parser factory for automatic parser selection based on file type.
"""
import logging
from pathlib import Path
from typing import Optional

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .pptx_parser import PPTXParser
from .code_parser import CodeParser
from .image_parser import ImageParser
from .text_parser import TextParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating appropriate parser instances based on file type.
    
    Examples:
        >>> factory = ParserFactory()
        >>> parser = factory.get_parser(Path("document.pdf"))
        >>> isinstance(parser, PDFParser)
        True
        
        >>> parser = factory.get_parser(Path("code.cpp"))
        >>> isinstance(parser, CodeParser)
        True
    """
    
    def __init__(self):
        """Initialize parser factory with all available parsers."""
        self._parsers = []
        self._parser_cache = {}
        
        # Try to initialize each parser
        self._register_parser(PDFParser, "PDF")
        self._register_parser(DOCXParser, "DOCX")
        self._register_parser(PPTXParser, "PPTX")
        self._register_parser(CodeParser, "Code")
        self._register_parser(ImageParser, "Image")
        self._register_parser(TextParser, "Text")
        
        logger.info(f"✓ ParserFactory initialized with {len(self._parsers)} parsers")
    
    def _register_parser(self, parser_class, name: str) -> None:
        """
        Try to register a parser class.
        
        Args:
            parser_class: Parser class to instantiate
            name: Parser name for logging
        """
        try:
            parser_instance = parser_class()
            self._parsers.append(parser_instance)
            logger.debug(f"✓ Registered {name} parser")
        except ImportError as e:
            logger.warning(f"⚠ {name} parser not available: {e}")
        except Exception as e:
            logger.error(f"✗ Failed to register {name} parser: {e}")
    
    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """
        Get appropriate parser for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Parser instance that supports the file, or None if no parser found
            
        Examples:
            >>> factory = ParserFactory()
            >>> parser = factory.get_parser(Path("example.pdf"))
            >>> parser is not None
            True
            
            >>> parser = factory.get_parser(Path("unknown.xyz"))
            >>> parser is None
            True
        """
        # Check cache first
        ext = file_path.suffix.lower()
        if ext in self._parser_cache:
            return self._parser_cache[ext]
        
        # Find suitable parser
        for parser in self._parsers:
            if parser.supports(file_path):
                self._parser_cache[ext] = parser
                logger.debug(f"Selected {parser.__class__.__name__} for {file_path.name}")
                return parser
        
        logger.warning(f"No parser found for file: {file_path}")
        return None
    
    def supports(self, file_path: Path) -> bool:
        """
        Check if any parser supports this file type.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if a parser is available for this file
        """
        return self.get_parser(file_path) is not None
    
    def list_supported_extensions(self) -> list:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        extensions = set()
        test_files = {
            '.pdf': 'test.pdf',
            '.docx': 'test.docx',
            '.pptx': 'test.pptx',
            '.cpp': 'test.cpp',
            '.py': 'test.py',
            '.java': 'test.java',
            '.js': 'test.js',
            '.png': 'test.png',
            '.jpg': 'test.jpg',
        }
        
        for ext, filename in test_files.items():
            if self.supports(Path(filename)):
                extensions.add(ext)
        
        return sorted(extensions)


# Singleton instance
_parser_factory: Optional[ParserFactory] = None


def get_parser_factory() -> ParserFactory:
    """
    Get or create singleton parser factory.
    
    Returns:
        Shared ParserFactory instance
    """
    global _parser_factory
    
    if _parser_factory is None:
        _parser_factory = ParserFactory()
    
    return _parser_factory


def get_parser(file_path: Path) -> Optional[BaseParser]:
    """
    Convenience function to get parser for a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Appropriate parser or None
        
    Examples:
        >>> parser = get_parser(Path("document.pdf"))
        >>> parser is not None
        True
    """
    factory = get_parser_factory()
    return factory.get_parser(file_path)
