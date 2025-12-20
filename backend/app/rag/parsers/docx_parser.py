"""
DOCX parser using Docling for Word document extraction.
"""
import logging
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

from .base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class DOCXParser(BaseParser):
    """
    Parser for Microsoft Word (.docx) documents using Docling.
    
    Preserves document structure including:
    - Headings and styles
    - Tables and lists
    - Text formatting
    
    Examples:
        >>> parser = DOCXParser()
        >>> parser.supports(Path("document.docx"))
        True
    """
    
    def __init__(self):
        """Initialize DOCX parser with Docling converter."""
        if DocumentConverter is None:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )
        
        try:
            self.converter = DocumentConverter()
            logger.info("✓ Docling DocumentConverter initialized for DOCX")
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is a DOCX document."""
        return file_path.suffix.lower() == '.docx'
    
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse DOCX document and extract text with structure.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            ParsedContent with extracted text and metadata
        """
        self.validate_file(file_path)
        
        if not self.supports(file_path):
            raise ValueError(f"Not a DOCX file: {file_path}")
        
        logger.info(f"Parsing DOCX: {file_path.name}")
        
        try:
            # Convert DOCX using Docling
            result = self.converter.convert(str(file_path))
            
            # Export to Markdown
            markdown_text = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                'parser': 'docling_docx',
                'format': 'docx'
            }
            
            # Try to extract title and author
            try:
                doc = result.document
                if hasattr(doc, 'title') and doc.title:
                    metadata['title'] = doc.title
                if hasattr(doc, 'author') and doc.author:
                    metadata['author'] = doc.author
            except Exception as e:
                logger.debug(f"Could not extract full metadata: {e}")
            
            logger.info(f"✓ Parsed {file_path.name}: {len(markdown_text)} chars")
            
            return ParsedContent(
                text=markdown_text,
                content_type="text",
                language="fr",
                file_path=file_path,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            raise
