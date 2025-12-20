"""
PPTX parser using Docling for PowerPoint presentation extraction.
"""
import logging
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

from .base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class PPTXParser(BaseParser):
    """
    Parser for Microsoft PowerPoint (.pptx) presentations using Docling.
    
    Extracts:
    - Slide text content
    - Titles and bullet points
    - Tables and notes
    
    Examples:
        >>> parser = PPTXParser()
        >>> parser.supports(Path("presentation.pptx"))
        True
    """
    
    def __init__(self):
        """Initialize PPTX parser with Docling converter."""
        if DocumentConverter is None:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )
        
        try:
            self.converter = DocumentConverter()
            logger.info("✓ Docling DocumentConverter initialized for PPTX")
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is a PPTX presentation."""
        return file_path.suffix.lower() == '.pptx'
    
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse PPTX presentation and extract text.
        
        Args:
            file_path: Path to PPTX file
            
        Returns:
            ParsedContent with extracted text and metadata
        """
        self.validate_file(file_path)
        
        if not self.supports(file_path):
            raise ValueError(f"Not a PPTX file: {file_path}")
        
        logger.info(f"Parsing PPTX: {file_path.name}")
        
        try:
            # Convert PPTX using Docling
            result = self.converter.convert(str(file_path))
            
            # Export to Markdown
            markdown_text = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                'parser': 'docling_pptx',
                'format': 'pptx'
            }
            
            # Try to extract slide count and title
            try:
                doc = result.document
                if hasattr(doc, 'title') and doc.title:
                    metadata['title'] = doc.title
                if hasattr(doc, 'pages'):
                    metadata['slide_count'] = len(doc.pages)
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
            logger.error(f"Error parsing PPTX {file_path}: {e}")
            raise
