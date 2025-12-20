"""
Base parser module with ParsedContent dataclass and abstract BaseParser interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Literal


@dataclass
class ParsedContent:
    """
    Structured representation of parsed document content.
    
    Attributes:
        text: The main textual content extracted from the document
        content_type: Type of content - "code" for source code, "text" for documents
        language: Programming language (for code) or natural language (for text)
        file_path: Path to the source file
        metadata: Additional metadata about the content
        
    Examples:
        >>> content = ParsedContent(
        ...     text="def hello(): pass",
        ...     content_type="code",
        ...     language="python",
        ...     file_path=Path("example.py"),
        ...     metadata={"functions": ["hello"]}
        ... )
        >>> content.content_type
        'code'
    """
    text: str
    content_type: Literal["code", "text"]
    language: str
    file_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        if not self.text:
            raise ValueError("ParsedContent text cannot be empty")
        
        if self.content_type not in ("code", "text"):
            raise ValueError(f"content_type must be 'code' or 'text', got: {self.content_type}")
        
        # Ensure file_path is a Path object
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        
        # Add file metadata
        self.metadata.setdefault("file_name", self.file_path.name)
        self.metadata.setdefault("file_type", self.file_path.suffix)
        self.metadata.setdefault("file_size", self.file_path.stat().st_size if self.file_path.exists() else 0)


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    
    All parsers must implement the parse() method to extract content from files.
    Subclasses should handle specific file formats (PDF, DOCX, code, etc.).
    """
    
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse a file and extract its content.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedContent object containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or unsupported
            Exception: For parsing errors (encoding, corruption, etc.)
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Check if this parser supports the given file type.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if this parser can handle the file, False otherwise
            
        Examples:
            >>> parser = PDFParser()
            >>> parser.supports(Path("document.pdf"))
            True
            >>> parser.supports(Path("code.py"))
            False
        """
        pass
    
    def validate_file(self, file_path: Path) -> None:
        """
        Validate that the file exists and is readable.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        if not file_path.stat().st_size:
            raise ValueError(f"File is empty: {file_path}")
