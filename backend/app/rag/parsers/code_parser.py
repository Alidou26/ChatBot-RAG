"""
Code parser using Tree-sitter for semantic code parsing.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import tree_sitter_cpp as tscpp
    import tree_sitter_python as tspython
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser, Node
except ImportError:
    tscpp = tspython = tsjava = tsjavascript = None
    Language = Parser = Node = None

from ...config import LANGUAGE_MAPPING, SUPPORTED_CODE_EXTENSIONS
from .base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class CodeParser(BaseParser):
    """
    Parser for source code files using Tree-sitter.
    
    Supports: C++, Python, Java, JavaScript
    
    Extracts semantic units:
    - Functions and methods
    - Classes and structures
    - Associated comments
    - Metadata: function names, signatures, line numbers
    
    Examples:
        >>> parser = CodeParser()
        >>> parser.supports(Path("example.cpp"))
        True
        >>> parser.supports(Path("document.pdf"))
        False
    """
    
    # Query patterns for different languages
    QUERIES = {
        'cpp': """
            (function_definition) @function
            (class_specifier) @class
            (struct_specifier) @struct
        """,
        'python': """
            (function_definition) @function
            (class_definition) @class
        """,
        'java': """
            (method_declaration) @method
            (class_declaration) @class
            (interface_declaration) @interface
        """,
        'javascript': """
            (function_declaration) @function
            (class_declaration) @class
            (method_definition) @method
        """
    }
    
    def __init__(self):
        """Initialize Tree-sitter parsers for supported languages."""
        if Parser is None:
            raise ImportError(
                "Tree-sitter libraries not installed. "
                "Install with: pip install tree-sitter tree-sitter-cpp "
                "tree-sitter-python tree-sitter-java tree-sitter-javascript"
            )
        
        self.parsers = {}
        self.languages = {}
        
        # Initialize parsers for each language
        self._init_language('cpp', tscpp)
        self._init_language('python', tspython)
        self._init_language('java', tsjava)
        self._init_language('javascript', tsjavascript)
        
        logger.info(f"✓ Initialized Tree-sitter parsers for: {list(self.parsers.keys())}")
    
    def _init_language(self, lang_name: str, lang_module) -> None:
        """Initialize parser for a specific language."""
        if lang_module is None:
            logger.warning(f"Language module not available: {lang_name}")
            return
        
        try:
            language = Language(lang_module.language())
            parser = Parser(language)
            
            self.languages[lang_name] = language
            self.parsers[lang_name] = parser
            
            logger.debug(f"✓ Initialized {lang_name} parser")
        except Exception as e:
            logger.error(f"Failed to initialize {lang_name} parser: {e}")
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is a supported code file."""
        ext = file_path.suffix.lower()
        return ext in SUPPORTED_CODE_EXTENSIONS
    
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse source code file and extract semantic structure.
        
        Args:
            file_path: Path to source code file
            
        Returns:
            ParsedContent with code and metadata about functions/classes
        """
        self.validate_file(file_path)
        
        if not self.supports(file_path):
            raise ValueError(f"Unsupported code file: {file_path}")
        
        # Determine language
        ext = file_path.suffix.lower()
        language = LANGUAGE_MAPPING.get(ext)
        
        if language not in self.parsers:
            raise ValueError(f"No parser available for language: {language}")
        
        logger.info(f"Parsing code file: {file_path.name} ({language})")
        
        try:
            # Read file with encoding handling
            code_bytes = self._read_file_safe(file_path)
            
            # Parse code
            parser = self.parsers[language]
            tree = parser.parse(code_bytes)
            
            # Extract semantic units
            code_text = code_bytes.decode('utf-8')
            units = self._extract_code_units(tree, code_text, language)
            
            # Build metadata
            metadata = {
                'parser': 'tree_sitter',
                'format': 'code',
                'language': language,
                'units_count': len(units),
                'units': units,
                'has_syntax_errors': tree.root_node.has_error
            }
            
            logger.info(
                f"✓ Parsed {file_path.name}: {len(code_text)} chars, "
                f"{len(units)} semantic units"
            )
            
            return ParsedContent(
                text=code_text,
                content_type="code",
                language=language,
                file_path=file_path,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing code file {file_path}: {e}")
            raise
    
    def _read_file_safe(self, file_path: Path) -> bytes:
        """
        Read file with encoding fallback.
        
        Try UTF-8 first, then Latin-1 as fallback.
        """
        try:
            return file_path.read_bytes()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _extract_code_units(
        self,
        tree,
        code_text: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Extract code units (functions, classes, etc.) from the syntax tree.
        
        Uses tree traversal instead of queries for compatibility.
        
        Args:
            tree: Tree-sitter syntax tree
            code_text: Source code text
            language: Programming language
            
        Returns:
            List of extracted code units with metadata
        """
        # Define node types to extract for each language
        target_types = {
            'cpp': ['function_definition', 'class_specifier', 'struct_specifier'],
            'python': ['function_definition', 'class_definition'],
            'java': ['method_declaration', 'class_declaration', 'interface_declaration'],
            'javascript': ['function_declaration', 'class_declaration', 'method_definition']
        }
        
        if language not in target_types:
            logger.warning(f"No target types defined for language: {language}")
            return []
        
        units = []
        types_to_find = target_types[language]
        
        def traverse(node):
            """Recursively traverse tree and collect matching nodes."""
            if node.type in types_to_find:
                # Determine capture name from node type
                capture_name = node.type.replace('_definition', '').replace('_declaration', '').replace('_specifier', '')
                unit = self._extract_unit_info(node, code_text, capture_name)
                if unit:
                    units.append(unit)
            
            # Recursively process children
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return units
    
    def _extract_unit_info(
        self,
        node: 'Node',
        code_text: str,
        unit_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract information about a code unit (function, class, etc.).
        
        Args:
            node: Tree-sitter node
            code_text: Full source code
            unit_type: Type of unit (function, class, etc.)
            
        Returns:
            Dictionary with unit metadata
        """
        try:
            start_byte = node.start_byte
            end_byte = node.end_byte
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            # Extract code text
            unit_text = code_text[start_byte:end_byte]
            
            # Try to extract name
            name = self._extract_name(node, code_text)
            
            # Look for preceding comments
            comments = self._extract_preceding_comments(node, code_text)
            
            return {
                'type': unit_type,
                'name': name,
                'start_line': start_line,
                'end_line': end_line,
                'start_byte': start_byte,
                'end_byte': end_byte,
                'text': unit_text,
                'comments': comments,
                'signature': unit_text.split('\n')[0][:100]  # First line as signature
            }
            
        except Exception as e:
            logger.debug(f"Could not extract unit info: {e}")
            return None
    
    def _extract_name(self, node: 'Node', code_text: str) -> Optional[str]:
        """Extract the name of a function/class from its node."""
        # Look for identifier child node
        for child in node.children:
            if 'identifier' in child.type or child.type == 'name':
                name_start = child.start_byte
                name_end = child.end_byte
                return code_text[name_start:name_end]
        
        return None
    
    def _extract_preceding_comments(
        self,
        node: 'Node',
        code_text: str
    ) -> List[str]:
        """
        Extract comments that precede a code unit.
        
        Args:
            node: Code unit node
            code_text: Source code
            
        Returns:
            List of comment strings
        """
        comments = []
        
        # Get previous sibling nodes
        current = node.prev_sibling
        
        while current:
            if 'comment' in current.type:
                comment_start = current.start_byte
                comment_end = current.end_byte
                comment_text = code_text[comment_start:comment_end]
                comments.insert(0, comment_text.strip())
                current = current.prev_sibling
            else:
                # Stop if we hit non-comment node
                break
        
        return comments
