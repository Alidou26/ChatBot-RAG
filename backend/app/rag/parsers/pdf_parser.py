"""
PDF parser using Docling for structure-preserving document extraction.
"""
import logging
from pathlib import Path
from typing import Optional  # (laisse pour compat)

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

from ...rag.parsers.base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    Parser for PDF documents using Docling.
    
    Docling preserves document structure including:
    - Headings and hierarchy
    - Tables and lists
    - Text formatting
    - Page metadata
    
    Exports to Markdown format for better downstream processing.
    
    Examples:
        >>> parser = PDFParser()
        >>> parser.supports(Path("document.pdf"))
        True
        >>> parser.supports(Path("code.py"))
        False
    """
    
    def __init__(self):
        """Initialize PDF parser with Docling converter."""
        if DocumentConverter is None:
            # Comportement d'origine conservé : on lève si Docling absent
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )
        try:
            self.converter = DocumentConverter()
            logger.info("✓ Docling DocumentConverter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise
    
    def supports(self, file_path: Path) -> bool:
        """
        Check if file is a PDF.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file extension is .pdf
        """
        return file_path.suffix.lower() == '.pdf'
    
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse a PDF document and extract text.

        Cette version garde la logique et les noms, mais :
        - essaie Docling, PyMuPDF/pdfminer et OCR,
        - compare la longueur des textes,
        - conserve le résultat le plus riche,
        - aplatit les métadonnées pour Chroma/LlamaIndex.
        """
        # Validate file
        self.validate_file(file_path)

        if not self.supports(file_path):
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")

        text_content: str = ""
        metadata: dict = {}
        common_meta = {'format': 'pdf'}

        # ---------- Méthode 1 : Docling (logique d'origine) ----------
        text_docling = ""
        meta_docling = {}
        try:
            result = self.converter.convert(str(file_path))
            markdown_text = result.document.export_to_markdown()
            meta_docling = self._extract_metadata(result, file_path)
            if markdown_text and markdown_text.strip():
                logger.info(
                    f"✓ Parsed {file_path.name} via Docling: {len(markdown_text)} chars"
                )
                text_docling = markdown_text
            else:
                logger.info("Docling returned empty content")
        except Exception as e:
            logger.warning(f"Docling failed for {file_path.name}: {e}")

        # ---------- Méthode 2 : PyMuPDF puis pdfminer ----------
        text_pymupdf = ""
        meta_pymupdf = {}
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(file_path))
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            extracted = "\n".join(parts).strip()
            if extracted:
                text_pymupdf = extracted
                meta_pymupdf = {'parser': 'pymupdf', 'page_count': doc.page_count}
                logger.info(f"✓ Extracted text via PyMuPDF: {len(text_pymupdf)} chars")
            else:
                meta_pymupdf = {'parser': 'pymupdf', 'page_count': doc.page_count}
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {file_path.name}: {e}")
            # Fallback pdfminer
            try:
                from pdfminer.high_level import extract_text  # type: ignore
                extracted = (extract_text(str(file_path)) or "").strip()
                if extracted:
                    text_pymupdf = extracted
                    meta_pymupdf = {'parser': 'pdfminer'}
                    logger.info(f"✓ Extracted text via pdfminer: {len(text_pymupdf)} chars")
                else:
                    meta_pymupdf = {'parser': 'pdfminer'}
            except Exception as e2:
                logger.warning(f"pdfminer extraction failed for {file_path.name}: {e2}")
                meta_pymupdf = {'parser': 'pdfminer', 'error': str(e2)}

        # ---------- Méthode 3 : OCR (pytesseract) ----------
        text_ocr = ""
        meta_ocr = {}
        try:
            from pdf2image import convert_from_path  # type: ignore
            import pytesseract  # type: ignore

            images = convert_from_path(str(file_path), dpi=200)
            ocr_parts = []
            for img in images:
                gray = img.convert("L")
                try:
                    ocr_text = pytesseract.image_to_string(gray, lang="fra+eng")
                except Exception:
                    ocr_text = pytesseract.image_to_string(gray)
                if ocr_text:
                    ocr_parts.append(ocr_text)
            extracted = "\n".join(ocr_parts).strip()
            if extracted:
                text_ocr = extracted
                meta_ocr = {'parser': 'ocr', 'ocr_engine': 'pytesseract', 'ocr_pages': len(images)}
                logger.info(f"✓ Extracted text via OCR: {len(text_ocr)} chars")
            else:
                meta_ocr = {'parser': 'ocr', 'ocr_engine': 'pytesseract', 'ocr_pages': len(images)}
                logger.warning("No text found via OCR")
        except Exception as e:
            meta_ocr = {'parser': 'ocr', 'error': str(e)}
            logger.warning(
                f"OCR extraction failed for {file_path.name}: {e}. "
                "Install poppler-utils, pdf2image, and tesseract for OCR."
            )

        # ---------- Sélection du meilleur résultat ----------
        candidates = {
            "docling": (text_docling, meta_docling),
            "pymupdf": (text_pymupdf, meta_pymupdf),
            "ocr": (text_ocr, meta_ocr),
        }
        best_key = max(candidates.keys(), key=lambda k: len(candidates[k][0]) if candidates[k][0] else 0)
        best_text, best_meta = candidates[best_key]

        # Si tout vide : placeholder sûr
        if not (best_text and best_text.strip()):
            placeholder = (
                f"[PDF: {file_path.name}]\n"
                "Ce document ne contient pas de texte extrait par les méthodes disponibles. "
                "Vérifiez Docling, PyMuPDF/pdfminer et l’OCR (pdf2image + tesseract + poppler)."
            )
            language = 'fr'
            final_meta = {
                **common_meta,
                'parser': 'pdf_stub',
                'best_method': None,
                'methods_tried': list(candidates.keys()),
            }
            return ParsedContent(
                text=placeholder,
                content_type="text",
                language=language,
                file_path=file_path,
                metadata=self._flatten_metadata(final_meta)
            )

        # ---------- Détection langue (heuristique fr/en inchangée) ----------
        lower_text = f" {best_text.lower()} "
        french_keywords = [' le ', ' la ', ' les ', ' des ', ' du ', ' une ', ' un ', ' et ']
        language = 'fr' if any(kw in lower_text for kw in french_keywords) else 'en'

        # ---------- Métadonnées finales (aplatis) ----------
        final_meta = {
            **common_meta,
            **best_meta,
            'best_method': best_key,
            'methods_tried': list(candidates.keys()),
        }
        final_meta.setdefault('page_count', best_meta.get('page_count') if isinstance(best_meta, dict) else None)

        return ParsedContent(
            text=best_text,
            content_type="text",
            language=language,
            file_path=file_path,
            metadata=self._flatten_metadata(final_meta)
        )
    
    def _extract_metadata(self, result, file_path: Path) -> dict:
        """
        Extract metadata from Docling conversion result.
        
        Args:
            result: Docling conversion result
            file_path: Source file path
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'parser': 'docling_pdf',
            'format': 'pdf'
        }
        
        try:
            doc = result.document
            
            # Page count
            if hasattr(doc, 'pages'):
                metadata['page_count'] = len(doc.pages)
            
            # Document title
            if hasattr(doc, 'title') and doc.title:
                metadata['title'] = doc.title
            
            # Author
            if hasattr(doc, 'author') and doc.author:
                metadata['author'] = doc.author
            
            # Additional document metadata
            if hasattr(doc, 'metadata'):
                doc_meta = doc.metadata
                if isinstance(doc_meta, dict):
                    metadata.update({
                        k: v for k, v in doc_meta.items()
                        if k not in metadata and v is not None
                    })
            
        except Exception as e:
            logger.warning(f"Could not extract full metadata: {e}")
        
        return metadata

    # ---------- Helper interne pour Chroma/LlamaIndex ----------
    def _flatten_metadata(self, meta: dict) -> dict:
        """
        Ensure metadata is flat: Chroma/LlamaIndex accept only str/int/float/None.
        Lists/sets/tuples are joined by comma; dicts/objects are stringified.
        """
        flat = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float)) or v is None:
                flat[k] = v
            elif isinstance(v, (list, tuple, set)):
                flat[k] = ", ".join(map(str, v))
            else:
                try:
                    flat[k] = str(v)
                except Exception:
                    flat[k] = None
        return flat
