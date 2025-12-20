"""
Image parser stub for future vision-based processing.
"""
import logging
from pathlib import Path

from ...config import SUPPORTED_IMAGE_EXTENSIONS
from .base_parser import BaseParser, ParsedContent

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """
    Stub parser for image files.
    
    Future implementation will use vision models (e.g., GPT-4 Vision, LLaVA)
    to extract:
    - Image descriptions
    - Text from diagrams/screenshots (OCR)
    - Technical content understanding
    
    Currently returns basic metadata only.
    
    Examples:
        >>> parser = ImageParser()
        >>> parser.supports(Path("diagram.png"))
        True
        >>> parser.supports(Path("document.pdf"))
        False
    """
    
    def __init__(self):
        """Initialize image parser."""
        logger.warning(
            "ImageParser is a stub. Vision-based parsing not yet implemented."
        )
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is an image."""
        return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    
    def parse(self, file_path: Path) -> ParsedContent:
        """
        Parse image file (stub implementation).
        
        Args:
            file_path: Path to image file
            
        Returns:
            ParsedContent with placeholder text and metadata
            
        Note:
            This is a stub implementation. Future versions will use
            vision models for actual content extraction.
        """
        self.validate_file(file_path)
        
        if not self.supports(file_path):
            raise ValueError(f"Not a supported image file: {file_path}")
        
        # Attempt to extract text from the image via OCR.  If OCR is unavailable or fails,
        # fall back to a placeholder message.  This allows the ingestion pipeline to
        # incorporate basic information even when OCR is not configured.
        logger.info(f"Processing image: {file_path.name}")

        try:
            from PIL import Image as PILImage  # type: ignore
            import pytesseract  # type: ignore

            # Open image and convert to RGB for OCR
            with PILImage.open(file_path) as img:
                # Convert to grayscale to improve OCR accuracy
                gray = img.convert("L")
                # Perform OCR (attempt French and English by default)
                try:
                    ocr_text = pytesseract.image_to_string(gray, lang="fra+eng")
                except Exception:
                    ocr_text = pytesseract.image_to_string(gray)

            extracted_text = ocr_text.strip()
            if not extracted_text:
                extracted_text = f"[IMAGE: {file_path.name}]\nNo visible text detected via OCR."

            metadata = {
                'parser': 'image_ocr',
                'format': file_path.suffix.lower(),
                'ocr_engine': 'pytesseract',
                'image_size_bytes': file_path.stat().st_size,
                'language_hint': 'fra+eng'
            }
            logger.info(f"✓ Extracted text from {file_path.name}")
            return ParsedContent(
                text=extracted_text,
                content_type="text",
                language="unknown",
                file_path=file_path,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to perform OCR on image {file_path}: {e}")
            # Fall back to stub if OCR fails
            file_stat = file_path.stat()
            placeholder_text = (
                f"[IMAGE: {file_path.name}]\n"
                f"Image file to be processed with vision model.\n"
                f"Type: {file_path.suffix}\n"
                f"Size: {file_stat.st_size} bytes"
            )
            metadata = {
                'parser': 'image_stub',
                'format': file_path.suffix.lower(),
                'image_size_bytes': file_stat.st_size,
                'needs_vision_processing': True,
                'note': 'Vision-based extraction not yet implemented'
            }
            logger.info(f"✓ Created placeholder for {file_path.name}")
            return ParsedContent(
                text=placeholder_text,
                content_type="text",
                language="unknown",
                file_path=file_path,
                metadata=metadata
            )


# TODO: Future implementation with vision model
"""
Future implementation example:

class VisionImageParser(ImageParser):
    def __init__(self, vision_model="gpt-4-vision"):
        self.vision_client = OpenAI()
        self.model = vision_model
    
    def parse(self, file_path: Path) -> ParsedContent:
        # Read image
        import base64
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Call vision API
        response = self.vision_client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this technical image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }]
        )
        
        description = response.choices[0].message.content
        
        return ParsedContent(
            text=description,
            content_type="text",
            language="en",
            file_path=file_path,
            metadata={
                'parser': 'vision_gpt4',
                'format': file_path.suffix,
                'vision_model': self.model
            }
        )
"""
