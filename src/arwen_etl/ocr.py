from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
    import numpy as np
    from PIL import Image
    from pdf2image import convert_from_path
    import pytesseract

    _OCR_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    convert_from_path = None  # type: ignore[assignment]
    pytesseract = None  # type: ignore[assignment]
    _OCR_AVAILABLE = False


class OCRProcessor:
    """Tesseract OCR with preprocessing"""

    def __init__(self, lang: str = 'eng'):
        if not _OCR_AVAILABLE:
            raise RuntimeError(
                "OCR dependencies not installed. Install with: uv sync --extra ocr"
                "numpy Pillow pdf2image pytesseract"
            )
        self.lang = lang
        self.tesseract_config = '--oem 3 --psm 6'
        # Build Tesseract path based on system
        if os.name == 'posix':
            self.tesseract_path = '/usr/bin/tesseract'
        else:
            self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR accuracy"""
        # Convert to gray if color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        # Denoising
        image = cv2.fastNlMeansDenoising(image)
        return image

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page metadata"""
        text_results = []
        images = convert_from_path(pdf_path)
        for i, page in enumerate(images):
            processed = self.preprocess_image(np.array(page))
            text = pytesseract.image_to_string(processed, lang=self.lang, config=self.tesseract_config)
            text_results.append({
                'page': i+1,
                'text': text.strip(),
                'page_num': i+1,
                'page_count': len(images),
                'confidence': self._calculate_confidence(processed)
            })
        return text_results

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        processed = self.preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang=self.lang, config=self.tesseract_config)
        return {
            'text': text.strip(),
            'confidence': self._calculate_confidence(processed),
            'dimensions': img.shape
        }

    def _calculate_confidence(self, image: np.ndarray) -> float:
        """Estimate OCR confidence from image characteristics"""
        # Simple heuristic based on edge detection
        edges = cv2.Canny(image, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = np.prod(image.shape)
        return min(1.0, edge_pixels / total_pixels * 0.8 + 0.2)


# Public API — instantiate only when OCR dependencies are available.
ocr_processor: Optional[OCRProcessor] = None
if _OCR_AVAILABLE:
    ocr_processor = OCRProcessor()