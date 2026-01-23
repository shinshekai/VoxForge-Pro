"""
OCR Processor Module
Text extraction from scanned PDFs using PaddleOCR with caching support.

NOTE: The "pkg_resources is deprecated" warning comes from paddleocr's 'perth' dependency.
This cannot be fixed here - paddleocr needs to update their dependency.
"""

# PROPER FIX: Set environment variable to disable connectivity check BEFORE importing paddleocr
# This is the correct way to disable the check, not suppression
import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

# Try to import OCR libraries
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None


@dataclass
class OCRWord:
    """A single word/text block from OCR."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class OCRResult:
    """OCR result for a single page."""
    page_number: int
    text: str
    words: List[OCRWord]
    average_confidence: float
    low_confidence_words: List[OCRWord]


class OCRProcessor:
    """
    OCR processor using PaddleOCR for text extraction from images.
    Supports caching and multiple languages.
    """
    
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ch": "Chinese (Simplified)",
        "chinese_cht": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu"
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        language: str = "en",
        use_gpu: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize OCR processor.
        
        Args:
            cache_dir: Directory for caching OCR results
            language: OCR language code
            use_gpu: Whether to use GPU for OCR
            confidence_threshold: Minimum confidence for "good" results
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle")
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.language = language
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        
        self._ocr: Optional[PaddleOCR] = None
    
    @property
    def ocr(self) -> PaddleOCR:
        """Lazy-load OCR engine."""
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                use_gpu=self.use_gpu,
                show_log=False
            )
        return self._ocr
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Calculate hash of an image for caching."""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def _get_cache_path(self, image_hash: str) -> Path:
        """Get cache file path for an image hash."""
        return self.cache_dir / f"{image_hash}.json"
    
    def _load_from_cache(self, image_hash: str) -> Optional[OCRResult]:
        """Load OCR result from cache."""
        cache_path = self._get_cache_path(image_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                words = [
                    OCRWord(
                        text=w['text'],
                        confidence=w['confidence'],
                        bbox=tuple(w['bbox'])
                    )
                    for w in data['words']
                ]
                
                low_conf_words = [
                    OCRWord(
                        text=w['text'],
                        confidence=w['confidence'],
                        bbox=tuple(w['bbox'])
                    )
                    for w in data['low_confidence_words']
                ]
                
                return OCRResult(
                    page_number=data['page_number'],
                    text=data['text'],
                    words=words,
                    average_confidence=data['average_confidence'],
                    low_confidence_words=low_conf_words
                )
            except (json.JSONDecodeError, KeyError):
                return None
        return None
    
    def _save_to_cache(self, image_hash: str, result: OCRResult):
        """Save OCR result to cache."""
        cache_path = self._get_cache_path(image_hash)
        
        data = {
            'page_number': result.page_number,
            'text': result.text,
            'words': [
                {'text': w.text, 'confidence': w.confidence, 'bbox': list(w.bbox)}
                for w in result.words
            ],
            'average_confidence': result.average_confidence,
            'low_confidence_words': [
                {'text': w.text, 'confidence': w.confidence, 'bbox': list(w.bbox)}
                for w in result.low_confidence_words
            ]
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def process_image(
        self,
        image: Image.Image,
        page_number: int = 1,
        use_cache: bool = True
    ) -> OCRResult:
        """
        Process a single image with OCR.
        
        Args:
            image: PIL Image to process
            page_number: Page number for reference
            use_cache: Whether to use cached results
            
        Returns:
            OCRResult with extracted text and confidence scores
        """
        # Check cache
        if use_cache:
            image_hash = self._get_image_hash(image)
            cached = self._load_from_cache(image_hash)
            if cached:
                cached.page_number = page_number
                return cached
        
        # Convert to numpy array for PaddleOCR
        image_np = np.array(image)
        
        # Run OCR
        result = self.ocr.ocr(image_np, cls=True)
        
        # Parse results
        words = []
        low_confidence_words = []
        text_lines = []
        total_confidence = 0
        
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert bbox to x1, y1, x2, y2
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bbox_rect = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                )
                
                word = OCRWord(
                    text=text,
                    confidence=confidence,
                    bbox=bbox_rect
                )
                words.append(word)
                text_lines.append(text)
                total_confidence += confidence
                
                if confidence < self.confidence_threshold:
                    low_confidence_words.append(word)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(words) if words else 0
        
        # Combine text
        full_text = "\n".join(text_lines)
        
        result = OCRResult(
            page_number=page_number,
            text=full_text,
            words=words,
            average_confidence=avg_confidence,
            low_confidence_words=low_confidence_words
        )
        
        # Cache result
        if use_cache:
            self._save_to_cache(image_hash, result)
        
        return result
    
    def process_pdf_page(
        self,
        pdf_handler,
        page_number: int,
        dpi: int = 300,
        use_cache: bool = True
    ) -> OCRResult:
        """
        Process a PDF page with OCR.
        
        Args:
            pdf_handler: PDFHandler instance with an open document
            page_number: Page number to process (1-indexed)
            dpi: DPI for rendering
            use_cache: Whether to use cached results
            
        Returns:
            OCRResult with extracted text
        """
        # Render page as image
        image = pdf_handler.render_page_for_ocr(page_number, dpi=dpi)
        
        # Process with OCR
        return self.process_image(image, page_number, use_cache)
    
    def clear_cache(self):
        """Clear all cached OCR results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_size(self) -> int:
        """Get total size of cache in bytes."""
        total = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                total += f.stat().st_size
        return total
    
    def set_language(self, language: str):
        """
        Change OCR language (requires reinitializing engine).
        
        Args:
            language: Language code (e.g., "en", "ch", "ja")
        """
        if language != self.language:
            self.language = language
            self._ocr = None  # Force reinitialization
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Return dictionary of supported language codes and names."""
        return cls.SUPPORTED_LANGUAGES.copy()
