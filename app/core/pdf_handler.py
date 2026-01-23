"""
PDF Handler Module
PDF parsing, text extraction, and page management using PyMuPDF.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import fitz  # PyMuPDF
from PIL import Image
import io


@dataclass
class PageInfo:
    """Information about a single PDF page."""
    number: int  # 1-indexed
    width: float
    height: float
    text: str
    has_text: bool
    is_scanned: bool
    image_count: int


@dataclass
class ChapterInfo:
    """Information about a detected chapter."""
    title: str
    start_page: int  # 1-indexed
    end_page: int  # 1-indexed
    confidence: float  # 0.0-1.0


@dataclass
class PDFDocument:
    """Container for PDF document information."""
    path: str
    filename: str
    page_count: int
    title: Optional[str]
    author: Optional[str]
    is_scanned: bool
    file_size: int
    md5_hash: str
    pages: List[PageInfo] = field(default_factory=list)
    chapters: List[ChapterInfo] = field(default_factory=list)


class PDFHandler:
    """
    Handler for PDF document processing.
    Supports both digital and scanned PDFs.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/pdf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._doc: Optional[fitz.Document] = None
        self._pdf_info: Optional[PDFDocument] = None
    
    def open(self, pdf_path: str) -> PDFDocument:
        """
        Open a PDF document and extract basic information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFDocument with metadata and page information
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Close any previously opened document
        self.close()
        
        # Open the PDF
        self._doc = fitz.open(str(pdf_path))
        
        # Calculate file hash
        with open(pdf_path, 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        
        # Extract metadata
        metadata = self._doc.metadata or {}
        
        # Analyze pages
        pages = []
        scanned_page_count = 0
        
        for i, page in enumerate(self._doc):
            text = page.get_text().strip()
            has_text = len(text) > 50  # Threshold for "has meaningful text"
            image_count = len(page.get_images())
            
            # Determine if page is likely scanned
            is_scanned = not has_text and image_count > 0
            if is_scanned:
                scanned_page_count += 1
            
            pages.append(PageInfo(
                number=i + 1,
                width=page.rect.width,
                height=page.rect.height,
                text=text,
                has_text=has_text,
                is_scanned=is_scanned,
                image_count=image_count
            ))
        
        # Document is considered scanned if >50% of pages are scanned
        is_scanned = scanned_page_count > len(pages) * 0.5
        
        self._pdf_info = PDFDocument(
            path=str(pdf_path),
            filename=pdf_path.name,
            page_count=len(self._doc),
            title=metadata.get('title') or pdf_path.stem,
            author=metadata.get('author'),
            is_scanned=is_scanned,
            file_size=pdf_path.stat().st_size,
            md5_hash=md5_hash,
            pages=pages
        )
        
        return self._pdf_info
    
    def close(self):
        """Close the current document."""
        if self._doc:
            self._doc.close()
            self._doc = None
            self._pdf_info = None
    
    def get_page_text(self, page_number: int) -> str:
        """
        Get text from a specific page (1-indexed).
        
        Args:
            page_number: Page number (1-indexed)
            
        Returns:
            Extracted text from the page
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        if page_number < 1 or page_number > len(self._doc):
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = self._doc[page_number - 1]
        return page.get_text()
    
    def get_page_image(
        self,
        page_number: int,
        zoom: float = 2.0,
        format: str = "png"
    ) -> bytes:
        """
        Render a page as an image.
        
        Args:
            page_number: Page number (1-indexed)
            zoom: Zoom factor for rendering (2.0 = 144 DPI)
            format: Output format ("png", "jpeg", "ppm")
            
        Returns:
            Image bytes
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        if page_number < 1 or page_number > len(self._doc):
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = self._doc[page_number - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        if format.lower() == "png":
            return pix.tobytes("png")
        elif format.lower() == "jpeg":
            return pix.tobytes("jpeg")
        else:
            return pix.tobytes()
    
    def get_page_thumbnail(
        self,
        page_number: int,
        max_size: int = 200
    ) -> bytes:
        """
        Generate a thumbnail for a page.
        
        Args:
            page_number: Page number (1-indexed)
            max_size: Maximum dimension in pixels
            
        Returns:
            PNG thumbnail bytes
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        page = self._doc[page_number - 1]
        
        # Calculate zoom to fit max_size
        rect = page.rect
        scale = max_size / max(rect.width, rect.height)
        mat = fitz.Matrix(scale, scale)
        
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    
    def extract_text_range(
        self,
        start_page: int,
        end_page: int
    ) -> str:
        """
        Extract text from a range of pages.
        
        Args:
            start_page: Start page (1-indexed, inclusive)
            end_page: End page (1-indexed, inclusive)
            
        Returns:
            Combined text from all pages in range
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        texts = []
        for i in range(start_page - 1, min(end_page, len(self._doc))):
            page = self._doc[i]
            text = page.get_text().strip()
            if text:
                texts.append(text)
        
        return "\n\n".join(texts)
    
    def extract_all_text(self) -> str:
        """Extract text from all pages."""
        if not self._doc:
            raise RuntimeError("No document is open")
        
        return self.extract_text_range(1, len(self._doc))
    
    def get_images_for_ocr(
        self,
        page_number: int,
        dpi: int = 300
    ) -> List[Image.Image]:
        """
        Extract images from a page suitable for OCR.
        
        Args:
            page_number: Page number (1-indexed)
            dpi: DPI for rendering
            
        Returns:
            List of PIL Images
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        page = self._doc[page_number - 1]
        images = []
        
        # If page has embedded images, extract them
        for img_info in page.get_images():
            xref = img_info[0]
            base_image = self._doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            images.append(img)
        
        # If no embedded images, render the page
        if not images:
            zoom = dpi / 72  # 72 is default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        return images
    
    def render_page_for_ocr(
        self,
        page_number: int,
        dpi: int = 300
    ) -> Image.Image:
        """
        Render entire page as image for OCR.
        
        Args:
            page_number: Page number (1-indexed)
            dpi: DPI for rendering
            
        Returns:
            PIL Image
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        page = self._doc[page_number - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    def get_toc(self) -> List[Tuple[int, str, int]]:
        """
        Get table of contents.
        
        Returns:
            List of (level, title, page_number) tuples
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        return self._doc.get_toc()
    
    def save_page_as_image(
        self,
        page_number: int,
        output_path: str,
        dpi: int = 150
    ) -> str:
        """
        Save a page as an image file.
        
        Args:
            page_number: Page number (1-indexed)
            output_path: Output file path
            dpi: DPI for rendering
            
        Returns:
            Path to saved image
        """
        if not self._doc:
            raise RuntimeError("No document is open")
        
        page = self._doc[page_number - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path))
        
        return str(output_path)
    
    @property
    def is_open(self) -> bool:
        """Check if a document is currently open."""
        return self._doc is not None
    
    @property
    def info(self) -> Optional[PDFDocument]:
        """Get current document info."""
        return self._pdf_info
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
