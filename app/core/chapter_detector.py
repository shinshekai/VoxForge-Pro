"""
Chapter Detector Module
Automatic detection of chapter boundaries in text and PDFs.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Chapter:
    """Represents a detected chapter."""
    number: int
    title: str
    start_page: int  # 1-indexed
    end_page: int  # 1-indexed
    start_position: int  # Character position in text
    end_position: int
    confidence: float


class ChapterDetector:
    """
    Detects chapter boundaries using multiple strategies:
    - Regex pattern matching
    - TOC parsing
    - Font size analysis (for PDFs)
    """
    
    # Common chapter patterns
    DEFAULT_PATTERNS = [
        # "Chapter 1", "Chapter One", "CHAPTER 1"
        r"(?i)^(?:chapter|chap\.?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[a-z]+)\s*[:\-\.]?\s*(.*)$",
        # "Part I", "Part 1", "PART ONE"
        r"(?i)^(?:part)\s+(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[:\-\.]?\s*(.*)$",
        # "Section 1", "Section A"
        r"(?i)^(?:section)\s+(\d+|[a-z])\s*[:\-\.]?\s*(.*)$",
        # "Book 1", "Book One"
        r"(?i)^(?:book)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[:\-\.]?\s*(.*)$",
        # "Prologue", "Epilogue", "Introduction"
        r"(?i)^(prologue|epilogue|introduction|preface|foreword|afterword|conclusion)\s*[:\-\.]?\s*(.*)$",
        # Roman numerals alone: "I.", "II.", "III."
        r"^([IVXLCDM]+)\.\s*(.*)$",
        # Numbered with period: "1.", "2."
        r"^(\d+)\.\s+(.+)$"
    ]
    
    # Number word to digit mapping
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }
    
    # Roman numeral mapping
    ROMAN_VALUES = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        Initialize chapter detector.
        
        Args:
            custom_patterns: Additional regex patterns to use
        """
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        self._compiled_patterns = [re.compile(p, re.MULTILINE) for p in self.patterns]
    
    def _roman_to_int(self, s: str) -> int:
        """Convert Roman numeral to integer."""
        s = s.upper()
        result = 0
        prev = 0
        
        for char in reversed(s):
            value = self.ROMAN_VALUES.get(char, 0)
            if value < prev:
                result -= value
            else:
                result += value
            prev = value
        
        return result
    
    def _parse_chapter_number(self, number_str: str) -> int:
        """Parse chapter number from string (digit, word, or roman numeral)."""
        number_str = number_str.strip().lower()
        
        # Try as integer
        try:
            return int(number_str)
        except ValueError:
            pass
        
        # Try as word
        if number_str in self.NUMBER_WORDS:
            return self.NUMBER_WORDS[number_str]
        
        # Try as Roman numeral
        if re.match(r'^[ivxlcdm]+$', number_str):
            return self._roman_to_int(number_str)
        
        return 0
    
    def detect_from_text(
        self,
        text: str,
        min_chapter_length: int = 500
    ) -> List[Chapter]:
        """
        Detect chapters from text content.
        
        Args:
            text: Full text content
            min_chapter_length: Minimum characters between chapter markers
            
        Returns:
            List of detected chapters
        """
        chapters = []
        matches = []
        
        # Find all pattern matches
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                # Get the full matched line
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                
                full_line = text[line_start:line_end].strip()
                
                # Parse groups
                groups = match.groups()
                if len(groups) >= 1:
                    number_str = groups[0]
                    title = groups[1] if len(groups) > 1 and groups[1] else full_line
                else:
                    number_str = "0"
                    title = full_line
                
                chapter_num = self._parse_chapter_number(number_str)
                
                matches.append({
                    'position': match.start(),
                    'number': chapter_num,
                    'title': title.strip(),
                    'line': full_line,
                    'pattern_index': self.patterns.index(pattern.pattern)
                })
        
        # Sort by position
        matches.sort(key=lambda x: x['position'])
        
        # Filter out false positives (too close together)
        filtered_matches = []
        last_pos = -min_chapter_length
        
        for m in matches:
            if m['position'] - last_pos >= min_chapter_length:
                filtered_matches.append(m)
                last_pos = m['position']
        
        # Create chapter objects
        for i, m in enumerate(filtered_matches):
            start_pos = m['position']
            end_pos = filtered_matches[i + 1]['position'] if i + 1 < len(filtered_matches) else len(text)
            
            # Calculate confidence based on pattern quality
            pattern_confidence = 1.0 - (m['pattern_index'] * 0.1)
            pattern_confidence = max(0.3, pattern_confidence)
            
            chapters.append(Chapter(
                number=m['number'] or (i + 1),
                title=m['title'] or f"Chapter {i + 1}",
                start_page=0,  # Will be filled in by caller
                end_page=0,
                start_position=start_pos,
                end_position=end_pos,
                confidence=pattern_confidence
            ))
        
        return chapters
    
    def detect_from_toc(
        self,
        toc: List[Tuple[int, str, int]]
    ) -> List[Chapter]:
        """
        Detect chapters from PDF table of contents.
        
        Args:
            toc: List of (level, title, page_number) from PDF
            
        Returns:
            List of detected chapters
        """
        chapters = []
        
        # Filter for top-level entries (level 1)
        top_level = [item for item in toc if item[0] == 1]
        
        for i, (level, title, page) in enumerate(top_level):
            end_page = top_level[i + 1][2] - 1 if i + 1 < len(top_level) else page + 50
            
            chapters.append(Chapter(
                number=i + 1,
                title=title,
                start_page=page,
                end_page=end_page,
                start_position=0,
                end_position=0,
                confidence=0.95  # High confidence for TOC-based detection
            ))
        
        return chapters
    
    def detect_from_pages(
        self,
        pages: List[str],
        min_chapter_length: int = 500
    ) -> List[Chapter]:
        """
        Detect chapters from a list of page texts with page tracking.
        
        Args:
            pages: List of text strings, one per page
            min_chapter_length: Minimum characters between chapter markers
            
        Returns:
            List of detected chapters with page numbers
        """
        # Build combined text with page markers
        combined_text = ""
        page_positions = []  # (start_pos, end_pos) for each page
        
        for i, page_text in enumerate(pages):
            start = len(combined_text)
            combined_text += page_text + "\n\n"
            end = len(combined_text)
            page_positions.append((start, end))
        
        # Detect chapters in combined text
        chapters = self.detect_from_text(combined_text, min_chapter_length)
        
        # Map positions to page numbers
        for chapter in chapters:
            # Find start page
            for page_num, (start, end) in enumerate(page_positions):
                if start <= chapter.start_position < end:
                    chapter.start_page = page_num + 1
                    break
            
            # Find end page
            for page_num, (start, end) in enumerate(page_positions):
                if start < chapter.end_position <= end:
                    chapter.end_page = page_num + 1
                    break
            
            # Ensure end_page >= start_page
            if chapter.end_page < chapter.start_page:
                chapter.end_page = chapter.start_page
        
        return chapters
    
    def add_pattern(self, pattern: str):
        """Add a custom chapter detection pattern."""
        self.patterns.append(pattern)
        self._compiled_patterns.append(re.compile(pattern, re.MULTILINE))
    
    def clear_custom_patterns(self):
        """Reset to default patterns only."""
        self.patterns = self.DEFAULT_PATTERNS.copy()
        self._compiled_patterns = [re.compile(p, re.MULTILINE) for p in self.patterns]
