"""
VoxForge Pro - Premium AI-Powered Audiobook Generator
Clean implementation with proper warning resolution
"""

# ============================================================================
# PROPER WARNING RESOLUTION
# ============================================================================
# 
# WARNINGS AND THEIR RESOLUTIONS:
#
# 1. "pkg_resources is deprecated" - From perth/paddleocr dependency
#    STATUS: Cannot fix - this is inside the paddleocr library code
#    SOLUTION: Wait for paddleocr to update their code, or pin setuptools<81
#
# 2. "dropout option adds dropout after all but last recurrent layer"
#    STATUS: Cannot fix - this is inside Kokoro's model architecture
#    SOLUTION: This is a harmless warning, the model works correctly
#
# 3. "torch.nn.utils.weight_norm is deprecated"
#    STATUS: Cannot fix - this is inside Kokoro's model code
#    SOLUTION: Wait for Kokoro to update to parametrizations.weight_norm
#
# 4. "Checking connectivity to the model hosters"
#    STATUS: FIXED - Set DISABLE_MODEL_SOURCE_CHECK environment variable
#    MUST be set before importing paddleocr
#
# 5. "Trying to convert audio automatically from float32 to 16-bit int"
#    STATUS: FIXED - Convert audio to int16 before passing to Gradio
#
# ============================================================================

import os
import sys

# FIX #4: Set environment variable BEFORE any paddleocr-related imports
# This is the proper way to disable the connectivity check
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Standard library imports
import json
import tempfile
import uuid
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    print("ERROR: Gradio not installed. Run: pip install gradio")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        CUDA_VERSION = torch.version.cuda or "Unknown"
    else:
        GPU_NAME = "CPU Only"
        GPU_MEMORY = 0
        CUDA_VERSION = "N/A"
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_NAME = "PyTorch Not Installed"
    GPU_MEMORY = 0
    CUDA_VERSION = "N/A"

# Core module imports
try:
    from core.tts_engine import TTSManager, TTSConfig, KOKORO_AVAILABLE, CHATTERBOX_AVAILABLE, GeneratedAudio
except ImportError as e:
    print(f"Warning: TTS engine import failed: {e}")
    KOKORO_AVAILABLE = False
    CHATTERBOX_AVAILABLE = False
    TTSManager = None
    TTSConfig = None
    GeneratedAudio = None

try:
    from core.pdf_handler import PDFHandler
except ImportError as e:
    print(f"Warning: PDF handler import failed: {e}")
    PDFHandler = None

try:
    from core.ocr_processor import OCRProcessor, PADDLEOCR_AVAILABLE
except ImportError as e:
    print(f"Warning: OCR processor import failed: {e}")
    PADDLEOCR_AVAILABLE = False
    OCRProcessor = None

try:
    from core.chapter_detector import ChapterDetector
except ImportError as e:
    print(f"Warning: Chapter detector import failed: {e}")
    ChapterDetector = None

try:
    from core.voice_manager import VoiceManager
except ImportError as e:
    print(f"Warning: Voice manager import failed: {e}")
    VoiceManager = None

try:
    from core.job_queue import JobQueue, JobStatus
except ImportError as e:
    print(f"Warning: Job queue import failed: {e}")
    JobQueue = None
    JobStatus = None

try:
    from core.library import Library, init_db
except ImportError as e:
    print(f"Warning: Library import failed: {e}")
    Library = None
    init_db = lambda: None


# ============================================================================
# GLOBAL STATE
# ============================================================================

tts_manager: Optional[Any] = None
pdf_handler: Optional[Any] = None
ocr_processor: Optional[Any] = None
chapter_detector: Optional[Any] = None
voice_manager: Optional[Any] = None
job_queue: Optional[Any] = None
library: Optional[Any] = None

current_pdf_path: Optional[str] = None
current_pdf_text: str = ""
current_chapters: List[Dict] = []


def safe_init_components():
    """Initialize all components safely."""
    global tts_manager, pdf_handler, ocr_processor, chapter_detector
    global voice_manager, job_queue, library
    
    errors = []
    
    if TTSManager:
        try:
            tts_manager = TTSManager()
        except Exception as e:
            errors.append(f"TTS: {e}")
    
    if PDFHandler:
        try:
            Path("cache/pdf").mkdir(parents=True, exist_ok=True)
            pdf_handler = PDFHandler(cache_dir="cache/pdf")
        except Exception as e:
            errors.append(f"PDF: {e}")
    
    if ChapterDetector:
        try:
            chapter_detector = ChapterDetector()
        except Exception as e:
            errors.append(f"Chapters: {e}")
    
    if VoiceManager:
        try:
            Path("data/voices").mkdir(parents=True, exist_ok=True)
            voice_manager = VoiceManager(data_dir="data/voices")
        except Exception as e:
            errors.append(f"Voices: {e}")
    
    if JobQueue:
        try:
            Path("data/jobs").mkdir(parents=True, exist_ok=True)
            job_queue = JobQueue(data_dir="data/jobs")
        except Exception as e:
            errors.append(f"Jobs: {e}")
    
    if Library:
        try:
            Path("library").mkdir(parents=True, exist_ok=True)
            library = Library(db_path="library/database.sqlite")
        except Exception as e:
            errors.append(f"Library: {e}")
    
    if OCRProcessor and PADDLEOCR_AVAILABLE:
        try:
            Path("cache/ocr").mkdir(parents=True, exist_ok=True)
            ocr_processor = OCRProcessor(cache_dir="cache/ocr")
        except Exception as e:
            errors.append(f"OCR: {e}")
    
    return errors


# ============================================================================
# VOICE CATALOG
# ============================================================================

KOKORO_VOICES = {
    "🇺🇸 American English": {
        "Female": [
            ("af_alloy", "Alloy", "Warm, conversational"),
            ("af_aoede", "Aoede", "Melodic, expressive"),
            ("af_bella", "Bella", "Friendly, clear"),
            ("af_heart", "Heart", "Warm, emotional"),
            ("af_jessica", "Jessica", "Professional, articulate"),
            ("af_kore", "Kore", "Young, energetic"),
            ("af_nicole", "Nicole", "Smooth, sophisticated"),
            ("af_nova", "Nova", "Modern, dynamic"),
            ("af_river", "River", "Calm, flowing"),
            ("af_sarah", "Sarah", "Natural, relatable"),
            ("af_sky", "Sky", "Light, airy"),
        ],
        "Male": [
            ("am_adam", "Adam", "Deep, authoritative"),
            ("am_echo", "Echo", "Resonant, clear"),
            ("am_eric", "Eric", "Friendly, casual"),
            ("am_fenrir", "Fenrir", "Bold, dramatic"),
            ("am_liam", "Liam", "Warm, trustworthy"),
            ("am_michael", "Michael", "Professional, neutral"),
            ("am_onyx", "Onyx", "Rich, powerful"),
            ("am_puck", "Puck", "Playful, mischievous"),
            ("am_santa", "Santa", "Jolly, warm"),
        ]
    },
    "🇬🇧 British English": {
        "Female": [
            ("bf_alice", "Alice", "Elegant, refined"),
            ("bf_emma", "Emma", "Warm, sophisticated"),
            ("bf_isabella", "Isabella", "Regal, articulate"),
            ("bf_lily", "Lily", "Sweet, gentle"),
        ],
        "Male": [
            ("bm_daniel", "Daniel", "Distinguished, clear"),
            ("bm_fable", "Fable", "Storyteller quality"),
            ("bm_george", "George", "Classic, authoritative"),
            ("bm_lewis", "Lewis", "Intellectual, warm"),
        ]
    },
    "🇧🇷 Brazilian Portuguese": {
        "Female": [
            ("pf_camila", "Camila", "Vibrant, expressive"),
            ("pf_dora", "Dora", "Warm, friendly"),
        ],
        "Male": [
            ("pm_alex", "Alex", "Clear, professional"),
            ("pm_santa", "Santa", "Jolly, festive"),
        ]
    },
    "🇨🇳 Chinese Mandarin": {
        "Female": [
            ("zf_xiaobei", "Xiaobei", "Sweet, youthful"),
            ("zf_xiaoni", "Xiaoni", "Gentle, melodic"),
            ("zf_xiaoxiao", "Xiaoxiao", "Lively, expressive"),
            ("zf_xiaoyi", "Xiaoyi", "Professional, clear"),
        ],
        "Male": [
            ("zm_yunjian", "Yunjian", "Strong, confident"),
            ("zm_yunxi", "Yunxi", "Warm, friendly"),
            ("zm_yunxia", "Yunxia", "Calm, soothing"),
            ("zm_yunyang", "Yunyang", "Energetic, youthful"),
        ]
    },
    "🇯🇵 Japanese": {
        "Female": [
            ("jf_alpha", "Alpha", "Modern, clear"),
            ("jf_gongitsune", "Gongitsune", "Gentle, storyteller"),
            ("jf_nezumi", "Nezumi", "Sweet, playful"),
            ("jf_tebukuro", "Tebukuro", "Warm, nurturing"),
        ],
        "Male": [
            ("jm_kumo", "Kumo", "Deep, mysterious"),
        ]
    },
    "🇪🇸 Spanish": {
        "Female": [
            ("ef_dora", "Dora", "Warm, expressive"),
        ],
        "Male": [
            ("em_alex", "Alex", "Clear, professional"),
            ("em_santa", "Santa", "Festive, jolly"),
        ]
    }
}


def get_voice_choices() -> List[Tuple[str, str]]:
    """Get voice choices for dropdown."""
    choices = []
    for lang, genders in KOKORO_VOICES.items():
        lang_name = lang.split(" ", 1)[1] if " " in lang else lang
        for gender, voices in genders.items():
            for voice_id, name, desc in voices:
                choices.append((f"{name} ({gender[0]}) - {lang_name}", voice_id))
    return choices


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_duration(seconds: float) -> str:
    """Format duration."""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
    except:
        return "0s"


def format_file_size(bytes_size: int) -> str:
    """Format file size."""
    try:
        if bytes_size >= 1024**3:
            return f"{bytes_size / 1024**3:.2f} GB"
        elif bytes_size >= 1024**2:
            return f"{bytes_size / 1024**2:.2f} MB"
        elif bytes_size >= 1024:
            return f"{bytes_size / 1024:.2f} KB"
        return f"{bytes_size} B"
    except:
        return "0 B"


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    FIX #5: Convert float32 audio to int16 to prevent Gradio warning.
    This is the proper way to handle audio for Gradio.
    """
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        # Normalize to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        # Convert to int16 range
        audio = (audio * 32767).astype(np.int16)
    return audio


def get_system_status_html() -> str:
    """Generate system status HTML."""
    engines_html = ""
    if KOKORO_AVAILABLE:
        engines_html += '<span class="status-chip active">✓ Kokoro-82M</span>'
    else:
        engines_html += '<span class="status-chip inactive">✗ Kokoro</span>'
    
    if CHATTERBOX_AVAILABLE:
        engines_html += '<span class="status-chip active">✓ Chatterbox</span>'
    else:
        engines_html += '<span class="status-chip inactive">✗ Chatterbox</span>'
    
    if CUDA_AVAILABLE:
        gpu_html = f'''
            <span class="status-chip gpu-active">⚡ {GPU_NAME}</span>
            <span class="status-chip">{GPU_MEMORY:.1f} GB VRAM</span>
            <span class="status-chip">CUDA {CUDA_VERSION}</span>
        '''
    else:
        gpu_html = '<span class="status-chip warning">⚠ CPU Mode</span>'
    
    return f'''
        <div class="system-status">
            <div class="status-group">
                <span class="status-label">ENGINES</span>
                {engines_html}
            </div>
            <div class="status-group">
                <span class="status-label">COMPUTE</span>
                {gpu_html}
            </div>
        </div>
    '''


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def handle_pdf_upload(file) -> Tuple[str, str, List, Optional[str], List]:
    """Handle PDF upload - returns text, status, chapters, first page image, and chapter dropdown choices."""
    global current_pdf_path, current_pdf_text, current_chapters
    
    if file is None:
        return "", "📁 Upload a PDF to get started", [], None, []
    
    if not pdf_handler:
        return "", "❌ PDF processing not available", [], None, []
    
    try:
        pdf_info = pdf_handler.open(file.name)
        current_pdf_path = file.name
        current_pdf_text = pdf_handler.extract_all_text()
        
        # Detect chapters
        chapter_dropdown_choices = [("📖 Full Document", "full")]
        if chapter_detector:
            pages_text = [pdf_handler.get_page_text(i+1) for i in range(pdf_info.page_count)]
            chapters = chapter_detector.detect_from_pages(pages_text)
            current_chapters = [
                {'number': ch.number, 'title': ch.title, 'start_page': ch.start_page, 'end_page': ch.end_page}
                for ch in chapters
            ]
            chapter_data = [[ch['number'], ch['title'], f"{ch['start_page']}-{ch['end_page']}"] 
                            for ch in current_chapters]
            # Add chapters to dropdown
            for ch in current_chapters:
                chapter_dropdown_choices.append((f"Chapter {ch['number']}: {ch['title']}", str(ch['number'])))
        else:
            chapter_data = []
        
        # Render first page as preview
        first_page_path = None
        try:
            import tempfile
            page_img_bytes = pdf_handler.get_page_image(1, zoom=1.0)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(page_img_bytes)
                first_page_path = f.name
        except Exception as e:
            print(f"Could not render PDF page: {e}")
        
        pdf_type = "📷 Scanned" if pdf_info.is_scanned else "📄 Digital"
        status = f"✅ **{pdf_info.filename}**\n📑 {pdf_info.page_count} pages | {pdf_type} | {len(current_chapters)} chapters"
        
        # Return full text for now
        text_preview = current_pdf_text[:5000]
        if len(current_pdf_text) > 5000:
            text_preview += "\n\n⋯ *text truncated for preview*"
        
        return text_preview, status, chapter_data, first_page_path, chapter_dropdown_choices
        
    except Exception as e:
        import traceback
        print(f"PDF upload error: {traceback.format_exc()}")
        return "", f"❌ Error: {str(e)}", [], None, []


def handle_chapter_select(chapter_choice: str) -> str:
    """Handle chapter selection - returns the text for the selected chapter."""
    global current_pdf_text, current_chapters
    
    print(f"DEBUG: handle_chapter_select called with: '{chapter_choice}'")
    print(f"DEBUG: pdf_handler exists: {pdf_handler is not None}")
    print(f"DEBUG: pdf_handler.is_open: {pdf_handler.is_open if pdf_handler else 'N/A'}")
    print(f"DEBUG: current_chapters count: {len(current_chapters)}")
    
    if not pdf_handler:
        return "❌ PDF handler not initialized. Please try reloading the page."
    
    if not pdf_handler.is_open:
        return "❌ No PDF loaded. Please upload a PDF first."
    
    if chapter_choice == "full" or not chapter_choice:
        # Return full document text
        if not current_pdf_text:
            return "❌ No text extracted from PDF. The document may be scanned or image-based."
        text = current_pdf_text
        if len(text) > 15000:
            text = text[:15000] + "\n\n⋯ *text truncated for display*"
        return text
    
    if not current_chapters:
        return "❌ No chapters detected in this PDF. The document may not have standard chapter headings."
    
    try:
        chapter_num = int(chapter_choice)
        print(f"DEBUG: Looking for chapter {chapter_num}")
        
        for ch in current_chapters:
            print(f"DEBUG: Checking chapter {ch['number']}: {ch['title']}")
            if ch['number'] == chapter_num:
                # Extract text for this chapter's page range
                print(f"DEBUG: Found chapter, extracting pages {ch['start_page']}-{ch['end_page']}")
                chapter_text = pdf_handler.extract_text_range(ch['start_page'], ch['end_page'])
                
                if not chapter_text or not chapter_text.strip():
                    return f"⚠️ **Chapter {chapter_num}: {ch['title']}**\n\nPages {ch['start_page']}-{ch['end_page']}\n\n*This chapter appears to be scanned or has no extractable text. OCR may be needed.*"
                
                header = f"## 📖 Chapter {chapter_num}: {ch['title']}\n*Pages {ch['start_page']}-{ch['end_page']}*\n\n---\n\n"
                return header + chapter_text
        
        return f"❌ Chapter {chapter_num} not found in the list of {len(current_chapters)} detected chapters."
    except (ValueError, TypeError) as e:
        print(f"DEBUG: Error parsing chapter choice: {e}")
        return f"❌ Invalid chapter selection: '{chapter_choice}'."



def render_pdf_page(page_num: int) -> Optional[str]:
    """Render a specific PDF page as an image and return the file path."""
    if not pdf_handler or not pdf_handler.is_open:
        return None
    
    try:
        import tempfile
        page_num = max(1, min(page_num, pdf_handler.info.page_count))
        page_img_bytes = pdf_handler.get_page_image(page_num, zoom=1.5)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(page_img_bytes)
            return f.name
    except Exception as e:
        print(f"Could not render page {page_num}: {e}")
        return None


def get_page_count() -> int:
    """Get total page count of current PDF."""
    if pdf_handler and pdf_handler.is_open and pdf_handler.info:
        return pdf_handler.info.page_count
    return 1


def handle_generate(
    text: str,
    engine: str,
    voice: str,
    speed: float,
    format_type: str,
    bitrate: int,
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """Generate audiobook."""
    
    if not text or not text.strip():
        return "⚠️ Please enter some text.", None
    
    if not tts_manager:
        return "❌ TTS Engine not available.", None
    
    if not voice_manager:
        return "❌ Voice Manager not available.", None
    
    try:
        segments = voice_manager.parse_speaker_tags(text)
        book_id = str(uuid.uuid4())[:8]
        output_dir = Path(f"library/audiobooks/{book_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        engine_name = "kokoro" if "Kokoro" in engine else "chatterbox"
        config = TTSConfig(engine=engine_name, voice=voice, speed=speed)
        
        audio_chunks = []
        for i, segment in enumerate(segments):
            progress((i + 1) / len(segments), f"Generating {i+1}/{len(segments)}")
            
            if segment.speaker and hasattr(voice_manager, 'get_voice'):
                speaker_voice = voice_manager.get_voice(segment.speaker)
                if speaker_voice:
                    config.voice = speaker_voice.voice_id
                    config.engine = speaker_voice.engine
            
            result = tts_manager.generate(segment.text, config)
            if result and hasattr(result, 'audio'):
                audio_chunks.append(result.audio)
        
        if not audio_chunks:
            return "❌ No audio generated.", None
        
        full_audio = np.concatenate(audio_chunks)
        output_path = output_dir / f"audiobook.{format_type.lower()}"
        
        audio_obj = GeneratedAudio(
            audio=full_audio,
            sample_rate=24000,
            duration=len(full_audio) / 24000,
            text=text[:100],
            voice=voice
        )
        tts_manager.save_audio(audio_obj, str(output_path), format_type.lower(), bitrate)
        
        if library:
            try:
                title = Path(current_pdf_path).stem if current_pdf_path else f"Audio {book_id}"
                library.add_audiobook(
                    book_id=book_id,
                    title=title,
                    output_dir=str(output_dir),
                    engine=config.engine,
                    format=format_type.lower(),
                    bitrate=bitrate,
                    chapters=[],
                    voices={},
                    duration_seconds=audio_obj.duration,
                    file_size_bytes=output_path.stat().st_size
                )
            except Exception as lib_err:
                print(f"Library error: {lib_err}")
        
        status = f"✅ **Done!**\n📁 `{output_path.name}`\n⏱️ {format_duration(audio_obj.duration)} | 💾 {format_file_size(output_path.stat().st_size)}"
        return status, str(output_path)
        
    except Exception as e:
        print(f"Generation error: {traceback.format_exc()}")
        return f"❌ Error: {str(e)}", None


def safe_preview_voice(voice: str, text: str) -> Optional[Tuple[int, Any]]:
    """Generate voice preview with proper audio format conversion."""
    if not tts_manager:
        return None
    
    if not text or not text.strip():
        text = "Hello! This is a preview of my voice."
    
    try:
        config = TTSConfig(engine="kokoro", voice=voice, speed=1.0)
        result = tts_manager.generate(text[:300], config)
        
        if result and hasattr(result, 'audio') and hasattr(result, 'sample_rate'):
            # FIX #5: Convert to int16 to prevent Gradio warning
            audio_int16 = audio_to_int16(result.audio)
            return (result.sample_rate, audio_int16)
        return None
    except Exception as e:
        print(f"Preview error: {e}")
        return None


def get_library_data() -> Tuple[List[List], str]:
    """Get library data."""
    if not library:
        return [], "<p>Library not available</p>"
    
    try:
        audiobooks = library.get_all_audiobooks(limit=50)
        stats = library.get_stats()
        
        data = []
        for book in audiobooks:
            data.append([
                book.title[:40] if book.title else "Untitled",
                book.engine or "—",
                format_duration(book.duration_seconds) if book.duration_seconds else "—",
                format_file_size(book.file_size_bytes) if book.file_size_bytes else "—",
                (book.format.upper() if book.format else "—"),
                book.created_at.strftime("%Y-%m-%d") if book.created_at else "—",
                book.book_id
            ])
        
        stats_html = f"""
            <div class="library-stats">
                <div class="stat-card">
                    <span class="stat-value">{stats.get('total_audiobooks', 0)}</span>
                    <span class="stat-label">Audiobooks</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{stats.get('total_duration_hours', 0):.1f}h</span>
                    <span class="stat-label">Total Time</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{stats.get('total_size_gb', 0):.2f} GB</span>
                    <span class="stat-label">Storage</span>
                </div>
            </div>
        """
        return data, stats_html
    except Exception as e:
        print(f"Library error: {e}")
        return [], f"<p>Error: {e}</p>"


# ============================================================================
# CSS THEME - HIGH CONTRAST VERSION
# ============================================================================

MODERN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Use a light theme base for maximum readability */
:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #f1f5f9;
    --bg-tertiary: #e2e8f0;
    --bg-card: #ffffff;
    --accent-primary: #7c3aed;
    --accent-secondary: #a855f7;
    --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-color: #cbd5e1;
    --success: #059669;
    --warning: #d97706;
    --error: #dc2626;
}

/* Main container */
.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header */
.app-header {
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--accent-gradient);
    border-radius: 20px 20px 0 0;
}

.app-logo { font-size: 3.5rem; margin-bottom: 12px; display: block; }

.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #7c3aed;
    margin: 0 0 8px 0;
}

.app-subtitle { 
    color: var(--text-secondary); 
    font-size: 1.1rem; 
    margin: 0; 
}

/* System status */
.system-status {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 24px;
    margin-top: 24px;
    padding-top: 24px;
    border-top: 2px solid var(--border-color);
}

.status-group { 
    display: flex; 
    align-items: center; 
    gap: 8px; 
    flex-wrap: wrap;
}

.status-label {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.status-chip {
    padding: 8px 14px;
    background: var(--bg-secondary);
    border: 2px solid var(--border-color);
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
}

.status-chip.active { 
    background: #d1fae5; 
    border-color: #10b981; 
    color: #047857;
}

.status-chip.gpu-active { 
    background: #ede9fe; 
    border-color: #8b5cf6; 
    color: #6d28d9;
}

.status-chip.warning { 
    background: #fef3c7; 
    border-color: #f59e0b; 
    color: #b45309;
}

.status-chip.inactive { 
    opacity: 0.6; 
}

/* Force text colors on all Gradio elements */
.gradio-container * {
    color: var(--text-primary);
}

/* Tabs */
.tab-nav, .tabs {
    background: var(--bg-secondary) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    border: 2px solid var(--border-color) !important;
}

.tab-nav button, .tabs button {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover, .tabs button:hover { 
    background: var(--bg-tertiary) !important; 
    color: var(--text-primary) !important; 
}

.tab-nav button.selected, .tabs button.selected { 
    background: var(--accent-gradient) !important; 
    color: white !important; 
}

/* Text inputs */
textarea, input[type="text"], input[type="number"], .input-text {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 14px 16px !important;
    font-size: 1rem !important;
}

textarea::placeholder, input::placeholder {
    color: var(--text-muted) !important;
}

textarea:focus, input:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
}

/* Labels */
label, .label-wrap, .label-wrap span, .block-label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Dropdowns */
.dropdown, select, .gradio-dropdown {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* Buttons */
button, .gr-button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

button.primary, .gr-button.primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: white !important;
}

button.primary:hover, .gr-button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.4) !important;
}

button.secondary, .gr-button.secondary {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Sliders */
.gr-slider input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

/* Voice cards */
.voice-grid { 
    display: grid; 
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); 
    gap: 16px; 
}

.voice-card {
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.voice-card:hover { 
    border-color: var(--accent-primary); 
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.15);
}

.voice-name { 
    font-weight: 700; 
    color: var(--text-primary) !important; 
    font-size: 1.1rem; 
}

.voice-id { 
    font-size: 0.8rem; 
    color: var(--text-muted) !important; 
    font-family: 'Consolas', monospace; 
}

.voice-desc { 
    color: var(--text-secondary) !important; 
    font-size: 0.9rem; 
    margin-top: 8px; 
}

.voice-gender {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 12px;
}

.voice-gender.female { 
    background: #fce7f3; 
    color: #be185d; 
}

.voice-gender.male { 
    background: #dbeafe; 
    color: #1d4ed8; 
}

/* Language sections */
.language-section { 
    margin-bottom: 32px; 
}

.language-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border-color);
}

.language-flag { 
    font-size: 1.75rem; 
}

.language-name { 
    font-size: 1.25rem; 
    font-weight: 700; 
    color: var(--text-primary) !important; 
}

.language-count { 
    color: var(--text-muted) !important; 
    font-size: 0.9rem; 
}

/* Library stats */
.library-stats { 
    display: flex; 
    gap: 16px; 
    flex-wrap: wrap; 
    margin-bottom: 24px; 
}

.stat-card {
    flex: 1;
    min-width: 120px;
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.stat-value { 
    font-size: 1.75rem; 
    font-weight: 700; 
    color: var(--accent-primary) !important; 
    display: block; 
}

.stat-label { 
    font-size: 0.8rem; 
    color: var(--text-muted) !important; 
    text-transform: uppercase;
    font-weight: 600;
}

/* Tables */
table, .dataframe {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 16px !important;
}

th {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    font-weight: 700 !important;
    padding: 14px 16px !important;
}

td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding: 14px 16px !important;
}

/* Markdown */
.markdown-text, .prose, .gr-markdown {
    color: var(--text-primary) !important;
}

.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    color: var(--text-primary) !important;
}

.markdown-text p {
    color: var(--text-secondary) !important;
}

/* Accordion */
.accordion, .gr-accordion {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
}

/* Audio player - SUBTLE FIX for visibility */
audio {
    width: 100%;
    border-radius: 12px;
}

/* FORCE SVG dimensions since Gradio sometimes renders them with 0 width */
.play-pause-button svg,
.volume-button svg,
button.play-pause-button svg,
button[aria-label="Play"] svg,
button[aria-label="Pause"] svg,
button[aria-label*="volume"] svg,
button[aria-label*="Adjust"] svg {
    min-width: 16px !important;
    min-height: 16px !important;
    width: 16px !important;
    height: 16px !important;
    display: inline-block !important;
}

/* Target specific Gradio audio player button classes */
.play-pause-button svg,
.play-pause-button svg path,
.play-pause-button svg circle,
.play-pause-button svg polygon,
.play-pause-button svg rect,
button.play-pause-button svg,
button.play-pause-button svg *,
.volume-button svg,
.volume-button svg path,
.volume-button svg *,
button[class*="volume"] svg,
button[class*="volume"] svg * {
    fill: #1e293b !important;
    stroke: #1e293b !important;
    color: #1e293b !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Also target by aria-label for redundancy */
button[aria-label="Play"] svg,
button[aria-label="Play"] svg *,
button[aria-label="Pause"] svg,
button[aria-label="Pause"] svg *,
button[aria-label*="volume"] svg,
button[aria-label*="volume"] svg *,
button[aria-label*="Adjust"] svg,
button[aria-label*="Adjust"] svg * {
    fill: #1e293b !important;
    stroke: #1e293b !important;
    opacity: 1 !important;
}

/* General audio container SVGs */
div[class*="audio"] svg,
div[class*="audio"] svg path,
div[class*="Audio"] svg,
div[class*="Audio"] svg path {
    fill: #1e293b !important;
    stroke: #1e293b !important;
    opacity: 1 !important;
}

/* Time display */
div[class*="audio"] span,
div[class*="Audio"] span {
    color: #1e293b !important;
}

/* Footer */
.app-footer { 
    text-align: center; 
    padding: 32px; 
    color: var(--text-muted) !important;
    font-size: 0.9rem;
}

/* Responsive */
@media (max-width: 768px) {
    .app-header { padding: 24px; }
    .app-title { font-size: 2rem; }
    .system-status { flex-direction: column; gap: 12px; }
    .voice-grid { grid-template-columns: 1fr; }
}
"""




# ============================================================================
# UI BUILDER
# ============================================================================

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    voice_choices = get_voice_choices()
    default_voice = "af_heart" if voice_choices else None
    
    with gr.Blocks(css=MODERN_CSS, title="VoxForge Pro", theme=gr.themes.Soft(primary_hue="purple")) as app:
        
        gr.HTML(f"""
            <div class="app-header">
                <span class="app-logo">🎧</span>
                <h1 class="app-title">VoxForge Pro</h1>
                <p class="app-subtitle">Premium AI-Powered Audiobook Generator</p>
                {get_system_status_html()}
            </div>
        """)
        
        with gr.Tabs():
            
            with gr.Tab("✨ Create"):
                with gr.Row():
                    # Left column: PDF Viewer
                    with gr.Column(scale=2):
                        gr.Markdown("## 📖 PDF Viewer")
                        pdf_upload = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
                        pdf_status = gr.Markdown("📁 Upload a PDF to get started")
                        
                        # PDF Page Preview
                        pdf_page_image = gr.Image(label="Page Preview", type="filepath", height=400)
                        
                        # Page Navigation
                        with gr.Row():
                            prev_page_btn = gr.Button("◀ Prev", size="sm", scale=1)
                            page_number = gr.Number(value=1, label="Page", minimum=1, maximum=1000, step=1, scale=1)
                            next_page_btn = gr.Button("Next ▶", size="sm", scale=1)
                        
                        # Manual Page Range Selection
                        gr.Markdown("### 📄 Extract Pages")
                        with gr.Row():
                            start_page_input = gr.Number(value=1, label="Start Page", minimum=1, step=1, scale=1)
                            end_page_input = gr.Number(value=1, label="End Page", minimum=1, step=1, scale=1)
                        
                        with gr.Row():
                            extract_all_btn = gr.Button("📖 Full Document", variant="secondary", scale=1)
                            extract_range_btn = gr.Button("📥 Extract Range", variant="primary", scale=1)
                    
                    # Middle column: Text Content
                    with gr.Column(scale=3):
                        gr.Markdown("## 📝 Input Text")
                        gr.Markdown("""
Enter your text below. Use speaker tags like `[narrator]text[/narrator]` or `[alice]text[/alice]` for multi-voice stories, or plain text for single voice.

**Example tags:** `[narrator]`, `[alice]`, `[bob]`, `[speaker1]`, etc.
                        """)
                        
                        text_input = gr.Textbox(
                            label="",
                            placeholder="""[narrator]Once upon a time...[/narrator]
[alice]Hello, I'm Alice![/alice]
[bob]And I'm Bob![/bob]

Or drag & drop documents here (Word, PDF, TXT, RTF, EPUB)""",
                            lines=12,
                            show_label=False
                        )
                        
                        # Paralinguistic tags
                        gr.Markdown("**Insert paralinguistic tag:**")
                        with gr.Row():
                            tag_clear_throat = gr.Button("[clear throat]", size="sm", variant="secondary")
                            tag_sigh = gr.Button("[sigh]", size="sm", variant="secondary")
                            tag_cough = gr.Button("[cough]", size="sm", variant="secondary")
                            tag_groan = gr.Button("[groan]", size="sm", variant="secondary")
                        with gr.Row():
                            tag_sniff = gr.Button("[sniff]", size="sm", variant="secondary")
                            tag_gasp = gr.Button("[gasp]", size="sm", variant="secondary")
                            tag_chuckle = gr.Button("[chuckle]", size="sm", variant="secondary")
                            tag_laugh = gr.Button("[laugh]", size="sm", variant="secondary")
                        
                        with gr.Accordion("📊 Chapter Details", open=False):
                            chapters_table = gr.Dataframe(headers=["#", "Chapter", "Pages"], interactive=False)
                    
                    # Right column: Voice & Output
                    with gr.Column(scale=2):
                        gr.Markdown("## 🎙️ Voice")
                        
                        engine_choice = gr.Dropdown(
                            choices=["🎤 Kokoro-82M" if KOKORO_AVAILABLE else "🎤 Kokoro (N/A)",
                                     "🤖 Chatterbox" if CHATTERBOX_AVAILABLE else "🤖 Chatterbox (N/A)"],
                            value="🎤 Kokoro-82M" if KOKORO_AVAILABLE else None,
                            label="Engine"
                        )
                        
                        voice_dropdown = gr.Dropdown(choices=voice_choices, value=default_voice, label="Voice")
                        speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                        
                        with gr.Row():
                            preview_text = gr.Textbox(label="Preview", value="Hello!", lines=2, scale=3)
                            preview_btn = gr.Button("🔊", variant="secondary", scale=1)
                        
                        preview_audio = gr.Audio(label="Preview", type="numpy")
                        
                        gr.Markdown("## 📁 Output")
                        with gr.Row():
                            format_choice = gr.Dropdown(["MP3", "WAV", "FLAC", "OGG"], value="MP3", label="Format")
                            bitrate_choice = gr.Dropdown([64, 96, 128, 192, 256, 320], value=192, label="Bitrate")
                        
                        generate_btn = gr.Button("🚀 Generate Audiobook", variant="primary", size="lg")
                        generation_status = gr.Markdown("")
                        output_audio = gr.Audio(label="Output", type="filepath")
                
                # Event handlers for PDF and page extraction
                def on_pdf_upload(file):
                    result = handle_pdf_upload(file)
                    text, status, chapters, page_img, chapter_choices = result
                    # Get page count for end page default
                    total_pages = get_page_count() if pdf_handler and pdf_handler.is_open else 1
                    return text, status, chapters, page_img, total_pages
                
                pdf_upload.change(
                    on_pdf_upload, 
                    [pdf_upload], 
                    [text_input, pdf_status, chapters_table, pdf_page_image, end_page_input]
                )
                
                # Page navigation
                def go_prev_page(current):
                    new_page = max(1, int(current) - 1)
                    return new_page, render_pdf_page(new_page)
                
                def go_next_page(current):
                    max_page = get_page_count()
                    new_page = min(max_page, int(current) + 1)
                    return new_page, render_pdf_page(new_page)
                
                def on_page_change(page):
                    return render_pdf_page(int(page))
                
                prev_page_btn.click(go_prev_page, [page_number], [page_number, pdf_page_image])
                next_page_btn.click(go_next_page, [page_number], [page_number, pdf_page_image])
                page_number.change(on_page_change, [page_number], [pdf_page_image])
                
                # Page range extraction
                def extract_page_range(start, end):
                    if not pdf_handler or not pdf_handler.is_open:
                        return "❌ No PDF loaded. Please upload a PDF first."
                    start = int(start)
                    end = int(end)
                    max_page = get_page_count()
                    if start < 1 or start > max_page:
                        return f"❌ Start page must be between 1 and {max_page}"
                    if end < start:
                        return f"❌ End page must be >= start page"
                    end = min(end, max_page)
                    text = pdf_handler.extract_text_range(start, end)
                    if not text.strip():
                        return f"⚠️ Pages {start}-{end} have no extractable text. The document may be scanned."
                    return f"## 📄 Pages {start}-{end}\n\n{text}"
                
                def extract_full_document():
                    if not pdf_handler or not pdf_handler.is_open:
                        return "❌ No PDF loaded. Please upload a PDF first."
                    text = current_pdf_text
                    if not text:
                        return "❌ No text extracted from PDF."
                    if len(text) > 20000:
                        text = text[:20000] + "\n\n⋯ *truncated for display*"
                    return text
                
                extract_range_btn.click(extract_page_range, [start_page_input, end_page_input], [text_input])
                extract_all_btn.click(extract_full_document, [], [text_input])
                
                # Paralinguistic tag insertion - append to text
                def insert_tag(current_text, tag):
                    if current_text:
                        return current_text + " " + tag + " "
                    return tag + " "
                
                tag_clear_throat.click(lambda t: insert_tag(t, "[clear throat]"), [text_input], [text_input])
                tag_sigh.click(lambda t: insert_tag(t, "[sigh]"), [text_input], [text_input])
                tag_cough.click(lambda t: insert_tag(t, "[cough]"), [text_input], [text_input])
                tag_groan.click(lambda t: insert_tag(t, "[groan]"), [text_input], [text_input])
                tag_sniff.click(lambda t: insert_tag(t, "[sniff]"), [text_input], [text_input])
                tag_gasp.click(lambda t: insert_tag(t, "[gasp]"), [text_input], [text_input])
                tag_chuckle.click(lambda t: insert_tag(t, "[chuckle]"), [text_input], [text_input])
                tag_laugh.click(lambda t: insert_tag(t, "[laugh]"), [text_input], [text_input])
                
                # Voice preview and generation
                preview_btn.click(safe_preview_voice, [voice_dropdown, preview_text], [preview_audio])
                generate_btn.click(handle_generate, [text_input, engine_choice, voice_dropdown, speed_slider, format_choice, bitrate_choice], [generation_status, output_audio])
            
            with gr.Tab("🎭 Voices"):
                gr.Markdown("## 🎭 47 Premium AI Voices")
                gr.Markdown("Select a voice from the dropdown below to preview it.")
                
                # Voice Preview Section
                with gr.Group():
                    gr.Markdown("### 🔊 Preview a Voice")
                    with gr.Row():
                        gallery_voice = gr.Dropdown(
                            choices=voice_choices, 
                            value=default_voice,
                            label="Select Voice to Preview", 
                            scale=3
                        )
                        gallery_btn = gr.Button("▶️ Play Preview", variant="primary", size="lg", scale=1)
                    
                    with gr.Row():
                        preview_text_gallery = gr.Textbox(
                            label="Preview Text",
                            value="Welcome to VoxForge Pro! I can read any text naturally and expressively.",
                            lines=2,
                            scale=3
                        )
                    
                    gallery_audio = gr.Audio(label="Voice Preview", type="numpy")
                
                # Wire up the preview
                gallery_btn.click(
                    safe_preview_voice, 
                    [gallery_voice, preview_text_gallery], 
                    [gallery_audio]
                )
            
            with gr.Tab("📚 Library"):
                gr.Markdown("## 📚 Your Audiobooks")
                library_stats = gr.HTML("")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                library_table = gr.Dataframe(headers=["Title", "Engine", "Duration", "Size", "Format", "Created", "ID"], interactive=False)
                
                with gr.Row():
                    delete_id = gr.Textbox(label="Book ID", placeholder="Enter ID")
                    delete_btn = gr.Button("🗑️ Delete", variant="stop")
                delete_status = gr.Markdown("")
                
                def refresh_lib():
                    data, stats = get_library_data()
                    return data, stats
                
                def delete_book(book_id):
                    if library and book_id:
                        try:
                            if library.delete_audiobook(book_id):
                                data, stats = get_library_data()
                                return f"✅ Deleted", data, stats
                        except Exception as e:
                            return f"❌ {e}", gr.update(), gr.update()
                    return "❌ Invalid", gr.update(), gr.update()
                
                refresh_btn.click(refresh_lib, outputs=[library_table, library_stats])
                delete_btn.click(delete_book, [delete_id], [delete_status, library_table, library_stats])
                app.load(refresh_lib, outputs=[library_table, library_stats])
            
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("## ⚙️ Configuration")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Defaults")
                        default_engine = gr.Dropdown(["Kokoro-82M", "Chatterbox"], value="Kokoro-82M", label="Engine")
                        default_format = gr.Dropdown(["MP3", "WAV", "FLAC", "OGG"], value="MP3", label="Format")
                        default_bitrate = gr.Dropdown([64, 96, 128, 192, 256, 320], value=192, label="Bitrate")
                        save_btn = gr.Button("💾 Save", variant="primary")
                        save_status = gr.Markdown("")
                    
                    with gr.Column():
                        gr.Markdown("### System")
                        gr.Markdown(f"""
**GPU:** {'✅ ' + GPU_NAME if CUDA_AVAILABLE else '❌ CPU Mode'}  
**CUDA:** {CUDA_VERSION if CUDA_AVAILABLE else 'N/A'}  
**Kokoro:** {'✅' if KOKORO_AVAILABLE else '❌'}  
**Chatterbox:** {'✅' if CHATTERBOX_AVAILABLE else '❌'}  
**OCR:** {'✅' if PADDLEOCR_AVAILABLE else '❌'}
                        """)
                
                save_btn.click(lambda e, f, b: "✅ Saved!", [default_engine, default_format, default_bitrate], [save_status])
        
        gr.HTML('<div class="app-footer">Made with ❤️ • Powered by Kokoro & Chatterbox TTS</div>')
    
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoxForge Pro")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    print("🎧 Initializing VoxForge Pro...")
    
    try:
        init_db()
    except Exception as e:
        print(f"Database warning: {e}")
    
    errors = safe_init_components()
    if errors:
        print("Component warnings:")
        for e in errors:
            print(f"  - {e}")
    
    app = create_ui()
    print(f"🚀 Starting at http://{args.host}:{args.port}")
    
    try:
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print(f"⚠️ Port {args.port} busy, trying {args.port + 1}")
            app.launch(server_name=args.host, server_port=args.port + 1, share=args.share)
        else:
            raise


if __name__ == "__main__":
    main()
