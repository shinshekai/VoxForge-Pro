"""
TTS Engine Module
Unified wrapper for Kokoro-82M and Chatterbox TTS engines.

WARNINGS THAT CANNOT BE FIXED (they come from the Kokoro library itself):

1. "dropout option adds dropout after all but last recurrent layer"
   - This is in Kokoro's LSTM model architecture
   - The warning is harmless - the model works correctly
   - Kokoro would need to change their model config to fix this

2. "torch.nn.utils.weight_norm is deprecated"
   - This is in Kokoro's model code using the old weight_norm API
   - Kokoro would need to migrate to torch.nn.utils.parametrizations.weight_norm
   - This is a known issue and doesn't affect functionality
"""

import os
import io
import torch
import numpy as np
import soundfile as sf
from typing import Optional, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path

# Configure pydub to find ffmpeg in Pinokio's miniconda
# This is needed because ffmpeg may not be in the system PATH
FFMPEG_PATH = Path("C:/pinokio/bin/miniconda/Library/bin/ffmpeg.exe")
FFPROBE_PATH = Path("C:/pinokio/bin/miniconda/Library/bin/ffprobe.exe")
if FFMPEG_PATH.exists():
    from pydub import AudioSegment
    AudioSegment.converter = str(FFMPEG_PATH)
    if FFPROBE_PATH.exists():
        AudioSegment.ffprobe = str(FFPROBE_PATH)

# Try to import TTS libraries
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KPipeline = None

try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    engine: str = "kokoro"  # "kokoro" or "chatterbox"
    voice: str = "af_heart"  # Kokoro voice ID or path to reference audio
    speed: float = 1.0
    pitch: float = 0.0  # semitones, for Kokoro
    exaggeration: float = 0.5  # for Chatterbox
    cfg_weight: float = 0.3  # for Chatterbox
    sample_rate: int = 24000


@dataclass
class GeneratedAudio:
    """Container for generated audio data."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str
    voice: str


class KokoroEngine:
    """Wrapper for Kokoro-82M TTS engine."""
    
    # All 47 Kokoro voices organized by language
    VOICES = {
        "American English": [
            ("af_alloy", "Af Alloy"), ("af_aoede", "Af Aoede"), ("af_bella", "Af Bella"),
            ("af_heart", "Af Heart"), ("af_jessica", "Af Jessica"), ("af_kore", "Af Kore"),
            ("af_nicole", "Af Nicole"), ("af_nova", "Af Nova"), ("af_river", "Af River"),
            ("af_sarah", "Af Sarah"), ("af_sky", "Af Sky"),
            ("am_adam", "Am Adam"), ("am_echo", "Am Echo"), ("am_eric", "Am Eric"),
            ("am_fenrir", "Am Fenrir"), ("am_liam", "Am Liam"), ("am_michael", "Am Michael"),
            ("am_onyx", "Am Onyx"), ("am_puck", "Am Puck"), ("am_santa", "Am Santa")
        ],
        "British English": [
            ("bf_alice", "Bf Alice"), ("bf_emma", "Bf Emma"), ("bf_isabella", "Bf Isabella"),
            ("bf_lily", "Bf Lily"),
            ("bm_daniel", "Bm Daniel"), ("bm_fable", "Bm Fable"), ("bm_george", "Bm George"),
            ("bm_lewis", "Bm Lewis")
        ],
        "Brazilian Portuguese": [
            ("pf_camila", "Pf Camila"), ("pf_dora", "Pf Dora"),
            ("pm_alex", "Pm Alex"), ("pm_santa", "Pm Santa")
        ],
        "Chinese": [
            ("zf_xiaobei", "Zf Xiaobei"), ("zf_xiaoni", "Zf Xiaoni"),
            ("zf_xiaoxiao", "Zf Xiaoxiao"), ("zf_xiaoyi", "Zf Xiaoyi"),
            ("zm_yunjian", "Zm Yunjian"), ("zm_yunxi", "Zm Yunxi"),
            ("zm_yunxia", "Zm Yunxia"), ("zm_yunyang", "Zm Yunyang")
        ],
        "Japanese": [
            ("jf_alpha", "Jf Alpha"), ("jf_gongitsune", "Jf Gongitsune"),
            ("jf_nezumi", "Jf Nezumi"), ("jf_tebukuro", "Jf Tebukuro"),
            ("jm_kumo", "Jm Kumo")
        ],
        "Spanish": [
            ("ef_dora", "Ef Dora"),
            ("em_alex", "Em Alex"), ("em_santa", "Em Santa")
        ]
    }
    
    def __init__(self, device: str = "cuda"):
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro is not installed. Run: pip install kokoro")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self._pipelines: Dict[str, KPipeline] = {}
        self.sample_rate = 24000
    
    def _get_pipeline(self, lang_code: str = "a") -> KPipeline:
        """Get or create a pipeline for the specified language."""
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M"
            )
        return self._pipelines[lang_code]
    
    def _get_lang_code(self, voice: str) -> str:
        """Determine language code from voice ID."""
        prefix = voice[:2].lower() if len(voice) >= 2 else "a"
        lang_map = {
            "af": "a", "am": "a",  # American English
            "bf": "b", "bm": "b",  # British English
            "pf": "p", "pm": "p",  # Brazilian Portuguese
            "zf": "z", "zm": "z",  # Chinese
            "jf": "j", "jm": "j",  # Japanese
            "ef": "e", "em": "e",  # Spanish
        }
        return lang_map.get(prefix, "a")
    
    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        pitch: float = 0.0
    ) -> GeneratedAudio:
        """Generate audio from text."""
        lang_code = self._get_lang_code(voice)
        pipeline = self._get_pipeline(lang_code)
        
        # Generate audio
        generator = pipeline(text, voice=voice, speed=speed)
        
        # Collect all audio chunks
        audio_chunks = []
        for _, _, audio_chunk in generator:
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)
        
        if not audio_chunks:
            raise ValueError("No audio generated")
        
        # Concatenate chunks
        audio = np.concatenate(audio_chunks)
        duration = len(audio) / self.sample_rate
        
        return GeneratedAudio(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            text=text,
            voice=voice
        )
    
    def generate_stream(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Generate audio in streaming mode, yielding (phonemes, audio) tuples."""
        lang_code = self._get_lang_code(voice)
        pipeline = self._get_pipeline(lang_code)
        
        for graphemes, phonemes, audio in pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                yield phonemes, audio
    
    def get_all_voices(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return all available voices organized by language."""
        return self.VOICES.copy()
    
    def preview_voice(self, voice: str) -> GeneratedAudio:
        """Generate a preview sample for a voice."""
        preview_text = f"Hello, this is {voice.replace('_', ' ')}. I'm one of the Kokoro voices available for text-to-speech synthesis."
        return self.generate(preview_text, voice=voice)


class ChatterboxEngine:
    """Wrapper for Chatterbox TTS engine with voice cloning."""
    
    def __init__(self, device: str = "cuda"):
        if not CHATTERBOX_AVAILABLE:
            raise ImportError("Chatterbox is not installed. Run: pip install chatterbox-tts")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self.sample_rate = 24000
    
    @property
    def model(self) -> "ChatterboxTTS":
        """Lazy-load the model."""
        if self._model is None:
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
        return self._model
    
    def generate(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.3
    ) -> GeneratedAudio:
        """
        Generate audio from text, optionally cloning a reference voice.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file (5-10s) for voice cloning
            exaggeration: Voice exaggeration factor (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
        """
        # Generate with or without reference audio
        if reference_audio and os.path.exists(reference_audio):
            audio = self.model.generate(
                text,
                audio_prompt_path=reference_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        else:
            audio = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        duration = len(audio) / self.sample_rate
        
        return GeneratedAudio(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            text=text,
            voice=reference_audio or "default"
        )


class TTSManager:
    """
    Unified TTS manager supporting both Kokoro and Chatterbox engines.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._kokoro: Optional[KokoroEngine] = None
        self._chatterbox: Optional[ChatterboxEngine] = None
    
    @property
    def kokoro(self) -> KokoroEngine:
        """Lazy-load Kokoro engine."""
        if self._kokoro is None:
            self._kokoro = KokoroEngine(self.device)
        return self._kokoro
    
    @property
    def chatterbox(self) -> ChatterboxEngine:
        """Lazy-load Chatterbox engine."""
        if self._chatterbox is None:
            self._chatterbox = ChatterboxEngine(self.device)
        return self._chatterbox
    
    def generate(
        self,
        text: str,
        config: TTSConfig
    ) -> GeneratedAudio:
        """Generate audio using the configured engine."""
        if config.engine == "kokoro":
            return self.kokoro.generate(
                text=text,
                voice=config.voice,
                speed=config.speed,
                pitch=config.pitch
            )
        elif config.engine == "chatterbox":
            return self.chatterbox.generate(
                text=text,
                reference_audio=config.voice if os.path.exists(str(config.voice)) else None,
                exaggeration=config.exaggeration,
                cfg_weight=config.cfg_weight
            )
        else:
            raise ValueError(f"Unknown engine: {config.engine}")
    
    def save_audio(
        self,
        audio: GeneratedAudio,
        output_path: str,
        format: str = "mp3",
        bitrate: int = 128
    ) -> str:
        """Save generated audio to file."""
        from pydub import AudioSegment
        
        # Convert numpy array to AudioSegment
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio.audio, audio.sample_rate, format='wav')
        audio_bytes.seek(0)
        
        segment = AudioSegment.from_wav(audio_bytes)
        
        # Export in requested format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "mp3":
            segment.export(str(output_path), format="mp3", bitrate=f"{bitrate}k")
        elif format.lower() == "wav":
            segment.export(str(output_path), format="wav")
        elif format.lower() == "flac":
            segment.export(str(output_path), format="flac")
        elif format.lower() == "ogg":
            segment.export(str(output_path), format="ogg")
        else:
            segment.export(str(output_path), format=format)
        
        return str(output_path)
    
    def get_available_engines(self) -> Dict[str, bool]:
        """Return availability status of each engine."""
        return {
            "kokoro": KOKORO_AVAILABLE,
            "chatterbox": CHATTERBOX_AVAILABLE
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Return GPU information."""
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "cuda_version": torch.version.cuda
            }
        return {"available": False}
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
