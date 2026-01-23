"""
Voice Manager Module
Speaker tag parsing, voice assignment, and custom voice blending.
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import uuid


@dataclass
class Speaker:
    """A speaker/character in the text."""
    name: str
    voice_id: str
    engine: str  # "kokoro" or "chatterbox"
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextSegment:
    """A segment of text with speaker assignment."""
    text: str
    speaker: Optional[str]
    start: int
    end: int


@dataclass
class CustomVoiceBlend:
    """A custom Kokoro voice blend."""
    id: str
    name: str
    voices: List[Tuple[str, float]]  # (voice_id, weight)
    language: str
    created_at: str


@dataclass
class ChatterboxVoice:
    """A Chatterbox voice clone reference."""
    id: str
    name: str
    audio_path: str
    created_at: str


class VoiceManager:
    """
    Manages voice assignments and custom voice creation.
    """
    
    # Speaker tag regex pattern
    SPEAKER_TAG_PATTERN = re.compile(
        r'\[([^\]\/]+)\](.*?)\[\/\1\]',
        re.DOTALL | re.IGNORECASE
    )
    
    # Alternative patterns
    ALT_PATTERNS = [
        # <speaker>text</speaker>
        re.compile(r'<([^>\/]+)>(.*?)</\1>', re.DOTALL | re.IGNORECASE),
        # narrator: text
        re.compile(r'^(\w+):\s*(.+)$', re.MULTILINE),
    ]
    
    # Paralinguistic tags
    PARALINGUISTIC_TAGS = [
        "[clear throat]", "[sigh]", "[cough]", "[groan]",
        "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
        "[whisper]", "[shout]", "[pause]"
    ]
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize voice manager.
        
        Args:
            data_dir: Directory for storing voice data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/voices")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.speakers: Dict[str, Speaker] = {}
        self.custom_blends: Dict[str, CustomVoiceBlend] = {}
        self.chatterbox_voices: Dict[str, ChatterboxVoice] = {}
        
        self._load_saved_voices()
    
    def _load_saved_voices(self):
        """Load saved custom voices from disk."""
        # Load custom blends
        blends_file = self.data_dir / "custom_blends.json"
        if blends_file.exists():
            try:
                with open(blends_file, 'r') as f:
                    data = json.load(f)
                for item in data:
                    blend = CustomVoiceBlend(
                        id=item['id'],
                        name=item['name'],
                        voices=[(v[0], v[1]) for v in item['voices']],
                        language=item.get('language', 'a'),
                        created_at=item['created_at']
                    )
                    self.custom_blends[blend.id] = blend
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Load Chatterbox voices
        cb_file = self.data_dir / "chatterbox_voices.json"
        if cb_file.exists():
            try:
                with open(cb_file, 'r') as f:
                    data = json.load(f)
                for item in data:
                    voice = ChatterboxVoice(
                        id=item['id'],
                        name=item['name'],
                        audio_path=item['audio_path'],
                        created_at=item['created_at']
                    )
                    self.chatterbox_voices[voice.id] = voice
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _save_custom_blends(self):
        """Save custom blends to disk."""
        blends_file = self.data_dir / "custom_blends.json"
        data = [
            {
                'id': b.id,
                'name': b.name,
                'voices': [[v[0], v[1]] for v in b.voices],
                'language': b.language,
                'created_at': b.created_at
            }
            for b in self.custom_blends.values()
        ]
        with open(blends_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_chatterbox_voices(self):
        """Save Chatterbox voices to disk."""
        cb_file = self.data_dir / "chatterbox_voices.json"
        data = [
            {
                'id': v.id,
                'name': v.name,
                'audio_path': v.audio_path,
                'created_at': v.created_at
            }
            for v in self.chatterbox_voices.values()
        ]
        with open(cb_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def parse_speaker_tags(self, text: str) -> List[TextSegment]:
        """
        Parse text with speaker tags into segments.
        
        Args:
            text: Text with speaker tags like [narrator]text[/narrator]
            
        Returns:
            List of TextSegment objects
        """
        segments = []
        last_end = 0
        
        # Find all speaker-tagged segments
        for match in self.SPEAKER_TAG_PATTERN.finditer(text):
            # Add any untagged text before this match
            if match.start() > last_end:
                untagged = text[last_end:match.start()].strip()
                if untagged:
                    segments.append(TextSegment(
                        text=untagged,
                        speaker=None,
                        start=last_end,
                        end=match.start()
                    ))
            
            # Add the tagged segment
            speaker = match.group(1).strip().lower()
            content = match.group(2).strip()
            
            if content:
                segments.append(TextSegment(
                    text=content,
                    speaker=speaker,
                    start=match.start(),
                    end=match.end()
                ))
            
            last_end = match.end()
        
        # Add any remaining untagged text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append(TextSegment(
                    text=remaining,
                    speaker=None,
                    start=last_end,
                    end=len(text)
                ))
        
        # If no tags found, return entire text as one segment
        if not segments:
            segments.append(TextSegment(
                text=text,
                speaker=None,
                start=0,
                end=len(text)
            ))
        
        return segments
    
    def get_unique_speakers(self, segments: List[TextSegment]) -> List[str]:
        """Get list of unique speakers from segments."""
        speakers = set()
        for seg in segments:
            if seg.speaker:
                speakers.add(seg.speaker)
        return sorted(speakers)
    
    def assign_voice(
        self,
        speaker_name: str,
        voice_id: str,
        engine: str = "kokoro",
        settings: Optional[Dict[str, Any]] = None
    ):
        """
        Assign a voice to a speaker.
        
        Args:
            speaker_name: Name of the speaker
            voice_id: Voice ID (Kokoro voice, blend ID, or Chatterbox voice)
            engine: TTS engine to use
            settings: Additional voice settings
        """
        self.speakers[speaker_name.lower()] = Speaker(
            name=speaker_name,
            voice_id=voice_id,
            engine=engine,
            settings=settings or {}
        )
    
    def get_voice(self, speaker_name: str) -> Optional[Speaker]:
        """Get voice assignment for a speaker."""
        return self.speakers.get(speaker_name.lower())
    
    def create_custom_blend(
        self,
        name: str,
        voices: List[Tuple[str, float]],
        language: str = "a"
    ) -> CustomVoiceBlend:
        """
        Create a custom Kokoro voice blend.
        
        Args:
            name: Name for the blend
            voices: List of (voice_id, weight) tuples
            language: Language code
            
        Returns:
            Created CustomVoiceBlend
        """
        from datetime import datetime
        
        blend = CustomVoiceBlend(
            id=f"custom_{uuid.uuid4().hex[:8]}",
            name=name,
            voices=voices,
            language=language,
            created_at=datetime.now().isoformat()
        )
        
        self.custom_blends[blend.id] = blend
        self._save_custom_blends()
        
        return blend
    
    def delete_custom_blend(self, blend_id: str) -> bool:
        """Delete a custom voice blend."""
        if blend_id in self.custom_blends:
            del self.custom_blends[blend_id]
            self._save_custom_blends()
            return True
        return False
    
    def add_chatterbox_voice(
        self,
        name: str,
        audio_path: str
    ) -> ChatterboxVoice:
        """
        Add a Chatterbox voice clone reference.
        
        Args:
            name: Name for the voice
            audio_path: Path to reference audio file
            
        Returns:
            Created ChatterboxVoice
        """
        from datetime import datetime
        
        # Copy audio to voices directory
        import shutil
        audio_path = Path(audio_path)
        dest_dir = self.data_dir / "reference_audio"
        dest_dir.mkdir(exist_ok=True)
        
        dest_path = dest_dir / f"{uuid.uuid4().hex[:8]}_{audio_path.name}"
        shutil.copy(audio_path, dest_path)
        
        voice = ChatterboxVoice(
            id=f"cb_{uuid.uuid4().hex[:8]}",
            name=name,
            audio_path=str(dest_path),
            created_at=datetime.now().isoformat()
        )
        
        self.chatterbox_voices[voice.id] = voice
        self._save_chatterbox_voices()
        
        return voice
    
    def delete_chatterbox_voice(self, voice_id: str) -> bool:
        """Delete a Chatterbox voice clone."""
        if voice_id in self.chatterbox_voices:
            voice = self.chatterbox_voices[voice_id]
            # Delete audio file if it exists
            audio_path = Path(voice.audio_path)
            if audio_path.exists():
                audio_path.unlink()
            
            del self.chatterbox_voices[voice_id]
            self._save_chatterbox_voices()
            return True
        return False
    
    def get_all_available_voices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available voices organized by category.
        
        Returns:
            Dictionary with 'kokoro', 'custom_blends', 'chatterbox' keys
        """
        from .tts_engine import KokoroEngine
        
        result = {
            'kokoro': [],
            'custom_blends': [],
            'chatterbox': []
        }
        
        # Kokoro voices
        try:
            kokoro_voices = KokoroEngine.VOICES
            for lang, voices in kokoro_voices.items():
                for voice_id, voice_name in voices:
                    result['kokoro'].append({
                        'id': voice_id,
                        'name': voice_name,
                        'language': lang
                    })
        except Exception:
            pass
        
        # Custom blends
        for blend in self.custom_blends.values():
            result['custom_blends'].append({
                'id': blend.id,
                'name': blend.name,
                'voices': blend.voices,
                'language': blend.language
            })
        
        # Chatterbox voices
        for voice in self.chatterbox_voices.values():
            result['chatterbox'].append({
                'id': voice.id,
                'name': voice.name,
                'audio_path': voice.audio_path
            })
        
        return result
    
    def clear_speaker_assignments(self):
        """Clear all speaker-to-voice assignments."""
        self.speakers.clear()
