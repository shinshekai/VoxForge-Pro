"""
Library Module
SQLite-based library management for generated audiobooks.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class AudiobookModel(Base):
    """SQLAlchemy model for audiobooks."""
    __tablename__ = 'audiobooks'
    
    id = Column(Integer, primary_key=True)
    book_id = Column(String(64), unique=True, nullable=False)
    title = Column(String(512), nullable=False)
    author = Column(String(256))
    
    source_path = Column(String(1024))
    source_type = Column(String(32))
    source_hash = Column(String(64))
    
    engine = Column(String(32))
    format = Column(String(16))
    bitrate = Column(Integer)
    
    duration_seconds = Column(Float)
    file_size_bytes = Column(Integer)
    chapter_count = Column(Integer)
    
    output_dir = Column(String(1024))
    chapters_json = Column(Text)  # JSON array of chapter metadata
    voices_json = Column(Text)  # JSON dict of voice assignments
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    tags = Column(String(512))
    notes = Column(Text)
    is_favorite = Column(Boolean, default=False)


@dataclass
class Audiobook:
    """Audiobook data class."""
    id: int
    book_id: str
    title: str
    author: Optional[str]
    source_path: Optional[str]
    source_type: Optional[str]
    source_hash: Optional[str]
    engine: str
    format: str
    bitrate: int
    duration_seconds: float
    file_size_bytes: int
    chapter_count: int
    output_dir: str
    chapters: List[Dict[str, Any]]
    voices: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    notes: Optional[str]
    is_favorite: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'book_id': self.book_id,
            'title': self.title,
            'author': self.author,
            'source_path': self.source_path,
            'source_type': self.source_type,
            'source_hash': self.source_hash,
            'engine': self.engine,
            'format': self.format,
            'bitrate': self.bitrate,
            'duration_seconds': self.duration_seconds,
            'file_size_bytes': self.file_size_bytes,
            'chapter_count': self.chapter_count,
            'output_dir': self.output_dir,
            'chapters': self.chapters,
            'voices': self.voices,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'tags': self.tags,
            'notes': self.notes,
            'is_favorite': self.is_favorite
        }


class Library:
    """
    Manages the audiobook library with SQLite storage.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize library.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = "library/database.sqlite"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def _model_to_dataclass(self, model: AudiobookModel) -> Audiobook:
        """Convert SQLAlchemy model to dataclass."""
        chapters = json.loads(model.chapters_json) if model.chapters_json else []
        voices = json.loads(model.voices_json) if model.voices_json else {}
        tags = model.tags.split(',') if model.tags else []
        
        return Audiobook(
            id=model.id,
            book_id=model.book_id,
            title=model.title,
            author=model.author,
            source_path=model.source_path,
            source_type=model.source_type,
            source_hash=model.source_hash,
            engine=model.engine,
            format=model.format,
            bitrate=model.bitrate,
            duration_seconds=model.duration_seconds or 0,
            file_size_bytes=model.file_size_bytes or 0,
            chapter_count=model.chapter_count or 0,
            output_dir=model.output_dir,
            chapters=chapters,
            voices=voices,
            created_at=model.created_at,
            updated_at=model.updated_at,
            tags=tags,
            notes=model.notes,
            is_favorite=model.is_favorite or False
        )
    
    def add_audiobook(
        self,
        book_id: str,
        title: str,
        output_dir: str,
        engine: str,
        format: str,
        bitrate: int,
        chapters: List[Dict[str, Any]],
        voices: Dict[str, str],
        author: Optional[str] = None,
        source_path: Optional[str] = None,
        source_type: Optional[str] = None,
        source_hash: Optional[str] = None,
        duration_seconds: float = 0,
        file_size_bytes: int = 0
    ) -> Audiobook:
        """Add a new audiobook to the library."""
        model = AudiobookModel(
            book_id=book_id,
            title=title,
            author=author,
            source_path=source_path,
            source_type=source_type,
            source_hash=source_hash,
            engine=engine,
            format=format,
            bitrate=bitrate,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            chapter_count=len(chapters),
            output_dir=output_dir,
            chapters_json=json.dumps(chapters),
            voices_json=json.dumps(voices)
        )
        
        self.session.add(model)
        self.session.commit()
        
        return self._model_to_dataclass(model)
    
    def get_audiobook(self, book_id: str) -> Optional[Audiobook]:
        """Get an audiobook by ID."""
        model = self.session.query(AudiobookModel).filter_by(book_id=book_id).first()
        return self._model_to_dataclass(model) if model else None
    
    def get_all_audiobooks(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        descending: bool = True
    ) -> List[Audiobook]:
        """Get all audiobooks with pagination."""
        query = self.session.query(AudiobookModel)
        
        # Apply sorting
        column = getattr(AudiobookModel, sort_by, AudiobookModel.created_at)
        if descending:
            query = query.order_by(column.desc())
        else:
            query = query.order_by(column)
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        return [self._model_to_dataclass(m) for m in query.all()]
    
    def search_audiobooks(
        self,
        query: str,
        limit: int = 50
    ) -> List[Audiobook]:
        """Search audiobooks by title or author."""
        pattern = f"%{query}%"
        results = self.session.query(AudiobookModel).filter(
            (AudiobookModel.title.like(pattern)) |
            (AudiobookModel.author.like(pattern))
        ).limit(limit).all()
        
        return [self._model_to_dataclass(m) for m in results]
    
    def update_audiobook(
        self,
        book_id: str,
        **kwargs
    ) -> Optional[Audiobook]:
        """Update audiobook fields."""
        model = self.session.query(AudiobookModel).filter_by(book_id=book_id).first()
        if not model:
            return None
        
        for key, value in kwargs.items():
            if key == 'chapters':
                model.chapters_json = json.dumps(value)
            elif key == 'voices':
                model.voices_json = json.dumps(value)
            elif key == 'tags':
                model.tags = ','.join(value) if isinstance(value, list) else value
            elif hasattr(model, key):
                setattr(model, key, value)
        
        self.session.commit()
        return self._model_to_dataclass(model)
    
    def delete_audiobook(self, book_id: str, delete_files: bool = True) -> bool:
        """Delete an audiobook from the library."""
        model = self.session.query(AudiobookModel).filter_by(book_id=book_id).first()
        if not model:
            return False
        
        # Delete audio files if requested
        if delete_files and model.output_dir:
            import shutil
            output_path = Path(model.output_dir)
            if output_path.exists():
                shutil.rmtree(output_path)
        
        self.session.delete(model)
        self.session.commit()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        audiobooks = self.session.query(AudiobookModel).all()
        
        total_duration = sum(m.duration_seconds or 0 for m in audiobooks)
        total_size = sum(m.file_size_bytes or 0 for m in audiobooks)
        
        return {
            'total_audiobooks': len(audiobooks),
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600,
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024**3),
            'engines': list(set(m.engine for m in audiobooks if m.engine)),
            'formats': list(set(m.format for m in audiobooks if m.format))
        }
    
    def find_duplicates(self) -> List[List[Audiobook]]:
        """Find duplicate audiobooks based on source hash."""
        from collections import defaultdict
        
        hash_groups = defaultdict(list)
        
        for model in self.session.query(AudiobookModel).filter(AudiobookModel.source_hash != None).all():
            hash_groups[model.source_hash].append(self._model_to_dataclass(model))
        
        # Return only groups with multiple items
        return [group for group in hash_groups.values() if len(group) > 1]
    
    def close(self):
        """Close database connection."""
        self.session.close()


def init_db(db_path: Optional[str] = None):
    """Initialize the database (create tables)."""
    if db_path is None:
        db_path = "library/database.sqlite"
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    
    print(f"Database initialized at: {db_path}")
