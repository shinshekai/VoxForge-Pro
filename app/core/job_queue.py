"""
Job Queue Module
Async job processing for audiobook generation with progress tracking.
"""

import os
import json
import uuid
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a job."""
    current_chapter: int = 0
    total_chapters: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    current_text: str = ""
    current_speaker: str = ""
    eta_seconds: float = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Job:
    """Represents an audiobook generation job."""
    id: str
    title: str
    source_path: str
    source_type: str  # "pdf", "text", "docx"
    status: JobStatus
    engine: str
    format: str
    bitrate: int
    chapters: List[Dict[str, Any]]
    voices: Dict[str, str]
    output_dir: str
    progress: JobProgress
    error: Optional[str] = None
    created_at: str = ""
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'source_path': self.source_path,
            'source_type': self.source_type,
            'status': self.status.value,
            'engine': self.engine,
            'format': self.format,
            'bitrate': self.bitrate,
            'chapters': self.chapters,
            'voices': self.voices,
            'output_dir': self.output_dir,
            'progress': {
                'current_chapter': self.progress.current_chapter,
                'total_chapters': self.progress.total_chapters,
                'current_chunk': self.progress.current_chunk,
                'total_chunks': self.progress.total_chunks,
                'current_text': self.progress.current_text[:100] if self.progress.current_text else "",
                'current_speaker': self.progress.current_speaker,
                'eta_seconds': self.progress.eta_seconds,
                'started_at': self.progress.started_at,
                'completed_at': self.progress.completed_at
            },
            'error': self.error,
            'created_at': self.created_at,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary."""
        progress = JobProgress(
            current_chapter=data['progress'].get('current_chapter', 0),
            total_chapters=data['progress'].get('total_chapters', 0),
            current_chunk=data['progress'].get('current_chunk', 0),
            total_chunks=data['progress'].get('total_chunks', 0),
            current_text=data['progress'].get('current_text', ''),
            current_speaker=data['progress'].get('current_speaker', ''),
            eta_seconds=data['progress'].get('eta_seconds', 0),
            started_at=data['progress'].get('started_at'),
            completed_at=data['progress'].get('completed_at')
        )
        
        return cls(
            id=data['id'],
            title=data['title'],
            source_path=data['source_path'],
            source_type=data['source_type'],
            status=JobStatus(data['status']),
            engine=data['engine'],
            format=data['format'],
            bitrate=data['bitrate'],
            chapters=data['chapters'],
            voices=data['voices'],
            output_dir=data['output_dir'],
            progress=progress,
            error=data.get('error'),
            created_at=data['created_at'],
            priority=data.get('priority', 0)
        )


class JobQueue:
    """
    Manages audiobook generation jobs with async processing.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_concurrent: int = 1
    ):
        """
        Initialize job queue.
        
        Args:
            data_dir: Directory for job persistence
            max_concurrent: Maximum concurrent jobs (usually 1 for GPU)
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/jobs")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent
        
        self._jobs: Dict[str, Job] = {}
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._current_job_id: Optional[str] = None
        self._callbacks: Dict[str, List[Callable]] = {
            'progress': [],
            'completed': [],
            'failed': [],
            'cancelled': []
        }
        
        self._load_jobs()
    
    def _load_jobs(self):
        """Load persisted jobs from disk."""
        jobs_file = self.data_dir / "jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    data = json.load(f)
                for job_data in data:
                    job = Job.from_dict(job_data)
                    self._jobs[job.id] = job
                    
                    # Re-queue pending jobs
                    if job.status == JobStatus.PENDING:
                        self._queue.put((job.priority, job.created_at, job.id))
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _save_jobs(self):
        """Persist jobs to disk."""
        jobs_file = self.data_dir / "jobs.json"
        with open(jobs_file, 'w') as f:
            data = [job.to_dict() for job in self._jobs.values()]
            json.dump(data, f, indent=2)
    
    def add_job(
        self,
        title: str,
        source_path: str,
        source_type: str,
        engine: str,
        format: str,
        bitrate: int,
        chapters: List[Dict[str, Any]],
        voices: Dict[str, str],
        output_dir: str,
        priority: int = 0
    ) -> Job:
        """
        Add a new job to the queue.
        
        Returns:
            Created Job object
        """
        job_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        job = Job(
            id=job_id,
            title=title,
            source_path=source_path,
            source_type=source_type,
            status=JobStatus.PENDING,
            engine=engine,
            format=format,
            bitrate=bitrate,
            chapters=chapters,
            voices=voices,
            output_dir=output_dir,
            progress=JobProgress(total_chapters=len(chapters)),
            created_at=created_at,
            priority=priority
        )
        
        with self._lock:
            self._jobs[job_id] = job
            # Priority queue uses (priority, timestamp, id) for ordering
            # Lower priority number = higher priority
            self._queue.put((-priority, created_at, job_id))
            self._save_jobs()
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs sorted by creation time (newest first)."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs
    
    def get_pending_jobs(self) -> List[Job]:
        """Get all pending jobs."""
        return [j for j in self._jobs.values() if j.status == JobStatus.PENDING]
    
    def get_current_job(self) -> Optional[Job]:
        """Get the currently processing job."""
        if self._current_job_id:
            return self._jobs.get(self._current_job_id)
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.PAUSED):
                job.status = JobStatus.CANCELLED
                self._save_jobs()
                self._trigger_callback('cancelled', job)
                return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a processing job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.PROCESSING:
                job.status = JobStatus.PAUSED
                self._save_jobs()
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.PAUSED:
                job.status = JobStatus.PENDING
                self._queue.put((-job.priority, job.created_at, job.id))
                self._save_jobs()
                return True
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job (only completed, failed, or cancelled)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                del self._jobs[job_id]
                self._save_jobs()
                return True
        return False
    
    def clear_completed(self):
        """Remove all completed jobs."""
        with self._lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.status == JobStatus.COMPLETED
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
            self._save_jobs()
    
    def update_progress(
        self,
        job_id: str,
        current_chapter: Optional[int] = None,
        current_chunk: Optional[int] = None,
        total_chunks: Optional[int] = None,
        current_text: Optional[str] = None,
        current_speaker: Optional[str] = None,
        eta_seconds: Optional[float] = None
    ):
        """Update job progress."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if current_chapter is not None:
                    job.progress.current_chapter = current_chapter
                if current_chunk is not None:
                    job.progress.current_chunk = current_chunk
                if total_chunks is not None:
                    job.progress.total_chunks = total_chunks
                if current_text is not None:
                    job.progress.current_text = current_text
                if current_speaker is not None:
                    job.progress.current_speaker = current_speaker
                if eta_seconds is not None:
                    job.progress.eta_seconds = eta_seconds
                
                self._trigger_callback('progress', job)
    
    def complete_job(self, job_id: str):
        """Mark a job as completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress.completed_at = datetime.now().isoformat()
                self._current_job_id = None
                self._save_jobs()
                self._trigger_callback('completed', job)
    
    def fail_job(self, job_id: str, error: str):
        """Mark a job as failed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.progress.completed_at = datetime.now().isoformat()
                self._current_job_id = None
                self._save_jobs()
                self._trigger_callback('failed', job)
    
    def on_progress(self, callback: Callable[[Job], None]):
        """Register a progress callback."""
        self._callbacks['progress'].append(callback)
    
    def on_completed(self, callback: Callable[[Job], None]):
        """Register a completion callback."""
        self._callbacks['completed'].append(callback)
    
    def on_failed(self, callback: Callable[[Job], None]):
        """Register a failure callback."""
        self._callbacks['failed'].append(callback)
    
    def _trigger_callback(self, event: str, job: Job):
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(job)
            except Exception:
                pass
    
    def get_queue_size(self) -> int:
        """Get number of pending jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
    
    def get_stats(self) -> Dict[str, int]:
        """Get job statistics."""
        stats = {
            'pending': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'paused': 0
        }
        
        for job in self._jobs.values():
            stats[job.status.value] = stats.get(job.status.value, 0) + 1
        
        return stats
