"""Thread-safe in-memory job store for asynchronous analyses."""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

from server.schemas import AnalysisResponse, AnalysisResult

_TTL_SECONDS = 3600


@dataclass
class _Entry:
    status: str = "pending"
    progress: float = 0.0
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class AnalysisStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[str, _Entry] = {}

    def _purge_expired(self):
        cutoff = time.time() - _TTL_SECONDS
        expired = [k for k, e in self._entries.items() if e.created_at < cutoff]
        for k in expired:
            del self._entries[k]

    def create(self) -> str:
        analysis_id = uuid.uuid4().hex
        with self._lock:
            self._purge_expired()
            self._entries[analysis_id] = _Entry()
        return analysis_id

    def set_running(self, analysis_id: str) -> None:
        with self._lock:
            if analysis_id in self._entries:
                self._entries[analysis_id].status = "running"

    def set_progress(self, analysis_id: str, progress: float) -> None:
        with self._lock:
            if analysis_id in self._entries:
                self._entries[analysis_id].progress = max(0.0, min(1.0, progress))

    def complete(self, analysis_id: str, result: AnalysisResult) -> None:
        with self._lock:
            if analysis_id in self._entries:
                e = self._entries[analysis_id]
                e.status = "completed"
                e.progress = 1.0
                e.result = result

    def fail(self, analysis_id: str, error: str) -> None:
        with self._lock:
            if analysis_id in self._entries:
                e = self._entries[analysis_id]
                e.status = "failed"
                e.error = error

    def get(self, analysis_id: str) -> Optional[AnalysisResponse]:
        with self._lock:
            e = self._entries.get(analysis_id)
            if e is None:
                return None
            return AnalysisResponse(
                analysis_id=analysis_id,
                status=e.status,
                progress=e.progress,
                result=e.result,
                error=e.error,
            )


EXECUTOR = ThreadPoolExecutor(max_workers=2)


def submit_analysis(store: AnalysisStore, orchestrator, conv) -> str:
    analysis_id = store.create()

    def _run():
        store.set_running(analysis_id)
        try:
            result = orchestrator.analyze(
                conv, progress_cb=lambda p: store.set_progress(analysis_id, p)
            )
            store.complete(analysis_id, result)
        except Exception as e:  # pragma: no cover - defensive
            store.fail(analysis_id, str(e))

    EXECUTOR.submit(_run)
    return analysis_id
