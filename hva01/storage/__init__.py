"""
Storage Subsystem for HVA-01.
Provides transactional, append-only persistence with cross-platform file locking and crash recovery.
"""

from .append_store import (
    AppendOnlyStore,
    CorrectionRecord,
    FileLock,
    FileLockError,
    StorageCorruptionError,
    ConflictingDuplicateSubmissionError,
)

__all__ = [
    "AppendOnlyStore",
    "CorrectionRecord",
    "FileLock",
    "FileLockError",
    "StorageCorruptionError",
    "ConflictingDuplicateSubmissionError",
]
