"""
analyst_dashboard/universe/acquisition/storage.py

Isolated local filesystem storage for raw directory response bytes and metadata sidecars.

Storage Safety Invariants:
- Preserves raw bytes exactly as received prior to any normalization.
- Uses atomic writes (.tmp write followed by os.replace) to prevent partial file corruption.
- Failed or malformed acquisitions NEVER overwrite previously accepted valid artifacts.
- Failed attempts are archived to isolated quarantine logs.
- Pure engineering/research archive: distinct from UniverseStore or database publication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AcquisitionAttempt, AcquisitionOutcome


class ArtifactStorage:
    """
    Manages safe, atomic persistence of raw directory bytes and metadata sidecars.
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir = self.base_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _determine_filename(self, source_id: str, content_type: str | None) -> str:
        """
        Derives standard artifact filename based on source ID and MIME type.
        Enforces strict path traversal sanitization.
        """
        raw_name = Path(source_id).name
        clean_id = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-"))
        if not clean_id:
            clean_id = "unknown_source"

        sid_lower = clean_id.lower()
        ct_lower = (content_type or "").lower()
        if "json" in ct_lower or "json" in sid_lower or "sec" in sid_lower:
            ext = ".json"
        else:
            ext = ".txt"
        return f"{clean_id}{ext}"

    def save_artifact(
        self,
        attempt: AcquisitionAttempt,
        raw_bytes: bytes | None,
    ) -> AcquisitionAttempt:
        """
        Saves raw bytes and metadata sidecar atomically for individual files.
        Enforces the Overwrite Safety Invariant:
        If acquisition failed or bytes are None, does NOT overwrite any existing accepted artifact.
        Instead, stores failure metadata to the quarantine log.

        Note: Provides atomic replacement of individual files via .tmp staging and replace();
        does not constitute a multi-file transactional bundle.
        """
        filename = self._determine_filename(attempt.source_id, attempt.content_type)
        target_path = (self.base_dir / filename).resolve()
        meta_path = (self.base_dir / f"{filename}.meta.json").resolve()

        # Enforce boundary containment: target cannot write outside base_dir
        if target_path.parent != self.base_dir:
            attempt.outcome = AcquisitionOutcome.FAILED
            attempt.error_message = f"Path traversal rejected: target '{target_path}' escapes base directory '{self.base_dir}'"
            return attempt

        # 1. Handle Failed / Incomplete Attempts
        if attempt.outcome != AcquisitionOutcome.SUCCESS or raw_bytes is None:
            # Save failure record in quarantine directory without touching valid artifacts
            timestamp_str = attempt.retrieval_start_utc.replace(":", "-").replace(".", "_")
            fail_meta_name = f"failed_{attempt.source_id}_{timestamp_str}.meta.json"
            fail_meta_path = self.quarantine_dir / fail_meta_name

            with open(fail_meta_path, "w", encoding="utf-8") as f:
                json.dump(attempt.to_dict(), f, indent=2)

            attempt.metadata_artifact_path = str(fail_meta_path)
            attempt.raw_artifact_path = None
            return attempt

        # 2. Atomic Write for Accepted Artifacts
        tmp_raw = self.base_dir / f".tmp_{filename}"
        tmp_meta = self.base_dir / f".tmp_{filename}.meta.json"

        try:
            # Write raw bytes
            with open(tmp_raw, "wb") as f:
                f.write(raw_bytes)

            attempt.raw_artifact_path = str(target_path)
            attempt.metadata_artifact_path = str(meta_path)

            # Write sidecar metadata
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(attempt.to_dict(), f, indent=2)

            # Atomic replace
            tmp_raw.replace(target_path)
            tmp_meta.replace(meta_path)

        except Exception as e:
            # Cleanup temp files on disk error
            if tmp_raw.exists():
                tmp_raw.unlink()
            if tmp_meta.exists():
                tmp_meta.unlink()
            attempt.outcome = AcquisitionOutcome.FAILED
            attempt.error_message = f"Failed to persist artifact atomically: {e}"

        return attempt

    def load_artifact(self, source_id: str) -> tuple[bytes | None, dict[str, Any] | None]:
        """Loads preserved artifact bytes and sidecar metadata if available."""
        # Check for both .json and .txt variants
        for ext in [".json", ".txt"]:
            raw_path = self.base_dir / f"{source_id}{ext}"
            meta_path = self.base_dir / f"{source_id}{ext}.meta.json"
            if raw_path.exists() and meta_path.exists():
                with open(raw_path, "rb") as f:
                    raw_bytes = f.read()
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return raw_bytes, meta

        return None, None
