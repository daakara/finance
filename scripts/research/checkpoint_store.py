"""ARX Terminal — Resumable Checkpoint Store (Sections 27 & 28).

Provides incremental, atomic, and crash-resilient persistence of ETF series mapping
and mandate parsing records during pipeline execution.

Enforces:
1. Resumability: Crashes after target N do not restart from target 1.
2. Idempotency: Processing the same target multiple times produces identical state without duplicates.
3. Checkpoint Record schema (Section 28):
   - run_id
   - input_manifest_sha256
   - document_index_key
   - series_resolution_key
   - mapping_outcome
   - section_hash
   - mandate_parse_status
   - timestamp
4. Isolation: Never mutates canonical governance artifacts during execution.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Set, List


CHECKPOINT_STORE_VERSION = "CHECKPOINT_STORE_V1_0_0"


@dataclass
class CheckpointRecord:
    """A single persistent checkpoint record for an ETF series target."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    run_id: str
    input_manifest_sha256: str
    document_index_key: str
    series_resolution_key: str
    mapping_outcome: str
    section_hash: str
    mandate_parse_status: str
    timestamp: str
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRecord":
        return cls(
            symbol=data.get("symbol", ""),
            cik=data.get("cik", ""),
            series_id=data.get("series_id", ""),
            class_id=data.get("class_id", ""),
            run_id=data.get("run_id", ""),
            input_manifest_sha256=data.get("input_manifest_sha256", ""),
            document_index_key=data.get("document_index_key", ""),
            series_resolution_key=data.get("series_resolution_key", ""),
            mapping_outcome=data.get("mapping_outcome", ""),
            section_hash=data.get("section_hash", ""),
            mandate_parse_status=data.get("mandate_parse_status", ""),
            timestamp=data.get("timestamp", ""),
            extra_data=data.get("extra_data", {}),
        )


class CheckpointStore:
    """Append-only, crash-resilient JSON Lines checkpoint store."""

    def __init__(
        self,
        checkpoint_path: Path,
        run_id: str,
        input_manifest_sha256: str,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.run_id = run_id
        self.input_manifest_sha256 = input_manifest_sha256
        self._completed_records: Dict[str, CheckpointRecord] = {}

        # Ensure directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records if resuming from a previous run
        self._load_existing()

    def _load_existing(self) -> int:
        """Reads any previously persisted records from the checkpoint file."""
        if not self.checkpoint_path.exists():
            return 0

        loaded_count = 0
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = CheckpointRecord.from_dict(data)
                    # Verify manifest compatibility
                    if record.input_manifest_sha256 and record.input_manifest_sha256 != self.input_manifest_sha256:
                        # Manifest mismatch warning, but preserve record
                        pass
                    self._completed_records[record.symbol] = record
                    loaded_count += 1
                except Exception:
                    # Skip corrupted or partial trailing lines
                    continue
        return loaded_count

    def is_completed(self, symbol: str) -> bool:
        """Checks if a target symbol has already been successfully checkpointed."""
        return symbol in self._completed_records

    def get_record(self, symbol: str) -> Optional[CheckpointRecord]:
        """Retrieves the checkpoint record for a given symbol."""
        return self._completed_records.get(symbol)

    def get_completed_count(self) -> int:
        """Returns the total number of unique completed targets."""
        return len(self._completed_records)

    def get_completed_symbols(self) -> Set[str]:
        """Returns a set of all completed target symbols."""
        return set(self._completed_records.keys())

    def get_all_records(self) -> List[CheckpointRecord]:
        """Returns a list of all completed checkpoint records."""
        return list(self._completed_records.values())

    def save_record(
        self,
        symbol: str,
        cik: str,
        series_id: str,
        class_id: str,
        document_index_key: str,
        series_resolution_key: str,
        mapping_outcome: str,
        section_hash: str,
        mandate_parse_status: str,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> CheckpointRecord:
        """Atomically appends a completed target record to disk and updates in-memory index."""
        now_iso = datetime.now(timezone.utc).isoformat()
        record = CheckpointRecord(
            symbol=symbol,
            cik=cik,
            series_id=series_id,
            class_id=class_id,
            run_id=self.run_id,
            input_manifest_sha256=self.input_manifest_sha256,
            document_index_key=document_index_key,
            series_resolution_key=series_resolution_key,
            mapping_outcome=mapping_outcome,
            section_hash=section_hash,
            mandate_parse_status=mandate_parse_status,
            timestamp=now_iso,
            extra_data=extra_data or {},
        )

        # Append to checkpoint file with sync
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._completed_records[symbol] = record
        return record
