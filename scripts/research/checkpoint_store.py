"""ARX Terminal — Resumable Checkpoint Store (V1.1.0).

Provides incremental, atomic, and crash-resilient persistence of ETF series mapping
and mandate parsing records during pipeline execution.

Version: CHECKPOINT_STORE_V1_1_0
Authoritative Schema:
Enforces deterministic semantic checkpoint identity containing:
- symbol
- cik
- series_id
- class_id
- source_accession
- source_sha256
- selector_version
- document_index_version
- series_resolver_version
- mandate_parser_version
- policy_version
- snapshot_boundary

A checkpoint record is valid and reusable ONLY when all semantic inputs match identically.
Mutation of any authority, source document, or version invalidates the checkpoint.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Set, List, Union


CHECKPOINT_STORE_VERSION = "CHECKPOINT_STORE_V1_1_0"


@dataclass
class CheckpointIdentity:
    """Canonical semantic input identity for a target checkpoint."""
    symbol: str
    cik: str = ""
    series_id: str = ""
    class_id: str = ""
    source_accession: str = ""
    source_sha256: str = ""
    selector_version: str = ""
    document_index_version: str = ""
    series_resolver_version: str = ""
    mandate_parser_version: str = ""
    policy_version: str = ""
    snapshot_boundary: str = ""

    def to_identity_key(self) -> str:
        return ":".join([
            self.symbol,
            self.cik,
            self.series_id,
            self.class_id,
            self.source_accession,
            self.source_sha256,
            self.selector_version,
            self.document_index_version,
            self.series_resolver_version,
            self.mandate_parser_version,
            self.policy_version,
            self.snapshot_boundary,
        ])


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
    # Semantic identity fields (CHECKPOINT_STORE_V1_1_0)
    source_accession: str = ""
    source_sha256: str = ""
    selector_version: str = ""
    document_index_version: str = ""
    series_resolver_version: str = ""
    mandate_parser_version: str = ""
    policy_version: str = ""
    snapshot_boundary: str = ""
    checkpoint_schema_version: str = CHECKPOINT_STORE_VERSION
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
            source_accession=data.get("source_accession", ""),
            source_sha256=data.get("source_sha256", ""),
            selector_version=data.get("selector_version", ""),
            document_index_version=data.get("document_index_version", ""),
            series_resolver_version=data.get("series_resolver_version", ""),
            mandate_parser_version=data.get("mandate_parser_version", ""),
            policy_version=data.get("policy_version", ""),
            snapshot_boundary=data.get("snapshot_boundary", ""),
            checkpoint_schema_version=data.get("checkpoint_schema_version", "CHECKPOINT_STORE_V1_0_0"),
            extra_data=data.get("extra_data", {}),
        )

    def matches_identity(self, identity: CheckpointIdentity) -> bool:
        """Verifies that all semantically relevant inputs match this record."""
        if identity.symbol and self.symbol != identity.symbol:
            return False
        if identity.cik and self.cik != identity.cik:
            return False
        if identity.series_id and self.series_id != identity.series_id:
            return False
        if identity.class_id and self.class_id != identity.class_id:
            return False
        if identity.source_accession and self.source_accession != identity.source_accession:
            return False
        if identity.source_sha256 and self.source_sha256 != identity.source_sha256:
            return False
        if identity.selector_version and self.selector_version != identity.selector_version:
            return False
        if identity.document_index_version and self.document_index_version != identity.document_index_version:
            return False
        if identity.series_resolver_version and self.series_resolver_version != identity.series_resolver_version:
            return False
        if identity.mandate_parser_version and self.mandate_parser_version != identity.mandate_parser_version:
            return False
        if identity.policy_version and self.policy_version != identity.policy_version:
            return False
        if identity.snapshot_boundary and self.snapshot_boundary != identity.snapshot_boundary:
            return False
        return True


class CheckpointStore:
    """Append-only, crash-resilient JSON Lines checkpoint store with versioned semantic keying."""

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
                    self._completed_records[record.symbol] = record
                    loaded_count += 1
                except Exception:
                    # Skip corrupted or partial trailing lines
                    continue
        return loaded_count

    def is_completed(
        self,
        symbol: str,
        identity: Optional[CheckpointIdentity] = None,
        **identity_kwargs: Any
    ) -> bool:
        """Checks if a target has been completed.

        If identity or identity_kwargs are supplied, the existing record must match
        ALL semantic inputs identically. Any difference invalidates completion.
        """
        record = self._completed_records.get(symbol)
        if record is None:
            return False

        if identity is not None:
            return record.matches_identity(identity)

        if identity_kwargs:
            check_id = CheckpointIdentity(symbol=symbol, **identity_kwargs)
            return record.matches_identity(check_id)

        # If no identity specified, require that the record was created under current manifest
        if record.input_manifest_sha256 and record.input_manifest_sha256 != self.input_manifest_sha256:
            return False

        return True

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
        source_accession: str = "",
        source_sha256: str = "",
        selector_version: str = "",
        document_index_version: str = "",
        series_resolver_version: str = "",
        mandate_parser_version: str = "",
        policy_version: str = "",
        snapshot_boundary: str = "",
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
            source_accession=source_accession,
            source_sha256=source_sha256,
            selector_version=selector_version,
            document_index_version=document_index_version,
            series_resolver_version=series_resolver_version,
            mandate_parser_version=mandate_parser_version,
            policy_version=policy_version,
            snapshot_boundary=snapshot_boundary,
            checkpoint_schema_version=CHECKPOINT_STORE_VERSION,
            extra_data=extra_data or {},
        )

        # Append to checkpoint file with sync
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._completed_records[symbol] = record
        return record
