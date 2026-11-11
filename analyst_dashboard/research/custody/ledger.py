"""
ARX Terminal — Tamper-Evident Custody Receipt Ledger (Gate F)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Operational Mandate:
- Append-only hash-chained JSON-Lines ledger stored in vault ledger partition.
- Genesis record anchored to 64-zero digest.
- Strict data minimization: no legal names, emails, raw ratings, or notes recorded.
- Cryptographic verification of chain continuity (detects modification, deletion, reordering).
- Explicit governance distinction: local tamper-evidence vs. external immutable anchoring.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from analyst_dashboard.research.custody.exceptions import LedgerIntegrityError

GENESIS_PREVIOUS_DIGEST = "0" * 64


def compute_receipt_digest(record_payload: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 digest of receipt record.
    Excludes 'receipt_digest' field itself and sorts keys canonically.
    """
    payload_copy = {k: v for k, v in record_payload.items() if k != "receipt_digest"}
    canonical_json = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class TamperEvidentLedger:
    """
    Append-only hash-chained receipt ledger for evidence custody events.
    """

    def __init__(self, ledger_file_path: Path | str):
        self.ledger_path = Path(ledger_file_path).resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_externally_anchored = False  # Local tamper-evidence; external anchoring held

    def get_all_receipts(self) -> List[Dict[str, Any]]:
        """Read and parse all receipts from the ledger file."""
        if not self.ledger_path.exists():
            return []

        receipts = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    receipts.append(record)
                except json.JSONDecodeError as e:
                    raise LedgerIntegrityError(
                        f"Corrupted receipt line {line_no} in ledger: {e}"
                    ) from e
        return receipts

    def get_last_receipt(self) -> Optional[Dict[str, Any]]:
        """Return the most recent receipt, or None if ledger is empty."""
        receipts = self.get_all_receipts()
        return receipts[-1] if receipts else None

    def append_receipt(
        self,
        pseudonymous_observer_id: str,
        batch_id: str,
        package_sha256: str,
        raw_submission_sha256: str,
        schema_validation_status: str,
        custody_state: str,
        receipt_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Construct, hash-chain, and append a new receipt to the ledger.
        """
        # Validate minimal fields
        if not pseudonymous_observer_id or not batch_id:
            raise LedgerIntegrityError("Observer ID and Batch ID are required for receipt.")

        last_receipt = self.get_last_receipt()
        previous_digest = (
            last_receipt["receipt_digest"] if last_receipt else GENESIS_PREVIOUS_DIGEST
        )

        now_utc = datetime.now(timezone.utc).isoformat()
        rcpt_id = receipt_id or f"RCPT-{uuid.uuid4().hex[:12].upper()}"

        record: Dict[str, Any] = {
            "receipt_id": rcpt_id,
            "timestamp_utc": now_utc,
            "pseudonymous_observer_id": pseudonymous_observer_id,
            "batch_id": batch_id,
            "package_sha256": package_sha256,
            "raw_submission_sha256": raw_submission_sha256,
            "schema_validation_status": schema_validation_status,
            "custody_state": custody_state,
            "previous_receipt_digest": previous_digest,
        }

        if custom_metadata:
            # Enforce data minimization on metadata
            forbidden_keys = {"name", "email", "phone", "notes", "ratings", "raw"}
            for k in custom_metadata:
                if k.lower() in forbidden_keys:
                    raise LedgerIntegrityError(
                        f"Forbidden identifying key '{k}' in receipt metadata."
                    )
            record["metadata"] = custom_metadata

        digest = compute_receipt_digest(record)
        record["receipt_digest"] = digest

        line_to_write = json.dumps(record, sort_keys=True) + "\n"

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(line_to_write)
            f.flush()
            os.fsync(f.fileno())

        return record

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify cryptographic hash-chain continuity across all receipts.
        Detects modification, reordering, deletion, or truncation.
        """
        receipts = self.get_all_receipts()
        if not receipts:
            return True, ["Ledger is empty (valid initial state)."]

        errors = []
        expected_prev_digest = GENESIS_PREVIOUS_DIGEST

        for idx, rcpt in enumerate(receipts):
            rcpt_id = rcpt.get("receipt_id", f"INDEX_{idx}")
            actual_prev = rcpt.get("previous_receipt_digest")

            if actual_prev != expected_prev_digest:
                errors.append(
                    f"Chain break at receipt {rcpt_id} (index {idx}): "
                    f"expected previous '{expected_prev_digest}', got '{actual_prev}'."
                )

            recorded_digest = rcpt.get("receipt_digest")
            computed_digest = compute_receipt_digest(rcpt)

            if recorded_digest != computed_digest:
                errors.append(
                    f"Digest mismatch at receipt {rcpt_id} (index {idx}): "
                    f"recorded '{recorded_digest}', computed '{computed_digest}'."
                )

            expected_prev_digest = recorded_digest

        is_valid = len(errors) == 0
        return is_valid, errors
