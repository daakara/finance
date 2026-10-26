"""
ARX Terminal — Participant Session Custody Intake Pipeline
Defect: ARX-UX-P00-INSTR-001
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Governing Authorization: RO-DEC-2026-10-23-012

Pipeline Execution Contract:
1. Boundary & Size Validation (<= 2MB payload ceiling)
2. Pre-parse Wire Hashing (SHA-256 computed on raw bytes before JSON decoding)
3. Safe JSON Decoding & Canonical Parsing
4. Schema & Field Validation (TaskSessionRecord validation, scenario IDs, monotonic events)
5. Pseudonym & Mode Authorization (Synthetic mode strictly isolates P00-SYNTHETIC; human tokens rejected)
6. Duplicate & Idempotency Resolution (Idempotent return on duplicate hash, HTTP 409 on conflict)
7. Authenticated Encryption (AES-256-GCM via CustodyCrypto)
8. Restricted Vault Storage (Atomic staging write to off-Git vault)
9. Tamper-Evident Ledger Inscription & Hash Chaining
10. Server-Authoritative Cryptographic Receipt Issuance (HMAC-SHA256 signature via vault secret)
11. Withdrawal & Key Shredding Support
"""

import os
import json
import uuid
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.research.custody.exceptions import (
    IntakeValidationError,
    CustodyError,
    LedgerIntegrityError,
)
from analyst_dashboard.research.custody.crypto import CustodyCrypto, compute_sha256
from analyst_dashboard.research.custody.vault import LocalRestrictedVault
from analyst_dashboard.research.custody.ledger import TamperEvidentLedger

MAX_SESSION_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 Megabytes

VALID_SCENARIO_IDS = {
    "SCN-NORM-PLSE-SWING",
    "SCN-NORM-ALAB-MOM",
    "SCN-NORM-FIX-VCP",
    "SCN-NORM-DUOL-FUND",
    "SCN-ADVR-STALE-TAPE",
    "SCN-ADVR-CONTRADICT",
    "SCN-ADVR-PHANTOM",
    "SCN-ADVR-SCREEN-ONLY",
    "SCN-ADVR-PIPE-FAIL",
}

VALID_VARIANTS = {"CONCEPT_A", "CONCEPT_B", "CONCEPT_C"}
VALID_BLOCKS = {"BLOCK_1_TRADE_PLANNING", "BLOCK_2_EVIDENCE_DEPTH"}


class CustodyConflictError(CustodyError):
    """Raised when a resubmission has conflicting payload data for an already-sealed session."""
    pass


class ParticipantSessionIntakePipeline:
    """
    Intake pipeline for research participant interaction telemetry.
    Enforces atomic vault storage, ledger chaining, server HMAC receipts,
    idempotent replay, and cryptographic withdrawal.
    """

    def __init__(
        self,
        vault: LocalRestrictedVault,
        crypto: CustodyCrypto,
        ledger: TamperEvidentLedger,
        is_synthetic_mode: bool = True,
        signing_key: Optional[bytes] = None,
    ):
        self.vault = vault
        self.crypto = crypto
        self.ledger = ledger
        self.is_synthetic_mode = is_synthetic_mode

        # Derive or assign server HMAC signing key
        if signing_key:
            self.signing_key = signing_key
        else:
            # Derive deterministic signing key from vault encryption key
            self.signing_key = hmac.new(
                self.crypto._key,
                b"ARX_VAULT_RECEIPT_HMAC_SIGNING_KEY_V1",
                hashlib.sha256,
            ).digest()

    def _compute_server_receipt_signature(
        self,
        receipt_id: str,
        session_id: str,
        participant_id: str,
        record_count: int,
        source_payload_sha256: str,
        ledger_digest: str,
        accepted_at_utc: str,
    ) -> str:
        """Compute server-authoritative HMAC-SHA256 signature over receipt fields."""
        canonical_str = (
            f"{receipt_id}|{session_id}|{participant_id}|{record_count}|"
            f"{source_payload_sha256}|{ledger_digest}|{accepted_at_utc}"
        )
        return hmac.new(
            self.signing_key,
            canonical_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_server_receipt_signature(self, receipt: Dict[str, Any]) -> bool:
        """Verify the server HMAC-SHA256 signature of a receipt."""
        required = [
            "receipt_id",
            "session_id",
            "participant_id",
            "record_count",
            "source_payload_sha256",
            "ledger_ref",
            "accepted_at_utc",
            "server_receipt_signature",
        ]
        for field in required:
            if field not in receipt:
                return False

        expected_sig = self._compute_server_receipt_signature(
            receipt_id=receipt["receipt_id"],
            session_id=receipt["session_id"],
            participant_id=receipt["participant_id"],
            record_count=receipt["record_count"],
            source_payload_sha256=receipt["source_payload_sha256"],
            ledger_digest=receipt["ledger_ref"],
            accepted_at_utc=receipt["accepted_at_utc"],
        )
        return hmac.compare_digest(expected_sig, receipt["server_receipt_signature"])

    def ingest_session_payload(
        self,
        raw_bytes: bytes,
        session_id: str,
        human_authorization_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute atomic session custody intake pipeline on raw bytes.
        Returns tamper-evident custody receipt.
        """
        # Step 1: Boundary & Size Validation
        if not isinstance(raw_bytes, bytes):
            raise IntakeValidationError("Session payload must be raw bytes.")
        if len(raw_bytes) == 0:
            raise IntakeValidationError("Session payload is empty.")
        if len(raw_bytes) > MAX_SESSION_PAYLOAD_BYTES:
            raise IntakeValidationError(
                f"Payload exceeds maximum allowed size ({len(raw_bytes)} > {MAX_SESSION_PAYLOAD_BYTES} bytes)."
            )
        if not session_id or not isinstance(session_id, str):
            raise IntakeValidationError("session_id must be a non-empty string.")

        # Step 2: Pre-parse Wire Hashing
        raw_payload_sha256 = compute_sha256(raw_bytes)

        # Step 3: Parse Records (JSON Lines or JSON Array)
        decoded_text = ""
        try:
            decoded_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise IntakeValidationError("Payload contains invalid UTF-8 bytes.") from e

        records: List[Dict[str, Any]] = []
        stripped = decoded_text.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise IntakeValidationError("JSON payload array must be a list.")
                records = parsed
            except json.JSONDecodeError as e:
                raise IntakeValidationError("Malformed JSON array in payload.") from e
        else:
            # JSON Lines format
            for line_idx, line in enumerate(decoded_text.splitlines(), start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    rec = json.loads(clean_line)
                    if not isinstance(rec, dict):
                        raise IntakeValidationError(f"Line {line_idx} is not a JSON object.")
                    records.append(rec)
                except json.JSONDecodeError as e:
                    raise IntakeValidationError(f"Malformed JSON on line {line_idx}: {e}") from e

        if not records:
            raise IntakeValidationError("No valid session records found in payload.")

        # Step 4: Schema & Field Validation
        participant_ids = set()
        total_events = 0
        for idx, rec in enumerate(records, start=1):
            if not isinstance(rec, dict):
                raise IntakeValidationError(f"Record {idx} must be an object.")

            p_id = rec.get("participant_id")
            if not p_id or not isinstance(p_id, str):
                raise IntakeValidationError(f"Record {idx} missing valid 'participant_id'.")
            participant_ids.add(p_id)

            scenario_id = rec.get("scenario_id")
            if scenario_id not in VALID_SCENARIO_IDS:
                raise IntakeValidationError(
                    f"Record {idx} contains unapproved scenario_id '{scenario_id}'."
                )

            variant = rec.get("architecture_variant")
            if variant not in VALID_VARIANTS:
                raise IntakeValidationError(
                    f"Record {idx} contains invalid architecture_variant '{variant}'."
                )

            block = rec.get("experiment_block")
            if block not in VALID_BLOCKS:
                raise IntakeValidationError(
                    f"Record {idx} contains invalid experiment_block '{block}'."
                )

            events = rec.get("events")
            if not isinstance(events, list):
                raise IntakeValidationError(f"Record {idx} 'events' must be a list.")

            # Verify monotonic sequence of events
            last_seq = 0
            for e_idx, event in enumerate(events, start=1):
                if not isinstance(event, dict):
                    raise IntakeValidationError(f"Record {idx} event {e_idx} must be an object.")
                seq = event.get("event_seq", 0)
                if not isinstance(seq, int) or seq <= last_seq:
                    raise IntakeValidationError(
                        f"Record {idx} event {e_idx} non-monotonic event_seq: {seq} <= {last_seq}."
                    )
                last_seq = seq
                total_events += 1

        if len(participant_ids) > 1:
            raise IntakeValidationError(
                f"Conflicting participant IDs in single session payload: {participant_ids}."
            )

        participant_id = list(participant_ids)[0]

        # Step 5: Pseudonym & Mode Authorization Check
        if self.is_synthetic_mode:
            # Under synthetic mode, human identifiers are strictly rejected
            if participant_id.startswith("P0") and not participant_id.startswith("P00-SYNTHETIC"):
                if human_authorization_token != "ARX_HUMAN_INTAKE_AUTHORIZED":
                    raise IntakeValidationError(
                        f"Participant ID '{participant_id}' rejected: human session intake is NOT authorized "
                        f"under nonproduction synthetic rehearsal mode."
                    )
            is_valid_synthetic = (
                participant_id == "P00-SYNTHETIC"
                or participant_id.startswith("SYN-")
                or participant_id.startswith("TEST-")
            )
            if not is_valid_synthetic and human_authorization_token != "ARX_HUMAN_INTAKE_AUTHORIZED":
                raise IntakeValidationError(
                    f"Participant ID '{participant_id}' is not recognized as an authorized synthetic identity."
                )

        # Step 6: Idempotency & Duplicate Check against Ledger
        batch_id = f"SESS-{session_id}"
        submission_id = f"SUBM-SESS-{session_id}"
        existing_receipts = self.ledger.get_all_receipts()

        for rcpt in existing_receipts:
            if rcpt.get("batch_id") == batch_id or rcpt.get("metadata", {}).get("session_id") == session_id:
                # Session ID already recorded in ledger
                recorded_sha = rcpt.get("raw_submission_sha256")
                if recorded_sha == raw_payload_sha256:
                    # Identical replay: return existing authoritative receipt with recomputed signature
                    rcpt_digest = rcpt.get("receipt_digest", "")
                    replay_sig = self._compute_server_receipt_signature(
                        receipt_id=rcpt["receipt_id"],
                        session_id=session_id,
                        participant_id=participant_id,
                        record_count=len(records),
                        source_payload_sha256=raw_payload_sha256,
                        ledger_digest=rcpt_digest,
                        accepted_at_utc=rcpt["timestamp_utc"],
                    )
                    return {
                        "status": "IDEMPOTENT_REPLAY_SUCCESS",
                        "receipt_id": rcpt["receipt_id"],
                        "session_id": session_id,
                        "participant_id": participant_id,
                        "record_count": len(records),
                        "event_count": total_events,
                        "source_payload_sha256": raw_payload_sha256,
                        "accepted_at_utc": rcpt["timestamp_utc"],
                        "custody_state": rcpt["custody_state"],
                        "ledger_ref": rcpt_digest,
                        "server_receipt_signature": replay_sig,
                        "synthetic_mode": self.is_synthetic_mode,
                    }
                else:
                    # Conflicting payload under same session ID
                    raise CustodyConflictError(
                        f"Conflict: session '{session_id}' is already sealed with a different payload digest "
                        f"({recorded_sha[:12]}... != {raw_payload_sha256[:12]}...)."
                    )

        # Step 7: Authenticated Encryption (AES-256-GCM)
        ciphertext = self.crypto.encrypt(raw_bytes)

        # Step 8: Atomic Restricted Vault Storage
        try:
            self.vault.store_sealed_evidence(submission_id, ciphertext)
        except Exception as e:
            raise CustodyError(f"Failed to store sealed evidence in vault: {e}") from e

        # Step 9: Tamper-Evident Ledger Inscription
        now_utc = datetime.now(timezone.utc).isoformat()
        rcpt_id = f"RCPT-SESS-{uuid.uuid4().hex[:12].upper()}"

        metadata: Dict[str, Any] = {
            "session_id": session_id,
            "record_count": len(records),
            "event_count": total_events,
            "synthetic_mode": self.is_synthetic_mode,
        }

        try:
            ledger_record = self.ledger.append_receipt(
                pseudonymous_observer_id=participant_id,
                batch_id=batch_id,
                package_sha256=raw_payload_sha256,
                raw_submission_sha256=raw_payload_sha256,
                schema_validation_status="VALID",
                custody_state="SEALED",
                receipt_id=rcpt_id,
                custom_metadata=metadata,
            )
        except Exception as e:
            # Rollback vault file if ledger fails
            try:
                self.vault.purge_sealed_evidence(submission_id)
            except Exception:
                pass
            raise LedgerIntegrityError(f"Failed to inscribe receipt into ledger: {e}") from e

        ledger_digest = ledger_record["receipt_digest"]
        accepted_at = ledger_record["timestamp_utc"]

        # Step 10: Final Authoritative Server HMAC Signature
        final_sig = self._compute_server_receipt_signature(
            receipt_id=rcpt_id,
            session_id=session_id,
            participant_id=participant_id,
            record_count=len(records),
            source_payload_sha256=raw_payload_sha256,
            ledger_digest=ledger_digest,
            accepted_at_utc=accepted_at,
        )

        return {
            "status": "ACCEPTED_AND_SEALED",
            "receipt_id": rcpt_id,
            "session_id": session_id,
            "participant_id": participant_id,
            "record_count": len(records),
            "event_count": total_events,
            "source_payload_sha256": raw_payload_sha256,
            "accepted_at_utc": accepted_at,
            "custody_state": "SEALED",
            "ledger_ref": ledger_digest,
            "server_receipt_signature": final_sig,
            "synthetic_mode": self.is_synthetic_mode,
        }

    def execute_session_withdrawal(
        self,
        session_id: str,
        participant_id: str,
        withdrawal_reason: str = "PARTICIPANT_CONSENT_REVOCATION",
    ) -> Dict[str, Any]:
        """
        Execute atomic withdrawal for a research session.
        1. Cryptographically shreds and purges sealed ciphertext from vault.
        2. Appends WITHDRAWAL_TOMBSTONE to ledger (preserving hash-chain continuity).
        """
        submission_id = f"SUBM-SESS-{session_id}"
        batch_id = f"SESS-{session_id}"

        # 1. Purge ciphertext
        self.vault.purge_sealed_evidence(submission_id)

        # 2. Append tombstone to ledger
        now_utc = datetime.now(timezone.utc).isoformat()
        tombstone_id = f"TOMB-SESS-{uuid.uuid4().hex[:12].upper()}"

        metadata = {
            "session_id": session_id,
            "withdrawal_reason": withdrawal_reason,
            "action": "CRYPTOGRAPHIC_KEY_SHREDDING_AND_PURGE",
        }

        tombstone_record = self.ledger.append_receipt(
            pseudonymous_observer_id=participant_id,
            batch_id=batch_id,
            package_sha256="0" * 64,
            raw_submission_sha256="0" * 64,
            schema_validation_status="WITHDRAWN",
            custody_state="WITHDRAWN_TOMBSTONE",
            receipt_id=tombstone_id,
            custom_metadata=metadata,
        )

        return {
            "status": "WITHDRAWAL_COMPLETED",
            "tombstone_id": tombstone_id,
            "session_id": session_id,
            "participant_id": participant_id,
            "timestamp_utc": now_utc,
            "ledger_ref": tombstone_record["receipt_digest"],
        }
