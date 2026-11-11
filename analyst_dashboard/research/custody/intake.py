"""
ARX Terminal — Controlled Synthetic Evidence Intake Pipeline (Gate E)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Pipeline Execution Contract (10 Steps):
1. Boundary & Size Validation (<= 2MB payload ceiling)
2. Pre-parse Wire Hashing (SHA-256 computed on raw bytes before JSON decoding)
3. Safe JSON Decoding (No raw payloads in exception traces)
4. Draft-07 JSON Schema Validation (tests against OBSERVER_RATING_SCHEMA.json)
5. Pseudonym & Mode Authorization (Synthetic mode strictly rejects human tokens)
6. Trial Completeness & Taxonomy Validation (All 30 trials: ALPHA, BETA, GAMMA)
7. Replay & Duplicate Protection (Rejects duplicate hashes and observer resubmissions)
8. Authenticated Encryption (AES-256-GCM via CustodyCrypto)
9. Restricted Vault Storage (Atomic staging write to off-Git vault)
10. Tamper-Evident Ledger Inscription & Blinding State Resolution
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import jsonschema

from analyst_dashboard.research.custody.exceptions import (
    IntakeValidationError,
    CustodyError,
)
from analyst_dashboard.research.custody.crypto import CustodyCrypto, compute_sha256
from analyst_dashboard.research.custody.vault import LocalRestrictedVault
from analyst_dashboard.research.custody.ledger import TamperEvidentLedger

MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 Megabytes

EXPECTED_30_TRIALS = {f"ALPHA-{i:02d}" for i in range(1, 11)} | {
    f"BETA-{i:02d}" for i in range(1, 11)
} | {f"GAMMA-{i:02d}" for i in range(1, 11)}

VALID_CM_CODES = {f"CM-{i:02d}" for i in range(1, 10)}
VALID_SM_CODES = {f"SM-{i:02d}" for i in range(1, 7)}
ALL_VALID_CODES = VALID_CM_CODES | VALID_SM_CODES

SCHEMA_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "qualification"
    / "observer_package"
    / "OBSERVER_RATING_SCHEMA.json"
)


class IntakePipeline:
    """
    Controlled evidence intake pipeline enforcing fail-closed boundary validation,
    cryptographic sealing, mutual blinding lockstep, and tamper-evident logging.
    """

    def __init__(
        self,
        vault: LocalRestrictedVault,
        crypto: CustodyCrypto,
        ledger: TamperEvidentLedger,
        schema_path: Optional[Path | str] = None,
        is_synthetic_mode: bool = True,
    ):
        self.vault = vault
        self.crypto = crypto
        self.ledger = ledger
        self.is_synthetic_mode = is_synthetic_mode

        # Load schema
        schema_file = Path(schema_path or SCHEMA_DEFAULT_PATH).resolve()
        if not schema_file.exists():
            raise CustodyError(f"Rating schema not found at '{schema_file}'.")
        try:
            self.schema = json.loads(schema_file.read_text(encoding="utf-8"))
            jsonschema.Draft7Validator.check_schema(self.schema)
            self.validator = jsonschema.Draft7Validator(self.schema)
        except Exception as e:
            raise CustodyError(f"Failed to initialize JSON schema validator: {e}") from e

    def ingest_submission(
        self,
        raw_bytes: bytes,
        registered_batch_id: str,
        expected_package_sha: str,
        human_authorization_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute 10-step atomic intake pipeline on raw submission bytes.
        Returns tamper-evident custody receipt.
        """
        # Step 1: Boundary & Size Validation
        if not isinstance(raw_bytes, bytes):
            raise IntakeValidationError("Submission payload must be raw bytes.")
        if len(raw_bytes) == 0:
            raise IntakeValidationError("Submission payload is empty.")
        if len(raw_bytes) > MAX_PAYLOAD_BYTES:
            raise IntakeValidationError(
                f"Submission exceeds maximum allowed size ({len(raw_bytes)} > {MAX_PAYLOAD_BYTES} bytes)."
            )

        if not registered_batch_id or not expected_package_sha:
            raise IntakeValidationError(
                "Batch identifier and package SHA-256 reference are mandatory."
            )

        # Step 2: Pre-parse SHA-256 Digest
        raw_submission_sha = compute_sha256(raw_bytes)

        # Step 3: Safe JSON Decoding
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise IntakeValidationError(
                "Malformed submission payload: invalid UTF-8 JSON syntax."
            ) from e

        if not isinstance(data, dict):
            raise IntakeValidationError("Root submission payload must be a JSON object.")

        # Step 4: Schema Validation
        validation_errors = list(self.validator.iter_errors(data))
        if validation_errors:
            # Capture error message without leaking sensitive raw values
            err_msg = validation_errors[0].message
            err_path = "/".join(str(p) for p in validation_errors[0].path)
            raise IntakeValidationError(
                f"Schema validation failed at path '{err_path}': {err_msg}"
            )

        observer_id = data.get("observer_id")
        transcript_id = data.get("transcript_id")
        ratings_list = data.get("ratings", [])

        # Step 5: Pseudonym & Mode Authorization Check
        if self.is_synthetic_mode:
            # In synthetic mode, real human observer identifiers are strictly forbidden
            if observer_id.startswith("OBS-HUMAN-"):
                if human_authorization_token != "ARX_HUMAN_INTAKE_AUTHORIZED":
                    raise IntakeValidationError(
                        f"Observer ID '{observer_id}' rejected: human intake not authorized "
                        f"under nonproduction synthetic custody mode."
                    )
            # Require synthetic provenance indication
            is_explicit_synthetic = (
                observer_id.startswith("OBS-SYNTHETIC-")
                or observer_id.startswith("OBS-TEST-")
                or data.get("synthetic_provenance") is True
            )
            if not is_explicit_synthetic:
                raise IntakeValidationError(
                    f"Observer ID '{observer_id}' rejected: synthetic mode requires synthetic pseudonym or provenance flag."
                )

        # Step 6: Trial Completeness & Taxonomy Code Verification
        seen_trials: Set[str] = set()
        for idx, rating in enumerate(ratings_list):
            tid = rating.get("trial_id")
            if not tid or tid not in EXPECTED_30_TRIALS:
                raise IntakeValidationError(
                    f"Trial at index {idx} has invalid or unapproved trial_id '{tid}'."
                )
            if tid in seen_trials:
                raise IntakeValidationError(f"Duplicate trial_id '{tid}' in ratings list.")
            seen_trials.add(tid)

            # Taxonomy validation
            cm_codes = rating.get("assigned_cm_codes", [])
            for code in cm_codes:
                if code not in VALID_CM_CODES:
                    raise IntakeValidationError(
                        f"Trial '{tid}' contains invalid canonical code '{code}'."
                    )

            sm_codes = rating.get("assigned_sm_codes", [])
            for code in sm_codes:
                if code not in VALID_SM_CODES:
                    raise IntakeValidationError(
                        f"Trial '{tid}' contains invalid supplemental code '{code}'."
                    )

        if seen_trials != EXPECTED_30_TRIALS:
            missing_trials = EXPECTED_30_TRIALS - seen_trials
            raise IntakeValidationError(
                f"Incomplete submission: missing {len(missing_trials)} trials (e.g. {sorted(missing_trials)[:3]})."
            )

        # Step 7: Replay & Duplicate Submission Protection
        existing_receipts = self.ledger.get_all_receipts()
        for rcpt in existing_receipts:
            if rcpt.get("raw_submission_sha256") == raw_submission_sha:
                raise IntakeValidationError(
                    "Submission replay rejected: exact byte digest already exists in custody ledger."
                )

        observer_receipts = [
            r
            for r in existing_receipts
            if r.get("batch_id") == registered_batch_id
            and r.get("pseudonymous_observer_id") == observer_id
        ]
        if observer_receipts and observer_receipts[-1].get("custody_state") != "WITHDRAWAL_TOMBSTONE":
            raise IntakeValidationError(
                f"Duplicate submission rejected: observer '{observer_id}' has an active sealed submission for batch '{registered_batch_id}'."
            )

        # Step 8: Authenticated Encryption
        submission_id = f"SUBM-{registered_batch_id}-{observer_id}"
        # Associate metadata with ciphertext authentication
        aad = f"{submission_id}|{raw_submission_sha}".encode("utf-8")
        ciphertext = self.crypto.encrypt(raw_bytes, associated_data=aad)

        # Step 9: Restricted Off-Git Vault Storage
        self.vault.store_sealed_evidence(submission_id, ciphertext)

        # Step 10: Ledger Inscription & Blinding State Resolution
        # Check if paired observer in batch has an active sealed submission
        other_observers = {
            r.get("pseudonymous_observer_id")
            for r in existing_receipts
            if r.get("batch_id") == registered_batch_id
            and r.get("pseudonymous_observer_id") != observer_id
        }
        active_peers = []
        for peer_id in other_observers:
            peer_receipts = [
                r
                for r in existing_receipts
                if r.get("batch_id") == registered_batch_id
                and r.get("pseudonymous_observer_id") == peer_id
            ]
            if peer_receipts and peer_receipts[-1].get("custody_state") != "WITHDRAWAL_TOMBSTONE":
                active_peers.append(peer_id)

        if len(active_peers) == 0:
            custody_state = "PENDING_INDEPENDENT_PAIR"
        else:
            custody_state = "PAIR_SEALED_ELIGIBLE_FOR_SCORING"

        receipt = self.ledger.append_receipt(
            pseudonymous_observer_id=observer_id,
            batch_id=registered_batch_id,
            package_sha256=expected_package_sha,
            raw_submission_sha256=raw_submission_sha,
            schema_validation_status="VALIDATED",
            custody_state=custody_state,
            receipt_id=f"RCPT-{submission_id}",
            custom_metadata={
                "submission_id": submission_id,
                "transcript_id": transcript_id,
                "trial_count": len(seen_trials),
                "synthetic_mode": self.is_synthetic_mode,
            },
        )

        return receipt
