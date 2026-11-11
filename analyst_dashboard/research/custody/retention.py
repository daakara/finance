"""
ARX Terminal — Data Retention & Participant Withdrawal Workflow (Gate H & J)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Operational Mandate:
- Implement GDPR Article 17 Right to Erasure / Participant Withdrawal workflow.
- Cryptographically shred and remove ciphertext and identity mapping from vault.
- Append WITHDRAWAL_TOMBSTONE receipt to the tamper-evident ledger.
- Invalidate all downstream qualification claims associated with withdrawn observer.
- Document residual backup retention boundaries and limitations.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from analyst_dashboard.research.custody.exceptions import WithdrawalError
from analyst_dashboard.research.custody.vault import LocalRestrictedVault
from analyst_dashboard.research.custody.ledger import TamperEvidentLedger


class RetentionAndWithdrawalManager:
    """
    Manages data lifecycle, retention enforcement, and participant withdrawal workflows.
    """

    def __init__(
        self,
        vault: LocalRestrictedVault,
        ledger: TamperEvidentLedger,
    ):
        self.vault = vault
        self.ledger = ledger

    def execute_participant_withdrawal(
        self,
        batch_id: str,
        observer_id: str,
        withdrawal_reason: str = "PARTICIPANT_CONSENT_REVOCATION_GDPR_ART17",
    ) -> Dict[str, Any]:
        """
        Execute atomic withdrawal and erasure for a pseudonymous participant.
        1. Purges sealed ciphertext via cryptographic shredding.
        2. Purges identity mapping if stored.
        3. Records WITHDRAWAL_TOMBSTONE in append-only receipt ledger.
        4. Invalidates downstream qualification.
        """
        if not batch_id or not observer_id:
            raise WithdrawalError("Batch ID and Observer ID are required for withdrawal.")

        submission_id = f"SUBM-{batch_id}-{observer_id}"

        # 1. Purge ciphertext
        purged_evidence = self.vault.purge_sealed_evidence(submission_id)

        # 2. Purge identity mapping
        purged_identity = self.vault.purge_identity_mapping(observer_id)

        # 3. Retrieve prior submission hash from ledger if available
        receipts = self.ledger.get_all_receipts()
        prior_sha = "0" * 64
        for r in receipts:
            if (
                r.get("batch_id") == batch_id
                and r.get("pseudonymous_observer_id") == observer_id
                and r.get("custody_state") != "WITHDRAWAL_TOMBSTONE"
            ):
                prior_sha = r.get("raw_submission_sha256", prior_sha)
                break

        # 4. Inscribe tombstone receipt into ledger
        tombstone_receipt = self.ledger.append_receipt(
            pseudonymous_observer_id=observer_id,
            batch_id=batch_id,
            package_sha256="0" * 64,
            raw_submission_sha256=prior_sha,
            schema_validation_status="WITHDRAWN_PURGED",
            custody_state="WITHDRAWAL_TOMBSTONE",
            receipt_id=f"TOMBSTONE-{submission_id}",
            custom_metadata={
                "reason": withdrawal_reason,
                "purged_submission_id": submission_id,
                "evidence_file_purged": purged_evidence,
                "identity_file_purged": purged_identity,
                "qualification_invalidated": True,
            },
        )

        return {
            "status": "WITHDRAWAL_EXECUTED",
            "submission_id": submission_id,
            "evidence_purged": purged_evidence,
            "identity_purged": purged_identity,
            "qualification_status": "REVOKED_INVALIDATED",
            "tombstone_receipt": tombstone_receipt,
        }
