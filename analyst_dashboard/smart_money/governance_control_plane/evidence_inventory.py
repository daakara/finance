"""
Evidence Inventory Subsystem
ARX Terminal — Smart Money

Maintains the authoritative inventory of governance and infrastructure evidence.
Enforces evidence completeness and zero-finding audit attestations.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from .canonicalization import sha256_canonical, domain_normalize


@dataclass
class ZeroFindingAttestation:
    audit_id: str
    evidence_hash: str
    reviewer: str
    policy_version: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceInventory:
    """
    Tracks and validates governance evidence completeness.
    Guarantees that selectively incomplete evidence sets cannot drive READY.
    """

    def __init__(
        self,
        inventory_id: str = "urn:arx:inventory:evidence:control-plane:v1",
        epoch: str = "epoch-gov-001",
        required_source_classes: Optional[List[str]] = None,
    ):
        self.inventory_id = inventory_id
        self.epoch = epoch
        self.required_source_classes = required_source_classes or [
            "ARCHITECTURE_SPEC",
            "SECURITY_ASSESSMENT",
            "CODE_VERIFICATION",
            "CRYPTOGRAPHIC_PROOF",
        ]
        self.expected_artifacts: Dict[str, Dict[str, Any]] = {}
        self.observed_artifacts: Dict[str, Dict[str, Any]] = {}
        self.missing_artifacts: List[str] = []
        self.revoked_artifacts: List[str] = []
        self.stale_artifacts: List[str] = []
        self.zero_finding_attestations: List[ZeroFindingAttestation] = []
        self.mandatory_audits: List[str] = ["AUDIT-SEC-001", "AUDIT-ARCH-001"]

    def add_expected_artifact(self, artifact_id: str, source_class: str, expected_hash: str) -> None:
        self.expected_artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "source_class": source_class,
            "expected_hash": expected_hash,
        }

    def add_observed_artifact(
        self, artifact_id: str, source_class: str, observed_hash: str, verified_at: str
    ) -> None:
        self.observed_artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "source_class": source_class,
            "observed_hash": observed_hash,
            "verified_at": verified_at,
        }

    def add_zero_finding_attestation(self, attestation: ZeroFindingAttestation) -> None:
        self.zero_finding_attestations.append(attestation)

    def reconcile(self, active_finding_audit_ids: Optional[List[str]] = None) -> None:
        """
        Reconciles expected vs observed artifacts and audit attestations.
        """
        self.missing_artifacts = []
        for aid, exp in self.expected_artifacts.items():
            if aid not in self.observed_artifacts:
                self.missing_artifacts.append(aid)
            elif self.observed_artifacts[aid]["observed_hash"] != exp["expected_hash"]:
                self.stale_artifacts.append(aid)

        # Audit surface completeness:
        # Every mandatory audit must emit either findings or a zero-finding attestation.
        active_audit_set = set(active_finding_audit_ids or [])
        attested_audit_set = {a.audit_id for a in self.zero_finding_attestations}
        for ma in self.mandatory_audits:
            if ma not in active_audit_set and ma not in attested_audit_set:
                self.missing_artifacts.append(f"MANDATORY_AUDIT_ATTESTATION:{ma}")

    def is_complete(self) -> bool:
        self.reconcile()
        return (
            len(self.missing_artifacts) == 0
            and len(self.revoked_artifacts) == 0
            and len(self.stale_artifacts) == 0
            and len(self.expected_artifacts) > 0
            and len(self.observed_artifacts) >= len(self.expected_artifacts)
        )

    def get_missing_count(self) -> int:
        return len(self.missing_artifacts)

    def get_unreconciled_count(self) -> int:
        return len(self.missing_artifacts) + len(self.stale_artifacts) + len(self.revoked_artifacts)

    def to_dict(self) -> Dict[str, Any]:
        self.reconcile()
        return {
            "inventory_id": self.inventory_id,
            "governance_epoch": self.epoch,
            "required_source_classes": sorted(self.required_source_classes),
            "expected_artifacts": sorted(
                list(self.expected_artifacts.values()), key=lambda x: x["artifact_id"]
            ),
            "observed_artifacts": sorted(
                list(self.observed_artifacts.values()), key=lambda x: x["artifact_id"]
            ),
            "missing_artifacts": sorted(self.missing_artifacts),
            "revoked_artifacts": sorted(self.revoked_artifacts),
            "stale_artifacts": sorted(self.stale_artifacts),
            "zero_finding_attestations": [
                a.to_dict()
                for a in sorted(self.zero_finding_attestations, key=lambda x: x.audit_id)
            ],
            "evidence_set_complete": self.is_complete(),
            "missing_required_evidence_refs": self.get_missing_count(),
            "unreconciled_required_evidence": self.get_unreconciled_count(),
        }

    def compute_inventory_hash(self) -> str:
        d = self.to_dict()
        return sha256_canonical(d)
