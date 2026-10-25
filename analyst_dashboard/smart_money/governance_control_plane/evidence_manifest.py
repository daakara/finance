"""
Canonical Evidence Manifest & Claim Registry Subsystem
ARX Terminal — Smart Money

Implements strict canonical evidence vetting, explicit NOT_ESTABLISHED absence modeling,
CanonicalEvidenceManifest generation, and ClaimRegistry semantic bindings.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
import re

from .canonicalization import sha256_canonical, domain_normalize


class EvidenceIntegrityError(Exception):
    """Raised when evidence fails canonical validation."""
    pass


class SyntheticEvidenceContaminationError(EvidenceIntegrityError):
    """Raised when synthetic or test-double evidence attempts canonical admission."""
    pass


class PlaceholderHashError(EvidenceIntegrityError):
    """Raised when a placeholder or repeated-character digest is detected."""
    pass


FORBIDDEN_PROVENANCE_CLASSES = {"SYNTHETIC_FIXTURE", "TEST_DOUBLE"}
REPEATED_CHAR_HASH_PATTERN = re.compile(r"^([a-f0-9])\1{63}$")


@dataclass
class EvidenceRecord:
    evidence_id: str
    claim_id: str
    evidence_state: str  # ESTABLISHED, NOT_ESTABLISHED, STALE, REVOKED
    provenance_class: str  # CANONICAL_LEDGER, EXTERNAL_OPERATIONAL, REPOSITORY_BOUND, OPERATOR_RATIFIED, LOCAL_REFERENCE, DESIGN_CONTRACT, NONE
    authority_class: str  # AUTHORITATIVE, NON_AUTHORITATIVE, NOT_APPLICABLE
    source: Optional[str]
    content_hash: Optional[str]
    validity: str  # CURRENT, STALE, REVOKED
    observed_at: str
    canonical_hash: Optional[str] = None

    def __post_init__(self):
        if not self.canonical_hash:
            data = {
                "evidence_id": self.evidence_id,
                "claim_id": self.claim_id,
                "evidence_state": self.evidence_state,
                "provenance_class": self.provenance_class,
                "authority_class": self.authority_class,
                "source": self.source,
                "content_hash": self.content_hash,
                "validity": self.validity,
                "observed_at": self.observed_at,
            }
            self.canonical_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_canonical_evidence_record(rec: EvidenceRecord) -> None:
    """
    Validates that an EvidenceRecord is eligible for inclusion in the CanonicalEvidenceManifest:
    1. Rejects SYNTHETIC_FIXTURE and TEST_DOUBLE
    2. Rejects placeholder / repeated-character hashes (e.g. '0'*64 or 'a'*64)
    3. Rejects test personas in source
    4. Validates NOT_ESTABLISHED records have null source/content_hash and provenance NONE
    """
    if rec.provenance_class in FORBIDDEN_PROVENANCE_CLASSES:
        raise SyntheticEvidenceContaminationError(
            f"Record {rec.evidence_id} has forbidden provenance_class '{rec.provenance_class}'."
        )

    if rec.source and any(p in rec.source.lower() for p in ["test_persona", "mock_operator", "synthetic"]):
        raise SyntheticEvidenceContaminationError(
            f"Record {rec.evidence_id} source contains test persona: '{rec.source}'."
        )

    if rec.content_hash:
        if REPEATED_CHAR_HASH_PATTERN.match(rec.content_hash.lower()):
            raise PlaceholderHashError(
                f"Record {rec.evidence_id} has forbidden repeated-character placeholder hash '{rec.content_hash}'."
            )

    if rec.evidence_state == "NOT_ESTABLISHED":
        if rec.content_hash is not None:
            raise EvidenceIntegrityError(
                f"NOT_ESTABLISHED record {rec.evidence_id} must have content_hash=None, got '{rec.content_hash}'."
            )
        if rec.provenance_class not in {"NONE", "NOT_APPLICABLE"}:
            raise EvidenceIntegrityError(
                f"NOT_ESTABLISHED record {rec.evidence_id} must have provenance_class 'NONE', got '{rec.provenance_class}'."
            )


@dataclass
class CanonicalEvidenceManifest:
    manifest_id: str
    manifest_version: str
    source_freeze_sha: str
    records: List[EvidenceRecord]
    summary: Dict[str, int]
    manifest_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "source_freeze_sha": self.source_freeze_sha,
            "records": [r.to_dict() for r in self.records],
            "summary": self.summary,
            "manifest_hash": self.manifest_hash,
        }


def build_canonical_evidence_manifest(
    records: List[EvidenceRecord],
    source_freeze_sha: str,
    manifest_id: str = "urn:arx:manifest:evidence:canonical:v1",
    manifest_version: str = "1.0.0",
) -> CanonicalEvidenceManifest:
    """
    Constructs a verified CanonicalEvidenceManifest.
    Fails closed if any record contains synthetic, test-double, or placeholder data.
    """
    validated_records: List[EvidenceRecord] = []
    summary = {
        "authoritative_present": 0,
        "supporting_present": 0,
        "not_established": 0,
        "stale": 0,
        "revoked": 0,
        "synthetic": 0,
        "test_double": 0,
    }

    for rec in records:
        validate_canonical_evidence_record(rec)
        validated_records.append(rec)
        if rec.evidence_state == "NOT_ESTABLISHED":
            summary["not_established"] += 1
        elif rec.validity == "STALE":
            summary["stale"] += 1
        elif rec.validity == "REVOKED":
            summary["revoked"] += 1
        elif rec.authority_class == "AUTHORITATIVE":
            summary["authoritative_present"] += 1
        else:
            summary["supporting_present"] += 1

    sorted_records = sorted(validated_records, key=lambda r: r.evidence_id)
    manifest_body = {
        "manifest_id": manifest_id,
        "manifest_version": manifest_version,
        "source_freeze_sha": source_freeze_sha,
        "records": [r.to_dict() for r in sorted_records],
        "summary": summary,
    }
    manifest_hash = sha256_canonical(manifest_body)

    return CanonicalEvidenceManifest(
        manifest_id=manifest_id,
        manifest_version=manifest_version,
        source_freeze_sha=source_freeze_sha,
        records=sorted_records,
        summary=summary,
        manifest_hash=manifest_hash,
    )


@dataclass
class CanonicalClaim:
    claim_id: str
    claim_type: str
    semantic_subject: str
    current_state: str  # ESTABLISHED, NOT_ESTABLISHED, STALE, REVOKED, CONFLICT
    source_evidence_ids: List[str]
    derivation_function: str
    derivation_version: str
    claim_hash: Optional[str] = None

    def __post_init__(self):
        if not self.claim_hash:
            data = {
                "claim_id": self.claim_id,
                "claim_type": self.claim_type,
                "semantic_subject": self.semantic_subject,
                "current_state": self.current_state,
                "source_evidence_ids": sorted(self.source_evidence_ids),
                "derivation_function": self.derivation_function,
                "derivation_version": self.derivation_version,
            }
            self.claim_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClaimRegistry:
    """
    Authoritative projection of Canonical Claims and Claim-Fact Semantic Bindings.
    """

    def __init__(
        self,
        registry_id: str = "urn:arx:registry:claims:v1",
        claims: Optional[List[CanonicalClaim]] = None,
        claim_fact_bindings: Optional[Dict[str, List[str]]] = None,
    ):
        self.registry_id = registry_id
        self._claims: Dict[str, CanonicalClaim] = {}
        if claims:
            for c in claims:
                self._claims[c.claim_id] = c

        # Claim-Fact Semantic Bindings: maps claim_id to all related fact aliases
        self.claim_fact_bindings: Dict[str, List[str]] = claim_fact_bindings or {
            "claim_governance_root": [
                "claim_governance_root",
                "root_quorum_established",
                "root_quorum_status",
                "governance_root_ready",
            ],
            "claim_external_ledger": [
                "claim_external_ledger",
                "external_ledger_established",
                "external_ledger_connected",
                "external_ledger_status",
            ],
            "claim_external_witness": [
                "claim_external_witness",
                "external_witness_established",
                "external_witness_connected",
                "external_witness_status",
            ],
            "claim_candidate_freeze": [
                "claim_candidate_freeze",
                "candidate_freeze_established",
                "candidate004_freeze_verified",
            ],
            "claim_denominator_alignment": [
                "claim_denominator_alignment",
                "denominator_predicate_aligned",
                "natural_denominator_admissible",
            ],
        }

    def add_claim(self, claim: CanonicalClaim) -> None:
        self._claims[claim.claim_id] = claim

    def get_claim(self, claim_id: str) -> Optional[CanonicalClaim]:
        return self._claims.get(claim_id)

    @property
    def claims(self) -> List[CanonicalClaim]:
        return sorted(self._claims.values(), key=lambda c: c.claim_id)

    def compute_registry_hash(self) -> str:
        data = {
            "registry_id": self.registry_id,
            "claims": [c.to_dict() for c in self.claims],
            "claim_fact_bindings": domain_normalize(self.claim_fact_bindings),
        }
        return sha256_canonical(data)

    def get_facts_for_claim(self, claim_id: str) -> Set[str]:
        return set(self.claim_fact_bindings.get(claim_id, [claim_id]))

    def get_claim_for_fact(self, fact_id: str) -> Optional[str]:
        for cid, facts in self.claim_fact_bindings.items():
            if fact_id in facts or fact_id == cid:
                return cid
        return None
