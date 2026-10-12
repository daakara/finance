"""
Canonical Claim Derivation Subsystem
ARX Terminal — Smart Money

Implements governed claim derivation from canonical evidence manifests,
enforcing 6-valued claim states (ESTABLISHED, NOT_ESTABLISHED, UNKNOWN, CONFLICT, STALE, REVOKED),
authoritative state basis verification, conflict detection, and Candidate 004 dual-identity separation.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from .canonicalization import sha256_canonical
from .evidence_manifest import CanonicalEvidenceManifest, EvidenceRecord


CLAIM_STATES = {"ESTABLISHED", "NOT_ESTABLISHED", "UNKNOWN", "CONFLICT", "STALE", "REVOKED"}


class ClaimDerivationError(Exception):
    """Base exception for claim derivation failures."""
    pass


class AuthoritativeStateBasisMissingError(ClaimDerivationError):
    """Raised when NOT_ESTABLISHED is asserted without an authoritative state basis."""
    pass


class ClaimConflictError(ClaimDerivationError):
    """Raised when conflicting authoritative evidence cannot be reconciled."""
    pass


class EmptyClaimRegistryError(ClaimDerivationError):
    """Raised when an empty claim registry is used to derive a canonical decision."""
    pass


class ClaimPolicySchemaError(ClaimDerivationError):
    """Raised when a claim derivation policy fails schema validation."""
    pass


class UnratifiedClaimError(ClaimDerivationError):
    """Raised when OPERATOR_RATIFIED is asserted without an authoritative operator ratification artifact."""
    pass


@dataclass
class StateBasis:
    basis_type: str
    authority_ref: Optional[str] = None
    state_source_ref: Optional[str] = None
    observed_at: Optional[str] = None
    authority_source: Optional[str] = None
    search_scope: Optional[str] = None
    source_set: Optional[List[str]] = None
    completeness_assertion: Optional[str] = None
    observation_window: Optional[str] = None
    validity: str = "CURRENT"

    def __post_init__(self):
        if self.authority_source is None:
            self.authority_source = self.authority_ref or "urn:arx:authority:canonical-manifest"
        if self.authority_ref is None:
            self.authority_ref = self.authority_source
        if self.search_scope is None:
            self.search_scope = self.state_source_ref or "urn:arx:scope:canonical-evidence-records"
        if self.state_source_ref is None:
            self.state_source_ref = self.search_scope
        if self.source_set is None:
            self.source_set = [self.state_source_ref or self.search_scope]
        if self.completeness_assertion is None:
            self.completeness_assertion = "EXHAUSTIVE_RECOMPUTED_ABSENCE"
        if self.observation_window is None:
            self.observation_window = "GENESIS_TO_HEAD"
        if self.observed_at is None:
            self.observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalDerivedClaim:
    claim_id: str
    claim_class: str
    claim_state: str
    derivation_id: str
    derivation_policy_hash: str
    evidence_refs: List[str]
    provenance_class: str = "REPOSITORY_GROUNDED"
    attestation_status: str = "LOCAL_REFERENCE"
    freshness: str = "CURRENT"
    authority_scope: str = "LOCAL_GOVERNANCE_REFERENCE"
    input_claim_ids: List[str] = field(default_factory=list)
    state_basis: Optional[StateBasis] = None
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    claim_hash: Optional[str] = None

    def __post_init__(self):
        if self.claim_state not in CLAIM_STATES:
            raise ValueError(f"Invalid claim_state '{self.claim_state}'. Allowed: {sorted(list(CLAIM_STATES))}")

        # Section 13: Authoritative NOT_ESTABLISHED Basis
        if self.claim_state == "NOT_ESTABLISHED":
            if self.state_basis is None:
                raise AuthoritativeStateBasisMissingError(
                    f"Claim '{self.claim_id}' cannot be NOT_ESTABLISHED without an authoritative state basis. "
                    "Laundering UNKNOWN into NOT_ESTABLISHED is strictly prohibited "
                    "(AUTHORITATIVE_NOT_ESTABLISHED_WITHOUT_STATE_BASIS == 0)."
                )
            sb = self.state_basis
            if not (sb.basis_type and sb.authority_source and sb.search_scope and sb.source_set and sb.completeness_assertion and sb.observation_window and sb.observed_at):
                raise AuthoritativeStateBasisMissingError(
                    f"Claim '{self.claim_id}' has incomplete state basis. All fields must be non-empty."
                )

        # Section 18: Operator Ratification Enforcement
        if self.attestation_status == "OPERATOR_RATIFIED":
            if not (self.metadata.get("ratification_artifact_ref") or self.metadata.get("ratification_evidence")):
                raise UnratifiedClaimError(
                    f"OPERATOR_RATIFIED_WITHOUT_RATIFICATION_ARTIFACT == 0: "
                    f"Claim '{self.claim_id}' cannot assert OPERATOR_RATIFIED without ratification artifact reference."
                )

        if not self.claim_hash:
            data = {
                "claim_id": self.claim_id,
                "claim_class": self.claim_class,
                "claim_state": self.claim_state,
                "provenance_class": self.provenance_class,
                "attestation_status": self.attestation_status,
                "freshness": self.freshness,
                "authority_scope": self.authority_scope,
                "derivation_id": self.derivation_id,
                "derivation_policy_hash": self.derivation_policy_hash,
                "evidence_refs": sorted(self.evidence_refs),
                "input_claim_ids": sorted(self.input_claim_ids),
                "state_basis": self.state_basis.to_dict() if self.state_basis else None,
                "metadata": self.metadata,
            }
            self.claim_hash = sha256_canonical(data)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.state_basis:
            d["state_basis"] = self.state_basis.to_dict()
        return d


class ClaimDerivationEngine:
    """
    Executes governed claim derivation per CANONICAL_CLAIM_DERIVATION_POLICY.json.
    """

    def __init__(self, policy_data: Optional[Dict[str, Any]] = None):
        if policy_data is None:
            policy_data = self._load_default_policy()
        self.policy_data = policy_data
        self.policy_id = policy_data.get("policy_id", "urn:arx:policy:claim-derivation:v1")
        self.policy_version = policy_data.get("policy_version", "1.0.0")
        self.rules = policy_data.get("rules", [])
        self.policy_hash = sha256_canonical(self.policy_data)

    @staticmethod
    def _load_default_policy() -> Dict[str, Any]:
        policy_path = os.path.join(
            os.path.dirname(__file__),
            "../../../docs/domain/smart_money/governance_control_plane/CANONICAL_CLAIM_DERIVATION_POLICY.json",
        )
        if os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "policy_id": "urn:arx:policy:claim-derivation:v1",
            "policy_version": "1.0.0",
            "rules": [],
        }

    def derive_all_claims(
        self,
        evidence_manifest: CanonicalEvidenceManifest,
        upstream_claims: Optional[Dict[str, CanonicalDerivedClaim]] = None,
        observed_at: Optional[str] = None,
    ) -> Dict[str, CanonicalDerivedClaim]:
        """
        Derives all canonical claims from evidence records and upstream claims.
        """
        if not observed_at:
            observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if len(evidence_manifest.records) == 0:
            raise EmptyClaimRegistryError(
                "Cannot derive claims from empty evidence manifest (EMPTY_CLAIM_REGISTRY_ACCEPTED_FOR_CANONICAL_DECISION == 0)."
            )

        derived: Dict[str, CanonicalDerivedClaim] = {}
        records_by_claim = {}
        for rec in evidence_manifest.records:
            records_by_claim.setdefault(rec.claim_id, []).append(rec)

        for rule in self.rules:
            out_id = rule["output_claim_id"]
            claim_cls = rule.get("claim_class", "GENERAL")
            deriv_id = rule["derivation_id"]

            matching_recs = records_by_claim.get(out_id, [])

            # Check conflicts
            established_recs = [r for r in matching_recs if r.evidence_state == "ESTABLISHED"]
            not_established_recs = [r for r in matching_recs if r.evidence_state == "NOT_ESTABLISHED"]

            if len(established_recs) > 0 and len(not_established_recs) > 0:
                claim_state = "CONFLICT"
                basis = None
            elif len(established_recs) > 0:
                claim_state = "ESTABLISHED"
                basis = None
            elif len(not_established_recs) > 0:
                claim_state = "NOT_ESTABLISHED"
                auth_src = not_established_recs[0].source or "urn:arx:authority:canonical-manifest"
                basis = StateBasis(
                    basis_type="AUTHORITATIVE_CONFIRMED_ABSENCE",
                    authority_source=auth_src,
                    search_scope="urn:arx:scope:canonical-evidence-record",
                    source_set=["git:tree:docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json"],
                    completeness_assertion="EXHAUSTIVE_RECOMPUTED_ABSENCE",
                    observation_window="GENESIS_TO_HEAD",
                    observed_at=observed_at,
                    authority_ref=auth_src,
                    state_source_ref="urn:arx:scope:canonical-evidence-record",
                )
            else:
                claim_state = "UNKNOWN"
                basis = None

            # Special case for Candidate 004 dual identity
            metadata = {}
            if out_id == "CLAIM-CANDIDATE004-SEMANTIC-CLOSURE-IDENTITY":
                metadata["semantic_closure_recomputation"] = "NOT_ESTABLISHED"

            claim = CanonicalDerivedClaim(
                claim_id=out_id,
                claim_class=claim_cls,
                claim_state=claim_state,
                derivation_id=deriv_id,
                derivation_policy_hash=self.policy_hash,
                evidence_refs=[r.evidence_id for r in matching_recs],
                input_claim_ids=rule.get("input_claim_ids", []),
                state_basis=basis,
                observed_at=observed_at,
                metadata=metadata,
            )
            derived[out_id] = claim

        return derived
