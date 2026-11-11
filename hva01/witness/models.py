"""
HVA-01 Canonical Evidence and Witness Data Models.
Implements RFC 8785 canonical serialization for Genesis commitments, attestation payloads, and witness envelopes.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..crypto.canonical import CanonicalJCS, canonical_sha256, canonicalize_json


@dataclass(frozen=True)
class GenesisCommitment:
    """
    Canonical pre-intake commitment sealing the research universe, denominators, and allocation seeds.
    Binds the 59 evaluable charts and 60-case manifest prior to reviewer intake.
    """
    genesis_id: str
    protocol_version: str = "HVA-00-V2.1"
    owner_pubkey_fingerprint: str = ""
    created_at_utc: str = ""
    total_evaluable_charts: int = 59
    manifest_total_cases: int = 60
    quarantined_cases: List[str] = field(default_factory=lambda: ["HVA00-REJ-04"])
    strata_breakdown: Dict[str, int] = field(default_factory=lambda: {
        "primary_specificity": 29,
        "macro_specificity": 19,
        "positive_sensitivity": 15,
        "developing_controls": 10,
        "sufficiency_controls": 5,
    })
    candidate_scenario_ids: List[str] = field(default_factory=list)
    reviewer_seeds: Dict[str, int] = field(default_factory=dict)
    restricted_mapping_digest: str = ""
    genesis_hash: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            object.__setattr__(self, "created_at_utc", datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        """Compute RFC 8785 SHA-256 digest of genesis fields (excluding genesis_hash)."""
        data = {
            "genesis_id": self.genesis_id,
            "protocol_version": self.protocol_version,
            "owner_pubkey_fingerprint": self.owner_pubkey_fingerprint,
            "created_at_utc": self.created_at_utc,
            "total_evaluable_charts": self.total_evaluable_charts,
            "manifest_total_cases": self.manifest_total_cases,
            "quarantined_cases": sorted(self.quarantined_cases),
            "strata_breakdown": self.strata_breakdown,
            "candidate_scenario_ids": sorted(self.candidate_scenario_ids),
            "reviewer_seeds": self.reviewer_seeds,
            "restricted_mapping_digest": self.restricted_mapping_digest,
        }
        return canonical_sha256(data)

    @classmethod
    def create(
        cls,
        genesis_id: str,
        owner_pubkey_fingerprint: str,
        candidate_scenario_ids: List[str],
        reviewer_seeds: Dict[str, int],
        restricted_mapping_digest: str,
        created_at_utc: Optional[str] = None,
    ) -> GenesisCommitment:
        now_utc = created_at_utc or datetime.now(timezone.utc).isoformat()
        temp = cls(
            genesis_id=genesis_id,
            owner_pubkey_fingerprint=owner_pubkey_fingerprint,
            created_at_utc=now_utc,
            candidate_scenario_ids=candidate_scenario_ids,
            reviewer_seeds=reviewer_seeds,
            restricted_mapping_digest=restricted_mapping_digest,
            genesis_hash="",
        )
        calculated_hash = temp.compute_hash()
        return cls(
            genesis_id=genesis_id,
            owner_pubkey_fingerprint=owner_pubkey_fingerprint,
            created_at_utc=now_utc,
            candidate_scenario_ids=candidate_scenario_ids,
            reviewer_seeds=reviewer_seeds,
            restricted_mapping_digest=restricted_mapping_digest,
            genesis_hash=calculated_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttestationPayload:
    """
    The exact payload signed by the independent witness.
    Binds the session, Merkle root, leaf count, and genesis commitment.
    """
    session_id: str
    protocol_version: str
    genesis_hash: str
    merkle_root: str  # format "sha256:<hex>"
    leaf_count: int
    leaf_hashes: List[str]

    def to_canonical_bytes(self) -> bytes:
        """Encode to deterministic RFC 8785 UTF-8 bytes for cryptographic signing."""
        data = {
            "session_id": self.session_id,
            "protocol_version": self.protocol_version,
            "genesis_hash": self.genesis_hash,
            "merkle_root": self.merkle_root,
            "leaf_count": self.leaf_count,
            "leaf_hashes": self.leaf_hashes,
        }
        return canonicalize_json(data)


@dataclass(frozen=True)
class WitnessAttestation:
    """
    The detached witness signature and dual-bound attestation digest.
    """
    witness_id: str
    witness_pubkey_fingerprint: str
    signature_algorithm: str
    signature_b64: str
    attestation_digest: str
    timestamp_token: Optional[Dict[str, Any]] = None
    attested_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitnessEnvelope:
    """
    The consolidated root evidence envelope.
    """
    envelope_version: str
    attestation_payload: AttestationPayload
    witness_attestation: WitnessAttestation
    status: str = "COMMITTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_version": self.envelope_version,
            "attestation_payload": asdict(self.attestation_payload),
            "witness_attestation": self.witness_attestation.to_dict(),
            "status": self.status,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonicalize_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WitnessEnvelope:
        payload_data = data["attestation_payload"]
        attestation_data = data["witness_attestation"]
        
        payload = AttestationPayload(
            session_id=payload_data["session_id"],
            protocol_version=payload_data["protocol_version"],
            genesis_hash=payload_data["genesis_hash"],
            merkle_root=payload_data["merkle_root"],
            leaf_count=payload_data["leaf_count"],
            leaf_hashes=list(payload_data["leaf_hashes"]),
        )
        
        attestation = WitnessAttestation(
            witness_id=attestation_data["witness_id"],
            witness_pubkey_fingerprint=attestation_data["witness_pubkey_fingerprint"],
            signature_algorithm=attestation_data["signature_algorithm"],
            signature_b64=attestation_data["signature_b64"],
            attestation_digest=attestation_data["attestation_digest"],
            timestamp_token=attestation_data.get("timestamp_token"),
            attested_at_utc=attestation_data.get("attested_at_utc", ""),
        )
        
        return cls(
            envelope_version=data.get("envelope_version", "2.1.0"),
            attestation_payload=payload,
            witness_attestation=attestation,
            status=data.get("status", "COMMITTED"),
        )
