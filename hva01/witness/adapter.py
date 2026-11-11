"""
HVA-01 Passive Witness Session Adapter.
Bridges Wave 1-3 Runner storage with Wave 4B cryptographic evidence sealing
without mutating FSM semantics or response-locking contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

from ..crypto.canonical import canonicalize_json
from ..crypto.merkle import (
    build_merkle_tree,
    compute_leaf_hash,
    generate_inclusion_proof,
)
from ..crypto.signatures import SyntheticWitnessKey, sign_detached
from ..models.response import ResponseRecord
from ..storage.append_store import AppendOnlyStore
from .models import AttestationPayload, GenesisCommitment, WitnessAttestation, WitnessEnvelope
from .timestamp import SyntheticTimestampEngine, compute_attestation_digest


class WitnessSessionAdapter:
    """
    Passive adapter that reads committed session records, constructs canonical
    Merkle tree commitments, issues synthetic witness signatures, and binds timestamps.
    """

    @classmethod
    def seal_session_evidence(
        cls,
        session_id: str,
        witness_key: SyntheticWitnessKey,
        witness_priv: ed25519.Ed25519PrivateKey,
        timestamp_engine: SyntheticTimestampEngine,
        genesis: GenesisCommitment,
        store: AppendOnlyStore,
        is_withdrawal: bool = False,
    ) -> Tuple[WitnessEnvelope, List[List[Dict[str, str]]]]:
        """
        Seal session evidence and output canonical witness envelope and inclusion proofs.
        """
        records, _ = store.load_master_records(session_id, verify=False)
        if not records:
            raise ValueError(f"Cannot seal evidence for session '{session_id}': No master records found.")

        # 1. Compute Leaf Hashes
        leaf_hashes = []
        leaf_hex_list = []
        for rec in records:
            rec_dict = rec.to_canonical_dict() if hasattr(rec, "to_canonical_dict") else rec.to_dict()
            rec_bytes = canonicalize_json(rec_dict)
            l_hash = compute_leaf_hash(rec_bytes)
            leaf_hashes.append(l_hash)
            leaf_hex_list.append(f"sha256:{l_hash.hex()}")

        # 2. Build Merkle Tree
        root_bytes, layers = build_merkle_tree(leaf_hashes)
        root_hex = f"sha256:{root_bytes.hex()}"

        # 3. Generate Inclusion Proofs
        proofs = []
        for i in range(len(records)):
            proof = generate_inclusion_proof(i, layers)
            proofs.append(proof)

        # 4. Construct Attestation Payload
        status_str = "WITHDRAWN" if is_withdrawal else "COMMITTED"
        payload = AttestationPayload(
            session_id=session_id,
            protocol_version=genesis.protocol_version,
            genesis_hash=genesis.genesis_hash,
            merkle_root=root_hex,
            leaf_count=len(records),
            leaf_hashes=leaf_hex_list,
        )

        # 5. Sign Canonical Payload with Witness Key
        signed_bytes = payload.to_canonical_bytes()
        sig_b64 = sign_detached(witness_priv, signed_bytes)

        # 6. Compute Dual-Bound Attestation Digest D_attest
        d_attest = compute_attestation_digest(
            merkle_root=root_hex,
            key_fingerprint=witness_key.public_key_fingerprint,
            signature_b64=sig_b64,
        )

        # 7. Request Synthetic Timestamp Token
        ts_token = timestamp_engine.issue_timestamp(d_attest)

        # 8. Assemble Envelope
        attestation = WitnessAttestation(
            witness_id=witness_key.witness_id,
            witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
            signature_algorithm="Ed25519",
            signature_b64=sig_b64,
            attestation_digest=d_attest,
            timestamp_token=ts_token.to_dict(),
            attested_at_utc=ts_token.timestamp_utc,
        )

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=attestation,
            status=status_str,
        )

        # 9. Persist Sealed Files to Session Directory
        session_dir = store.get_session_dir(session_id)
        if session_dir.exists():
            envelope_path = session_dir / "witness_envelope.json"
            proofs_path = session_dir / "inclusion_proofs.json"
            envelope_path.write_text(json.dumps(envelope.to_dict(), indent=2), encoding="utf-8")
            proofs_path.write_text(json.dumps(proofs, indent=2), encoding="utf-8")

        return envelope, proofs
