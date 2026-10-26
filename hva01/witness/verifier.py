"""
HVA-01 End-to-End Evidence Batch Verifier.
Executes multi-tier cryptographic verification linking Genesis commitments,
sequential hash chains, Merkle roots, detached witness signatures, and timestamp tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..crypto.canonical import CanonicalJCS, canonicalize_json
from ..crypto.merkle import (
    build_merkle_tree,
    compute_leaf_hash,
    verify_inclusion_proof,
)
from ..engine.hasher import CanonicalHasher
from ..models.response import ResponseRecord
from .models import GenesisCommitment, WitnessEnvelope
from .policy import PolicyEvaluationResult, TrustStatus, WitnessTrustPolicy
from .timestamp import SyntheticTimestampEngine


class EvidenceBatchVerifier:
    """
    Fail-closed end-to-end evidence verifier for HVA-01 evaluation records.
    """

    def __init__(
        self,
        trust_policy: WitnessTrustPolicy,
        timestamp_engine: SyntheticTimestampEngine,
    ) -> None:
        self.trust_policy = trust_policy
        self.timestamp_engine = timestamp_engine

    def verify_batch(
        self,
        genesis: GenesisCommitment,
        response_records: List[ResponseRecord],
        envelope: WitnessEnvelope,
        inclusion_proofs: Optional[List[List[Dict[str, str]]]] = None,
        reference_time_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify the complete evidentiary chain for a completed evaluation session.
        Returns a structured dictionary with verification results.
        """
        errors = []

        # 1. Verify Genesis Hash
        calculated_genesis_hash = genesis.compute_hash()
        if genesis.genesis_hash != calculated_genesis_hash:
            errors.append(
                f"GENESIS_HASH_MISMATCH: Genesis hash corrupted. "
                f"Stored '{genesis.genesis_hash}', calculated '{calculated_genesis_hash}'."
            )

        # 2. Verify Session Genesis Binding
        if envelope.attestation_payload.genesis_hash != genesis.genesis_hash:
            errors.append(
                f"ENVELOPE_GENESIS_BINDING_MISMATCH: Envelope binds genesis "
                f"'{envelope.attestation_payload.genesis_hash}', expected '{genesis.genesis_hash}'."
            )

        # 3. Verify Leaf Count & Completeness
        num_records = len(response_records)
        if num_records != envelope.attestation_payload.leaf_count:
            errors.append(
                f"LEAF_COUNT_MISMATCH: Record count {num_records} does not match "
                f"envelope leaf_count {envelope.attestation_payload.leaf_count}."
            )

        if envelope.status == "COMMITTED" and num_records != genesis.total_evaluable_charts:
            errors.append(
                f"INCOMPLETE_EVALUATION_BATCH: Committed session has {num_records} leaves, "
                f"expected {genesis.total_evaluable_charts} from Genesis."
            )

        # 4. Verify Sequential Hash Chain
        valid_chain, chain_err = CanonicalHasher.verify_chain(
            response_records,
            genesis_hash=genesis.genesis_hash,
            expected_total_cases=num_records,
        )
        if not valid_chain:
            errors.append(f"HASH_CHAIN_INTEGRITY_FAILURE: {chain_err}")

        # 5. Recompute Merkle Tree from Leaf Records
        leaf_hashes = []
        for rec in response_records:
            rec_dict = rec.to_canonical_dict() if hasattr(rec, "to_canonical_dict") else rec.to_dict()
            rec_canonical_bytes = canonicalize_json(rec_dict)
            leaf_hash = compute_leaf_hash(rec_canonical_bytes)
            leaf_hashes.append(leaf_hash)

        calculated_root_bytes, layers = build_merkle_tree(leaf_hashes)
        calculated_root_hex = f"sha256:{calculated_root_bytes.hex()}"

        if calculated_root_hex != envelope.attestation_payload.merkle_root:
            errors.append(
                f"MERKLE_ROOT_MISMATCH: Calculated root '{calculated_root_hex}' does not match "
                f"envelope root '{envelope.attestation_payload.merkle_root}'."
            )

        # 6. Verify Inclusion Proofs (if provided)
        if inclusion_proofs:
            if len(inclusion_proofs) != num_records:
                errors.append(
                    f"INCLUSION_PROOF_COUNT_MISMATCH: Provided {len(inclusion_proofs)} proofs "
                    f"for {num_records} records."
                )
            else:
                for idx, (l_hash, proof) in enumerate(zip(leaf_hashes, inclusion_proofs)):
                    if not verify_inclusion_proof(
                        l_hash,
                        proof,
                        calculated_root_bytes,
                        leaf_index=idx,
                        tree_size=num_records,
                    ):
                        errors.append(f"INCLUSION_PROOF_INVALID: Leaf at index {idx} failed proof verification.")

        # 7. Evaluate Trust Policy on Envelope
        policy_result = self.trust_policy.evaluate_attestation(
            envelope,
            self.timestamp_engine,
            reference_time_utc=reference_time_utc,
        )
        if not policy_result.is_accepted:
            errors.append(f"TRUST_POLICY_FAILURE: [{policy_result.status.value}] {policy_result.rejection_reason}")

        # 8. Produce Final Verdict
        is_verified = (len(errors) == 0)
        return {
            "verified": is_verified,
            "session_id": envelope.attestation_payload.session_id,
            "leaf_count": num_records,
            "merkle_root": calculated_root_hex,
            "trust_status": policy_result.status.value,
            "errors": errors,
        }
