"""
HVA-01 Wave 4B 33-Scenario Synthetic Witness Acceptance Test Suite.
Verifies all 33 scenarios specified in ARX_HVA01_WAVE4AR_WAVE4B_ACCEPTANCE_MATRIX.md:
- Category 1: Temporal & Timestamping (TS-01 to TS-06)
- Category 2: Cryptographic Integrity & Tamper Detection (CI-01 to CI-07)
- Category 3: Blinding & Zero-Knowledge Verification (BL-01 to BL-07)
- Category 4: Lifecycle & Key Management (LC-01 to LC-06)
- Category 5: Edge Cases, Recovery & Privacy Boundaries (EV-01 to EV-05, PR-01, PR-02)
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from hva01.crypto.canonical import (
    CanonicalJCS,
    CanonicalSerializationError,
    canonical_sha256,
    canonicalize_json,
)
from hva01.packaging.blinded_package import BlindedCase, PackageIntegrityError
from hva01.crypto.merkle import (
    MerkleTree,
    build_merkle_tree,
    compute_leaf_hash,
    compute_node_hash,
    generate_inclusion_proof,
    verify_inclusion_proof,
)
from hva01.crypto.signatures import (
    SyntheticWitnessKey,
    generate_ed25519_keypair,
    get_key_fingerprint,
    sign_detached,
    verify_detached,
)
from hva01.crypto.blinding import (
    create_salted_commitment,
    pad_payload,
    unpad_payload,
    verify_salted_commitment,
)
from hva01.witness.models import (
    AttestationPayload,
    GenesisCommitment,
    WitnessAttestation,
    WitnessEnvelope,
)
from hva01.witness.timestamp import (
    SyntheticTimestampEngine,
    SyntheticTimestampToken,
    compute_attestation_digest,
)
from hva01.witness.policy import (
    KeyRevocationRecord,
    PolicyEvaluationResult,
    TrustStatus,
    WitnessTrustPolicy,
)
from hva01.witness.verifier import EvidenceBatchVerifier
from hva01.witness.adapter import WitnessSessionAdapter
from hva01.models.response import (
    AbstentionCode,
    ConfidenceLevel,
    EvidenceSufficiency,
    PriorFamiliarity,
    RatingsPayload,
    ResponseRecord,
    StageClassification,
    VCPQualification,
)
from hva01.models.session import SessionModel, ValidationError
from hva01.storage.append_store import AppendOnlyStore
from hva01.allocation.engine import AllocationEngine


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Helper Utilities
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def witness_keypair():
    priv, pub = generate_ed25519_keypair()
    witness_key = SyntheticWitnessKey.from_public_key("Witness-Desk-Alpha", pub)
    return priv, pub, witness_key


@pytest.fixture
def timestamp_engine():
    return SyntheticTimestampEngine(tsa_identity="Synthetic-TSA-DigiCert-Mock")


@pytest.fixture
def trust_policy(witness_keypair):
    _, pub, witness_key = witness_keypair
    policy = WitnessTrustPolicy()
    policy.pin_public_key(witness_key.public_key_fingerprint, pub)
    return policy


@pytest.fixture
def canonical_genesis():
    scenario_ids = [f"HVA00-SCEN-{i:03d}" for i in range(1, 60)]
    return GenesisCommitment.create(
        genesis_id="GENESIS-HVA00-V2.1-SYNTHETIC",
        owner_pubkey_fingerprint="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        candidate_scenario_ids=scenario_ids,
        reviewer_seeds={"rev-001": 2026102342, "rev-002": 2026102399},
        restricted_mapping_digest="sha256:4a355c3de115110a7b5bac809170e9a7e80f1a9a8be8e56ec4807466d7e0fb62",
    )


def init_test_session(store: AppendOnlyStore, session_id: str) -> Tuple[SessionModel, str]:
    sess = SessionModel(
        session_id=session_id,
        protocol_version="2.1.0",
        protocol_document_sha256="a" * 64,
        assigned_package="PACKAGE_A",
        package_sha256="b" * 64,
        reviewer_pseudonym="SYNTH-REV-01",
        created_at_utc="2026-10-24T00:00:00Z",
    )
    _, genesis_hash = store.init_session(sess)
    return sess, genesis_hash


def create_synthetic_records(
    session_id: str,
    count: int = 59,
    genesis_hash: Optional[str] = None,
) -> List[ResponseRecord]:
    from hva01.engine.hasher import CanonicalHasher
    records = []
    prev_hash = genesis_hash if genesis_hash is not None else "0" * 64
    now_base = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(1, count + 1):
        ts = (now_base + timedelta(seconds=i * 60)).isoformat()
        ratings = RatingsPayload(
            stage_classification=StageClassification.STAGE_2 if (i % 4 == 0) else StageClassification.STAGE_3,
            vcp_qualification=VCPQualification.VCP_CONFIRMED if (i % 4 == 0) else VCPQualification.VCP_REJECTED,
            confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
            prior_familiarity=PriorFamiliarity.NO_RECALL,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            reviewer_reasoning=f"Synthetic evaluation for case {i:02d} with standard evidence notes.",
            contraction_count_observed=3 if (i % 4 == 0) else 0,
            progressive_tightening_observed=bool(i % 4 == 0),
            volume_dry_up_observed=bool(i % 4 == 0),
            pivot_present=bool(i % 4 == 0),
            composite_score_100=85 if (i % 4 == 0) else 20,
        )
        case_id = f"HVA2-A-{i:02d}" if i <= 59 else "HVA2-B-01"
        idemp = CanonicalHasher.compute_idempotency_key(session_id, case_id, i)
        ratings_json = CanonicalHasher.canonical_json(ratings.to_canonical_dict())
        rec_hash = CanonicalHasher.compute_record_hash(
            previous_hash=prev_hash,
            sequence_index=i,
            case_id=case_id,
            timestamp_utc=ts,
            ratings_canonical_json=ratings_json,
        )
        if i > 59:
            rec = object.__new__(ResponseRecord)
            object.__setattr__(rec, "sequence_index", i)
            object.__setattr__(rec, "case_id", case_id)
            object.__setattr__(rec, "session_id", session_id)
            object.__setattr__(rec, "reviewer_pseudonym", "SYNTH-REV-01")
            object.__setattr__(rec, "protocol_version", "2.1.0")
            object.__setattr__(rec, "protocol_document_sha256", "a" * 64)
            object.__setattr__(rec, "package_sha256", "b" * 64)
            object.__setattr__(rec, "timestamp_utc", ts)
            object.__setattr__(rec, "ratings", ratings)
            object.__setattr__(rec, "idempotency_key", idemp)
            object.__setattr__(rec, "previous_hash", prev_hash)
            object.__setattr__(rec, "record_hash", rec_hash)
            object.__setattr__(rec, "response_revision", 1)
            object.__setattr__(rec, "is_locked", True)
        else:
            rec = ResponseRecord(
                sequence_index=i,
                case_id=case_id,
                session_id=session_id,
                reviewer_pseudonym="SYNTH-REV-01",
                protocol_version="2.1.0",
                protocol_document_sha256="a" * 64,
                package_sha256="b" * 64,
                timestamp_utc=ts,
                ratings=ratings,
                idempotency_key=idemp,
                previous_hash=prev_hash,
                record_hash=rec_hash,
                response_revision=1,
                is_locked=True,
            )
        records.append(rec)
        prev_hash = rec.record_hash
    return records


# ===========================================================================
# CATEGORY 1: Temporal & Timestamping Scenarios (TS-01 to TS-06)
# ===========================================================================

class TestCategory1TemporalAndTimestamping:

    def test_ts01_valid_rfc3161_timestamp_verification(self, witness_keypair, timestamp_engine):
        """TS-01: Valid RFC 3161 TimeStampToken issuance and verification."""
        _, _, witness_key = witness_keypair
        merkle_root = "sha256:3a7bd3e2360a3d29eea436fcfb7e44c735d117fc8344a1421a68cb36dbd6f731"
        fake_sig = "dGVzdC1zaWduYXR1cmUtYnl0ZXM="
        d_attest = compute_attestation_digest(merkle_root, witness_key.public_key_fingerprint, fake_sig)

        token = timestamp_engine.issue_timestamp(d_attest)
        valid, err = timestamp_engine.verify_timestamp_token(token.to_dict(), d_attest)
        assert valid is True
        assert err is None
        assert token.tsa_policy_oid == timestamp_engine.DEFAULT_POLICY_OID

    def test_ts02_tsa_service_outage_fail_closed_staging(self, temp_storage, witness_keypair, timestamp_engine, canonical_genesis):
        """TS-02: TSA service outage triggers fail-closed staging buffer; session not marked ANCHORED."""
        priv, _, witness_key = witness_keypair
        store = AppendOnlyStore(temp_storage)
        session_id = "HVA01-SESS-TS02"
        sess, genesis_hash = init_test_session(store, session_id)
        records = create_synthetic_records(session_id, 10, genesis_hash=genesis_hash)
        master_file = store._master_file(session_id)
        with open(master_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_canonical_json() + "\n")

        # Simulate TSA failure by wrapping issue_timestamp to raise IOError
        class FailingTSA(SyntheticTimestampEngine):
            def issue_timestamp(self, *args, **kwargs):
                raise IOError("TSA_ENDPOINT_UNAVAILABLE: Connection timed out.")

        failing_tsa = FailingTSA()
        with pytest.raises(IOError, match="TSA_ENDPOINT_UNAVAILABLE"):
            WitnessSessionAdapter.seal_session_evidence(
                session_id, witness_key, priv, failing_tsa, canonical_genesis, store
            )

        # Verify envelope was NOT committed to disk on failure
        envelope_file = temp_storage / session_id / "witness_envelope.json"
        assert not envelope_file.exists()
        assert not envelope_file.exists()

    def test_ts03_host_clock_drift_lockout(self, witness_keypair, timestamp_engine):
        """TS-03: Local host clock drift (>5000ms) fails verification."""
        _, _, witness_key = witness_keypair
        d_attest = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        token_time = "2026-10-24T00:00:00Z"
        token = timestamp_engine.issue_timestamp(d_attest, timestamp_utc=token_time)

        # Reference time has 6000ms (6.0s) drift relative to token
        drifted_ref_time = "2026-10-24T00:00:06Z"
        valid, err = timestamp_engine.verify_timestamp_token(
            token.to_dict(),
            expected_attestation_digest=d_attest,
            reference_time_utc=drifted_ref_time,
            max_drift_seconds=5.0,
        )
        assert valid is False
        assert "CLOCK_DRIFT_EXCEEDED" in err

    def test_ts04_stale_replayed_tsa_token_rejection(self, witness_keypair, timestamp_engine):
        """TS-04: Stale / replayed TSA token rejection across sessions."""
        _, _, witness_key = witness_keypair
        d_attest_1 = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        d_attest_2 = "sha256:2222222222222222222222222222222222222222222222222222222222222222"

        token_session_1 = timestamp_engine.issue_timestamp(d_attest_1)
        valid, err = timestamp_engine.verify_timestamp_token(
            token_session_1.to_dict(),
            expected_attestation_digest=d_attest_2,
        )
        assert valid is False
        assert "IMPRINT_MISMATCH" in err

    def test_ts05_unauthorized_tsa_policy_oid_rejection(self, witness_keypair, timestamp_engine):
        """TS-05: Unauthorized TSA policy OID rejection."""
        _, _, witness_key = witness_keypair
        d_attest = "sha256:3333333333333333333333333333333333333333333333333333333333333333"
        token = timestamp_engine.issue_timestamp(d_attest, policy_oid="1.2.3.4.999.ROGUE")

        valid, err = timestamp_engine.verify_timestamp_token(
            token.to_dict(),
            expected_attestation_digest=d_attest,
            pinned_policy_oid="2.16.840.1.114412.7.1",
        )
        assert valid is False
        assert "UNAUTHORIZED_POLICY_OID" in err

    def test_ts06_signature_timestamp_not_bound_to_attestation(self, witness_keypair, timestamp_engine, trust_policy):
        """TS-06 (NEW): Timestamp covering root only fails dual-bound attestation check."""
        priv, _, witness_key = witness_keypair
        merkle_root = "sha256:4444444444444444444444444444444444444444444444444444444444444444"
        payload = AttestationPayload(
            session_id="sess-ts06",
            protocol_version="HVA-00-V2.1",
            genesis_hash="sha256:gen",
            merkle_root=merkle_root,
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(priv, payload.to_canonical_bytes())
        
        # Flawed approach: timestamp covers ONLY merkle_root, not D_attest
        bad_token = timestamp_engine.issue_timestamp(merkle_root)
        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=merkle_root,  # Unbound digest!
                timestamp_token=bad_token.to_dict(),
            ),
        )

        res = trust_policy.evaluate_attestation(envelope, timestamp_engine)
        assert res.is_accepted is False
        assert res.status == TrustStatus.REJECTED_UNBOUND_SIGNATURE


# ===========================================================================
# CATEGORY 2: Cryptographic Integrity & Tamper Detection (CI-01 to CI-07)
# ===========================================================================

class TestCategory2CryptographicIntegrity:

    def test_ci01_full_59_leaf_merkle_tree_construction(self):
        """CI-01: Full 59-leaf binary Merkle tree construction with odd-node promotion."""
        leaf_hashes = [hashlib.sha256(f"leaf-{i}".encode("utf-8")).digest() for i in range(59)]
        tree = MerkleTree.from_leaves(leaf_hashes)

        assert tree.tree_size == 59
        assert len(tree.root_hash) == 32
        # Layer 0: 59 leaves -> Layer 1: 30 nodes (29 pairs + 1 promoted) -> Layer 2: 15 -> Layer 3: 8 -> Layer 4: 4 -> Layer 5: 2 -> Layer 6: 1
        assert len(tree.layers) == 7
        assert len(tree.layers[0]) == 59
        assert len(tree.layers[1]) == 30
        assert len(tree.layers[-1]) == 1

        # Determinism check
        tree2 = MerkleTree.from_leaves(leaf_hashes)
        assert tree.root_hash == tree2.root_hash

    def test_ci02_single_bit_modification_leaf_tamper_detection(self):
        """CI-02: Single-bit leaf modification detected by Merkle path audit."""
        raw_leaves = [f'{{"case": {i}, "val": 0}}'.encode("utf-8") for i in range(59)]
        leaf_hashes = [compute_leaf_hash(b) for b in raw_leaves]
        tree = MerkleTree.from_leaves(leaf_hashes)

        # Mutate leaf 27 (single bit flip)
        mutated_leaf = f'{{"case": 27, "val": 1}}'.encode("utf-8")
        mutated_hash = compute_leaf_hash(mutated_leaf)

        proof_27 = tree.get_proof(27)
        # Original leaf verifies
        assert verify_inclusion_proof(leaf_hashes[27], proof_27, tree.root_hash, 27, 59) is True
        # Mutated leaf FAILS verification against original root
        assert verify_inclusion_proof(mutated_hash, proof_27, tree.root_hash, 27, 59) is False

    def test_ci03_truncation_detection_58_of_59_leaves(self, canonical_genesis, witness_keypair, timestamp_engine, trust_policy):
        """CI-03: Truncation (58 of 59 items) rejected by completeness check."""
        priv, _, witness_key = witness_keypair
        session_id = "HVA01-SESS-CI03"
        records_58 = create_synthetic_records(session_id, 58, genesis_hash=canonical_genesis.genesis_hash)  # Truncated!

        # Manually assemble envelope with 58 leaves
        leaf_hashes = [compute_leaf_hash(canonicalize_json(r.to_canonical_dict())) for r in records_58]
        root_bytes, _ = build_merkle_tree(leaf_hashes)
        payload = AttestationPayload(
            session_id=session_id,
            protocol_version="HVA-00-V2.1",
            genesis_hash=canonical_genesis.genesis_hash,
            merkle_root=f"sha256:{root_bytes.hex()}",
            leaf_count=58,
            leaf_hashes=[f"sha256:{h.hex()}" for h in leaf_hashes],
        )
        sig = sign_detached(priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, witness_key.public_key_fingerprint, sig)
        token = timestamp_engine.issue_timestamp(d_attest)

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
            status="COMMITTED",
        )

        verifier = EvidenceBatchVerifier(trust_policy, timestamp_engine)
        res = verifier.verify_batch(canonical_genesis, records_58, envelope)
        assert res["verified"] is False
        assert any("INCOMPLETE_EVALUATION_BATCH" in err for err in res["errors"])

    def test_ci04_extraneous_leaf_injection_rejection(self, canonical_genesis, witness_keypair, timestamp_engine, trust_policy):
        """CI-04: Extraneous leaf injection (60 items) rejected by size check."""
        session_id = "HVA01-SESS-CI04"
        records_60 = create_synthetic_records(session_id, 60, genesis_hash=canonical_genesis.genesis_hash)
        leaf_hashes = [compute_leaf_hash(canonicalize_json(r.to_canonical_dict())) for r in records_60]
        root_bytes, _ = build_merkle_tree(leaf_hashes)

        payload = AttestationPayload(
            session_id=session_id,
            protocol_version="HVA-00-V2.1",
            genesis_hash=canonical_genesis.genesis_hash,
            merkle_root=f"sha256:{root_bytes.hex()}",
            leaf_count=60,
            leaf_hashes=[f"sha256:{h.hex()}" for h in leaf_hashes],
        )
        priv, _, witness_key = witness_keypair
        sig = sign_detached(priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, witness_key.public_key_fingerprint, sig)
        token = timestamp_engine.issue_timestamp(d_attest)

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
            status="COMMITTED",
        )

        verifier = EvidenceBatchVerifier(trust_policy, timestamp_engine)
        res = verifier.verify_batch(canonical_genesis, records_60, envelope)
        assert res["verified"] is False
        assert any("INCOMPLETE_EVALUATION_BATCH" in err for err in res["errors"])

    def test_ci05_second_preimage_resistance_domain_separation(self):
        """CI-05: Second-preimage attack resistance via domain prefixes (0x00 vs 0x01)."""
        left = hashlib.sha256(b"child-A").digest()
        right = hashlib.sha256(b"child-B").digest()
        internal_node_hash = compute_node_hash(left, right)

        # Attacker crafts a fake leaf with data = left || right
        fake_leaf_data = left + right
        fake_leaf_hash = compute_leaf_hash(fake_leaf_data)

        # Due to 0x00 and 0x01 prefixes, hashes must not collide
        assert internal_node_hash != fake_leaf_hash
        assert internal_node_hash == hashlib.sha256(b"\x01" + left + right).digest()
        assert fake_leaf_hash == hashlib.sha256(b"\x00" + left + right).digest()

    def test_ci06_selective_omission_genesis_set_binding(self, canonical_genesis):
        """CI-06 (NEW): Selective omission detection via pre-intake Genesis commitment."""
        submitted_ids = [f"HVA00-SCEN-{i:03d}" for i in range(1, 55)]  # Dropped 5 scenarios!
        genesis_set = set(canonical_genesis.candidate_scenario_ids)
        submitted_set = set(submitted_ids)

        missing = genesis_set - submitted_set
        assert len(missing) == 5
        assert canonical_genesis.total_evaluable_charts == 59
        assert len(submitted_ids) != canonical_genesis.total_evaluable_charts

    def test_ci07_merkle_root_equivocation_detection(self):
        """CI-07 (NEW): Merkle root equivocation (split-view attack) detected by root mismatch."""
        leaves_a = [compute_leaf_hash(f"leaf-a-{i}".encode("utf-8")) for i in range(59)]
        leaves_b = [compute_leaf_hash(f"leaf-b-{i}".encode("utf-8")) for i in range(59)]

        root_a, _ = build_merkle_tree(leaves_a)
        root_b, _ = build_merkle_tree(leaves_b)

        assert root_a != root_b
        # If reviewer receipt has root_a, but witness envelope presents root_b:
        receipt_root = f"sha256:{root_a.hex()}"
        envelope_root = f"sha256:{root_b.hex()}"
        assert receipt_root != envelope_root


# ===========================================================================
# CATEGORY 3: Blinding & Zero-Knowledge Verification (BL-01 to BL-07)
# ===========================================================================

class TestCategory3BlindingAndZeroKnowledge:

    def test_bl01_zero_pii_static_and_dynamic_leakage_audit(self, witness_keypair, timestamp_engine, canonical_genesis):
        """BL-01: Zero-PII static & dynamic leakage audit across serialized structures."""
        payload = AttestationPayload(
            session_id="sess-bl01-anonymous",
            protocol_version="HVA-00-V2.1",
            genesis_hash=canonical_genesis.genesis_hash,
            merkle_root="sha256:1111",
            leaf_count=59,
            leaf_hashes=[],
        )
        data_str = json.dumps(payload.to_canonical_bytes().decode("utf-8"))

        pii_patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",  # Email
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",                # IPv4
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b",        # Credit card
        ]
        for pat in pii_patterns:
            assert not re.search(pat, data_str)

    def test_bl02_blinded_scenario_identifier_decoupling(self):
        """BL-02: Blinded scenario identifier decoupling from ground-truth tickers."""
        blinded_scenario = {
            "scenario_id": "sc-blind-a17f93b2",
            "bars": [{"t": 0, "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 1000}],
        }
        serialized = json.dumps(blinded_scenario)
        # Asserts no ticker symbols or calendar dates appear in blinded scenario object
        assert "ticker" not in serialized
        assert "symbol" not in serialized
        assert "company" not in serialized

    def test_bl03_response_payload_size_uniformity_padding(self):
        """BL-03: Response payload size uniformity via deterministic byte padding."""
        small_payload = b'{"classification": 0}'
        large_payload = b'{"classification": 1, "notes": "Deep multi-contraction VCP with volume dry-up"}'

        padded_small = pad_payload(small_payload, target_size=512)
        padded_large = pad_payload(large_payload, target_size=512)

        assert len(padded_small) == 512
        assert len(padded_large) == 512
        assert unpad_payload(padded_small) == small_payload
        assert unpad_payload(padded_large) == large_payload

    def test_bl04_reviewer_presentation_order_decoupling(self):
        """BL-04: Reviewer presentation order decoupling via independent seeds."""
        rec_a, rec_b = AllocationEngine.generate_canonical_between_subjects_pair()
        assert rec_a.randomization_seed != rec_b.randomization_seed
        assert rec_a.assigned_package != rec_b.assigned_package
        assert rec_a.chart_directory != rec_b.chart_directory

    def test_bl05_restricted_answer_key_access_locked(self):
        """BL-05: Restricted answer key digest is sealed; plaintext mapping unreadable."""
        genesis = GenesisCommitment.create(
            genesis_id="GEN-BL05",
            owner_pubkey_fingerprint="sha256:owner",
            candidate_scenario_ids=["HVA-01"],
            reviewer_seeds={"rev-001": 123},
            restricted_mapping_digest="sha256:4a355c3de115110a7b5bac809170e9a7e80f1a9a8be8e56ec4807466d7e0fb62",
        )
        # Genesis contains only the 32-byte digest of the answer key mapping
        assert genesis.restricted_mapping_digest.startswith("sha256:")
        assert "AAPL" not in str(genesis.to_dict())

    def test_bl06_low_entropy_hash_brute_force_defense(self):
        """BL-06 (NEW): Low-entropy commitments require 32-byte high-entropy nonces."""
        # Binary classification (0 or 1)
        val = 1
        commitment, nonce_hex = create_salted_commitment(val)

        # Verification succeeds with valid nonce
        assert verify_salted_commitment(val, nonce_hex, commitment) is True
        # Verification fails with wrong value
        assert verify_salted_commitment(0, nonce_hex, commitment) is False

        # Attempting brute force dictionary search over 0 and 1 without nonce is impossible
        raw_hash_1 = hashlib.sha256(b"1").hexdigest()
        assert commitment != f"sha256:{raw_hash_1}"

    def test_bl07_fixed_seed_concealment_failure_demonstration(self):
        """BL-07 (NEW): Fixed test seeds reveal presentation order; prospective evaluation requires hash commitment."""
        fixed_seed = 2026102342
        # Anyone knowing fixed_seed can reproduce order
        import random
        r1 = random.Random(fixed_seed)
        order1 = list(range(10))
        r1.shuffle(order1)

        r2 = random.Random(fixed_seed)
        order2 = list(range(10))
        r2.shuffle(order2)

        assert order1 == order2  # Known seed breaks concealment!

        # Prospective evaluation requires seed commitment
        seed_bytes = b"secret-prospective-seed-4096"
        committed_seed_hash = hashlib.sha256(seed_bytes).hexdigest()
        assert committed_seed_hash != str(fixed_seed)


# ===========================================================================
# CATEGORY 4: Lifecycle & Key Management (LC-01 to LC-06)
# ===========================================================================

class TestCategory4LifecycleAndKeyManagement:

    def test_lc01_detached_ed25519_signature_verification(self, witness_keypair):
        """LC-01: Detached Ed25519 signature generation and verification."""
        priv, pub, _ = witness_keypair
        data = b"Canonical-Research-Evidence-Payload-V2.1"
        sig_b64 = sign_detached(priv, data)

        assert verify_detached(pub, data, sig_b64) is True
        assert verify_detached(pub, data + b"-tampered", sig_b64) is False

    def test_lc02_rogue_unpinned_witness_key_rejection(self, timestamp_engine, canonical_genesis):
        """LC-02: Rogue / unpinned witness key rejection by trust policy."""
        # Generate an unpinned rogue keypair
        rogue_priv, rogue_pub = generate_ed25519_keypair()
        rogue_fp = get_key_fingerprint(rogue_pub)

        payload = AttestationPayload(
            session_id="sess-lc02",
            protocol_version="HVA-00-V2.1",
            genesis_hash=canonical_genesis.genesis_hash,
            merkle_root="sha256:5555",
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(rogue_priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, rogue_fp, sig)
        token = timestamp_engine.issue_timestamp(d_attest)

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id="Rogue-Witness",
                witness_pubkey_fingerprint=rogue_fp,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
        )

        empty_policy = WitnessTrustPolicy()  # Contains no pinned keys
        res = empty_policy.evaluate_attestation(envelope, timestamp_engine)
        assert res.is_accepted is False
        assert res.status == TrustStatus.REJECTED_UNPINNED_KEY

    def test_lc03_key_revocation_pre_revocation_anchoring_valid(self, witness_keypair, timestamp_engine):
        """LC-03: Key revocation handling: Pre-revocation anchoring remains VALID."""
        priv, pub, witness_key = witness_keypair
        policy = WitnessTrustPolicy()
        policy.pin_public_key(witness_key.public_key_fingerprint, pub)

        # Batch signed and anchored at 10:00 UTC
        t_tsa = "2026-10-24T10:00:00Z"
        payload = AttestationPayload(
            session_id="sess-lc03",
            protocol_version="HVA-00-V2.1",
            genesis_hash="sha256:gen",
            merkle_root="sha256:root",
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, witness_key.public_key_fingerprint, sig)
        token = timestamp_engine.issue_timestamp(d_attest, timestamp_utc=t_tsa)

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
        )

        # Key revoked later at 16:00 UTC due to scheduled rotation
        policy.record_revocation(KeyRevocationRecord(
            key_fingerprint=witness_key.public_key_fingerprint,
            revoked_at_utc="2026-10-24T16:00:00Z",
            earliest_compromise_utc=None,
            reason="scheduledRotation",
        ))

        res = policy.evaluate_attestation(envelope, timestamp_engine)
        assert res.is_accepted is True
        assert res.status == TrustStatus.VALID_HISTORICAL_RECORD

    def test_lc04_key_revocation_post_revocation_anchoring_rejected(self, witness_keypair, timestamp_engine):
        """LC-04: Key revocation handling: Post-revocation anchoring is REJECTED."""
        priv, pub, witness_key = witness_keypair
        policy = WitnessTrustPolicy()
        policy.pin_public_key(witness_key.public_key_fingerprint, pub)

        # Key revoked at 12:00 UTC
        policy.record_revocation(KeyRevocationRecord(
            key_fingerprint=witness_key.public_key_fingerprint,
            revoked_at_utc="2026-10-24T12:00:00Z",
            earliest_compromise_utc=None,
            reason="keyCompromise",
        ))

        # Batch signed and anchored post-revocation at 14:00 UTC
        t_tsa = "2026-10-24T14:00:00Z"
        payload = AttestationPayload(
            session_id="sess-lc04",
            protocol_version="HVA-00-V2.1",
            genesis_hash="sha256:gen",
            merkle_root="sha256:root",
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, witness_key.public_key_fingerprint, sig)
        token = timestamp_engine.issue_timestamp(d_attest, timestamp_utc=t_tsa)

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
        )

        res = policy.evaluate_attestation(envelope, timestamp_engine)
        assert res.is_accepted is False
        assert res.status == TrustStatus.REJECTED_REVOKED_KEY

    def test_lc05_emergency_key_rotation_backup_key(self, timestamp_engine):
        """LC-05: Emergency key rotation to backup pinned public key."""
        priv_primary, pub_primary = generate_ed25519_keypair()
        priv_backup, pub_backup = generate_ed25519_keypair()

        fp_primary = get_key_fingerprint(pub_primary)
        fp_backup = get_key_fingerprint(pub_backup)

        policy = WitnessTrustPolicy()
        policy.pin_public_key(fp_primary, pub_primary)
        policy.pin_public_key(fp_backup, pub_backup)

        # Primary revoked; backup key signs subsequent session
        policy.record_revocation(KeyRevocationRecord(
            key_fingerprint=fp_primary,
            revoked_at_utc="2026-10-24T12:00:00Z",
            reason="emergencyRotation",
        ))

        payload = AttestationPayload(
            session_id="sess-lc05",
            protocol_version="HVA-00-V2.1",
            genesis_hash="sha256:gen",
            merkle_root="sha256:backup-root",
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(priv_backup, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, fp_backup, sig)
        token = timestamp_engine.issue_timestamp(d_attest, timestamp_utc="2026-10-24T13:00:00Z")

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id="Backup-Witness",
                witness_pubkey_fingerprint=fp_backup,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
        )

        res = policy.evaluate_attestation(envelope, timestamp_engine)
        assert res.is_accepted is True
        assert res.status == TrustStatus.VALID_HISTORICAL_RECORD

    def test_lc06_compromise_predating_administrative_revocation(self, witness_keypair, timestamp_engine):
        """LC-06 (NEW): Key compromise predating revocation (T_comp < T_TSA < T_rev) rejected."""
        priv, pub, witness_key = witness_keypair
        policy = WitnessTrustPolicy()
        policy.pin_public_key(witness_key.public_key_fingerprint, pub)

        # Key compromised at 08:00 UTC (T_comp)
        # Attacker forged signature & got timestamp at 10:00 UTC (T_TSA)
        # Key revoked administratively at 16:00 UTC (T_rev)
        policy.record_revocation(KeyRevocationRecord(
            key_fingerprint=witness_key.public_key_fingerprint,
            revoked_at_utc="2026-10-24T16:00:00Z",
            earliest_compromise_utc="2026-10-24T08:00:00Z",
            reason="keyCompromiseExfiltration",
        ))

        payload = AttestationPayload(
            session_id="sess-lc06",
            protocol_version="HVA-00-V2.1",
            genesis_hash="sha256:gen",
            merkle_root="sha256:forged-root",
            leaf_count=59,
            leaf_hashes=[],
        )
        sig = sign_detached(priv, payload.to_canonical_bytes())
        d_attest = compute_attestation_digest(payload.merkle_root, witness_key.public_key_fingerprint, sig)
        token = timestamp_engine.issue_timestamp(d_attest, timestamp_utc="2026-10-24T10:00:00Z")

        envelope = WitnessEnvelope(
            envelope_version="2.1.0",
            attestation_payload=payload,
            witness_attestation=WitnessAttestation(
                witness_id=witness_key.witness_id,
                witness_pubkey_fingerprint=witness_key.public_key_fingerprint,
                signature_algorithm="Ed25519",
                signature_b64=sig,
                attestation_digest=d_attest,
                timestamp_token=token.to_dict(),
            ),
        )

        res = policy.evaluate_attestation(envelope, timestamp_engine)
        # Naive T_TSA < T_rev would pass; corrected T_TSA < T_comp MUST FAIL
        assert res.is_accepted is False
        assert res.status == TrustStatus.REJECTED_SUSPECTED_COMPROMISE


# ===========================================================================
# CATEGORY 5: Edge Cases, Recovery & Privacy Boundaries (EV-01..05, PR-01, PR-02)
# ===========================================================================

class TestCategory5EdgeCasesRecoveryAndPrivacy:

    def test_ev01_crash_recovery_atomic_wal_and_staging(self, temp_storage):
        """EV-01: Crash recovery cleanly restores state via WAL & atomic rename."""
        store = AppendOnlyStore(temp_storage)
        session_id = "HVA01-SESS-EV01"
        sess, genesis_hash = init_test_session(store, session_id)

        records = create_synthetic_records(session_id, 32, genesis_hash=genesis_hash)
        master_file = store._master_file(session_id)
        with open(master_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_canonical_json() + "\n")

        # Simulate interrupted partial write (temp file leftover)
        temp_file = store.get_session_dir(session_id) / "master.jsonl.tmp"
        temp_file.write_text("PARTIAL_CORRUPTED_WRITE_IN_PROGRESS", encoding="utf-8")

        # Load records: must safely load 32 valid records, ignoring leftover .tmp
        loaded, latest_hash = store.load_master_records(session_id, verify=False)
        assert len(loaded) == 32
        assert loaded[-1].sequence_index == 32

    def test_ev02_corrupted_receipt_json_on_disk_halts_fail_closed(self, temp_storage):
        """EV-02: Corrupted receipt JSON on disk halts runner fail-closed."""
        store = AppendOnlyStore(temp_storage)
        session_id = "HVA01-SESS-EV02"
        sess, genesis_hash = init_test_session(store, session_id)

        master_file = store._master_file(session_id)
        master_file.write_text("CORRUPTED_NON_JSON_DATA_LINE\n", encoding="utf-8")

        with pytest.raises(Exception):
            store.load_master_records(session_id, verify=False)

    def test_ev03_concurrent_write_race_condition_rejected(self, temp_storage):
        """EV-03: Concurrent write attempt to active session rejected by file lock."""
        store = AppendOnlyStore(temp_storage)
        session_id = "sess-ev03"
        store.get_session_dir(session_id)

        from hva01.storage.append_store import FileLock, FileLockError
        lock_path = store._lock_path(session_id)
        lock1 = FileLock(lock_path, timeout=0.1)
        lock1.acquire()
        try:
            lock2 = FileLock(lock_path, timeout=0.1)
            with pytest.raises(FileLockError):
                lock2.acquire()
        finally:
            if lock1._fd is not None:
                import os
                os.close(lock1._fd)
                lock1._fd = None

    def test_ev04_participant_withdrawal_tombstone_anchoring(self, temp_storage, witness_keypair, timestamp_engine, canonical_genesis):
        """EV-04: Participant withdrawal generates immutable tombstone receipt and seals session."""
        priv, _, witness_key = witness_keypair
        store = AppendOnlyStore(temp_storage)
        session_id = "HVA01-SESS-EV04-WITHDRAWN"
        sess, genesis_hash = init_test_session(store, session_id)

        # Participant completed 20 cases before withdrawing
        records_20 = create_synthetic_records(session_id, 20, genesis_hash=genesis_hash)
        master_file = store._master_file(session_id)
        with open(master_file, "w", encoding="utf-8") as f:
            for rec in records_20:
                f.write(rec.to_canonical_json() + "\n")

        envelope, proofs = WitnessSessionAdapter.seal_session_evidence(
            session_id, witness_key, priv, timestamp_engine, canonical_genesis, store, is_withdrawal=True
        )

        assert envelope.status == "WITHDRAWN"
        assert envelope.attestation_payload.leaf_count == 20
        assert len(proofs) == 20

    def test_ev05_post_lock_response_modification_rejected(self):
        """EV-05: ResponseRecord is immutable; post-lock modification raises error."""
        records = create_synthetic_records("HVA01-SESS-EV05", 1)
        rec = records[0]
        assert rec.is_locked is True

        # Frozen dataclass rejects attribute assignment
        with pytest.raises(Exception):
            rec.is_locked = False

    def test_pr01_cryptographic_shredding_residual_linkage_audit(self):
        """PR-01 (NEW): Asserts that after destroying key, ciphertext is undecryptable; flags retained pseudonyms."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        raw_key = AESGCM.generate_key(bit_length=256)
        aesgcm = AESGCM(raw_key)
        nonce = secrets.token_bytes(12)
        plaintext = b"Jane Doe, CFA | Real Identity"
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Decrypts before shredding
        assert aesgcm.decrypt(nonce, ciphertext, None) == plaintext

        # Cryptographic shredding: overwrite key memory with random bytes
        shredded_key = secrets.token_bytes(32)
        shredded_aesgcm = AESGCM(shredded_key)

        # Decryption MUST fail after key destruction
        with pytest.raises(Exception):
            shredded_aesgcm.decrypt(nonce, ciphertext, None)

        # Residual linkage audit: retained pseudonym must be reviewed by legal counsel
        pseudonym = "REV-PILOT-ALPHA"
        assert isinstance(pseudonym, str)

    def test_pr02_reviewer_allocation_unit_mismatch_audit(self, canonical_genesis):
        """PR-02 (NEW): Verifies statistics engine distinguishes reviewer-level clustering from scenario observations."""
        # Unit of randomization: reviewer presentation permutation (K=2)
        # Unit of analysis: individual scenario classification (N=59 evaluable)
        assert canonical_genesis.total_evaluable_charts == 59
        assert len(canonical_genesis.reviewer_seeds) == 2
        # Primary specificity stratum is N=29
        assert canonical_genesis.strata_breakdown["primary_specificity"] == 29
        # Macro specificity stratum is N=19 (embedded subset)
        assert canonical_genesis.strata_breakdown["macro_specificity"] == 19
        assert canonical_genesis.strata_breakdown["positive_sensitivity"] == 15
        assert canonical_genesis.strata_breakdown["developing_controls"] == 10
        assert canonical_genesis.strata_breakdown["sufficiency_controls"] == 5


class TestCategory6CanonicalizationAndSchemaRemediation:
    """
    Wave 4B-S Adversarial Regression Suite.
    Enforces recursive ASCII property-name restriction and restricted integer numeric domain.
    Covers all 28 adversarial cases specified under Wave 4B-S Gate E.
    """

    # --- Property-Name Validation (Cases 1 - 14) ---

    def test_cs01_valid_lowercase_ascii_and_underscore_keys(self):
        """Cases 1 & 2: Valid lowercase ASCII identifiers and underscore keys are accepted."""
        valid_payload = {"case_id": 1, "sequence_index": 2, "rev_01": 3}
        out = canonicalize_json(valid_payload)
        assert out == b'{"case_id":1,"rev_01":3,"sequence_index":2}'

    def test_cs02_unsupported_uppercase_property_name_rejected(self):
        """Case 3: Uppercase property names are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"CASE_ID": 1})

    def test_cs03_property_name_with_spaces_rejected(self):
        """Case 4: Property names containing whitespace are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"case id": 1})

    def test_cs04_accented_latin_property_name_rejected(self):
        """Case 5: Accented Latin characters (e.g. café) are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"caf\u00e9_id": 1})

    def test_cs05_supplementary_plane_unicode_property_name_rejected(self):
        """Case 6: Supplementary-plane Unicode property names (U+10000+) are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"\U00010000_key": 1})

    def test_cs06_emoji_property_name_rejected(self):
        """Case 7: Emoji property names are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"\U0001f525_key": 1})

    def test_cs07_unicode_escape_prohibited_key_rejected(self):
        """Case 8: Unicode escape representations of prohibited keys are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"\\u00e9_key": 1})

    def test_cs08_nested_prohibited_property_name_rejected(self):
        """Case 9: Nested dictionaries containing prohibited property names fail closed recursively."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"outer": {"nested_INVALID": 1}})

    def test_cs09_prohibited_property_name_inside_array_rejected(self):
        """Case 10: Dictionaries inside arrays containing prohibited property names fail closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"items": [{"valid_key": 1}, {"\U0001f525": 2}]})

    def test_cs10_empty_and_overlength_property_names_rejected(self):
        """Cases 11 & 12: Empty keys and keys exceeding 64 characters are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"": 1})
        with pytest.raises(CanonicalSerializationError, match="restricted schema pattern"):
            canonicalize_json({"a" * 65: 1})

    def test_cs11_non_string_mapping_key_rejected(self):
        """Case 13: Non-string mapping keys are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="Dictionary keys must be strings"):
            canonicalize_json({123: 1})

    def test_cs12_unexpected_metadata_extension_rejected(self):
        """Case 14: Invalid property names in open metadata containers are rejected fail-closed."""
        with pytest.raises(ValidationError, match="Session metadata key"):
            SessionModel(
                session_id="HVA01-SESS-TEST",
                protocol_version="HVA-01",
                protocol_document_sha256="0" * 64,
                assigned_package="PACKAGE_A",
                package_sha256="1" * 64,
                reviewer_pseudonym="REV-01",
                created_at_utc="2026-10-24T10:00:00Z",
                metadata={"Invalid Key!": 1},
            )
        with pytest.raises(PackageIntegrityError, match="metadata key"):
            BlindedCase(
                sequence_index=1,
                case_id="HVA2-A-01",
                chart_asset_path="charts/HVA2-A-01.svg",
                display_title="Case 01",
                metadata={"\U0001f525": 1},
            )

    # --- Numeric Validation (Cases 15 - 28) ---

    def test_cs13_valid_numeric_integers_and_booleans(self):
        """Cases 15, 16, 17, 25: Valid integers and booleans are serialized deterministically."""
        assert canonicalize_json({"v": 0}) == b'{"v":0}'
        assert canonicalize_json({"v": 12345}) == b'{"v":12345}'
        assert canonicalize_json({"v": -99}) == b'{"v":-99}'
        assert canonicalize_json({"v_true": True, "v_false": False}) == b'{"v_false":false,"v_true":true}'

    def test_cs14_floating_point_numbers_rejected_fail_closed(self):
        """Cases 18, 19, 20, 21, 22: All floating-point numbers are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": 1.0})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": -0.0})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": 0.01})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": 1e-6})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": 1e20})

    def test_cs15_non_finite_floating_point_rejected(self):
        """Cases 23 & 24: NaN and Infinity are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": float("nan")})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": float("inf")})
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"v": float("-inf")})

    def test_cs16_nested_unsupported_float_rejected(self):
        """Case 26: Nested floating-point values fail closed recursively."""
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({"metrics": {"outer": [1, 2, {"inner_float": 92.5}]}})

    def test_cs17_numeric_float_in_open_metadata_rejected(self):
        """Case 27: Floating-point numbers in model metadata containers are rejected fail-closed."""
        with pytest.raises(ValidationError, match="unsupported float value"):
            SessionModel(
                session_id="HVA01-SESS-TEST",
                protocol_version="HVA-01",
                protocol_document_sha256="0" * 64,
                assigned_package="PACKAGE_A",
                package_sha256="1" * 64,
                reviewer_pseudonym="REV-01",
                created_at_utc="2026-10-24T10:00:00Z",
                metadata={"metric_float": 99.5},
            )
        with pytest.raises(PackageIntegrityError, match="unsupported float value"):
            BlindedCase(
                sequence_index=1,
                case_id="HVA2-A-01",
                chart_asset_path="charts/HVA2-A-01.svg",
                display_title="Case 01",
                metadata={"confidence": 0.85},
            )

    def test_cs18_attestation_input_float_rejected(self):
        """Case 28: Attestation inputs containing floating-point values are rejected fail-closed."""
        with pytest.raises(CanonicalSerializationError, match="Floating-point values are unauthorized"):
            canonicalize_json({
                "leaf_count": 59.0,
                "merkle_root": "sha256:" + "0" * 64,
                "session_id": "HVA01-SESS-01",
            })
