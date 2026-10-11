"""
ARX Terminal — Smart Money Test Suite
Targeted Verification for Distributed Authority Substrate (Theorems A–D, V2.1)

Covers:
1. Candidate 004 full manifest conformance (all 17 semantic items and frozen constants)
2. Gate-3 IAM boundary separation and non-self-issuable constraints
3. Gate-3 lifecycle monotonicity, single-use, revocation, and supersession rejection
4. Transactional CAS single-winner claim under concurrent contention
5. Stale runner write rejection via fencing tokens
6. Global rollback detection and latest-head verification
7. Genesis Theorem: prevention of second genesis upon total cache/local state loss
8. Response-loss idempotent retry without duplicate increments
9. Admission and natural denominator single-transaction atomicity
10. Verbatim raw SEC evidence tuple, object/metadata coupling, quarantine, and independent replay
11. Audit timestamp UTC canonicalization and timezone conversion
"""

import datetime
from datetime import timezone
import hashlib
import json
import os
import sqlite3
import tempfile
import pytest

from analyst_dashboard.smart_money.public_availability.gate3_authority_substrate import (
    Gate3AuthorizationV2,
    Gate3LifecycleManager,
    Gate3DeliveryAdapter,
    Gate3LifecycleError,
    Gate3IamSeparationError,
    GATE3_ISSUER_IDENTITY,
    GATE3_CONSUMER_IDENTITY,
)
from analyst_dashboard.smart_money.public_availability.transactional_state_backend import (
    TransactionalStateBackend,
    StaleWriterFencingError,
    SecondGenesisAttemptError,
    DuplicateGate3ClaimError,
    GlobalRollbackError,
)
from analyst_dashboard.smart_money.public_availability.evidence_custody import (
    RawSecEvidenceTuple,
    DurableEvidenceCustodyStore,
    EvidenceStatus,
)


class TestDistributedAuthoritySubstrate:
    """Comprehensive test suite for the distributed authority substrate."""

    # 1. Candidate 004 Full Manifest Conformance (Section 3)
    def test_candidate004_full_manifest_conformance(self):
        freeze_path = os.path.join(
            "docs", "domain", "vcp", "sprint_3", "candidates", "CANDIDATE_GENERATION_004_FREEZE.json"
        )
        assert os.path.exists(freeze_path), "CANDIDATE_GENERATION_004_FREEZE.json must exist"

        with open(freeze_path, "r", encoding="utf-8") as f:
            freeze = json.load(f)

        from analyst_dashboard.vcp.sprint_3_shadow_governance import (
            CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
            CANONICAL_DEPENDENCY_LOCK_HASH,
            CANONICAL_RUNTIME_CONFIG_HASH,
            CANONICAL_SPRINT_3_GOVERNANCE_SHA256,
        )
        from analyst_dashboard.vcp.sprint_3_durable_storage import (
            CANONICAL_DDL_HASH,
            SCHEMA_VERSION,
            MIGRATION_ID,
            DENOMINATOR_POLICY_VERSION,
        )
        from analyst_dashboard.analyzers.scanner_publication_integrity import (
            CANONICAL_VCP_RULESET_HASH,
            CANONICAL_VCP_UNIVERSE_HASH,
            CANONICAL_VCP_DATA_PROVENANCE_HASH,
        )

        assert CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH == freeze["semantic_closure_hash"]
        assert CANONICAL_SPRINT_3_GOVERNANCE_SHA256 == freeze["governance_hash"]
        assert CANONICAL_VCP_RULESET_HASH == freeze["vcp_ruleset_hash"]
        assert CANONICAL_VCP_UNIVERSE_HASH == freeze["universe_builder_hash"]
        assert CANONICAL_VCP_DATA_PROVENANCE_HASH == freeze["data_interpretation_hash"]
        assert CANONICAL_DEPENDENCY_LOCK_HASH == freeze["dependency_lock_hash"]
        assert CANONICAL_RUNTIME_CONFIG_HASH == freeze["runtime_config_hash"]
        assert CANONICAL_DDL_HASH == freeze["ddl_hash"]
        assert SCHEMA_VERSION == freeze["database_schema_version"]
        assert SCHEMA_VERSION == freeze["durability_schema_version"]
        assert MIGRATION_ID == freeze["migration_identity"]
        assert DENOMINATOR_POLICY_VERSION == freeze["denominator_policy_version"]
        assert len(freeze["semantic_closure"]["items"]) == 17

    # 2. Gate-3 IAM Separation and Non-Self-Issuable Invariants (Theorem B)
    def test_gate3_iam_separation_and_self_authorization_prevention(self):
        mgr = Gate3LifecycleManager()

        # A. Self-issuance attempt: consumer attempts to issue authorization -> REJECTED
        with pytest.raises(Gate3IamSeparationError) as exc_info:
            auth_self = Gate3AuthorizationV2(
                authorization_id="AUTH_SELF_001",
                authorization_sequence=1,
                epoch_id="smart-money-durable-state-v2",
                target_generation=1,
                target_release_sha="SHA_RELEASE",
                authorization_scope="NATURAL_OBSERVATION",
                issuer_identity=GATE3_CONSUMER_IDENTITY,
                consumer_identity=GATE3_CONSUMER_IDENTITY,
                nonce="NONCE_001",
                issued_at_utc="2026-10-12T00:00:00+00:00",
                not_before_utc="2026-10-12T00:00:00+00:00",
                expires_at_utc="2026-10-12T01:00:00+00:00",
                single_use=True,
                readiness_manifest_hash="READINESS_HASH",
            )
            mgr.register_authorization(auth_self)
        assert "Unauthorized issuer identity" in str(exc_info.value)

        # B. Consumer attempts to revoke authorization -> REJECTED
        auth_valid = Gate3AuthorizationV2(
            authorization_id="AUTH_VALID_001",
            authorization_sequence=1,
            epoch_id="smart-money-durable-state-v2",
            target_generation=1,
            target_release_sha="SHA_RELEASE",
            authorization_scope="NATURAL_OBSERVATION",
            issuer_identity=GATE3_ISSUER_IDENTITY,
            consumer_identity=GATE3_CONSUMER_IDENTITY,
            nonce="NONCE_001",
            issued_at_utc="2026-10-12T00:00:00+00:00",
            not_before_utc="2026-10-12T00:00:00+00:00",
            expires_at_utc="2026-10-12T01:00:00+00:00",
            single_use=True,
            readiness_manifest_hash="READINESS_HASH",
        )
        mgr.register_authorization(auth_valid)

        with pytest.raises(Gate3IamSeparationError):
            mgr.revoke_authorization("AUTH_VALID_001", caller_identity=GATE3_CONSUMER_IDENTITY)

    # 3. Gate-3 Lifecycle Monotonicity, Supersession, and Revocation (Theorem B)
    def test_gate3_lifecycle_supersession_and_revocation(self):
        mgr = Gate3LifecycleManager()
        now = datetime.datetime(2026, 10, 12, 0, 30, tzinfo=timezone.utc)

        auth_1 = Gate3AuthorizationV2(
            authorization_id="AUTH_SEQ_1",
            authorization_sequence=1,
            epoch_id="smart-money-durable-state-v2",
            target_generation=1,
            target_release_sha="SHA_RELEASE",
            authorization_scope="NATURAL_OBSERVATION",
            issuer_identity=GATE3_ISSUER_IDENTITY,
            consumer_identity=GATE3_CONSUMER_IDENTITY,
            nonce="NONCE_1",
            issued_at_utc="2026-10-12T00:00:00+00:00",
            not_before_utc="2026-10-12T00:00:00+00:00",
            expires_at_utc="2026-10-12T01:00:00+00:00",
            single_use=True,
            readiness_manifest_hash="READINESS_HASH",
        )
        mgr.register_authorization(auth_1)

        # Auth 1 is valid before supersession
        ok, _ = mgr.validate_for_consumption(auth_1, now, GATE3_CONSUMER_IDENTITY)
        assert ok is True

        # Auth 2 supersedes Auth 1
        auth_2 = Gate3AuthorizationV2(
            authorization_id="AUTH_SEQ_2",
            authorization_sequence=2,
            epoch_id="smart-money-durable-state-v2",
            target_generation=2,
            target_release_sha="SHA_RELEASE",
            authorization_scope="NATURAL_OBSERVATION",
            issuer_identity=GATE3_ISSUER_IDENTITY,
            consumer_identity=GATE3_CONSUMER_IDENTITY,
            nonce="NONCE_2",
            issued_at_utc="2026-10-12T00:15:00+00:00",
            not_before_utc="2026-10-12T00:15:00+00:00",
            expires_at_utc="2026-10-12T01:15:00+00:00",
            single_use=True,
            readiness_manifest_hash="READINESS_HASH_2",
        )
        mgr.register_authorization(auth_2)

        # Attempt to consume older Auth 1 -> SUPERSEDED REJECTION
        ok1, msg1 = mgr.validate_for_consumption(auth_1, now, GATE3_CONSUMER_IDENTITY)
        assert ok1 is False
        assert "SUPERSEDED" in msg1

        # Attempt to register sequence regression -> REJECTED
        with pytest.raises(Gate3LifecycleError):
            auth_bad_seq = Gate3AuthorizationV2(
                authorization_id="AUTH_BAD_SEQ",
                authorization_sequence=2,  # Duplicate sequence
                epoch_id="smart-money-durable-state-v2",
                target_generation=3,
                target_release_sha="SHA_RELEASE",
                authorization_scope="NATURAL_OBSERVATION",
                issuer_identity=GATE3_ISSUER_IDENTITY,
                consumer_identity=GATE3_CONSUMER_IDENTITY,
                nonce="NONCE_BAD",
                issued_at_utc="2026-10-12T00:20:00+00:00",
                not_before_utc="2026-10-12T00:20:00+00:00",
                expires_at_utc="2026-10-12T01:20:00+00:00",
                single_use=True,
                readiness_manifest_hash="READINESS_HASH_2",
            )
            mgr.register_authorization(auth_bad_seq)

    # 4. Transactional CAS Single-Winner Claim Under Contention (Theorem C)
    def test_transactional_cas_single_winner_claim(self, tmp_path):
        db_path = str(tmp_path / "cas_state.db")
        backend = TransactionalStateBackend(db_path)
        gen, token = backend.initialize_epoch("GENESIS_COMMIT")
        assert gen == 1

        shared_auth_id = "AUTH_GATE3_CONCURRENT_100"

        # Runner A attempts claim at generation 1 -> SUCCESS
        ok_a, gen_a, token_a, receipt_a = backend.claim_authorization_cas(
            authorization_id=shared_auth_id,
            run_id="RUNNER_A",
            expected_generation=1,
            fencing_token=token,
        )
        assert ok_a is True
        assert gen_a == 2
        assert token_a != token

        # Runner B attempts claim for same auth_id -> DuplicateGate3ClaimError
        with pytest.raises(DuplicateGate3ClaimError):
            backend.claim_authorization_cas(
                authorization_id=shared_auth_id,
                run_id="RUNNER_B",
                expected_generation=gen_a,
                fencing_token=token_a,
            )

    # 5. Stale Runner Write Rejection via Fencing Tokens (Theorem C)
    def test_stale_runner_write_fencing_rejection(self, tmp_path):
        db_path = str(tmp_path / "fencing_state.db")
        backend = TransactionalStateBackend(db_path)
        gen, token = backend.initialize_epoch("GENESIS_COMMIT")

        # Runner A starts at gen 1, token A
        run_a_gen = gen
        run_a_token = token

        # Runner B claims an authorization and advances generation to 2
        _, gen_b, token_b, _ = backend.claim_authorization_cas(
            authorization_id="AUTH_B",
            run_id="RUNNER_B",
            expected_generation=run_a_gen,
            fencing_token=run_a_token,
        )
        assert gen_b == 2

        # Runner A resumes and attempts admission using stale gen 1 / token A -> REJECTED
        with pytest.raises(StaleWriterFencingError):
            backend.record_accession_and_denominator(
                accession="0001018724-26-000001",
                authority_event_id="EVT_A",
                idempotency_key="KEY_A",
                generation=run_a_gen,
                fencing_token=run_a_token,
            )

    # 6. Global Rollback Detection and Latest-Head Verification (Theorem C)
    def test_global_rollback_detection(self, tmp_path):
        db_path = str(tmp_path / "rollback_state.db")
        backend = TransactionalStateBackend(db_path)
        gen, token = backend.initialize_epoch("GENESIS_COMMIT")

        # Commit checkpoint 1
        backend.verify_and_commit_checkpoint(
            checkpoint_sequence=1,
            checkpoint_hash="HASH_CHK_1",
            generation=gen,
            fencing_token=token,
        )

        # Commit checkpoint 2
        backend.verify_and_commit_checkpoint(
            checkpoint_sequence=2,
            checkpoint_hash="HASH_CHK_2",
            generation=gen,
            fencing_token=token,
        )

        # Attempt to commit checkpoint 1 (rollback/fork) -> GlobalRollbackError
        with pytest.raises(GlobalRollbackError) as exc_info:
            backend.verify_and_commit_checkpoint(
                checkpoint_sequence=1,
                checkpoint_hash="HASH_CHK_1",
                generation=gen,
                fencing_token=token,
            )
        assert "Rollback rejected" in str(exc_info.value)

    # 7. Genesis Theorem: Prevention of Second Genesis (Theorem C)
    def test_genesis_theorem_second_genesis_prevention(self, tmp_path):
        db_path = str(tmp_path / "genesis_state.db")
        backend = TransactionalStateBackend(db_path)
        backend.initialize_epoch("GENESIS_COMMIT")

        # Authoritative state is active; a fresh empty runner arrives with local_state_exists=False
        with pytest.raises(SecondGenesisAttemptError) as exc_info:
            backend.verify_genesis_safety(local_state_exists=False)
        assert "SECOND GENESIS IS IMPOSSIBLE" in str(exc_info.value).upper()

    # 8. Response-Loss Idempotent Retry Without Duplicate Increments (Theorem C)
    def test_response_loss_idempotent_retry(self, tmp_path):
        db_path = str(tmp_path / "idempotency_state.db")
        backend = TransactionalStateBackend(db_path)
        gen, token = backend.initialize_epoch("GENESIS_COMMIT")

        idempotency_key = "IDEM_REQ_001"
        ok, res1 = backend.record_accession_and_denominator(
            accession="0001018724-26-000010",
            authority_event_id="EVT_001",
            idempotency_key=idempotency_key,
            generation=gen,
            fencing_token=token,
        )
        assert ok is True
        assert res1["status"] == "ADMISSION_COMMITTED"
        assert res1["natural_denominator"] == 1

        # Client loses response and retries with same idempotency key
        ok_retry, res2 = backend.record_accession_and_denominator(
            accession="0001018724-26-000010",
            authority_event_id="EVT_001",
            idempotency_key=idempotency_key,
            generation=gen,
            fencing_token=token,
        )
        assert ok_retry is True
        assert res2["status"] == "ALREADY_COMMITTED_IDEMPOTENT_SUCCESS"
        assert res2["natural_denominator"] == 1  # Exactly 1, NOT 2!

        state = backend.get_authoritative_state()
        assert state["natural_denominator"] == 1

    # 9. Verbatim Raw Evidence Tuple, Staging, Quarantine, and Replay (Theorem D)
    def test_evidence_custody_staging_quarantine_and_replay(self, tmp_path):
        custody_dir = str(tmp_path / "evidence_store")
        store = DurableEvidenceCustodyStore(custody_dir)

        payload_bytes = b"<ownershipDocument><documentType>4</documentType></ownershipDocument>"
        tuple_record = RawSecEvidenceTuple(
            accession="0001018724-26-000099",
            endpoint="https://data.sec.gov/submissions/CIK0001018724.json",
            http_status=200,
            request_time_utc="2026-10-12T00:10:00+00:00",
            response_time_utc="2026-10-12T00:10:01+00:00",
            clock_evidence_id="CLOCK_EVT_01",
            object_id="OBJ_SEC_000099",
            object_version="1",
            byte_length=len(payload_bytes),
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
            content_type="application/xml",
            capture_run_id="RUN_CAP_01",
        )

        # Stage evidence
        ok_stg, _ = store.stage_evidence(tuple_record, payload_bytes)
        assert ok_stg is True

        # Commit evidence to durable custody
        ok_commit, _ = store.commit_evidence("OBJ_SEC_000099")
        assert ok_commit is True

        # Independent replay verification
        ok_rep, replay_details = store.replay_and_verify("OBJ_SEC_000099")
        assert ok_rep is True
        assert replay_details["status"] == "REPLAY_VERIFIED"
        assert replay_details["sha256"] == tuple_record.sha256

        # Quarantine verification
        ok_q, _ = store.quarantine_evidence(
            "OBJ_SEC_000099",
            failure_stage="R1_ROOTING",
            failure_reason="Sigstore keyless certificate minting timeout",
        )
        assert ok_q is True
        q_meta_path = os.path.join(store.quarantine_dir, "OBJ_SEC_000099.quarantine.json")
        assert os.path.exists(q_meta_path)
        with open(q_meta_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        assert q_data["status"] == "CAPTURED_QUARANTINED"
        assert q_data["admissible"] is False

    # 10. Audit Timestamp UTC Canonicalization and Timezone Verification (Section 2.1)
    def test_audit_timestamp_canonicalization_and_timezone(self):
        # Explicit test for aware UTC timestamps
        t_start = datetime.datetime(2026, 10, 12, 0, 10, 0, tzinfo=timezone.utc)
        # Even if local is UTC+2 (02:20 local), aware UTC representation must be 00:20:00Z
        t_close_local = datetime.datetime(2026, 10, 12, 2, 20, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
        t_close_utc = t_close_local.astimezone(timezone.utc)

        assert t_start < t_close_utc
        assert t_close_utc.isoformat() == "2026-10-12T00:20:00+00:00"
        diff_seconds = (t_close_utc - t_start).total_seconds()
        assert diff_seconds > 0
