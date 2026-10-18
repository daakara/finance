"""
tests/test_registration_execution_wrapper.py

Unit tests for controlled identity registration execution wrapper and SQLite backup API.
Uses strictly synthetic test fixtures; zero real-candidate application.

Selected by Tier 2 Path-Filtered Gates: tier2c.
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionAuthorization,
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
    ProposedAssociationAdoption,
    ProposedEntityAdoption,
    AdoptionPlanSummary,
)
from analyst_dashboard.universe.identity_persistence import (
    IdentityApplicationStatus,
    InternalSecurityRecord,
    SqliteIdentityRepository,
)
from scripts.apply_registration_plan_wrapper import (
    ApprovalManifestError,
    BackupError,
    IdentityAlreadyAppliedError,
    create_verified_sqlite_backup,
    execute_controlled_registration,
    verify_explicit_candidate_bindings,
)

pytestmark = pytest.mark.tier2c


def _build_synthetic_unapproved_plan() -> IdentityAdoptionPlan:
    """Builds an isolated synthetic unapproved plan fixture."""
    entities = [
        ProposedEntityAdoption(
            provisional_security_id="PROV_SEC_SYNTH_01",
            primary_symbol="SYNTH_A",
            share_class=None,
            issuer_cik="0009999991",
            status=IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_SYNTH_DURABLE_01",
            rationale="Synthetic test entity unapproved.",
        ),
        ProposedEntityAdoption(
            provisional_security_id="PROV_SEC_SYNTH_02",
            primary_symbol="SYNTH_B",
            share_class="CLASS_B",
            issuer_cik="0009999992",
            status=IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_security_id="ARX_SEC_SYNTH_DURABLE_02",
            rationale="Synthetic test entity unapproved.",
        ),
    ]

    associations = [
        ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_SYNTH_01",
            provisional_security_id="PROV_SEC_SYNTH_01",
            target_durable_security_id="ARX_SEC_SYNTH_DURABLE_01",
            symbol="SYNTH_A",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_ARX_SEC_SYNTH_DURABLE_01_XNAS_SYNTH_A_2026-10-17",
            conflict_reason="EXACT_PLAN_APPROVAL_ABSENT",
        ),
        ProposedAssociationAdoption(
            provisional_association_id="PROV_ASSOC_SYNTH_02",
            provisional_security_id="PROV_SEC_SYNTH_02",
            target_durable_security_id="ARX_SEC_SYNTH_DURABLE_02",
            symbol="SYNTH_B",
            operating_mic="XNAS",
            effective_from="2026-10-17",
            effective_to=None,
            status=IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED,
            is_creation=True,
            is_mechanically_plannable=True,
            proposed_durable_association_id="ASSOC_ARX_SEC_SYNTH_DURABLE_02_XNAS_SYNTH_B_2026-10-17",
            conflict_reason="EXACT_PLAN_APPROVAL_ABSENT",
        ),
    ]

    summary = AdoptionPlanSummary(
        total_proposals=2,
        entities_matched_existing=0,
        entities_approved_creation=0,
        entities_blocked=0,
        associations_matched_existing=0,
        associations_approved_creation=0,
        associations_blocked=0,
        conflicts_count=0,
        entities_mechanically_ready=2,
        associations_mechanically_ready=2,
    )

    prov_map = {
        "PROV_SEC_SYNTH_01": "ARX_SEC_SYNTH_DURABLE_01",
        "PROV_SEC_SYNTH_02": "ARX_SEC_SYNTH_DURABLE_02",
    }

    content_digest = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
        entities=entities,
        associations=associations,
        provisional_to_durable_map=prov_map,
        summary=summary,
    )

    auth = IdentityAdoptionAuthorization(
        is_authorized=False,
        authorized_by=None,
        scope="REFERENCE_IDENTITY_REGISTRATION",
        exact_plan_approval="ABSENT",
        application_authorized=False,
    )

    return IdentityAdoptionPlan(
        plan_id="PLAN_ADOPT_SYNTH_TEST_001",
        created_at_utc="2026-10-18T16:00:00Z",
        evaluation_reference_date="2026-10-17",
        target_state_digest="ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee",
        plan_content_digest=content_digest,
        authorization=auth,
        entities=entities,
        associations=associations,
        provisional_to_durable_map=prov_map,
        conflicts=[],
        blocked_operations=[],
        missing_evidence_or_authorization=[],
        primary_listing_states={},
        summary=summary,
    )


class TestSqliteConsistentBackupApi:
    """Verifies SQLite backup API behavior on WAL databases with transactions."""

    def test_backup_creates_consistent_independent_copy_in_wal_mode(self, tmp_path):
        src_path = str(tmp_path / "src.db")
        bck_path = str(tmp_path / "bck.db")

        # 1. Initialize repository with WAL mode
        repo = SqliteIdentityRepository(src_path, auto_init=True)
        with repo.transaction() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            sec = InternalSecurityRecord(
                internal_security_id="ARX_SEC_SYNTH_99",
                issuer_cik="0009999999",
                primary_symbol="SYNTH99",
                share_class=None,
                created_at_utc="2026-10-18T16:00:00Z",
                status="ACTIVE",
                continuity_basis="TEST",
                certifies_continuity=False,
            )
            repo.save_security(sec, conn=conn)

        src_digest = repo.compute_target_state_digest()

        # 2. Perform online backup using supported SQLite backup API
        bck_digest = create_verified_sqlite_backup(src_path, bck_path)

        # 3. Verify backup is consistent and can be read independently
        assert bck_digest == src_digest
        with SqliteIdentityRepository(bck_path, read_only=True) as bck_repo:
            secs = bck_repo.list_all_securities()
            assert len(secs) == 1
            assert secs[0].internal_security_id == "ARX_SEC_SYNTH_99"
            assert bck_repo.compute_target_state_digest() == src_digest

    def test_backup_fails_if_source_does_not_exist(self, tmp_path):
        nonexistent = str(tmp_path / "nonexistent.db")
        bck_path = str(tmp_path / "bck.db")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            create_verified_sqlite_backup(nonexistent, bck_path)

    def test_backup_refuses_to_overwrite_existing_file(self, tmp_path):
        src_path = str(tmp_path / "src.db")
        bck_path = str(tmp_path / "bck.db")
        SqliteIdentityRepository(src_path, auto_init=True)

        # Pre-create the destination backup file
        Path(bck_path).write_text("prior_backup_data", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            create_verified_sqlite_backup(src_path, bck_path)

        # Ensure prior file was not clobbered
        assert Path(bck_path).read_text(encoding="utf-8") == "prior_backup_data"

    def test_backup_fails_if_schema_incomplete(self, tmp_path):
        import sqlite3
        src_path = str(tmp_path / "plain_sqlite.db")
        bck_path = str(tmp_path / "bck.db")

        # Create a raw SQLite DB without required identity tables
        conn = sqlite3.connect(src_path)
        conn.execute("CREATE TABLE some_random_table (id INT);")
        conn.commit()
        conn.close()

        with pytest.raises(BackupError, match="BACKUP_SCHEMA_INCOMPLETE"):
            create_verified_sqlite_backup(src_path, bck_path)


class TestRegistrationExecutionWrapperGovernance:
    """Verifies pre-execution gates, explicit bindings, and fail-closed behaviors of wrapper."""

    def test_wrapper_rejects_missing_approval_manifest(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(plan.to_json(), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="APPROVAL_MANIFEST_MISSING"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(tmp_path / "missing_manifest.json"),
                db_path=str(tmp_path / "identity.db"),
            )

    def test_wrapper_rejects_missing_or_incomplete_manifest_keys(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(plan.to_json(), encoding="utf-8")

        # Incomplete manifest missing 'authorized_by'
        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": "some_sha",
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": str(tmp_path / "identity.db"),
            "expected_target_state_digest": plan.target_state_digest,
            # authorized_by missing!
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ApprovalManifestError, match="Missing required key 'authorized_by'"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "identity.db"),
            )

    def test_wrapper_rejects_unsupported_scope(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(plan.to_json(), encoding="utf-8")

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": "some_sha",
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": str(tmp_path / "identity.db"),
            "expected_target_state_digest": plan.target_state_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "UNIVERSE_PUBLICATION",  # Unauthorized scope!
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ApprovalManifestError, match="UNSUPPORTED_SCOPE"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "identity.db"),
            )

    def test_wrapper_rejects_plan_file_hash_tampering(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(plan.to_json(), encoding="utf-8")

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": str(tmp_path / "identity.db"),
            "expected_target_state_digest": plan.target_state_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="PLAN_FILE_TAMPERED"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "identity.db"),
            )

    def test_wrapper_rejects_unapproved_target_path(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": str(tmp_path / "approved_target.db"),
            "expected_target_state_digest": plan.target_state_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="TARGET_PATH_UNAPPROVED"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "unapproved_target.db"),
            )

    def test_wrapper_rejects_absent_target_database(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": plan.target_state_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="TARGET_DATABASE_ABSENT"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
            )

    def test_wrapper_rejects_target_state_digest_mismatch(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        repo = SqliteIdentityRepository(db_path, auto_init=True)
        # Modify DB state
        with repo.transaction() as conn:
            sec = InternalSecurityRecord(
                internal_security_id="ARX_SEC_MUTATED",
                issuer_cik="0009999999",
                primary_symbol="MUTATED",
                share_class=None,
                created_at_utc="2026-10-18T16:00:00Z",
                status="ACTIVE",
                continuity_basis="TEST",
                certifies_continuity=False,
            )
            repo.save_security(sec, conn=conn)

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee",
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="TARGET_STATE_DIGEST_MISMATCH"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
            )

    def test_wrapper_rejects_swapped_or_mismatched_candidate_bindings(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        # Intentionally swapped durable IDs in bindings
        swapped_bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",  # Swapped!
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",  # Swapped!
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": swapped_bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="CANDIDATE_BINDING_MISMATCH"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
            )

    def test_wrapper_rejects_unexpected_entity_action_state(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        # Mutate entity status to BLOCKED in reviewed plan
        mutated_entities = list(plan.entities)
        mutated_entities[0] = dataclasses.replace(
            mutated_entities[0],
            status=IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED,
        )
        plan_mut = dataclasses.replace(
            plan,
            entities=mutated_entities,
            plan_content_digest=IdentityAdoptionDryRunEngine.compute_plan_content_digest(
                entities=mutated_entities,
                associations=plan.associations,
                provisional_to_durable_map=plan.provisional_to_durable_map,
                summary=plan.summary,
            ),
        )

        plan_file = tmp_path / "plan.json"
        plan_bytes = plan_mut.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan_mut.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan_mut.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="UNEXPECTED_ACTION_STATE"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
            )

    def test_wrapper_rejects_unexpected_association_conflict_reason(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        # Mutate association conflict reason
        mutated_assocs = list(plan.associations)
        mutated_assocs[0] = dataclasses.replace(
            mutated_assocs[0],
            conflict_reason="UNEXPECTED_CONFLICT",
        )
        plan_mut = dataclasses.replace(
            plan,
            associations=mutated_assocs,
            plan_content_digest=IdentityAdoptionDryRunEngine.compute_plan_content_digest(
                entities=plan.entities,
                associations=mutated_assocs,
                provisional_to_durable_map=plan.provisional_to_durable_map,
                summary=plan.summary,
            ),
        )

        plan_file = tmp_path / "plan.json"
        plan_bytes = plan_mut.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan_mut.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan_mut.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="UNEXPECTED_ACTION_STATE"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
            )

    def test_wrapper_successful_application_and_readback(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "preserved_applied_plan.json")

        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        correct_bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": correct_bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        outcome = execute_controlled_registration(
            reviewed_plan_path=str(plan_file),
            approval_manifest_path=str(manifest_file),
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )

        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert outcome.applied_entities_count == 2
        assert outcome.applied_associations_count == 2
        assert outcome.target_state_digest_before == init_digest
        assert outcome.target_state_digest_after != init_digest

        # Verify applied plan artifact is preserved on disk
        assert os.path.exists(applied_plan_out)
        preserved_plan = json.loads(Path(applied_plan_out).read_text(encoding="utf-8"))
        assert preserved_plan["authorization"]["exact_plan_approval"] == "APPROVED"
        assert preserved_plan["summary"]["entities_approved_creation"] == 2

        # Verify pre-apply backup is preserved and intact
        assert os.path.exists(backup_path)
        with SqliteIdentityRepository(backup_path, read_only=True) as bck_repo:
            assert bck_repo.compute_target_state_digest() == init_digest
            assert len(bck_repo.list_all_securities()) == 0

        # Verify database readback
        with SqliteIdentityRepository(db_path, read_only=True) as verified_repo:
            secs = verified_repo.list_all_securities()
            assocs = verified_repo.list_all_associations()
            assert len(secs) == 2
            assert len(assocs) == 2
            sec_ids = {s.internal_security_id for s in secs}
            assert sec_ids == {"ARX_SEC_SYNTH_DURABLE_01", "ARX_SEC_SYNTH_DURABLE_02"}

    def test_wrapper_retry_detects_already_applied_without_clobbering_backup(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "preserved_applied_plan.json")

        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        correct_bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": correct_bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        # 1. First run succeeds
        outcome1 = execute_controlled_registration(
            reviewed_plan_path=str(plan_file),
            approval_manifest_path=str(manifest_file),
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )
        assert outcome1.status == IdentityApplicationStatus.APPLIED
        backup_stat_before = os.stat(backup_path)

        # 2. Retry run immediately detects ALREADY_APPLIED
        with pytest.raises(IdentityAlreadyAppliedError, match="ALREADY_APPLIED"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_out,
            )

        # 3. Verify backup was NOT clobbered or touched on retry
        backup_stat_after = os.stat(backup_path)
        assert backup_stat_before.st_mtime_ns == backup_stat_after.st_mtime_ns
        with SqliteIdentityRepository(backup_path, read_only=True) as bck_repo:
            assert bck_repo.compute_target_state_digest() == init_digest
            assert len(bck_repo.list_all_securities()) == 0

        # 4. Verify no duplicated records in DB
        with SqliteIdentityRepository(db_path, read_only=True) as verified_repo:
            assert len(verified_repo.list_all_securities()) == 2
            assert len(verified_repo.list_all_associations()) == 2

    def test_wrapper_rejects_existing_output_artifact(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "preserved_applied_plan.json")
        # Pre-create output artifact
        Path(applied_plan_out).write_text("prior_artifact", encoding="utf-8")

        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(FileExistsError, match="APPLIED_ARTIFACT_EXISTS"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_out,
            )

    def test_wrapper_handles_post_commit_artifact_write_failure_as_committed(self, tmp_path):
        plan = _build_synthetic_unapproved_plan()
        plan_file = tmp_path / "plan.json"
        plan_bytes = plan.to_json().encode("utf-8")
        plan_file.write_bytes(plan_bytes)

        db_path = str(tmp_path / "identity.db")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "preserved_applied_plan.json")

        repo = SqliteIdentityRepository(db_path, auto_init=True)
        init_digest = repo.compute_target_state_digest()

        bindings = {
            "SYNTH_A": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_01",
                "issuer_cik": "0009999991",
                "share_class": None,
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
            "SYNTH_B": {
                "durable_security_id": "ARX_SEC_SYNTH_DURABLE_02",
                "issuer_cik": "0009999992",
                "share_class": "CLASS_B",
                "operating_mic": "XNAS",
                "effective_from": "2026-10-17",
            },
        }

        manifest = {
            "approval_id": "APPR_SYNTH_001",
            "reviewed_plan_id": plan.plan_id,
            "reviewed_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "reviewed_plan_content_digest": plan.plan_content_digest,
            "approved_target_db_path": db_path,
            "expected_target_state_digest": init_digest,
            "authorized_by": "SYNTH_OPERATOR",
            "authorization_timestamp_utc": "2026-10-18T16:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": bindings,
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        # Mock Path.write_text to fail only for the output artifact write
        real_write_text = Path.write_text

        def mock_write(path_self, *args, **kwargs):
            if str(path_self) == applied_plan_out:
                raise OSError("Disk quota exceeded on artifact storage")
            return real_write_text(path_self, *args, **kwargs)

        with patch.object(Path, "write_text", side_effect=mock_write):
            outcome = execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=str(manifest_file),
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_out,
            )

        # Must report as COMMITTED (APPLIED), not falsely rolled back
        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert any("POST_COMMIT_ARTIFACT_WRITE_FAILED" in d for d in outcome.details)

        # Database rows must be committed
        with SqliteIdentityRepository(db_path, read_only=True) as verified_repo:
            assert len(verified_repo.list_all_securities()) == 2
            assert len(verified_repo.list_all_associations()) == 2
