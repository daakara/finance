"""
tests/test_registration_execution_wrapper.py

Unit tests for controlled identity registration execution wrapper and SQLite backup API.
Uses strictly synthetic test fixtures; zero real-candidate application.

Selected by Tier 2 Path-Filtered Gates: tier2c.
"""

import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    EffectiveDatedListingAssociation,
    IdentityApplicationStatus,
    InternalSecurityRecord,
    SqliteIdentityRepository,
)
from scripts.apply_registration_plan_wrapper import (
    ApprovalManifestError,
    BackupError,
    IdentityAlreadyAppliedError,
    archive_and_verify_audit_bundle,
    create_verified_sqlite_backup,
    diagnose_execution_state,
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


EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence" / "coverage_expansion"


class TestPackageAArtifactTransportAndStagingVerification:
    """
    Unit tests for the concrete Package A artifact transport, staging, and pre-execution
    verification runbook, demonstrating fail-closed behavior in an isolated disposable replica.
    """

    def test_transport_byte_identical_transfer_and_hash_verification(self, tmp_path):
        """Verifies byte-for-byte fidelity across simulated stream/base64 transport."""
        source_plan_path = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        assert source_plan_path.exists(), "Source adoption plan file must exist in evidence directory."

        source_bytes = source_plan_path.read_bytes()
        expected_sha = hashlib.sha256(source_bytes).hexdigest()
        assert expected_sha == "14b7f5edc1e1bbe41b5ed79bcbf367f9052199fc47e02ebb83ae0c0c8b9fcaef"

        # Simulate transport stream via base64 encoding (matching the runbook recommendation)
        transport_stream = base64.b64encode(source_bytes)
        staged_dir = tmp_path / "staging" / "expansion"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_plan_path = staged_dir / "package_a_identity_adoption_plan.json"
        staged_plan_path.write_bytes(base64.b64decode(transport_stream))

        # 1. Byte-level verification
        staged_bytes = staged_plan_path.read_bytes()
        staged_sha = hashlib.sha256(staged_bytes).hexdigest()
        assert staged_sha == expected_sha, "Transferred plan SHA-256 must match source byte-for-byte."

        # 2. Schema and logical digest verification
        plan_data = json.loads(staged_bytes.decode("utf-8"))
        assert plan_data["plan_id"] == "PLAN_ADOPT_9A2E4E04E878"
        assert plan_data["plan_content_digest"] == "9a2e4e04e878b667e25a2a75d56292a0b3e679b15890211cef13db96f7a8bfe5"
        assert plan_data["target_state_digest"] == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"
        assert len(plan_data["entities"]) == 130
        assert len(plan_data["associations"]) == 130

    def test_transport_corrupted_byte_rejected(self, tmp_path):
        """Verifies that a single changed or corrupted byte in transport is detected and rejected."""
        source_plan_path = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        source_bytes = source_plan_path.read_bytes()
        expected_sha = "14b7f5edc1e1bbe41b5ed79bcbf367f9052199fc47e02ebb83ae0c0c8b9fcaef"

        # Corrupt a single byte in the middle of payload
        corrupted_bytes = bytearray(source_bytes)
        corrupted_bytes[1024] ^= 0xFF
        corrupted_sha = hashlib.sha256(corrupted_bytes).hexdigest()

        staged_dir = tmp_path / "staging" / "expansion"
        staged_dir.mkdir(parents=True, exist_ok=True)
        corrupted_plan_path = staged_dir / "package_a_identity_adoption_plan.json"
        corrupted_plan_path.write_bytes(corrupted_bytes)

        # Pre-execution verification fails immediately
        assert corrupted_sha != expected_sha
        assert hashlib.sha256(corrupted_plan_path.read_bytes()).hexdigest() != expected_sha

        # Fails closed if passed to wrapper
        fake_manifest = {
            "approval_id": "APPR_TEST_001",
            "reviewed_plan_id": "PLAN_ADOPT_9A2E4E04E878",
            "reviewed_plan_file_sha256": expected_sha,
            "reviewed_plan_content_digest": "9a2e4e04e878b667e25a2a75d56292a0b3e679b15890211cef13db96f7a8bfe5",
            "approved_target_db_path": str(tmp_path / "absent.db"),
            "expected_target_state_digest": "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7",
            "authorized_by": "TEST_OP",
            "authorization_timestamp_utc": "2026-10-20T08:00:00Z",
            "scope": "REFERENCE_IDENTITY_REGISTRATION",
            "candidate_bindings": {},
        }
        manifest_file = staged_dir / "manifest.json"
        manifest_file.write_text(json.dumps(fake_manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="PLAN_FILE_TAMPERED"):
            execute_controlled_registration(
                reviewed_plan_path=str(corrupted_plan_path),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "absent.db"),
                backup_path=str(tmp_path / "backup.db"),
                output_applied_plan_path=str(tmp_path / "applied.json"),
            )

    def test_transport_missing_artifact_rejected(self, tmp_path):
        """Verifies that missing staged artifacts fail closed before any database connection."""
        absent_plan = tmp_path / "staging" / "expansion" / "absent_plan.json"
        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(tmp_path / "target.db")
        manifest_data["authorized_by"] = "TEST_OP"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        manifest_file = tmp_path / "staging" / "manifest.json"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="REVIEWED_PLAN_MISSING"):
            execute_controlled_registration(
                reviewed_plan_path=str(absent_plan),
                approval_manifest_path=str(manifest_file),
                db_path=str(tmp_path / "target.db"),
                backup_path=str(tmp_path / "backup.db"),
                output_applied_plan_path=str(tmp_path / "applied.json"),
            )

    def test_staging_verification_is_read_only_and_cannot_mutate_database(self, tmp_path):
        """Verifies that pre-execution inspection commands perform zero database writes or schema modifications."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        assert export_file.exists(), "target_identity_state_export.json must exist."
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
            initial_digest = repo.compute_target_state_digest()

        assert initial_digest == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"

        # Record file mtime and size before inspection
        initial_stat = Path(db_path).stat()

        # Execute read-only preflight commands
        with SqliteIdentityRepository(db_path, read_only=True) as check_repo:
            observed_digest = check_repo.compute_target_state_digest()
            securities = check_repo.list_all_securities()
            associations = check_repo.list_all_associations()
            assert observed_digest == initial_digest
            assert len(securities) == 3
            assert len(associations) == 3

        # Target database file must be unmodified
        final_stat = Path(db_path).stat()
        assert final_stat.st_size == initial_stat.st_size
        assert not Path(f"{db_path}.pre_apply.bak").exists()
        assert not Path(f"{db_path}-journal").exists()

    def test_staging_backup_non_clobber_enforced(self, tmp_path):
        """Verifies exclusive backup creation fails closed if a backup already exists."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
            initial_digest = repo.compute_target_state_digest()

        assert initial_digest == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"

        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        # Pre-create the backup destination file
        Path(backup_path).write_text("PRIOR_OPERATOR_BACKUP", encoding="utf-8")

        plan_path = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        template_data = json.loads(template_file.read_text(encoding="utf-8"))
        template_data["approved_target_db_path"] = str(Path(db_path).resolve())
        template_data["authorized_by"] = "TEST_OP"
        template_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        manifest_path = str(tmp_path / "manifest.json")
        Path(manifest_path).write_text(json.dumps(template_data), encoding="utf-8")

        with pytest.raises(FileExistsError, match="BACKUP_DESTINATION_EXISTS"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_path),
                approval_manifest_path=manifest_path,
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=str(tmp_path / "applied.json"),
            )

        assert Path(backup_path).read_text(encoding="utf-8") == "PRIOR_OPERATOR_BACKUP"

    def test_staged_replica_execution_preserves_three_pilot_identities_and_rejects_repeat(self, tmp_path):
        """Rehearses full execution on an isolated replica, verifying pilot preservation and repeat rejection."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
            initial_digest = repo.compute_target_state_digest()

        assert initial_digest == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"

        # Staging path setup
        staged_dir = tmp_path / "staging"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_plan = staged_dir / "package_a_identity_adoption_plan.json"
        staged_plan.write_bytes((EVIDENCE_DIR / "package_a_identity_adoption_plan.json").read_bytes())

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "TEST_REHEARSAL_OPERATOR"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        staged_manifest = staged_dir / "approval_manifest.json"
        staged_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "preserved_applied_plan.json")

        # 1. Clean execution
        outcome = execute_controlled_registration(
            reviewed_plan_path=str(staged_plan),
            approval_manifest_path=str(staged_manifest),
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )

        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert outcome.applied_entities_count == 130
        assert outcome.applied_associations_count == 130
        assert outcome.target_state_digest_before == initial_digest
        assert outcome.target_state_digest_after != initial_digest
        assert Path(backup_path).exists()
        assert Path(applied_plan_out).exists()

        # 2. Verify all 3 pilot identities remain intact
        with SqliteIdentityRepository(db_path, read_only=True) as repo:
            securities = {s.internal_security_id: s for s in repo.list_all_securities()}
            associations = {a.association_id: a for a in repo.list_all_associations()}
            assert len(securities) == 133
            assert len(associations) == 133

            # Check pilot identities explicitly
            for pilot_id in ["ARX_SEC_0B4C728FE0BE", "ARX_SEC_B7EF9ED7A35A", "ARX_SEC_A11A24D442E2"]:
                assert pilot_id in securities, f"Pilot security {pilot_id} must be preserved."
            aapl_sec = securities["ARX_SEC_0B4C728FE0BE"]
            assert aapl_sec.primary_symbol == "AAPL"
            assert aapl_sec.issuer_cik == "0000320193"

            goog_sec = securities["ARX_SEC_A11A24D442E2"]
            assert goog_sec.primary_symbol == "GOOG"
            assert goog_sec.share_class == "CLASS_C"
            assert goog_sec.issuer_cik == "0001652044"

            googl_sec = securities["ARX_SEC_B7EF9ED7A35A"]
            assert googl_sec.primary_symbol == "GOOGL"
            assert googl_sec.share_class == "CLASS_A"
            assert googl_sec.issuer_cik == "0001652044"

        # 3. Repeat execution must be rejected with IdentityAlreadyAppliedError (not a silent no-op)
        with pytest.raises(IdentityAlreadyAppliedError, match="ALREADY_APPLIED"):
            execute_controlled_registration(
                reviewed_plan_path=str(staged_plan),
                approval_manifest_path=str(staged_manifest),
                db_path=db_path,
                backup_path=str(tmp_path / "second.bak"),
                output_applied_plan_path=str(tmp_path / "second.json"),
            )


class TestPackageAExecutionRecoveryAndAuditRetention:
    """Tests for durable audit retention and uncertain-execution diagnostic triage."""

    def test_durable_audit_retention_survives_staging_cleanup(self, tmp_path):
        """Verifies that writing applied plan to durable audit directory survives staging cleanup."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        # Setup target database
        data_dir = tmp_path / "analyst_dashboard" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / "identity.db")

        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
            initial_digest = repo.compute_target_state_digest()

        # Temporary staging directory
        staging_dir = tmp_path / "staging" / "expansion"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_plan = staging_dir / "package_a_identity_adoption_plan.json"
        staged_plan.write_bytes((EVIDENCE_DIR / "package_a_identity_adoption_plan.json").read_bytes())

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "DAVID_AKARA"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        staged_manifest = staging_dir / "approval_manifest.json"
        staged_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

        # Durable audit directory and backup path
        audit_dir = tmp_path / "analyst_dashboard" / "audit" / "identity"
        audit_dir.mkdir(parents=True, exist_ok=True)
        applied_plan_out = str(audit_dir / "applied_plan_PLAN_ADOPT_9A2E4E04E878.json")
        backup_path = str(audit_dir / "identity.db.pre_apply.bak")

        # 1. Execute controlled registration
        outcome = execute_controlled_registration(
            reviewed_plan_path=str(staged_plan),
            approval_manifest_path=str(staged_manifest),
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )

        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert outcome.applied_entities_count == 130
        assert outcome.applied_associations_count == 130
        assert Path(applied_plan_out).exists()
        assert Path(backup_path).exists()

        # Retain all 7 audit artifacts in durable audit directory prior to staging cleanup
        # using the real shared production retention implementation (SIP-06)
        receipt = archive_and_verify_audit_bundle(
            staging_dir=str(staging_dir),
            audit_dir=str(audit_dir),
            db_path=db_path,
            plan_id="PLAN_ADOPT_9A2E4E04E878",
            execution_outcome=outcome,
            backup_path=backup_path,
            clean_staging_on_success=True,
        )
        assert receipt["status"] == "VERIFIED_AND_RETAINED"
        assert receipt["artifacts_verified_count"] == 7
        assert not staging_dir.exists(), "Staging directory must be cleaned up on verified retention"

        # 3. Verify ALL 7 durable audit artifacts survive staging cleanup intact
        required_artifacts = [
            "package_a_identity_adoption_plan.json",
            "package_a_approval_manifest.json",
            "applied_plan_PLAN_ADOPT_9A2E4E04E878.json",
            "execution_summary.json",
            "post_execution_readback.json",
            "target_identity_state_export.json",
            "identity.db.pre_apply.bak",
        ]
        for art_name in required_artifacts:
            art_path = audit_dir / art_name
            assert art_path.exists(), f"Required audit artifact '{art_name}' missing from durable audit directory!"
            assert art_path.stat().st_size > 0, f"Audit artifact '{art_name}' must not be empty!"

        applied_data = json.loads(Path(applied_plan_out).read_text(encoding="utf-8"))
        assert applied_data["plan_id"] == "PLAN_ADOPT_9A2E4E04E878"
        assert len(applied_data["entities"]) == 130
        assert len(applied_data["associations"]) == 130
        assert applied_data["summary"]["entities_approved_creation"] == 130

        assert Path(backup_path).exists(), "Pre-apply backup must survive staging cleanup!"
        with SqliteIdentityRepository(backup_path, read_only=True) as bck_repo:
            assert bck_repo.compute_target_state_digest() == initial_digest
            assert len(bck_repo.list_all_securities()) == 3

        # 4. Target repository state remains intact
        with SqliteIdentityRepository(db_path, read_only=True) as target_repo:
            assert len(target_repo.list_all_securities()) == 133
            assert len(target_repo.list_all_associations()) == 133

    def test_recovery_triage_not_applied_state(self, tmp_path):
        """Verifies diagnostic triage detects clean NOT_APPLIED state on pre-commit abort."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
            initial_digest = repo.compute_target_state_digest()

        # Simulate execution failure before commit by passing invalid scope
        invalid_manifest = tmp_path / "invalid_manifest.json"
        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        m_data = json.loads(template_file.read_text(encoding="utf-8"))
        m_data["scope"] = "INVALID_SCOPE"
        invalid_manifest.write_text(json.dumps(m_data), encoding="utf-8")

        with pytest.raises(ApprovalManifestError):
            execute_controlled_registration(
                reviewed_plan_path=str(EVIDENCE_DIR / "package_a_identity_adoption_plan.json"),
                approval_manifest_path=str(invalid_manifest),
                db_path=db_path,
            )

        # Run diagnostic triage
        diagnosis = diagnose_execution_state(db_path=db_path, plan_id="PLAN_ADOPT_9A2E4E04E878")
        assert diagnosis["triage_state"] == "NOT_APPLIED"
        assert diagnosis["current_digest"] == initial_digest
        assert diagnosis["securities_count"] == 3
        assert diagnosis["associations_count"] == 3
        assert diagnosis["safe_to_retry"] is True

    def test_recovery_triage_applied_and_verified_state(self, tmp_path):
        """Verifies diagnostic triage detects APPLIED_AND_VERIFIED state if post-commit artifact writing fails."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "DAVID_AKARA"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        manifest_path = str(tmp_path / "approval_manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest_data), encoding="utf-8")

        plan_path = str(EVIDENCE_DIR / "package_a_identity_adoption_plan.json")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        unwritable_artifact_path = str(tmp_path / "unwritable_dir" / "applied_plan.json")

        # Execute where output artifact directory does not exist and fails to write
        outcome = execute_controlled_registration(
            reviewed_plan_path=plan_path,
            approval_manifest_path=manifest_path,
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=unwritable_artifact_path,
        )

        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert any("POST_COMMIT_ARTIFACT_WRITE_FAILED" in d for d in outcome.details)

        # R5: In recovery triage, omitting reviewed_plan_path returns MISSING_REVIEWED_PLAN_EVIDENCE
        diag_no_plan = diagnose_execution_state(db_path=db_path, plan_id="PLAN_ADOPT_9A2E4E04E878")
        assert diag_no_plan["triage_state"] == "MISSING_REVIEWED_PLAN_EVIDENCE"
        assert diag_no_plan["safe_to_retry"] is False

        # With reviewed_plan_path provided, diagnosis succeeds as APPLIED_AND_VERIFIED
        diagnosis = diagnose_execution_state(
            db_path=db_path,
            plan_id="PLAN_ADOPT_9A2E4E04E878",
            reviewed_plan_path=plan_path,
        )
        assert diagnosis["triage_state"] == "APPLIED_AND_VERIFIED"
        assert diagnosis["securities_count"] == 133
        assert diagnosis["associations_count"] == 133
        assert diagnosis["safe_to_retry"] is False

        # Attempting to re-run fails closed with IdentityAlreadyAppliedError
        with pytest.raises(IdentityAlreadyAppliedError):
            execute_controlled_registration(
                reviewed_plan_path=plan_path,
                approval_manifest_path=manifest_path,
                db_path=db_path,
                backup_path=str(tmp_path / "second.bak"),
                output_applied_plan_path=str(tmp_path / "second.json"),
            )

    def test_recovery_triage_applied_with_subsequent_audited_writes(self, tmp_path):
        """Verifies diagnostic triage detects APPLIED_WITH_SUBSEQUENT_AUDITED_WRITES state."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "DAVID_AKARA"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        manifest_path = str(tmp_path / "approval_manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest_data), encoding="utf-8")

        plan_path = str(EVIDENCE_DIR / "package_a_identity_adoption_plan.json")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "applied_plan.json")

        outcome = execute_controlled_registration(
            reviewed_plan_path=plan_path,
            approval_manifest_path=manifest_path,
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )
        assert outcome.status == IdentityApplicationStatus.APPLIED

        # Simulate subsequent audited write
        with SqliteIdentityRepository(db_path, auto_init=False) as repo:
            subsequent_sec = InternalSecurityRecord(
                internal_security_id="ARX_SEC_SUBSEQUENT_001",
                primary_symbol="SUBS",
                issuer_cik="0009876543",
                share_class=None,
                created_at_utc="2026-10-21T00:00:00Z",
                status="ACTIVE",
            )
            with repo._get_connection() as conn:
                repo.save_security(subsequent_sec, conn=conn)
                subsequent_digest = repo.compute_target_state_digest(conn=conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO applied_plan_audit_ledger
                    (plan_id, plan_content_digest, applied_at_utc, authorized_by, target_state_digest_before, target_state_digest_after, authorization_scope, entities_applied, associations_applied, summary_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "PLAN_FUTURE_SUBSEQUENT_001",
                        "digest_subsequent_001",
                        "2026-10-21T00:00:00Z",
                        "DAVID_AKARA",
                        outcome.target_state_digest_after,
                        subsequent_digest,
                        "REFERENCE_IDENTITY_REGISTRATION",
                        1,
                        0,
                        json.dumps({"status": "APPLIED"}),
                    ),
                )
                conn.commit()

        diagnosis = diagnose_execution_state(db_path=db_path, reviewed_plan_path=plan_path)
        assert diagnosis["triage_state"] == "APPLIED_WITH_SUBSEQUENT_AUDITED_WRITES"
        assert diagnosis["safe_to_retry"] is False
        assert diagnosis["subsequent_audit_records_count"] == 1
        assert diagnosis["securities_count"] == 134
        assert diagnosis["associations_count"] == 133

    def test_recovery_triage_inconsistent_state(self, tmp_path):
        """Verifies diagnostic triage flags inconsistent or unexpected state and halts."""
        db_path = str(tmp_path / "inconsistent.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            # Insert 1 arbitrary security
            repo.save_security(InternalSecurityRecord(
                internal_security_id="ARX_SEC_ANOMALOUS",
                primary_symbol="ANOM",
                issuer_cik="0001234567",
                share_class=None,
                created_at_utc="2026-10-20T08:00:00Z",
                status="ACTIVE",
            ))

        diagnosis = diagnose_execution_state(db_path=db_path, plan_id="PLAN_ADOPT_9A2E4E04E878")
        assert diagnosis["triage_state"] == "INCONSISTENT_STATE"
        assert diagnosis["safe_to_retry"] is False
        assert diagnosis["securities_count"] == 1

    def test_recovery_triage_rejects_malformed_plan_file(self, tmp_path):
        """SIP-06: Malformed or unparseable plan file fails closed without silent default fallback."""
        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            pass

        bad_plan = tmp_path / "corrupt_plan.json"
        bad_plan.write_text("NOT_VALID_JSON{:::broken", encoding="utf-8")

        diagnosis = diagnose_execution_state(
            db_path=db_path,
            reviewed_plan_path=str(bad_plan),
        )
        assert diagnosis["triage_state"] == "CORRUPT_OR_MALFORMED_PLAN"
        assert diagnosis["safe_to_retry"] is False

    def test_recovery_triage_rejects_tampered_plan_content_digest(self, tmp_path):
        """SIP-06: Tampered candidate CIK with unchanged declared digest fails closed."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        # Create tampered plan: alter first candidate CIK but leave declared plan_content_digest intact
        plan_file = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        plan_dict = json.loads(plan_file.read_text(encoding="utf-8"))
        plan_dict["entities"][0]["issuer_cik"] = "0009999999"  # Tampered CIK

        tampered_plan_path = tmp_path / "tampered_plan.json"
        tampered_plan_path.write_text(json.dumps(plan_dict), encoding="utf-8")

        diagnosis = diagnose_execution_state(
            db_path=db_path,
            reviewed_plan_path=str(tampered_plan_path),
        )
        assert diagnosis["triage_state"] == "PLAN_INTEGRITY_TAMPERED"
        assert diagnosis["safe_to_retry"] is False

    def test_recovery_triage_rejects_altered_pilot_attributes(self, tmp_path):
        """SIP-06: Altered pilot attributes (e.g. CIK modification) fail closed in diagnostic."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                rec = InternalSecurityRecord(**sec_data)
                if rec.primary_symbol == "AAPL":
                    # Alter AAPL CIK
                    rec = dataclasses.replace(rec, issuer_cik="0000000000")
                repo.save_security(rec)
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        diagnosis = diagnose_execution_state(
            db_path=db_path,
            reviewed_plan_path=str(EVIDENCE_DIR / "package_a_identity_adoption_plan.json"),
        )
        assert diagnosis["triage_state"] == "INCONSISTENT_STATE"
        assert diagnosis["pilots_preserved"] is False
        assert diagnosis["safe_to_retry"] is False

    def test_archive_and_verify_preserves_staging_on_missing_artifact(self, tmp_path):
        """SIP-06: Retention failure preserves transient staging directory."""
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "package_a_identity_adoption_plan.json").write_text("{}", encoding="utf-8")
        # Omit other 6 artifacts

        audit_dir = tmp_path / "audit"
        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True):
            pass

        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                clean_staging_on_success=True,
            )

        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in str(exc_info.value)
        assert staging_dir.exists(), "Staging directory MUST be preserved when retention fails!"

    def test_archive_and_verify_rejects_empty_json_preserves_staging(self, tmp_path):
        """R4: Semantic audit retention fails closed on empty JSON objects {} and preserves staging."""
        staging_dir = tmp_path / "staging_r4"
        staging_dir.mkdir()
        # Create all 7 files, but make one an empty JSON object {}
        (staging_dir / "package_a_identity_adoption_plan.json").write_text("{}", encoding="utf-8")
        (staging_dir / "package_a_approval_manifest.json").write_text("{}", encoding="utf-8")
        (staging_dir / "applied_plan_PLAN_ADOPT_9A2E4E04E878.json").write_text("{}", encoding="utf-8")
        (staging_dir / "execution_summary.json").write_text("{}", encoding="utf-8")
        (staging_dir / "post_execution_readback.json").write_text("{}", encoding="utf-8")
        (staging_dir / "target_identity_state_export.json").write_text("{}", encoding="utf-8")

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True):
            pass
        bak_path = str(tmp_path / "identity.db.pre_apply.bak")
        shutil.copy2(db_path, bak_path)

        audit_dir = tmp_path / "audit_r4"
        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                backup_path=bak_path,
                clean_staging_on_success=True,
            )
        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in str(exc_info.value)
        assert staging_dir.exists(), "Staging must not be removed when semantic verification fails!"

    def test_recovery_triage_detects_tampered_association_attributes(self, tmp_path):
        """R5: Tampered candidate association attributes (e.g. wrong operating_mic or date) fail closed."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        db_path = str(tmp_path / "identity.db")
        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "DAVID_AKARA"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        manifest_path = str(tmp_path / "approval_manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest_data), encoding="utf-8")

        plan_path = str(EVIDENCE_DIR / "package_a_identity_adoption_plan.json")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_out = str(tmp_path / "applied_plan.json")

        outcome = execute_controlled_registration(
            reviewed_plan_path=plan_path,
            approval_manifest_path=manifest_path,
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )
        assert outcome.status == IdentityApplicationStatus.APPLIED

        # Tamper with one candidate association in the database (e.g. change operating_mic to XNYS for NASDAQ symbol)
        with SqliteIdentityRepository(db_path, auto_init=False) as repo:
            with repo._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE listing_associations SET operating_mic = 'XNYS' WHERE symbol = 'NVDA'"
                )
                conn.commit()

        diagnosis = diagnose_execution_state(
            db_path=db_path,
            plan_id="PLAN_ADOPT_9A2E4E04E878",
            reviewed_plan_path=plan_path,
        )
        assert diagnosis["triage_state"] in ("ASSOCIATION_ATTRIBUTE_MISMATCH", "INCONSISTENT_STATE")
        assert diagnosis["safe_to_retry"] is False


# ────────────────────────────────────────────────────────────
# 11. Defect B: Applied Receipt Retention Integrity & Staging Preservation
# ────────────────────────────────────────────────────────────

class TestDefectBAppliedReceiptRetentionIntegrity:
    @staticmethod
    def _setup_rehearsal_bundle(tmp_path: Path):
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))

        data_dir = tmp_path / "analyst_dashboard" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / "identity.db")

        with SqliteIdentityRepository(db_path, auto_init=True) as repo:
            for sec_data in export_data["state"]["security_registry"].values():
                repo.save_security(InternalSecurityRecord(**sec_data))
            for assoc_data in export_data["state"]["association_ledger"].values():
                repo.save_association(EffectiveDatedListingAssociation(**assoc_data))

        staging_dir = tmp_path / "staging" / "expansion"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_plan = staging_dir / "package_a_identity_adoption_plan.json"
        staged_plan.write_bytes((EVIDENCE_DIR / "package_a_identity_adoption_plan.json").read_bytes())

        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        manifest_data = json.loads(template_file.read_text(encoding="utf-8"))
        manifest_data["approved_target_db_path"] = str(Path(db_path).resolve())
        manifest_data["authorized_by"] = "DAVID_AKARA"
        manifest_data["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        staged_manifest = staging_dir / "approval_manifest.json"
        staged_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

        audit_dir = tmp_path / "analyst_dashboard" / "audit" / "identity"
        audit_dir.mkdir(parents=True, exist_ok=True)
        applied_plan_out = str(audit_dir / "applied_plan_PLAN_ADOPT_9A2E4E04E878.json")
        backup_path = str(audit_dir / "identity.db.pre_apply.bak")

        outcome = execute_controlled_registration(
            reviewed_plan_path=str(staged_plan),
            approval_manifest_path=str(staged_manifest),
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_out,
        )
        assert outcome.status == IdentityApplicationStatus.APPLIED
        return staging_dir, audit_dir, db_path, backup_path, applied_plan_out, outcome

    def test_defect_b_altered_entity_cik_fails_retention_and_preserves_staging(self, tmp_path):
        """Defect B: Altering an entity's CIK in the applied receipt causes retention failure and preserves staging."""
        staging_dir, audit_dir, db_path, backup_path, applied_plan_out, outcome = self._setup_rehearsal_bundle(tmp_path)

        # In-place modify applied plan: change first entity's issuer_cik
        p_applied = Path(applied_plan_out)
        data = json.loads(p_applied.read_text(encoding="utf-8"))
        data["entities"][0]["issuer_cik"] = "0000000001"
        p_applied.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                plan_id="PLAN_ADOPT_9A2E4E04E878",
                execution_outcome=outcome,
                backup_path=backup_path,
                clean_staging_on_success=True,
            )

        err_msg = str(exc_info.value)
        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in err_msg
        assert "issuer_cik" in err_msg or "content digest" in err_msg or "attribute mismatch" in err_msg
        assert staging_dir.exists(), "Staging directory MUST be preserved when receipt verification fails!"

    def test_defect_b_altered_association_attributes_fails_retention_and_preserves_staging(self, tmp_path):
        """Defect B: Altering an association's operating_mic in applied receipt causes retention failure and preserves staging."""
        staging_dir, audit_dir, db_path, backup_path, applied_plan_out, outcome = self._setup_rehearsal_bundle(tmp_path)

        p_applied = Path(applied_plan_out)
        data = json.loads(p_applied.read_text(encoding="utf-8"))
        data["associations"][0]["operating_mic"] = "XNYS"
        p_applied.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                plan_id="PLAN_ADOPT_9A2E4E04E878",
                execution_outcome=outcome,
                backup_path=backup_path,
                clean_staging_on_success=True,
            )

        err_msg = str(exc_info.value)
        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in err_msg
        assert "operating_mic" in err_msg or "content digest" in err_msg or "attribute mismatch" in err_msg
        assert staging_dir.exists(), "Staging directory MUST be preserved when receipt verification fails!"

    def test_defect_b_self_consistent_tampered_plan_digest_rejected(self, tmp_path):
        """Defect B: Self-consistent receipt with altered content and recomputed digest is rejected."""
        staging_dir, audit_dir, db_path, backup_path, applied_plan_out, outcome = self._setup_rehearsal_bundle(tmp_path)

        p_applied = Path(applied_plan_out)
        data = json.loads(p_applied.read_text(encoding="utf-8"))
        # Tamper content and fake digest
        data["entities"][0]["primary_symbol"] = "TAMPERED_SYM"
        data["plan_content_digest"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        p_applied.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                plan_id="PLAN_ADOPT_9A2E4E04E878",
                execution_outcome=outcome,
                backup_path=backup_path,
                clean_staging_on_success=True,
            )

        err_msg = str(exc_info.value)
        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in err_msg
        assert staging_dir.exists(), "Staging directory MUST be preserved when receipt digest mismatches reviewed plan!"

    def test_defect_b_contradictory_authorization_flags_rejected(self, tmp_path):
        """Defect B: Contradictory authorization flags in applied receipt are rejected."""
        staging_dir, audit_dir, db_path, backup_path, applied_plan_out, outcome = self._setup_rehearsal_bundle(tmp_path)

        p_applied = Path(applied_plan_out)
        data = json.loads(p_applied.read_text(encoding="utf-8"))
        data["authorization"]["is_authorized"] = False
        data["authorization"]["exact_plan_approval"] = "REJECTED"
        p_applied.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(RuntimeError) as exc_info:
            archive_and_verify_audit_bundle(
                staging_dir=str(staging_dir),
                audit_dir=str(audit_dir),
                db_path=db_path,
                plan_id="PLAN_ADOPT_9A2E4E04E878",
                execution_outcome=outcome,
                backup_path=backup_path,
                clean_staging_on_success=True,
            )

        err_msg = str(exc_info.value)
        assert "AUDIT_RETENTION_VERIFICATION_FAILED" in err_msg
        assert "authorization" in err_msg.lower()
        assert staging_dir.exists(), "Staging directory MUST be preserved when authorization flags are contradictory!"



