"""
scripts/apply_registration_plan_wrapper.py

Controlled execution wrapper for applying a reviewed, approved identity
registration plan against an explicitly verified target repository.

Enforces strict governance constraints:
1. Requires a valid, external operator approval manifest.
2. Verifies reviewed plan file SHA-256 and plan content digest before execution.
3. Enforces explicit candidate identity bindings (preventing positional/swapped allocations).
4. Verifies target freshness and checks for already-applied plans BEFORE creating backups.
5. Creates and independently verifies a consistent SQLite backup via the SQLite backup API.
   Enforces exclusive non-clobber creation so prior backups are never overwritten.
6. Re-verifies target database path, existence, and target-state digest before calling the service.
7. Delegates atomic execution to IdentityPlanApplicationService.apply_plan under transaction lock.
8. Preserves the approved applied plan artifact and approval reference for audit (no deletion).
9. Ensures post-commit artifact write failures are reported as committed, not falsely rolled back.
10. Fails closed with diagnostics on any discrepancy; never automatically deletes or overwrites databases.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionAuthorization,
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
)
from analyst_dashboard.universe.identity_persistence import (
    IdentityApplicationOutcome,
    IdentityApplicationStatus,
    IdentityPlanApplicationService,
    SqliteIdentityRepository,
)


class IdentityWrapperError(Exception):
    """Base error for identity execution wrapper failures."""
    pass


class IdentityAlreadyAppliedError(IdentityWrapperError):
    """Raised when a plan has already been applied to the target repository."""
    pass


class BackupError(IdentityWrapperError):
    """Raised when backup creation, verification, or schema validation fails."""
    pass


class ApprovalManifestError(IdentityWrapperError):
    """Raised when the approval manifest is missing, invalid, or untrusted."""
    pass


def create_verified_sqlite_backup(source_db_path: str, backup_db_path: str) -> str:
    """
    Creates a consistent SQLite backup using the supported SQLite backup API
    and verifies that the backup can be independently opened and its schema and
    logical state digest match the source database.

    Enforces exclusive creation (no-clobber) to guarantee that prior backups are
    never silently overwritten.
    """
    p_src = Path(source_db_path).resolve()
    if not p_src.is_file():
        raise FileNotFoundError(f"Source database '{source_db_path}' does not exist.")

    p_bck = Path(backup_db_path).resolve()
    if p_bck.exists():
        raise FileExistsError(
            f"BACKUP_DESTINATION_EXISTS: Backup path '{backup_db_path}' already exists. "
            "Exclusive creation required; refusing to overwrite existing backup."
        )

    # Ensure parent directory for backup exists
    p_bck.parent.mkdir(parents=True, exist_ok=True)

    # 1. Connect source using read-only URI mode (?mode=ro)
    # This guarantees the SQLite engine cannot create a missing file or mutate source
    src_uri = f"{p_src.as_uri()}?mode=ro"
    src_conn = sqlite3.connect(src_uri, uri=True, check_same_thread=False)
    bck_conn = sqlite3.connect(str(p_bck), check_same_thread=False)
    try:
        src_conn.backup(bck_conn)
    finally:
        bck_conn.close()
        src_conn.close()

    # 2. Independently verify backup schema before computing digest
    with SqliteIdentityRepository(str(p_bck), read_only=True) as bck_repo:
        with bck_repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('internal_securities', 'listing_associations', 'applied_plan_audit_ledger', 'identity_evidence_references')"
            )
            found_tables = {row[0] for row in cur.fetchall()}
            expected_tables = {
                "internal_securities",
                "listing_associations",
                "applied_plan_audit_ledger",
                "identity_evidence_references",
            }
            if found_tables != expected_tables:
                raise BackupError(
                    f"BACKUP_SCHEMA_INCOMPLETE: Backup missing required tables: {expected_tables - found_tables}"
                )

        bck_digest = bck_repo.compute_target_state_digest()

    with SqliteIdentityRepository(str(p_src), read_only=True) as src_repo:
        src_digest = src_repo.compute_target_state_digest()

    if src_digest != bck_digest:
        raise BackupError(
            f"BACKUP_VERIFICATION_FAILED: Source state digest '{src_digest}' does not match "
            f"backup state digest '{bck_digest}'."
        )

    return bck_digest


def verify_explicit_candidate_bindings(
    plan: IdentityAdoptionPlan,
    expected_bindings: Dict[str, Dict[str, Any]],
) -> None:
    """
    Verifies that planned entities and associations match the expected authoritative
    candidate bindings explicitly by symbol, CIK, share class, and durable ID.
    Rejects any missing, duplicated, or swapped bindings.
    """
    entity_by_sym = {e.primary_symbol: e for e in plan.entities}
    assoc_by_sym = {a.symbol: a for a in plan.associations}

    if set(entity_by_sym.keys()) != set(expected_bindings.keys()):
        raise ValueError(
            f"CANDIDATE_BINDING_MISMATCH: Planned entity symbols {set(entity_by_sym.keys())} "
            f"do not match expected symbols {set(expected_bindings.keys())}."
        )

    for sym, expected in expected_bindings.items():
        ent = entity_by_sym[sym]
        exp_cik = expected.get("issuer_cik")
        exp_class = expected.get("share_class")
        exp_durable_id = expected.get("durable_security_id")
        exp_mic = expected.get("operating_mic")
        exp_eff_from = expected.get("effective_from")

        if ent.issuer_cik != exp_cik:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: CIK mismatch for {sym}: expected '{exp_cik}', got '{ent.issuer_cik}'."
            )
        if ent.share_class != exp_class:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: Share class mismatch for {sym}: expected '{exp_class}', got '{ent.share_class}'."
            )
        if ent.proposed_durable_security_id != exp_durable_id:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: Durable security ID mismatch for {sym}: "
                f"expected '{exp_durable_id}', got '{ent.proposed_durable_security_id}'."
            )

        assoc = assoc_by_sym.get(sym)
        if not assoc:
            raise ValueError(f"CANDIDATE_BINDING_MISMATCH: Missing association for symbol '{sym}'.")
        if assoc.target_durable_security_id != exp_durable_id:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: Association target security ID mismatch for {sym}: "
                f"expected '{exp_durable_id}', got '{assoc.target_durable_security_id}'."
            )
        if exp_mic and assoc.operating_mic != exp_mic:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: Operating MIC mismatch for {sym}: expected '{exp_mic}', got '{assoc.operating_mic}'."
            )
        if exp_eff_from and assoc.effective_from != exp_eff_from:
            raise ValueError(
                f"CANDIDATE_BINDING_MISMATCH: Effective from date mismatch for {sym}: expected '{exp_eff_from}', got '{assoc.effective_from}'."
            )


def execute_controlled_registration(
    reviewed_plan_path: str,
    approval_manifest_path: str,
    db_path: str,
    backup_path: Optional[str] = None,
    output_applied_plan_path: Optional[str] = None,
) -> IdentityApplicationOutcome:
    """
    Executes controlled identity plan application enforcing all pre-execution and
    transaction-level governance gates.
    """
    # 1. Load and validate operator approval manifest
    p_manifest = Path(approval_manifest_path)
    if not p_manifest.exists():
        raise FileNotFoundError(f"APPROVAL_MANIFEST_MISSING: Approval manifest '{approval_manifest_path}' does not exist.")

    manifest_data = json.loads(p_manifest.read_text(encoding="utf-8"))
    required_manifest_keys = [
        "approval_id",
        "reviewed_plan_id",
        "reviewed_plan_file_sha256",
        "reviewed_plan_content_digest",
        "approved_target_db_path",
        "expected_target_state_digest",
        "authorized_by",
        "authorization_timestamp_utc",
        "scope",
        "candidate_bindings",
    ]
    for k in required_manifest_keys:
        if k not in manifest_data or manifest_data[k] is None:
            raise ApprovalManifestError(f"APPROVAL_MANIFEST_INVALID: Missing required key '{k}' in approval manifest.")

    if manifest_data["scope"] != "REFERENCE_IDENTITY_REGISTRATION":
        raise ApprovalManifestError(
            f"UNSUPPORTED_SCOPE: Execution requires scope 'REFERENCE_IDENTITY_REGISTRATION', "
            f"got '{manifest_data['scope']}'."
        )

    # 2. Load and verify reviewed unapproved plan
    p_plan = Path(reviewed_plan_path)
    if not p_plan.exists():
        raise FileNotFoundError(f"REVIEWED_PLAN_MISSING: Plan file '{reviewed_plan_path}' does not exist.")

    plan_raw_bytes = p_plan.read_bytes()
    computed_file_sha256 = hashlib.sha256(plan_raw_bytes).hexdigest()
    if computed_file_sha256 != manifest_data["reviewed_plan_file_sha256"]:
        raise ValueError(
            f"PLAN_FILE_TAMPERED: Plan file SHA-256 '{computed_file_sha256}' does not match "
            f"approved manifest SHA-256 '{manifest_data['reviewed_plan_file_sha256']}'."
        )

    reviewed_plan = IdentityAdoptionPlan.from_json(plan_raw_bytes.decode("utf-8"))
    if not reviewed_plan.is_plan_content_valid():
        raise ValueError("PLAN_INTEGRITY_TAMPERED: Recomputed content digest does not match declared digest.")

    if reviewed_plan.plan_content_digest != manifest_data["reviewed_plan_content_digest"]:
        raise ValueError(
            f"PLAN_CONTENT_DIGEST_MISMATCH: Reviewed plan content digest '{reviewed_plan.plan_content_digest}' "
            f"does not match approved manifest digest '{manifest_data['reviewed_plan_content_digest']}'."
        )

    if reviewed_plan.plan_id != manifest_data["reviewed_plan_id"]:
        raise ValueError(
            f"PLAN_ID_MISMATCH: Reviewed plan ID '{reviewed_plan.plan_id}' does not match "
            f"approved manifest plan ID '{manifest_data['reviewed_plan_id']}'."
        )

    # 3. Target database path verification & pre-flight existence
    resolved_db_path = str(Path(db_path).resolve())
    approved_db_path = str(Path(manifest_data["approved_target_db_path"]).resolve())
    if resolved_db_path != approved_db_path:
        raise ValueError(
            f"TARGET_PATH_UNAPPROVED: Provided db_path '{resolved_db_path}' does not match "
            f"approved target path '{approved_db_path}' in manifest."
        )

    if not os.path.exists(resolved_db_path) or not os.path.isfile(resolved_db_path):
        raise FileNotFoundError(
            f"TARGET_DATABASE_ABSENT: Target repository '{resolved_db_path}' does not exist. "
            "Repository schema must be explicitly initialized prior to plan application."
        )

    # 4. Pre-application target freshness check & retry detection (BEFORE backup creation)
    with SqliteIdentityRepository(resolved_db_path, read_only=True) as check_repo:
        current_target_digest = check_repo.compute_target_state_digest()

        # Check if plan was already applied
        with check_repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT audit_id, applied_at_utc, target_state_digest_after FROM applied_plan_audit_ledger WHERE plan_id = ?",
                (reviewed_plan.plan_id,),
            )
            already_applied_row = cur.fetchone()
            if already_applied_row is not None:
                raise IdentityAlreadyAppliedError(
                    f"ALREADY_APPLIED: Plan '{reviewed_plan.plan_id}' is already recorded as applied in repository "
                    f"'{resolved_db_path}' (Audit ID {already_applied_row[0]} applied at {already_applied_row[1]}, "
                    f"target digest after: {already_applied_row[2]}). Current digest: '{current_target_digest}'. "
                    "Stop and inspect."
                )

    if current_target_digest != manifest_data["expected_target_state_digest"]:
        raise ValueError(
            f"TARGET_STATE_DIGEST_MISMATCH: Current target state digest '{current_target_digest}' "
            f"does not match expected target state digest '{manifest_data['expected_target_state_digest']}'."
        )

    # 5. Explicit candidate identity bindings verification & action disposition validation
    verify_explicit_candidate_bindings(reviewed_plan, manifest_data["candidate_bindings"])

    for ent in reviewed_plan.entities:
        if ent.status != IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED:
            raise ValueError(
                f"UNEXPECTED_ACTION_STATE: Entity '{ent.primary_symbol}' has unexpected status '{ent.status.value}'. "
                "Expected 'MECHANICALLY_READY_UNAPPROVED'."
            )
        if not ent.is_creation or not ent.is_mechanically_plannable:
            raise ValueError(f"UNEXPECTED_ACTION_STATE: Entity '{ent.primary_symbol}' is not a plannable creation.")

    for assoc in reviewed_plan.associations:
        if assoc.status != IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED:
            raise ValueError(
                f"UNEXPECTED_ACTION_STATE: Association '{assoc.symbol}' has unexpected status '{assoc.status.value}'. "
                "Expected 'MECHANICALLY_READY_UNAPPROVED'."
            )
        if assoc.conflict_reason != "EXACT_PLAN_APPROVAL_ABSENT":
            raise ValueError(
                f"UNEXPECTED_ACTION_STATE: Association '{assoc.symbol}' has unexpected conflict reason "
                f"'{assoc.conflict_reason}'. Expected 'EXACT_PLAN_APPROVAL_ABSENT'."
            )
        if not assoc.is_creation or not assoc.is_mechanically_plannable:
            raise ValueError(f"UNEXPECTED_ACTION_STATE: Association '{assoc.symbol}' is not a plannable creation.")

    # 6. Consistent SQLite Backup & Verification (exclusive non-clobber)
    resolved_backup_path = backup_path or f"{resolved_db_path}.pre_apply.bak"
    create_verified_sqlite_backup(resolved_db_path, resolved_backup_path)

    # 7. Check output artifact path does not already exist
    out_applied_plan_path = output_applied_plan_path or str(
        Path(resolved_db_path).parent / f"applied_plan_{reviewed_plan.plan_id}_{reviewed_plan.plan_content_digest[:12]}.json"
    )
    if Path(out_applied_plan_path).exists():
        raise FileExistsError(
            f"APPLIED_ARTIFACT_EXISTS: Output artifact path '{out_applied_plan_path}' already exists. "
            "Refusing to overwrite existing execution artifact."
        )

    # 8. Construct approved executable plan bound to verified authorization context
    approved_auth = IdentityAdoptionAuthorization(
        is_authorized=True,
        authorized_by=manifest_data["authorized_by"],
        authorization_timestamp_utc=manifest_data["authorization_timestamp_utc"],
        scope=manifest_data["scope"],
        allow_reference_identity_registration=True,
        exact_plan_approval="APPROVED",
        application_authorized=True,
        production_target=resolved_db_path,
        notes=f"Approved under manifest {manifest_data['approval_id']}.",
    )

    approved_entities = [
        dataclasses.replace(
            e,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            rationale=(
                "Approved under architectural principle: evidence-resolved internal security "
                "identity registered independently of strategy eligibility and R06 status. "
                "Grants zero universe admission or publication authority."
            ),
        )
        for e in reviewed_plan.entities
    ]

    approved_associations = [
        dataclasses.replace(
            a,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            conflict_reason=None,
        )
        for a in reviewed_plan.associations
    ]

    approved_summary = dataclasses.replace(
        reviewed_plan.summary,
        entities_approved_creation=len(approved_entities),
        associations_approved_creation=len(approved_associations),
        entities_mechanically_ready=0,
        associations_mechanically_ready=0,
    )

    approved_content_digest = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
        entities=approved_entities,
        associations=approved_associations,
        provisional_to_durable_map=reviewed_plan.provisional_to_durable_map,
        summary=approved_summary,
    )

    approved_plan = dataclasses.replace(
        reviewed_plan,
        authorization=approved_auth,
        entities=approved_entities,
        associations=approved_associations,
        summary=approved_summary,
        plan_content_digest=approved_content_digest,
        target_state_digest=manifest_data["expected_target_state_digest"],
    )

    if not approved_plan.is_plan_content_valid():
        raise RuntimeError("INTERNAL_ERROR: Constructed approved plan content digest validation failed.")

    # 9. Execute application via released service under SQLite transaction lock
    with SqliteIdentityRepository(resolved_db_path, auto_init=False) as repo:
        outcome = IdentityPlanApplicationService.apply_plan(
            plan=approved_plan,
            repository=repo,
            authorization=approved_auth,
            allow_partial=False,
        )

    if outcome.status != IdentityApplicationStatus.APPLIED:
        raise RuntimeError(f"PLAN_APPLICATION_FAILED: Outcome status was '{outcome.status.value}'. Details: {outcome.details}")

    # 10. Post-commit artifact handling (transaction is COMMITTED in SQLite)
    try:
        Path(out_applied_plan_path).write_text(approved_plan.to_json(indent=2), encoding="utf-8")
    except Exception as e:
        warning_msg = (
            f"POST_COMMIT_ARTIFACT_WRITE_FAILED: Database transaction was COMMITTED to '{resolved_db_path}' "
            f"(target digest: {outcome.target_state_digest_after}), but writing applied plan artifact to "
            f"'{out_applied_plan_path}' failed: {e}. Do NOT roll back or re-apply."
        )
        outcome = dataclasses.replace(
            outcome,
            details=list(outcome.details) + [warning_msg],
        )
        print(f"WARNING: {warning_msg}", file=sys.stderr)

    return outcome


TARGET_BASELINE_STATE_DIGEST = "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"


def compute_approved_plan_content_digest(plan: IdentityAdoptionPlan) -> str:
    """Computes the approved-form plan content digest from a reviewed adoption plan."""
    approved_entities = [
        dataclasses.replace(
            e,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            rationale=(
                "Approved under architectural principle: evidence-resolved internal security "
                "identity registered independently of strategy eligibility and R06 status. "
                "Grants zero universe admission or publication authority."
            ),
        )
        if e.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED else e
        for e in plan.entities
    ]
    approved_associations = [
        dataclasses.replace(
            a,
            status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
            conflict_reason=None,
        )
        if a.status == IdentityAdoptionStatus.MECHANICALLY_READY_UNAPPROVED else a
        for a in plan.associations
    ]
    approved_summary = dataclasses.replace(
        plan.summary,
        entities_approved_creation=len(approved_entities),
        associations_approved_creation=len(approved_associations),
        entities_mechanically_ready=0,
        associations_mechanically_ready=0,
    )
    return IdentityAdoptionDryRunEngine.compute_plan_content_digest(
        entities=approved_entities,
        associations=approved_associations,
        provisional_to_durable_map=plan.provisional_to_durable_map,
        summary=approved_summary,
    )


def diagnose_execution_state(
    db_path: str,
    plan_id: str = "PLAN_ADOPT_9A2E4E04E878",
    reviewed_plan_path: Optional[str] = None,
    expected_initial_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read-only diagnostic triage for uncertain execution outcomes (SIP-06 / R5).
    Binds verification to reviewed plan ID, content digest, candidate identities,
    pilot preservation, and derives counts dynamically from reviewed scope.
    Inspects audit trail to distinguish:
    - 'NOT_APPLIED': No audit record, target state digest matches expected pre-application digest,
      pilot records intact. Safe to address root cause and re-run.
    - 'APPLIED_AND_VERIFIED': Transaction committed in SQLite, audit ledger has entry, target state
      digest matches recorded digest, candidate additions present, pilots preserved. Re-running is rejected.
      DO NOT restore backup.
    - 'APPLIED_WITH_SUBSEQUENT_AUDITED_WRITES': Plan was applied, and subsequent legitimate audited
      transactions exist in applied_plan_audit_ledger. Target state matches latest audit record.
    - 'INCONSISTENT_STATE': Database does not match clean pre-state, verified post-state, or valid audit trail.
      Operator MUST halt without blind retry.
    """
    p = Path(db_path).resolve()
    if not p.is_file():
        return {
            "triage_state": "DATABASE_ABSENT",
            "db_path": str(p),
            "safe_to_retry": False,
        }

    expected_additions = 130
    expected_associations = 130
    expected_entities: Dict[str, Any] = {}
    expected_assocs: Dict[str, Any] = {}
    reviewed_plan: Optional[IdentityAdoptionPlan] = None

    if reviewed_plan_path:
        p_plan = Path(reviewed_plan_path)
        if not p_plan.is_file():
            return {
                "triage_state": "PLAN_FILE_NOT_FOUND",
                "db_path": str(p),
                "plan_path": str(p_plan),
                "safe_to_retry": False,
            }
        try:
            plan_raw = p_plan.read_text(encoding="utf-8")
            reviewed_plan = IdentityAdoptionPlan.from_json(plan_raw)
        except Exception as e:
            return {
                "triage_state": "CORRUPT_OR_MALFORMED_PLAN",
                "db_path": str(p),
                "plan_path": str(p_plan),
                "error": str(e),
                "safe_to_retry": False,
            }

        # SIP-06: Fail closed if declared content digest does not match recomputed content
        if not reviewed_plan.is_plan_content_valid():
            return {
                "triage_state": "PLAN_INTEGRITY_TAMPERED",
                "db_path": str(p),
                "plan_id": reviewed_plan.plan_id,
                "error": "Plan content digest does not match declared digest; candidate attributes or bindings have been altered.",
                "safe_to_retry": False,
            }

        plan_id = reviewed_plan.plan_id
        expected_entities = {
            e.proposed_durable_security_id: e for e in reviewed_plan.entities if e.proposed_durable_security_id
        }
        expected_assocs = {
            a.proposed_durable_association_id: a for a in reviewed_plan.associations if a.proposed_durable_association_id
        }
        expected_additions = len(reviewed_plan.entities)
        expected_associations = len(reviewed_plan.associations)
        expected_initial_digest = expected_initial_digest or reviewed_plan.target_state_digest

    default_initial_digest = expected_initial_digest or TARGET_BASELINE_STATE_DIGEST

    with SqliteIdentityRepository(str(p), read_only=True) as repo:
        current_digest = repo.compute_target_state_digest()
        securities = {s.internal_security_id: s for s in repo.list_all_securities()}
        associations = {a.association_id: a for a in repo.list_all_associations()}
        with repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT audit_id, plan_id, plan_content_digest, target_state_digest_before, target_state_digest_after, applied_at_utc FROM applied_plan_audit_ledger ORDER BY audit_id ASC"
            )
            all_audit_rows = cur.fetchall()

    matching_audit = [row for row in all_audit_rows if row[1] == plan_id]
    subsequent_audit = [row for row in all_audit_rows if matching_audit and row[0] > matching_audit[-1][0]]

    # Pilot preservation check verifying both durable ID existence AND authoritative attributes
    canonical_pilots = {
        "ARX_SEC_0B4C728FE0BE": {"symbol": "AAPL", "cik": "0000320193", "share_class": None},
        "ARX_SEC_A11A24D442E2": {"symbol": "GOOG", "cik": "0001652044", "share_class": "CLASS_C"},
        "ARX_SEC_B7EF9ED7A35A": {"symbol": "GOOGL", "cik": "0001652044", "share_class": "CLASS_A"},
    }
    pilots_preserved = True
    for pid, canon in canonical_pilots.items():
        sec = securities.get(pid)
        if not sec:
            pilots_preserved = False
            break
        if (
            sec.primary_symbol != canon["symbol"]
            or sec.issuer_cik != canon["cik"]
            or sec.share_class != canon["share_class"]
        ):
            pilots_preserved = False
            break

    # 1. NOT_APPLIED: clean baseline pre-state
    if not matching_audit and current_digest == default_initial_digest and len(securities) == len(canonical_pilots) and pilots_preserved:
        return {
            "triage_state": "NOT_APPLIED",
            "plan_id": plan_id,
            "current_digest": current_digest,
            "securities_count": len(securities),
            "associations_count": len(associations),
            "pilots_preserved": pilots_preserved,
            "safe_to_retry": True,
        }

    # If canonical pilots are altered or missing, state is inconsistent
    if not pilots_preserved:
        return {
            "triage_state": "INCONSISTENT_STATE",
            "plan_id": plan_id,
            "current_digest": current_digest,
            "securities_count": len(securities),
            "associations_count": len(associations),
            "pilots_preserved": False,
            "safe_to_retry": False,
            "error": "Canonical pilot securities have been altered or deleted from target database.",
        }

    # If there are NO audit records at all, but the database does not match clean baseline:
    if not matching_audit:
        return {
            "triage_state": "INCONSISTENT_STATE",
            "plan_id": plan_id,
            "current_digest": current_digest,
            "securities_count": len(securities),
            "associations_count": len(associations),
            "pilots_preserved": pilots_preserved,
            "safe_to_retry": False,
            "error": "No matching audit record, but database does not match clean baseline pilot state.",
        }

    # R5: If audit records exist or database state has additions, authoritative reviewed plan is required to verify!
    if not reviewed_plan:
        return {
            "triage_state": "MISSING_REVIEWED_PLAN_EVIDENCE",
            "db_path": str(p),
            "plan_id": plan_id,
            "current_digest": current_digest,
            "securities_count": len(securities),
            "associations_count": len(associations),
            "error": "Applied database state requires authoritative reviewed plan evidence to certify candidate entity and association integrity.",
            "safe_to_retry": False,
        }

    # R5: Verify plan linkage with applied audit record
    if matching_audit:
        rec = matching_audit[-1]
        rec_content_digest = rec[2]
        approved_digest = compute_approved_plan_content_digest(reviewed_plan)
        valid_plan_digests = {reviewed_plan.plan_content_digest, approved_digest}
        if rec_content_digest not in valid_plan_digests:
            return {
                "triage_state": "PLAN_CONTENT_DIGEST_MISMATCH_WITH_AUDIT_LEDGER",
                "db_path": str(p),
                "plan_id": plan_id,
                "current_digest": current_digest,
                "supplied_plan_content_digest": reviewed_plan.plan_content_digest,
                "recorded_audit_content_digest": rec_content_digest,
                "error": "Supplied reviewed plan content digest does not match the recorded digest in the applied audit ledger.",
                "safe_to_retry": False,
            }

    # Candidate entity and association verification
    candidates_match = True
    if expected_entities:
        for exp_id, exp_ent in expected_entities.items():
            sec = securities.get(exp_id)
            if not sec:
                candidates_match = False
                break
            if (
                sec.primary_symbol != exp_ent.primary_symbol
                or sec.issuer_cik != exp_ent.issuer_cik
                or sec.share_class != exp_ent.share_class
            ):
                candidates_match = False
                break

    assocs_match = True
    if expected_assocs:
        for exp_assoc_id, exp_assoc in expected_assocs.items():
            assoc = associations.get(exp_assoc_id)
            if not assoc:
                matching = [
                    a for a in associations.values()
                    if a.internal_security_id == exp_assoc.target_durable_security_id
                    and a.symbol == exp_assoc.symbol
                ]
                if not matching:
                    assocs_match = False
                    break
                assoc = matching[0]
            if (
                assoc.symbol != exp_assoc.symbol
                or assoc.internal_security_id != exp_assoc.target_durable_security_id
                or (exp_assoc.operating_mic and assoc.operating_mic != exp_assoc.operating_mic)
                or (exp_assoc.effective_from and assoc.effective_from != exp_assoc.effective_from)
                or (exp_assoc.effective_to != assoc.effective_to)
            ):
                assocs_match = False
                break

    all_candidates_verified = candidates_match and assocs_match
    if not all_candidates_verified:
        return {
            "triage_state": "ASSOCIATION_ATTRIBUTE_MISMATCH" if not assocs_match else "CANDIDATE_ATTRIBUTES_MISMATCH",
            "db_path": str(p),
            "plan_id": plan_id,
            "current_digest": current_digest,
            "candidates_match": candidates_match,
            "assocs_match": assocs_match,
            "error": "Candidate entities or listing associations in target database do not match reviewed plan bindings.",
            "safe_to_retry": False,
        }

    # 2. APPLIED_AND_VERIFIED (clean post-state without subsequent writes)
    if matching_audit and not subsequent_audit:
        rec = matching_audit[-1]
        rec_digest_after = rec[4]
        expected_final_count = len(canonical_pilots) + expected_additions
        expected_final_assocs = len(canonical_pilots) + expected_associations

        if (
            current_digest == rec_digest_after
            and len(securities) == expected_final_count
            and len(associations) == expected_final_assocs
            and pilots_preserved
            and all_candidates_verified
        ):
            return {
                "triage_state": "APPLIED_AND_VERIFIED",
                "plan_id": plan_id,
                "current_digest": current_digest,
                "audit_id": rec[0],
                "applied_at_utc": rec[5],
                "target_state_digest_after": rec[4],
                "securities_count": len(securities),
                "associations_count": len(associations),
                "pilots_preserved": pilots_preserved,
                "candidate_additions_verified": all_candidates_verified,
                "safe_to_retry": False,
            }

    # 3. APPLIED_WITH_SUBSEQUENT_AUDITED_WRITES
    if matching_audit and subsequent_audit:
        latest_rec = all_audit_rows[-1]
        if current_digest == latest_rec[4] and pilots_preserved and all_candidates_verified:
            return {
                "triage_state": "APPLIED_WITH_SUBSEQUENT_AUDITED_WRITES",
                "plan_id": plan_id,
                "current_digest": current_digest,
                "audit_id": matching_audit[-1][0],
                "applied_at_utc": matching_audit[-1][5],
                "subsequent_audit_records_count": len(subsequent_audit),
                "latest_audit_id": latest_rec[0],
                "securities_count": len(securities),
                "associations_count": len(associations),
                "pilots_preserved": pilots_preserved,
                "candidate_additions_verified": all_candidates_verified,
                "safe_to_retry": False,
            }

    # 4. Otherwise: INCONSISTENT_STATE
    return {
        "triage_state": "INCONSISTENT_STATE",
        "plan_id": plan_id,
        "current_digest": current_digest,
        "securities_count": len(securities),
        "associations_count": len(associations),
        "pilots_preserved": pilots_preserved,
        "safe_to_retry": False,
        "error": "Database state does not match clean baseline, verified post-state, or valid audit record.",
    }
    return {
        "triage_state": "INCONSISTENT_STATE",
        "plan_id": plan_id,
        "current_digest": current_digest,
        "matching_audit_records": len(matching_audit),
        "total_audit_records": len(all_audit_rows),
        "securities_count": len(securities),
        "associations_count": len(associations),
        "pilots_preserved": pilots_preserved,
        "candidate_additions_verified": all_candidates_verified,
        "safe_to_retry": False,
    }


REQUIRED_AUDIT_ARTIFACT_NAMES = [
    "package_a_identity_adoption_plan.json",
    "package_a_approval_manifest.json",
    "applied_plan_PLAN_ADOPT_9A2E4E04E878.json",
    "execution_summary.json",
    "post_execution_readback.json",
    "target_identity_state_export.json",
    "identity.db.pre_apply.bak",
]


def archive_and_verify_audit_bundle(
    staging_dir: Optional[str],
    audit_dir: str,
    db_path: str,
    plan_id: str = "PLAN_ADOPT_9A2E4E04E878",
    execution_outcome: Optional[IdentityApplicationOutcome] = None,
    backup_path: Optional[str] = None,
    clean_staging_on_success: bool = True,
) -> Dict[str, Any]:
    """
    SIP-06: Shared, production-ready durable audit retention and staging cleanup.
    Ensures all 7 required audit artifacts are gathered, verified for non-emptiness,
    valid JSON / SQLite schema, and hashed before transient staging is removed.
    Fails closed and PRESERVES staging if any artifact is missing or invalid.
    """
    p_audit = Path(audit_dir).resolve()
    p_audit.mkdir(parents=True, exist_ok=True)
    p_staging = Path(staging_dir).resolve() if staging_dir else None

    # 1. package_a_identity_adoption_plan.json
    dest_plan = p_audit / "package_a_identity_adoption_plan.json"
    if not dest_plan.exists() and p_staging:
        src = p_staging / "package_a_identity_adoption_plan.json"
        if src.is_file():
            shutil.copy2(str(src), str(dest_plan))

    # 2. package_a_approval_manifest.json
    dest_manifest = p_audit / "package_a_approval_manifest.json"
    if not dest_manifest.exists() and p_staging:
        for candidate_name in ["package_a_approval_manifest.json", "approval_manifest.json"]:
            src = p_staging / candidate_name
            if src.is_file():
                shutil.copy2(str(src), str(dest_manifest))
                break

    # 3. applied_plan_<plan_id>.json
    applied_name = f"applied_plan_{plan_id}.json"
    dest_applied = p_audit / applied_name
    if not dest_applied.exists():
        if p_staging:
            matches = list(p_staging.glob(f"applied_plan_{plan_id}*.json"))
            if matches:
                shutil.copy2(str(matches[0]), str(dest_applied))
        if not dest_applied.exists():
            p_db = Path(db_path).resolve()
            db_matches = list(p_db.parent.glob(f"applied_plan_{plan_id}*.json"))
            if db_matches:
                shutil.copy2(str(db_matches[0]), str(dest_applied))

    # 4. execution_summary.json
    dest_summary = p_audit / "execution_summary.json"
    if not dest_summary.exists():
        if execution_outcome:
            dest_summary.write_text(execution_outcome.to_json(indent=2), encoding="utf-8")
        elif p_staging and (p_staging / "execution_summary.json").is_file():
            shutil.copy2(str(p_staging / "execution_summary.json"), str(dest_summary))
    else:
        try:
            existing_sum = json.loads(dest_summary.read_text(encoding="utf-8"))
            if existing_sum.get("plan_id") != plan_id:
                raise RuntimeError(
                    f"AUDIT_EVIDENCE_CONFLICT: Existing execution_summary.json belongs to plan '{existing_sum.get('plan_id')}' != '{plan_id}'."
                )
        except json.JSONDecodeError:
            pass

    # 5. post_execution_readback.json
    dest_readback = p_audit / "post_execution_readback.json"
    p_db = Path(db_path).resolve()
    if p_db.is_file():
        with SqliteIdentityRepository(str(p_db), read_only=True) as repo:
            secs = repo.list_all_securities()
            assocs = repo.list_all_associations()
            d_after = repo.compute_target_state_digest()
            rb_data = {
                "plan_id": plan_id,
                "securities_count": len(secs),
                "associations_count": len(assocs),
                "target_state_digest_before": TARGET_BASELINE_STATE_DIGEST,
                "target_state_digest_after": d_after,
                "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            if dest_readback.exists():
                try:
                    existing_rb = json.loads(dest_readback.read_text(encoding="utf-8"))
                    if existing_rb.get("target_state_digest_after") != d_after or existing_rb.get("plan_id") != plan_id:
                        raise RuntimeError(
                            f"AUDIT_EVIDENCE_CONFLICT: Existing post_execution_readback.json belongs to a different "
                            f"execution state (digest '{existing_rb.get('target_state_digest_after')}') than current target ('{d_after}'). "
                            "Refusing to overwrite earlier execution evidence."
                        )
                except json.JSONDecodeError:
                    pass
            dest_readback.write_text(json.dumps(rb_data, indent=2), encoding="utf-8")

    # 6. target_identity_state_export.json
    dest_export = p_audit / "target_identity_state_export.json"
    if not dest_export.exists():
        if p_staging and (p_staging / "target_identity_state_export.json").is_file():
            shutil.copy2(str(p_staging / "target_identity_state_export.json"), str(dest_export))
        else:
            repo_ev = repo_root / "evidence" / "coverage_expansion" / "target_identity_state_export.json"
            if repo_ev.is_file():
                shutil.copy2(str(repo_ev), str(dest_export))

    # 7. identity.db.pre_apply.bak
    dest_bak = p_audit / "identity.db.pre_apply.bak"
    if not dest_bak.exists():
        src_bak = Path(backup_path).resolve() if backup_path else Path(f"{db_path}.pre_apply.bak").resolve()
        if src_bak.is_file():
            shutil.copy2(str(src_bak), str(dest_bak))

    # Strict Verification of ALL 7 Artifacts
    missing_or_empty = []
    artifact_hashes = {}
    for art_name in REQUIRED_AUDIT_ARTIFACT_NAMES:
        art_path = p_audit / art_name
        if not art_path.is_file():
            missing_or_empty.append(f"MISSING: {art_name}")
            continue
        if art_path.stat().st_size == 0:
            missing_or_empty.append(f"EMPTY (0 bytes): {art_name}")
            continue

        raw = art_path.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        artifact_hashes[art_name] = {
            "size_bytes": len(raw),
            "sha256": h,
        }

        # Verify JSON validity for JSON artifacts
        if art_name.endswith(".json"):
            try:
                json.loads(raw.decode("utf-8"))
            except Exception as e:
                missing_or_empty.append(f"INVALID_JSON: {art_name} ({e})")

    # Verify backup SQLite database integrity
    if dest_bak.is_file() and dest_bak.stat().st_size > 0:
        try:
            with SqliteIdentityRepository(str(dest_bak), read_only=True) as bck_repo:
                bck_digest = bck_repo.compute_target_state_digest()
                artifact_hashes["identity.db.pre_apply.bak"]["sqlite_target_state_digest"] = bck_digest
        except Exception as e:
            missing_or_empty.append(f"INVALID_SQLITE_BACKUP: {e}")

    # R4: Semantic Content & Cross-Artifact Identity Verification
    semantic_defects = []

    # 1. Verify reviewed plan content integrity
    plan_obj = None
    try:
        plan_raw = (p_audit / "package_a_identity_adoption_plan.json").read_text(encoding="utf-8")
        plan_data = json.loads(plan_raw)
        if not isinstance(plan_data, dict) or not plan_data:
            semantic_defects.append("package_a_identity_adoption_plan.json is empty or not an object.")
        else:
            plan_obj = IdentityAdoptionPlan.from_json(plan_raw)
            if plan_obj.plan_id != plan_id:
                semantic_defects.append(f"plan_id mismatch in plan: expected '{plan_id}', got '{plan_obj.plan_id}'.")
            if not plan_obj.is_plan_content_valid():
                semantic_defects.append("Plan content digest does not match declared digest in plan.")
            if len(plan_obj.entities) != 130 or len(plan_obj.associations) != 130:
                semantic_defects.append(f"Expected 130 entities and 130 associations in plan, got {len(plan_obj.entities)} and {len(plan_obj.associations)}.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate package_a_identity_adoption_plan.json: {e}")

    # 2. Verify approval manifest authorization and linkage
    try:
        manifest_raw = (p_audit / "package_a_approval_manifest.json").read_text(encoding="utf-8")
        manifest_data = json.loads(manifest_raw)
        if not isinstance(manifest_data, dict) or not manifest_data:
            semantic_defects.append("package_a_approval_manifest.json is empty or not an object.")
        else:
            if manifest_data.get("authorized_by") is None:
                semantic_defects.append("Approval manifest has null authorized_by.")
            if manifest_data.get("scope") != "REFERENCE_IDENTITY_REGISTRATION":
                semantic_defects.append(f"Approval manifest scope '{manifest_data.get('scope')}' != 'REFERENCE_IDENTITY_REGISTRATION'.")
            if manifest_data.get("reviewed_plan_id") != plan_id:
                semantic_defects.append(f"Manifest reviewed_plan_id '{manifest_data.get('reviewed_plan_id')}' != '{plan_id}'.")
            if plan_obj and manifest_data.get("reviewed_plan_content_digest") != plan_obj.plan_content_digest:
                semantic_defects.append("Manifest plan_content_digest does not match reviewed plan.")
            if "package_a_identity_adoption_plan.json" in artifact_hashes:
                expected_sha = artifact_hashes["package_a_identity_adoption_plan.json"]["sha256"]
                if manifest_data.get("reviewed_plan_file_sha256") != expected_sha:
                    semantic_defects.append("Manifest reviewed_plan_file_sha256 does not match reviewed plan file hash.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate package_a_approval_manifest.json: {e}")

    # 3. Verify applied plan receipt (Defect B)
    applied_data = None
    applied_obj = None
    try:
        applied_raw = dest_applied.read_text(encoding="utf-8")
        applied_data = json.loads(applied_raw)
        if not isinstance(applied_data, dict) or not applied_data:
            semantic_defects.append(f"{applied_name} is empty or not an object.")
        else:
            try:
                applied_obj = IdentityAdoptionPlan.from_json(applied_raw)
            except Exception as deser_err:
                semantic_defects.append(f"Failed to deserialize {applied_name} into IdentityAdoptionPlan: {deser_err}")

            if applied_obj:
                if applied_obj.plan_id != plan_id:
                    semantic_defects.append(f"Applied plan plan_id '{applied_obj.plan_id}' != '{plan_id}'.")

                # B1 & B2: Verify recomputed content digest matches declared digest (detects in-place tampering)
                if not applied_obj.is_plan_content_valid():
                    semantic_defects.append("Applied plan content digest does not match its internal elements (in-place modification detected).")

                # B3: Verify applied plan digest matches expected approved transformation of reviewed plan
                expected_approved_digest = compute_approved_plan_content_digest(plan_obj) if plan_obj else None
                if expected_approved_digest and applied_obj.plan_content_digest != expected_approved_digest:
                    semantic_defects.append(
                        f"Applied plan content digest '{applied_obj.plan_content_digest}' does not match "
                        f"expected approved transformation digest '{expected_approved_digest}'."
                    )

                # B1 & B2: Compare individual entity and association contents against reviewed plan (not just counts!)
                if plan_obj:
                    plan_entities_by_id = {e.provisional_security_id: e for e in plan_obj.entities}
                    for app_e in applied_obj.entities:
                        orig_e = plan_entities_by_id.get(app_e.provisional_security_id)
                        if not orig_e:
                            semantic_defects.append(f"Applied entity '{app_e.provisional_security_id}' not present in reviewed plan.")
                            continue
                        if (
                            app_e.primary_symbol != orig_e.primary_symbol
                            or app_e.issuer_cik != orig_e.issuer_cik
                            or app_e.share_class != orig_e.share_class
                            or app_e.proposed_durable_security_id != orig_e.proposed_durable_security_id
                        ):
                            semantic_defects.append(
                                f"Applied entity '{app_e.provisional_security_id}' attribute mismatch: "
                                f"actual(sym={app_e.primary_symbol}, cik={app_e.issuer_cik}, sc={app_e.share_class}, dur={app_e.proposed_durable_security_id}) != "
                                f"expected(sym={orig_e.primary_symbol}, cik={orig_e.issuer_cik}, sc={orig_e.share_class}, dur={orig_e.proposed_durable_security_id})."
                            )

                    plan_assocs_by_id = {a.provisional_association_id: a for a in plan_obj.associations}
                    for app_a in applied_obj.associations:
                        orig_a = plan_assocs_by_id.get(app_a.provisional_association_id)
                        if not orig_a:
                            semantic_defects.append(f"Applied association '{app_a.provisional_association_id}' not present in reviewed plan.")
                            continue
                        if (
                            app_a.target_durable_security_id != orig_a.target_durable_security_id
                            or app_a.symbol != orig_a.symbol
                            or app_a.operating_mic != orig_a.operating_mic
                            or app_a.effective_from != orig_a.effective_from
                            or app_a.effective_to != orig_a.effective_to
                        ):
                            semantic_defects.append(
                                f"Applied association '{app_a.provisional_association_id}' attribute mismatch: "
                                f"actual(sec={app_a.target_durable_security_id}, mic={app_a.operating_mic}, from={app_a.effective_from}, to={app_a.effective_to}) != "
                                f"expected(sec={orig_a.target_durable_security_id}, mic={orig_a.operating_mic}, from={orig_a.effective_from}, to={orig_a.effective_to})."
                            )

                # B5: Check coherent authorization fields
                auth = applied_obj.authorization
                if not auth:
                    semantic_defects.append("Applied plan has missing authorization block.")
                else:
                    if not auth.is_authorized or not auth.application_authorized or auth.exact_plan_approval != "APPROVED":
                        semantic_defects.append("Applied plan authorization flags are contradictory, unauthorized, or not approved.")
                    if auth.scope != "REFERENCE_IDENTITY_REGISTRATION":
                        semantic_defects.append(f"Applied plan authorization scope '{auth.scope}' != 'REFERENCE_IDENTITY_REGISTRATION'.")
                    if manifest_data and auth.authorized_by != manifest_data.get("authorized_by"):
                        semantic_defects.append(f"Applied plan authorized_by '{auth.authorized_by}' != manifest '{manifest_data.get('authorized_by')}'.")

                if len(applied_obj.entities) != 130 or len(applied_obj.associations) != 130:
                    semantic_defects.append(f"Applied plan item count mismatch: {len(applied_obj.entities)} entities, {len(applied_obj.associations)} associations.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate {applied_name}: {e}")

    # 4. Verify execution summary agreement
    try:
        summary_raw = dest_summary.read_text(encoding="utf-8")
        summary_data = json.loads(summary_raw)
        if not isinstance(summary_data, dict) or not summary_data:
            semantic_defects.append("execution_summary.json is empty or not an object.")
        else:
            if summary_data.get("plan_id") != plan_id:
                semantic_defects.append(f"Execution summary plan_id '{summary_data.get('plan_id')}' != '{plan_id}'.")
            if summary_data.get("status") not in ("APPLIED", "COMMITTED"):
                semantic_defects.append(f"Execution summary status '{summary_data.get('status')}' is not APPLIED/COMMITTED.")
            if summary_data.get("applied_entities_count") != 130 or summary_data.get("applied_associations_count") != 130:
                semantic_defects.append(f"Execution summary additions mismatch: entities={summary_data.get('applied_entities_count')}, assocs={summary_data.get('applied_associations_count')}.")
            if applied_data and summary_data.get("target_state_digest_before") != applied_data.get("target_state_digest"):
                semantic_defects.append("Execution summary target_state_digest_before does not match applied plan pre-state target digest.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate execution_summary.json: {e}")

    # 5. Verify post-execution readback agreement
    try:
        readback_raw = dest_readback.read_text(encoding="utf-8")
        readback_data = json.loads(readback_raw)
        if not isinstance(readback_data, dict) or not readback_data:
            semantic_defects.append("post_execution_readback.json is empty or not an object.")
        else:
            if readback_data.get("securities_count") != 133 or readback_data.get("associations_count") != 133:
                semantic_defects.append(f"Readback entity count mismatch: secs={readback_data.get('securities_count')}, assocs={readback_data.get('associations_count')}; expected 133/133.")
            if summary_data and readback_data.get("target_state_digest_after") != summary_data.get("target_state_digest_after"):
                semantic_defects.append("Readback target_state_digest_after does not match execution summary.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate post_execution_readback.json: {e}")

    # 6. Verify pre-state target export agreement
    try:
        export_raw = dest_export.read_text(encoding="utf-8")
        export_data = json.loads(export_raw)
        if not isinstance(export_data, dict) or not export_data:
            semantic_defects.append("target_identity_state_export.json is empty or not an object.")
        else:
            export_digest = export_data.get("target_state_digest") or export_data.get("metadata", {}).get("target_state_digest")
            if export_digest != TARGET_BASELINE_STATE_DIGEST:
                semantic_defects.append(f"Target export digest '{export_digest}' != baseline '{TARGET_BASELINE_STATE_DIGEST}'.")
            state = export_data.get("state", {})
            secs = state.get("security_registry", {})
            if len(secs) != 3 or "ARX_SEC_0B4C728FE0BE" not in secs:
                semantic_defects.append("Target export does not contain canonical 3 pilot securities.")
    except Exception as e:
        semantic_defects.append(f"Failed to validate target_identity_state_export.json: {e}")

    # 7. Verify pre-apply backup agreement
    if dest_bak.is_file() and dest_bak.stat().st_size > 0:
        try:
            with SqliteIdentityRepository(str(dest_bak), read_only=True) as bck_repo:
                bck_digest = bck_repo.compute_target_state_digest()
                bck_secs = bck_repo.list_all_securities()
                bck_assocs = bck_repo.list_all_associations()
                if bck_digest != TARGET_BASELINE_STATE_DIGEST:
                    semantic_defects.append(f"Backup digest '{bck_digest}' != baseline '{TARGET_BASELINE_STATE_DIGEST}'.")
                if len(bck_secs) != 3 or len(bck_assocs) != 3:
                    semantic_defects.append(f"Backup contains {len(bck_secs)} secs and {len(bck_assocs)} assocs; expected exactly 3 pilots.")
        except Exception as e:
            semantic_defects.append(f"Backup repository inspection failed: {e}")

    # B4: Verify target database audit ledger linkage if target database exists
    if p_db.is_file():
        try:
            with SqliteIdentityRepository(str(p_db), read_only=True) as check_repo:
                with check_repo._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT plan_id, plan_content_digest, target_state_digest_before, target_state_digest_after, authorized_by "
                        "FROM applied_plan_audit_ledger WHERE plan_id = ?",
                        (plan_id,)
                    )
                    audit_rows = cur.fetchall()
                    if not audit_rows:
                        semantic_defects.append(f"Target database has zero audit ledger records for plan '{plan_id}'.")
                    else:
                        ledger_row = audit_rows[-1]
                        if applied_obj and ledger_row[1] != applied_obj.plan_content_digest:
                            semantic_defects.append(
                                f"Audit ledger plan_content_digest '{ledger_row[1]}' does not match applied receipt '{applied_obj.plan_content_digest}'."
                            )
                        if ledger_row[2] != TARGET_BASELINE_STATE_DIGEST:
                            semantic_defects.append(
                                f"Audit ledger target_state_digest_before '{ledger_row[2]}' does not match baseline '{TARGET_BASELINE_STATE_DIGEST}'."
                            )
                        if summary_data and ledger_row[3] != summary_data.get("target_state_digest_after"):
                            semantic_defects.append(
                                f"Audit ledger target_state_digest_after '{ledger_row[3]}' does not match execution summary '{summary_data.get('target_state_digest_after')}'."
                            )
                        if manifest_data and ledger_row[4] != manifest_data.get("authorized_by"):
                            semantic_defects.append(
                                f"Audit ledger authorized_by '{ledger_row[4]}' does not match approval manifest '{manifest_data.get('authorized_by')}'."
                            )
        except Exception as e:
            semantic_defects.append(f"Failed to inspect target database audit ledger: {e}")

    all_defects = missing_or_empty + semantic_defects
    if all_defects:
        raise RuntimeError(
            f"AUDIT_RETENTION_VERIFICATION_FAILED: Staging cleanup aborted. "
            f"Defects detected: {all_defects}. Staging directory preserved at '{p_staging}'."
        )

    receipt = {
        "status": "VERIFIED_AND_RETAINED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_directory": str(p_audit),
        "artifacts_verified_count": len(artifact_hashes),
        "artifacts": artifact_hashes,
    }
    receipt_path = p_audit / "audit_bundle_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # Clean staging ONLY AFTER 100% verification
    if clean_staging_on_success and p_staging and p_staging.exists():
        shutil.rmtree(str(p_staging))

    return receipt


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal: Controlled Identity Registration Plan Execution Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reviewed-plan", default=None, help="Path to reviewed unapproved IdentityAdoptionPlan JSON file.")
    parser.add_argument("--approval-manifest", default=None, help="Path to operator approval manifest JSON file (required unless --diagnose or --archive-audit is set).")
    parser.add_argument("--db-path", required=True, help="Explicit target SQLite database path.")
    parser.add_argument("--backup-path", default=None, help="Explicit backup database path (default: <db-path>.pre_apply.bak).")
    parser.add_argument("--output-applied-plan", default=None, help="Path to preserve approved applied plan artifact for audit.")
    parser.add_argument("--diagnose", action="store_true", default=False, help="Run read-only execution state diagnosis instead of applying plan.")
    parser.add_argument("--archive-audit", action="store_true", default=False, help="Archive and verify all 7 required audit artifacts into durable audit directory before staging cleanup.")
    parser.add_argument("--staging-dir", default=None, help="Path to staging directory containing pre-execution files.")
    parser.add_argument("--audit-dir", default=None, help="Path to durable audit directory.")

    args = parser.parse_args(argv)

    if args.diagnose:
        diag = diagnose_execution_state(
            db_path=args.db_path,
            reviewed_plan_path=args.reviewed_plan,
        )
        print(json.dumps(diag, indent=2))
        return 0

    if args.archive_audit:
        if not args.audit_dir:
            print("EXECUTION_ERROR: --audit-dir is required when --archive-audit is active.", file=sys.stderr)
            return 1
        try:
            receipt = archive_and_verify_audit_bundle(
                staging_dir=args.staging_dir,
                audit_dir=args.audit_dir,
                db_path=args.db_path,
                backup_path=args.backup_path,
            )
            print("AUDIT_BUNDLE_ARCHIVED_AND_VERIFIED:")
            print(json.dumps(receipt, indent=2))
            return 0
        except Exception as e:
            print(f"AUDIT_RETENTION_ERROR: {e}", file=sys.stderr)
            return 1

    if not args.reviewed_plan:
        print("EXECUTION_ERROR: --reviewed-plan is required when applying plan.", file=sys.stderr)
        return 1

    if not args.approval_manifest:
        print("EXECUTION_ERROR: --approval-manifest is required when not in --diagnose or --archive-audit mode.", file=sys.stderr)
        return 1

    try:
        outcome = execute_controlled_registration(
            reviewed_plan_path=args.reviewed_plan,
            approval_manifest_path=args.approval_manifest,
            db_path=args.db_path,
            backup_path=args.backup_path,
            output_applied_plan_path=args.output_applied_plan,
        )
        print("EXECUTION_SUCCESSFUL:")
        print(outcome.to_json(indent=2))
        return 0
    except IdentityAlreadyAppliedError as e:
        print(f"ALREADY_APPLIED: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"EXECUTION_ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
