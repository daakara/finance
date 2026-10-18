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
import hashlib
import json
import os
from pathlib import Path
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal: Controlled Identity Registration Plan Execution Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reviewed-plan", required=True, help="Path to reviewed unapproved IdentityAdoptionPlan JSON file.")
    parser.add_argument("--approval-manifest", required=True, help="Path to operator approval manifest JSON file.")
    parser.add_argument("--db-path", required=True, help="Explicit target SQLite database path.")
    parser.add_argument("--backup-path", default=None, help="Explicit backup database path (default: <db-path>.pre_apply.bak).")
    parser.add_argument("--output-applied-plan", default=None, help="Path to preserve approved applied plan artifact for audit.")

    args = parser.parse_args(argv)

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
