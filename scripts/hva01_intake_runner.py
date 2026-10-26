#!/usr/bin/env python3
"""
HVA-01 Intake Runner Command-Line Interface.
Provides air-gapped, local command-line operations for session initialization,
case response submission, response locking, verification, and crash recovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hva01 import (
    AbstentionCode,
    AllocationEngine,
    AllocationLedger,
    AllocationRecord,
    AllocationStateError,
    AppendOnlyStore,
    AuthenticationError,
    AuthorizationError,
    ConfidenceLevel,
    EvidenceSufficiency,
    IntakeRunner,
    PackageAccessViolation,
    Permission,
    Principal,
    PriorFamiliarity,
    RatingsPayload,
    ResponseValidationError,
    RestrictedLabelAccessError,
    Role,
    SecurityEnforcer,
    StageClassification,
    StateTransitionError,
    StorageCorruptionError,
    StorageExhaustedError,
    VCPQualification,
    ValidationError,
)


def cmd_init(args: argparse.Namespace) -> int:
    try:
        runner = IntakeRunner.create_session(
            session_id=args.session_id,
            reviewer_pseudonym=args.pseudonym,
            assigned_package=args.package,
            protocol_version=args.protocol_version,
            protocol_document_sha256=args.protocol_sha,
            package_sha256=args.package_sha,
            base_dir=args.store_dir,
            total_cases=args.total_cases,
            expected_protocol_sha256=getattr(args, "expected_protocol_sha", None),
            expected_package_sha256=getattr(args, "expected_package_sha", None),
        )
        print(f"[OK] Session initialized: {runner.session.session_id}")
        print(f"     Reviewer Pseudonym:  {runner.session.reviewer_pseudonym}")
        print(f"     Assigned Package:    {runner.session.assigned_package}")
        print(f"     Total Cases:         {runner.session.total_cases}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to initialize session: {exc}", file=sys.stderr)
        return 1


def cmd_auth(args: argparse.Namespace) -> int:
    try:
        runner = IntakeRunner.resume_session(args.session_id, base_dir=args.store_dir)
        runner.authorize()
        print(f"[OK] Session authorized: {args.session_id}")
        print(f"     Current State:    {runner.fsm.current_state.value}")
        print(f"     Current Sequence: {runner.fsm.current_sequence}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to authorize session: {exc}", file=sys.stderr)
        return 1


def cmd_submit(args: argparse.Namespace) -> int:
    try:
        runner = IntakeRunner.resume_session(args.session_id, base_dir=args.store_dir)

        ratings = RatingsPayload(
            stage_classification=StageClassification(args.stage),
            vcp_qualification=VCPQualification(args.vcp),
            confidence_level=ConfidenceLevel(args.confidence),
            prior_familiarity=PriorFamiliarity(args.familiarity),
            evidence_sufficiency=EvidenceSufficiency(args.sufficiency),
            reviewer_reasoning=args.reasoning,
            abstention_code=AbstentionCode(args.abstention),
            contraction_count_observed=args.contractions,
            progressive_tightening_observed=args.tightening,
            volume_dry_up_observed=args.volume_dry_up,
            pivot_present=args.pivot,
            composite_score_100=args.composite_score,
        )

        record = runner.submit_and_lock_response(
            sequence_index=args.seq,
            case_id=args.case_id,
            ratings=ratings,
        )

        print(f"[OK] Response locked for Sequence {record.sequence_index} ({record.case_id})")
        print(f"     Record Hash: {record.record_hash}")
        print(f"     Next State:  {runner.fsm.current_state.value}")
        print(f"     Next Seq:    {runner.fsm.current_sequence}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to submit response: {exc}", file=sys.stderr)
        return 1


def cmd_withdraw(args: argparse.Namespace) -> int:
    try:
        runner = IntakeRunner.resume_session(args.session_id, base_dir=args.store_dir)
        result = runner.withdraw_session(justification=args.justification)
        print(f"[OK] Session withdrawn: {args.session_id}")
        print(f"     Committed Cases: {result['committed_cases']}")
        print(f"     Coverage Ratio:  {result['coverage_ratio']}")
        print(f"     Status:          SEALED (WITHDRAWN)")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to withdraw session: {exc}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        runner = IntakeRunner.resume_session(args.session_id, base_dir=args.store_dir)
        report = runner.verify_integrity()
        print(json.dumps(report, indent=2))
        return 0 if report["chain_valid"] else 2
    except Exception as exc:
        print(f"[ERROR] Verification failed: {exc}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    try:
        store = AppendOnlyStore(args.store_dir)
        status = store.get_crash_recovery_status(args.session_id)
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] Status inspection failed: {exc}", file=sys.stderr)
        return 1


def cmd_alloc_gen(args: argparse.Namespace) -> int:
    try:
        admin_principal = Principal.create_allocation_admin(args.admin_id)
        ledger = AllocationLedger(base_dir=args.store_dir)
        rec_a, rec_b = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version=args.protocol_version,
            package_a_sha256=args.package_a_sha,
            package_b_sha256=args.package_b_sha,
        )
        ledger.register_allocation(admin_principal, rec_a)
        ledger.register_allocation(admin_principal, rec_b)
        print(f"[OK] Generated and registered between-subjects allocations:")
        print(f"     A: {rec_a.allocation_id} -> {rec_a.reviewer_pseudonym} ({rec_a.assigned_package})")
        print(f"     B: {rec_b.allocation_id} -> {rec_b.reviewer_pseudonym} ({rec_b.assigned_package})")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to generate allocations: {exc}", file=sys.stderr)
        return 1


def cmd_alloc_lock(args: argparse.Namespace) -> int:
    try:
        admin_principal = Principal.create_allocation_admin(args.admin_id)
        ledger = AllocationLedger(base_dir=args.store_dir)
        locked = ledger.lock_allocation(admin_principal, args.allocation_id)
        print(f"[OK] Allocation locked: {locked.allocation_id} ({locked.reviewer_pseudonym})")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to lock allocation: {exc}", file=sys.stderr)
        return 1


def cmd_alloc_list(args: argparse.Namespace) -> int:
    try:
        admin_principal = Principal.create_allocation_admin(args.admin_id)
        ledger = AllocationLedger(base_dir=args.store_dir)
        allocations = ledger.list_allocations(admin_principal)
        print(json.dumps([a.to_dict() for a in allocations], indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to list allocations: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HVA-01 Intake Runner CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: init
    p_init = subparsers.add_parser("init", help="Initialize a new session")
    p_init.add_argument("--session-id", required=True, help="Session identifier (HVA01-SESS-...)")
    p_init.add_argument("--pseudonym", required=True, help="Reviewer pseudonym (SYNTH-REV-... or REV-...)")
    p_init.add_argument("--package", required=True, choices=["PACKAGE_A", "PACKAGE_B"], help="Package assignment")
    p_init.add_argument("--protocol-version", default="2.1.0", help="Protocol version")
    p_init.add_argument("--protocol-sha", required=True, help="64-char hex SHA-256 of ratified protocol document")
    p_init.add_argument("--package-sha", required=True, help="64-char hex SHA-256 of package manifest")
    p_init.add_argument("--expected-protocol-sha", default=None, help="Expected protocol SHA for verification")
    p_init.add_argument("--expected-package-sha", default=None, help="Expected package SHA for verification")
    p_init.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_init.add_argument("--total-cases", type=int, default=59, help="Total cases in evaluation sequence")
    p_init.set_defaults(func=cmd_init)

    # Subcommand: auth
    p_auth = subparsers.add_parser("auth", help="Authorize session and open draft")
    p_auth.add_argument("--session-id", required=True, help="Session identifier")
    p_auth.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_auth.set_defaults(func=cmd_auth)

    # Subcommand: submit
    p_submit = subparsers.add_parser("submit", help="Submit and permanently lock a case response")
    p_submit.add_argument("--session-id", required=True, help="Session identifier")
    p_submit.add_argument("--seq", type=int, required=True, help="Sequence index (1..59)")
    p_submit.add_argument("--case-id", required=True, help="Case identifier (HVA2-A-xx / HVA2-B-xx)")
    p_submit.add_argument("--stage", required=True, choices=[s.value for s in StageClassification], help="Stage classification")
    p_submit.add_argument("--vcp", required=True, choices=[v.value for v in VCPQualification], help="VCP qualification")
    p_submit.add_argument("--confidence", required=True, choices=[c.value for c in ConfidenceLevel], help="Confidence level")
    p_submit.add_argument("--familiarity", required=True, choices=[f.value for f in PriorFamiliarity], help="Prior familiarity")
    p_submit.add_argument("--sufficiency", required=True, choices=[s.value for s in EvidenceSufficiency], help="Evidence sufficiency")
    p_submit.add_argument("--reasoning", required=True, help="Reviewer technical reasoning (>= 5 chars)")
    p_submit.add_argument("--abstention", default="NONE", choices=[a.value for a in AbstentionCode], help="Abstention code")
    p_submit.add_argument("--contractions", type=int, default=None, help="Contraction count observed (0..10)")
    p_submit.add_argument("--tightening", type=lambda x: (str(x).lower() == 'true'), default=None, help="Progressive tightening")
    p_submit.add_argument("--volume-dry-up", type=lambda x: (str(x).lower() == 'true'), default=None, help="Volume dry-up observed")
    p_submit.add_argument("--pivot", type=lambda x: (str(x).lower() == 'true'), default=None, help="Pivot present")
    p_submit.add_argument("--composite-score", type=int, default=None, help="Composite score (0..100)")
    p_submit.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_submit.set_defaults(func=cmd_submit)

    # Subcommand: withdraw
    p_withdraw = subparsers.add_parser("withdraw", help="Formally withdraw session and seal log")
    p_withdraw.add_argument("--session-id", required=True, help="Session identifier")
    p_withdraw.add_argument("--justification", required=True, help="Reason for withdrawal")
    p_withdraw.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_withdraw.set_defaults(func=cmd_withdraw)

    # Subcommand: verify
    p_verify = subparsers.add_parser("verify", help="Verify session hash chain and integrity")
    p_verify.add_argument("--session-id", required=True, help="Session identifier")
    p_verify.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_verify.set_defaults(func=cmd_verify)

    # Subcommand: status
    p_status = subparsers.add_parser("status", help="Get session crash recovery status")
    p_status.add_argument("--session-id", required=True, help="Session identifier")
    p_status.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_status.set_defaults(func=cmd_status)

    # Subcommand: alloc-gen
    p_agen = subparsers.add_parser("alloc-gen", help="Generate canonical between-subjects allocations")
    p_agen.add_argument("--admin-id", default="ALLOC-ADMIN-01", help="Administrator principal ID")
    p_agen.add_argument("--protocol-version", default="2.1.0-ratified", help="Protocol version")
    p_agen.add_argument("--package-a-sha", required=True, help="Package A manifest SHA-256")
    p_agen.add_argument("--package-b-sha", required=True, help="Package B manifest SHA-256")
    p_agen.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_agen.set_defaults(func=cmd_alloc_gen)

    # Subcommand: alloc-lock
    p_alock = subparsers.add_parser("alloc-lock", help="Lock an allocation permanently")
    p_alock.add_argument("--allocation-id", required=True, help="Allocation ID to lock")
    p_alock.add_argument("--admin-id", default="ALLOC-ADMIN-01", help="Administrator principal ID")
    p_alock.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_alock.set_defaults(func=cmd_alloc_lock)

    # Subcommand: alloc-list
    p_alist = subparsers.add_parser("alloc-list", help="List registered allocations")
    p_alist.add_argument("--admin-id", default="ALLOC-ADMIN-01", help="Administrator principal ID")
    p_alist.add_argument("--store-dir", default="./hva01_sessions", help="Base storage directory")
    p_alist.set_defaults(func=cmd_alloc_list)

    parsed = parser.parse_args()
    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
