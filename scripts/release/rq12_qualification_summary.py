#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-12: Final Qualification Summary Aggregation
Directive 074-A4.1: Hosted PR Merge Identity Contract Closure.
Ingests, validates, and aggregates authentic stage receipts (RQ-01 through RQ-11).
Enforces single authoritative identity contract across Source, Execution, and Release dimensions.
Zero acceptance of 7-character prefixes, abbreviated SHAs, or substituted commits.
Validates execution identity, workflow run ID, attempt, PR number, repository, and policy version.
Fails closed with RELEASE_QUALIFICATION_BLOCKED if any stage fails, is missing, stale,
substituted, malformed, or has broken evidence linkage.
Emits noncircular summary and receipt.
"""

import sys, os, json, time, datetime, argparse

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import (
    create_stage_receipt, save_stage_receipt, load_stage_receipt,
    validate_receipt_integrity, resolve_identity_contract, get_git_commit, get_receipts_dir,
    compute_file_sha256, QUALIFICATION_POLICY_VERSION, SHA40_REGEX
)

RECEIPTS_DIR = get_receipts_dir()

MANDATORY_CONTROLS = [
    ("RQ-01", "Candidate Identity & Lineage"),
    ("RQ-02", "Release Diff Integrity"),
    ("RQ-03", "Backend Regression Suite"),
    ("RQ-04", "Frontend Regression & Typecheck"),
    ("RQ-05", "Cross-Engine Invariants"),
    ("RQ-06", "Governance & Backlog Invariants"),
    ("RQ-07", "Deployment Safety & Secret Isolation"),
    ("RQ-08", "Artifact Provenance & Digest Binding"),
    ("RQ-09", "Four-Way Cross-Platform Compatibility"),
    ("RQ-10", "Release Configuration Integrity"),
    ("RQ-11", "Rollback Readiness & Target Verification")
]

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    parser = argparse.ArgumentParser(description="RQ-12: Qualification Summary Aggregator")
    parser.add_argument("--candidate-sha", default=None, help="Expected exact 40-character source candidate SHA")
    parser.add_argument("--execution-sha", default=None, help="Expected exact 40-character execution commit SHA")
    parser.add_argument("--target-branch", default=None, help="Expected target branch")
    args = parser.parse_args()

    print("===============================================================")
    print("   ARX TERMINAL - RQ-12 FINAL QUALIFICATION AGGREGATION")
    print("===============================================================")

    # 1. Authoritative Identity Resolution
    ok, identity, err_msg = resolve_identity_contract()
    if not ok:
        print(f"FATAL: Identity contract resolution failed closed: {err_msg}")
        errors.append(f"Identity resolution failure: {err_msg}")
        sys.exit(1)

    expected_source_sha = args.candidate_sha or identity["source_sha"]
    expected_execution_sha = args.execution_sha or identity["execution_sha"]
    expected_target = args.target_branch or identity["target_branch"]

    # Enforce that expected_source_sha must be a fully resolved 40-character SHA
    if not SHA40_REGEX.match(expected_source_sha):
        print(f"FATAL: Target candidate SHA '{expected_source_sha}' is not a valid 40-character hex SHA")
        sys.exit(1)
    if not SHA40_REGEX.match(expected_execution_sha):
        print(f"FATAL: Execution SHA '{expected_execution_sha}' is not a valid 40-character hex SHA")
        sys.exit(1)

    stage_summaries = {}
    passed_count = 0
    failed_count = 0
    blocked_count = 0
    missing_count = 0

    print(f"Target Source Candidate SHA: {expected_source_sha}")
    print(f"Verified Execution Commit SHA: {expected_execution_sha}")
    print(f"Execution Context: Run ID={identity['workflow_run_id']}, PR={identity['pr_number']}, Event={identity['event_name']}")
    print(f"Target Branch: {expected_target}")
    print(f"Qualification Policy: {QUALIFICATION_POLICY_VERSION}")
    print("\nAuditing stage receipts:")

    # Check for unknown or rogue receipts in RECEIPTS_DIR
    if os.path.exists(RECEIPTS_DIR):
        valid_filenames = {f"{cid.lower().replace('-', '')}_receipt.json" for cid, _ in MANDATORY_CONTROLS}
        valid_filenames.add("rq12_receipt.json")
        for fname in sorted(os.listdir(RECEIPTS_DIR)):
            if fname.endswith("_receipt.json") and fname not in valid_filenames:
                errors.append(f"Unexpected or rogue receipt file detected: {fname}")
                blocked_count += 1
                print(f"  [ROGUE_RECEIPT] {fname:<37} : UNEXPECTED")

    for cid, name in MANDATORY_CONTROLS:
        receipt = load_stage_receipt(cid)
        if not receipt:
            missing_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "MISSING",
                "valid": False,
                "reason": f"Receipt file for {cid} not found in {RECEIPTS_DIR}"
            }
            errors.append(f"Missing receipt for mandatory stage {cid}")
            print(f"  [{cid}] {name:<42} : MISSING (No evidence emitted)")
            continue

        # 1. Validate complete receipt schema, source & execution digests
        is_valid, validation_msg = validate_receipt_integrity(
            receipt,
            expected_source_sha=expected_source_sha,
            expected_execution_sha=expected_execution_sha
        )
        if not is_valid:
            failed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "CORRUPTED",
                "valid": False,
                "reason": validation_msg
            }
            errors.append(f"Tampered or corrupted receipt for {cid}: {validation_msg}")
            print(f"  [{cid}] {name:<42} : CORRUPTED ({validation_msg})")
            continue

        # 2. Strict Exact 40-Character Candidate Identity Verification
        receipt_cand_sha = receipt.get("candidate_sha", "")
        receipt_src_sha = receipt.get("source_sha", "")
        receipt_exec_sha = receipt.get("execution_sha", "")

        if receipt_cand_sha.lower() != expected_source_sha.lower() or receipt_src_sha.lower() != expected_source_sha.lower():
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "SUBSTITUTED",
                "valid": False,
                "reason": f"Candidate SHA mismatch in receipt: {receipt_src_sha} vs expected {expected_source_sha}"
            }
            errors.append(f"Candidate SHA mismatch in {cid}: {receipt_src_sha} vs expected {expected_source_sha}")
            print(f"  [{cid}] {name:<42} : SUBSTITUTED (Expected {expected_source_sha[:10]}..., got {receipt_src_sha[:10]}...)")
            continue

        if receipt_exec_sha.lower() != expected_execution_sha.lower():
            failed_count += 1
            msg = f"Execution SHA mismatch in {cid}: receipt has {receipt_exec_sha} vs expected {expected_execution_sha}"
            stage_summaries[cid] = {
                "name": name,
                "status": "EXECUTION_SHA_MISMATCH",
                "valid": False,
                "reason": msg
            }
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : EXECUTION_SHA_MISMATCH (Expected {expected_execution_sha[:10]}..., got {receipt_exec_sha[:10]}...)")
            continue

        # 3. Validate Execution Identity & Workflow Context
        r_ctx = receipt.get("execution_context", {})

        # Repository match
        if r_ctx.get("repository") != identity["repository"]:
            failed_count += 1
            msg = f"Repository mismatch in {cid}: {r_ctx.get('repository')} vs {identity['repository']}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : REPOSITORY_MISMATCH")
            continue

        # Workflow Run ID validation
        if identity.get("workflow_run_id") and r_ctx.get("workflow_run_id") != identity["workflow_run_id"]:
            failed_count += 1
            msg = f"Workflow Run ID mismatch in {cid}: receipt has {r_ctx.get('workflow_run_id')} vs current {identity['workflow_run_id']}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : RUN_ID_MISMATCH")
            continue

        # Workflow Run Attempt validation
        if identity.get("workflow_run_attempt") and int(r_ctx.get("workflow_run_attempt", 0)) != int(identity["workflow_run_attempt"]):
            failed_count += 1
            msg = f"Workflow Run Attempt mismatch in {cid}: receipt has attempt {r_ctx.get('workflow_run_attempt')} vs current {identity['workflow_run_attempt']}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : RUN_ATTEMPT_MISMATCH")
            continue

        # PR number validation
        if identity.get("pr_number") and r_ctx.get("pr_number") != identity["pr_number"]:
            failed_count += 1
            msg = f"PR number mismatch in {cid}: receipt has PR #{r_ctx.get('pr_number')} vs current PR #{identity['pr_number']}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : PR_MISMATCH")
            continue

        # Target branch validation
        r_target = receipt.get("target_branch", "")
        clean_target = r_target.replace("refs/heads/", "").strip()
        clean_expected = expected_target.replace("refs/heads/", "").strip()
        if clean_target not in (clean_expected, "release/production", "main"):
            blocked_count += 1
            msg = f"Unauthorized target branch in {cid}: '{clean_target}'"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : TARGET_BRANCH_MISMATCH ('{clean_target}')")
            continue

        # Policy Version validation
        r_policy_ctx = r_ctx.get("qualification_policy_version")
        r_policy_meta = receipt.get("metadata", {}).get("policy_version")
        if r_policy_ctx != QUALIFICATION_POLICY_VERSION or (r_policy_meta and r_policy_meta != QUALIFICATION_POLICY_VERSION):
            failed_count += 1
            bad_pol = r_policy_ctx if r_policy_ctx != QUALIFICATION_POLICY_VERSION else r_policy_meta
            msg = f"Qualification policy version mismatch in {cid}: {bad_pol} vs {QUALIFICATION_POLICY_VERSION}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : POLICY_VERSION_MISMATCH ({bad_pol})")
            continue

        # 4. Status Check
        stage_status = receipt.get("status")
        if stage_status != "PASS":
            failed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": stage_status,
                "valid": False,
                "reason": f"Stage {cid} did not achieve PASS status (observed: {stage_status})"
            }
            errors.append(f"Stage {cid} failed with status {stage_status}")
            print(f"  [{cid}] {name:<42} : {stage_status}")
            continue

        # 5. Evidence Path & Digest Validation
        evidence_paths = receipt.get("evidence_paths", [])
        evidence_digests = receipt.get("evidence_digests", {})
        evidence_ok = True
        missing_ev = []
        for ep in evidence_paths:
            abs_ep = ep if os.path.isabs(ep) else os.path.join(PROJECT_ROOT, ep)
            if not os.path.exists(abs_ep):
                evidence_ok = False
                missing_ev.append(ep)
            else:
                recorded_digest = evidence_digests.get(ep)
                if recorded_digest and recorded_digest != "FILE_NOT_FOUND":
                    actual_digest = compute_file_sha256(abs_ep)
                    if recorded_digest != actual_digest:
                        evidence_ok = False
                        missing_ev.append(f"{ep} (Digest tampered: {actual_digest[:8]} != {recorded_digest[:8]})")

        if not evidence_ok:
            failed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "EVIDENCE_MISSING",
                "valid": False,
                "reason": f"Required evidence artifacts missing or tampered: {missing_ev}"
            }
            errors.append(f"Stage {cid} missing or tampered evidence artifacts: {missing_ev}")
            print(f"  [{cid}] {name:<42} : EVIDENCE_MISSING ({', '.join(missing_ev)})")
            continue

        # Stage passed all checks
        passed_count += 1
        duration_val = receipt.get("duration_sec", 0.0)
        stage_summaries[cid] = {
            "name": name,
            "status": "PASS",
            "valid": True,
            "duration_sec": duration_val
        }
        print(f"  [{cid}] {name:<42} : PASS ({duration_val}s)")

    print("\n===============================================================")
    print("   QUALIFICATION SUMMARY VERDICT")
    print("===============================================================")
    total_controls = len(MANDATORY_CONTROLS)
    print(f"Total Controls: {total_controls} | Passed: {passed_count} | Failed: {failed_count} | Blocked: {blocked_count} | Missing: {missing_count}")

    overall_pass = (passed_count == total_controls and len(errors) == 0)
    verdict = "RELEASE_QUALIFIED" if overall_pass else "RELEASE_QUALIFICATION_BLOCKED"

    print(f"OVERALL VERDICT: {verdict}")

    duration = time.time() - t0
    assertions.append({
        "assertion": "All 11 mandatory release qualification stages passed",
        "observed": {
            "total": total_controls,
            "passed": passed_count,
            "failed": failed_count,
            "blocked": blocked_count,
            "missing": missing_count
        },
        "passed": overall_pass
    })

    summary_payload = {
        "summary_id": "ARX_RELEASE_QUALIFICATION_SUMMARY",
        "verdict": verdict,
        "policy_version": QUALIFICATION_POLICY_VERSION,
        "candidate_sha": expected_source_sha,
        "source_sha": expected_source_sha,
        "execution_sha": expected_execution_sha,
        "target_branch": expected_target,
        "execution_context": identity,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_controls": total_controls,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "blocked_count": blocked_count,
        "missing_count": missing_count,
        "stages": stage_summaries,
        "errors": errors
    }

    summary_file = os.path.join(PROJECT_ROOT, "scripts", "release", "release_qualification_summary.json")
    with open(summary_file, "w", encoding="utf-8") as out:
        json.dump(summary_payload, out, indent=2)

    # RQ-12 own receipt (noncircular)
    rq12_status = "PASS" if overall_pass else "FAIL"
    receipt = create_stage_receipt(
        control_id="RQ-12",
        name="Qualification Summary Aggregation",
        status=rq12_status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=expected_source_sha,
        target_branch=expected_target,
        assertions=assertions,
        evidence_paths=["scripts/release/release_qualification_summary.json"],
        errors=errors,
        metadata={
            "summary_verdict": verdict,
            "passed_controls": passed_count,
            "total_controls": total_controls
        }
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Summary written to: {summary_file}")

    if overall_pass:
        print("=== [RQ-12] RESULT: PASS (RELEASE QUALIFIED) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-12] RESULT: FAIL ({verdict} - {len(errors)} errors) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
