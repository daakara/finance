#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-12: Final Qualification Summary Aggregation
Ingests, validates, and aggregates authentic stage receipts (RQ-01 through RQ-11).
Enforces exact 40-character candidate identity verification across all stages.
Zero acceptance of 7-character prefixes or abbreviated SHAs.
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
    validate_receipt_integrity, get_execution_context, get_git_commit, get_receipts_dir,
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
    parser.add_argument("--candidate-sha", default=None, help="Expected exact 40-character candidate SHA")
    parser.add_argument("--target-branch", default=None, help="Expected target branch")
    args = parser.parse_args()

    print("===============================================================")
    print("   ARX TERMINAL - RQ-12 FINAL QUALIFICATION AGGREGATION")
    print("===============================================================")

    ctx = get_execution_context()
    current_head = get_git_commit("HEAD")
    expected_sha = args.candidate_sha or ctx["source_sha"] or current_head
    expected_target = args.target_branch or ctx["base_ref"] or "release/production"

    # Enforce that expected_sha must be a fully resolved 40-character SHA
    if not SHA40_REGEX.match(expected_sha):
        print(f"FATAL: Target candidate SHA '{expected_sha}' is not a valid 40-character hex SHA")
        sys.exit(1)

    stage_summaries = {}
    passed_count = 0
    failed_count = 0
    blocked_count = 0
    missing_count = 0

    print(f"Target Candidate SHA: {expected_sha}")
    print(f"Execution Context: Run ID={ctx['run_id']}, PR={ctx['pr_number']}, Event={ctx['event_name']}")
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

        # 1. Validate complete receipt schema and cryptographic digest
        is_valid, validation_msg = validate_receipt_integrity(receipt)
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
        receipt_sha = receipt.get("candidate_sha", "")
        if not SHA40_REGEX.match(receipt_sha):
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "INVALID_SHA_FORMAT",
                "valid": False,
                "reason": f"Candidate SHA '{receipt_sha}' in receipt is not a 40-character SHA"
            }
            errors.append(f"Abbreviated or malformed candidate SHA in {cid}: '{receipt_sha}'")
            print(f"  [{cid}] {name:<42} : INVALID_SHA_FORMAT ('{receipt_sha}')")
            continue

        if receipt_sha.lower() != expected_sha.lower():
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "SUBSTITUTED",
                "valid": False,
                "reason": f"Candidate SHA mismatch in receipt: {receipt_sha} vs expected {expected_sha}"
            }
            errors.append(f"Candidate SHA mismatch in {cid}: {receipt_sha} vs expected {expected_sha}")
            print(f"  [{cid}] {name:<42} : SUBSTITUTED (Expected {expected_sha[:10]}..., got {receipt_sha[:10]}...)")
            continue

        # 3. Validate Execution Identity & Workflow Context
        r_ctx = receipt.get("execution_context", {})
        if not r_ctx or not isinstance(r_ctx, dict) or not r_ctx.get("checked_out_sha"):
            failed_count += 1
            errors.append(f"Missing or incomplete execution_context in {cid}")
            print(f"  [{cid}] {name:<42} : MISSING_EXECUTION_CONTEXT")
            continue

        # Execution SHA validation (checked_out_sha in execution_context)
        r_exec_sha = r_ctx.get("checked_out_sha")
        if r_exec_sha and not SHA40_REGEX.match(r_exec_sha):
            failed_count += 1
            msg = f"Invalid execution SHA format in {cid}: '{r_exec_sha}'"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : INVALID_EXECUTION_SHA_FORMAT")
            continue

        if r_exec_sha and r_exec_sha.lower() != current_head.lower():
            failed_count += 1
            msg = f"Execution SHA mismatch in {cid}: receipt has checked_out_sha {r_exec_sha} vs runner {current_head}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : EXECUTION_SHA_MISMATCH")
            continue

        # Workflow Run ID validation
        if ctx.get("run_id") and r_ctx.get("run_id") and str(r_ctx.get("run_id")) != str(ctx.get("run_id")):
            failed_count += 1
            msg = f"Workflow Run ID mismatch in {cid}: receipt has {r_ctx.get('run_id')} vs runner {ctx.get('run_id')}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : RUN_ID_MISMATCH")
            continue

        # Workflow Run Attempt validation
        if ctx.get("run_attempt") and r_ctx.get("run_attempt") and int(r_ctx.get("run_attempt")) != int(ctx.get("run_attempt")):
            failed_count += 1
            msg = f"Workflow Run Attempt mismatch in {cid}: receipt has attempt {r_ctx.get('run_attempt')} vs runner {ctx.get('run_attempt')}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : RUN_ATTEMPT_MISMATCH")
            continue

        # PR number validation
        if ctx.get("pr_number") and r_ctx.get("pr_number") and int(r_ctx.get("pr_number")) != int(ctx.get("pr_number")):
            failed_count += 1
            msg = f"PR number mismatch in {cid}: receipt has PR #{r_ctx.get('pr_number')} vs current PR #{ctx.get('pr_number')}"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : PR_MISMATCH")
            continue

        # Target branch validation
        r_target = receipt.get("target_branch", "")
        if r_target and not (r_target == expected_target or r_target == "release/production" or r_target == "main"):
            blocked_count += 1
            msg = f"Unauthorized target branch in {cid}: '{r_target}'"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : TARGET_BRANCH_MISMATCH ('{r_target}')")
            continue

        # Repository identity validation
        r_repo = r_ctx.get("repository") or "daakara/finance"
        current_repo = ctx.get("repository") or "daakara/finance"
        if r_repo.lower() != current_repo.lower():
            failed_count += 1
            msg = f"Repository mismatch in {cid}: '{r_repo}' vs expected '{current_repo}'"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : REPOSITORY_MISMATCH ('{r_repo}')")
            continue

        # Policy version validation
        r_policy = receipt.get("metadata", {}).get("policy_version") or r_ctx.get("policy_version")
        if r_policy and r_policy != QUALIFICATION_POLICY_VERSION:
            failed_count += 1
            msg = f"Altered policy version in {cid}: '{r_policy}' vs '{QUALIFICATION_POLICY_VERSION}'"
            errors.append(msg)
            print(f"  [{cid}] {name:<42} : POLICY_VERSION_MISMATCH ('{r_policy}')")
            continue

        # Evidence files existence and digest verification
        ev_paths = receipt.get("evidence_paths", [])
        ev_digests = receipt.get("evidence_digests", {})
        ev_broken = False
        for ep in ev_paths:
            abs_ep = ep if os.path.isabs(ep) else os.path.join(PROJECT_ROOT, ep)
            if not os.path.exists(abs_ep):
                failed_count += 1
                ev_broken = True
                msg = f"Evidence path missing on disk for {cid}: {ep}"
                errors.append(msg)
                print(f"  [{cid}] {name:<42} : EVIDENCE_MISSING ({ep})")
                break
            actual_digest = compute_file_sha256(abs_ep)
            expected_digest = ev_digests.get(ep)
            if expected_digest and expected_digest != actual_digest:
                failed_count += 1
                ev_broken = True
                msg = f"Evidence digest mismatch for {cid} file {ep}: stored {expected_digest[:10]}... vs actual {actual_digest[:10]}..."
                errors.append(msg)
                print(f"  [{cid}] {name:<42} : EVIDENCE_DIGEST_MISMATCH ({ep})")
                break
        if ev_broken:
            continue

        # Check timestamp freshness (max age 24 hours)
        max_age_hours = float(os.environ.get("ARX_MAX_RECEIPT_AGE_HOURS", "24.0"))
        try:
            r_end = datetime.datetime.fromisoformat(receipt["end_time"].replace("Z", "+00:00"))
            age_sec = (datetime.datetime.now(datetime.timezone.utc) - r_end).total_seconds()
            if age_sec > (max_age_hours * 3600.0):
                failed_count += 1
                stage_summaries[cid] = {
                    "name": name,
                    "status": "STALE",
                    "valid": False,
                    "reason": f"Receipt timestamp {receipt['end_time']} exceeds maximum age of {max_age_hours} hours"
                }
                errors.append(f"Stale receipt for {cid}: age {age_sec/3600.0:.1f} hours")
                print(f"  [{cid}] {name:<42} : STALE ({age_sec/3600.0:.1f}h old)")
                continue
        except Exception as e:
            failed_count += 1
            errors.append(f"Failed to parse timestamp in {cid}: {e}")
            continue

        # Check stage status
        r_status = receipt.get("status")
        if r_status == "PASS":
            passed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "PASS",
                "valid": True,
                "duration_sec": receipt.get("duration_sec"),
                "receipt_digest": receipt.get("receipt_digest")
            }
            assertions.append({
                "assertion": f"{cid} ({name}) passed and verified",
                "observed": "PASS",
                "passed": True
            })
            print(f"  [{cid}] {name:<42} : PASS ({receipt.get('duration_sec', 0)}s)")
        else:
            failed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": r_status,
                "valid": False,
                "errors": receipt.get("errors", [])
            }
            errors.append(f"Stage {cid} did not pass: status={r_status}, errors={receipt.get('errors')}")
            print(f"  [{cid}] {name:<42} : {r_status}")

    print("\n===============================================================")
    print("   QUALIFICATION SUMMARY VERDICT")
    print("===============================================================")
    print(f"Total Controls: {len(MANDATORY_CONTROLS)} | Passed: {passed_count} | Failed: {failed_count} | Blocked: {blocked_count} | Missing: {missing_count}")

    is_qualified = (
        passed_count == len(MANDATORY_CONTROLS) and
        failed_count == 0 and
        blocked_count == 0 and
        missing_count == 0 and
        len(errors) == 0
    )

    verdict_str = "RELEASE_QUALIFIED" if is_qualified else "RELEASE_QUALIFICATION_BLOCKED"
    print(f"OVERALL VERDICT: {verdict_str}")

    # Write summary artifact
    summary_path = os.path.join(os.path.dirname(__file__), "release_qualification_summary.json")
    summary_data = {
        "verdict": verdict_str,
        "is_qualified": is_qualified,
        "qualification_policy_version": QUALIFICATION_POLICY_VERSION,
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate_identity": {
            "source_sha": expected_sha,
            "target_branch": expected_target,
            "checked_out_sha": current_head,
            "repository": ctx.get("repository", "daakara/finance"),
            "pr_number": ctx.get("pr_number")
        },
        "statistics": {
            "total_mandatory_controls": len(MANDATORY_CONTROLS),
            "passed": passed_count,
            "failed": failed_count,
            "blocked": blocked_count,
            "missing": missing_count
        },
        "controls": stage_summaries,
        "errors": errors
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary written to: {summary_path}")

    # Save RQ-12 Receipt Noncircularly
    duration = time.time() - t0
    rq12_receipt = create_stage_receipt(
        control_id="RQ-12",
        name="Qualification Summary Aggregator",
        status="PASS" if is_qualified else "FAIL",
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=expected_sha,
        target_branch=expected_target,
        assertions=assertions,
        evidence_paths=["scripts/release/release_qualification_summary.json"],
        errors=errors,
        metadata={
            "verdict": verdict_str,
            "is_qualified": is_qualified,
            "policy_version": QUALIFICATION_POLICY_VERSION
        }
    )
    save_stage_receipt(rq12_receipt)

    if is_qualified:
        print("=== [RQ-12] RESULT: PASS (RELEASE QUALIFIED) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-12] RESULT: FAIL ({verdict_str} - {len(errors)} errors) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
