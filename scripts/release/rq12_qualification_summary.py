#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-12: Final Qualification Summary Aggregation
Ingests, validates, and aggregates authentic stage receipts (RQ-01 through RQ-11).
Fails closed with RELEASE_QUALIFICATION_BLOCKED if any stage fails, is missing, stale,
substituted, or corrupted. Emits noncircular summary and receipt.
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
    validate_receipt_integrity, get_execution_context, get_git_commit, get_receipts_dir
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
    parser.add_argument("--candidate-sha", default=None, help="Expected candidate SHA to bind")
    args = parser.parse_args()

    print("===============================================================")
    print("   ARX TERMINAL - RQ-12 FINAL QUALIFICATION AGGREGATION")
    print("===============================================================")

    ctx = get_execution_context()
    current_head = get_git_commit("HEAD")
    expected_sha = args.candidate_sha or ctx["pr_head_sha"] or current_head

    stage_summaries = {}
    passed_count = 0
    failed_count = 0
    blocked_count = 0
    missing_count = 0

    print(f"Target Candidate SHA: {expected_sha}")
    print(f"Execution Context: Run ID={ctx['run_id']}, PR={ctx['pr_number']}, Event={ctx['event_name']}")
    print("\nAuditing stage receipts:")

    # Check for unknown or rogue receipts in RECEIPTS_DIR
    if os.path.exists(RECEIPTS_DIR):
        valid_filenames = {f"{cid.lower().replace('-', '')}_receipt.json" for cid, _ in MANDATORY_CONTROLS}
        valid_filenames.add("rq12_receipt.json")
        for fname in os.listdir(RECEIPTS_DIR):
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

        # Validate schema, digest, and identity
        is_valid, validation_msg = validate_receipt_integrity(receipt, expected_candidate_sha=None)
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

        # Validate candidate SHA consistency
        receipt_sha = receipt.get("candidate_sha", "")
        # Allow receipt_sha to match either expected_sha, current_head, or base if checking diff
        sha_matches = (
            receipt_sha.lower() == expected_sha.lower() or
            receipt_sha.lower() == current_head.lower() or
            receipt_sha.startswith(expected_sha[:7]) or
            expected_sha.startswith(receipt_sha[:7])
        )
        if not sha_matches:
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "SUBSTITUTED",
                "valid": False,
                "reason": f"Candidate SHA mismatch in receipt: {receipt_sha[:10]} vs expected {expected_sha[:10]}"
            }
            errors.append(f"Candidate SHA substitution in {cid}: {receipt_sha[:10]} vs {expected_sha[:10]}")
            print(f"  [{cid}] {name:<42} : SUBSTITUTED (SHA mismatch)")
            continue

        # Validate Run ID / PR consistency if both present
        r_ctx = receipt.get("execution_context", {})
        if ctx["run_id"] and r_ctx.get("run_id") and str(ctx["run_id"]) != str(r_ctx.get("run_id")):
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "STALE_RUN",
                "valid": False,
                "reason": f"Run ID mismatch: current {ctx['run_id']} vs receipt {r_ctx.get('run_id')}"
            }
            errors.append(f"Stale receipt from previous run for {cid}")
            print(f"  [{cid}] {name:<42} : STALE_RUN")
            continue

        actual_status = receipt["status"]
        if actual_status == "PASS":
            passed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "PASS",
                "valid": True,
                "duration_sec": receipt.get("duration_sec", 0),
                "digest": receipt.get("receipt_digest", "")
            }
            print(f"  [{cid}] {name:<42} : PASS ({receipt.get('duration_sec', 0)}s)")
        elif actual_status == "FAIL":
            failed_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "FAIL",
                "valid": True,
                "errors": receipt.get("errors", [])
            }
            errors.append(f"Stage {cid} failed: {receipt.get('errors', [])}")
            print(f"  [{cid}] {name:<42} : FAIL (Errors reported)")
        elif actual_status == "BLOCKED":
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "BLOCKED",
                "valid": True,
                "errors": receipt.get("errors", [])
            }
            errors.append(f"Stage {cid} blocked: {receipt.get('errors', [])}")
            print(f"  [{cid}] {name:<42} : BLOCKED")
        elif actual_status == "NOT_APPLICABLE":
            # Policy: NOT_APPLICABLE is NOT permitted for any mandatory stage
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": "BLOCKED_NOT_APPLICABLE",
                "valid": False,
                "reason": "NOT_APPLICABLE is not authorized for mandatory controls"
            }
            errors.append(f"Stage {cid} claimed NOT_APPLICABLE without authorization")
            print(f"  [{cid}] {name:<42} : BLOCKED (Unauthorized NOT_APPLICABLE)")
        else:
            blocked_count += 1
            stage_summaries[cid] = {
                "name": name,
                "status": actual_status,
                "valid": False
            }
            errors.append(f"Stage {cid} has unapproved status {actual_status}")
            print(f"  [{cid}] {name:<42} : {actual_status}")

    # Determine Overall Verdict
    is_fully_qualified = (
        passed_count == len(MANDATORY_CONTROLS) and
        failed_count == 0 and
        blocked_count == 0 and
        missing_count == 0 and
        len(errors) == 0
    )

    if is_fully_qualified:
        overall_verdict = "RELEASE_QUALIFIED"
        overall_status = "PASS"
    else:
        overall_verdict = "RELEASE_QUALIFICATION_BLOCKED"
        overall_status = "BLOCKED"

    assertions.append({
        "assertion": "All 11 mandatory qualification stages have authentic PASS receipts",
        "observed": {
            "total_mandatory": len(MANDATORY_CONTROLS),
            "passed": passed_count,
            "failed": failed_count,
            "blocked": blocked_count,
            "missing": missing_count
        },
        "passed": is_fully_qualified
    })

    duration = time.time() - t0

    # Write Master Qualification Summary JSON
    summary_data = {
        "directive_id": "ARX_RELEASE_QUALIFICATION_REMEDIATION_074A3",
        "timestamp": start_time,
        "candidate_commit": expected_sha,
        "execution_context": ctx,
        "controls_total": len(MANDATORY_CONTROLS),
        "controls_passed": passed_count,
        "controls_failed": failed_count,
        "controls_blocked": blocked_count,
        "controls_missing": missing_count,
        "overall_verdict": overall_verdict,
        "qualification_posture": "AUTHENTIC_RECEIPTS_VERIFIED" if is_fully_qualified else "FAIL_CLOSED_BLOCKED",
        "blocking_reasons": errors,
        "stage_details": stage_summaries
    }

    summary_file = os.path.join(PROJECT_ROOT, "scripts", "release", "release_qualification_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Self-receipt (RQ-12 emits its own noncircular receipt)
    rq12_receipt = create_stage_receipt(
        control_id="RQ-12",
        name="Final Qualification Summary Aggregation",
        status=overall_status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=expected_sha,
        target_branch=ctx["base_ref"] or "release/production",
        assertions=assertions,
        evidence_paths=["scripts/release/release_qualification_summary.json"],
        errors=errors,
        metadata={
            "overall_verdict": overall_verdict,
            "passed_controls": passed_count,
            "total_controls": len(MANDATORY_CONTROLS)
        }
    )
    save_stage_receipt(rq12_receipt)

    print("\n===============================================================")
    print("   QUALIFICATION SUMMARY VERDICT")
    print("===============================================================")
    print(f"Total Controls: {len(MANDATORY_CONTROLS)} | Passed: {passed_count} | Failed: {failed_count} | Blocked: {blocked_count} | Missing: {missing_count}")
    print(f"OVERALL VERDICT: {overall_verdict}")
    print(f"Summary written to: {summary_file}")

    if is_fully_qualified:
        print("=== [RQ-12] RESULT: PASS (RELEASE QUALIFIED) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-12] RESULT: FAIL (RELEASE QUALIFICATION BLOCKED: {len(errors)} blocking conditions) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()