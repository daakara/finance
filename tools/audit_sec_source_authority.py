"""
Independent SEC Source Authority & PDS Closure Replay Auditor (Epoch 001)

Governance:
- Pure READ-ONLY independent reconstruction.
- Does NOT import or share certification decision helpers:
  - NO PdsSourceClosureTheorem.evaluate()
  - NO ScopedDomainDerivationTheorem.evaluate()
  - NO DecisionInputEvaluator.evaluate()
- Directly reconstructs exact set differences, content conflicts, hash chains,
  and audit proofs from raw bundle artifacts.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Set, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash


class IndependentSecSourceAuthorityAuditor:
    """
    Independent read-only auditor for SEC Source Authority bundles and closure claims.
    Reconstructs truth strictly from raw artifact inputs without shared decision logic.
    """

    def __init__(self):
        self.mode = "READ_ONLY"

    def audit_raw_eod_and_accounted_artifacts(
        self,
        raw_eod_content: str,
        accounted_records: List[Dict[str, Any]],
        expected_business_day: str,
    ) -> Dict[str, Any]:
        """
        Directly and independently audits a raw EOD string and list of accounted artifacts.
        """
        trace_log: List[str] = []
        failures: List[str] = []

        trace_log.append("Step 1: Commencing independent raw EOD audit")

        # 1. Parse raw EOD lines directly
        lines = raw_eod_content.splitlines()
        expected_filenames: Set[str] = set()
        duplicate_filenames: List[str] = []
        malformed_lines: List[str] = []
        unknown_extension_lines: List[str] = []

        for line_idx, line in enumerate(lines, start=1):
            s = line.strip()
            if not s:
                continue

            # Check format
            lower = s.lower()
            if not (lower.endswith(".dissem") or lower.endswith(".paper") or lower.endswith(".pr") or lower.endswith(".pc")):
                unknown_extension_lines.append(s)

            if len(s) < 5 or " " in s:
                malformed_lines.append(s)

            if s in expected_filenames:
                duplicate_filenames.append(s)
            expected_filenames.add(s)

        if duplicate_filenames:
            failures.append(f"AUDIT_DUPLICATE_FILENAMES: {len(duplicate_filenames)}")
        if malformed_lines:
            failures.append(f"AUDIT_MALFORMED_LINES: {len(malformed_lines)}")
        if unknown_extension_lines:
            failures.append(f"AUDIT_UNKNOWN_EXTENSIONS: {len(unknown_extension_lines)}")

        trace_log.append(f"Step 2: Parsed {len(expected_filenames)} distinct expected files from EOD")

        # 2. Audit accounted records
        accounted_by_day: Dict[str, List[Dict[str, Any]]] = {}
        hashes_by_filename: Dict[str, Set[str]] = {}
        unclassified_cross_day: List[str] = []

        for rec in accounted_records:
            fname = rec.get("pds_filename", "")
            src_day = rec.get("source_business_day", "")
            rcpt_day = rec.get("receipt_day", "")
            method = rec.get("receipt_method", "")
            raw_sha = rec.get("raw_bytes_sha256", "")

            accounted_by_day.setdefault(src_day, []).append(rec)
            hashes_by_filename.setdefault(fname, set()).add(raw_sha)

            if rcpt_day != src_day and method not in (
                "PDS_RETRANSMISSION", "PDS_STAGING_RECOVERY", "PDS_FTP_RECOVERY", "PDS_HELPDESK_RECOVERY"
            ):
                unclassified_cross_day.append(fname)

        # Check content conflicts (same filename, differing bytes)
        content_conflicts = [fn for fn, hset in hashes_by_filename.items() if len(hset) > 1]
        if content_conflicts:
            failures.append(f"AUDIT_CONTENT_CONFLICTS: {len(content_conflicts)} files ({content_conflicts[:3]})")

        if unclassified_cross_day:
            failures.append(f"AUDIT_UNCLASSIFIED_CROSS_DAY: {len(unclassified_cross_day)} files")

        trace_log.append("Step 3: Accounted records integrity and cross-day classification checked")

        # 3. Exact Set Reconciliation for target business day
        curr_day_records = accounted_by_day.get(expected_business_day, [])
        accounted_filenames = {r.get("pds_filename") for r in curr_day_records}

        missing = sorted(list(expected_filenames - accounted_filenames))
        unexpected = sorted(list(accounted_filenames - expected_filenames))

        if missing:
            failures.append(f"AUDIT_MISSING_FILES: {len(missing)} files missing from accounted set")
        if unexpected:
            failures.append(f"AUDIT_UNEXPECTED_FILES: {len(unexpected)} unexpected files in accounted set")

        trace_log.append(f"Step 4: Set reconciliation computed (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

        reconstructed_eod_hash = compute_canonical_hash(sorted(list(expected_filenames)))
        reconstructed_acc_hash = compute_canonical_hash(sorted(list(accounted_filenames)))

        if reconstructed_eod_hash != reconstructed_acc_hash:
            failures.append("AUDIT_SET_HASH_MISMATCH")

        audit_pass = len(failures) == 0
        trace_log.append(f"Step 5: Independent audit completed with result: {'PASS' if audit_pass else 'FAIL'}")

        trace_text = "\n".join(trace_log)
        proof_trace_sha256 = hashlib.sha256(trace_text.encode("utf-8")).hexdigest()

        return {
            "audit_result": "PASS" if audit_pass else "FAIL",
            "auditor_mode": self.mode,
            "reconstructed_eod_expected_count": len(expected_filenames),
            "reconstructed_pds_accounted_count": len(accounted_filenames),
            "reconstructed_missing_count": len(missing),
            "reconstructed_unexpected_count": len(unexpected),
            "reconstructed_content_conflict_count": len(content_conflicts),
            "reconstructed_eod_set_hash": reconstructed_eod_hash,
            "reconstructed_pds_accounted_set_hash": reconstructed_acc_hash,
            "proof_trace_sha256": proof_trace_sha256,
            "trace_log": trace_log,
            "failure_reasons": failures,
        }

    def audit_evidence_bundle_chain(
        self,
        current_bundle_dict: Dict[str, Any],
        previous_bundle_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Independently audits bundle manifest, seals, and hash chain.
        """
        trace: List[str] = []
        failures: List[str] = []

        trace.append("Step 1: Auditing evidence bundle integrity")

        b_id = current_bundle_dict.get("bundle_id")
        seq = current_bundle_dict.get("bundle_sequence")
        prev_id = current_bundle_dict.get("previous_bundle_id")
        prev_sha = current_bundle_dict.get("previous_bundle_sha256")
        stored_sha = current_bundle_dict.get("bundle_sha256")

        # Recompute hash without bundle_sha256 field
        copy_data = dict(current_bundle_dict)
        copy_data.pop("bundle_sha256", None)
        recomputed_sha = compute_canonical_hash(copy_data)

        if stored_sha and stored_sha != recomputed_sha:
            failures.append(f"BUNDLE_SHA_MISMATCH: Stored {stored_sha} != Recomputed {recomputed_sha}")

        trace.append(f"Step 2: Bundle hash verified for {b_id}")

        if seq == 1:
            if prev_id is not None or prev_sha is not None:
                failures.append("GENESIS_BUNDLE_MUST_HAVE_NULL_PREVIOUS")
            trace.append("Step 3: Genesis bundle chain link verified")
        else:
            if not previous_bundle_dict:
                failures.append("PREVIOUS_BUNDLE_REQUIRED_FOR_NON_GENESIS")
            else:
                prev_copy = dict(previous_bundle_dict)
                prev_copy.pop("bundle_sha256", None)
                recomputed_prev_sha = compute_canonical_hash(prev_copy)
                if prev_sha != recomputed_prev_sha:
                    failures.append(f"PREVIOUS_BUNDLE_HASH_MISMATCH: Link {prev_sha} != Actual {recomputed_prev_sha}")
                trace.append("Step 3: Sequential bundle hash chain link verified")

        audit_pass = len(failures) == 0
        trace.append(f"Step 4: Bundle chain audit finished: {'PASS' if audit_pass else 'FAIL'}")

        trace_text = "\n".join(trace)
        proof_trace_sha = hashlib.sha256(trace_text.encode("utf-8")).hexdigest()

        return {
            "audit_result": "PASS" if audit_pass else "FAIL",
            "auditor_mode": self.mode,
            "bundle_id": b_id,
            "recomputed_bundle_sha256": recomputed_sha,
            "proof_trace_sha256": proof_trace_sha,
            "trace_log": trace,
            "failure_reasons": failures,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_sec_source_authority.py <bundle_or_manifest.json>")
        sys.exit(1)

    target_file = sys.argv[1]
    if not os.path.isfile(target_file):
        print(f"File not found: {target_file}")
        sys.exit(1)

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    auditor = IndependentSecSourceAuthorityAuditor()
    if "bundle_schema_version" in data:
        res = auditor.audit_evidence_bundle_chain(data)
    else:
        print("Unrecognized evidence format for CLI audit")
        sys.exit(1)

    print(json.dumps(res, indent=2))
    sys.exit(0 if res["audit_result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
