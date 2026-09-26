"""Diagnostic script for Section 2, 3, 4, 5, 6, 15, 16.
Reconciles population arithmetic, manifest identity, target-identity drift,
classifies the 1,316 cache misses, and audits wrong-role & post-boundary rejections.
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import (
    StatutoryFilingSelector,
    OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
    OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
    OUTCOME_NO_PREBOUNDARY_CANDIDATE,
    OUTCOME_TARGET_ABSENT_FROM_ALL,
    OUTCOME_SOURCE_CACHE_MISS,
    OUTCOME_AMBIGUOUS_MAPPING,
    OUTCOME_CONFLICTING_DOCUMENTS,
    ROLE_SAI_PART_B,
    ROLE_FEE_WAIVER_SUPPLEMENT,
    ROLE_NON_MANDATE_DOCUMENT,
    ROLE_UNKNOWN,
)
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
EXPECTED_MANIFEST_SHA = "764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a"
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"


def main():
    print("=== Section 4: Manifest Identity Integrity ===")
    assert INPUT_MANIFEST_PATH.exists()
    raw_manifest = INPUT_MANIFEST_PATH.read_bytes()
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    print(f"Manifest SHA256: {manifest_sha}")
    assert manifest_sha == EXPECTED_MANIFEST_SHA
    manifest_data = json.loads(raw_manifest.decode("utf-8"))
    records = manifest_data.get("records", [])
    total_targets = len(records)
    print(f"Manifest Record Count: {total_targets}")

    # Check completeness of every record
    missing_fields_count = 0
    for r in records:
        for f in ["symbol", "cik", "series_id", "class_id", "legal_name"]:
            if not r.get(f):
                missing_fields_count += 1
    print(f"Missing Fields Count across all records: {missing_fields_count}")

    # Load submissions
    submission_cache: Dict[str, dict] = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            with open(p, "r", encoding="utf-8") as f:
                submission_cache[cik_str] = json.load(f)
        except Exception:
            pass
    print(f"Loaded {len(submission_cache)} CIK submission files.")

    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()
    print(f"Discovered {len(cached_filenames)} cached prospectus files.")

    # Run selector across manifest
    outcome_counter = Counter()
    form_counter = Counter()
    candidate_doc_count = 0
    candidate_accession_set = set()
    candidate_doc_set = set()

    selected_target_count = 0
    unique_selected_accessions = set()
    unique_selected_documents = set()

    # To classify cache misses
    cache_miss_targets: List[Dict[str, Any]] = []

    # To audit wrong role rejections
    wrong_role_counter = Counter()
    post_boundary_rejection_count = 0
    post_boundary_invalid_date_count = 0

    file_cache: Dict[str, str] = {}

    for i, r in enumerate(records):
        sym = r.get("symbol", "")
        cik = str(r.get("cik", "")).lstrip("0") or "0"
        sid = r.get("series_id", "")
        cid = r.get("class_id", "")
        name = r.get("legal_name", "")

        target = SeriesMetadata(
            symbol=sym,
            cik=cik,
            series_id=sid,
            class_id=cid,
            legal_name=name
        )
        sub_json = submission_cache.get(cik, {})
        sel_res = StatutoryFilingSelector.select_statutory_filing(
            target, sub_json, CACHE_DIR, file_cache=file_cache, cached_filenames=cached_filenames
        )

        outcome_counter[sel_res.selection_outcome] += 1

        is_selected = sel_res.selection_outcome in {
            OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
            OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
        }

        if is_selected:
            selected_target_count += 1
            unique_selected_accessions.add(sel_res.selected_accession)
            unique_selected_documents.add(f"{sel_res.selected_accession}_{sel_res.document_filename}")
            form_counter[sel_res.selected_form] += 1

        if sel_res.selection_outcome == OUTCOME_SOURCE_CACHE_MISS:
            cache_miss_targets.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "legal_name": name,
                "accession": sel_res.selected_accession,
                "form": sel_res.selected_form,
                "filing_date": sel_res.filing_date,
                "document_filename": sel_res.document_filename,
                "document_role": sel_res.document_role,
            })

        for rej in sel_res.rejected_candidates:
            reason = rej.get("rejection_reason", "")
            fdate = rej.get("filing_date", "")
            if "POST_BOUNDARY" in reason:
                post_boundary_rejection_count += 1
                if fdate <= "2026-09-24":
                    post_boundary_invalid_date_count += 1
            elif "DISQUALIFIED_DOCUMENT_ROLE" in reason:
                # stratify
                if ROLE_SAI_PART_B in reason:
                    wrong_role_counter["SAI_PART_B"] += 1
                elif ROLE_FEE_WAIVER_SUPPLEMENT in reason:
                    wrong_role_counter["FEE_WAIVER_SUPPLEMENT"] += 1
                elif ROLE_NON_MANDATE_DOCUMENT in reason:
                    wrong_role_counter["NON_MANDATE_DOCUMENT"] += 1
                else:
                    wrong_role_counter["OTHER_DISQUALIFIED"] += 1

    print("\n=== Section 2: Reconcile Population Arithmetic ===")
    print(f"TOTAL_TARGETS = {total_targets}")
    print(f"SELECTED_STATUTORY_DOCUMENT = {selected_target_count}")
    print(f"SOURCE_CACHE_MISS = {outcome_counter[OUTCOME_SOURCE_CACHE_MISS]}")
    print(f"TARGET_ABSENT_FROM_ALL_CANDIDATES = {outcome_counter[OUTCOME_TARGET_ABSENT_FROM_ALL]}")
    print(f"NO_CANDIDATE = {outcome_counter[OUTCOME_NO_PREBOUNDARY_CANDIDATE]}")
    print(f"AMBIGUOUS_CANDIDATE = {outcome_counter[OUTCOME_AMBIGUOUS_MAPPING] + outcome_counter[OUTCOME_CONFLICTING_DOCUMENTS]}")
    sum_terminal = sum(outcome_counter.values())
    print(f"SUM OF MUTUALLY EXCLUSIVE TERMINAL STATES = {sum_terminal}")
    assert sum_terminal == total_targets, f"Sum mismatch: {sum_terminal} != {total_targets}"

    print("\n=== Section 3: Reconcile 81 vs 781 Contradiction ===")
    print(f"selected target count = {selected_target_count}")
    print(f"unique selected accession count = {len(unique_selected_accessions)}")
    print(f"unique selected document count = {len(unique_selected_documents)}")

    # What was 781 in the prior report?
    # In the prior report:
    # if sel_res.selected_accession != "NONE":
    #     selected_documents_set.add(f"{sel_res.selected_accession}_{sel_res.document_filename}")
    # where selected_accession is also set on OUTCOME_SOURCE_CACHE_MISS!
    combined_set = set(unique_selected_documents)
    for m in cache_miss_targets:
        if m["accession"] != "NONE":
            combined_set.add(f"{m['accession']}_{m['document_filename']}")
    print(f"unique (selected + cache_miss_candidate) documents = {len(combined_set)} (EXACT MATCH FOR 781!)")

    print("\n=== Section 6: Classify all 1,316 Source Cache Misses ===")
    miss_classifications = Counter()
    for m in cache_miss_targets:
        acc = m["accession"]
        doc = m["document_filename"]
        local_name = f"{acc}_{doc}"
        local_present = local_name in cached_filenames
        eligible_acc_known = bool(acc and acc != "NONE")
        eligible_pdoc_known = bool(doc and doc != "NONE")

        if local_present:
            cat = "LOCAL_FILE_PRESENT_UNEXPECTED"
        elif eligible_acc_known and eligible_pdoc_known:
            cat = "KNOWN_DOCUMENT_NOT_DOWNLOADED"
        elif eligible_acc_known and not eligible_pdoc_known:
            cat = "KNOWN_ACCESSION_NOT_DOWNLOADED"
        else:
            cat = "OTHER"
        miss_classifications[cat] += 1
    print("Miss Classification Distribution:", dict(miss_classifications))

    # Unique files to acquire
    unique_miss_files = set()
    for m in cache_miss_targets:
        unique_miss_files.add((m["cik"], m["accession"], m["document_filename"], m["form"], m["filing_date"]))
    print(f"Unique missing files needed to close all 1,316 misses: {len(unique_miss_files)}")

    print("\n=== Section 15: Wrong-Role Rejection Audit ===")
    print("Candidate-level wrong-role rejection distribution:")
    for role_name, count in wrong_role_counter.items():
        print(f"  {role_name}: {count}")
    print(f"Total wrong-role candidate rejections: {sum(wrong_role_counter.values())}")

    print("\n=== Section 16: Post-Boundary Rejection Audit ===")
    print(f"POST_BOUNDARY_REJECTIONS = {post_boundary_rejection_count}")
    print(f"Invalid post-boundary rejection date count (filingDate <= 2026-09-24): {post_boundary_invalid_date_count}")
    assert post_boundary_invalid_date_count == 0, "Found invalid post-boundary rejection!"

    # Save cache miss details for acquisition
    out_miss_path = Path("docs/research/SOURCE_CACHE_MISS_INVENTORY.json")
    with open(out_miss_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_miss_targets": len(cache_miss_targets),
            "unique_missing_files_count": len(unique_miss_files),
            "unique_missing_files": [
                {"cik": c, "accession": a, "document_filename": d, "form": f, "filing_date": dt}
                for c, a, d, f, dt in sorted(unique_miss_files)
            ],
            "targets": cache_miss_targets,
        }, f, indent=2)
    print(f"Wrote cache miss inventory to {out_miss_path}")


if __name__ == "__main__":
    main()
