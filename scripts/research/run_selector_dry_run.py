"""ARX Terminal — Statutory Filing Selector Population Dry-Run (Section 29).

Executes selection-layer dry run across all targets from the frozen input manifest:
INPUT_MANIFEST_SHA256 = 764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a

Does NOT perform mandate classification.
Produces only filing-selection diagnostics.
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
    FilingSelectionResult,
    OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
    OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
    OUTCOME_NO_PREBOUNDARY_CANDIDATE,
    OUTCOME_TARGET_ABSENT_FROM_ALL,
    OUTCOME_SOURCE_CACHE_MISS,
    OUTCOME_AMBIGUOUS_MAPPING,
    OUTCOME_CONFLICTING_DOCUMENTS,
)
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
EXPECTED_MANIFEST_SHA = "764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a"
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
OUTPUT_DIAGNOSTICS_PATH = Path("docs/research/STATUTORY_SELECTOR_DRY_RUN_DIAGNOSTICS.json")


def run_dry_run() -> Dict[str, Any]:
    print("=" * 80)
    print("ARX TERMINAL — POPULATION STATUTORY FILING SELECTION DRY-RUN (SECTION 29)")
    print("=" * 80)

    # 1. Verify Manifest Identity
    assert INPUT_MANIFEST_PATH.exists(), f"Missing input manifest: {INPUT_MANIFEST_PATH}"
    with open(INPUT_MANIFEST_PATH, "rb") as f:
        manifest_bytes = f.read()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest_sha == EXPECTED_MANIFEST_SHA, (
        f"Manifest SHA mismatch: expected {EXPECTED_MANIFEST_SHA}, got {manifest_sha}"
    )
    print(f"Verified Input Manifest SHA256: {manifest_sha}")

    data = json.loads(manifest_bytes.decode("utf-8"))
    records = data.get("records", [])
    total_targets = len(records)
    print(f"Loaded {total_targets} targets from manifest.")

    # Cache submissions in memory to avoid repeated JSON disk reads
    submission_cache: Dict[str, dict] = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            with open(p, "r", encoding="utf-8") as f:
                submission_cache[cik_str] = json.load(f)
        except Exception:
            pass
    print(f"Cached {len(submission_cache)} CIK submission files in memory.")
    file_cache: Dict[str, str] = {}
    prospectus_dir = CACHE_DIR / "sec_prospectus"
    cached_filenames: Set[str] = {p.name for p in prospectus_dir.iterdir()} if prospectus_dir.exists() else set()
    print(f"Discovered {len(cached_filenames)} cached prospectus files.")

    # Counters
    outcome_counter = Counter()
    form_counter = Counter()
    candidate_count_distribution = Counter()
    wrong_role_rejection_count = 0
    post_boundary_rejection_count = 0
    selected_documents_set = set()

    results: List[Dict[str, Any]] = []

    for i, row in enumerate(records):
        if (i + 1) % 500 == 0 or i == 0 or i == total_targets - 1:
            print(f"Processing target [{i+1}/{total_targets}] ({row.get('symbol')})...")
        sym = row.get("symbol", "")
        cik = str(row.get("cik", "")).lstrip("0") or "0"
        sid = row.get("series_id", "")
        cid = row.get("class_id", "")
        name = row.get("legal_name", "")

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
        candidate_count_distribution[sel_res.candidate_count] += 1

        is_selected = sel_res.selection_outcome in {
            OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
            OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
        }
        if is_selected:
            form_counter[sel_res.selected_form] += 1
            selected_documents_set.add(f"{sel_res.selected_accession}_{sel_res.document_filename}")

        # Tally rejections
        for r in sel_res.rejected_candidates:
            reason = r.get("rejection_reason", "")
            if "POST_BOUNDARY" in reason:
                post_boundary_rejection_count += 1
            elif "DISQUALIFIED_DOCUMENT_ROLE" in reason:
                wrong_role_rejection_count += 1

        results.append({
            "symbol": sym,
            "cik": cik,
            "series_id": sid,
            "class_id": cid,
            "selection_outcome": sel_res.selection_outcome,
            "selected_accession": sel_res.selected_accession,
            "selected_form": sel_res.selected_form,
            "candidate_count": sel_res.candidate_count,
            "cache_key": sel_res.cache_key,
        })

    # Summary
    selected_statutory_count = (
        outcome_counter[OUTCOME_SELECTED_STATUTORY_PROSPECTUS] +
        outcome_counter[OUTCOME_SELECTED_SUMMARY_PROSPECTUS]
    )

    diagnostics = {
        "selector_version": StatutoryFilingSelector.VERSION,
        "input_manifest_sha256": manifest_sha,
        "total_targets": total_targets,
        "selected_statutory_documents_count": selected_statutory_count,
        "unique_documents_selected": len(selected_documents_set),
        "outcomes": dict(outcome_counter),
        "selected_forms": dict(form_counter),
        "no_candidate_count": outcome_counter[OUTCOME_NO_PREBOUNDARY_CANDIDATE],
        "target_absent_count": outcome_counter[OUTCOME_TARGET_ABSENT_FROM_ALL],
        "source_cache_miss_count": outcome_counter[OUTCOME_SOURCE_CACHE_MISS],
        "ambiguous_count": outcome_counter[OUTCOME_AMBIGUOUS_MAPPING] + outcome_counter[OUTCOME_CONFLICTING_DOCUMENTS],
        "wrong_role_rejection_count": wrong_role_rejection_count,
        "post_boundary_rejection_count": post_boundary_rejection_count,
        "candidate_count_distribution_sample": {
            k: candidate_count_distribution[k] for k in sorted(candidate_count_distribution.keys())[:10]
        }
    }

    OUTPUT_DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDRY-RUN RESULTS SUMMARY:")
    print(f"Total Targets: {total_targets}")
    print(f"Selected Statutory Documents: {selected_statutory_count}")
    print(f"Unique Documents Selected: {len(selected_documents_set)}")
    print(f"Outcomes: {dict(outcome_counter)}")
    print(f"No Pre-boundary Candidate: {diagnostics['no_candidate_count']}")
    print(f"Target Absent From All: {diagnostics['target_absent_count']}")
    print(f"Source Cache Miss: {diagnostics['source_cache_miss_count']}")
    print(f"Ambiguous Selection: {diagnostics['ambiguous_count']}")
    print(f"Wrong Role Rejections: {wrong_role_rejection_count}")
    print(f"Post-Boundary Rejections: {post_boundary_rejection_count}")
    print(f"Saved diagnostics to: {OUTPUT_DIAGNOSTICS_PATH}")
    print("=" * 80)
    return diagnostics


if __name__ == "__main__":
    run_dry_run()
