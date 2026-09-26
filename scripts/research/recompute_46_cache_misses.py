"""ARX Terminal — Recompute Residual Cache Misses from Scratch (Section 3).

Re-runs selector resolution against the frozen manifest, enumerates every target
whose terminal outcome is SOURCE_CACHE_MISS, records required fields, and freezes
them before any acquisition is attempted.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import (
    StatutoryFilingSelector,
    OUTCOME_SOURCE_CACHE_MISS,
)
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
EXPECTED_MANIFEST_SHA = "764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a"
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
OUTPUT_FROZEN_PATH = Path("docs/research/FROZEN_RESIDUAL_46_CACHE_MISSES.json")


def main():
    print("=" * 80)
    print("ARX TERMINAL — RECOMPUTING RESIDUAL CACHE MISSES FROM SCRATCH")
    print("=" * 80)

    assert INPUT_MANIFEST_PATH.exists(), f"Missing input manifest: {INPUT_MANIFEST_PATH}"
    manifest_bytes = INPUT_MANIFEST_PATH.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest_sha == EXPECTED_MANIFEST_SHA, (
        f"Manifest SHA mismatch: expected {EXPECTED_MANIFEST_SHA}, got {manifest_sha}"
    )
    print(f"Verified Input Manifest SHA256: {manifest_sha}")

    records = json.loads(manifest_bytes.decode("utf-8"))["records"]
    print(f"Loaded {len(records)} targets from manifest.")

    submission_cache: Dict[str, dict] = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            with open(p, "r", encoding="utf-8") as f:
                submission_cache[cik_str] = json.load(f)
        except Exception:
            pass
    print(f"Cached {len(submission_cache)} CIK submission files in memory.")

    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()
    print(f"Discovered {len(cached_filenames)} cached prospectus files.")

    file_cache: Dict[str, str] = {}
    cache_miss_targets: List[Dict[str, Any]] = []

    for i, row in enumerate(records):
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

        if sel_res.selection_outcome == OUTCOME_SOURCE_CACHE_MISS:
            expected_filename = f"{sel_res.selected_accession}_{sel_res.document_filename}"
            expected_cache_path = str(PROSPECTUS_DIR / expected_filename)
            cache_miss_targets.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "legal_name": name,
                "candidate_accession": sel_res.selected_accession,
                "candidate_form": sel_res.selected_form,
                "candidate_filing_date": sel_res.filing_date,
                "candidate_document_filename": sel_res.document_filename,
                "selection_reason": sel_res.selection_evidence,
                "selection_rule_id": sel_res.selection_rule_id,
                "cache_path_expected": expected_cache_path,
                "expected_filename": expected_filename,
            })

    print(f"\nSOURCE_CACHE_MISS_RECOMPUTED = {len(cache_miss_targets)}")

    output_data = {
        "source_cache_miss_recomputed": len(cache_miss_targets),
        "manifest_sha256": manifest_sha,
        "total_targets_evaluated": len(records),
        "cached_prospectus_count_at_evaluation": len(cached_filenames),
        "records": cache_miss_targets
    }

    with open(OUTPUT_FROZEN_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved frozen residual cache-miss inventory to: {OUTPUT_FROZEN_PATH}")

    # Print summary table
    print("\nResidual Targets Detail:")
    print(f"{'SYM':<6} {'CIK':<10} {'FORM':<8} {'ACCESSION':<22} {'DOC':<30}")
    print("-" * 80)
    for m in cache_miss_targets:
        print(f"{m['symbol']:<6} {m['cik']:<10} {m['candidate_form']:<8} {m['candidate_accession']:<22} {m['candidate_document_filename']:<30}")


if __name__ == "__main__":
    main()
