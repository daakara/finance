"""ARX Terminal — Re-evaluate Residual 46 Cache Misses (Section 8).

Runs selector resolution for the 46 frozen residual targets with the newly acquired sources.
Records Prior State, Root Cause, Acquisition Needed, New State, and Selected Accession.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

FROZEN_PATH = Path("docs/research/FROZEN_RESIDUAL_46_CACHE_MISSES.json")
CLASSIFICATION_PATH = Path("docs/research/RESIDUAL_46_CLASSIFICATION.json")
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
OUTPUT_EVAL_PATH = Path("docs/research/RESIDUAL_46_EVALUATION.json")


def main():
    print("=" * 80)
    print("ARX TERMINAL — RE-EVALUATING RESIDUAL 46 TARGETS (SECTION 8)")
    print("=" * 80)

    data = json.load(open(FROZEN_PATH, encoding="utf-8"))["records"]
    classif_data = json.load(open(CLASSIFICATION_PATH, encoding="utf-8"))["records"]
    rc_map = {r["symbol"]: r["root_cause"] for r in classif_data}

    submission_cache: Dict[str, dict] = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            with open(p, "r", encoding="utf-8") as f:
                submission_cache[cik_str] = json.load(f)
        except Exception:
            pass

    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()
    file_cache: Dict[str, str] = {}

    eval_results: List[Dict[str, Any]] = []

    print(f"{'Target':<7} {'Prior State':<18} {'Root Cause':<35} {'Acq Needed':<11} {'New State':<38} {'Selected Accession':<22}")
    print("-" * 140)

    outcome_counts = {}

    for r in data:
        sym = r["symbol"]
        cik = r["cik"]
        sid = r["series_id"]
        cid = r["class_id"]
        name = r["legal_name"]

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

        prior_state = "SOURCE_CACHE_MISS"
        root_cause = rc_map.get(sym, "KNOWN_ELIGIBLE_DOCUMENT_NOT_CACHED")
        acq_needed = "YES" if root_cause == "KNOWN_ELIGIBLE_DOCUMENT_NOT_CACHED" else "NO"
        new_state = sel_res.selection_outcome
        sel_acc = sel_res.selected_accession

        outcome_counts[new_state] = outcome_counts.get(new_state, 0) + 1

        eval_results.append({
            "target": sym,
            "cik": cik,
            "series_id": sid,
            "class_id": cid,
            "legal_name": name,
            "prior_state": prior_state,
            "root_cause": root_cause,
            "acquisition_needed": acq_needed,
            "new_state": new_state,
            "selected_accession": sel_acc,
            "selected_form": sel_res.selected_form,
            "selected_document": sel_res.document_filename,
        })

        print(f"{sym:<7} {prior_state:<18} {root_cause:<35} {acq_needed:<11} {new_state:<38} {sel_acc:<22}")

    output = {
        "total_targets": len(eval_results),
        "outcome_distribution": outcome_counts,
        "records": eval_results
    }

    with open(OUTPUT_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("-" * 140)
    print("\nOutcome Summary for Residual 46:")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v}")

    print(f"\nSaved residual 46 evaluation to: {OUTPUT_EVAL_PATH}")


if __name__ == "__main__":
    main()
