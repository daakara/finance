"""Evaluation script for STATUTORY_FILING_SELECTOR_V1_2_0 against V1.1.0 failure corpus.

Evaluates:
1. 7 confirmed false-negative targets (Section 18, 30)
2. 35 residual cache-miss targets (Section 17, 29)
3. 10 original certified targets
4. 16 residual historical-crawl targets
"""

import json
from pathlib import Path
from scripts.research.series_prospectus_mapper import SeriesMetadata
from scripts.research.statutory_filing_selector import StatutoryFilingSelector

CORPUS_PATH = Path("docs/research/V1_1_0_REMEDIATION_FAILURE_CORPUS.json")
SUBMISSIONS_DIR = Path("data/research/cache/sec_submissions")
PROSPECTUS_DIR = Path("data/research/cache/sec_prospectus")


def main():
    print("=" * 80)
    print("ARX TERMINAL — EVALUATING V1.2.0 SELECTOR AGAINST FAILURE CORPUS")
    print("=" * 80)

    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    StatutoryFilingSelector.reset_history_cache()

    # In-memory document cache to avoid repeated disk reads
    file_cache = {}
    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()

    # 1. Evaluate 7 Confirmed False Negatives
    print("\n--- 1. EVALUATING 7 CONFIRMED FALSE NEGATIVES ---")
    fn_records = corpus["confirmed_7_false_negatives"]
    fn_remediated = 0
    fn_results = []

    for item in fn_records:
        sym = item["symbol"]
        cik = str(item["cik"]).zfill(10)
        sub_path = SUBMISSIONS_DIR / f"CIK{cik}.json"
        sub_json = json.load(open(sub_path, encoding="utf-8"))

        target = SeriesMetadata(
            cik=cik,
            series_id=item["series_id"],
            class_id=item["class_id"],
            symbol=sym,
            legal_name=item["legal_name"],
        )

        res = StatutoryFilingSelector.select_statutory_filing(
            target_series=target,
            submission_json=sub_json,
            file_cache=file_cache,
            cached_filenames=cached_filenames,
        )

        is_selected = res.selection_outcome in {
            "SELECTED_TARGET_STATUTORY_PROSPECTUS",
            "SELECTED_TARGET_SUMMARY_PROSPECTUS",
        }
        if is_selected:
            fn_remediated += 1

        fn_results.append({
            "symbol": sym,
            "cik": cik,
            "v1_1_0_outcome": item["v1_1_0_outcome"],
            "v1_2_0_outcome": res.selection_outcome,
            "selected_accession": res.selected_accession,
            "selected_doc": res.document_filename,
            "remediated": is_selected,
        })
        print(f"[{sym}] V1.1.0: {item['v1_1_0_outcome']} -> V1.2.0: {res.selection_outcome} ({res.selected_accession} / {res.document_filename}) | REMEDIATED: {is_selected}")

    print(f"\nKNOWN_FALSE_NEGATIVES_REMEDIATED: {fn_remediated} / {len(fn_records)}")

    # 2. Evaluate Residual 35 Cache Misses
    print("\n--- 2. EVALUATING RESIDUAL 35 CACHE MISSES ---")
    miss_records = corpus["residual_35_cache_misses"]
    miss_outcomes = {}
    miss_results = []
    end_cache_misses = 0

    for item in miss_records:
        sym = item["symbol"]
        cik = str(item["cik"]).zfill(10)
        sub_path = SUBMISSIONS_DIR / f"CIK{cik}.json"
        sub_json = json.load(open(sub_path, encoding="utf-8"))

        target = SeriesMetadata(
            cik=cik,
            series_id=item["series_id"],
            class_id=item["class_id"],
            symbol=sym,
            legal_name=item["legal_name"],
        )

        res = StatutoryFilingSelector.select_statutory_filing(
            target_series=target,
            submission_json=sub_json,
            file_cache=file_cache,
            cached_filenames=cached_filenames,
        )

        miss_outcomes[res.selection_outcome] = miss_outcomes.get(res.selection_outcome, 0) + 1
        if res.selection_outcome == "SOURCE_CACHE_MISS":
            end_cache_misses += 1

        miss_results.append({
            "symbol": sym,
            "cik": cik,
            "legal_name": item["legal_name"],
            "v1_1_0_outcome": item["v1_1_0_outcome"],
            "v1_2_0_outcome": res.selection_outcome,
            "selected_accession": res.selected_accession,
            "selected_doc": res.document_filename,
        })
        print(f"[{sym}] V1.1.0: {item['v1_1_0_outcome']} -> V1.2.0: {res.selection_outcome} ({res.selected_accession} / {res.document_filename})")

    print(f"\nRESIDUAL 35 OUTCOME DISTRIBUTION:")
    for outcome, count in sorted(miss_outcomes.items()):
        print(f"  {outcome}: {count}")
    print(f"SOURCE_CACHE_MISS_END = {end_cache_misses}")

    # 3. Evaluate Original 10 Certified Targets
    print("\n--- 3. EVALUATING ORIGINAL 10 CERTIFIED TARGETS ---")
    orig_10 = corpus["original_10_certified_targets"]
    orig_selected = 0
    for item in orig_10:
        sym = item["symbol"]
        cik = str(item["cik"]).zfill(10)
        sub_path = SUBMISSIONS_DIR / f"CIK{cik}.json"
        sub_json = json.load(open(sub_path, encoding="utf-8"))

        target = SeriesMetadata(
            cik=cik,
            series_id=item["series_id"],
            class_id=item["class_id"],
            symbol=sym,
            legal_name=item["legal_name"],
        )

        res = StatutoryFilingSelector.select_statutory_filing(
            target_series=target,
            submission_json=sub_json,
            file_cache=file_cache,
            cached_filenames=cached_filenames,
        )

        is_sel = res.selection_outcome in {"SELECTED_TARGET_STATUTORY_PROSPECTUS", "SELECTED_TARGET_SUMMARY_PROSPECTUS"}
        if is_sel:
            orig_selected += 1
        print(f"[{sym}] {res.selection_outcome} ({res.selected_accession} / {res.document_filename}) | SELECTED: {is_sel}")

    print(f"ORIGINAL_10_SELECTED: {orig_selected} / {len(orig_10)}")

    # 4. Evaluate Residual 16 Historical-Crawl Targets
    print("\n--- 4. EVALUATING RESIDUAL 16 HISTORICAL-CRAWL TARGETS ---")
    res_16 = corpus["residual_16_historical_crawl_targets"]
    res_16_selected = 0
    for item in res_16:
        sym = item["symbol"]
        cik = str(item["cik"]).zfill(10)
        sub_path = SUBMISSIONS_DIR / f"CIK{cik}.json"
        sub_json = json.load(open(sub_path, encoding="utf-8"))

        target = SeriesMetadata(
            cik=cik,
            series_id=item.get("series_id", ""),
            class_id=item.get("class_id", ""),
            symbol=sym,
            legal_name=item["legal_name"],
        )

        res = StatutoryFilingSelector.select_statutory_filing(
            target_series=target,
            submission_json=sub_json,
            file_cache=file_cache,
            cached_filenames=cached_filenames,
        )

        is_sel = res.selection_outcome in {"SELECTED_TARGET_STATUTORY_PROSPECTUS", "SELECTED_TARGET_SUMMARY_PROSPECTUS"}
        if is_sel:
            res_16_selected += 1
        print(f"[{sym}] {res.selection_outcome} ({res.selected_accession} / {res.document_filename}) | SELECTED: {is_sel}")

    print(f"RESIDUAL_16_SELECTED: {res_16_selected} / {len(res_16)}")

    # Save summary report
    report = {
        "selector_version": StatutoryFilingSelector.VERSION,
        "known_false_negatives_remediated": f"{fn_remediated} / {len(fn_records)}",
        "residual_35_cache_misses_start": 35,
        "residual_35_cache_misses_end": end_cache_misses,
        "residual_35_distribution": miss_outcomes,
        "original_10_selected": f"{orig_10_selected if 'orig_10_selected' in locals() else orig_selected} / {len(orig_10)}",
        "residual_16_selected": f"{res_16_selected} / {len(res_16)}",
        "fn_results": fn_results,
        "miss_results": miss_results,
    }
    with open("docs/research/V1_2_0_CORPUS_EVAL_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved eval report to docs/research/V1_2_0_CORPUS_EVAL_REPORT.json")


if __name__ == "__main__":
    main()
