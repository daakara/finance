"""ARX Terminal — Freeze V1.1.0 Remediation Failure Corpus (Section 3).

Freezes an immutable failure corpus containing:
1. Residual 35 SOURCE_CACHE_MISS targets
2. 7 confirmed false-negative targets (ARMH, ASMH, AFOS, ADIV, ADPV, AIQ, ATTR)
3. 40 sampled absence targets
4. Original certified 10/20 target certification set
5. Residual 16 historical-crawl remediation set
"""

import json
from pathlib import Path

MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
RESIDUAL_46_EVAL_PATH = Path("docs/research/RESIDUAL_46_EVALUATION.json")
ABSENCE_REPORT_PATH = Path("docs/research/ABSENCE_AUDIT_REPORT.json")
RESIDUAL_16_PATH = Path("docs/research/FROZEN_RESIDUAL_16_TARGETS.json")
OUTPUT_PATH = Path("docs/research/V1_1_0_REMEDIATION_FAILURE_CORPUS.json")


def main():
    print("=" * 80)
    print("ARX TERMINAL — FREEZING V1.1.0 FAILURE CORPUS (SECTION 3)")
    print("=" * 80)

    manifest_records = {r["symbol"]: r for r in json.load(open(MANIFEST_PATH, encoding="utf-8"))["records"]}

    # 1. Residual 35 cache-miss targets
    eval_46 = json.load(open(RESIDUAL_46_EVAL_PATH, encoding="utf-8"))["records"]
    residual_35 = [r for r in eval_46 if r["new_state"] == "SOURCE_CACHE_MISS"]
    print(f"Residual 35 cache misses loaded: {len(residual_35)}")

    # 2. 7 confirmed false-negative absence targets
    absence_data = json.load(open(ABSENCE_REPORT_PATH, encoding="utf-8"))
    adj_records = absence_data["adjudication_records"]
    confirmed_7 = [r for r in adj_records if r["has_qualifying_preboundary_doc"] == "YES"]
    print(f"Confirmed 7 false negatives loaded: {len(confirmed_7)}")

    # 3. 40 sampled absence targets
    sampled_40 = adj_records
    print(f"Sampled 40 absence targets loaded: {len(sampled_40)}")

    # 4. Original certified 10 targets
    targets_10 = [
        ("GQQQ", "1592900", "S000088111", "C000254131", "Astoria US Quality Growth Kings ETF", "EA Series TRUST"),
        ("ROE",  "1592900", "S000081203", "C000244015", "Astoria US Equal Weight Quality Kings ETF", "EA Series TRUST"),
        ("AGGA", "1592900", "S000091819", "C000259601", "EA Astoria Beacon Dynamic Core US Fixed Income ETF", "EA Series TRUST"),
        ("CGCP", "1870117", "S000074251", "C000231860", "Capital Group Core Plus Income ETF", "Capital Group Fixed Income ETF Trust"),
        ("CGMS", "1870117", "S000077688", "C000238176", "Capital Group Short Duration Municipal Income ETF", "Capital Group Fixed Income ETF Trust"),
        ("CGSD", "1870117", "S000074252", "C000231861", "Capital Group Short Duration Income ETF", "Capital Group Fixed Income ETF Trust"),
        ("CGHY", "1870117", "S000080123", "C000241982", "Capital Group High Yield Bond ETF", "Capital Group Fixed Income ETF Trust"),
        ("CGGG", "2034928", "S000092695", "C000260959", "Capital Group U.S. Large Growth ETF", "Capital Group Active ETF Trust"),
        ("CGMM", "2034928", "S000088874", "C000255476", "Capital Group U.S. Small and Mid Cap ETF", "Capital Group Active ETF Trust"),
        ("CGVV", "2034928", "S000092696", "C000260960", "Capital Group U.S. Large Value ETF", "Capital Group Active ETF Trust"),
    ]
    print(f"Original 10 certified targets loaded: {len(targets_10)}")

    # 5. Residual 16 historical-crawl set
    res_16_raw = json.load(open(RESIDUAL_16_PATH, encoding="utf-8"))["by_cik"]
    res_16 = []
    for cik_info in res_16_raw.values():
        res_16.extend(cik_info["targets"])
    print(f"Residual 16 historical-crawl targets loaded: {len(res_16)}")

    corpus = {
        "residual_35_cache_misses": [
            {
                "symbol": r["target"],
                "cik": r["cik"],
                "series_id": r["series_id"],
                "class_id": r["class_id"],
                "legal_name": r["legal_name"],
                "v1_1_0_outcome": "SOURCE_CACHE_MISS",
                "candidate_accession": r.get("selected_accession", ""),
                "candidate_form": r.get("selected_form", ""),
                "candidate_document": r.get("selected_document", ""),
                "failure_class": "FALSE_POSITIVE_BUFFER_MATCHING" if "buffer" in r["legal_name"].lower() else "SHARE_CLASS_OR_SUBMISSION_MISS"
            }
            for r in residual_35
        ],
        "confirmed_7_false_negatives": [
            {
                "symbol": r["target"],
                "cik": r["cik"],
                "legal_name": r["legal_name"],
                "series_id": manifest_records.get(r["target"], {}).get("series_id", ""),
                "class_id": manifest_records.get(r["target"], {}).get("class_id", ""),
                "v1_1_0_outcome": "TARGET_ABSENT_FROM_ALL_CANDIDATES",
                "expected_qualifying_document": r.get("matching_document", ""),
                "expected_qualifying_accession": r.get("matching_accession", ""),
                "failure_class": r["failure_class"]
            }
            for r in confirmed_7
        ],
        "sampled_40_absence_targets": [
            {
                "symbol": r["target"],
                "cik": r["cik"],
                "legal_name": r["legal_name"],
                "series_id": manifest_records.get(r["target"], {}).get("series_id", ""),
                "class_id": manifest_records.get(r["target"], {}).get("class_id", ""),
                "v1_1_0_outcome": "TARGET_ABSENT_FROM_ALL_CANDIDATES",
                "audited_verdict": "FALSE_NEGATIVE" if r["has_qualifying_preboundary_doc"] == "YES" else "CORRECT_ABSENCE",
                "failure_class": r["failure_class"]
            }
            for r in sampled_40
        ],
        "original_10_certified_targets": [
            {
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "legal_name": name,
                "trust_name": trust,
                "expected_outcome": "SELECTED_TARGET_STATUTORY_PROSPECTUS"
            }
            for sym, cik, sid, cid, name, trust in targets_10
        ],
        "residual_16_historical_crawl_targets": [
            {
                "symbol": r["symbol"],
                "cik": r["cik"],
                "series_id": r["series_id"],
                "class_id": r["class_id"],
                "legal_name": r["legal_name"],
                "expected_outcome": "SELECTED_STATUTORY_DOCUMENT"
            }
            for r in res_16
        ]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nSaved frozen remediation failure corpus to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
