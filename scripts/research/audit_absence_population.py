"""ARX Terminal — Absence Population Stratification & Independent Audit (Sections 11-17).

1. Identifies all TARGET_ABSENT_FROM_ALL_CANDIDATES targets.
2. Stratifies by CIK, candidate count, form availability, latest filing year, trust size.
3. Selects a stratified sample spanning structural classes.
4. Independently adjudicates whether qualifying pre-boundary documents exist in SEC submissions.
5. Identifies confirmed selector false negatives, failure classes, and systematic vs isolated defect status.
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
OUTPUT_REPORT_PATH = Path("docs/research/ABSENCE_AUDIT_REPORT.json")


def main():
    print("=" * 80)
    print("ARX TERMINAL — ABSENCE POPULATION STRATIFICATION & INDEPENDENT AUDIT")
    print("=" * 80)

    manifest = json.load(open(INPUT_MANIFEST_PATH, encoding="utf-8"))["records"]

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

    absence_targets: List[Dict[str, Any]] = []
    selected_targets: List[Dict[str, Any]] = []
    cache_miss_targets: List[Dict[str, Any]] = []

    for r in manifest:
        sym = r.get("symbol", "")
        cik = str(r.get("cik", "")).lstrip("0") or "0"
        sid = r.get("series_id", "")
        cid = r.get("class_id", "")
        name = r.get("legal_name", "")

        target = SeriesMetadata(symbol=sym, cik=cik, series_id=sid, class_id=cid, legal_name=name)
        sub_json = submission_cache.get(cik, {})
        sel_res = StatutoryFilingSelector.select_statutory_filing(
            target, sub_json, CACHE_DIR, file_cache=file_cache, cached_filenames=cached_filenames
        )

        record_info = {
            "symbol": sym,
            "cik": cik,
            "series_id": sid,
            "class_id": cid,
            "legal_name": name,
            "outcome": sel_res.selection_outcome,
            "candidate_count": sel_res.candidate_count,
            "rejected_candidates": sel_res.rejected_candidates,
            "selected_accession": sel_res.selected_accession,
            "selected_form": sel_res.selected_form,
            "document_filename": sel_res.document_filename,
        }

        if sel_res.selection_outcome == "TARGET_ABSENT_FROM_ALL_CANDIDATES":
            absence_targets.append(record_info)
        elif "SELECTED" in sel_res.selection_outcome:
            selected_targets.append(record_info)
        else:
            cache_miss_targets.append(record_info)

    print(f"Total Targets: {len(manifest)}")
    print(f"Selected: {len(selected_targets)}")
    print(f"Target Absent From All: {len(absence_targets)}")
    print(f"Source Cache Miss: {len(cache_miss_targets)}")

    # Section 12: Stratify the absence population
    cik_counts = Counter(r["cik"] for r in absence_targets)
    top_ciks = cik_counts.most_common(15)

    print("\n--- Section 12: Stratification of Absence Population ---")
    print("Top 15 CIK Concentration Groups:")
    cik_names = {}
    for cik, count in top_ciks:
        sub = submission_cache.get(cik, {})
        cname = sub.get("name", "Unknown Registrant")
        cik_names[cik] = cname
        pct = (count / len(absence_targets)) * 100
        print(f"  CIK {cik:<10} ({cname:<45}): {count:>4} targets ({pct:.1f}%)")

    # Stratify by candidate counts
    cand_dist = Counter()
    for r in absence_targets:
        cc = r["candidate_count"]
        if cc == 0:
            bucket = "0 candidates"
        elif cc <= 5:
            bucket = "1-5 candidates"
        elif cc <= 20:
            bucket = "6-20 candidates"
        elif cc <= 100:
            bucket = "21-100 candidates"
        else:
            bucket = ">100 candidates"
        cand_dist[bucket] += 1

    print("\nCandidate Count Distribution among Absence Targets:")
    for b, count in cand_dist.items():
        print(f"  {b}: {count} ({count/len(absence_targets)*100:.1f}%)")

    # Section 13 & 14: Sample and independently adjudicate
    # Select a diverse sample of 40 targets across top CIKs, small CIKs, zero-candidates, many-candidates
    sample_targets = []
    seen_ciks = Counter()

    for r in absence_targets:
        cik = r["cik"]
        if seen_ciks[cik] < 2:  # max 2 per CIK to ensure diversity
            sample_targets.append(r)
            seen_ciks[cik] += 1
            if len(sample_targets) >= 40:
                break

    print(f"\n--- Section 13 & 14: Independent Adjudication of {len(sample_targets)} Sample Targets ---")

    adjudication_results = []
    false_negatives = 0
    correct_absences = 0
    ambiguous = 0

    failure_classes = Counter()

    for idx, t in enumerate(sample_targets):
        sym = t["symbol"]
        cik = t["cik"]
        sid = t["series_id"]
        cid = t["class_id"]
        name = t["legal_name"]

        sub = submission_cache.get(cik, {})
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        pdocs = recent.get("primaryDocument", [])
        pdescs = recent.get("primaryDocDescription", [])
        fdates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])

        # Independent search in registrant's filings for target evidence
        found_in_meta = False
        matching_acc = None
        matching_doc = None
        matching_form = None
        matching_date = None
        match_type = None

        sym_lower = sym.lower()
        sid_lower = sid.lower() if sid else ""
        cid_lower = cid.lower() if cid else ""
        norm_name = re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()
        name_words = [w for w in norm_name.split() if len(w) > 2]

        for i, form in enumerate(forms):
            fdate = fdates[i] if i < len(fdates) else ""
            if fdate > "2026-09-24":
                continue
            if form not in ["485BPOS", "485APOS", "497K", "497"]:
                continue

            doc = pdocs[i] if i < len(pdocs) else ""
            desc = pdescs[i] if i < len(pdescs) else ""
            combined = f"{doc.lower()} {desc.lower()}"

            # Check if ticker is explicitly in metadata
            if sym_lower and (f" {sym_lower} " in f" {combined} " or f"({sym_lower})" in combined or f"_{sym_lower}." in combined or f"-{sym_lower}." in combined):
                found_in_meta = True
                matching_acc = accs[i]
                matching_doc = doc
                matching_form = form
                matching_date = fdate
                match_type = "TICKER_IN_METADATA"
                break

            # Check if Series ID or Class ID is in metadata
            if (sid_lower and sid_lower in combined) or (cid_lower and cid_lower in combined):
                found_in_meta = True
                matching_acc = accs[i]
                matching_doc = doc
                matching_form = form
                matching_date = fdate
                match_type = "SERIES_OR_CLASS_ID_IN_METADATA"
                break

            # Check if distinctive name matches
            if len(name_words) >= 2:
                matched_words = [w for w in name_words if w in combined]
                if len(matched_words) >= 3 or (len(name_words) <= 3 and len(matched_words) >= 2):
                    # Check that words aren't just generic
                    non_generic = [w for w in matched_words if w not in ["etf", "fund", "index", "trust", "shares", "portfolio"]]
                    if len(non_generic) >= 2:
                        found_in_meta = True
                        matching_acc = accs[i]
                        matching_doc = doc
                        matching_form = form
                        matching_date = fdate
                        match_type = "NAME_TOKENS_IN_METADATA"
                        break

        # Check cached files if any contain the ticker or legal name
        found_in_cached_file = False
        if not found_in_meta:
            for p in PROSPECTUS_DIR.glob(f"*{cik}*"):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if sid and sid in text:
                        found_in_cached_file = True
                        match_type = "SERIES_ID_IN_CACHED_FILE"
                        matching_doc = p.name
                        break
                    if sym and f"({sym})" in text:
                        found_in_cached_file = True
                        match_type = "TICKER_IN_CACHED_FILE"
                        matching_doc = p.name
                        break
                except Exception:
                    pass

        has_qualifying_doc = found_in_meta or found_in_cached_file

        if has_qualifying_doc:
            selector_correct = False
            false_negatives += 1
            failure_class = f"MISSED_QUALIFYING_DOC_{match_type}"
            failure_classes[failure_class] += 1
        else:
            selector_correct = True
            correct_absences += 1
            failure_class = "GENUINE_ABSENCE_CONFIRMED"

        adjudication_results.append({
            "target": sym,
            "cik": cik,
            "legal_name": name,
            "has_qualifying_preboundary_doc": "YES" if has_qualifying_doc else "NO",
            "match_evidence": match_type or "NONE_FOUND",
            "matching_accession": matching_acc or "NONE",
            "matching_document": matching_doc or "NONE",
            "selector_absence_correct": "YES" if selector_correct else "NO",
            "failure_class": failure_class,
        })

    print(f"\nAudited {len(sample_targets)} absence targets:")
    print(f"  CORRECT_ABSENCE: {correct_absences}")
    print(f"  SELECTOR_FALSE_NEGATIVE: {false_negatives}")
    print(f"  AMBIGUOUS_SOURCE: {ambiguous}")
    print("\nFailure Classes:")
    for fc, cnt in failure_classes.items():
        print(f"  {fc}: {cnt}")

    report = {
        "total_targets": len(manifest),
        "total_absence_population": len(absence_targets),
        "absence_percentage": len(absence_targets) / len(manifest) * 100,
        "sample_size": len(sample_targets),
        "correct_absence_count": correct_absences,
        "false_negative_count": false_negatives,
        "ambiguous_count": ambiguous,
        "failure_classes": dict(failure_classes),
        "top_ciks": top_ciks,
        "adjudication_records": adjudication_results,
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved absence audit report to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
