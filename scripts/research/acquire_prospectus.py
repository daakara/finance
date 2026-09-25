"""ARX Terminal — Statutory Prospectus Acquisition and Deterministic Mandate Resolution Pipeline.

Enforces:
1. Target population: exactly 2,956 mandate-blocked ETFs from ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet.
2. Snapshot boundary: 2026-09-24T23:59:59Z.
3. Policy v1.1.0 authorized statutory filings (485BPOS, 485APOS, N-1A, S-6, 497).
4. No positive classification from ticker, security name, marketing title, or fund name.
5. Deterministic series disambiguation: multi-series ambiguity fails closed to AMBIGUOUS_SERIES_MAPPING.
6. Governed prospectus caching under data/research/cache/sec_prospectus/.
7. Complete attempted population accounting (SOURCE_FOUND + SOURCE_NOT_FOUND == 2956).
8. Frozen DeterministicMandateParser ruleset execution.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.mandate_parser import DeterministicMandateParser, MandateParseResult

SNAPSHOT_BOUNDARY = "2026-09-24"
CACHE_DIR = Path("data/research/cache")
PROSPECTUS_CACHE_DIR = CACHE_DIR / "sec_prospectus"
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
BLOCKER_LEDGER_PATH = Path("docs/research/ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet")
MANDATE_EVIDENCE_PATH = Path("data/research/etf_mandate_evidence_v1.json")

HEADERS = {
    "User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"
}

STATUTORY_FORMS = {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A", "497"}


def get_cik_submission(cik: str) -> dict:
    """Load cached submission JSON for a CIK."""
    p = SUBMISSIONS_DIR / f"CIK{int(cik):010d}.json"
    if not p.exists():
        p = SUBMISSIONS_DIR / f"CIK{cik}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def download_filing(cik: str, accession: str, primary_doc: str) -> tuple[bool, str, str]:
    """Download a primary document from EDGAR and cache locally."""
    PROSPECTUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_filename = f"{accession}_{primary_doc}"
    local_path = PROSPECTUS_CACHE_DIR / local_filename

    if local_path.exists():
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return True, str(local_path), sha

    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"
    
    try:
        time.sleep(0.12)  # Respect SEC rate limit (max 10 req/s)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            with open(local_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(resp.text)
            sha = hashlib.sha256(resp.text.encode("utf-8")).hexdigest()
            return True, str(local_path), sha
        else:
            return False, f"HTTP_{resp.status_code}", ""
    except Exception as e:
        return False, str(e), ""


def run_mandate_acquisition():
    print("=" * 70)
    print("ARX TERMINAL — STATUTORY PROSPECTUS ACQUISITION & MANDATE PIPELINE")
    print("=" * 70)

    # 1. Load Mandate Blocked Population
    assert BLOCKER_LEDGER_PATH.exists(), f"Missing {BLOCKER_LEDGER_PATH}"
    df_ledger = pd.read_parquet(BLOCKER_LEDGER_PATH)
    mandate_pop = df_ledger[df_ledger["blocker_type"] == "MANDATE_BLOCKED"].copy()

    total_attempted = len(mandate_pop)
    assert total_attempted == 2956, f"Expected 2956 mandate blocked rows, found {total_attempted}"
    assert mandate_pop["symbol"].nunique() == 2956, "Duplicate symbols found in mandate population"

    print(f"Target Mandate Population: {total_attempted} unique ETFs across {mandate_pop['CIK'].nunique()} CIKs")

    # 2. Check CIK series counts
    cik_series_counts = mandate_pop.groupby("CIK")["series_id"].nunique().to_dict()

    # 3. Process every ETF deterministically
    attempted_records = []
    source_found_cnt = 0
    source_not_found_cnt = 0
    download_succeeded_cnt = 0
    download_failed_cnt = 0
    series_ambiguous_cnt = 0
    parse_failed_cnt = 0
    confirmatory_resolved_cnt = 0
    other_etf_resolved_cnt = 0
    mandate_remaining_cnt = 0

    downloaded_accessions = {}

    for idx, (_, row) in enumerate(mandate_pop.iterrows()):
        sym = row["symbol"]
        cik = str(row["CIK"])
        sid = row["series_id"]
        cid = row["class_id"]
        b_reason = row["blocker_reason"]

        sub = get_cik_submission(cik)
        recent = sub.get("filings", {}).get("recent", {}) if sub else {}
        forms = recent.get("form", [])
        fdates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        pdocs = recent.get("primaryDocument", [])

        # Find latest pre-boundary statutory filing
        candidates = []
        for i, form in enumerate(forms):
            if form in STATUTORY_FORMS and fdates[i] <= SNAPSHOT_BOUNDARY:
                candidates.append((fdates[i], form, accs[i], pdocs[i] if i < len(pdocs) else ""))

        if not candidates:
            # Source not found on EDGAR
            source_not_found_cnt += 1
            mandate_remaining_cnt += 1
            attempted_records.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "mandate_status": "SOURCE_NOT_FOUND",
                "derived_mandate_classification": "UNRESOLVED_MANDATE",
                "is_sector_specific_mandate": False,
                "approved_sector": None,
                "is_broad_or_multi_sector_mandate": False,
                "government_debt_mandate": False,
                "corporate_credit_mandate": False,
                "non_confirmatory_mandate": False,
                "source_document": "NONE",
                "filing_accession": "NONE",
                "exact_source_section": "NONE",
                "parser_rule_id": "NO_SOURCE",
                "confidence_state": "SOURCE_NOT_FOUND",
                "evidence_snippet": "No pre-boundary statutory filing located on EDGAR"
            })
            continue

        source_found_cnt += 1

        # Prefer 485BPOS, then 485APOS, then 497, then N-1A
        candidates.sort(key=lambda x: (x[0], x[1] == "485BPOS", x[1] == "485APOS"), reverse=True)
        chosen_date, chosen_form, chosen_acc, chosen_pdoc = candidates[0]

        # Check series-level mapping
        is_single_series_trust = (cik_series_counts.get(cik, 0) == 1)
        if not is_single_series_trust:
            # Multi-series trust without individual series index mapping fails closed to AMBIGUOUS_SERIES_MAPPING
            series_ambiguous_cnt += 1
            mandate_remaining_cnt += 1
            attempted_records.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "mandate_status": "AMBIGUOUS_SERIES_MAPPING",
                "derived_mandate_classification": "UNRESOLVED_MANDATE",
                "is_sector_specific_mandate": False,
                "approved_sector": None,
                "is_broad_or_multi_sector_mandate": False,
                "government_debt_mandate": False,
                "corporate_credit_mandate": False,
                "non_confirmatory_mandate": False,
                "source_document": f"SEC_{chosen_form}",
                "filing_accession": chosen_acc,
                "exact_source_section": "NONE",
                "parser_rule_id": "AMBIGUOUS_MULTI_SERIES_TRUST",
                "confidence_state": "AMBIGUOUS_SOURCE_MAPPING",
                "evidence_snippet": f"Multi-series trust CIK {cik} with {cik_series_counts.get(cik)} series; series-specific document unpartitioned"
            })
            continue

        # Single-series trust: download and parse filing
        if not chosen_pdoc:
            chosen_pdoc = f"{chosen_acc}.htm"

        # Download filing
        success, local_path, raw_sha = download_filing(cik, chosen_acc, chosen_pdoc)
        if not success:
            download_failed_cnt += 1
            mandate_remaining_cnt += 1
            attempted_records.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "mandate_status": "DOWNLOAD_FAILED",
                "derived_mandate_classification": "UNRESOLVED_MANDATE",
                "is_sector_specific_mandate": False,
                "approved_sector": None,
                "is_broad_or_multi_sector_mandate": False,
                "government_debt_mandate": False,
                "corporate_credit_mandate": False,
                "non_confirmatory_mandate": False,
                "source_document": f"SEC_{chosen_form}",
                "filing_accession": chosen_acc,
                "exact_source_section": "NONE",
                "parser_rule_id": "DOWNLOAD_ERROR",
                "confidence_state": "DOWNLOAD_FAILED",
                "evidence_snippet": f"Download failed: {local_path}"
            })
            continue

        download_succeeded_cnt += 1

        # Read content and run frozen deterministic parser
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                html_text = f.read()
            strategy_text, sec_name = DeterministicMandateParser.extract_strategy_text(html_text)
            
            if not strategy_text or len(strategy_text.strip()) < 50:
                parse_failed_cnt += 1
                mandate_remaining_cnt += 1
                attempted_records.append({
                    "symbol": sym,
                    "cik": cik,
                    "series_id": sid,
                    "class_id": cid,
                    "mandate_status": "PARSE_FAILURE",
                    "derived_mandate_classification": "UNRESOLVED_MANDATE",
                    "is_sector_specific_mandate": False,
                    "approved_sector": None,
                    "is_broad_or_multi_sector_mandate": False,
                    "government_debt_mandate": False,
                    "corporate_credit_mandate": False,
                    "non_confirmatory_mandate": False,
                    "source_document": f"SEC_{chosen_form}",
                    "filing_accession": chosen_acc,
                    "exact_source_section": sec_name,
                    "parser_rule_id": "EMPTY_OR_UNEXTRACTABLE_SECTION",
                    "confidence_state": "PARSE_FAILED",
                    "evidence_snippet": "Strategy section not extractable"
                })
                continue

            parse_res = DeterministicMandateParser.parse_mandate(
                strategy_text=strategy_text,
                accession=chosen_acc,
                section_name=sec_name
            )

            # Map to resolution status under Policy v1.1
            if parse_res.non_confirmatory_mandate:
                other_etf_resolved_cnt += 1
                mandate_status = "RESOLVED_NON_CONFIRMATORY_MANDATE"
                derived_class = "NON_CONFIRMATORY_MANDATE"
            elif parse_res.government_debt_mandate:
                confirmatory_resolved_cnt += 1
                mandate_status = "RESOLVED_CONFIRMATORY_MANDATE"
                derived_class = "GOVERNMENT_DEBT_MANDATE"
            elif parse_res.corporate_credit_mandate:
                confirmatory_resolved_cnt += 1
                mandate_status = "RESOLVED_CONFIRMATORY_MANDATE"
                derived_class = "CORPORATE_CREDIT_MANDATE"
            elif parse_res.sector_specific_mandate and parse_res.approved_sector:
                confirmatory_resolved_cnt += 1
                mandate_status = "RESOLVED_CONFIRMATORY_MANDATE"
                derived_class = "DESIGNATED_SECTOR_MANDATE"
            elif parse_res.broad_or_multi_sector_mandate:
                confirmatory_resolved_cnt += 1
                mandate_status = "RESOLVED_CONFIRMATORY_MANDATE"
                derived_class = "BROAD_OR_MULTI_SECTOR_EQUITY_MANDATE"
            else:
                mandate_remaining_cnt += 1
                mandate_status = "UNRESOLVED_AMBIGUOUS_MANDATE"
                derived_class = "UNRESOLVED_MANDATE"

            attempted_records.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "mandate_status": mandate_status,
                "derived_mandate_classification": derived_class,
                "is_sector_specific_mandate": parse_res.sector_specific_mandate,
                "approved_sector": parse_res.approved_sector,
                "is_broad_or_multi_sector_mandate": parse_res.broad_or_multi_sector_mandate,
                "government_debt_mandate": parse_res.government_debt_mandate,
                "corporate_credit_mandate": parse_res.corporate_credit_mandate,
                "non_confirmatory_mandate": parse_res.non_confirmatory_mandate,
                "source_document": f"SEC_{chosen_form}",
                "filing_accession": chosen_acc,
                "exact_source_section": sec_name,
                "parser_rule_id": parse_res.parser_rule_id,
                "confidence_state": parse_res.confidence_state,
                "normalized_evidence_record_sha256": parse_res.evidence_text_hash,
                "evidence_snippet": parse_res.evidence_snippet
            })

        except Exception as e:
            parse_failed_cnt += 1
            mandate_remaining_cnt += 1
            attempted_records.append({
                "symbol": sym,
                "cik": cik,
                "series_id": sid,
                "class_id": cid,
                "mandate_status": "PARSE_FAILURE",
                "derived_mandate_classification": "UNRESOLVED_MANDATE",
                "is_sector_specific_mandate": False,
                "approved_sector": None,
                "is_broad_or_multi_sector_mandate": False,
                "government_debt_mandate": False,
                "corporate_credit_mandate": False,
                "non_confirmatory_mandate": False,
                "source_document": f"SEC_{chosen_form}",
                "filing_accession": chosen_acc,
                "exact_source_section": "ERROR",
                "parser_rule_id": "EXCEPTION",
                "confidence_state": "PARSE_FAILED",
                "evidence_snippet": str(e)
            })

    # Exact Accounting Assertions
    assert source_found_cnt + source_not_found_cnt == 2956, "SOURCE_FOUND + SOURCE_NOT_FOUND must equal 2956"
    assert confirmatory_resolved_cnt + other_etf_resolved_cnt + mandate_remaining_cnt == 2956, (
        f"CONFIRMATORY_RESOLVED ({confirmatory_resolved_cnt}) + OTHER_ETF_RESOLVED ({other_etf_resolved_cnt}) + "
        f"MANDATE_BLOCKERS_REMAINING ({mandate_remaining_cnt}) must equal 2956"
    )

    print("\n--- ACQUISITION STATUS ACCOUNTING ---")
    print(f"SOURCE_FOUND              = {source_found_cnt}")
    print(f"SOURCE_NOT_FOUND          = {source_not_found_cnt}")
    print(f"DOWNLOAD_SUCCEEDED        = {download_succeeded_cnt}")
    print(f"DOWNLOAD_FAILED           = {download_failed_cnt}")
    print(f"SERIES_MAPPING_AMBIGUOUS  = {series_ambiguous_cnt}")

    print("\n--- MANDATE RESOLUTION ACCOUNTING ---")
    print(f"MANDATE_BLOCKERS_START     = {total_attempted}")
    print(f"CONFIRMATORY_RESOLVED      = {confirmatory_resolved_cnt}")
    print(f"OTHER_ETF_RESOLVED         = {other_etf_resolved_cnt}")
    print(f"SOURCE_NOT_FOUND           = {source_not_found_cnt}")
    print(f"PARSE_FAILED               = {parse_failed_cnt}")
    print(f"AMBIGUOUS                  = {series_ambiguous_cnt}")
    print(f"MANDATE_BLOCKERS_REMAINING = {mandate_remaining_cnt}")

    # 4. Combine with existing canonical mandate evidence (77 entries)
    canonical_entries = []
    if MANDATE_EVIDENCE_PATH.exists():
        with open(MANDATE_EVIDENCE_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            # Only keep the 77 original entries (prevent duplicate appending)
            canonical_entries = [e for e in old_data.get("entries", []) if e["symbol"] not in mandate_pop["symbol"].values]

    full_entries = canonical_entries + attempted_records

    mandate_database = {
        "metadata": {
            "mandate_parser_version": "1.1.0",
            "mandate_parser_ruleset": DeterministicMandateParser.RULESET_ID,
            "governing_policy": "docs/research/ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1.json",
            "total_canonical_entries": len(canonical_entries),
            "total_attempted_entries": len(attempted_records),
            "total_mandate_database_entries": len(full_entries),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classification_boundary": f"{SNAPSHOT_BOUNDARY}T23:59:59Z",
            "post_boundary_source_used": 0
        },
        "entries": full_entries
    }

    with open(MANDATE_EVIDENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(mandate_database, f, indent=2)

    print(f"\nSaved updated mandate evidence database to {MANDATE_EVIDENCE_PATH}")
    print(f"Total entries: {len(full_entries)} (77 canonical + 2,956 attempted)")
    return {
        "source_found": source_found_cnt,
        "source_not_found": source_not_found_cnt,
        "download_succeeded": download_succeeded_cnt,
        "download_failed": download_failed_cnt,
        "series_ambiguous": series_ambiguous_cnt,
        "parse_failed": parse_failed_cnt,
        "confirmatory_resolved": confirmatory_resolved_cnt,
        "other_etf_resolved": other_etf_resolved_cnt,
        "mandate_remaining": mandate_remaining_cnt,
    }


if __name__ == "__main__":
    run_mandate_acquisition()
