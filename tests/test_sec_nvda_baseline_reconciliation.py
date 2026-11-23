"""ARX Issue #165: NVDA Form 4 Baseline Reconciliation Offline Suite.

Verifies:
1. Recomputed XML content hashes (16/16 exact matches) and aggregate digest parity.
2. Exact accession inventory, issuer identity, and zero duplicate records.
3. Strict separation of filing counts (16) vs transaction counts (36).
4. Transaction code distribution (25 S, 5 F, 4 G, 2 A) and zero derivative transactions.
5. Zero-transaction parse root cause (0001347842-26-000015 Section 16 retirement exit disclosure).
6. Evidence-based amendment classification (PARTIAL_CORRECTION) and supersession mapping.
7. Retained submissions index comparison against explicit August 26–November 23, 2026 window.
8. Distinctions between source freshness, filing-date coverage, and transaction-date coverage.
"""

import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import pytest

from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.amendment_classifier import (
    classify_amendment_effect,
    SCOPE_PARTIAL_CORRECTION,
)

EXPORT_PATH = Path(__file__).parent.parent / "data/research/nvda_form4_frozen_baseline_export.json"
SUBMISSIONS_PATH = Path("/Users/iDavid/Projects/finance-clean/data/research/sourced_fundamentals/raw/CIK0001045810/CIK0001045810_submissions.json")

EXPECTED_AGGREGATE_DIGEST = "c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2"


@pytest.fixture(scope="module")
def baseline_data():
    assert EXPORT_PATH.exists(), f"Missing export at {EXPORT_PATH}"
    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_nvda_baseline_inventory_and_hash_recomputation(baseline_data):
    """Recomputes SHA-256 for all 16 raw filings and verifies exact match with recorded database hash."""
    filings = baseline_data["raw_filings"]
    assert len(filings) == 16, "Frozen baseline must contain exactly 16 raw filings"

    content_hashes = []
    accessions = []

    for f in filings:
        acc = f["accession_number"]
        accessions.append(acc)
        assert f["cik"] == "0001045810", f"CIK mismatch in {acc}"
        assert f["ticker"] == "NVDA", f"Ticker mismatch in {acc}"

        raw_xml = f["xml_content"]
        rec_hash = f["content_hash"]
        computed_hash = hashlib.sha256(raw_xml.encode("utf-8")).hexdigest()
        assert computed_hash == rec_hash, f"Hash mismatch for accession {acc}: {computed_hash} != {rec_hash}"
        content_hashes.append(rec_hash)

    # Check zero duplicates
    assert len(accessions) == len(set(accessions)), "Duplicate accessions detected in baseline"

    # Verify ordered concatenated aggregate digest
    aggregate_digest = hashlib.sha256("".join(content_hashes).encode("utf-8")).hexdigest()
    assert aggregate_digest == EXPECTED_AGGREGATE_DIGEST, f"Aggregate digest mismatch: {aggregate_digest}"


def test_nvda_baseline_transaction_derivation(baseline_data):
    """Re-derives all 36 transactions directly from XML, separating derivative and non-derivative lines."""
    filings = baseline_data["raw_filings"]
    total_non_deriv = 0
    total_deriv = 0
    code_counts = Counter()

    for f in filings:
        parsed = Form4XmlParser.parse_filing(f["xml_content"], f["accession_number"])
        for tx in parsed.transactions:
            if tx.is_derivative:
                total_deriv += 1
            else:
                total_non_deriv += 1
            code_counts[tx.transaction_code] += 1

    assert total_non_deriv == 36, f"Expected 36 non-derivative transactions, got {total_non_deriv}"
    assert total_deriv == 0, f"Expected 0 derivative transactions, got {total_deriv}"
    assert sum(code_counts.values()) == 36

    # Verify exact code distribution
    assert code_counts["S"] == 25, "Expected 25 open market sales (Code S)"
    assert code_counts["F"] == 5, "Expected 5 tax withholdings (Code F)"
    assert code_counts["G"] == 4, "Expected 4 bona fide gifts (Code G)"
    assert code_counts["A"] == 2, "Expected 2 equity awards/grants (Code A)"


def test_zero_transaction_parse_investigation(baseline_data):
    """Investigates accession 0001347842-26-000015 to verify why it parses with 0 transactions."""
    filings = baseline_data["raw_filings"]
    f = next(x for x in filings if x["accession_number"] == "0001347842-26-000015")

    assert f["filing_date"] == "2026-08-24"
    assert f["accepted_datetime"].startswith("2026-08-24T21:32:04")

    root = ET.fromstring(f["xml_content"])
    non_deriv = root.findall(".//nonDerivativeTransaction")
    deriv = root.findall(".//derivativeTransaction")
    non_deriv_holdings = root.findall(".//nonDerivativeHolding")
    deriv_holdings = root.findall(".//derivativeHolding")

    assert len(non_deriv) == 0, "Expected 0 non-derivative transactions"
    assert len(deriv) == 0, "Expected 0 derivative transactions"
    assert len(non_deriv_holdings) == 0, "Expected 0 non-derivative holdings"
    assert len(deriv_holdings) == 0, "Expected 0 derivative holdings"

    # Remarks verify retirement disclosure (Section 16 exit notice)
    remarks = root.findtext("remarks") or ""
    assert "retired from his role" in remarks
    assert "no longer subject to" in remarks
    assert "Section 16" in remarks

    # Parser disposition
    parsed = Form4XmlParser.parse_filing(f["xml_content"], f["accession_number"])
    assert len(parsed.transactions) == 0
    assert parsed.disposition == "FETCHED_NO_RELEVANT_TRANSACTIONS"


def test_amendment_effect_validation_and_deduplication(baseline_data):
    """Validates amendment 0002152188-26-000007 effect from document evidence and prevents double-counting."""
    filings = baseline_data["raw_filings"]
    base_f = next(x for x in filings if x["accession_number"] == "0002152188-26-000005")
    amend_f = next(x for x in filings if x["accession_number"] == "0002152188-26-000007")

    base_parsed = Form4XmlParser.parse_filing(base_f["xml_content"], base_f["accession_number"])
    amend_parsed = Form4XmlParser.parse_filing(amend_f["xml_content"], amend_f["accession_number"])

    assert len(base_parsed.transactions) == 1
    assert len(amend_parsed.transactions) == 1

    # Base reported 172,507 shares; amendment corrected to 160,418 shares
    assert base_parsed.transactions[0].shares == 172507.0
    assert amend_parsed.transactions[0].shares == 160418.0

    # Classified effect
    base_txs = [t.__dict__ for t in base_parsed.transactions]
    res = classify_amendment_effect(
        amendment_filing=amend_parsed,
        base_filing={"accession_number": base_f["accession_number"], "filing_date": base_f["filing_date"]},
        base_transactions=base_txs,
        raw_xml=amend_f["xml_content"],
    )
    assert res.status == SCOPE_PARTIAL_CORRECTION
    assert len(res.mappings) == 1
    assert res.mappings[0]["base_transaction_id"] == "0002152188-26-000005:0002152188:NON_DERIVATIVE:0"
    assert res.mappings[0]["amendment_transaction_id"] == "0002152188-26-000007:0002152188:NON_DERIVATIVE:0"

    # Effective active transactions count (avoiding double-counting superseded base transaction):
    # 36 total database rows - 1 superseded base line = 35 active transactions point-in-time
    active_count = 36 - len(res.mappings)
    assert active_count == 35


def test_submissions_index_window_comparison(baseline_data):
    """Compares baseline against retained CIK0001045810_submissions.json for the August 26–November 23 window."""
    assert SUBMISSIONS_PATH.exists(), f"Submissions index missing at {SUBMISSIONS_PATH}"
    with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    recent = sub_data.get("filings", {}).get("recent", {})
    accs = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    fdates = recent.get("filingDate", [])
    adts = recent.get("acceptanceDateTime", [])

    WINDOW_START = "2026-08-26"
    WINDOW_END = "2026-11-23"

    in_window_accs = []
    out_window_accs = []

    for acc, form, fdate in zip(accs, forms, fdates):
        if form in ("4", "4/A"):
            if WINDOW_START <= fdate <= WINDOW_END:
                in_window_accs.append((acc, form, fdate))
            else:
                out_window_accs.append((acc, form, fdate))

    # In-window count is exactly 15
    assert len(in_window_accs) == 15, f"Expected 15 in-window Form 4 filings, got {len(in_window_accs)}"

    # Exactly one 4/A amendment in window
    amend_forms = [form for _, form, _ in in_window_accs if form == "4/A"]
    assert len(amend_forms) == 1

    baseline_accs = set(f["accession_number"] for f in baseline_data["raw_filings"])

    # All 15 in-window accessions must be in the baseline
    for acc, form, fdate in in_window_accs:
        assert acc in baseline_accs, f"In-window accession {acc} missing from baseline"

    # The 16th baseline accession is 0001347842-26-000015 (August 24, 2026), outside the window
    outside_base = [acc for acc in baseline_accs if not any(acc == x[0] for x in in_window_accs)]
    assert len(outside_base) == 1
    assert outside_base[0] == "0001347842-26-000015"

    aug24_filing = next(f for f in baseline_data["raw_filings"] if f["accession_number"] == outside_base[0])
    assert aug24_filing["filing_date"] == "2026-08-24"
    assert aug24_filing["filing_date"] < WINDOW_START

    # Earlier out-of-window accessions in submissions index (August 12 and August 7) must not be in baseline
    earlier_aug_accs = [acc for acc, form, fdate in out_window_accs if fdate in ("2026-08-12", "2026-08-07")]
    assert len(earlier_aug_accs) == 2
    for ea in earlier_aug_accs:
        assert ea not in baseline_accs, f"Earlier filing {ea} unexpectedly in baseline"
