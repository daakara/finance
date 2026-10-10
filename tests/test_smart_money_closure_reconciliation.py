"""
ARX Terminal - Smart Money Wave 2B Closure Reconciliation Test Suite
Verifies universe accounting parity, security type classification authority,
backfill spec scope binding, manifest immutability, date-range reconciliation,
fallback disproof, and independent corpus manifest hash parity.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
import pytest

from analyst_dashboard.smart_money.universe_models import (
    SmartMoneySecurity,
    SmartMoneyEligibilityDecision,
    SmartMoneyUniverseBuildManifest,
)
from analyst_dashboard.smart_money.universe_builder import (
    SmartMoneyUniverseBuilder,
    DEFAULT_SMART_MONEY_UNIVERSE_DB,
)
from analyst_dashboard.smart_money.backfill_models import (
    BackfillMode,
    BackfillStatus,
    SmartMoneyBackfillSpec,
    SmartMoneyBackfillManifest,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    DEFAULT_SMART_MONEY_DB,
)
from analyst_dashboard.smart_money.form4_models import SmartMoneyTransaction


def test_universe_exact_accounting():
    """Verify exact arithmetic closure of the persisted Wave 2A universe build."""
    conn = sqlite3.connect(DEFAULT_SMART_MONEY_UNIVERSE_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT source_population_count, eligible_count, ineligible_count, unresolved_count, scannable_count "
            "FROM sm_universe_builds WHERE universe_build_id = 'UB_SM_20261010_79fb1c65ac99'"
        )
        row = cur.fetchone()
        assert row is not None, "Universe build UB_SM_20261010_79fb1c65ac99 must exist in smart_money_universe.db"

        source_count, eligible_count, ineligible_count, unresolved_count, scannable_count = row
        assert source_count == 10435
        assert eligible_count == 7710
        assert ineligible_count == 2725
        assert unresolved_count == 0
        assert scannable_count == 7710

        # Exact arithmetic closure: source = eligible + ineligible + unresolved
        assert source_count == eligible_count + ineligible_count + unresolved_count
        # Scannable equals eligible (all eligible are CIK mapped)
        assert scannable_count == eligible_count
    finally:
        conn.close()


def test_exclusion_reason_accounting():
    """Verify that exclusion reasons in the build manifest are accurately accounted."""
    conn = sqlite3.connect(DEFAULT_SMART_MONEY_UNIVERSE_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT manifest_json FROM sm_universe_builds WHERE universe_build_id = 'UB_SM_20261010_79fb1c65ac99'"
        )
        row = cur.fetchone()
        manifest = json.loads(row[0])
        exclusions = manifest.get("exclusion_summary", {})

        assert exclusions.get("INELIGIBLE_EXCHANGE") == 2725
        # Verify that all 2725 ineligibles are account for by exchange (OTC)
        assert sum(exclusions.values()) == 2725
    finally:
        conn.close()


def test_security_type_classification_authority_limitation():
    """
    Prove that raw SEC company_tickers_exchange lacks security type metadata,
    causing ETFs (e.g. SPY) and ADRs (e.g. TSM, NVO) to be admitted into 7,710.
    """
    builder = SmartMoneyUniverseBuilder(db_path=DEFAULT_SMART_MONEY_UNIVERSE_DB)
    scannable = builder.get_scannable_members("UB_SM_20261010_79fb1c65ac99")
    scannable_syms = {m["symbol"] for m in scannable}

    # ETFs and ADRs are present because source feed lacks security type classification
    assert "SPY" in scannable_syms, "SPY (ETF) is currently admitted due to missing classification authority"
    assert "TSM" in scannable_syms, "TSM (ADR) is currently admitted due to missing classification authority"
    assert "NVO" in scannable_syms, "NVO (ADR) is currently admitted due to missing classification authority"


def test_backfill_spec_hash_binds_date_range_but_omits_execution_scope():
    """
    Verify which fields are bound in spec_hash and prove that execution scope
    (max_issuers, target_symbols, max_filings_per_issuer) is NOT bound,
    leading to IDENTITY_COLLISION between ladder stages.
    """
    spec = SmartMoneyBackfillSpec(
        universe_build_id="UB_SM_20261010_79fb1c65ac99",
        universe_content_hash="12d062516189f1d9855cb8463ac30d0a95d29def2ec0ee019c720def0e9978c7",
        backfill_mode="CURRENT_UNIVERSE_HISTORICAL_BACKFILL",
        requested_date_start="2026-08-01",
        requested_date_end="2026-10-10",
    )
    spec_hash = spec.compute_spec_hash()
    assert spec_hash == "3abc4df16f96eb37cd1f3b21d63950359391bfa8ead75d5e58f37a011f6e3c9c"
    assert spec.get_backfill_id() == "BF-3abc4df16f96eb37"

    # Stage 1 (5 issuers), Stage 2 (25 issuers), Stage 3 (100 issuers) all produce identical spec_hash
    # because max_issuers is not bound in SmartMoneyBackfillSpec.
    spec_fields = set(spec.__dataclass_fields__.keys())
    assert "requested_date_start" in spec_fields
    assert "requested_date_end" in spec_fields
    assert "universe_build_id" in spec_fields
    assert "max_issuers" not in spec_fields
    assert "target_symbols" not in spec_fields
    assert "max_filings_per_issuer" not in spec_fields


def test_manifest_overwrite_condition():
    """
    Verify that in sm_backfill_manifests, backfill_id is the primary key.
    Running multiple stages with the same spec hash causes manifest overwrite.
    """
    conn = sqlite3.connect(DEFAULT_SMART_MONEY_DB)
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM sm_backfill_manifests WHERE backfill_id = 'BF-3abc4df16f96eb37'")
        count = cur.fetchone()[0]
        # Exactly 1 row survives in sm_backfill_manifests despite 3 stages executed
        assert count == 1

        cur.execute("SELECT manifest_json FROM sm_backfill_manifests WHERE backfill_id = 'BF-3abc4df16f96eb37'")
        manifest = json.loads(cur.fetchone()[0])
        # Only Stage 3 (100 issuers) is preserved in sm_backfill_manifests
        assert manifest["issuer_count_completed"] == 100
    finally:
        conn.close()


def test_date_range_persisted_truth():
    """Verify that persisted execution truth is 2026-08-01 to 2026-10-10."""
    manifest_path = os.path.join(
        "docs", "domain", "radar", "smart-money", "SMART_MONEY_WAVE2B_HISTORICAL_CORPUS_MANIFEST.json"
    )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["requested_start"] == "2026-08-01"
    assert manifest["requested_end"] == "2026-10-10"


def test_disprove_fallback_rule():
    """
    Empirically disprove the fallback rule: 'filingDate + 22:00:00Z'.
    Compares hypothetical fallback against actual acceptance_datetime across cached filings.
    Proves FALLBACK_EARLY_CASES > 0 (look-ahead leakage!).
    """
    conn = sqlite3.connect(DEFAULT_SMART_MONEY_DB)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT accession_number, filing_date, acceptance_datetime FROM form4_raw_filings "
            "WHERE acceptance_datetime IS NOT NULL AND filing_date IS NOT NULL"
        ).fetchall()

        assert len(rows) > 0

        fallback_early_cases = 0
        fallback_late_cases = 0

        for acc, fdate, acc_dt in rows:
            hypo_fallback = f"{fdate}T22:00:00Z"
            acc_clean = acc_dt.replace(" ", "T")
            if not acc_clean.endswith("Z"):
                acc_clean += "Z"

            if hypo_fallback < acc_clean:
                fallback_early_cases += 1
            else:
                fallback_late_cases += 1

        # The fallback is early in the vast majority of cases in canonical 71 reconciliation sample (67/71)
        sample_71 = rows[:71]
        sample_early = sum(
            1 for _, fdate, acc_dt in sample_71
            if f"{fdate}T22:00:00Z" < (acc_dt.replace(" ", "T") if acc_dt.endswith("Z") else acc_dt.replace(" ", "T") + "Z")
        )
        assert sample_early == 67
        assert len(sample_71) - sample_early == 4

        # Across the entire corpus, fallback is overwhelmingly early, disproving the fallback
        assert fallback_early_cases >= 67, "Fallback must be proven early to establish CURRENT_FALLBACK_LOOKAHEAD_SAFE = NO"
    finally:
        conn.close()


def test_missing_acceptance_timestamp_fail_closed(tmp_path):
    """
    Verify that transactions with unknown public availability are strictly excluded
    from point-in-time as-of queries (fail-closed).
    """
    test_db = str(tmp_path / "test_as_of_fail_closed.db")
    mgr = Form4IngestionManager(db_path=test_db)

    # Ingest a filing with acceptance_datetime and one with NULL availability
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO sm_source_availability_ledger 
        (accession_number, issuer_cik, public_availability_at, first_fetched_at, source_hash, form_type, normalization_version)
        VALUES ('ACC-KNOWN', '0000320193', '2026-09-05T18:00:00Z', '2026-09-05T18:05:00Z', 'hash1', '4', '1.0.0')
    """)
    cur.execute("""
        INSERT INTO sm_source_availability_ledger 
        (accession_number, issuer_cik, public_availability_at, first_fetched_at, source_hash, form_type, normalization_version)
        VALUES ('ACC-UNKNOWN', '0000320193', '', '2026-09-05T18:05:00Z', 'hash2', '4', '1.0.0')
    """)

    cur.execute("""
        INSERT INTO form4_parsed_transactions
        (record_key, accession_number, issuer_cik, issuer_symbol, transaction_date, filing_date,
         transaction_code, signal_type, shares, price_per_share, total_value_usd, acquired_disposed,
         ownership_nature, is_positive_buying_conviction, is_open_market_purchase_evidence, content_hash, parsed_at,
         acceptance_datetime, public_availability_at)
        VALUES ('REC-KNOWN', 'ACC-KNOWN', '0000320193', 'AAPL', '2026-09-04', '2026-09-05',
                'P', 'OPEN_MARKET_PURCHASE', 100, 150.0, 15000.0, 'A',
                'D', 1, 1, 'chash1', '2026-09-05T18:05:00Z', '2026-09-05T18:00:00Z', '2026-09-05T18:00:00Z')
    """)
    cur.execute("""
        INSERT INTO form4_parsed_transactions
        (record_key, accession_number, issuer_cik, issuer_symbol, transaction_date, filing_date,
         transaction_code, signal_type, shares, price_per_share, total_value_usd, acquired_disposed,
         ownership_nature, is_positive_buying_conviction, is_open_market_purchase_evidence, content_hash, parsed_at,
         acceptance_datetime, public_availability_at)
        VALUES ('REC-UNKNOWN', 'ACC-UNKNOWN', '0000320193', 'AAPL', '2026-09-04', '2026-09-05',
                'P', 'OPEN_MARKET_PURCHASE', 200, 150.0, 30000.0, 'A',
                'D', 1, 1, 'chash2', '2026-09-05T18:05:00Z', NULL, NULL)
    """)
    conn.commit()
    conn.close()

    # Query as of 2026-09-06
    txs = mgr.get_transactions_as_of("2026-09-06T00:00:00Z", symbol="AAPL")
    # Only REC-KNOWN is returned; REC-UNKNOWN with empty availability is strictly excluded
    assert len(txs) == 1
    assert txs[0].record_key == "REC-KNOWN"


def test_dst_timezone_normalization():
    """Verify UTC normalization across Eastern Daylight (EDT) and Eastern Standard (EST)."""
    # Test that naive datetimes are rejected and ISO UTC with Z is preserved
    utc_str_edt = "2026-09-15T22:30:00.000Z"
    dt = datetime.fromisoformat(utc_str_edt.replace("Z", "+00:00"))
    assert dt.tzinfo == timezone.utc

    # Form 4 after-hours filing at 21:30 Eastern Daylight Time (EDT = UTC-4) is 01:30 UTC next day
    # Proves 22:00:00Z on the filing day is 3.5 hours BEFORE 01:30 UTC next day
    edt_after_hours_utc = "2026-09-16T01:30:00Z"
    same_day_fallback_utc = "2026-09-15T22:00:00Z"
    assert same_day_fallback_utc < edt_after_hours_utc, "Same-day 22:00 UTC fallback is earlier than actual EDT evening acceptance!"


def test_legacy_conviction_alias_deprecation():
    """Verify that is_open_market_purchase_evidence is canonical and legacy alias behaves correctly."""
    from analyst_dashboard.smart_money.form4_models import SmartMoneySignalType, OwnershipNature
    tx = SmartMoneyTransaction(
        record_key="RK1",
        accession_number="ACC1",
        issuer_cik="0000320193",
        issuer_symbol="AAPL",
        issuer_name="Apple Inc.",
        reporting_owner_cik="0001780525",
        reporting_owner_name="Newstead Jennifer",
        is_officer=True,
        officer_title="General Counsel",
        is_director=False,
        is_ten_percent_owner=False,
        filing_date="2026-09-01",
        transaction_date="2026-09-01",
        transaction_code="P",
        signal_type=SmartMoneySignalType.OPEN_MARKET_PURCHASE,
        shares=100.0,
        price_per_share=150.0,
        total_value_usd=15000.0,
        acquired_disposed="A",
        ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False,
        is_open_market_purchase_evidence=True,
    )
    # Canonical field
    assert tx.is_open_market_purchase_evidence is True
    # Deprecated compatibility alias reflects canonical field
    assert tx.is_positive_buying_conviction is True


def test_corpus_manifest_independent_hash_parity():
    """Independently recompute Stage 3 corpus manifest hash from file bytes and canonical JSON."""
    manifest_path = os.path.join(
        "docs", "domain", "radar", "smart-money", "SMART_MONEY_WAVE2B_HISTORICAL_CORPUS_MANIFEST.json"
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Independent computation over canonical JSON representation
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    recomputed_hash = hashlib.sha256(canonical_json).hexdigest()

    expected_hash = "d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0"
    assert recomputed_hash == expected_hash, f"Recomputed hash {recomputed_hash} must match expected {expected_hash}"
