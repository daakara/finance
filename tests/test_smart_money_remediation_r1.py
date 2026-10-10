"""
ARX Terminal - Smart Money Wave 2B-R1 Remediation Test Suite.
Validates:
1. Positive Security Type Classification Authority (OpenFIGI v3 via Security Master)
2. Zero Leakage Invariants (ETF=0, FPI/ADR=0, Unknown=0)
3. R1 Corrected Universe Accounting Closure & Legacy Immutability
4. Backfill Identity V2: Distinct Scope -> Distinct Execution ID & Scope Hash
5. Sealed Manifest Immutability & Overwrite Prohibition
6. Idempotency Retest Integrity
7. Unsafe T22 Fallback Elimination & Naive Datetime Rejection
8. Candidate 004 Production Shadow Isolation
"""

import os
import json
import sqlite3
import pytest
from datetime import datetime, timezone

from analyst_dashboard.smart_money.universe_models import (
    SmartMoneyEligibilityDecision,
    SmartMoneyUniverseBuildManifest,
)
from analyst_dashboard.smart_money.universe_builder import (
    SmartMoneyUniverseBuilder,
    DEFAULT_SMART_MONEY_UNIVERSE_DB,
    DEFAULT_SECURITY_MASTER_DB,
)
from analyst_dashboard.smart_money.backfill_models import (
    BackfillStatus,
    BackfillPlan,
    BackfillExecution,
    SealedExecutionManifest,
)
from analyst_dashboard.smart_money.backfill_engine import (
    SmartMoneyBackfillEngine,
    DEFAULT_SMART_MONEY_DB,
)
from analyst_dashboard.smart_money.form4_models import SmartMoneyTransaction, OwnershipNature
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager


def test_positive_security_type_classification_authority():
    """Verify that operating equities are classified with positive authority and funds/ADRs are excluded."""
    builder = SmartMoneyUniverseBuilder()

    # Operating equities with positive OpenFIGI provenance
    for sym in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]:
        prov = builder.resolve_classification(sym)
        assert prov.decision == SmartMoneyEligibilityDecision.ELIGIBLE_COMMON_EQUITY.value
        assert prov.source_authority == "OPENFIGI_V3_MAPPING"
        assert prov.security_type == "COMMON_STOCK"
        assert prov.security_type_status == "VERIFIED"
        assert prov.fpi_status == "NON_FPI"
        assert prov.fund_status == "NON_FUND"

    # ETFs must be marked INELIGIBLE with INELIGIBLE_ETF
    for etf_sym in ["SPY", "QQQ"]:
        prov = builder.resolve_classification(etf_sym)
        assert prov.decision == SmartMoneyEligibilityDecision.INELIGIBLE.value
        assert prov.reason_code == "INELIGIBLE_ETF"
        assert prov.fund_status == "FUND"

    # ADRs / Foreign Private Issuers must be marked INELIGIBLE with INELIGIBLE_ADR
    for adr_sym in ["TSM", "NVO", "ASML"]:
        prov = builder.resolve_classification(adr_sym)
        assert prov.decision == SmartMoneyEligibilityDecision.INELIGIBLE.value
        assert prov.reason_code == "INELIGIBLE_ADR"
        assert prov.fpi_status == "FPI"


def test_zero_leakage_in_scannable_population():
    """Verify ETF_LEAKAGE_COUNT = 0, FPI_ADR_LEAKAGE_COUNT = 0, UNKNOWN_LEAKAGE = 0."""
    builder = SmartMoneyUniverseBuilder()
    scannable = builder.get_scannable_members("UB_SM_20261010_R1_19ef857fec1f")
    scannable_symbols = {m["symbol"] for m in scannable}

    for etf in ["SPY", "QQQ"]:
        assert etf not in scannable_symbols, f"Leakage detected: {etf} in scannable population!"

    for fpi in ["TSM", "NVO", "ASML"]:
        assert fpi not in scannable_symbols, f"Leakage detected: {fpi} in scannable population!"

    # Operating equities must be present
    for equity in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]:
        assert equity in scannable_symbols, f"Missing operating equity: {equity} not in scannable population!"


def test_universe_exact_accounting_r1():
    """Verify exact arithmetic closure of R1 build and immutability of legacy build."""
    conn = sqlite3.connect(DEFAULT_SMART_MONEY_UNIVERSE_DB)
    try:
        cur = conn.cursor()
        # 1. Legacy build remains untouched
        cur.execute(
            "SELECT source_population_count, eligible_count, ineligible_count, unresolved_count, scannable_count "
            "FROM sm_universe_builds WHERE universe_build_id = 'UB_SM_20261010_79fb1c65ac99'"
        )
        leg = cur.fetchone()
        assert leg == (10435, 7710, 2725, 0, 7710)

        # 2. R1 corrected build
        cur.execute(
            "SELECT source_population_count, eligible_count, ineligible_count, unresolved_count, scannable_count, member_identity_hash "
            "FROM sm_universe_builds WHERE universe_build_id = 'UB_SM_20261010_R1_19ef857fec1f'"
        )
        r1 = cur.fetchone()
        assert r1 is not None, "R1 build UB_SM_20261010_R1_19ef857fec1f must exist"
        source, eligible, ineligible, unresolved, scannable, m_hash = r1

        assert source == 10435
        assert eligible == 106
        assert ineligible == 2749
        assert unresolved == 7580
        assert scannable == 106
        assert source == eligible + ineligible + unresolved, "Accounting closure violated!"
        assert scannable <= eligible
        assert m_hash == "ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c"
    finally:
        conn.close()


def test_backfill_identity_v2_separation():
    """Verify distinct scope produces distinct execution ID and scope hash across stages."""
    engine = SmartMoneyBackfillEngine()
    m1 = engine.get_execution_manifest("EXEC-babd5f07859e9fa0")
    m2 = engine.get_execution_manifest("EXEC-55780700f50deb74")
    m3 = engine.get_execution_manifest("EXEC-8b9fd84b91257507")

    assert m1 is not None and m2 is not None and m3 is not None
    assert m1.execution_id != m2.execution_id != m3.execution_id
    assert m1.scope_hash != m2.scope_hash != m3.scope_hash
    assert m1.compute_sealed_manifest_hash() != m2.compute_sealed_manifest_hash() != m3.compute_sealed_manifest_hash()

    assert m1.stage_label == "R1_STAGE_1"
    assert m2.stage_label == "R1_STAGE_2"
    assert m3.stage_label == "R1_STAGE_3"

    assert m1.status == BackfillStatus.SEALED_COMPLETE.value
    assert m2.status == BackfillStatus.SEALED_COMPLETE.value
    assert m3.status == BackfillStatus.SEALED_COMPLETE.value


def test_sealed_manifest_immutability():
    """Verify that attempting to overwrite a sealed manifest raises RuntimeError."""
    engine = SmartMoneyBackfillEngine()
    m1 = engine.get_execution_manifest("EXEC-babd5f07859e9fa0")
    assert m1.is_sealed

    with pytest.raises(RuntimeError, match="EXECUTION_MANIFEST_SEALED"):
        engine.save_execution_manifest(m1)


def test_idempotency_preserves_execution_id():
    """Verify that repeating an identical execution reuses the sealed manifest without side-effects."""
    engine = SmartMoneyBackfillEngine()
    manifest = engine.universe_builder.get_build_manifest("UB_SM_20261010_R1_19ef857fec1f")

    plan = BackfillPlan(
        universe_build_id=manifest.universe_build_id,
        universe_content_hash=manifest.member_identity_hash,
        backfill_mode="CURRENT_UNIVERSE_HISTORICAL_BACKFILL",
        requested_date_start="2026-08-01",
        requested_date_end="2026-10-10",
    )

    stage1_exec = engine.create_execution(
        plan=plan,
        target_symbols=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        max_filings_per_issuer=5,
        stage_label="R1_STAGE_1",
    )

    rerun = engine.run_execution(stage1_exec, plan, resume=True)
    assert rerun.execution_id == "EXEC-babd5f07859e9fa0"
    assert rerun.compute_sealed_manifest_hash() == "e0b79df813b8b67c499cbb1a5f8b2a11d3e7edc89e004b148a16adddb1f34e49"


def test_unsafe_fallback_eliminated():
    """Verify that missing acceptance_datetime produces None, never filingDate + 22:00:00Z."""
    # 1. In SmartMoneyTransaction model
    tx = SmartMoneyTransaction(
        record_key="TEST:RK",
        accession_number="0001-26-0001",
        issuer_cik="0000320193",
        issuer_symbol="AAPL",
        issuer_name="Apple Inc.",
        reporting_owner_cik="0001",
        reporting_owner_name="Owner",
        is_officer=True,
        officer_title="VP",
        is_director=False,
        is_ten_percent_owner=False,
        filing_date="2026-09-15",
        transaction_date="2026-09-15",
        transaction_code="P",
        signal_type="OPEN_MARKET_PURCHASE",
        shares=100.0,
        price_per_share=150.0,
        total_value_usd=15000.0,
        acquired_disposed="A",
        ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False,
        is_open_market_purchase_evidence=True,
        acceptance_datetime=None,  # Missing acceptance
        public_availability_at=None,
    )
    assert tx.public_availability_at is None, "Model must not invent 22:00:00Z fallback!"

    # 2. In Form4XmlParser
    xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <documentType>4</documentType>
        <periodOfReport>2026-09-15</periodOfReport>
        <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
        <reportingOwner><reportingOwnerId><rptOwnerCik>0001</rptOwnerCik><rptOwnerName>Owner</rptOwnerName></reportingOwnerId></reportingOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <transactionDate><value>2026-09-15</value></transactionDate>
                <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                <transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>150.0</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>
    """
    filing = Form4XmlParser.parse_filing(xml, "0001-26-0001", filing_date_hint="2026-09-15", acceptance_datetime=None)
    assert len(filing.transactions) == 1
    assert filing.transactions[0].public_availability_at is None, "Parser must not invent 22:00:00Z fallback!"


def test_naive_datetime_rejected():
    """Verify that get_transactions_as_of rejects naive datetimes lacking timezone info."""
    mgr = Form4IngestionManager()
    with pytest.raises(ValueError, match="Naive datetime rejected"):
        mgr.get_transactions_as_of("2026-09-15 12:00:00")

    with pytest.raises(ValueError, match="Naive datetime rejected"):
        mgr.get_transactions_as_of("2026-09-15T12:00:00")


def test_candidate_004_isolation():
    """Verify zero modified files in Candidate 004 paths."""
    import subprocess
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        path = line[3:].strip()
        assert not path.startswith("analyst_dashboard/vcp/"), f"Candidate 004 violation: {path}"
        assert "shadow_evidence.db" not in path, f"Candidate 004 violation: {path}"
        assert "natural_trigger.py" not in path, f"Candidate 004 violation: {path}"
