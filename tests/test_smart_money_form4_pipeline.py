"""
Comprehensive Test Suite for SEC Form 4 Smart Money Pipeline.
Verifies parsing, signal semantics, deduplication, false-positive controls,
readiness transitions, and arithmetic ranking.
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneySignalType,
    OwnershipNature,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    RateLimiter,
)
from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner
from analyst_dashboard.smart_money.universe_builder import SmartMoneyUniverseBuilder

SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-08-20</periodOfReport>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001234567</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-18</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

SAMPLE_GRANT_AND_EXERCISE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-08-21</periodOfReport>
  <issuer>
    <issuerCik>0000789019</issuerCik>
    <issuerName>MICROSOFT CORP</issuerName>
    <issuerTradingSymbol>MSFT</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0009876543</rptOwnerCik>
      <rptOwnerName>NADELLA SATYA</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-19</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>25000</value></transactionShares>
        <transactionPricePerShare><value>0.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-20</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>0.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-20</value></transactionDate>
      <transactionCoding><transactionCode>F</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>4500</value></transactionShares>
        <transactionPricePerShare><value>415.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_form4_parser_open_market_purchase():
    """Verify that code P + Acquired A + price > 0 is correctly identified as positive buying conviction."""
    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_XML,
        accession_number="0000320193-26-000100",
    )
    assert filing.issuer_symbol == "AAPL"
    assert filing.issuer_cik == "0000320193"
    assert len(filing.transactions) == 1

    tx = filing.transactions[0]
    assert tx.transaction_code == "P"
    assert tx.signal_type == SmartMoneySignalType.OPEN_MARKET_PURCHASE
    assert tx.shares == 5000.0
    assert tx.price_per_share == 225.50
    assert tx.total_value_usd == 1127500.0
    assert tx.is_positive_buying_conviction is True
    assert tx.officer_title == "Chief Executive Officer"
    assert tx.is_officer is True
    assert tx.is_director is True
    assert tx.ownership_nature == OwnershipNature.DIRECT


def test_false_positive_controls_grant_and_exercise():
    """Verify that grants (A), exercises (M), and tax withholdings (F) fail closed and are NOT positive conviction."""
    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_GRANT_AND_EXERCISE_XML,
        accession_number="0000789019-26-000200",
    )
    assert filing.issuer_symbol == "MSFT"
    assert len(filing.transactions) == 3

    grant_tx = filing.transactions[0]
    assert grant_tx.transaction_code == "A"
    assert grant_tx.signal_type == SmartMoneySignalType.EQUITY_GRANT
    assert grant_tx.is_positive_buying_conviction is False  # False-positive guard

    exercise_tx = filing.transactions[1]
    assert exercise_tx.transaction_code == "M"
    assert exercise_tx.signal_type == SmartMoneySignalType.OPTION_EXERCISE
    assert exercise_tx.is_positive_buying_conviction is False  # False-positive guard

    tax_tx = filing.transactions[2]
    assert tax_tx.transaction_code == "F"
    assert tax_tx.signal_type == SmartMoneySignalType.TAX_WITHHOLDING
    assert tax_tx.is_positive_buying_conviction is False


def test_false_positive_controls_zero_price_purchase():
    """Verify that code P with zero or missing price fails closed."""
    zero_price_xml = SAMPLE_FORM4_XML.replace("<value>225.50</value>", "<value>0.00</value>")
    filing = Form4XmlParser.parse_filing(
        raw_content=zero_price_xml,
        accession_number="0000320193-26-000101",
    )
    tx = filing.transactions[0]
    assert tx.transaction_code == "P"
    assert tx.price_per_share == 0.0
    assert tx.is_positive_buying_conviction is False


def test_deduplication_record_key():
    """Verify unique composite primary keys for Form 4 transactions."""
    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_XML,
        accession_number="0000320193-26-000100",
    )
    tx = filing.transactions[0]
    expected_key = "0000320193-26-000100:0001234567:NON_DERIVATIVE:0"
    assert tx.record_key == expected_key
    assert len(tx.content_hash) == 64


def test_ingestion_and_sqlite_persistence(tmp_path):
    """Verify that Form4IngestionManager saves and loads transactions idempotently."""
    db_path = str(tmp_path / "test_smart_money.db")
    mgr = Form4IngestionManager(db_path=db_path)

    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_XML,
        accession_number="0000320193-26-000100",
        source_url="https://sec.gov/test",
    )
    mgr.save_raw_filing_and_transactions(filing, "https://sec.gov/test")

    # Query back
    txs = mgr.get_parsed_transactions(symbol="AAPL")
    assert len(txs) == 1
    assert txs[0].record_key == "0000320193-26-000100:0001234567:NON_DERIVATIVE:0"
    assert txs[0].is_positive_buying_conviction is True

    # Second save (idempotency check)
    mgr.save_raw_filing_and_transactions(filing, "https://sec.gov/test")
    txs_after = mgr.get_parsed_transactions(symbol="AAPL")
    assert len(txs_after) == 1  # No duplicate rows created


def test_scanner_clustering_and_conviction_scoring(tmp_path):
    """Verify that multiple insider purchases trigger cluster buy bonus and C-Suite score."""
    db_path = str(tmp_path / "test_scanner.db")
    mgr = Form4IngestionManager(db_path=db_path)

    # Insider 1: CEO buys $1.12M
    filing1 = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_XML,
        accession_number="0000320193-26-000100",
    )
    mgr.save_raw_filing_and_transactions(filing1, "https://sec.gov/test1")

    # Insider 2: CFO buys $300k
    cfo_xml = SAMPLE_FORM4_XML.replace("0001234567", "0007654321")
    cfo_xml = cfo_xml.replace("COOK TIMOTHY D", "MAESTRI LUCA")
    cfo_xml = cfo_xml.replace("Chief Executive Officer", "Chief Financial Officer")
    cfo_xml = cfo_xml.replace("<value>5000</value>", "<value>1500</value>")
    cfo_xml = cfo_xml.replace("<value>225.50</value>", "<value>200.00</value>")
    filing2 = Form4XmlParser.parse_filing(
        cfo_xml,
        accession_number="0000320193-26-000102",
    )
    mgr.save_raw_filing_and_transactions(filing2, "https://sec.gov/test2")

    scanner = SmartMoneyForm4Scanner(ingestion_mgr=mgr)
    state = scanner.evaluate_issuer_transactions(
        symbol="AAPL",
        transactions=mgr.get_parsed_transactions(symbol="AAPL"),
        lookback_days=30,
        now_date="2026-08-25",
    )

    assert state.cluster_buy is True
    assert state.distinct_buyer_count == 2
    assert state.c_suite_buying is True
    assert state.gross_purchase_value_usd == 1127500.0 + 300000.0
    assert state.conviction_score >= 90.0  # High conviction from value + cluster + C-Suite
    assert state.qualifies_for_radar is True


def test_readiness_contract_false_zero_prevention(tmp_path):
    """Verify that unready scanner outputs PIPELINE_PENDING with matched_count: None."""
    db_path = str(tmp_path / "test_universe.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)
    manifest, _ = builder.build_universe(
        raw_records=[
            {"ticker": "AAPL", "exchange": "NASDAQ", "name": "APPLE INC", "cik": 320193, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
            {"ticker": "MSFT", "exchange": "NASDAQ", "name": "MICROSOFT CORP", "cik": 789019, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
        ],
        source_snapshot_id="SNAP_TEST_001",
        as_of="2026-10-10T12:00:00Z",
    )

    scanner = SmartMoneyForm4Scanner(universe_builder=builder)

    # 1. Market-Wide Mode without universe_build_id: Must fail closed with ValueError
    with pytest.raises(ValueError, match="SNAPSHOT_WITHOUT_UNIVERSE_BUILD_IN_MARKET_WIDE_MODE"):
        scanner.scan_issuers(
            symbols=["AAPL", "MSFT"],
            is_bounded_sample=False,
        )

    # 2. Market-Wide Mode with universe_build_id: Must fail closed to PIPELINE_PENDING with matched_count == None
    prod_snap = scanner.scan_issuers(
        universe_build_id=manifest.universe_build_id,
        is_bounded_sample=False,
    )
    assert prod_snap.status == "PIPELINE_PENDING"
    assert prod_snap.matched_count is None  # Invariant: Never false zero
    assert len(prod_snap.items) == 0
    assert prod_snap.universe_build_id == manifest.universe_build_id
    assert prod_snap.scannable_count == 2
    assert prod_snap.eligible_count == 2

    # 3. Readiness checklist evaluation
    readiness = scanner.evaluate_readiness(is_market_wide_governed=False)
    assert readiness.status == "PIPELINE_PENDING"
    assert readiness.to_dict()["checks"]["market_wide_universe_governed"] is False
    assert readiness.market_wide_universe_governed is False


def test_bounded_sample_snapshot_with_zero_matches(tmp_path):
    """Verify that when bounded sample completes with 0 matches, it reports NO_MATCHES with matched_count == 0."""
    db_path = str(tmp_path / "test_empty.db")
    mgr = Form4IngestionManager(db_path=db_path)
    scanner = SmartMoneyForm4Scanner(ingestion_mgr=mgr)

    snap = scanner.scan_issuers(
        symbols=["XYZ"],
        is_bounded_sample=True,
    )
    assert snap.status == "NO_MATCHES"
    assert snap.matched_count == 0
    assert snap.scannable_count == 1
