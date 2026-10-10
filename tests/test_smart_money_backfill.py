"""
Comprehensive Test Suite for Smart Money Wave 2B:
Point-in-Time Historical Form 4 Ingestion, Backfill Engine, Source Availability Ledger,
Knowledge-Time Semantics, Amendment Policy, and Resumability.
"""

import os
import json
import sqlite3
import pytest
from unittest.mock import MagicMock

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
    SmartMoneySignalType,
    OwnershipNature,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager
from analyst_dashboard.smart_money.universe_builder import SmartMoneyUniverseBuilder
from analyst_dashboard.smart_money.backfill_models import (
    BackfillMode,
    BackfillStatus,
    FilingDisposition,
    AmendmentRelationship,
    SmartMoneyBackfillSpec,
    SmartMoneyBackfillManifest,
    SourceAvailabilityRecord,
)
from analyst_dashboard.smart_money.backfill_engine import SmartMoneyBackfillEngine

SAMPLE_FORM4_AAPL_XML = """<?xml version="1.0"?>
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

SAMPLE_FORM4_AAPL_AMENDMENT_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4/A</documentType>
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
        <transactionShares><value>5500</value></transactionShares>
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

SAMPLE_FORM4_NVDA_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-08-22</periodOfReport>
  <issuer>
    <issuerCik>0001045810</issuerCik>
    <issuerName>NVIDIA CORP</issuerName>
    <issuerTradingSymbol>NVDA</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001999999</rptOwnerCik>
      <rptOwnerName>HUANG JEN HSUN</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>President &amp; CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-21</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>120.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


@pytest.fixture
def test_env(tmp_path):
    """Sets up isolated SQLite databases for universe and smart money cache."""
    cache_db = str(tmp_path / "smart_money_cache.db")
    universe_db = str(tmp_path / "smart_money_universe.db")

    universe_builder = SmartMoneyUniverseBuilder(db_path=universe_db)
    raw_pop = [
        {"ticker": "AAPL", "exchange": "NASDAQ", "name": "APPLE INC", "cik": 320193, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
        {"ticker": "NVDA", "exchange": "NASDAQ", "name": "NVIDIA CORP", "cik": 1045810, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
        {"ticker": "MSFT", "exchange": "NASDAQ", "name": "MICROSOFT CORP", "cik": 789019, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    ]
    manifest, _ = universe_builder.build_universe(
        raw_records=raw_pop,
        source_snapshot_id="SNAP_TEST_001",
        as_of="2026-10-10T12:00:00Z",
    )

    ingestion_mgr = Form4IngestionManager(db_path=cache_db)
    engine = SmartMoneyBackfillEngine(
        db_path=cache_db,
        ingestion_mgr=ingestion_mgr,
        universe_builder=universe_builder,
    )
    return {
        "cache_db": cache_db,
        "universe_db": universe_db,
        "universe_build_id": manifest.universe_build_id,
        "universe_builder": universe_builder,
        "ingestion_mgr": ingestion_mgr,
        "engine": engine,
    }


def test_backfill_mode_gating_and_survivorship_warning(test_env):
    """Verify Mode B is blocked, Mode A succeeds and attaches mandatory survivorship warning."""
    engine = test_env["engine"]
    u_id = test_env["universe_build_id"]

    # 1. Mode B: PIT_HISTORICAL_UNIVERSE_BACKFILL must fail closed
    with pytest.raises(ValueError, match="PIT_HISTORICAL_UNIVERSE_BACKFILL_NOT_AUTHORIZED"):
        engine.create_backfill_plan(
            universe_build_id=u_id,
            backfill_mode=BackfillMode.PIT_HISTORICAL_UNIVERSE_BACKFILL.value,
            requested_date_start="2026-07-01",
            requested_date_end="2026-10-01",
        )

    # 2. Mode A: CURRENT_UNIVERSE_HISTORICAL_BACKFILL succeeds
    spec = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-07-01",
        requested_date_end="2026-10-01",
    )
    assert spec.backfill_mode == "CURRENT_UNIVERSE_HISTORICAL_BACKFILL"
    assert spec.universe_build_id == u_id

    # 3. Manifest contains mandatory survivorship warning
    manifest = SmartMoneyBackfillManifest(
        backfill_id=spec.get_backfill_id(),
        backfill_spec_hash=spec.compute_spec_hash(),
        backfill_mode=spec.backfill_mode,
        universe_build_id=spec.universe_build_id,
        universe_content_hash=spec.universe_content_hash,
        requested_start=spec.requested_date_start,
        requested_end=spec.requested_date_end,
        started_at="2026-10-10T12:00:00Z",
    )
    assert "survivorship bias" in manifest.survivorship_bias_warning.lower()
    assert manifest.historical_market_coverage_claim == "NOT_AUTHORIZED"
    assert manifest.historical_performance_claim == "NOT_AUTHORIZED"


def test_spec_hash_determinism_and_wall_clock_invariance(test_env):
    """Verify that backfill specification hash is deterministic and invariant to execution time."""
    engine = test_env["engine"]
    u_id = test_env["universe_build_id"]

    spec_1 = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-07-01",
        requested_date_end="2026-10-01",
    )
    spec_2 = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-07-01",
        requested_date_end="2026-10-01",
    )
    assert spec_1.compute_spec_hash() == spec_2.compute_spec_hash()
    assert spec_1.get_backfill_id() == spec_2.get_backfill_id()


def test_accession_identity_and_cache_reuse(test_env):
    """Verify that identical accession + content hash reuses cached raw filing and increments cache hits."""
    mgr = test_env["ingestion_mgr"]
    accession = "0000320193-26-000100"

    # 1. Mock fetcher returning raw XML on first call
    def mock_fetch(url, *args, **kwargs):
        mgr._request_count += 1
        return SAMPLE_FORM4_AAPL_XML

    mgr._fetch_url_with_retry = mock_fetch
    filing_1 = mgr.ingest_filing_by_accession(
        cik="0000320193",
        accession_number=accession,
        filing_date="2026-08-20",
        acceptance_dt="2026-08-20T17:35:12Z",
        force_refresh=False,
    )
    assert filing_1 is not None
    assert len(filing_1.transactions) == 1
    stats_1 = mgr.get_stats()
    assert stats_1["request_count"] == 1
    assert stats_1["cache_hits"] == 0

    # 2. Second ingestion of identical accession reuses cache (0 network requests, 1 cache hit)
    filing_2 = mgr.ingest_filing_by_accession(
        cik="0000320193",
        accession_number=accession,
        filing_date="2026-08-20",
        acceptance_dt="2026-08-20T17:35:12Z",
        force_refresh=False,
    )
    assert filing_2 is not None
    stats_2 = mgr.get_stats()
    assert stats_2["request_count"] == 1  # Unchanged
    assert stats_2["cache_hits"] == 1    # Incremented


def test_raw_source_hashing_and_retention(test_env):
    """Verify that raw source bytes are stored, SHA-256 verifiable, and traceable."""
    mgr = test_env["ingestion_mgr"]
    accession = "0000320193-26-000100"

    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=accession,
        filing_date_hint="2026-08-20",
        acceptance_datetime="2026-08-20T17:35:12Z",
        source_url="https://sec.gov/test.xml",
    )
    mgr.save_raw_filing_and_transactions(filing, source_url="https://sec.gov/test.xml")

    # Verify raw retention in SQLite
    cached = mgr.get_cached_raw_filing(accession)
    assert cached is not None
    assert cached["raw_content"] == filing.raw_xml
    expected_hash = filing.source_content_hash
    assert cached["source_content_hash"] == expected_hash
    assert expected_hash != ""


def test_transaction_identity_uniqueness_and_no_collisions():
    """Verify that composite transaction keys prevent collisions across joint owners, codes, and sections."""
    joint_xml = """<?xml version="1.0"?>
    <ownershipDocument>
      <documentType>4</documentType>
      <periodOfReport>2026-08-20</periodOfReport>
      <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0001111111</rptOwnerCik><rptOwnerName>INSIDER A</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-08-18</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>200.0</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
          <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-08-18</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>200.0</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
          <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <derivativeTable>
        <derivativeTransaction>
          <transactionDate><value>2026-08-18</value></transactionDate>
          <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>50.0</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
          <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </derivativeTransaction>
      </derivativeTable>
    </ownershipDocument>
    """
    filing = Form4XmlParser.parse_filing(raw_content=joint_xml, accession_number="ACC-JOINT-001")
    keys = [tx.record_key for tx in filing.transactions]
    assert len(keys) == 3
    # Invariant: TRANSACTION_IDENTITY_COLLISION_COUNT == 0
    assert len(set(keys)) == len(keys)
    assert keys[0] == "ACC-JOINT-001:0001111111:NON_DERIVATIVE:0"
    assert keys[1] == "ACC-JOINT-001:0001111111:NON_DERIVATIVE:1"
    assert keys[2] == "ACC-JOINT-001:0001111111:DERIVATIVE:0"


def test_form4_amendment_policy_and_lineage(test_env):
    """Verify Form 4 vs Form 4/A lineage tracking; original filing preserved and marked SUPERSEDED."""
    mgr = test_env["ingestion_mgr"]
    acc_orig = "0000320193-26-000100"
    acc_amend = "0000320193-26-000101"

    # 1. Ingest Original
    filing_orig = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=acc_orig,
        filing_date_hint="2026-08-20",
        acceptance_datetime="2026-08-20T17:35:12Z",
    )
    mgr.save_raw_filing_and_transactions(filing_orig, source_url="https://sec.gov/orig.xml")

    # 2. Ingest Amendment
    filing_amend = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_AMENDMENT_XML,
        accession_number=acc_amend,
        filing_date_hint="2026-08-21",
        acceptance_datetime="2026-08-21T16:00:00Z",
    )
    mgr.save_raw_filing_and_transactions(filing_amend, source_url="https://sec.gov/amend.xml")

    # 3. Check lineage relationships in sm_amendments
    conn = sqlite3.connect(test_env["cache_db"])
    cursor = conn.cursor()
    cursor.execute("SELECT relationship FROM sm_amendments WHERE accession_number = ?", (acc_amend,))
    rel_amend = cursor.fetchone()[0]
    cursor.execute("SELECT relationship FROM sm_amendments WHERE accession_number = ?", (acc_orig,))
    rel_orig = cursor.fetchone()[0]
    conn.close()

    assert rel_amend == "AMENDMENT"
    assert rel_orig == "SUPERSEDED_OR_CORRECTED"

    # 4. Invariant: Original raw filing still exists (never deleted)
    cached_orig = mgr.get_cached_raw_filing(acc_orig)
    assert cached_orig is not None


def test_source_availability_ledger_cannot_be_rewritten_earlier(test_env):
    """Verify that once recorded, source public availability cannot be backdated."""
    mgr = test_env["ingestion_mgr"]
    acc = "0000320193-26-000150"

    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=acc,
        filing_date_hint="2026-08-20",
        acceptance_datetime="2026-08-20T17:35:12Z",
    )
    mgr.save_raw_filing_and_transactions(filing, source_url="")

    # Verify ledger recorded
    conn = sqlite3.connect(test_env["cache_db"])
    cursor = conn.cursor()
    cursor.execute("SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number = ?", (acc,))
    initial_avail = cursor.fetchone()[0]
    assert initial_avail == "2026-08-20T17:35:12Z"

    # Attempt to backdate filing
    filing_backdated = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=acc,
        filing_date_hint="2026-08-10",
        acceptance_datetime="2026-08-10T10:00:00Z",
    )
    mgr.save_raw_filing_and_transactions(filing_backdated, source_url="")

    # Verify ledger was NOT rewritten earlier
    cursor.execute("SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number = ?", (acc,))
    persisted_avail = cursor.fetchone()[0]
    conn.close()
    assert persisted_avail == initial_avail  # Invariant: SOURCE_AVAILABILITY_CAN_BE_REWRITTEN_EARLIER = NO


def test_as_of_query_zero_look_ahead_leakage(test_env):
    """
    Point-in-Time Test:
    Filing accepted on 2026-08-20T17:35:12Z with trade date 2026-08-18.
    Must NOT leak at trade date or pre-acceptance timestamps.
    """
    mgr = test_env["ingestion_mgr"]
    acc = "0000320193-26-000200"

    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=acc,
        filing_date_hint="2026-08-20",
        acceptance_datetime="2026-08-20T17:35:12Z",
    )
    mgr.save_raw_filing_and_transactions(filing, source_url="")

    # 1. As-of trade date (2026-08-18): 0 records (no trade-date leakage)
    txs_trade_date = mgr.get_transactions_as_of(as_of_time="2026-08-18T23:59:59Z", symbol="AAPL")
    assert len(txs_trade_date) == 0

    # 2. As-of 1 second before acceptance: 0 records
    txs_before_acc = mgr.get_transactions_as_of(as_of_time="2026-08-20T17:35:11Z", symbol="AAPL")
    assert len(txs_before_acc) == 0

    # 3. As-of exact acceptance second: 1 record
    txs_at_acc = mgr.get_transactions_as_of(as_of_time="2026-08-20T17:35:12Z", symbol="AAPL")
    assert len(txs_at_acc) == 1
    assert txs_at_acc[0].record_key == f"{acc}:0001234567:NON_DERIVATIVE:0"

    # 4. As-of subsequent day: 1 record
    txs_future = mgr.get_transactions_as_of(as_of_time="2026-08-21T12:00:00Z", symbol="AAPL")
    assert len(txs_future) == 1


def test_backfill_crash_and_resume(test_env):
    """Verify that interrupted backfill safely resumes from durable checkpoints without duplicate fetches."""
    engine = test_env["engine"]
    u_id = test_env["universe_build_id"]

    spec = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-08-01",
        requested_date_end="2026-08-31",
    )

    # Mock SEC responses
    def mock_ingest(symbol=None, cik=None, **kwargs):
        if cik == "0000320193" or symbol == "AAPL":
            return [Form4XmlParser.parse_filing(SAMPLE_FORM4_AAPL_XML, "ACC-AAPL-001")]
        elif cik == "0001045810" or symbol == "NVDA":
            return [Form4XmlParser.parse_filing(SAMPLE_FORM4_NVDA_XML, "ACC-NVDA-001")]
        return []

    engine.ingestion_mgr.ingest_issuer = MagicMock(side_effect=mock_ingest)

    # 1. First run bounded to 1 issuer (simulating crash after first issuer)
    manifest_1 = engine.run_backfill(spec, max_issuers=1, resume=True)
    assert manifest_1.issuer_count_completed == 1

    # Verify AAPL checkpoint exists
    completed = engine.get_completed_issuers_for_backfill(spec.get_backfill_id())
    assert len(completed) == 1

    # 2. Second run with max_issuers=3 (resumes from checkpoint)
    manifest_2 = engine.run_backfill(spec, max_issuers=3, resume=True)
    assert manifest_2.issuer_count_completed == 3
    # Invariant: DUPLICATE_TRANSACTIONS_AFTER_RESUME == 0
    assert manifest_2.duplicates_detected == 0


def test_backfill_idempotency_identical_rerun(test_env):
    """Verify that running identical backfill twice yields 0 new filings, 0 new transactions, and identical hashes."""
    engine = test_env["engine"]
    u_id = test_env["universe_build_id"]

    spec = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-08-01",
        requested_date_end="2026-08-31",
    )

    def mock_ingest(symbol=None, cik=None, **kwargs):
        f = Form4XmlParser.parse_filing(SAMPLE_FORM4_AAPL_XML, f"ACC-IDEMP-{symbol}")
        engine.ingestion_mgr.save_raw_filing_and_transactions(f, "")
        return [f]

    engine.ingestion_mgr.ingest_issuer = MagicMock(side_effect=mock_ingest)

    manifest_1 = engine.run_backfill(spec, max_issuers=2, resume=False, force_refresh=True)
    manifest_2 = engine.run_backfill(spec, max_issuers=2, resume=True, force_refresh=False)

    assert manifest_1.source_manifest_hash == manifest_2.source_manifest_hash
    assert manifest_1.transaction_manifest_hash == manifest_2.transaction_manifest_hash
    assert manifest_2.status == "COMPLETE"


def test_deterministic_parser_replay(test_env):
    """Verify that re-parsing cached raw filings yields 0 mismatches."""
    engine = test_env["engine"]
    acc = "0000320193-26-000300"

    filing = Form4XmlParser.parse_filing(
        raw_content=SAMPLE_FORM4_AAPL_XML,
        accession_number=acc,
        filing_date_hint="2026-08-20",
        acceptance_datetime="2026-08-20T17:35:12Z",
    )
    engine.ingestion_mgr.save_raw_filing_and_transactions(filing, source_url="")

    # Replay parser 5 times
    hashes = set()
    for _ in range(5):
        s_hash, tx_hash = engine.replay_parser_for_accession(acc)
        hashes.add((s_hash, tx_hash))

    assert len(hashes) == 1  # Invariant: PARSER_REPLAY_MISMATCHES == 0


def test_source_disposition_closure(test_env):
    """Verify that all filing dispositions are classified into the closed enum set (zero unclassified)."""
    engine = test_env["engine"]
    u_id = test_env["universe_build_id"]

    spec = engine.create_backfill_plan(
        universe_build_id=u_id,
        backfill_mode=BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value,
        requested_date_start="2026-08-01",
        requested_date_end="2026-08-31",
    )

    def mock_ingest(symbol=None, cik=None, **kwargs):
        f = Form4XmlParser.parse_filing(SAMPLE_FORM4_AAPL_XML, f"ACC-DISP-{symbol}")
        f.disposition = FilingDisposition.FETCHED_AND_PARSED.value
        return [f]

    engine.ingestion_mgr.ingest_issuer = MagicMock(side_effect=mock_ingest)
    manifest = engine.run_backfill(spec, max_issuers=2, resume=False)

    assert manifest.disposition_counts[FilingDisposition.FETCHED_AND_PARSED.value] >= 1
    # Verify no unclassified keys
    allowed_keys = {d.value for d in FilingDisposition}
    for k in manifest.disposition_counts.keys():
        assert k in allowed_keys
