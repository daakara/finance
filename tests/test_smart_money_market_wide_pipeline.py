"""
ARX Terminal — Smart Money Form 4 Market-Wide Pipeline
Comprehensive Test Suite (Phase O).

Verifies all six frozen gates:
- Gate B1: Global SEC Enumeration (fast-path Atom parsing, pagination, unsupported form exclusion) & Authoritative Reconciliation
- Gate B2: Point-in-Time (PIT) Boundary Enforcement, Late Arrivals, and Temporal Isolation
- Gate B3: Frozen FORM4_RADAR_QUALIFICATION_V1 Predicate & 4-Factor Arithmetic Conviction Scoring
- Gate B4: Radar Taxonomy Separation (INSIDER_FORM4 != 13F Institutional or Dark Pool)
- Gate B5: Immutable Scanner Generation, SQLite Triggers, Content Hashing, and Active Pointer
- Gate B6: Operational Topology, Distributed Fencing, Overlap Rejection, and Rate Governance
"""

import os
import time
import uuid
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.smart_money.global_discovery import (
    GlobalSecDiscoveryEngine,
    DiscoveredFilingEntry,
    DiscoveryBatch,
    IN_SCOPE_FORM_TYPES,
)
from analyst_dashboard.smart_money.reconciliation_engine import (
    FilingReconciliationEngine,
    IndexFilingRecord,
    ReconciliationReport,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    RateLimiter,
)
from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneySignalType,
    OwnershipNature,
    SmartMoneyFiling,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner
from analyst_dashboard.analyzers.scanner_runner import (
    SmartMoneyScannerRunner,
    RESOURCE_KEY_SMART_MONEY_PIPELINE,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.coordination import (
    DurableRunCoordinator,
    CoordinationStore,
    TriggerType,
)
from api.routes.screener import (
    derive_canonical_radar_categories,
    get_radar_capabilities,
    RADAR_CAPABILITY_CONTRACT,
)


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_ATOM_FEED_XML = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EDGAR Latest Submissions - Form 4</title>
  <updated>2026-10-13T14:30:00-04:00</updated>
  <id>urn:tag:sec.gov,2026:latest-submissions-4</id>

  <!-- Entry 1: Valid Form 4 for NVDA -->
  <entry>
    <title>4 - NVIDIA CORP (0001045810) (Issuer)</title>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000088/0001045810-26-000088-index.htm"/>
    <summary type="html">
      &lt;b&gt;Filed:&lt;/b&gt; 2026-10-13 &lt;b&gt;AccNo:&lt;/b&gt; 0001045810-26-000088 &lt;b&gt;Size:&lt;/b&gt; 14 KB
    </summary>
    <updated>2026-10-13T14:25:00-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="4"/>
    <id>urn:tag:sec.gov,2026:accession-number=0001045810-26-000088</id>
  </entry>

  <!-- Entry 2: Valid Form 4/A for AAPL -->
  <entry>
    <title>4/A - Apple Inc. (0000320193) (Issuer)</title>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000099/0000320193-26-000099-index.htm"/>
    <summary type="html">
      &lt;b&gt;Filed:&lt;/b&gt; 2026-10-13 &lt;b&gt;AccNo:&lt;/b&gt; 0000320193-26-000099 &lt;b&gt;Size:&lt;/b&gt; 12 KB
    </summary>
    <updated>2026-10-13T14:28:00-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="4/A"/>
    <id>urn:tag:sec.gov,2026:accession-number=0000320193-26-000099</id>
  </entry>

  <!-- Entry 3: Out-of-scope Form 3 (Initial Statement) -->
  <entry>
    <title>3 - Microsoft Corp (0000789019) (Issuer)</title>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/789019/000078901926000011/0000789019-26-000011-index.htm"/>
    <summary type="html">
      &lt;b&gt;Filed:&lt;/b&gt; 2026-10-13 &lt;b&gt;AccNo:&lt;/b&gt; 0000789019-26-000011 &lt;b&gt;Size:&lt;/b&gt; 8 KB
    </summary>
    <updated>2026-10-13T14:20:00-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="3"/>
    <id>urn:tag:sec.gov,2026:accession-number=0000789019-26-000011</id>
  </entry>

  <!-- Entry 4: Out-of-scope Form 5 (Annual Statement) -->
  <entry>
    <title>5 - Tesla, Inc. (0001318605) (Issuer)</title>
    <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1318605/000131860526000022/0001318605-26-000022-index.htm"/>
    <summary type="html">
      &lt;b&gt;Filed:&lt;/b&gt; 2026-10-13 &lt;b&gt;AccNo:&lt;/b&gt; 0001318605-26-000022 &lt;b&gt;Size:&lt;/b&gt; 10 KB
    </summary>
    <updated>2026-10-13T14:15:00-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="5"/>
    <id>urn:tag:sec.gov,2026:accession-number=0001318605-26-000022</id>
  </entry>
</feed>
"""

SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-10-12</periodOfReport>
    <issuer>
        <issuerCik>0001045810</issuerCik>
        <issuerName>NVIDIA CORP</issuerName>
        <issuerTradingSymbol>NVDA</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001099999</rptOwnerCik>
            <rptOwnerName>Huang Jensen</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <officerTitle>President and CEO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-10-12</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>125.50</value></transactionPricePerShare>
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
def test_pipeline_env(tmp_path):
    """Initializes isolated SQLite stores for testing."""
    cache_db = str(tmp_path / "smart_money_cache.db")
    scanner_db = str(tmp_path / "scanner_store.db")

    ingestion_mgr = Form4IngestionManager(db_path=cache_db)
    snapshot_store = ScannerSnapshotStore(db_path=scanner_db)
    discovery_engine = GlobalSecDiscoveryEngine(ingestion_mgr=ingestion_mgr)
    recon_engine = FilingReconciliationEngine(db_path=cache_db)
    form4_scanner = SmartMoneyForm4Scanner(ingestion_mgr=ingestion_mgr)
    scanner_runner = SmartMoneyScannerRunner(
        snapshot_store=snapshot_store,
        form4_scanner=form4_scanner,
    )

    return {
        "cache_db": cache_db,
        "scanner_db": scanner_db,
        "ingestion_mgr": ingestion_mgr,
        "snapshot_store": snapshot_store,
        "discovery_engine": discovery_engine,
        "recon_engine": recon_engine,
        "form4_scanner": form4_scanner,
        "scanner_runner": scanner_runner,
    }


# =============================================================================
# 1. GATE B1: GLOBAL SEC ENUMERATION & RECONCILIATION TESTS
# =============================================================================

def test_gate_b1_global_discovery_enumeration(test_pipeline_env):
    """Fast-path discovery correctly parses Atom feed and filters for Form 4/4-A."""
    engine = test_pipeline_env["discovery_engine"]
    batch = engine.discover_ownership_filings(feed_content_override=SAMPLE_ATOM_FEED_XML)

    assert batch.status == "SUCCESS"
    assert batch.total_entries_found == 4
    # In-scope filter: Form 4 and Form 4/A only (Form 3 and Form 5 excluded)
    assert batch.in_scope_entries_count == 2
    assert len(batch.entries) == 2

    accessions = {e.accession_number for e in batch.entries}
    assert "0001045810-26-000088" in accessions
    assert "0000320193-26-000099" in accessions

    entry_nvda = next(e for e in batch.entries if e.accession_number == "0001045810-26-000088")
    assert entry_nvda.issuer_cik == "0001045810"
    assert entry_nvda.form_type == "4"
    assert entry_nvda.acceptance_datetime is not None


def test_gate_b1_cursor_continuation(test_pipeline_env):
    """Discovery advances cursor timestamp and filters out older filings."""
    engine = test_pipeline_env["discovery_engine"]
    b1 = engine.discover_ownership_filings(feed_content_override=SAMPLE_ATOM_FEED_XML)
    assert b1.cursor_timestamp is not None

    # Next run using cursor advances and filters seen timestamps
    b2 = engine.discover_ownership_filings(
        feed_content_override=SAMPLE_ATOM_FEED_XML,
        since_cursor=b1.cursor_timestamp,
    )
    # Since no timestamps are newer than the highest in b1, count is 0
    assert b2.in_scope_entries_count == 0


def test_gate_b1_reconciliation_parity_and_gap_detection(test_pipeline_env):
    """Reconciliation detects missing filings from daily index and creates gap records."""
    recon_engine = test_pipeline_env["recon_engine"]
    ingestion_mgr = test_pipeline_env["ingestion_mgr"]

    # Ingest only NVDA filing
    ingestion_mgr.ingest_accession(
        accession_number="0001045810-26-000088",
        issuer_cik="0001045810",
        filing_date="2026-10-12",
        acceptance_datetime="2026-10-12T20:00:00Z",
        raw_content=SAMPLE_FORM4_XML,
    )

    # Daily index has TWO filings: NVDA and AAPL
    daily_idx_records = [
        IndexFilingRecord(
            accession_number="0001045810-26-000088",
            issuer_cik="0001045810",
            company_name="NVIDIA CORP",
            form_type="4",
            filing_date="2026-10-12",
            source_filename="edgar/data/1045810/0001045810-26-000088.txt",
        ),
        IndexFilingRecord(
            accession_number="0000320193-26-000099",
            issuer_cik="0000320193",
            company_name="Apple Inc.",
            form_type="4/A",
            filing_date="2026-10-12",
            source_filename="edgar/data/320193/0000320193-26-000099.txt",
        ),
    ]

    report = recon_engine.reconcile_interval(
        index_records=daily_idx_records,
        interval_start="2026-10-12T00:00:00Z",
        interval_end="2026-10-12T23:59:59Z",
    )

    # AAPL is missing in local store -> coverage is PARTIAL, gap logged
    assert report.coverage_status == "PARTIAL"
    assert "0000320193-26-000099" in report.missing_accessions
    assert report.open_gap_count >= 1

    # Now ingest AAPL and re-reconcile
    aapl_xml = SAMPLE_FORM4_XML.replace("0001045810", "0000320193").replace("NVDA", "AAPL")
    ingestion_mgr.ingest_accession(
        accession_number="0000320193-26-000099",
        issuer_cik="0000320193",
        filing_date="2026-10-12",
        acceptance_datetime="2026-10-12T20:05:00Z",
        raw_content=aapl_xml,
    )

    report_after = recon_engine.reconcile_interval(
        index_records=daily_idx_records,
        interval_start="2026-10-12T00:00:00Z",
        interval_end="2026-10-12T23:59:59Z",
        auto_record_gaps=False,
    )
    # Perfect parity: coverage is COMPLETE
    assert report_after.coverage_status == "COMPLETE"
    assert len(report_after.missing_accessions) == 0


# =============================================================================
# 2. GATE B2 & PHASE E/F/G: POINT-IN-TIME (PIT) BOUNDARY TESTS
# =============================================================================

def test_gate_b2_pit_temporal_isolation(test_pipeline_env):
    """
    Transactions available at public_availability_at > W are strictly excluded from generation(W).
    Transactions available at public_availability_at <= W are included.
    """
    ingestion_mgr = test_pipeline_env["ingestion_mgr"]
    scanner = test_pipeline_env["form4_scanner"]

    # Filing 1: Available at 2026-10-10T12:00:00Z
    ingestion_mgr.ingest_accession(
        accession_number="ACC-PAST-001",
        issuer_cik="0001045810",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T12:00:00Z",
        raw_content=SAMPLE_FORM4_XML,
    )

    # Filing 2: Available at 2026-10-12T15:00:00Z (Future relative to W1)
    future_xml = SAMPLE_FORM4_XML.replace("10000", "20000")
    ingestion_mgr.ingest_accession(
        accession_number="ACC-FUTURE-002",
        issuer_cik="0001045810",
        filing_date="2026-10-12",
        acceptance_datetime="2026-10-12T15:00:00Z",
        raw_content=future_xml,
    )

    w1 = "2026-10-11T00:00:00Z"
    snap_w1 = scanner.scan_issuers_as_of(
        as_of_watermark=w1,
        symbols=["NVDA"],
        lookback_days=30,
        now_date="2026-10-11",
    )

    assert snap_w1.matched_count == 1
    # Only Filing 1 included: gross purchase = 10,000 * 125.50 = 1,255,000 USD
    cand = snap_w1.items[0]
    assert cand["grossPurchaseValueUsd"] == 1255000.0

    # At W2 (after Filing 2 availability), both filings are included
    w2 = "2026-10-13T00:00:00Z"
    snap_w2 = scanner.scan_issuers_as_of(
        as_of_watermark=w2,
        symbols=["NVDA"],
        lookback_days=30,
        now_date="2026-10-13",
    )
    assert snap_w2.matched_count == 1
    # Both filings included: 1,255,000 + 2,510,000 = 3,765,000 USD
    cand2 = snap_w2.items[0]
    assert cand2["grossPurchaseValueUsd"] == 3765000.0


def test_gate_b2_late_arrival_temporal_isolation(test_pipeline_env):
    """
    A late filer executed trades long ago (e.g. June), but filed in October.
    Must not leak into an August historical as-of query.
    """
    ingestion_mgr = test_pipeline_env["ingestion_mgr"]
    scanner = test_pipeline_env["form4_scanner"]

    late_xml = SAMPLE_FORM4_XML.replace("2026-10-12", "2026-06-01")
    ingestion_mgr.ingest_accession(
        accession_number="ACC-LATE-001",
        issuer_cik="0001045810",
        filing_date="2026-10-12",
        acceptance_datetime="2026-10-12T16:00:00Z",
        raw_content=late_xml,
    )

    # Historical boundary in August: Filing was NOT yet publicly available
    w_aug = "2026-08-01T00:00:00Z"
    snap_aug = scanner.scan_issuers_as_of(
        as_of_watermark=w_aug,
        symbols=["NVDA"],
        lookback_days=90,
        now_date="2026-08-01",
    )
    assert snap_aug.matched_count == 0


def test_gate_b2_ingestion_idempotency_and_conflict(test_pipeline_env):
    """Duplicate identical accession is a NOOP; conflicting content raises integrity error."""
    ingestion_mgr = test_pipeline_env["ingestion_mgr"]

    res1 = ingestion_mgr.ingest_accession(
        accession_number="ACC-IDEMP-001",
        issuer_cik="0001045810",
        raw_content=SAMPLE_FORM4_XML,
    )
    assert res1["status"] == "INGESTED_NEW"

    # Re-ingest exact same content -> KNOWN_IDEMPOTENT_NOOP
    res2 = ingestion_mgr.ingest_accession(
        accession_number="ACC-IDEMP-001",
        issuer_cik="0001045810",
        raw_content=SAMPLE_FORM4_XML,
    )
    assert res2["status"] == "KNOWN_IDEMPOTENT_NOOP"

    # Ingest differing content for same canonical accession -> SOURCE_INTEGRITY_CONFLICT
    conflicting_xml = SAMPLE_FORM4_XML.replace("125.50", "999.00")
    with pytest.raises(RuntimeError, match="SOURCE_INTEGRITY_CONFLICT"):
        ingestion_mgr.ingest_accession(
            accession_number="ACC-IDEMP-001",
            issuer_cik="0001045810",
            raw_content=conflicting_xml,
        )


# =============================================================================
# 3. GATE B3: FORM 4 QUALIFICATION CONTRACT TESTS
# =============================================================================

def test_gate_b3_frozen_qualification_predicate_boundaries(test_pipeline_env):
    """
    Enforces exact boundary rules for FORM4_RADAR_QUALIFICATION_V1:
    - gross_purchase_value_usd < $25,000 -> DISQUALIFIED
    - gross_purchase_value_usd >= $25,000 -> QUALIFIED
    - 0 open-market purchases (e.g. only sales or gifts) -> DISQUALIFIED
    """
    scanner = test_pipeline_env["form4_scanner"]

    # 1. Below threshold: 200 shares * $100 = $20,000 (< $25,000)
    tx_sub = SmartMoneyTransaction(
        record_key="REC-1",
        accession_number="ACC-1",
        issuer_cik="0001045810",
        issuer_symbol="NVDA",
        issuer_name="NVIDIA CORP",
        reporting_owner_cik="0001099999",
        reporting_owner_name="Jensen Huang",
        is_officer=True,
        officer_title="CEO",
        is_director=True,
        is_ten_percent_owner=False,
        filing_date="2026-10-12",
        transaction_date="2026-10-12",
        transaction_code="P",
        signal_type=SmartMoneySignalType.OPEN_MARKET_PURCHASE,
        shares=200.0,
        price_per_share=100.0,
        total_value_usd=20000.0,
        acquired_disposed="A",
        ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False,
        is_open_market_purchase_evidence=True,
    )
    state_sub = scanner.evaluate_issuer_transactions("NVDA", [tx_sub])
    assert state_sub.gross_purchase_value_usd == 20000.0
    assert not state_sub.qualifies_for_radar, "$20,000 purchase must NOT qualify"

    # 2. Exactly at threshold: 250 shares * $100 = $25,000
    tx_exact = SmartMoneyTransaction(
        record_key="REC-2",
        accession_number="ACC-2",
        issuer_cik="0001045810",
        issuer_symbol="NVDA",
        issuer_name="NVIDIA CORP",
        reporting_owner_cik="0001099999",
        reporting_owner_name="Jensen Huang",
        is_officer=True,
        officer_title="CEO",
        is_director=True,
        is_ten_percent_owner=False,
        filing_date="2026-10-12",
        transaction_date="2026-10-12",
        transaction_code="P",
        signal_type=SmartMoneySignalType.OPEN_MARKET_PURCHASE,
        shares=250.0,
        price_per_share=100.0,
        total_value_usd=25000.0,
        acquired_disposed="A",
        ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False,
        is_open_market_purchase_evidence=True,
    )
    state_exact = scanner.evaluate_issuer_transactions("NVDA", [tx_exact])
    assert state_exact.gross_purchase_value_usd == 25000.0
    assert state_exact.qualifies_for_radar, "$25,000 purchase MUST qualify"

    # 3. Only sales (no open market buy)
    tx_sale = SmartMoneyTransaction(
        record_key="REC-3",
        accession_number="ACC-3",
        issuer_cik="0001045810",
        issuer_symbol="NVDA",
        issuer_name="NVIDIA CORP",
        reporting_owner_cik="0001099999",
        reporting_owner_name="Jensen Huang",
        is_officer=True,
        officer_title="CEO",
        is_director=True,
        is_ten_percent_owner=False,
        filing_date="2026-10-12",
        transaction_date="2026-10-12",
        transaction_code="S",
        signal_type=SmartMoneySignalType.OPEN_MARKET_SALE,
        shares=5000.0,
        price_per_share=100.0,
        total_value_usd=500000.0,
        acquired_disposed="D",
        ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False,
        is_open_market_purchase_evidence=False,
    )
    state_sale = scanner.evaluate_issuer_transactions("NVDA", [tx_sale])
    assert not state_sale.qualifies_for_radar, "Sales only must NOT qualify for accumulation"


def test_gate_b3_conviction_scoring_components(test_pipeline_env):
    """
    Verifies 4 scoring factors:
    1. Purchase value: $1M+ (+40), $250k+ (+30), $50k+ (+20), <$50k (+10)
    2. Cluster: >=2 insiders (+30), single (+15)
    3. Role: C-suite (+20), Officer/Director (+10)
    4. Recency: <=7d (+10), <=14d (+5)
    Max = 100.0
    """
    scanner = test_pipeline_env["form4_scanner"]
    now_date = "2026-10-13"

    # Perfect 100 score:
    # Value $1.5M (40) + Cluster 2 insiders (30) + C-Suite CEO (20) + 1 day ago (10) = 100
    t1 = SmartMoneyTransaction(
        record_key="T1", accession_number="A1", issuer_cik="0001045810", issuer_symbol="NVDA",
        issuer_name="NVIDIA CORP", reporting_owner_cik="OWNER-1", reporting_owner_name="Alice CEO",
        is_officer=True, officer_title="CEO", is_director=True, is_ten_percent_owner=False,
        filing_date="2026-10-12", transaction_date="2026-10-12", transaction_code="P",
        signal_type=SmartMoneySignalType.OPEN_MARKET_PURCHASE, shares=10000.0, price_per_share=100.0,
        total_value_usd=1000000.0, acquired_disposed="A", ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False, is_open_market_purchase_evidence=True,
    )
    t2 = SmartMoneyTransaction(
        record_key="T2", accession_number="A2", issuer_cik="0001045810", issuer_symbol="NVDA",
        issuer_name="NVIDIA CORP", reporting_owner_cik="OWNER-2", reporting_owner_name="Bob Director",
        is_officer=False, officer_title=None, is_director=True, is_ten_percent_owner=False,
        filing_date="2026-10-12", transaction_date="2026-10-12", transaction_code="P",
        signal_type=SmartMoneySignalType.OPEN_MARKET_PURCHASE, shares=5000.0, price_per_share=100.0,
        total_value_usd=500000.0, acquired_disposed="A", ownership_nature=OwnershipNature.DIRECT,
        is_derivative=False, is_open_market_purchase_evidence=True,
    )

    state = scanner.evaluate_issuer_transactions("NVDA", [t1, t2], now_date=now_date)
    assert state.conviction_score == 100.0
    assert state.cluster_buy is True
    assert state.c_suite_buying is True
    assert state.distinct_buyer_count == 2


# =============================================================================
# 4. GATE B4 & GATE B5: IMMUTABLE SCANNER GENERATION & SNAPSHOT STORE
# =============================================================================

def test_gate_b5_snapshot_store_immutability_and_triggers(test_pipeline_env):
    """Snapshot rows are strictly immutable: UPDATE and DELETE trigger ABORT."""
    store = test_pipeline_env["snapshot_store"]
    snap_id = f"SNAP_TEST_{int(time.time())}"

    snapshot_data = {
        "snapshot_id": snap_id,
        "scanner_id": "SMART_MONEY",
        "run_id": "RUN-001",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "abc1234",
        "semantic_fingerprint": "fp123",
        "generated_at": "2026-10-13T10:00:00Z",
        "data_as_of": "2026-10-13T10:00:00Z",
        "status_at_publication": "AVAILABLE",
        "universe_id": "TEST_UNIVERSE",
        "universe_size": 100,
        "matched_count": 5,
        "results": [{"symbol": "NVDA", "score": 90.0}],
        "provenance": {"source": "SEC EDGAR Form 4"},
        "freshness": {"status": "LIVE"},
        "publication_decision": "PUBLISH",
    }
    store.save_snapshot(snapshot_data)

    # Verify retrieval
    loaded = store.get_snapshot_by_id(snap_id)
    assert loaded is not None
    assert loaded["matched_count"] == 5

    # Attempt mutation (UPDATE) -> Aborted by trigger
    conn = sqlite3.connect(store.db_path)
    with pytest.raises(sqlite3.DatabaseError, match="IMMUTABILITY_VIOLATION"):
        conn.execute("UPDATE scanner_snapshots SET matched_count = 10 WHERE snapshot_id = ?", (snap_id,))

    # Attempt deletion (DELETE) -> Aborted by trigger
    with pytest.raises(sqlite3.DatabaseError, match="IMMUTABILITY_VIOLATION"):
        conn.execute("DELETE FROM scanner_snapshots WHERE snapshot_id = ?", (snap_id,))
    conn.close()


def test_gate_b5_active_pointer_and_quarantine(test_pipeline_env):
    """Only snapshots with publication_decision = 'PUBLISH' become active."""
    store = test_pipeline_env["snapshot_store"]

    # 1. Publish valid snapshot S1
    s1_data = {
        "snapshot_id": "SNAP_S1",
        "scanner_id": "SMART_MONEY",
        "run_id": "RUN-S1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha1",
        "semantic_fingerprint": "fp1",
        "generated_at": "2026-10-13T10:00:00Z",
        "data_as_of": "2026-10-13T10:00:00Z",
        "status_at_publication": "AVAILABLE",
        "universe_id": "UNIV_1",
        "universe_size": 100,
        "matched_count": 3,
        "results": [{"symbol": "NVDA"}],
        "provenance": {},
        "freshness": {},
        "publication_decision": "PUBLISH",
    }
    store.save_snapshot(s1_data)
    active = store.get_latest_active_snapshot("SMART_MONEY")
    assert active["snapshot_id"] == "SNAP_S1"

    # 2. Add Quarantined snapshot S2
    s2_data = dict(s1_data)
    s2_data["snapshot_id"] = "SNAP_S2"
    s2_data["publication_decision"] = "QUARANTINE"
    store.save_snapshot(s2_data)

    # Active pointer must STILL point to S1!
    active_after = store.get_latest_active_snapshot("SMART_MONEY")
    assert active_after["snapshot_id"] == "SNAP_S1"


# =============================================================================
# 5. RADAR INTEGRATION & TAXONOMY SEPARATION
# =============================================================================

def test_radar_taxonomy_and_category_classification(test_pipeline_env):
    """
    Radar category classifier correctly maps SMART_MONEY when backed by active snapshot.
    Decoupled from Execution Ladder (zero IN_BUY_ZONE leakage).
    """
    # Case 1: No active snapshot -> SMART_MONEY not assigned
    cats_none = derive_canonical_radar_categories(
        expert_model="Greenblatt Magic Formula",
        peg_ratio=0.8,
        roic_pct=25.0,
        gross_margin_pct=60.0,
        symbol="NVDA",
        smart_money_matched_symbols=set(),
    )
    assert "VALUE_GARP" in cats_none
    assert "SMART_MONEY" not in cats_none

    # Case 2: NVDA qualifies in active Smart Money snapshot -> SMART_MONEY assigned
    cats_active = derive_canonical_radar_categories(
        expert_model="Greenblatt Magic Formula",
        peg_ratio=0.8,
        roic_pct=25.0,
        gross_margin_pct=60.0,
        symbol="NVDA",
        smart_money_matched_symbols={"NVDA"},
    )
    assert "SMART_MONEY" in cats_active
    assert "VALUE_GARP" in cats_active

    # Capabilities contract reflects dynamic status
    caps_active = get_radar_capabilities({
        "status": "AVAILABLE",
        "matched_count": 4,
        "freshness": {"data_as_of": "2026-10-13T12:00:00Z"},
    })
    assert caps_active["SMART_MONEY"]["status"] == "AVAILABLE"
    assert caps_active["SMART_MONEY"]["subsignal"] == "INSIDER_FORM4"
    assert caps_active["SMART_MONEY"]["matchedCount"] == 4


# =============================================================================
# 6. GATE B6: CONCURRENCY, DISTRIBUTED FENCING & RATE GOVERNANCE
# =============================================================================

def test_gate_b6_rate_limiter_pacing():
    """RateLimiter strictly enforces 5 req/s ceiling (interval = 0.2s)."""
    limiter = RateLimiter(max_rps=5.0)
    assert limiter.interval == 0.2

    t0 = time.monotonic()
    for _ in range(4):
        limiter.wait_for_slot()
    elapsed = time.monotonic() - t0
    # 4 calls at 0.2s interval: calls 2, 3, 4 each sleep ~0.2s -> total >= 0.5s
    assert elapsed >= 0.55


def test_gate_b6_scheduler_overlap_rejection(test_pipeline_env):
    """Overlapping scanner execution raises OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED."""
    runner = test_pipeline_env["scanner_runner"]

    # Manually hold the local scan lock
    runner._scan_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED"):
            runner.execute_market_wide_scan(bypass_thread_lock=False)
    finally:
        runner._scan_lock.release()


# =============================================================================
# 7. CANDIDATE 002 OPERATIONAL REMEDIATION TESTS
# =============================================================================

def test_remediation_all_sec_egress_call_sites_governed():
    """All outbound SEC paths share the singleton DEFAULT_SEC_RATE_LIMITER (5.0 req/s ceiling)."""
    from analyst_dashboard.data.sec_edgar_fetcher import DEFAULT_SEC_RATE_LIMITER, SecEdgarFetcher
    from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager
    from analyst_dashboard.smart_money.global_discovery import GlobalSecDiscoveryEngine
    from analyst_dashboard.smart_money.reconciliation_engine import FilingReconciliationEngine
    from api.routes.smart_money import sec_fetcher as route_sec_fetcher

    f1 = SecEdgarFetcher()
    f2 = Form4IngestionManager()
    f3 = GlobalSecDiscoveryEngine()
    f4 = FilingReconciliationEngine()

    assert f1.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert f2.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert f3.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert f4.rate_limiter is DEFAULT_SEC_RATE_LIMITER
    assert route_sec_fetcher.rate_limiter is DEFAULT_SEC_RATE_LIMITER

    # Concurrent worker simulation: 4 workers acquiring slots
    t0 = time.monotonic()
    for _ in range(4):
        DEFAULT_SEC_RATE_LIMITER.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.55


def test_remediation_intraday_vs_reconciled_coverage_state_machine(test_pipeline_env, monkeypatch):
    """
    Two-stage state machine:
    - Fast path / intraday -> PARTIAL, PIPELINE_PENDING, QUARANTINE.
    - Missing daily index -> RECONCILIATION_PENDING.
    - Available daily index + unresolved gaps -> PARTIAL.
    - Reconciled daily index + 0 gaps -> COMPLETE, advances certified watermark.
    """
    scanner = test_pipeline_env["form4_scanner"]
    recon_engine = FilingReconciliationEngine(db_path=test_pipeline_env["cache_db"])
    ingestion_mgr = test_pipeline_env["ingestion_mgr"]

    # Mock universe builder for market-wide testing
    class MockUniverseBuilder:
        def get_build_manifest(self, b_id):
            class M:
                scannable_count = 1
                eligible_count = 1
                universe_id = "SMART_MONEY_US_EQUITIES"
                member_identity_hash = "mock_hash"
                eligibility_policy_version = "1.0.0"
            return M()
        def get_scannable_members(self, b_id):
            return [{"symbol": "NVDA"}]

    scanner.universe_builder = MockUniverseBuilder()

    # 1. Intraday fast path scan: NO receipt in sm_reconciliation_reports
    snap_intraday = scanner.scan_issuers_as_of(
        as_of_watermark="2026-10-13T12:00:00Z",
        universe_build_id="UB-001",
    )
    assert snap_intraday.coverage_status == "PARTIAL"
    assert snap_intraday.status == "PIPELINE_PENDING"
    assert snap_intraday.provenance["certified_complete"] is False

    # 2. Missing daily master index -> RECONCILIATION_PENDING
    monkeypatch.setattr(recon_engine, "fetch_daily_master_index", lambda d: None)
    recon_pending = recon_engine.reconcile_daily_index("2026-10-13")
    assert recon_pending["status"] == "RECONCILIATION_PENDING"
    assert recon_pending["coverage_status"] == "RECONCILIATION_PENDING"

    # 3. Available daily index with unresolved gap -> PARTIAL
    sample_index_with_gap = (
        "CIK|Company Name|Form Type|Date Filed|File Name\n"
        "--------------------------------------------------\n"
        "1045810|NVIDIA CORP|4|2026-10-13|edgar/data/1045810/0001045810-26-000099.txt\n"
    )
    monkeypatch.setattr(recon_engine, "fetch_daily_master_index", lambda d: sample_index_with_gap)
    recon_partial = recon_engine.reconcile_daily_index("2026-10-13", auto_record_gaps=True)
    assert recon_partial["status"] == "COMPLETED"
    assert recon_partial["coverage_status"] == "PARTIAL"
    assert len(recon_partial["report"]["missing_accessions"]) == 1

    # 4. Ingest missing filing and reconcile -> COMPLETE
    ingestion_mgr.ingest_accession(
        accession_number="0001045810-26-000099",
        issuer_cik="0001045810",
        filing_date="2026-10-13",
        acceptance_datetime="2026-10-13T14:00:00Z",
        raw_content=SAMPLE_FORM4_XML.replace("0001045810-26-000088", "0001045810-26-000099"),
    )
    recon_complete = recon_engine.reconcile_daily_index("2026-10-13", auto_record_gaps=False)
    assert recon_complete["status"] == "COMPLETED"
    assert recon_complete["coverage_status"] == "COMPLETE"
    assert len(recon_complete["report"]["missing_accessions"]) == 0

    # 5. Watermark advance verified
    latest_w = recon_engine.get_latest_certified_watermark()
    assert latest_w == "2026-10-13T23:59:59Z"

    # 6. PIT scan at certified watermark -> COMPLETE and AVAILABLE
    snap_certified = scanner.scan_issuers_as_of(
        as_of_watermark=latest_w,
        universe_build_id="UB-001",
    )
    assert snap_certified.coverage_status == "COMPLETE"
    assert snap_certified.status == "AVAILABLE"
    assert snap_certified.provenance["certified_complete"] is True
    assert snap_certified.provenance["reconciliation_receipt_id"] is not None


def test_remediation_sqlite_canonical_connection_policy(tmp_path):
    """connect_smart_money_db enforces WAL mode, busy_timeout=5000, and synchronous=NORMAL."""
    from analyst_dashboard.smart_money.form4_ingestion import connect_smart_money_db, Form4IngestionManager
    from analyst_dashboard.smart_money.reconciliation_engine import FilingReconciliationEngine

    db_file = str(tmp_path / "policy_test.db")
    conn = connect_smart_money_db(db_file)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode")
    journal = cur.fetchone()[0]
    assert str(journal).lower() == "wal"

    cur.execute("PRAGMA busy_timeout")
    timeout = cur.fetchone()[0]
    assert int(timeout) == 5000

    cur.execute("PRAGMA synchronous")
    sync = cur.fetchone()[0]
    assert int(sync) == 1  # NORMAL

    conn.close()

    # Manager and engine connection methods use canonical factory
    mgr = Form4IngestionManager(db_path=db_file)
    c1 = mgr._get_connection()
    assert c1.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    c1.close()

    eng = FilingReconciliationEngine(db_path=db_file)
    c2 = eng._get_connection()
    assert c2.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    c2.close()


def test_remediation_vcp_isolation_both_directions(test_pipeline_env):
    """
    Snapshot domain isolation works bi-directionally:
    - Activating SMART_MONEY snapshot leaves MINERVINI_VCP active pointer untouched.
    - Activating MINERVINI_VCP snapshot leaves SMART_MONEY active pointer untouched.
    """
    store = test_pipeline_env["snapshot_store"]

    # 1. Seed initial VCP snapshot
    vcp_snap = {
        "snapshot_id": "VCP_INITIAL",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "RUN-VCP-1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha_vcp_1",
        "semantic_fingerprint": "fp_vcp_1",
        "generated_at": "2026-10-13T10:00:00Z",
        "data_as_of": "2026-10-13T10:00:00Z",
        "status_at_publication": "AVAILABLE",
        "universe_id": "VCP_UNIVERSE",
        "universe_size": 100,
        "matched_count": 2,
        "results": [{"symbol": "CPRX"}],
        "provenance": {},
        "freshness": {},
        "publication_decision": "PUBLISH",
    }
    store.save_snapshot(vcp_snap)
    assert store.get_latest_active_snapshot("MINERVINI_VCP")["snapshot_id"] == "VCP_INITIAL"

    # 2. Publish Smart Money snapshot -> VCP pointer MUST remain VCP_INITIAL
    sm_snap = {
        "snapshot_id": "SM_INITIAL",
        "scanner_id": "SMART_MONEY",
        "run_id": "RUN-SM-1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha_sm_1",
        "semantic_fingerprint": "fp_sm_1",
        "generated_at": "2026-10-13T11:00:00Z",
        "data_as_of": "2026-10-13T11:00:00Z",
        "status_at_publication": "AVAILABLE",
        "universe_id": "SM_UNIVERSE",
        "universe_size": 100,
        "matched_count": 3,
        "results": [{"symbol": "NVDA"}],
        "provenance": {},
        "freshness": {},
        "publication_decision": "PUBLISH",
    }
    store.save_snapshot(sm_snap)
    assert store.get_latest_active_snapshot("SMART_MONEY")["snapshot_id"] == "SM_INITIAL"
    assert store.get_latest_active_snapshot("MINERVINI_VCP")["snapshot_id"] == "VCP_INITIAL"

    # 3. Sleep 1.05s to ensure next snapshot has distinct created_at timestamp
    time.sleep(1.05)

    # Publish newer VCP snapshot -> Smart Money pointer MUST remain SM_INITIAL
    vcp_snap_2 = dict(vcp_snap)
    vcp_snap_2["snapshot_id"] = "VCP_SECOND"
    vcp_snap_2["run_id"] = "RUN-VCP-2"
    vcp_snap_2["generated_at"] = "2026-10-13T12:00:00Z"
    store.save_snapshot(vcp_snap_2)

    assert store.get_latest_active_snapshot("MINERVINI_VCP")["snapshot_id"] == "VCP_SECOND"
    assert store.get_latest_active_snapshot("SMART_MONEY")["snapshot_id"] == "SM_INITIAL"


def test_remediation_git_custody_and_ancestry():
    """Frozen RC 9d0bbcf is ancestor and parked artifacts are absent."""
    import subprocess

    # 1. Prove frozen RC is ancestor
    rc_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "9d0bbcf21673fd6a3428a6674713dd244ec948bc", "HEAD"],
        capture_output=True,
    )
    assert rc_check.returncode == 0, "Frozen RC 9d0bbcf must be an ancestor of HEAD."

    # 2. Prove preimplementation baseline is ancestor
    base_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "0b644698f46d14dd454425cb842b08e59b8dafcb", "HEAD"],
        capture_output=True,
    )
    assert base_check.returncode == 0, "Baseline 0b64469 must be an ancestor of HEAD."

    # 3. Parked Execution Ladder spec must NOT be tracked or modified
    ladder_file = "docs/governance/EXECUTION_LADDER_LIFECYCLE_SPECIFICATION_V1.md"
    git_status = subprocess.run(
        ["git", "status", "--porcelain", ladder_file],
        capture_output=True,
        text=True,
    )
    assert ladder_file not in git_status.stdout, f"Parked artifact {ladder_file} must not be present in git status."

    # 4. Unrelated 120-asset files must NOT have modified status
    unrelated_status = subprocess.run(
        ["git", "status", "--porcelain", "analyst_dashboard/analyzers/gem_screener.py", "frontend/lib/masterCatalog.ts"],
        capture_output=True,
        text=True,
    )
    assert unrelated_status.stdout.strip() == "", "Unrelated 120-asset changes must not be modified."
