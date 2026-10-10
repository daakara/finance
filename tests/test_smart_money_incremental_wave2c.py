"""
ARX Terminal — Smart Money Wave 2C Incremental Pipeline Test Suite.
Validates:
1. Three Distinct Progress Concepts (Max Observed vs Cursor vs Certified Watermark)
2. Partition Watermarks & Property Tests (>= 500 cases)
3. Gap Ledger, Gap Frontier Rule & Non-Skipping Invariants
4. Late Arrival Forensics & Watermark Integrity Breach
5. Form 4/A Amendment Lineage & Retroactive Non-Leakage
6. Incremental vs Full Rebuild Oracle Parity (Content Delta = 0)
7. Crash / Restart Resilience
8. Concurrency Safety & Lease Locking
9. Freshness Authority & Readiness Semantics
10. Schema 2.0.0 to Wave 2C Migration Ladder
"""

import os
import sys
import json
import sqlite3
import random
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager
from analyst_dashboard.smart_money.universe_builder import SmartMoneyUniverseBuilder
from analyst_dashboard.smart_money.incremental_models import (
    IncrementalRunStatus,
    GapType,
    GapStatus,
    LateArrivalEventType,
    FreshnessStatus,
    ReadinessStatus,
    IncrementalCycle,
    SourceGap,
    PartitionWatermark,
    WatermarkReceipt,
    LateArrivalEvent,
    FreshnessEvaluation,
)
from analyst_dashboard.smart_money.incremental_engine import SmartMoneyIncrementalEngine


SAMPLE_XML_TEMPLATE = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>{doc_type}</documentType>
    <periodOfReport>{period}</periodOfReport>
    <issuer>
        <issuerCik>{cik}</issuerCik>
        <issuerTradingSymbol>{symbol}</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001000000</rptOwnerCik>
            <rptOwnerName>Corporate Insider</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>{tx_date}</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>{shares}</value></transactionShares>
                <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
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
def temp_engine(tmp_path):
    """Provides a fresh isolated SmartMoneyIncrementalEngine in a temporary directory."""
    db_file = str(tmp_path / "test_incremental_cache.db")
    u_db_file = str(tmp_path / "test_universe.db")

    # Scaffolding mock scannable universe with 5 members for focused tests
    u_builder = SmartMoneyUniverseBuilder(db_path=u_db_file)
    mock_members = [
        {"cik": "0000320193", "symbol": "AAPL", "security_type": "COMMON_STOCK", "exchange": "NASDAQ", "listing_status": "ACTIVE", "classification_status": "VERIFIED"},
        {"cik": "0000789019", "symbol": "MSFT", "security_type": "COMMON_STOCK", "exchange": "NASDAQ", "listing_status": "ACTIVE", "classification_status": "VERIFIED"},
        {"cik": "0001045810", "symbol": "NVDA", "security_type": "COMMON_STOCK", "exchange": "NASDAQ", "listing_status": "ACTIVE", "classification_status": "VERIFIED"},
        {"cik": "0001652044", "symbol": "GOOGL", "security_type": "COMMON_STOCK", "exchange": "NASDAQ", "listing_status": "ACTIVE", "classification_status": "VERIFIED"},
        {"cik": "0001018724", "symbol": "AMZN", "security_type": "COMMON_STOCK", "exchange": "NASDAQ", "listing_status": "ACTIVE", "classification_status": "VERIFIED"},
    ]
    u_manifest, _ = u_builder.build_universe(
        raw_records=mock_members,
        source_authority="MOCK_TEST_UNIVERSE",
        source_snapshot_id="MOCK_SNAP_001",
    )

    engine = SmartMoneyIncrementalEngine(
        db_path=db_file,
        universe_db_path=u_db_file,
        universe_builder=u_builder,
    )
    # Initialize partition state with baseline 2026-10-10T01:02:33Z
    engine.init_partitions_from_universe(
        universe_build_id=u_manifest.universe_build_id,
        default_baseline_timestamp="2026-10-10T01:02:33.000Z",
    )
    return engine, u_manifest.universe_build_id


def test_three_progress_concepts(temp_engine):
    """Verify that Max Observed, Processing Cursor, and Certified Watermark are decoupled."""
    engine, u_build_id = temp_engine

    # Initial baseline state
    max_obs = engine.get_max_observed_source_time()
    cursor = engine.get_global_processing_cursor()
    watermark = engine.get_global_certified_completeness_watermark()

    assert max_obs == "2026-10-10T01:02:33.000Z"
    assert cursor == "2026-10-10T01:02:33.000Z"
    assert watermark == "2026-10-10T01:02:33.000Z"

    # Advance processing cursor of AAPL without advancing watermark
    engine.update_partition_poll_cursor("0000320193", "2026-10-10T08:00:00Z", poll_succeeded=True)
    assert engine.get_global_processing_cursor() == "2026-10-10T01:02:33.000Z"  # Min across all partitions remains baseline

    # Ingest a new filing for AAPL with acceptance at 2026-10-10T06:00:00Z
    raw_xml = SAMPLE_XML_TEMPLATE.format(
        doc_type="4", period="2026-10-10", cik="0000320193", symbol="AAPL",
        tx_date="2026-10-10", shares=500, price=220.0
    )
    engine.ingest_filing_incrementally(
        accession_number="0000320193-26-000001",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T06:00:00Z",
        document_type="4",
        raw_xml=raw_xml,
    )

    # Max observed is now 06:00:00Z, but certified watermark is STILL 01:02:33Z!
    assert engine.get_max_observed_source_time() == "2026-10-10T06:00:00Z"
    assert engine.get_global_certified_completeness_watermark() == "2026-10-10T01:02:33.000Z"
    assert engine.get_max_observed_source_time() != engine.get_global_certified_completeness_watermark()


def test_gap_frontier_rule_blocks_watermark(temp_engine):
    """Verify that earliest open gap strictly blocks candidate watermark advancement."""
    engine, u_build_id = temp_engine

    cycle = engine.plan_cycle(
        universe_build_id=u_build_id,
        window_end="2026-10-10T12:00:00Z",
    )
    assert engine.acquire_cycle_lease(cycle, "DELIV-1", "EXEC-1")

    # Set all partitions processing cursor to 12:00:00Z
    for cik in cycle.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T12:00:00Z", poll_succeeded=True)

    # Candidate before gap would be 12:00:00Z
    cand = engine.compute_candidate_watermark(u_build_id)
    assert cand == "2026-10-10T12:00:00Z"

    # Now create an open FETCH_FAILURE gap for MSFT at 2026-10-10T03:30:00Z
    gap = engine.register_gap(
        cycle_id=cycle.get_cycle_id(),
        issuer_cik="0000789019",
        universe_build_id=u_build_id,
        gap_type=GapType.FETCH_FAILURE,
        window_start="2026-10-10T03:30:00Z",
        window_end="2026-10-10T12:00:00Z",
        accession_number="0000789019-26-000099",
        source_event_time="2026-10-10T03:30:00Z",
    )

    # Candidate watermark MUST now be bounded at 03:30:00Z!
    cand_blocked = engine.compute_candidate_watermark(u_build_id)
    assert cand_blocked == "2026-10-10T03:30:00Z"

    # Attempting to certify 12:00:00Z must fail with RuntimeError
    with pytest.raises(RuntimeError, match="BLOCKING_GAPS_EXIST"):
        engine.certify_watermark_advancement(cycle, "2026-10-10T12:00:00Z")

    # Later filings can still be ingested while gap is open
    raw_xml_nvda = SAMPLE_XML_TEMPLATE.format(
        doc_type="4", period="2026-10-10", cik="0001045810", symbol="NVDA",
        tx_date="2026-10-10", shares=1000, price=130.0
    )
    res = engine.ingest_filing_incrementally(
        accession_number="0001045810-26-000055",
        issuer_cik="0001045810",
        symbol="NVDA",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T09:00:00Z",
        document_type="4",
        raw_xml=raw_xml_nvda,
    )
    assert res["status"] == "INGESTED_NEW"

    # Resolving the gap allows advancement
    assert engine.resolve_gap(gap.gap_id, "MANUAL_REPLAY", "Successfully fetched and parsed on retry")
    cand_freed = engine.compute_candidate_watermark(u_build_id)
    assert cand_freed == "2026-10-10T12:00:00Z"

    # Advancement now succeeds atomically
    receipt = engine.certify_watermark_advancement(cycle, "2026-10-10T12:00:00Z")
    assert receipt is not None
    assert receipt.new_certified_watermark == "2026-10-10T12:00:00Z"
    assert engine.get_global_certified_completeness_watermark() == "2026-10-10T12:00:00Z"


def test_late_arrival_integrity_breach(temp_engine):
    """Verify that a late filing arriving before certified watermark creates an integrity breach and freezes advancement."""
    engine, u_build_id = temp_engine

    # Certified watermark is at 2026-10-10T01:02:33.000Z
    # Ingest a filing with acceptance BEFORE certified watermark: 2026-10-09T18:00:00Z
    raw_xml = SAMPLE_XML_TEMPLATE.format(
        doc_type="4", period="2026-10-09", cik="0000320193", symbol="AAPL",
        tx_date="2026-10-09", shares=200, price=215.0
    )
    res = engine.ingest_filing_incrementally(
        accession_number="0000320193-26-LATE01",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-09",
        acceptance_datetime="2026-10-09T18:00:00Z",  # Before certified watermark!
        document_type="4",
        raw_xml=raw_xml,
    )
    assert res["status"] == "INGESTED_NEW"

    # Integrity breach MUST be detected
    assert engine.has_active_integrity_breach() is True

    # Freshness evaluation must flag DATA_INCOMPLETE
    freshness = engine.get_freshness_evaluation("2026-10-10T12:00:00Z")
    assert freshness.freshness_status == FreshnessStatus.DATA_INCOMPLETE.value
    assert freshness.readiness_status == ReadinessStatus.DATA_INCOMPLETE.value
    assert freshness.active_breach_count == 1

    # Further certified watermark advancement must be FROZEN
    cycle = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T18:00:00Z")
    engine.acquire_cycle_lease(cycle, "DELIV-2", "EXEC-2")
    with pytest.raises(RuntimeError, match="WATERMARK_INTEGRITY_BREACH"):
        engine.certify_watermark_advancement(cycle, "2026-10-10T18:00:00Z")


def test_form4a_amendment_lineage_and_pit_isolation(temp_engine):
    """Verify Form 4/A amendment handling and zero retroactive leakage in as-of queries."""
    engine, u_build_id = temp_engine
    mgr = engine.ingestion_mgr

    # Ingest original Form 4 at 2026-10-10T02:00:00Z (100 shares)
    xml_orig = SAMPLE_XML_TEMPLATE.format(
        doc_type="4", period="2026-10-10", cik="0000320193", symbol="AAPL",
        tx_date="2026-10-09", shares=100, price=220.0
    )
    engine.ingest_filing_incrementally(
        accession_number="0000320193-26-ORIG01",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T02:00:00Z",
        document_type="4",
        raw_xml=xml_orig,
    )

    # Ingest Form 4/A amendment at 2026-10-10T05:00:00Z (amending to 150 shares)
    xml_amend = SAMPLE_XML_TEMPLATE.format(
        doc_type="4/A", period="2026-10-10", cik="0000320193", symbol="AAPL",
        tx_date="2026-10-09", shares=150, price=220.0
    )
    engine.ingest_filing_incrementally(
        accession_number="0000320193-26-AMEND01",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T05:00:00Z",
        document_type="4/A",
        raw_xml=xml_amend,
    )

    # Query as of 2026-10-10T03:00:00Z (BEFORE amendment) -> Returns original 100 shares!
    txs_before = mgr.get_transactions_as_of("2026-10-10T03:00:00Z", symbol="AAPL")
    assert len(txs_before) == 1
    assert txs_before[0].accession_number == "0000320193-26-ORIG01"
    assert txs_before[0].shares == 100.0

    # Query as of 2026-10-10T06:00:00Z (AFTER amendment) -> Amendment is visible!
    txs_after = mgr.get_transactions_as_of("2026-10-10T06:00:00Z", symbol="AAPL")
    accs = [t.accession_number for t in txs_after]
    assert "0000320193-26-AMEND01" in accs


def test_partition_watermark_property_randomized(temp_engine):
    """Property test across 500 randomized partition frontier combinations."""
    engine, u_build_id = temp_engine
    random.seed(42)

    base_time = datetime(2026, 10, 10, 1, 0, 0, tzinfo=timezone.utc)
    ciks = ["0000320193", "0000789019", "0001045810", "0001652044", "0001018724"]

    failures = 0
    for trial in range(500):
        # Generate random cursors for each partition (0 to 10 hours from base)
        cursors = {}
        for cik in ciks:
            offset = random.randint(10, 600)
            cursors[cik] = (base_time + timedelta(minutes=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
            engine.update_partition_poll_cursor(cik, cursors[cik], poll_succeeded=True)

        # Expected global minimum cursor
        expected_min = min(cursors.values())

        # Compute candidate without gaps
        cand = engine.compute_candidate_watermark(u_build_id)
        if cand != expected_min:
            failures += 1

    assert failures == 0, f"Property test failed in {failures}/500 trials!"


def test_incremental_vs_full_rebuild_oracle_parity(temp_engine):
    """Verify exact parity (content delta = 0) between incremental pipeline and clean full rebuild."""
    engine, u_build_id = temp_engine

    # Ingest 3 distinct incremental filings
    for idx, (sym, cik) in enumerate([("AAPL", "0000320193"), ("MSFT", "0000789019"), ("NVDA", "0001045810")]):
        raw_xml = SAMPLE_XML_TEMPLATE.format(
            doc_type="4", period="2026-10-10", cik=cik, symbol=sym,
            tx_date="2026-10-10", shares=100 * (idx + 1), price=150.0 + idx
        )
        engine.ingest_filing_incrementally(
            accession_number=f"INC-{cik}-00{idx}",
            issuer_cik=cik,
            symbol=sym,
            filing_date="2026-10-10",
            acceptance_datetime=f"2026-10-10T04:0{idx}:00Z",
            document_type="4",
            raw_xml=raw_xml,
        )

    # Reconcile against independent full rebuild through 2026-10-10T05:00:00Z
    parity, audit = engine.reconcile_incremental_vs_full_rebuild(
        as_of_time="2026-10-10T05:00:00Z",
        universe_build_id=u_build_id,
    )
    assert parity is True, f"Full rebuild reconciliation failed: {audit}"
    assert audit["record_key_delta_count"] == 0
    assert audit["hash_mismatch_count"] == 0
    assert audit["parity_result"] == "PASS"


def test_concurrency_lease_collision(temp_engine):
    """Verify that concurrent attempts to acquire lease for the same cycle fail-closed."""
    engine, u_build_id = temp_engine

    cycle = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T12:00:00Z")

    # First attempt acquires lease
    assert engine.acquire_cycle_lease(cycle, "DELIV-A", "EXEC-A") is True

    # Concurrent attempt with same cycle ID is strictly REJECTED
    assert engine.acquire_cycle_lease(cycle, "DELIV-B", "EXEC-B") is False


def test_schema_v2_to_wave2c_migration_parity(tmp_path):
    """Verify schema migration from 2.0.0 to Wave 2C (2.1.0) preserves all legacy tables and records."""
    db_file = str(tmp_path / "test_migration_cache.db")

    # 1. Create a Schema 2.0.0 database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE form4_raw_filings (
            accession_number TEXT PRIMARY KEY,
            issuer_cik TEXT, issuer_symbol TEXT, filing_date TEXT,
            acceptance_datetime TEXT, document_type TEXT, raw_content TEXT,
            source_content_hash TEXT, source_url TEXT, fetched_at TEXT, disposition TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY, accession_number TEXT, issuer_cik TEXT, issuer_symbol TEXT,
            filing_date TEXT, transaction_date TEXT, is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT, public_availability_at TEXT, content_hash TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE sm_source_availability_ledger (
            accession_number TEXT PRIMARY KEY, issuer_cik TEXT, public_availability_at TEXT,
            first_fetched_at TEXT, source_hash TEXT, form_type TEXT, normalization_version TEXT
        )
    """)
    cur.execute("""
        INSERT INTO form4_raw_filings (accession_number, issuer_cik, issuer_symbol, filing_date, raw_content, source_content_hash)
        VALUES ('LEGACY-001', '0000320193', 'AAPL', '2026-10-09', '<xml/>', 'hash1')
    """)
    conn.commit()
    conn.close()

    # 2. Instantiate Form4IngestionManager on the V2 database (triggers migration)
    mgr = Form4IngestionManager(db_path=db_file)

    # 3. Verify Wave 2C tables exist and legacy data is intact
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    for required_table in [
        "sm_incremental_runs",
        "sm_incremental_partitions",
        "sm_partition_watermarks",
        "sm_source_gaps",
        "sm_watermark_receipts",
        "sm_late_arrival_events",
    ]:
        assert required_table in tables, f"Missing Wave 2C table: {required_table}"

    # Verify legacy row is preserved
    cur.execute("SELECT accession_number FROM form4_raw_filings WHERE accession_number = 'LEGACY-001'")
    assert cur.fetchone() is not None
    conn.close()


def test_rate_limit_and_http_429_gap_creation(temp_engine):
    """Section 32: Verify rate limit / 429 gap creation blocks watermark and enforces SEC compliance."""
    engine, u_build_id = temp_engine
    mgr = engine.ingestion_mgr

    # Verify SEC compliance invariants
    assert mgr.rate_limiter.interval >= 0.1, "Rate limit interval must correspond to <= 10 req/s threshold"
    assert "@" in mgr.session.headers.get("User-Agent", ""), "SEC User-Agent must include contact email"

    # Plan cycle
    cycle = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T05:00:00Z")
    engine.acquire_cycle_lease(cycle, "DELIV-429", "EXEC-429")

    # Simulate HTTP 429 encountered during polling for AAPL
    gap = engine.register_gap(
        cycle_id=cycle.get_cycle_id(),
        issuer_cik="0000320193",
        universe_build_id=u_build_id,
        gap_type=GapType.RATE_LIMITED,
        window_start="2026-10-10T02:00:00Z",
        window_end="2026-10-10T05:00:00Z",
    )
    assert gap.status == GapStatus.OPEN.value
    assert gap.gap_type == GapType.RATE_LIMITED.value

    # Update candidate through to window_end for all partitions
    for cik in cycle.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T05:00:00Z", poll_succeeded=True)

    # Candidate watermark calculation must be blocked at gap window_start (02:00:00Z)
    cand_watermark = engine.compute_candidate_watermark(u_build_id)
    assert cand_watermark == "2026-10-10T02:00:00Z"

    # Resolve gap after backoff
    res = engine.resolve_gap(gap.gap_id, "WAVE2C_RECOVERY_AGENT", "HTTP_429_BACKOFF_RETRY_SUCCEEDED")
    assert res is True

    # Candidate watermark now advances freely to window_end
    cand_watermark_after = engine.compute_candidate_watermark(u_build_id)
    assert cand_watermark_after == "2026-10-10T05:00:00Z"


def test_local_scheduler_simulation_and_cadence(temp_engine):
    """Section 34: Verify deterministic multi-cycle scheduling and missed-tick recovery."""
    engine, u_build_id = temp_engine

    # Tick 1: 02:00:00Z
    c1 = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T02:00:00Z")
    assert engine.acquire_cycle_lease(c1, "SCHED-TICK-1", "EXEC-TICK-1")
    for cik in c1.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T02:00:00Z", poll_succeeded=True)
    r1 = engine.certify_watermark_advancement(c1, "2026-10-10T02:00:00Z")
    assert r1.new_certified_watermark == "2026-10-10T02:00:00Z"

    # Tick 2: 02:15:00Z
    c2 = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T02:15:00Z")
    assert c2.previous_certified_watermark == "2026-10-10T02:00:00Z"
    assert engine.acquire_cycle_lease(c2, "SCHED-TICK-2", "EXEC-TICK-2")
    for cik in c2.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T02:15:00Z", poll_succeeded=True)
    r2 = engine.certify_watermark_advancement(c2, "2026-10-10T02:15:00Z")
    assert r2.new_certified_watermark == "2026-10-10T02:15:00Z"

    # Missed Tick: Skip 02:30:00Z, Tick 4 arrives at 02:45:00Z
    c4 = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T02:45:00Z")
    assert c4.previous_certified_watermark == "2026-10-10T02:15:00Z"
    assert engine.acquire_cycle_lease(c4, "SCHED-TICK-4", "EXEC-TICK-4")
    for cik in c4.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T02:45:00Z", poll_succeeded=True)
    r4 = engine.certify_watermark_advancement(c4, "2026-10-10T02:45:00Z")
    assert r4.new_certified_watermark == "2026-10-10T02:45:00Z"


def test_crash_recovery_resumption_without_duplication(temp_engine):
    """Section 30: Verify crash interruption leaves durable state intact and resumes idempotently."""
    engine, u_build_id = temp_engine

    # Step 1: Start cycle and ingest 1 filing
    c1 = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T03:00:00Z")
    engine.acquire_cycle_lease(c1, "CRASH-TEST-1", "EXEC-CRASH-1")

    xml_content = SAMPLE_XML_TEMPLATE.format(
        doc_type="4", period="2026-10-10", cik="0000320193", symbol="AAPL",
        tx_date="2026-10-10", shares=50, price=225.0
    )
    engine.ingest_filing_incrementally(
        accession_number="0000320193-26-CRASH01",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T02:30:00Z",
        document_type="4",
        raw_xml=xml_content,
    )

    # CRASH SIMULATION: Process terminates abruptly before cycle completes or certifies watermark!
    # Certified watermark must still be at baseline
    assert engine.get_global_certified_completeness_watermark().startswith("2026-10-10T01:02:33")

    # Step 2: Resumption after crash
    c_resume = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T03:00:00Z")
    engine.acquire_cycle_lease(c_resume, "CRASH-TEST-2", "EXEC-CRASH-2")

    # Ingest the same filing again: Idempotent NOOP
    res = engine.ingest_filing_incrementally(
        accession_number="0000320193-26-CRASH01",
        issuer_cik="0000320193",
        symbol="AAPL",
        filing_date="2026-10-10",
        acceptance_datetime="2026-10-10T02:30:00Z",
        document_type="4",
        raw_xml=xml_content,
    )
    assert res["status"] == "KNOWN_IDEMPOTENT_NOOP"

    # Complete resumption
    for cik in c_resume.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T03:00:00Z", poll_succeeded=True)
    r_res = engine.certify_watermark_advancement(c_resume, "2026-10-10T03:00:00Z")
    assert r_res.new_certified_watermark == "2026-10-10T03:00:00Z"

    # Verify no duplicate transactions created
    conn = engine._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM form4_parsed_transactions WHERE accession_number = '0000320193-26-CRASH01'")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_retry_exhaustion_blocks_watermark_advancement(temp_engine):
    """Section 14: Verify retry exhaustion pins watermark and preserves DATA_INCOMPLETE."""
    engine, u_build_id = temp_engine

    cycle = engine.plan_cycle(universe_build_id=u_build_id, window_end="2026-10-10T06:00:00Z")
    engine.acquire_cycle_lease(cycle, "EXHAUST-TEST", "EXEC-EXHAUST")

    # Register gap and simulate repeated retries
    gap = engine.register_gap(
        cycle_id=cycle.get_cycle_id(),
        issuer_cik="0000320193",
        universe_build_id=u_build_id,
        gap_type=GapType.FETCH_FAILURE,
        window_start="2026-10-10T02:30:00Z",
        window_end="2026-10-10T06:00:00Z",
    )
    assert gap.attempt_count == 1

    # Retry 1
    engine.register_gap(
        cycle_id=cycle.get_cycle_id(),
        issuer_cik="0000320193",
        universe_build_id=u_build_id,
        gap_type=GapType.FETCH_FAILURE,
        window_start="2026-10-10T02:30:00Z",
        window_end="2026-10-10T06:00:00Z",
    )
    # Retry 2 (Exhaustion reached)
    engine.register_gap(
        cycle_id=cycle.get_cycle_id(),
        issuer_cik="0000320193",
        universe_build_id=u_build_id,
        gap_type=GapType.FETCH_FAILURE,
        window_start="2026-10-10T02:30:00Z",
        window_end="2026-10-10T06:00:00Z",
    )

    # Verify attempt count is 3 and gap remains blocking
    conn = engine._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT attempt_count, status FROM sm_source_gaps WHERE gap_id = ?", (gap.gap_id,))
    row = cur.fetchone()
    assert row[0] == 3
    assert row[1] == GapStatus.RETRYING.value
    conn.close()
    # Update candidate through to window_end for all partitions
    for cik in cycle.partition_ciks:
        engine.update_partition_poll_cursor(cik, "2026-10-10T06:00:00Z", poll_succeeded=True)

    # Candidate watermark is strictly pinned at gap window_start
    cand_watermark = engine.compute_candidate_watermark(u_build_id)
    assert cand_watermark == "2026-10-10T02:30:00Z"

    # Evaluation reflects DATA_INCOMPLETE
    freshness = engine.get_freshness_evaluation()
    assert freshness.readiness_status == ReadinessStatus.DATA_INCOMPLETE.value
    assert freshness.open_gap_count >= 1
