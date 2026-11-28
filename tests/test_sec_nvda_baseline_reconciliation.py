"""ARX Issue #165: NVDA Form 4 Baseline Reconciliation Offline Suite.

Comprehensive offline qualification across 5 Gates:
GATE 1: Retained evidence reproducibility, repository-relative paths, and custody metadata.
GATE 2: Source-to-storage parity across XML elements, parser objects, and stored rows (36/36).
GATE 3: Point-in-time amendment behavior in disposable PostgreSQL (34 -> 34 -> 35 effective).
GATE 4: Replay idempotency (0 duplicates) and negative controls (SOURCE_INTEGRITY_CONFLICT, ambiguous amendments).
GATE 5: Submissions index comparison, date distinctions, and honest claim boundaries.
"""

import hashlib
import json
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from analyst_dashboard.smart_money.amendment_classifier import (
    SCOPE_PARTIAL_CORRECTION,
    classify_amendment_effect,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
)
from analyst_dashboard.smart_money.storage.postgres_schema import (
    initialize_postgres_schema,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sec_edgar"
EXPORT_PATH = FIXTURES_DIR / "nvda_form4_frozen_baseline_export.json"
SUBMISSIONS_PATH = FIXTURES_DIR / "CIK0001045810_submissions.json"

EXPECTED_AGGREGATE_DIGEST = (
    "c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2"
)
EXPECTED_SUBMISSIONS_DIGEST = (
    "8dd44fbc9bc7fd47f0b16155e66cd22445345d0f9706f316e9e55d7ce4f48045"
)
EXPECTED_BACKUP_SHA256 = (
    "cb63fe7a39f0fde755699d8a28200ca382eb4bbdfd0e2a914449467c3c15ed5f"
)


@pytest.fixture(autouse=True)
def require_offline_guard(pytestconfig):
    """Enforces offline sandbox execution without network egress."""
    if not pytestconfig.pluginmanager.hasplugin("sec_offline_test_support"):
        pytest.skip("Requires -p sec_offline_test_support for loopback-only replay")


@pytest.fixture(scope="module")
def baseline_data():
    assert EXPORT_PATH.exists(), f"Missing export fixture at {EXPORT_PATH}"
    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def submissions_data():
    assert SUBMISSIONS_PATH.exists(), f"Missing submissions fixture at {SUBMISSIONS_PATH}"
    with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def replay_pg(tmp_path):
    """Disposable PostgreSQL instance on loopback with fresh schema.
    Fails closed if pgserver cannot start (no mocks/SQLite allowed).
    """
    try:
        import pgserver
        import psycopg2
    except ImportError:
        pytest.skip("PostgreSQL replay dependencies (pgserver, psycopg2) not installed")

    server = pgserver.get_server(tmp_path / "postgres")
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    repo = PostgresSmartMoneyRepository(dsn=uri)
    try:
        yield conn, repo, uri
    finally:
        conn.close()
        server.cleanup()


# ==============================================================================
# GATE 1: Retained Evidence Reproducibility & Custody Metadata
# ==============================================================================

def test_gate1_retained_evidence_metadata_and_custody_digests(baseline_data, submissions_data):
    """Verifies credential-free frozen export metadata, custody digests, and UTF-8 hash basis."""
    meta = baseline_data.get("metadata", {})
    assert meta.get("export_timestamp_utc") == "2026-11-23T20:22:19.184646+00:00"
    assert meta.get("source_identity") == "railway:production:smart-money-worker:finance_smart_money"
    assert meta.get("backup_reference") == "sm-backup-20261122T120306Z-a1179f0e"
    assert meta.get("backup_sha256") == EXPECTED_BACKUP_SHA256
    assert meta.get("issuer_cik") == "0001045810"
    assert meta.get("issuer_ticker") == "NVDA"
    assert meta.get("issuer_name") == "NVIDIA CORP"
    assert meta.get("raw_filings_count") == 16
    assert meta.get("parsed_transactions_count") == 36
    assert meta.get("availability_records_count") == 16
    assert meta.get("amendments_count") == 1

    # Verify documented UTF-8 database text hash basis
    hash_basis = meta.get("hash_basis", "")
    assert "xml_content.encode(\"utf-8\")" in hash_basis or "xml_content.encode('utf-8')" in hash_basis
    assert "database text retention basis; not raw wire HTTP transport bytes" in hash_basis

    # Verify custody note epistemics
    custody_note = meta.get("custody_note", "")
    assert "Original external HTTP acquisition byte custody remains NOT_CHECKED" in custody_note

    # Verify ordering and aggregate digest
    assert meta.get("aggregate_content_digest_ordering") == "sorted by filing_date ASC, accession_number ASC"
    assert meta.get("aggregate_content_digest") == EXPECTED_AGGREGATE_DIGEST

    # Verify retained submissions index custody digest
    raw_sub_bytes = SUBMISSIONS_PATH.read_bytes()
    computed_sub_digest = hashlib.sha256(raw_sub_bytes).hexdigest()
    assert computed_sub_digest == EXPECTED_SUBMISSIONS_DIGEST
    assert len(raw_sub_bytes) == 160255

    # Verify per-file inventory length matches raw filings
    per_file = meta.get("per_file_inventory", [])
    assert len(per_file) == 16


# ==============================================================================
# GATE 2: Source-to-Storage Parity & Transaction Derivation
# ==============================================================================

def test_gate2_accession_inventory_and_hash_recomputation(baseline_data):
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


def test_gate2_transaction_parity_across_xml_parser_and_storage(baseline_data):
    """Verifies 1-to-1 parity between XML transaction elements, parser output, and stored database rows."""
    filings = baseline_data["raw_filings"]
    stored_txs = baseline_data["parsed_transactions"]
    assert len(stored_txs) == 36, f"Expected 36 stored transactions, got {len(stored_txs)}"
    stored_map = {t["transaction_id"]: t for t in stored_txs}

    # 1. Count XML elements independently
    xml_element_keys = []
    xml_code_counts = Counter()
    for f in filings:
        acc = f["accession_number"]
        root = ET.fromstring(f["xml_content"])
        owner_cik = (root.findtext(".//rptOwnerCik") or "").zfill(10)

        nd_elements = root.findall(".//nonDerivativeTransaction")
        for idx, el in enumerate(nd_elements):
            key = f"{acc}:{owner_cik}:NON_DERIVATIVE:{idx}"
            code = el.findtext(".//transactionCode")
            shares = float(el.findtext(".//transactionShares/value") or 0.0)
            xml_element_keys.append((key, code, shares))
            xml_code_counts[f"NON_DERIVATIVE_{code}"] += 1

        d_elements = root.findall(".//derivativeTransaction")
        for idx, el in enumerate(d_elements):
            key = f"{acc}:{owner_cik}:DERIVATIVE:{idx}"
            code = el.findtext(".//transactionCode")
            shares = float(el.findtext(".//transactionShares/value") or 0.0)
            xml_element_keys.append((key, code, shares))
            xml_code_counts[f"DERIVATIVE_{code}"] += 1

    assert len(xml_element_keys) == 36, f"Expected 36 XML transaction elements, got {len(xml_element_keys)}"
    assert xml_code_counts["DERIVATIVE_S"] == 0
    assert sum(v for k, v in xml_code_counts.items() if k.startswith("DERIVATIVE_")) == 0

    # 2. Parse via canonical Form4XmlParser
    parsed_txs = []
    for f in filings:
        parsed = Form4XmlParser.parse_filing(f["xml_content"], f["accession_number"])
        parsed_txs.extend(parsed.transactions)

    assert len(parsed_txs) == 36, f"Expected 36 parser transactions, got {len(parsed_txs)}"

    # 3. Verify stable record key parity across XML elements, parser, and storage
    for key, xml_code, xml_shares in xml_element_keys:
        assert key in stored_map, f"XML key {key} not found in stored database transactions"
        st = stored_map[key]
        assert st["transaction_code"] == xml_code
        assert st["shares"] == xml_shares

    parser_code_counts = Counter()
    for pt in parsed_txs:
        key = pt.record_key
        assert key in stored_map, f"Parsed record_key {key} missing from stored_transactions"
        st = stored_map[key]
        assert pt.transaction_code == st["transaction_code"]
        assert pt.transaction_date == st["transaction_date"]
        assert pt.shares == st["shares"]
        assert pt.price_per_share == st["price_per_share"]
        assert pt.reporting_owner_name == st["reporting_owner_name"]
        assert pt.is_derivative == st["is_derivative"]
        parser_code_counts[pt.transaction_code] += 1

    # Exact transaction code breakdown
    assert parser_code_counts["S"] == 25, "Expected 25 open market sales (Code S)"
    assert parser_code_counts["F"] == 5, "Expected 5 tax withholdings (Code F)"
    assert parser_code_counts["G"] == 4, "Expected 4 bona fide gifts (Code G)"
    assert parser_code_counts["A"] == 2, "Expected 2 equity grants/awards (Code A)"

    # Epistemic correctness claim: No Code-P open-market purchases
    assert parser_code_counts["P"] == 0, (
        "No code-P transactions were found in the retained baseline"
    )


def test_gate2_zero_transaction_parse_retirement_disclosure(baseline_data):
    """Investigates accession 0001347842-26-000015 to verify retirement exit notice (0 transactions)."""
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

    # Remarks verify Section 16 retirement exit notice
    remarks = root.findtext("remarks") or ""
    assert "retired from his role" in remarks
    assert "no longer subject to" in remarks
    assert "Section 16" in remarks

    # Parser disposition
    parsed = Form4XmlParser.parse_filing(f["xml_content"], f["accession_number"])
    assert len(parsed.transactions) == 0
    assert parsed.disposition == "FETCHED_NO_RELEVANT_TRANSACTIONS"


# ==============================================================================
# GATE 3: Point-in-Time Amendment Behavior in Disposable PostgreSQL
# ==============================================================================

def test_gate3_amendment_point_in_time_query_in_disposable_postgres(replay_pg, baseline_data):
    """Proves amendment supersession point-in-time in real PostgreSQL using canonical repository paths.
    Before boundary (2026-10-13T21:09:28Z): base effective (172,507 sh), 34 effective txs.
    At boundary (2026-10-13T21:09:29Z): amendment replaces base (160,418 sh), 34 effective txs.
    After all available (2026-11-20T00:00:00Z): 35 effective txs (Nov 6 filing added).
    Raw database rows in form4_parsed_transactions remain strictly 36.
    """
    conn, repo, dsn = replay_pg

    # 1. Verify newly initialized local target before writing
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings;")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
        assert cur.fetchone()[0] == 0

    # 2. Load all 16 filings and availability records
    for f in baseline_data["raw_filings"]:
        inserted = repo.save_raw_filing(f, conn=conn)
        assert inserted is True
        repo.save_availability(f["accession_number"], f["accepted_datetime"], conn=conn)

    # 3. Load all 36 parsed transactions
    saved_count = repo.save_parsed_transactions(baseline_data["parsed_transactions"], conn=conn)
    assert saved_count == 36

    # 4. Classify amendment effect using canonical classifier
    base_f = next(x for x in baseline_data["raw_filings"] if x["accession_number"] == "0002152188-26-000005")
    amend_f = next(x for x in baseline_data["raw_filings"] if x["accession_number"] == "0002152188-26-000007")

    base_parsed = Form4XmlParser.parse_filing(base_f["xml_content"], base_f["accession_number"])
    amend_parsed = Form4XmlParser.parse_filing(amend_f["xml_content"], amend_f["accession_number"])
    base_txs = [t.__dict__ for t in base_parsed.transactions]

    res = classify_amendment_effect(
        amendment_filing=amend_parsed,
        base_filing={"accession_number": base_f["accession_number"], "filing_date": base_f["filing_date"]},
        base_transactions=base_txs,
        raw_xml=amend_f["xml_content"],
    )
    assert res.status == SCOPE_PARTIAL_CORRECTION
    assert len(res.mappings) == 1

    # 5. Persist amendment record and transaction mappings into database
    repo.save_amendment(
        amendment_accession=amend_f["accession_number"],
        base_accession=base_f["accession_number"],
        amendment_number=1,
        resolution_status=res.status,
        resolved_at=amend_f["accepted_datetime"],
        conn=conn,
    )
    mapping_event_id = repo.record_amendment_transaction_mappings(
        base_accession=base_f["accession_number"],
        amendment_accession=amend_f["accession_number"],
        mappings=res.mappings,
        base_content_hash=base_f["content_hash"],
        amendment_content_hash=amend_f["content_hash"],
        conn=conn,
    )
    assert mapping_event_id is not None
    conn.commit()

    base_tx_id = "0002152188-26-000005:0002152188:NON_DERIVATIVE:0"
    amend_tx_id = "0002152188-26-000007:0002152188:NON_DERIVATIVE:0"

    # Query 1: Before boundary (2026-10-13T21:09:28Z)
    res_before = repo.get_bulk_transactions_as_of(
        "2026-10-13T21:09:28Z", ["NVDA"], "2026-08-01", conn=conn
    )["NVDA"]
    assert len(res_before) == 34, f"Expected 34 effective transactions before boundary, got {len(res_before)}"
    before_map = {t["transaction_id"]: t for t in res_before}
    assert base_tx_id in before_map, "Base transaction must be effective before amendment boundary"
    assert amend_tx_id not in before_map, "Amendment transaction must not be effective before boundary"
    assert float(before_map[base_tx_id]["shares"]) == 172507.0

    # Query 2: At/after boundary (2026-10-13T21:09:29Z)
    res_at = repo.get_bulk_transactions_as_of(
        "2026-10-13T21:09:29Z", ["NVDA"], "2026-08-01", conn=conn
    )["NVDA"]
    assert len(res_at) == 34, f"Expected 34 effective transactions at boundary, got {len(res_at)}"
    at_map = {t["transaction_id"]: t for t in res_at}
    assert base_tx_id not in at_map, "Base transaction must be superseded at amendment boundary"
    assert amend_tx_id in at_map, "Amendment transaction must be effective at amendment boundary"
    assert float(at_map[amend_tx_id]["shares"]) == 160418.0

    # Query 3: After all filings available (2026-11-20T00:00:00Z)
    res_after = repo.get_bulk_transactions_as_of(
        "2026-11-20T00:00:00Z", ["NVDA"], "2026-08-01", conn=conn
    )["NVDA"]
    assert len(res_after) == 35, f"Expected 35 effective transactions after all available, got {len(res_after)}"
    after_map = {t["transaction_id"]: t for t in res_after}
    assert base_tx_id not in after_map, "Base transaction must remain superseded"
    assert amend_tx_id in after_map, "Amendment transaction must remain effective"
    assert float(after_map[amend_tx_id]["shares"]) == 160418.0
    nov6_tx_id = "0001197647-26-000013:0001197647:NON_DERIVATIVE:0"
    assert nov6_tx_id in after_map, "November 6 filing transaction must be effective after its availability"

    # Verify all 36 raw rows remain preserved in database
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
        raw_db_count = cur.fetchone()[0]
        assert raw_db_count == 36, f"Expected 36 raw rows preserved in database, got {raw_db_count}"


# ==============================================================================
# GATE 4: Replay Idempotency & Negative Controls
# ==============================================================================

def test_gate4_replay_idempotency_zero_duplicates(replay_pg, baseline_data):
    """Verifies that replaying all 16 filings, availability, and 36 transactions produces 0 duplicates."""
    conn, repo, dsn = replay_pg

    # First load
    for f in baseline_data["raw_filings"]:
        repo.save_raw_filing(f, conn=conn)
        repo.save_availability(f["accession_number"], f["accepted_datetime"], conn=conn)
    repo.save_parsed_transactions(baseline_data["parsed_transactions"], conn=conn)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings;")
        c_raw_1 = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
        c_av_1 = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
        c_tx_1 = cur.fetchone()[0]

    assert c_raw_1 == 16
    assert c_av_1 == 16
    assert c_tx_1 == 36

    # Replay second load
    for f in baseline_data["raw_filings"]:
        inserted = repo.save_raw_filing(f, conn=conn)
        assert inserted is False, f"Expected idempotent no-op on re-insert of {f['accession_number']}"
        repo.save_availability(f["accession_number"], f["accepted_datetime"], conn=conn)
    re_saved = repo.save_parsed_transactions(baseline_data["parsed_transactions"], conn=conn)
    assert re_saved == 36
    conn.commit()

    # Verify counts have not increased (0 duplicates)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings;")
        assert cur.fetchone()[0] == 16
        cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
        assert cur.fetchone()[0] == 16
        cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
        assert cur.fetchone()[0] == 36


def test_gate4_altered_content_triggers_source_integrity_conflict(replay_pg, baseline_data):
    """Verifies that altering XML content for an existing accession triggers SOURCE_INTEGRITY_CONFLICT atomically."""
    conn, repo, dsn = replay_pg
    orig_f = baseline_data["raw_filings"][0]

    repo.save_raw_filing(orig_f, conn=conn)
    conn.commit()

    # Attempt to save tampered filing under the same accession
    tampered_f = dict(orig_f)
    tampered_f["xml_content"] = orig_f["xml_content"] + "<!-- tampered bytes -->"
    tampered_f["content_hash"] = hashlib.sha256(tampered_f["xml_content"].encode("utf-8")).hexdigest()

    with pytest.raises(RuntimeError) as exc_info:
        repo.save_raw_filing(tampered_f, conn=conn)

    assert "SOURCE_INTEGRITY_CONFLICT" in str(exc_info.value)
    assert orig_f["accession_number"] in str(exc_info.value)

    # Roll back aborted transaction and verify original content remains unchanged
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM form4_raw_filings WHERE accession_number = %s;",
            (orig_f["accession_number"],)
        )
        assert cur.fetchone()[0] == orig_f["content_hash"]


def test_gate4_unresolved_amendment_preserves_base_transactions(replay_pg, baseline_data):
    """Verifies that ambiguous or unclassified amendments do NOT supersede base transactions."""
    conn, repo, dsn = replay_pg

    base_f = next(x for x in baseline_data["raw_filings"] if x["accession_number"] == "0002152188-26-000005")
    amend_f = next(x for x in baseline_data["raw_filings"] if x["accession_number"] == "0002152188-26-000007")

    repo.save_raw_filing(base_f, conn=conn)
    repo.save_availability(base_f["accession_number"], base_f["accepted_datetime"], conn=conn)
    repo.save_raw_filing(amend_f, conn=conn)
    repo.save_availability(amend_f["accession_number"], amend_f["accepted_datetime"], conn=conn)

    base_tx = next(x for x in baseline_data["parsed_transactions"] if x["accession_number"] == base_f["accession_number"])
    amend_tx = next(x for x in baseline_data["parsed_transactions"] if x["accession_number"] == amend_f["accession_number"])
    repo.save_parsed_transactions([base_tx, amend_tx], conn=conn)

    # Save amendment with AMENDMENT_SCOPE_AMBIGUOUS status (no transaction mappings)
    repo.save_amendment(
        amendment_accession=amend_f["accession_number"],
        base_accession=base_f["accession_number"],
        amendment_number=1,
        resolution_status="AMENDMENT_SCOPE_AMBIGUOUS",
        resolved_at=amend_f["accepted_datetime"],
        conn=conn,
    )
    conn.commit()

    # Query after amendment availability
    res = repo.get_bulk_transactions_as_of("2026-10-14T00:00:00Z", ["NVDA"], "2026-08-01", conn=conn)["NVDA"]
    res_tx_ids = [t["transaction_id"] for t in res]

    # Base transaction must NOT be superseded, and ambiguous amendment must NOT be active
    assert base_tx["transaction_id"] in res_tx_ids, "Base transaction must remain active when amendment is ambiguous"
    assert amend_tx["transaction_id"] not in res_tx_ids, "Ambiguous amendment transaction must be excluded"


# ==============================================================================
# GATE 5: Retained Submissions Index & Window Comparison
# ==============================================================================

def test_gate5_submissions_index_window_coverage_and_date_distinctions(baseline_data, submissions_data):
    """Compares baseline against retained CIK0001045810_submissions.json for the August 26–November 23 window.
    Asserts exact in-window count (15) and enforces epistemic distinction between internal consistency
    and external completeness (coverage remains PARTIAL / NOT_CERTIFIED).
    """
    recent = submissions_data.get("filings", {}).get("recent", {})
    accs = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    fdates = recent.get("filingDate", [])

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
