r"""
tests/test_radar_universe_readiness_auditor_166.py

Comprehensive test suite qualifying the ARX Radar readiness assessment auditor under Issue #166.
Verifies all quality gates:
1. Dynamic summary calculation from symbol_level_evaluations across varied synthetic populations.
2. Exact arithmetic closure ($N == \sum \text{breakdowns}$).
3. Insufficient slope history ($200 \le \text{bars} < 222$).
4. Stale completed sessions ($\text{bars} \ge 222$, $\text{latest\_date} < \text{expected\_completed\_session}$).
5. Missing provenance tracking ($\text{bars} \ge 222$, fresh, but 0 provenance rows).
6. Duplicate manifest symbols handling and reconciliation.
7. Filtering observations strictly to pinned cutoff date.
8. Missing evidence never produces READY.
9. Identical inputs and cutoff reproduce identical assessment content (excluding execution timestamp).
10. Explicit evidence custody (mtime, acquisition time unrecorded, production authority NOT_CHECKED).
11. Separation of historical rehearsal cutoff from assessment cutoff.
12. Next production dependency (attributable export, zero production hydration recommendation).
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List
import pytest

from scripts.diagnostics.audit_radar_universe_readiness_166 import (
    compute_file_sha256,
    derive_expected_completed_session,
    generate_markdown_report,
    run_audit,
)


def _init_synthetic_market_db(db_path: Path, records: List[Dict[str, Any]], prov_records: List[Dict[str, Any]] = None):
    """Creates a temporary SQLite market store with daily bars and optional provenance."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE asset_ohlcv_daily (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
    """)
    cur.execute("""
        CREATE TABLE asset_ohlcv_provenance (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            provider TEXT NOT NULL,
            ingestion_source TEXT NOT NULL,
            observed_at TEXT,
            observed_date TEXT,
            observation_precision TEXT NOT NULL,
            observation_source TEXT NOT NULL,
            ingested_at TEXT,
            adjustment_state TEXT NOT NULL,
            structural_quality TEXT NOT NULL,
            fallback_status INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (symbol, trade_date)
        )
    """)
    for r in records:
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["symbol"], r["trade_date"], r.get("open", 100.0), r.get("high", 105.0), r.get("low", 95.0), r.get("close", 102.0), r.get("volume", 1000000)),
        )
    if prov_records:
        for p in prov_records:
            cur.execute(
                "INSERT INTO asset_ohlcv_provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (p["symbol"], p["trade_date"], "ALPACA", "SIP", "2026-11-20T21:00:00Z", p["trade_date"], "MINUTE_BAR", "SIP_CONSOLIDATED", "2026-11-20T21:05:00Z", "SPLIT_ADJUSTED", "CERTIFIED", 0),
            )
    conn.commit()
    conn.close()


def _init_empty_aux_dbs(universe_db: Path, scanner_db: Path):
    """Creates empty universe and scanner SQLite databases."""
    conn = sqlite3.connect(universe_db)
    conn.execute("CREATE TABLE universe_builds (build_id TEXT PRIMARY KEY, created_at TEXT)")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(scanner_db)
    conn.execute("CREATE TABLE scanner_snapshots (snapshot_id TEXT PRIMARY KEY, scanner_id TEXT, run_id TEXT, implementation_release_sha TEXT, status_at_publication TEXT, created_at TEXT, data_as_of TEXT)")
    conn.commit()
    conn.close()


def test_derive_expected_completed_session():
    """Verifies exchange calendar session derivation with date, datetime, weekends, and holidays."""
    # 1. Friday date-only: completed EOD session
    obs, sess, tele = derive_expected_completed_session("2026-11-20")
    assert obs == "2026-11-20"
    assert sess == "2026-11-20"

    # 2. Sunday date-only: rolls to prior Friday session
    obs_sun, sess_sun, _ = derive_expected_completed_session("2026-11-22")
    assert obs_sun == "2026-11-22"
    assert sess_sun == "2026-11-20"

    # 3. Friday morning timestamp (before 16:45 ET settlement): rolls to Thursday
    obs_morn, sess_morn, _ = derive_expected_completed_session("2026-11-20T09:30:00-05:00")
    assert obs_morn == "2026-11-20"
    assert sess_morn == "2026-11-19"

    # 4. Friday evening timestamp (after 16:45 ET settlement): resolves Friday
    obs_eve, sess_eve, _ = derive_expected_completed_session("2026-11-20T17:00:00-05:00")
    assert obs_eve == "2026-11-20"
    assert sess_eve == "2026-11-20"

    # 5. Explicit CLI override
    obs_ov, sess_ov, tele_ov = derive_expected_completed_session("2026-11-20", explicit_expected_session="2026-11-19")
    assert sess_ov == "2026-11-19"
    assert tele_ov["authority"] == "EXPLICIT_CLI_OVERRIDE"


def test_synthetic_fixtures_and_closure(tmp_path: Path):
    """
    Tests dynamic summary calculation and arithmetic closure across different populations
    and history depth buckets using synthetic fixtures.
    """
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    _init_empty_aux_dbs(universe_db, scanner_db)

    # 6 symbols with diverse histories:
    # SYM_ZERO: 0 bars -> MISSING_MARKET_DATA
    # SYM_SHALLOW: 40 bars -> INSUFFICIENT_HISTORY (<50)
    # SYM_MID: 100 bars -> INSUFFICIENT_HISTORY (50-199)
    # SYM_SLOPE: 210 bars -> INSUFFICIENT_SLOPE_HISTORY (200-221)
    # SYM_STALE: 252 bars, latest date 2026-11-13 < 2026-11-20 -> STALE_MARKET_DATA
    # SYM_NOPROV: 252 bars, latest date 2026-11-20, no provenance -> MISSING_PROVENANCE
    # SYM_READY: 252 bars, latest date 2026-11-20, with provenance -> READY

    symbols = ["SYM_ZERO", "SYM_SHALLOW", "SYM_MID", "SYM_SLOPE", "SYM_STALE", "SYM_NOPROV", "SYM_READY"]
    cand_manifest.write_text(json.dumps({"manifest_build_id": "TEST-CAND", "membership_hash": "hash_cand", "symbols": symbols}))
    prom_manifest.write_text(json.dumps({"manifest_build_id": "TEST-PROM", "membership_hash": "hash_prom", "symbols": ["SYM_READY", "SYM_STALE"]}))

    # Generate daily bars
    records = []
    base_date = date(2026, 11, 20)

    # SYM_SHALLOW: 40 bars ending 2026-11-20
    for i in range(40):
        records.append({"symbol": "SYM_SHALLOW", "trade_date": (base_date - timedelta(days=39 - i)).isoformat()})

    # SYM_MID: 100 bars ending 2026-11-20
    for i in range(100):
        records.append({"symbol": "SYM_MID", "trade_date": (base_date - timedelta(days=99 - i)).isoformat()})

    # SYM_SLOPE: 210 bars ending 2026-11-20
    for i in range(210):
        records.append({"symbol": "SYM_SLOPE", "trade_date": (base_date - timedelta(days=209 - i)).isoformat()})

    # SYM_STALE: 252 bars ending 2026-11-13 (stale compared to 2026-11-20)
    stale_date = date(2026, 11, 13)
    for i in range(252):
        records.append({"symbol": "SYM_STALE", "trade_date": (stale_date - timedelta(days=251 - i)).isoformat()})

    # SYM_NOPROV: 252 bars ending 2026-11-20
    for i in range(252):
        records.append({"symbol": "SYM_NOPROV", "trade_date": (base_date - timedelta(days=251 - i)).isoformat()})

    # SYM_READY: 252 bars ending 2026-11-20
    for i in range(252):
        records.append({"symbol": "SYM_READY", "trade_date": (base_date - timedelta(days=251 - i)).isoformat()})

    prov_records = [{"symbol": "SYM_READY", "trade_date": "2026-11-20"}]

    _init_synthetic_market_db(market_db, records, prov_records)

    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
        output_md_path=out_md,
    )

    s_cand = payload["summary_metrics_candidate_universe"]
    assert s_cand["total_symbols"] == 7
    assert s_cand["readiness_breakdown"]["MISSING_MARKET_DATA"] == 1
    assert s_cand["readiness_breakdown"]["INSUFFICIENT_HISTORY"] == 2
    assert s_cand["readiness_breakdown"]["INSUFFICIENT_SLOPE_HISTORY"] == 1
    assert s_cand["readiness_breakdown"]["STALE_MARKET_DATA"] == 1
    assert s_cand["readiness_breakdown"]["MISSING_PROVENANCE"] == 1
    assert s_cand["readiness_breakdown"]["READY"] == 1
    assert s_cand["accounting_closure"]["is_closed"] is True

    # Depth breakdown
    assert s_cand["history_depth_breakdown"]["zero_bars"] == 1
    assert s_cand["history_depth_breakdown"]["lt_50"] == 1
    assert s_cand["history_depth_breakdown"]["50_to_199"] == 1
    assert s_cand["history_depth_breakdown"]["200_to_221"] == 1
    assert s_cand["history_depth_breakdown"]["ge_252"] == 3

    # VCP Stage 2 eligibility
    vcp_s2 = s_cand["vcp_stage2_eligibility_breakdown"]
    assert vcp_s2["MISSING_MARKET_DATA"] == 1
    assert vcp_s2["INSUFFICIENT_HISTORY_SMA200"] == 2
    assert vcp_s2["INSUFFICIENT_HISTORY_SLOPE22"] == 1
    assert vcp_s2["ELIGIBLE"] == 3

    # Promoted reference cohort: 2 symbols (SYM_READY, SYM_STALE)
    s_prom = payload["summary_metrics_promoted_reference"]
    assert s_prom["total_symbols"] == 2
    assert s_prom["readiness_breakdown"]["READY"] == 1
    assert s_prom["readiness_breakdown"]["STALE_MARKET_DATA"] == 1
    assert s_prom["accounting_closure"]["is_closed"] is True

    # Markdown generation
    assert out_md.exists()
    md_text = out_md.read_text(encoding="utf-8")
    assert "Candidate Universe Closure Proof" in md_text
    assert "Promoted Cohort Closure Proof" in md_text


def test_duplicate_manifest_symbols(tmp_path: Path):
    """Verifies that duplicate manifest symbols are detected, tracked, and reconciled without crashing."""
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"

    _init_empty_aux_dbs(universe_db, scanner_db)
    _init_synthetic_market_db(market_db, [])

    # Manifest with 5 entries including duplicates for AAPL (3x) and MSFT (2x)
    raw_symbols = ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"]
    cand_manifest.write_text(json.dumps({"manifest_build_id": "TEST-DUP", "membership_hash": "hash_dup", "symbols": raw_symbols}))
    prom_manifest.write_text(json.dumps({"manifest_build_id": "TEST-PROM", "membership_hash": "hash_prom", "symbols": ["AAPL"]}))

    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
    )

    manifest_meta = payload["authoritative_manifests"][0]
    assert manifest_meta["manifest_symbol_count"] == 5
    assert manifest_meta["distinct_symbol_count"] == 2
    assert manifest_meta["duplicate_symbol_count"] == 3
    assert set(manifest_meta["duplicate_symbols"]) == {"AAPL", "MSFT"}

    s_cand = payload["summary_metrics_candidate_universe"]
    assert s_cand["total_symbols"] == 2
    assert s_cand["readiness_breakdown"]["MISSING_MARKET_DATA"] == 2
    assert s_cand["accounting_closure"]["is_closed"] is True


def test_observation_cutoff_filtering(tmp_path: Path):
    """Verifies that observations are strictly filtered to trade_date <= observation_cutoff."""
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"

    _init_empty_aux_dbs(universe_db, scanner_db)

    # 10 bars from 2026-11-01 to 2026-11-10
    records = []
    for d in range(1, 11):
        dt_str = f"2026-11-{d:02d}"
        records.append({"symbol": "TESTSYM", "trade_date": dt_str, "close": float(100 + d)})

    _init_synthetic_market_db(market_db, records)

    cand_manifest.write_text(json.dumps({"manifest_build_id": "TEST", "membership_hash": "h", "symbols": ["TESTSYM"]}))
    prom_manifest.write_text(json.dumps({"manifest_build_id": "TEST", "membership_hash": "h", "symbols": ["TESTSYM"]}))

    # Cutoff at 2026-11-05: only 5 bars should be observed
    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-05",
        output_json_path=out_json,
    )

    sym_eval = payload["symbol_level_evaluations"][0]
    assert sym_eval["bar_count"] == 5
    assert sym_eval["latest_date"] == "2026-11-05"
    assert sym_eval["latest_price"] == 105.0


def test_missing_evidence_never_produces_ready(tmp_path: Path):
    """Verifies that missing files, missing tables, or unverified production authority never produces READY."""
    missing_market_db = tmp_path / "non_existent.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"

    _init_empty_aux_dbs(universe_db, scanner_db)
    cand_manifest.write_text(json.dumps({"symbols": ["AAPL", "NVDA"]}))
    prom_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))

    payload = run_audit(
        market_db_path=missing_market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
    )

    assert payload["assessment_metadata"]["final_disposition"] == "ASSESSED_AS_OF_EVIDENCE / BLOCKED_BY_EVIDENCE"
    assert payload["summary_metrics_candidate_universe"]["readiness_breakdown"]["READY"] == 0
    assert payload["summary_metrics_candidate_universe"]["readiness_breakdown"]["MISSING_MARKET_DATA"] == 2


def test_reproducibility_identical_inputs(tmp_path: Path):
    """Verifies identical inputs and cutoff reproduce identical assessment content (excluding executed_at_utc)."""
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json_1 = tmp_path / "out1.json"
    out_json_2 = tmp_path / "out2.json"

    _init_empty_aux_dbs(universe_db, scanner_db)
    _init_synthetic_market_db(market_db, [{"symbol": "AAPL", "trade_date": "2026-11-20", "close": 230.0}])
    cand_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))
    prom_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))

    p1 = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json_1,
    )

    p2 = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json_2,
    )

    p1_clean = copy.deepcopy(p1)
    p2_clean = copy.deepcopy(p2)
    p1_clean["assessment_metadata"].pop("executed_at_utc")
    p2_clean["assessment_metadata"].pop("executed_at_utc")

    assert p1_clean == p2_clean


def test_evidence_custody_and_dependency_governance(tmp_path: Path):
    """Verifies mtime labeling, production authority NOT_CHECKED, and attributable export dependency."""
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"

    _init_empty_aux_dbs(universe_db, scanner_db)
    _init_synthetic_market_db(market_db, [])
    cand_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))
    prom_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))

    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
    )

    custody = payload["evidence_custody"]["inspected_market_database"]
    assert "file_mtime_utc" in custody
    assert "acquisition_time_status" in custody
    assert "NOT_RECORDED" in custody["acquisition_time_status"]

    meta = payload["assessment_metadata"]
    assert meta["production_authority_status"] == "NOT_CHECKED"
    assert meta["historical_rehearsal_cutoff"] == "2026-11-06"
    assert meta["assessment_cutoff"] == "2026-11-20"

    pop = payload["population_reconciliation"]
    assert pop["historical_promotion_evidence"]["status"] == "HISTORICALLY_PROMOTED_REFERENCE"
    assert pop["current_production_membership_status"] == "NOT_CHECKED"

    blocker = payload["demonstrated_blockers_and_limitations"]["blocker_1_market_data_deficit_in_local_store"]
    assert "attributable, consistent export" in blocker["next_dependency"].lower()


def test_custody_manifest_confirms_production_authority(tmp_path: Path):
    """Verifies that providing an attributable custody manifest confirms production authority and updates disposition."""
    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    custody_manifest = tmp_path / "custody_manifest.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    _init_empty_aux_dbs(universe_db, scanner_db)
    _init_synthetic_market_db(market_db, [])
    cand_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))
    prom_manifest.write_text(json.dumps({"symbols": ["AAPL"]}))

    custody_data = {
        "acquisition_metadata": {
            "target_service": "web",
            "railway_service_id": "56e38249-b78c-4a60-80d7-303ce7451bd8",
            "railway_deployment_id": "9f28fb4e-c804-4b6a-9a25-1092c496026d",
            "running_revision": "868d7afcd528006c2323e6b79b5a17f2f20f677c",
            "volume_mount": "/root",
            "acquisition_start_utc": "2026-11-20T18:31:12.614167+00:00",
            "acquisition_end_utc": "2026-11-20T18:31:39.403200+00:00",
            "identifiers_unchanged": True,
        },
        "published_bindings": {
            "universe_build_id": "UB-ARX-EXPANSION-500-20261108",
            "membership_hash": "5f60b47d5402c0fc29fa545b244b778ca99a017f47448f47ebdfc257db8881b0",
            "snapshot_id": "vcp-snap-1794130818-3c0eb357",
        },
        "transferred_exports": {},
    }
    custody_manifest.write_text(json.dumps(custody_data))

    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
        output_md_path=out_md,
        custody_manifest_path=custody_manifest,
    )

    meta = payload["assessment_metadata"]
    assert meta["production_authority_status"] == "AUTHORITY_CONFIRMED"
    assert meta["final_disposition"] == "ASSESSED_AS_OF_PRODUCTION_EXPORT"

    pop = payload["population_reconciliation"]
    assert "CONFIRMED_PUBLISHED_PRODUCTION_BUILD" in pop["current_production_membership_status"]

    custody = payload["evidence_custody"]["inspected_market_database"]
    assert custody["source"] == "Attributable Production Store Export (sqlite3_backup with committed WAL frames)"
    assert custody["acquisition_time_status"] == "CONFIRMED_VIA_CUSTODY_MANIFEST"
    assert custody["universe_build_binding"] == "UB-ARX-EXPANSION-500-20261108"

    assert out_md.exists()
    md_text = out_md.read_text()
    assert "AUTHORITY_CONFIRMED" in md_text
    assert "ASSESSED_AS_OF_PRODUCTION_EXPORT" in md_text
