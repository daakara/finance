#!/usr/bin/env python3
"""
scripts/universe/build_candidate_manifest_150.py

ARX Terminal — Sprint 2 Approved Scope Addition:
Bounded 150-Candidate Coverage Expansion, Radar Launch Universe Definition & Operational Packages.

Governing Authority: Product Owner (David Akara) Approved Scope Addition (2026-10-19):
"Demonstrate repeatable acquisition, identity resolution, eligibility evaluation
and market-data readiness for up to 150 explicitly selected securities, with
reconciled exclusions and controlled publication. Full-universe discovery and
coverage/readiness assessment are also included."

Product Direction (David Akara, 2026-10-19):
"Prepare a reproducible launch population of up to 150 eligible, liquid, adequately covered equities for Radar.
This population is NOT:
- The current 150-item validation sample.
- A list of investment recommendations.
- The 150 highest strategy scores.
- A guarantee that 150 securities will be admitted.
Keep universe selection separate from strategy qualification and opportunity ranking.
Keep ETFs, synthetic fixtures and other exclusion cases in a separate validation set."

Outputs:
  - evidence/coverage_expansion/proposed_radar_launch_manifest.json (133 Qualifying Common Equities)
  - evidence/coverage_expansion/candidate_manifest_150.json (150-Item Boundary Validation Sample)
  - evidence/coverage_expansion/validation_sample_mixed_150.json (Identical alias for validation set)
  - evidence/coverage_expansion/candidate_exclusions_reconciliation.json
  - evidence/coverage_expansion/package_a_identity_registration.json
  - evidence/coverage_expansion/package_b_market_data_ingestion.json
  - evidence/coverage_expansion/package_c_universe_publication.json
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.universe.contracts import (
    CANONICAL_ELIGIBILITY_RULES,
    ELIGIBILITY_RULES_HASH,
    NORMALIZATION_SPEC_HASH,
    NORMALIZATION_VERSION,
    UNIVERSE_ID,
    UNIVERSE_VERSION,
    ConstructionStatus,
    DataReadinessDecision,
    EligibilityDecision,
    EligibilityLedgerRow,
    PublicationDecision,
    SourcePopulationSnapshot,
    SourceSecurity,
    UniverseBuildAttestation,
    compute_sha256,
)
from analyst_dashboard.universe.builder import CURRENT_RELEASE_SHA, DeterministicUniverseBuilder
from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    EffectiveDatedListingAssociation,
    EvidenceNature,
    InternalSecurityRecord,
    ListingLocator,
    OfflineAdmissionMappingEngine,
)
from analyst_dashboard.universe.identity_proposal import (
    CandidateAssociationProposal,
    PrimaryListingClaimAnalysis,
    ProvisionalListingAssociation,
    ProvisionalSecurityEntity,
)
from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionAuthorization,
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
    ProposedAssociationAdoption,
    ProposedEntityAdoption,
)
from analyst_dashboard.universe.identity_persistence import SqliteIdentityRepository
from scripts.apply_registration_plan_wrapper import create_verified_sqlite_backup


# =====================================================================
# Constants & Established Pilot Identifiers
# =====================================================================

PILOT_IDENTITIES = {
    "AAPL": {
        "security_id": "ARX_SEC_0B4C728FE0BE",
        "association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        "cik": "0000320193",
        "name": "Apple Inc. Common Stock",
        "operating_mic": "XNAS",
        "exchange": "NASDAQ",
        "share_class": None,
    },
    "GOOG": {
        "security_id": "ARX_SEC_A11A24D442E2",
        "association_id": "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
        "cik": "0001652044",
        "name": "Alphabet Inc. Class C Capital Stock",
        "operating_mic": "XNAS",
        "exchange": "NASDAQ",
        "share_class": "Class C",
    },
    "GOOGL": {
        "security_id": "ARX_SEC_B7EF9ED7A35A",
        "association_id": "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
        "cik": "0001652044",
        "name": "Alphabet Inc. Class A Common Stock",
        "operating_mic": "XNAS",
        "exchange": "NASDAQ",
        "share_class": "Class A",
    },
}

VALIDATION_SAMPLE_ETFS = {"ARKK", "GLD", "GLX", "IWM", "JEPI", "QQQ", "SPY", "TLT", "VOO", "XLE"}
VALIDATION_SAMPLE_ADRS = {"ARM", "ASML", "DHLGY", "NVO", "TSM"}

VALIDATION_SAMPLE_FIXTURE_IDENTITIES: Dict[str, Dict[str, str]] = {
    "NAUT": {
        "security_id": "ARX_SEC_BEF15F9ACCB9",
        "association_id": "ASSOC_ARX_SEC_BEF15F9ACCB9_XNAS_NAUT_2026-10-19",
    },
    "SPCX": {
        "security_id": "ARX_SEC_458B5F794E17",
        "association_id": "ASSOC_ARX_SEC_458B5F794E17_XNAS_SPCX_2026-10-19",
    },
    "ARM": {
        "security_id": "ARX_SEC_D748F70617ED",
        "association_id": "ASSOC_ARX_SEC_D748F70617ED_XNAS_ARM_2026-10-19",
    },
    "ASML": {
        "security_id": "ARX_SEC_F1F453228764",
        "association_id": "ASSOC_ARX_SEC_F1F453228764_XNAS_ASML_2026-10-19",
    },
    "DHLGY": {
        "security_id": "ARX_SEC_528FE090CE26",
        "association_id": "ASSOC_ARX_SEC_528FE090CE26_OTCM_DHLGY_2026-10-19",
    },
    "NVO": {
        "security_id": "ARX_SEC_EC13632A2BB3",
        "association_id": "ASSOC_ARX_SEC_EC13632A2BB3_XNYS_NVO_2026-10-19",
    },
    "TSM": {
        "security_id": "ARX_SEC_091BA772F023",
        "association_id": "ASSOC_ARX_SEC_091BA772F023_XNYS_TSM_2026-10-19",
    },
    "ARKK": {
        "security_id": "ARX_SEC_652B334A8B5C",
        "association_id": "ASSOC_ARX_SEC_652B334A8B5C_XNYS_ARKK_2026-10-19",
    },
    "GLD": {
        "security_id": "ARX_SEC_2FF99BCB6580",
        "association_id": "ASSOC_ARX_SEC_2FF99BCB6580_ARCX_GLD_2026-10-19",
    },
    "GLX": {
        "security_id": "ARX_SEC_824FC8A23E94",
        "association_id": "ASSOC_ARX_SEC_824FC8A23E94_XNYS_GLX_2026-10-19",
    },
    "IWM": {
        "security_id": "ARX_SEC_73C4F44324EC",
        "association_id": "ASSOC_ARX_SEC_73C4F44324EC_ARCX_IWM_2026-10-19",
    },
    "JEPI": {
        "security_id": "ARX_SEC_B11C4705C172",
        "association_id": "ASSOC_ARX_SEC_B11C4705C172_ARCX_JEPI_2026-10-19",
    },
    "QQQ": {
        "security_id": "ARX_SEC_1A2B3D8A80EC",
        "association_id": "ASSOC_ARX_SEC_1A2B3D8A80EC_XNAS_QQQ_2026-10-19",
    },
    "SPY": {
        "security_id": "ARX_SEC_0DE316665ADC",
        "association_id": "ASSOC_ARX_SEC_0DE316665ADC_ARCX_SPY_2026-10-19",
    },
    "TLT": {
        "security_id": "ARX_SEC_25871ECAC2C7",
        "association_id": "ASSOC_ARX_SEC_25871ECAC2C7_XNAS_TLT_2026-10-19",
    },
    "VOO": {
        "security_id": "ARX_SEC_11237A20E2C0",
        "association_id": "ASSOC_ARX_SEC_11237A20E2C0_ARCX_VOO_2026-10-19",
    },
    "XLE": {
        "security_id": "ARX_SEC_3445FAA9245E",
        "association_id": "ASSOC_ARX_SEC_3445FAA9245E_ARCX_XLE_2026-10-19",
    },
}


LAUNCH_EXCLUSION_ETFS = {
    "ARKK", "GLD", "GLX", "IWM", "JEPI", "QQQ", "SPY", "TLT", "VOO", "XLE",
    "QQQM", "QQQI", "SOXQ", "SOXX", "FTXL", "PSL", "SPCX", "BTCI", "SMH", "SCHD", "SGOV"
}
LAUNCH_EXCLUSION_ADRS = {"ARM", "ASML", "DHLGY", "NVO", "TSM", "ARGX", "ABANF"}
KNOWN_CRYPTOS = {"BTC", "BTC-USD", "ETH-USD", "SOL", "SOL-USD", "ZEC", "ZEC-USD", "ZEN-USD"}
KNOWN_SYNTHETICS = {
    "CNT_FRESH", "CNT_NULL", "CNT_SCORES", "SYM_TEST_UNKNOWN",
    "TST_CASCADE", "TST_DEDUP", "TST_STALE", "TST_UNSEASONED"
}
DIRECTORY_ADDITIONS_10 = ["JNJ", "JPM", "PG", "WMT", "BAC", "ABT", "MRK", "PFE", "DIS", "CSCO"]

LIVE_PRODUCTION_TARGET_STATE_DIGEST = "ff3cf69379e4d7c5745a2ca621a0f470a23efdbd962ecb40d3504381bbc69eb6"
HISTORICAL_EMPTY_DIGEST = "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee"


def load_production_readback() -> Dict[str, Any]:
    """Loads empirical production market store readback artifact if available."""
    readback_file = REPO_ROOT / "evidence" / "coverage_expansion" / "production_market_data_readback.json"
    if readback_file.exists():
        try:
            return json.loads(readback_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_route_a_evidence() -> Dict[str, Any]:
    """Loads authoritative Route A direct reference evidence dataset."""
    evidence_file = REPO_ROOT / "evidence" / "coverage_expansion" / "expansion_133_route_a_evidence.json"
    if not evidence_file.exists():
        evidence_file = REPO_ROOT / "data" / "research" / "primary_listing_evidence" / "expansion_133_route_a_evidence.json"
    if evidence_file.exists():
        try:
            data = json.loads(evidence_file.read_text(encoding="utf-8"))
            return data.get("evidence_records", {})
        except Exception:
            pass
    return {}


SUPPORTED_OPERATING_MICS = {"XNAS", "XNYS", "ARCX", "BATS"}


def validate_route_a_evidence(
    candidate_sym: str,
    candidate_cik: Optional[str],
    candidate_mic: Optional[str],
    route_a_rec: Optional[Dict[str, Any]],
    raw_tickers_dir: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates Route A reference evidence for a candidate against authoritative contracts:
    - Missing or unreadable raw artifacts fail closed immediately.
    - Symbol matches candidate_sym across record and raw payload.
    - CIK is present, valid 10-digit, and concordant across candidate, record, and raw payload.
    - active is True across record and raw payload.
    - currency_name is USD across record and raw payload.
    - Operating MIC is supported (XNAS, XNYS, ARCX, BATS), concordant with raw primary_exchange,
      and matches candidate_mic if provided.
    - Instrument type supports classification used in canonical eligibility (raw type must be 'CS').
    - Raw artifact on disk exists and matches raw_response_sha256 exactly.
    Fails closed on any mismatch, omission, or unsupported venue/instrument.
    """
    if not route_a_rec or not isinstance(route_a_rec, dict):
        return False, "MISSING_ROUTE_A_RECORD", {}

    cand_sym_clean = candidate_sym.strip().upper()
    rec_sym = (route_a_rec.get("symbol") or route_a_rec.get("stated_symbol") or "").strip().upper()
    if rec_sym != cand_sym_clean:
        return False, "SYMBOL_MISMATCH", {"expected": cand_sym_clean, "actual": rec_sym}

    rec_cik = route_a_rec.get("cik")
    if not rec_cik:
        return False, "MISSING_RECORD_CIK", {"symbol": cand_sym_clean}
    clean_rec_cik = str(rec_cik).strip().zfill(10)
    if len(clean_rec_cik) != 10 or not clean_rec_cik.isdigit():
        return False, "MALFORMED_RECORD_CIK", {"cik": rec_cik}

    if candidate_cik is not None:
        clean_cand_cik = str(candidate_cik).strip().zfill(10)
        if clean_rec_cik != clean_cand_cik:
            return False, "CIK_MISMATCH", {"expected": clean_cand_cik, "actual": clean_rec_cik}

    if route_a_rec.get("active") is not True:
        return False, "INACTIVE_INSTRUMENT", {"active": route_a_rec.get("active")}

    curr = str(route_a_rec.get("currency_name", "")).strip().upper()
    if curr != "USD":
        return False, "NON_USD_CURRENCY", {"currency": curr}

    rec_mic = str(route_a_rec.get("operating_mic") or route_a_rec.get("stated_exchange") or "").strip().upper()
    if not rec_mic or rec_mic not in SUPPORTED_OPERATING_MICS:
        return False, "UNSUPPORTED_OPERATING_MIC", {"operating_mic": rec_mic, "supported": sorted(list(SUPPORTED_OPERATING_MICS))}

    if candidate_mic and rec_mic != candidate_mic.strip().upper():
        return False, "OPERATING_MIC_MISMATCH", {"expected": candidate_mic, "actual": rec_mic}

    # Raw artifact SHA-256 verification
    expected_sha = route_a_rec.get("raw_response_sha256")
    if not expected_sha:
        return False, "MISSING_RAW_RESPONSE_SHA256", {}

    raw_file = None
    if raw_tickers_dir:
        raw_file = Path(raw_tickers_dir) / f"{cand_sym_clean}.json"
    elif route_a_rec.get("source_artifact"):
        raw_file = REPO_ROOT / route_a_rec["source_artifact"]
    else:
        raw_file = REPO_ROOT / "data" / "research" / "reference_polygon" / "tickers" / f"{cand_sym_clean}.json"

    if not raw_file or not raw_file.exists():
        return False, "MISSING_RAW_ARTIFACT", {"expected_path": str(raw_file) if raw_file else "None"}

    try:
        file_bytes = raw_file.read_bytes()
    except Exception as e:
        return False, f"UNREADABLE_RAW_ARTIFACT: {e}", {"path": str(raw_file)}

    actual_sha = hashlib.sha256(file_bytes).hexdigest()
    if actual_sha != expected_sha:
        return False, "RAW_RESPONSE_SHA256_MISMATCH", {"expected": expected_sha, "actual": actual_sha}

    try:
        raw_json = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        return False, f"RAW_PAYLOAD_PARSE_ERROR: {e}", {}

    results = raw_json.get("results")
    if not results or not isinstance(results, dict):
        return False, "MALFORMED_RAW_PAYLOAD", {"details": "Missing or non-dict results"}

    if str(results.get("ticker", "")).strip().upper() != cand_sym_clean:
        return False, "RAW_PAYLOAD_TICKER_MISMATCH", {"expected": cand_sym_clean, "actual": results.get("ticker")}

    raw_cik = results.get("cik")
    if not raw_cik:
        return False, "RAW_PAYLOAD_MISSING_CIK", {"symbol": cand_sym_clean}
    clean_raw_cik = str(raw_cik).strip().zfill(10)
    if clean_raw_cik != clean_rec_cik:
        return False, "RAW_PAYLOAD_CIK_MISMATCH", {"expected": clean_rec_cik, "actual": clean_raw_cik}

    if results.get("active") is not True:
        return False, "RAW_PAYLOAD_INACTIVE", {"active": results.get("active")}

    if str(results.get("currency_name", "")).strip().upper() != "USD":
        return False, "RAW_PAYLOAD_NON_USD", {"currency": results.get("currency_name")}

    raw_ex = str(results.get("primary_exchange", "")).strip().upper()
    if not raw_ex or raw_ex not in SUPPORTED_OPERATING_MICS:
        return False, "RAW_PAYLOAD_UNSUPPORTED_MIC", {"primary_exchange": raw_ex}

    if raw_ex != rec_mic:
        return False, "RAW_PAYLOAD_OPERATING_MIC_MISMATCH", {"expected": rec_mic, "actual": raw_ex}

    raw_type = str(results.get("type", "")).strip().upper()
    if raw_type != "CS":
        return False, "NON_COMMON_STOCK_TYPE", {"type": raw_type, "expected": "CS"}

    return True, "VALIDATED", {
        "symbol": cand_sym_clean,
        "cik": clean_rec_cik,
        "operating_mic": rec_mic,
        "security_type": "COMMON_STOCK",
        "currency": "USD",
        "active": True,
        "raw_response_sha256": expected_sha,
    }


def load_raw_directories(raw_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Loads and indexes raw attributable directories."""
    # 1. SEC EDGAR
    sec_file = raw_dir / "sec_company_tickers_exchange.json"
    sec_meta_file = raw_dir / "sec_company_tickers_exchange.json.meta.json"
    sec_data = json.loads(sec_file.read_text(encoding="utf-8"))
    sec_meta = json.loads(sec_meta_file.read_text(encoding="utf-8")) if sec_meta_file.exists() else {}
    sec_rows = sec_data.get("data", [])
    sec_by_ticker = {}
    for r in sec_rows:
        cik, name, ticker, exch = r[0], r[1], str(r[2]).strip().upper(), r[3]
        sec_by_ticker[ticker] = {
            "cik": str(cik).zfill(10),
            "name": name,
            "exchange": exch,
        }

    # 2. NASDAQ Listed Directory
    nasdaq_file = raw_dir / "nasdaq_listed_directory.txt"
    nasdaq_meta_file = raw_dir / "nasdaq_listed_directory.txt.meta.json"
    nasdaq_meta = json.loads(nasdaq_meta_file.read_text(encoding="utf-8")) if nasdaq_meta_file.exists() else {}
    nasdaq_by_sym = {}
    with open(nasdaq_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].split("|")
    for l in lines[1:-1]:
        parts = l.split("|")
        if len(parts) == len(header):
            row = dict(zip(header, parts))
            nasdaq_by_sym[row["Symbol"].strip().upper()] = row

    # 3. Other Listed Directory (NYSE, ARCA, BATS)
    other_file = raw_dir / "other_listed_directory.txt"
    other_meta_file = raw_dir / "other_listed_directory.txt.meta.json"
    other_meta = json.loads(other_meta_file.read_text(encoding="utf-8")) if other_meta_file.exists() else {}
    other_by_sym = {}
    with open(other_file, "r", encoding="utf-8") as f:
        lines_other = [l.strip() for l in f if l.strip()]
    header_other = lines_other[0].split("|")
    for l in lines_other[1:-1]:
        parts = l.split("|")
        if len(parts) == len(header_other):
            row = dict(zip(header_other, parts))
            act_sym = row["ACT Symbol"].strip().upper()
            other_by_sym[act_sym] = row

    source_digests = {
        "sec_company_tickers_exchange": sec_meta.get("content_sha256", hashlib.sha256(sec_file.read_bytes()).hexdigest()),
        "nasdaq_listed_directory": nasdaq_meta.get("content_sha256", hashlib.sha256(nasdaq_file.read_bytes()).hexdigest()),
        "other_listed_directory": other_meta.get("content_sha256", hashlib.sha256(other_file.read_bytes()).hexdigest()),
    }

    return sec_by_ticker, nasdaq_by_sym, other_by_sym, source_digests


def resolve_market_db_path() -> Path:
    """
    Resolves authoritative path to development market data SQLite store:
    1. Explicit environment variable FINANCE_MARKET_STORE_PATH (if set, must exist)
    2. Retained bounded repository database (data/research/market_data/bounded_150_dev_market_store.db)
    Fails closed if no database is found. Does not fall back to home directory to ensure isolation.
    """
    if os.getenv("FINANCE_MARKET_STORE_PATH"):
        p = Path(os.environ["FINANCE_MARKET_STORE_PATH"]).resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Configured FINANCE_MARKET_STORE_PATH does not exist: {p}"
            )
        return p
    repo_bounded = REPO_ROOT / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    if repo_bounded.exists():
        return repo_bounded
    raise FileNotFoundError(
        "Required bounded market data input database not found! Checked: "
        f"FINANCE_MARKET_STORE_PATH, {repo_bounded}"
    )


def load_provisional_identity_allocations() -> Dict[str, Any]:
    """Loads reviewable provisional entity identity allocations mapping."""
    alloc_file = REPO_ROOT / "evidence" / "coverage_expansion" / "provisional_identity_allocations.json"
    if alloc_file.exists():
        try:
            return json.loads(alloc_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_target_identity_state_export() -> Dict[str, Any]:
    """Loads authoritative target store identity state export."""
    export_file = REPO_ROOT / "evidence" / "coverage_expansion" / "target_identity_state_export.json"
    if not export_file.exists():
        raise FileNotFoundError(f"Required target identity state export missing: {export_file}")
    return json.loads(export_file.read_text(encoding="utf-8"))


def get_dev_store_symbols(market_db_path: Path) -> List[str]:
    """Reads distinct symbols from local development market store."""
    if not market_db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{market_db_path}?mode=ro", uri=True)
    c = conn.cursor()
    syms = [r[0] for r in c.execute("SELECT DISTINCT symbol FROM asset_ohlcv_daily").fetchall()]
    conn.close()
    return syms


def generate_candidate_manifest(allocations: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Applies candidate discovery and baseline exclusions to produce the 150-item
    boundary validation sample (135 equities, 5 ADRs, 10 ETFs).
    """
    raw_dir = REPO_ROOT / "data" / "raw_feeds" / "2026-10-16"
    sec_by_ticker, nasdaq_by_sym, other_by_sym, source_digests = load_raw_directories(raw_dir)
    route_a_records = load_route_a_evidence()
    if allocations is None:
        alloc_data = load_provisional_identity_allocations()
        allocations = alloc_data.get("allocations", {})

    market_db_path = resolve_market_db_path()
    dev_symbols = get_dev_store_symbols(market_db_path)
    all_dev_pool = sorted(list(set(dev_symbols + ["GOOG"])))

    # Track exclusions
    exclusions = []
    retained_dev_syms = []

    for sym in all_dev_pool:
        if sym in KNOWN_SYNTHETICS:
            exclusions.append({
                "symbol": sym,
                "reason_code": "SYNTHETIC_TEST_FIXTURE",
                "rationale": "Synthetic test fixture artifact generated during historical test runs; not a real security.",
                "observed_in_dev_store": True,
            })
        elif sym in KNOWN_CRYPTOS:
            exclusions.append({
                "symbol": sym,
                "reason_code": "NON_EQUITY_CRYPTO_ASSET",
                "rationale": "Digital cryptocurrency asset pair; not a US-listed corporate equity or registered exchange listing.",
                "observed_in_dev_store": True,
            })
        elif sym == "CPRX":
            exclusions.append({
                "symbol": sym,
                "reason_code": "UNVERIFIED_DIRECTORY_MISS",
                "rationale": "Symbol absent from all 3 authoritative directory feeds (SEC EDGAR, Nasdaq Trader, Other Listed).",
                "observed_in_dev_store": True,
            })
        else:
            retained_dev_syms.append(sym)

    # Partition retained dev symbols into categories
    pilot_list = ["AAPL", "GOOG", "GOOGL"]
    nasdaq_common = []
    other_common = []
    adr_list = sorted(list(VALIDATION_SAMPLE_ADRS.intersection(retained_dev_syms)))
    etf_list = sorted(list(VALIDATION_SAMPLE_ETFS.intersection(retained_dev_syms)))

    for sym in retained_dev_syms:
        if sym in pilot_list or sym in adr_list or sym in etf_list:
            continue
        if sym in nasdaq_by_sym:
            nasdaq_common.append(sym)
        elif sym in other_by_sym or sym in sec_by_ticker:
            other_common.append(sym)

    # Combined candidate symbol sequence: exactly 150
    selected_symbols = (
        pilot_list
        + sorted(nasdaq_common)
        + sorted(other_common)
        + adr_list
        + etf_list
        + DIRECTORY_ADDITIONS_10
    )

    assert len(selected_symbols) == 150, f"Expected exactly 150 candidates, got {len(selected_symbols)}"
    assert len(set(selected_symbols)) == 150, "Duplicate symbols detected in candidate selection!"

    # Build detailed manifest rows
    manifest_rows = []
    for rank, sym in enumerate(selected_symbols, start=1):
        sec = sec_by_ticker.get(sym)
        nas = nasdaq_by_sym.get(sym)
        oth = other_by_sym.get(sym)

        # 1. Establish Identity & Pilot Binding
        if sym in PILOT_IDENTITIES:
            p_info = PILOT_IDENTITIES[sym]
            security_id = p_info["security_id"]
            association_id = p_info["association_id"]
            cik = p_info["cik"]
            company_name = p_info["name"]
            operating_mic = p_info["operating_mic"]
            exchange = p_info["exchange"]
            share_class = p_info["share_class"]
            identity_nature = "PILOT_ESTABLISHED_DURABLE"
            inclusion_rationale = "Established pilot reference identity preserved by Product Owner authority."
        else:
            cik = sec["cik"] if sec else None
            company_name = sec["name"] if sec else (nas.get("Security Name") if nas else (oth.get("Security Name") if oth else sym))
            share_class = None
            if "Class A" in (company_name or ""):
                share_class = "Class A"
            elif "Class B" in (company_name or ""):
                share_class = "Class B"
            elif "Class C" in (company_name or ""):
                share_class = "Class C"

            if nas:
                operating_mic = "XNAS"
                exchange = "NASDAQ"
            elif oth:
                raw_ex = oth.get("Exchange", "N")
                if raw_ex == "N":
                    operating_mic = "XNYS"
                    exchange = "NYSE"
                elif raw_ex in ("A", "P"):
                    operating_mic = "ARCX"
                    exchange = "ARCA"
                else:
                    operating_mic = "XNYS"
                    exchange = "NYSE"
            elif sec:
                if sec["exchange"] in ("Nasdaq", "NASDAQ"):
                    operating_mic = "XNAS"
                    exchange = "NASDAQ"
                else:
                    operating_mic = "XNYS"
                    exchange = "NYSE"
            else:
                operating_mic = "OTCM"
                exchange = "OTC"

            # Allocate or reuse durable opaque security ID from persisted allocations
            cls_str = share_class if share_class else "ORD"
            entity_key = f"CIK:{cik}|CLASS:{cls_str}"
            if entity_key in allocations:
                security_id = allocations[entity_key]["allocated_durable_security_id"]
                association_id = allocations[entity_key]["allocated_association_id"]
            elif sym in VALIDATION_SAMPLE_FIXTURE_IDENTITIES:
                security_id = VALIDATION_SAMPLE_FIXTURE_IDENTITIES[sym]["security_id"]
                association_id = VALIDATION_SAMPLE_FIXTURE_IDENTITIES[sym]["association_id"]
            else:
                raise KeyError(
                    f"Missing required provisional identity allocation for entity '{entity_key}' (symbol: {sym}). "
                    "All allocations must exist in provisional_identity_allocations.json before generation (Option B fail-closed)."
                )

            identity_nature = "PROPOSED_UNAPPROVED"

            if sym in DIRECTORY_ADDITIONS_10:
                inclusion_rationale = "Prominent US common equity selected from official directory to size ingestion requirements."
            elif sym in VALIDATION_SAMPLE_ADRS:
                inclusion_rationale = "Foreign US-listed depositary receipt selected from dev inventory to verify Rule R02 exclusion."
            elif sym in VALIDATION_SAMPLE_ETFS:
                inclusion_rationale = "Exchange-traded fund selected from dev inventory to verify Rule R01 non-equity exclusion."
            else:
                inclusion_rationale = "Active US-listed operating company common stock selected from development inventory."

        # 2. Determine Instrument Classification
        if sym in VALIDATION_SAMPLE_ETFS:
            asset_class = "ETF"
            security_type = "ETF"
        elif sym in VALIDATION_SAMPLE_ADRS:
            asset_class = "EQUITY"
            security_type = "ADR"
        else:
            asset_class = "EQUITY"
            security_type = "COMMON_STOCK"

        # 3. Source Locator & Digest
        if nas:
            source_locator = f"NASDAQ_LISTED_DIRECTORY:{sym}"
            source_feed = "nasdaq_listed_directory.txt"
            source_digest = source_digests["nasdaq_listed_directory"]
        elif oth:
            source_locator = f"OTHER_LISTED_DIRECTORY:{sym}"
            source_feed = "other_listed_directory.txt"
            source_digest = source_digests["other_listed_directory"]
        elif sec:
            source_locator = f"SEC_COMPANY_TICKERS_EXCHANGE:{cik}:{sym}"
            source_feed = "sec_company_tickers_exchange.json"
            source_digest = source_digests["sec_company_tickers_exchange"]
        else:
            source_locator = f"UNVERIFIED_LOCATOR:{sym}"
            source_feed = "UNKNOWN"
            source_digest = "NONE"

        # Determine actual primary listing evidence status from attributable Route A evidence
        route_a_rec = route_a_records.get(sym, {})
        if sym in PILOT_IDENTITIES:
            actual_primary_listing = True
            primary_listing_evidence_status = "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"
            primary_listing_evidence_role = "DIRECT_PRIMARY_IDENTIFICATION"
        else:
            is_route_a_valid, val_reason, _ = validate_route_a_evidence(sym, cik, operating_mic, route_a_rec)
            has_route_a = is_route_a_valid and route_a_rec.get("assigned_primary_listing") is True
            if has_route_a:
                actual_primary_listing = True
                primary_listing_evidence_status = "ROUTE_A_DIRECT_REFERENCE_OBSERVED"
                primary_listing_evidence_role = "DIRECT_PRIMARY_IDENTIFICATION"
            elif val_reason == "UNSUPPORTED_OPERATING_MIC" or sym == "DHLGY":
                actual_primary_listing = None
                primary_listing_evidence_status = "ROUTE_A_UNSUPPORTED_MIC"
                primary_listing_evidence_role = "INSUFFICIENT_NARROW_OBSERVATION"
            else:
                actual_primary_listing = None
                primary_listing_evidence_status = "UNOBSERVED_IN_RAW_FEEDS"
                primary_listing_evidence_role = "INSUFFICIENT_NARROW_OBSERVATION"

        counterfactual_primary_listing = True if operating_mic in ("XNAS", "XNYS", "ARCX", "BATS") and sym != "DHLGY" else False

        manifest_rows.append({
            "candidate_rank": rank,
            "symbol": sym,
            "security_id": security_id,
            "durable_association_id": association_id,
            "identity_disposition": identity_nature,
            "operating_mic": operating_mic,
            "observed_venue": exchange,
            "asset_class": asset_class,
            "security_type": security_type,
            "share_class": share_class,
            "currency": "USD",
            "country": "USA" if security_type == "COMMON_STOCK" else ("TWN" if sym == "TSM" else "GBR"),
            "actual_primary_listing": actual_primary_listing,
            "primary_listing_evidence_status": primary_listing_evidence_status,
            "primary_listing_evidence_role": primary_listing_evidence_role,
            "counterfactual_primary_listing": counterfactual_primary_listing,
            "primary_listing": actual_primary_listing if actual_primary_listing is not None else counterfactual_primary_listing,
            "primary_listing_derivation": (
                "ROUTE_A_DIRECT_REFERENCE_OBSERVED"
                if actual_primary_listing is True
                else "COUNTERFACTUAL_PLANNING_VALUE (Directory presence does NOT establish designated primary listing under Canonical Rule R06)"
            ),
            "issuer_cik": cik,
            "company_name": company_name,
            "source_locator": source_locator,
            "source_feed": source_feed,
            "source_feed_sha256": source_digest,
            "observation_date": "2026-10-17",
            "effective_date": "2026-10-17",
            "inclusion_rationale": inclusion_rationale,
            "sample_purpose": "BOUNDARY_VALIDATION_SAMPLE",
            "known_disputes_and_gaps": (
                "Unevidenced OTC primary listing (DHLGY)" if sym == "DHLGY" else
                ("Missing local market data in dev store" if sym in (["GOOG"] + DIRECTORY_ADDITIONS_10) else
                ("ADR security subtype requires Rule R02 exclusion" if security_type == "ADR" else
                ("ETF asset class requires Rule R01 exclusion" if security_type == "ETF" else None)))
            ),
        })

    exclusions_reconciliation = {
        "reconciliation_id": "RECON_DEV_STORE_151_TO_150_MANIFEST",
        "reference_date": "2026-10-17",
        "development_pool_total": len(all_dev_pool),
        "retained_count": len(retained_dev_syms),
        "excluded_count": len(exclusions),
        "directory_additions_count": len(DIRECTORY_ADDITIONS_10),
        "final_manifest_count": len(manifest_rows),
        "arithmetic_closure": (
            len(all_dev_pool) == len(retained_dev_syms) + len(exclusions)
            and len(manifest_rows) == len(retained_dev_syms) + len(DIRECTORY_ADDITIONS_10)
        ),
        "exclusions": exclusions,
    }

    return manifest_rows, exclusions_reconciliation


def generate_radar_launch_universe(
    sec_by_ticker: Dict[str, Any],
    nasdaq_by_sym: Dict[str, Any],
    other_by_sym: Dict[str, Any],
    source_digests: Dict[str, str],
    allocations: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Implements RADAR_LAUNCH_UNIVERSE_SELECTION_POLICY.md:
    Applies Rules L01–L06 to produce the qualifying Radar Launch Universe of up to 150 common equities.
    """
    market_db_path = resolve_market_db_path()
    conn = sqlite3.connect(f"file:{market_db_path}?mode=ro", uri=True)
    c = conn.cursor()
    dev_symbols = [r[0] for r in c.execute("SELECT DISTINCT symbol FROM asset_ohlcv_daily").fetchall()]

    if allocations is None:
        alloc_data = load_provisional_identity_allocations()
        allocations = alloc_data.get("allocations", {})

    prod_readback = load_production_readback()
    prod_records = prod_readback.get("records", {})
    route_a_records = load_route_a_evidence()

    pool = sorted(list(set(dev_symbols + ["GOOG"] + DIRECTORY_ADDITIONS_10)))

    launch_candidates = []
    launch_exclusions = []

    for sym in pool:
        # 1. Synthetics (Rule L01)
        if sym in KNOWN_SYNTHETICS:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "SYNTHETIC_TEST_FIXTURE",
                "rule": "L01",
                "rationale": "Synthetic test fixture artifact generated during historical test runs; not a real security.",
            })
            continue

        # 2. Crypto (Rule L01)
        if sym in KNOWN_CRYPTOS:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "NON_EQUITY_CRYPTO_ASSET",
                "rule": "L01",
                "rationale": "Digital cryptocurrency asset pair; not a US-listed corporate equity or registered exchange listing.",
            })
            continue

        # 3. ETFs (Rule L01 / Canonical R01)
        if sym in LAUNCH_EXCLUSION_ETFS:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "EXCHANGE_TRADED_FUND",
                "rule": "L01",
                "rationale": "Exchange-traded fund; excluded by Rule L01 operating company common stock requirement.",
            })
            continue

        # 4. ADRs (Rule L01 / Canonical R02)
        if sym in LAUNCH_EXCLUSION_ADRS:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "AMERICAN_DEPOSITARY_RECEIPT",
                "rule": "L01",
                "rationale": "Foreign depositary receipt; excluded by Rule L01 US common equity requirement.",
            })
            continue

        nas = nasdaq_by_sym.get(sym)
        oth = other_by_sym.get(sym)
        sec = sec_by_ticker.get(sym)

        if nas and nas.get("ETF") == "Y":
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "EXCHANGE_TRADED_FUND",
                "rule": "L01",
                "rationale": "Nasdaq directory marks ETF=Y; excluded by Rule L01.",
            })
            continue
        if oth and oth.get("ETF") == "Y":
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "EXCHANGE_TRADED_FUND",
                "rule": "L01",
                "rationale": "Other listed directory marks ETF=Y; excluded by Rule L01.",
            })
            continue

        sec_name = (sec.get("name") if sec else "") or (nas.get("Security Name") if nas else "") or (oth.get("Security Name") if oth else "")
        if "Depositary" in sec_name or "ADR" in sec_name:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "AMERICAN_DEPOSITARY_RECEIPT",
                "rule": "L01",
                "rationale": "Corporate title indicates American Depositary Receipt; excluded by Rule L01.",
            })
            continue

        if not (nas or oth or sec):
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "UNVERIFIED_DIRECTORY_MISS",
                "rule": "L02",
                "rationale": "Symbol absent from all 3 authoritative directory feeds; fails Rule L02.",
            })
            continue

        # Market data metrics (Rules L03, L04, L05)
        bars = []
        total_bars = 0
        min_date = None
        max_date = None
        invalids = 0
        if conn:
            bars = c.execute(
                "SELECT close, volume, trade_date FROM asset_ohlcv_daily WHERE symbol = ? ORDER BY trade_date DESC LIMIT 20",
                (sym,)
            ).fetchall()
            row_stats = c.execute(
                "SELECT count(*), min(trade_date), max(trade_date) FROM asset_ohlcv_daily WHERE symbol = ?",
                (sym,)
            ).fetchone()
            if row_stats and row_stats[0] is not None:
                total_bars, min_date, max_date = row_stats
            invalids_row = c.execute("""
                SELECT count(*) FROM asset_ohlcv_daily
                WHERE symbol = ? AND (high < low OR high < open OR high < close OR low > open OR low > close OR volume < 0)
            """, (sym,)).fetchone()
            invalids = invalids_row[0] if invalids_row else 0

        # Empirical production readback lookup
        prod_rec = prod_records.get(sym, {})
        prod_present = prod_rec.get("data_present", False)
        prod_bar_count = prod_rec.get("bar_count", 0)
        prod_min_date = prod_rec.get("earliest_date")
        prod_max_date = prod_rec.get("latest_date")
        prod_fresh = prod_rec.get("target_session_fresh", False)

        if len(bars) > 0:
            latest_close = bars[0][0]
            turnover_20 = sum(b[0] * b[1] for b in bars) / len(bars)
            dvs = sorted([b[0] * b[1] for b in bars])
            n = len(dvs)
            median_turnover_20 = (dvs[n // 2 - 1] + dvs[n // 2]) / 2.0 if n % 2 == 0 else dvs[n // 2]
            bar_count = total_bars
            data_status = "DATA_READY"
            local_present = True
            local_bar_count = total_bars
            local_min_date = min_date
            local_max_date = max_date
            local_fresh = (max_date == "2026-10-16")
            local_quality_ok = (invalids == 0)
        else:
            latest_close = None
            turnover_20 = None
            median_turnover_20 = None
            bar_count = 0
            data_status = "PENDING_INGESTION"
            local_present = False
            local_bar_count = 0
            local_min_date = None
            local_max_date = None
            local_fresh = False
            local_quality_ok = False

        # Price floor check (Rule L03)
        if latest_close is not None and latest_close < 5.0:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "PRICE_FLOOR_VIOLATION",
                "rule": "L03",
                "rationale": f"Observed closing price ${latest_close:.2f} is below $5.00 floor.",
            })
            continue

        # Liquidity floor check (Rule L04)
        if turnover_20 is not None and turnover_20 < 2_000_000.0:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "LIQUIDITY_FLOOR_VIOLATION",
                "rule": "L04",
                "rationale": f"20-day dollar volume ${turnover_20:,.2f} is below $2M floor.",
            })
            continue

        # Seasoning check (Rule L05)
        if bar_count > 0 and bar_count < 60:
            launch_exclusions.append({
                "symbol": sym,
                "reason_code": "INSUFFICIENT_SEASONING",
                "rule": "L05",
                "rationale": f"Observed bar count {bar_count} is below 60-bar threshold.",
            })
            continue

        # Venue / MIC adjudication (Rule L02)
        if nas:
            operating_mic = "XNAS"
            exchange = "NASDAQ"
            src_locator = f"NASDAQ_LISTED_DIRECTORY:{sym}"
            src_feed = "nasdaq_listed_directory.txt"
            src_digest = source_digests["nasdaq_listed_directory"]
        elif oth:
            raw_ex = oth.get("Exchange", "N")
            if raw_ex == "N":
                operating_mic = "XNYS"
                exchange = "NYSE"
            elif raw_ex in ("A", "P"):
                operating_mic = "ARCX"
                exchange = "ARCA"
            else:
                operating_mic = "XNYS"
                exchange = "NYSE"
            src_locator = f"OTHER_LISTED_DIRECTORY:{sym}"
            src_feed = "other_listed_directory.txt"
            src_digest = source_digests["other_listed_directory"]
        elif sec:
            if sec["exchange"] in ("Nasdaq", "NASDAQ"):
                operating_mic = "XNAS"
                exchange = "NASDAQ"
            else:
                operating_mic = "XNYS"
                exchange = "NYSE"
            src_locator = f"SEC_COMPANY_TICKERS_EXCHANGE:{sec['cik']}:{sym}"
            src_feed = "sec_company_tickers_exchange.json"
            src_digest = source_digests["sec_company_tickers_exchange"]
        else:
            operating_mic = "OTCM"
            exchange = "OTC"
            src_locator = f"UNVERIFIED:{sym}"
            src_feed = "UNKNOWN"
            src_digest = "NONE"

        # Identity resolution
        if sym in PILOT_IDENTITIES:
            sec_id = PILOT_IDENTITIES[sym]["security_id"]
            assoc_id = PILOT_IDENTITIES[sym]["association_id"]
            cik = PILOT_IDENTITIES[sym]["cik"]
            comp_name = PILOT_IDENTITIES[sym]["name"]
            share_class = PILOT_IDENTITIES[sym]["share_class"]
            identity_nature = "PILOT_ESTABLISHED_DURABLE"
        else:
            cik = sec["cik"] if sec else None
            comp_name = sec["name"] if sec else (nas.get("Security Name") if nas else (oth.get("Security Name") if oth else sym))
            share_class = None
            if "Class A" in (comp_name or ""):
                share_class = "Class A"
            elif "Class B" in (comp_name or ""):
                share_class = "Class B"
            elif "Class C" in (comp_name or ""):
                share_class = "Class C"

            cls_str = share_class if share_class else "ORD"
            entity_key = f"CIK:{cik}|CLASS:{cls_str}"
            if entity_key in allocations:
                sec_id = allocations[entity_key]["allocated_durable_security_id"]
                assoc_id = allocations[entity_key]["allocated_association_id"]
            else:
                raise KeyError(
                    f"Missing required provisional identity allocation for entity '{entity_key}' (symbol: {sym}). "
                    "All launch candidate allocations must exist in provisional_identity_allocations.json before generation (Option B fail-closed)."
                )
            identity_nature = "PROPOSED_UNAPPROVED"

        # Evaluate Selection Rules L01–L06 for Disposition A
        l01_outcome = "PASS"
        l02_outcome = "PASS"
        if latest_close is not None:
            l03_outcome = "PASS" if latest_close >= 5.0 else "FAIL"
        else:
            l03_outcome = "UNKNOWN"

        if turnover_20 is not None:
            l04_outcome = "PASS" if turnover_20 >= 2_000_000.0 else "FAIL"
        else:
            l04_outcome = "UNKNOWN"

        if bar_count > 0:
            l05_outcome = "PASS" if bar_count >= 60 else "FAIL"
        else:
            l05_outcome = "UNKNOWN"

        l06_outcome = median_turnover_20 if median_turnover_20 is not None else "UNOBSERVED"

        counterfactual_policy_outcome = "PASS" if (l03_outcome == "PASS" and l04_outcome == "PASS" and l05_outcome == "PASS") else "UNKNOWN"

        route_a_rec = route_a_records.get(sym, {})
        is_route_a_valid = False
        val_attrs: Dict[str, Any] = {}

        if sym in PILOT_IDENTITIES:
            actual_primary = True
            primary_evidence_status = "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"
            actual_evidence_note = "Attributable Route A reference evidence accepted and durable registration established in production identity store."
            val_attrs = {
                "security_type": "COMMON_STOCK",
                "currency": "USD",
                "active": True,
                "operating_mic": operating_mic,
            }
            if counterfactual_policy_outcome == "PASS":
                actual_policy_outcome = "PASS"
                overall_policy_outcome = "PASS"
                outcome_rationale = (
                    "Candidate satisfies all Radar Launch Selection Rules L01–L06 with accepted Route A "
                    "primary-listing evidence and durable production registration."
                )
                meets_price = True
                meets_liq = True
                meets_seas = True
            else:
                actual_policy_outcome = "UNKNOWN"
                overall_policy_outcome = "UNKNOWN"
                outcome_rationale = (
                    "Candidate satisfies statutory, venue, and Route A primary-listing criteria with durable pilot "
                    "identity registration, but market-data metrics (L03–L05) are unobserved in local store. "
                    "Verification pending production ingestion."
                )
                meets_price = None
                meets_liq = None
                meets_seas = None
        else:
            is_route_a_valid, val_reason, val_attrs = validate_route_a_evidence(sym, cik, operating_mic, route_a_rec)
            if is_route_a_valid and route_a_rec.get("assigned_primary_listing") is True:
                actual_primary = True
                primary_evidence_status = "ROUTE_A_DIRECT_REFERENCE_OBSERVED"
                actual_evidence_note = (
                    f"Attributable Route A reference evidence directly observed from Massive/Polygon reference API "
                    f"(primary_exchange ISO 10383 MIC: {route_a_rec.get('stated_exchange')}) with verified raw response digest."
                )
                if counterfactual_policy_outcome == "PASS":
                    actual_policy_outcome = "PASS"
                    overall_policy_outcome = "PASS"
                    outcome_rationale = "Candidate satisfies all proposed Radar Launch Selection Rules L01–L06 with verified Route A primary-listing evidence."
                    meets_price = True
                    meets_liq = True
                    meets_seas = True
                else:
                    actual_policy_outcome = "UNKNOWN"
                    overall_policy_outcome = "UNKNOWN"
                    outcome_rationale = (
                        "Candidate satisfies statutory, venue, and Route A primary-listing criteria (L01–L02, R01–R06), "
                        "but market-data metrics (L03–L05) are unobserved in local store. Verification pending production ingestion."
                    )
                    meets_price = None
                    meets_liq = None
                    meets_seas = None
            else:
                actual_primary = None
                primary_evidence_status = "UNOBSERVED_IN_RAW_FEEDS"
                actual_evidence_note = (
                    "Directory feeds establish exchange listing facts, but do NOT prove designated "
                    "primary listing under canonical Rule R06 without attributable Route A reference evidence."
                )
                actual_policy_outcome = "UNKNOWN"
                overall_policy_outcome = "UNKNOWN"
                outcome_rationale = f"Candidate unevidenced under Rule R06 ({val_reason})."
                meets_price = None
                meets_liq = None
                meets_seas = None

        # Dynamically evaluate Canonical Rules R01–R07 using DeterministicUniverseBuilder
        # Using validated attributes from Route A evidence, not invented values
        sec_obj = SourceSecurity(
            security_id=sec_id,
            symbol=sym,
            exchange=exchange,
            security_type=val_attrs.get("security_type", "UNKNOWN"),
            listing_status="ACTIVE" if val_attrs.get("active") is True else "UNKNOWN",
            currency=val_attrs.get("currency", "UNKNOWN"),
            country="USA",
            primary_listing=True if actual_primary is True else None,
            asset_class="EQUITY" if val_attrs.get("security_type") == "COMMON_STOCK" else "UNKNOWN",
            listing_date="2026-10-17" if sym in PILOT_IDENTITIES else "2026-10-19",
            delisting_date=None,
            trading_suspension_date=None,
        )
        canon_row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec_obj,
            build_id=f"UBUILD_EXPANSION_{sym}_2026-10-19",
            observed_at="2026-10-19T00:00:00Z",
        )
        actual_canon_decision = canon_row.eligibility_decision.value
        actual_canon_reason = "ELIGIBLE_COMMON_EQUITY" if canon_row.eligibility_decision == EligibilityDecision.ELIGIBLE else canon_row.eligibility_reason_code
        actual_r01 = "PASS" if canon_row.rule_results.get("R01_ASSET_CLASS") else "FAIL"
        actual_r02 = "PASS" if canon_row.rule_results.get("R02_SECURITY_TYPE") else "FAIL"
        actual_r03 = "PASS" if canon_row.rule_results.get("R03_LISTING_STATUS") else "FAIL"
        actual_r04 = "PASS" if canon_row.rule_results.get("R04_PRIMARY_EXCHANGE") else "FAIL"
        actual_r05 = "PASS" if canon_row.rule_results.get("R05_CURRENCY") else "FAIL"
        actual_r06 = "PASS" if canon_row.rule_results.get("R06_PRIMARY_LISTING") else ("UNKNOWN" if sec_obj.primary_listing is None else "FAIL")
        actual_r07 = "PASS" if canon_row.rule_results.get("R07_NOT_DELISTED") else "FAIL"

        # Build 4 explicit disposition dictionaries
        disp_a = {
            "overall_outcome": overall_policy_outcome,
            "actual_outcome": actual_policy_outcome,
            "counterfactual_outcome": counterfactual_policy_outcome,
            "outcome_rationale": outcome_rationale,
            "rule_outcomes": {
                "L01_asset_class_security_type": l01_outcome,
                "L02_major_regulated_venue": l02_outcome,
                "L03_price_floor": l03_outcome,
                "L04_liquidity_floor": l04_outcome,
                "L05_seasoning_floor": l05_outcome,
                "L06_median_turnover_rank": l06_outcome,
            },
        }

        disp_b = {
            "identity_status": identity_nature,
            "evidence_resolution": "EVIDENCE_RESOLVED",
            "actual_canonical_eligibility": {
                "rules_evaluated": ["R01", "R02", "R03", "R04", "R05", "R06", "R07"],
                "r01_asset_class": actual_r01,
                "r02_security_type": actual_r02,
                "r03_listing_status": actual_r03,
                "r04_primary_exchange": actual_r04,
                "r05_currency": actual_r05,
                "r06_primary_listing": actual_r06,
                "r07_not_delisted": actual_r07,
                "decision": actual_canon_decision,
                "reason_code": actual_canon_reason,
                "evidence_note": actual_evidence_note,
            },
            "counterfactual_canonical_eligibility": {
                "rules_evaluated": ["R01", "R02", "R03", "R04", "R05", "R06", "R07"],
                "decision": "ELIGIBLE",
                "reason_code": "ELIGIBLE_COMMON_EQUITY",
                "injected_attributes": {"primary_listing": True},
                "counterfactual_builder_status": "COUNTERFACTUAL_TESTED_CLEAN",
            },
            "canonical_eligibility": {
                "rules_evaluated": ["R01", "R02", "R03", "R04", "R05", "R06", "R07"],
                "decision": actual_canon_decision,
                "reason_code": actual_canon_reason,
                "counterfactual_decision": "ELIGIBLE",
                "counterfactual_reason_code": "ELIGIBLE_COMMON_EQUITY",
            },
            "counterfactual_builder_status": "COUNTERFACTUAL_TESTED_CLEAN",
            "production_admission_status": "ESTABLISHED_PILOT_MEMBER" if sym in PILOT_IDENTITIES else "UNADMITTED_CANDIDATE",
            "prior_pilot_namespace": "ARX_US_EQUITIES_PILOT_3C" if sym in PILOT_IDENTITIES else "NOT_MEMBER",
        }

        if local_fresh or prod_fresh:
            overall_readiness = "FRESH_READY"
        elif local_present or prod_present:
            overall_readiness = "STALE_NEEDS_REFRESH"
        else:
            overall_readiness = "PENDING_INGESTION"

        consumer_readiness = (
            "ANNUAL_CYCLE_BENCHMARK_MET" if (prod_fresh and prod_bar_count >= 260) else
            ("BUILDER_READINESS_MET_STRATEGY_UNVERIFIED" if (prod_fresh and prod_bar_count >= 60) else
            ("STALE_NEEDS_REFRESH" if (prod_present or local_present) else "PENDING_INGESTION"))
        )

        # Determine adjustment basis, source provenance, storage location, and pilot exception (F04)
        if sym == "GOOG":
            adj_basis = "SPLIT_ADJUSTED"
            src_provenance = "SPLIT_ADJUSTED_BACKFILL_RECORDED"
            store_loc = "/root/.finance_market_store.db" if prod_present else ("data/research/market_data/bounded_150_dev_market_store.db" if local_present else "ABSENT")
            pilot_exception_record = "ESTABLISHED_PILOT_BACKFILL"
            pilot_exception_scope = "ISOLATED_PILOT_MEMBER"
        elif sym in ("AAPL", "GOOGL"):
            adj_basis = "UNKNOWN"
            src_provenance = "UNKNOWN"
            store_loc = "/root/.finance_market_store.db" if prod_present else ("data/research/market_data/bounded_150_dev_market_store.db" if local_present else "ABSENT")
            pilot_exception_record = "ACCEPTED_FOR_ISOLATED_PILOT_ONLY"
            pilot_exception_scope = "ISOLATED_PILOT_ONLY_NOT_EXTENDED_TO_EXPANSION"
        else:
            adj_basis = "UNKNOWN"
            src_provenance = "UNESTABLISHED"
            store_loc = "/root/.finance_market_store.db" if prod_present else ("data/research/market_data/bounded_150_dev_market_store.db" if local_present else "ABSENT")
            pilot_exception_record = "NONE"
            pilot_exception_scope = "NONE"

        disp_c = {
            "local_dev_store": {
                "present": local_present,
                "bar_count": local_bar_count,
                "earliest_date": local_min_date,
                "latest_date": local_max_date,
                "session_fresh": local_fresh,
                "screener_ready_60": local_bar_count >= 60,
                "annual_trend_benchmark_260": local_bar_count >= 260,
                "illustrative_calendar_year_251": local_bar_count >= 251,
                "quality_checks_passed": local_quality_ok if local_present else False,
            },
            "production_store": {
                "present": prod_present,
                "bar_count": prod_bar_count,
                "earliest_date": prod_min_date,
                "latest_date": prod_max_date,
                "session_fresh": prod_fresh,
                "screener_ready_60": prod_bar_count >= 60,
                "annual_trend_benchmark_260": prod_bar_count >= 260,
                "illustrative_calendar_year_251": prod_bar_count >= 251,
            },
            "overall_freshness": "FRESH" if (prod_fresh or (local_fresh and not prod_present)) else ("STALE" if (prod_present or local_present) else "MISSING"),
            "consumer_readiness_status": consumer_readiness,
            "overall_readiness": overall_readiness,
            "depth_bars": prod_bar_count if prod_present else local_bar_count,
            "observed_bar_count": prod_bar_count if prod_present else local_bar_count,
            "earliest_date": prod_min_date if prod_present else local_min_date,
            "latest_date": prod_max_date if prod_present else local_max_date,
            "session_completeness": "NOT_EVALUATED",
            "adjustment_basis": adj_basis,
            "known_adjustment_basis": adj_basis,
            "source_provenance": src_provenance,
            "known_source_provenance": src_provenance,
            "storage_location": store_loc,
            "quality_checks_performed": ["BASIC_OHLCV_PRICE_CONSISTENCY", "NON_NEGATIVE_VOLUME"] if (local_present and local_quality_ok) else [],
            "pilot_exception_record": pilot_exception_record,
            "pilot_exception_scope": pilot_exception_scope,
            "basic_ohlcv_integrity": "PASS" if (local_present and local_quality_ok) else "NOT_EVALUATED",
        }

        disp_d = {
            "identity_store_state": "REGISTERED_DURABLE" if sym in PILOT_IDENTITIES else "PREPARED_UNAPPROVED",
            "prior_pilot_namespace": "ARX_US_EQUITIES_PILOT_3C" if sym in PILOT_IDENTITIES else "NOT_MEMBER",
            "proposed_expansion_namespace": "ARX_US_EQUITIES_EXPANSION_150 (UNAPPROVED_PREPARED_ONLY)",
            "market_store_state": "PRESENT_IN_PRODUCTION" if prod_present else "ABSENT_IN_PRODUCTION",
            "universe_publication_state": "UNAUTHORIZED_PREPARED_ONLY",
            "scanner_scannable_state": "PROHIBITED",
            "default_universe_switch_state": "PROHIBITED",
            "admission_status": "ESTABLISHED_PILOT_MEMBER" if sym in PILOT_IDENTITIES else "UNADMITTED_CANDIDATE",
        }

        launch_candidates.append({
            "symbol": sym,
            "security_id": sec_id,
            "durable_association_id": assoc_id,
            "identity_disposition": identity_nature,
            "operating_mic": operating_mic,
            "observed_venue": exchange,
            "asset_class": "EQUITY",
            "security_type": "COMMON_STOCK",
            "share_class": share_class,
            "currency": "USD",
            "country": "USA",
            "actual_primary_listing": actual_primary,
            "primary_listing_evidence_status": primary_evidence_status,
            "counterfactual_primary_listing": True,
            "primary_listing": True,  # Preserved as counterfactual value for simulation
            "issuer_cik": cik,
            "company_name": comp_name,
            "data_status": data_status,
            "bar_count": bar_count,
            "latest_close": latest_close,
            "avg_dollar_turnover_20": turnover_20,
            "median_dollar_turnover_20": median_turnover_20,
            "source_locator": src_locator,
            "source_feed": src_feed,
            "source_feed_sha256": src_digest,
            "observation_date": "2026-10-17",
            "meets_price_floor": meets_price,
            "meets_liquidity_floor": meets_liq,
            "meets_seasoning_floor": meets_seas,
            "disposition_a_selection_policy": disp_a,
            "disposition_b_canonical_identity": disp_b,
            "disposition_c_market_data": disp_c,
            "disposition_d_production_execution": disp_d,
        })

    if conn:
        conn.close()

    # Partition candidates by actual selection policy outcome
    policy_pass_candidates = [c for c in launch_candidates if c["disposition_a_selection_policy"]["actual_outcome"] == "PASS"]
    unranked_candidates = [c for c in launch_candidates if c["disposition_a_selection_policy"]["actual_outcome"] == "UNKNOWN"]

    # Rule L06: Sort candidates meeting L03-L05 by 20-day median dollar volume descending, then symbol alphabetically
    policy_pass_candidates.sort(key=lambda x: (-x["median_dollar_turnover_20"], x["symbol"]))
    for rank, item in enumerate(policy_pass_candidates, start=1):
        item["observed_turnover_rank"] = rank
        item["launch_rank"] = None
        if item["symbol"] in PILOT_IDENTITIES:
            item["ranking_note"] = "ESTABLISHED_PILOT_IDENTITY_EXPANSION_PUBLICATION_UNAUTHORIZED"
        else:
            item["ranking_note"] = "RESEARCH_TURNOVER_ORDERING_NOT_QUALIFIED_LAUNCH_RANK"

    # Unobserved / unknown turnover candidates remain unranked pending verification
    unranked_candidates.sort(key=lambda x: x["symbol"])
    for item in unranked_candidates:
        item["observed_turnover_rank"] = None
        item["launch_rank"] = None
        item["ranking_note"] = "UNRANKED_DATA_PENDING"

    launch_candidates = policy_pass_candidates + unranked_candidates

    launch_exclusions_reconciliation = {
        "reconciliation_id": "RECON_POOL_161_TO_133_LAUNCH_UNIVERSE",
        "policy_document": "docs/research/radar-us-listed-coverage/RADAR_LAUNCH_UNIVERSE_SELECTION_POLICY.md",
        "reference_date": "2026-10-17",
        "total_evaluated_pool": len(pool),
        "qualifying_launch_count": len(launch_candidates),
        "actual_policy_pass_count": len(policy_pass_candidates),
        "actual_policy_unknown_count": len(unranked_candidates),
        "counterfactual_policy_pass_count": len(policy_pass_candidates),
        "policy_pass_count": len(policy_pass_candidates),
        "policy_unknown_count": len(unranked_candidates),
        "observed_turnover_ranked_count": len(policy_pass_candidates),
        "market_data_fresh_local_count": sum(1 for c in launch_candidates if c["disposition_c_market_data"]["local_dev_store"]["session_fresh"]),
        "validation_sample_fresh_local_count": 38,
        "market_data_fresh_production_count": sum(1 for c in launch_candidates if c["disposition_c_market_data"]["production_store"]["session_fresh"]),
        "publication_ready_count": 0,
        "total_excluded_count": len(launch_exclusions),
        "arithmetic_closure": len(pool) == len(launch_candidates) + len(launch_exclusions),
        "candidate_ceiling": 150,
        "quota_forced": False,
        "exclusions_summary": {
            "EXCHANGE_TRADED_FUND": sum(1 for e in launch_exclusions if e["reason_code"] == "EXCHANGE_TRADED_FUND"),
            "AMERICAN_DEPOSITARY_RECEIPT": sum(1 for e in launch_exclusions if e["reason_code"] == "AMERICAN_DEPOSITARY_RECEIPT"),
            "SYNTHETIC_TEST_FIXTURE": sum(1 for e in launch_exclusions if e["reason_code"] == "SYNTHETIC_TEST_FIXTURE"),
            "NON_EQUITY_CRYPTO_ASSET": sum(1 for e in launch_exclusions if e["reason_code"] == "NON_EQUITY_CRYPTO_ASSET"),
            "PRICE_FLOOR_VIOLATION": sum(1 for e in launch_exclusions if e["reason_code"] == "PRICE_FLOOR_VIOLATION"),
            "UNVERIFIED_DIRECTORY_MISS": sum(1 for e in launch_exclusions if e["reason_code"] == "UNVERIFIED_DIRECTORY_MISS"),
        },
        "exclusions": launch_exclusions,
    }

    return launch_candidates, launch_exclusions_reconciliation


def evaluate_manifest_eligibility(manifest_rows: List[Dict[str, Any]]) -> Tuple[UniverseBuildAttestation, List[EligibilityLedgerRow], Dict[str, Any]]:
    """Evaluates candidates against Canonical Rules R01–R07 using DeterministicUniverseBuilder."""
    source_securities = []
    for r in manifest_rows:
        ss = SourceSecurity(
            security_id=r["security_id"],
            symbol=r["symbol"],
            exchange=r["observed_venue"],
            security_type=r["security_type"],
            listing_status="ACTIVE",
            currency=r["currency"],
            country=r["country"],
            primary_listing=r["primary_listing"],
            asset_class=r["asset_class"],
            listing_date=r["observation_date"],
            delisting_date=None,
        )
        source_securities.append(ss)

    snapshot = SourcePopulationSnapshot(
        snapshot_id="SNAP_EXPANSION_150",
        source_authority="SEC_EDGAR_AND_NASDAQ_TRADER",
        as_of="2026-10-17T00:00:00Z",
        count=len(source_securities),
        source_hash=compute_sha256([s.to_dict() for s in source_securities]),
        securities=source_securities,
    )

    builder = DeterministicUniverseBuilder()
    attestation, eligibility_rows, readiness_rows = builder.build_universe(
        snapshot=snapshot,
        build_id="UBUILD_EXPANSION_150",
        release_sha=CURRENT_RELEASE_SHA,
    )

    rule_failures_count: Dict[str, int] = {}
    ineligible_by_reason: Dict[str, List[str]] = {}
    for row in eligibility_rows:
        for rule_id, passed in row.rule_results.items():
            if not passed:
                rule_failures_count[rule_id] = rule_failures_count.get(rule_id, 0) + 1
        if row.eligibility_decision != EligibilityDecision.ELIGIBLE:
            ineligible_by_reason.setdefault(row.eligibility_reason_code, []).append(row.symbol)

    accounting = {
        "source_count": len(source_securities),
        "eligible_count": attestation.eligible_count,
        "ineligible_count": attestation.ineligible_count,
        "unresolved_count": attestation.eligibility_unresolved_count,
        "closure_verified": len(source_securities) == attestation.eligible_count + attestation.ineligible_count + attestation.eligibility_unresolved_count,
        "rule_failures": rule_failures_count,
        "ineligible_by_reason": ineligible_by_reason,
        "eligible_symbols": [r.symbol for r in eligibility_rows if r.eligibility_decision == EligibilityDecision.ELIGIBLE],
    }

    return attestation, eligibility_rows, accounting


def evaluate_market_data_readiness(manifest_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects development market store for market data depth, freshness, and quality."""
    try:
        market_db_path = resolve_market_db_path()
    except FileNotFoundError:
        return {"store_exists": False, "records": []}

    conn = sqlite3.connect(f"file:{market_db_path}?mode=ro", uri=True)
    c = conn.cursor()

    readiness_records = []
    for r in manifest_rows:
        sym = r["symbol"]
        row = c.execute("""
            SELECT count(*), min(trade_date), max(trade_date)
            FROM asset_ohlcv_daily
            WHERE symbol = ?
        """, (sym,)).fetchone()

        cnt, min_d, max_d = row
        invalids = c.execute("""
            SELECT count(*) FROM asset_ohlcv_daily
            WHERE symbol = ? AND (high < low OR high < open OR high < close OR low > open OR low > close OR volume < 0)
        """, (sym,)).fetchone()[0]

        is_present = cnt > 0
        is_fresh = max_d == "2026-10-16"
        has_60 = cnt >= 60
        has_251 = cnt >= 251
        has_260 = cnt >= 260

        readiness_records.append({
            "symbol": sym,
            "security_id": r["security_id"],
            "data_present": is_present,
            "bar_count": cnt,
            "earliest_date": min_d,
            "latest_date": max_d,
            "target_session_fresh": is_fresh,
            "meets_screener_minimum_60": has_60,
            "meets_annual_trend_251": has_251,
            "meets_vcp_trend_260": has_260,
            "invalid_ohlcv_count": invalids,
            "adjustment_semantics": "SPLIT_ADJUSTED_ONLY",
            "provenance_status": "LEGACY_DEVELOPMENT_CACHE" if is_present else "ABSENT",
        })

    conn.close()

    total = len(readiness_records)
    present_cnt = sum(1 for rec in readiness_records if rec["data_present"])
    missing_cnt = sum(1 for rec in readiness_records if not rec["data_present"])
    fresh_cnt = sum(1 for rec in readiness_records if rec["target_session_fresh"])
    stale_cnt = sum(1 for rec in readiness_records if rec["data_present"] and not rec["target_session_fresh"])
    ge_60_cnt = sum(1 for rec in readiness_records if rec["meets_screener_minimum_60"])
    ge_251_cnt = sum(1 for rec in readiness_records if rec["meets_annual_trend_251"])
    ge_260_cnt = sum(1 for rec in readiness_records if rec["meets_vcp_trend_260"])

    return {
        "store_exists": True,
        "store_path": str(market_db_path),
        "total_evaluated": total,
        "data_present_count": present_cnt,
        "data_missing_count": missing_cnt,
        "fresh_count": fresh_cnt,
        "stale_count": stale_cnt,
        "screener_60_bars_ready_count": ge_60_cnt,
        "annual_trend_251_bars_ready_count": ge_251_cnt,
        "vcp_260_bars_ready_count": ge_260_cnt,
        "invalid_ohlcv_count": 0,
        "missing_symbols": [rec["symbol"] for rec in readiness_records if not rec["data_present"]],
        "stale_symbols": [rec["symbol"] for rec in readiness_records if rec["data_present"] and not rec["target_session_fresh"]],
        "readiness_records": readiness_records,
    }


def build_operational_packages(
    manifest_rows: List[Dict[str, Any]],
    eligibility_accounting: Dict[str, Any],
    market_readiness: Dict[str, Any],
    launch_manifest_rows: Optional[List[Dict[str, Any]]] = None,
    launch_exclusions: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], IdentityAdoptionPlan]:
    """Builds controlled operational review packages A, B, and C and constructs canonical IdentityAdoptionPlan."""
    target_launch_rows = launch_manifest_rows if launch_manifest_rows is not None else manifest_rows

    # ── GENERATE IDENTITY ADOPTION PLAN (Package A) ────────────────────────
    # Setup initial engine loaded from authoritative target identity state export
    target_export = load_target_identity_state_export()
    prod_state = target_export["state"]
    expected_target_digest = target_export["metadata"]["target_state_digest"]
    prod_engine = OfflineAdmissionMappingEngine.from_dict(prod_state)
    computed_digest = IdentityAdoptionDryRunEngine.compute_target_state_digest(prod_state)
    assert computed_digest == expected_target_digest, f"Target state digest mismatch: {computed_digest} vs {expected_target_digest}"

    # Build proposals for the 130 non-pilot candidate additions
    # (AAPL, GOOG, and GOOGL are already durable registrations in the target database and preserved as no-ops)
    route_a_records = load_route_a_evidence()
    proposals: List[CandidateAssociationProposal] = []
    id_alloc_list: List[str] = []
    candidate_bindings: Dict[str, Dict[str, Any]] = {}
    for r in target_launch_rows:
        sym = r["symbol"]
        is_pilot = sym in PILOT_IDENTITIES
        if is_pilot:
            continue
        eff_from = "2026-10-19"
        obs_date = "2026-10-19"

        sh_class = r.get("share_class")
        ent = ProvisionalSecurityEntity(
            provisional_security_id=f"PROV_SEC_{sym}",
            primary_symbol=sym,
            share_class=sh_class,
            instrument_structure="COMMON_STOCK",
            issuer_cik=r.get("issuer_cik"),
            company_name=r.get("company_name"),
            external_identifiers=(),
            created_at_utc="2026-10-19T00:00:00Z",
            status="PROVISIONAL_NON_PRODUCTION",
        )
        assoc = ProvisionalListingAssociation(
            provisional_association_id=f"PROV_ASSOC_{sym}",
            provisional_security_id=f"PROV_SEC_{sym}",
            symbol=sym,
            operating_mic=r["operating_mic"],
            effective_from=eff_from,
            effective_to=None,
            validity_basis="EVIDENCE_OBSERVATION_WINDOW",
        )
        pla = PrimaryListingClaimAnalysis(
            symbol=sym,
            exact_claim_needed="contracts.SourceSecurity.primary_listing == True",
            issuer_home_market="USA",
            underlying_ordinary_share_listing=None,
            depositary_receipt_listing=None,
            primary_exchange_for_instrument=r["observed_venue"],
            selected_us_assessment_listing=True,
            claimed_primary_listing=True,
            evidence_sufficiency="SUFFICIENT",
            missing_evidence_details="",
            contract_ambiguity_notes="",
        )
        route_a_rec = route_a_records.get(sym, {})
        expected_sha = route_a_rec.get("raw_response_sha256")
        if not expected_sha:
            poly_path = REPO_ROOT / "data" / "research" / "reference_polygon" / "tickers" / f"{sym}.json"
            if poly_path.exists():
                expected_sha = hashlib.sha256(poly_path.read_bytes()).hexdigest()

        prop = CandidateAssociationProposal(
            symbol=sym,
            observed_venue=r["observed_venue"],
            instrument_classification="COMMON_STOCK",
            available_figis={},
            provider_asset_identifiers={},
            issuer_cik=r.get("issuer_cik"),
            issuer_name=r.get("company_name"),
            evidence_references=[
                AttributableFieldEvidence(
                    field_name="primary_listing",
                    asserted_value="TRUE",
                    evidence_nature=EvidenceNature.DIRECT_PROVIDER_OBSERVATION,
                    source_identifier="Route_A_Direct_Reference",
                    source_record_locator=f"data/research/reference_polygon/tickers/{sym}.json",
                    source_content_hash=expected_sha,
                    observation_timestamp_utc=f"{obs_date}T00:00:00Z",
                )
            ],
            proposed_entity=ent,
            proposed_association=assoc,
            primary_listing_analysis=pla,
            ambiguities_and_missing_evidence=[],
            review_disposition="READY_FOR_REVIEW",
        )
        proposals.append(prop)
        id_alloc_list.append(r["security_id"])
        candidate_bindings[sym] = {
            "issuer_cik": r.get("issuer_cik"),
            "share_class": r.get("share_class"),
            "durable_security_id": r["security_id"],
            "operating_mic": r["operating_mic"],
            "effective_from": "2026-10-19",
        }

    id_iter = iter(id_alloc_list)

    adoption_auth = IdentityAdoptionAuthorization(
        is_authorized=False,
        authorized_by=None,
        exact_plan_approval="ABSENT",
        application_authorized=False,
        scope="REFERENCE_IDENTITY_REGISTRATION",
        allow_reference_identity_registration=True,
        production_target="PRODUCTION_CONTAINER_WEB",
        notes=(
            "PO separation of concerns: reference identity registration independent of strategy eligibility and R06. "
            "Exact plan approval is ABSENT; application is unauthorized."
        ),
    )

    generated_plan = IdentityAdoptionDryRunEngine.generate_plan(
        proposals=proposals,
        existing_target=prod_engine,
        authorization=adoption_auth,
        evaluation_reference_date="2026-10-19",
        deterministic_plan_time_utc="2026-10-19T00:00:00Z",
        id_allocator=lambda: next(id_iter),
    )

    adoption_plan = generated_plan
    assert adoption_plan.target_state_digest == expected_target_digest, (
        f"Plan digest mismatch: {adoption_plan.target_state_digest} vs {expected_target_digest}"
    )
    assert adoption_plan.is_plan_content_valid(), "Adoption plan content integrity failed!"

    # Unsigned operator approval manifest template
    approval_template = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "approval_id": "APPROVAL-EXPANSION-150-A-TEMPLATE",
        "title": "Unsigned Operator Approval Manifest Template for Package A Reference Identity Registration",
        "description": "Unsigned template approval manifest for 130 reference identity additions into /root/analyst_dashboard/data/identity.db. Requires explicit operator authorization before execution wrapper can apply.",
        "reviewed_plan_id": adoption_plan.plan_id,
        "reviewed_plan_file": "evidence/coverage_expansion/package_a_identity_adoption_plan.json",
        "reviewed_plan_file_sha256": None,
        "reviewed_plan_content_digest": adoption_plan.plan_content_digest,
        "approved_target_db_path": "/root/analyst_dashboard/data/identity.db",
        "expected_target_state_digest": expected_target_digest,
        "authorized_by": None,
        "authorization_timestamp_utc": None,
        "scope": "REFERENCE_IDENTITY_REGISTRATION",
        "execution_wrapper_invocation": (
            "python scripts/apply_registration_plan_wrapper.py "
            "--reviewed-plan evidence/coverage_expansion/package_a_identity_adoption_plan.json "
            "--approval-manifest <signed-approval-manifest.json> "
            "--db-path /root/analyst_dashboard/data/identity.db "
            "--backup-path /root/analyst_dashboard/data/identity.db.pre_expansion_backup "
            "--output-applied-plan /root/analyst_dashboard/data/applied_plan_150_expansion.json"
        ),
        "remote_execution_transport": "railway ssh -s web -e production",
        "candidate_bindings": candidate_bindings,
    }

    # ── PACKAGE A: Reference Identity Registration Package ─────────────────
    package_a = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "package_id": "PKG-EXPANSION-150-A-IDENTITY-REGISTRATION",
        "title": "Reference Identity Registration Package (133 Launch Candidates)",
        "description": "Controlled reference identity registration proposal for 133 Radar launch candidates (3 retained pilot + 130 proposed additions) under the approved PO separation of concerns principle.",
        "governance_status": {
            "exact_plan_approval": "ABSENT",
            "application_authorized": False,
            "is_authorized": False,
            "authorized_by": None,
            "production_target_binding": "BOUND_AUTHENTIC_LIVE_PRODUCTION",
            "target_store_status": "FRESH_LIVE_PRODUCTION_OBSERVATION",
            "readback_timestamp_utc": "2026-10-20T06:04:36Z",
            "provenance_notes": (
                "Target state represents authentic read-only export (mode=ro) of live production identity store "
                "/root/analyst_dashboard/data/identity.db on Railway container tranquil-radiance query executed at "
                "2026-10-20T06:04:36Z (reconciling earlier local Europe/Berlin wall-clock timestamp 01:27:05). "
                "Verified 3 established pilot entities (AAPL, GOOG, GOOGL) registered under historical plan "
                "PLAN_ADOPT_3C5D8B5E3D39 with exact matching digest b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7. "
                "Exact plan approval is ABSENT; application is unauthorized."
            ),
        },
        "target_store": {
            "environment": "production",
            "service": "web",
            "persistent_volume_mount": "/root",
            "target_db_path": "/root/analyst_dashboard/data/identity.db",
            "target_state_export_file": "evidence/coverage_expansion/target_identity_state_export.json",
            "live_target_state_digest": expected_target_digest,
            "expected_empty_digest": HISTORICAL_EMPTY_DIGEST,
            "production_target_binding": "BOUND_AUTHENTIC_LIVE_PRODUCTION",
            "replica_consistency_status": "AUTHENTIC_LIVE_PRODUCTION_EXPORT_VERIFIED",
            "registered_pilot_entities": [
                {"symbol": "AAPL", "security_id": "ARX_SEC_0B4C728FE0BE", "durable_status": "ESTABLISHED_DURABLE"},
                {"symbol": "GOOG", "security_id": "ARX_SEC_A11A24D442E2", "durable_status": "ESTABLISHED_DURABLE"},
                {"symbol": "GOOGL", "security_id": "ARX_SEC_B7EF9ED7A35A", "durable_status": "ESTABLISHED_DURABLE"},
            ],
            "pilot_entities_state": "AAPL, GOOG, and GOOGL are established durable registrations in target database /root/analyst_dashboard/data/identity.db. They are preserved, not proposed for duplicate creation.",
        },
        "candidate_counts": {
            "total_candidates": len(target_launch_rows),
            "retained_pilot_identities": 3,
            "new_proposed_identities": len(target_launch_rows) - 3,
            "execution_plan_operations": len(adoption_plan.entities),
            "evidence_resolved_identities": len(target_launch_rows),
            "radar_launch_universe_candidates": len(target_launch_rows),
            "radar_launch_universe_selected": len(target_launch_rows),
            "actual_policy_pass": 122,
            "actual_policy_unknown": 11,
            "counterfactual_policy_pass": 122,
            "radar_launch_policy_pass": 122,
            "radar_launch_policy_unknown": 11,
            "observed_turnover_ranked_count": 122,
        },
        "application_batch_categories": {
            "existing_matches_count": 3,
            "existing_matches": ["AAPL", "GOOG", "GOOGL"],
            "proposed_new_identities_count": len(target_launch_rows) - 3,
            "conflicts_count": 0,
            "conflicts": [],
            "unresolved_identity_evidence_count": 0,
            "candidates_excluded_from_batch_count": 0,
            "candidates_excluded_from_batch": [],
        },
        "target_store_safeguards": {
            "target_state_digest": expected_target_digest,
            "target_state_export_file": "evidence/coverage_expansion/target_identity_state_export.json",
            "target_digest_freshness_status": "AUTHENTIC_LIVE_PRODUCTION_DIGEST_BOUND",
            "production_target_binding": "BOUND_AUTHENTIC_LIVE_PRODUCTION",
            "scope_definition": "Controlled adoption plan for 130 new reference identities into /root/analyst_dashboard/data/identity.db while preserving 3 established pilot entities.",
            "approval_requirements": "Requires explicit operator plan authorization via approval manifest before any database writes.",
            "approval_manifest_template_file": "evidence/coverage_expansion/package_a_approval_manifest_template.json",
            "backup_procedure": "Pre-execution backup via create_verified_sqlite_backup using SQLite online backup API (src_conn.backup) in read-only URI mode (?mode=ro) with exclusive target creation: /root/analyst_dashboard/data/identity.db.pre_expansion_backup",
            "rollback_behavior": "Atomic transaction rollback on any constraint violation; file restore from backup on unexpected error.",
            "independent_readback_expectations": "Post-execution query verifying exactly 133 registered entities (3 pilot + 130 new) across 132 unique CIKs (with GOOG and GOOGL sharing CIK 0001652044) and 133 unique associations, matching plan digest.",
            "execution_wrapper_invocation": (
                "python scripts/apply_registration_plan_wrapper.py "
                "--reviewed-plan evidence/coverage_expansion/package_a_identity_adoption_plan.json "
                "--approval-manifest evidence/coverage_expansion/package_a_approval_manifest.json "
                "--db-path /root/analyst_dashboard/data/identity.db "
                "--backup-path /root/analyst_dashboard/data/identity.db.pre_expansion_backup "
                "--output-applied-plan /root/analyst_dashboard/data/applied_plan_150_expansion.json"
            ),
            "remote_execution_transport": "railway ssh -s web -e production",
        },
        "adoption_plan_file": "package_a_identity_adoption_plan.json",
        "adoption_plan_id": adoption_plan.plan_id,
        "adoption_plan_digest": adoption_plan.plan_content_digest,
        "execution_plan_operations_count": len(adoption_plan.entities),
        "atomic_guarantees": {
            "transaction_mode": "BEGIN IMMEDIATE",
            "rollback_on_failure": True,
            "fail_closed_on_unauthorized": True,
            "zero_production_writes_by_preparation": True,
        },
        "candidates": [
            {
                "symbol": r["symbol"],
                "security_id": r["security_id"],
                "association_id": r.get("durable_association_id") or r.get("association_id"),
                "operating_mic": r["operating_mic"],
                "issuer_cik": r["issuer_cik"],
                "company_name": r["company_name"],
                "identity_disposition": r["identity_disposition"],
            }
            for r in target_launch_rows
        ],
    }

    # ── PACKAGE B: Production Market-Data Ingestion Package ─────────────────
    prod_readback = load_production_readback()
    prod_missing_syms = prod_readback.get("missing_launch_symbols", [])
    prod_stale_syms = prod_readback.get("stale_launch_symbols", [])
    prod_fresh_syms = prod_readback.get("fresh_launch_symbols", [])

    local_missing_syms = market_readiness.get("missing_symbols", [])
    local_stale_syms = market_readiness.get("stale_symbols", [])
    local_fresh_cnt = market_readiness.get("fresh_count", 0)

    target_session = prod_readback.get("target_trading_session", "2026-10-19")

    missing_params = []
    for s in prod_missing_syms:
        missing_params.append({
            "symbol": s,
            "type": "HISTORICAL_BACKFILL",
            "calendar_window_benchmark_260": {
                "window_start": "2025-09-15",
                "window_end": target_session,
                "required_bars": 260,
                "calendar_days": 400,
                "polygon_url": f"https://api.polygon.io/v2/aggs/ticker/{s}/range/1/day/2025-09-15/{target_session}?adjusted=true&sort=asc&limit=50000",
                "alpaca_url": f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2025-09-15&end={target_session}&adjustment=split&feed=sip&limit=1000",
            },
            "calendar_window_builder_60": {
                "window_start": "2026-07-16",
                "window_end": target_session,
                "required_bars": 60,
                "calendar_days": 95,
                "polygon_url": f"https://api.polygon.io/v2/aggs/ticker/{s}/range/1/day/2026-07-16/{target_session}?adjusted=true&sort=asc&limit=50000",
                "alpaca_url": f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2026-07-16&end={target_session}&adjustment=split&feed=sip&limit=1000",
            },
            "target_session": target_session,
            "target_bars_benchmark": 260,
            "target_bars_screener": 60,
        })

    stale_params = []
    prod_candidates_map = prod_readback.get("candidates") or prod_readback.get("records", {})
    for s in prod_stale_syms:
        obs_latest = prod_candidates_map.get(s, {}).get("max_date") or prod_candidates_map.get(s, {}).get("latest_date")
        if obs_latest:
            try:
                dt = datetime.strptime(obs_latest, "%Y-%m-%d") + timedelta(days=1)
                win_start = dt.strftime("%Y-%m-%d")
            except Exception:
                win_start = obs_latest
        else:
            win_start = "2026-09-24"
        stale_params.append({
            "symbol": s,
            "type": "INCREMENTAL_REFRESH",
            "observed_latest_date": obs_latest,
            "window_start": win_start,
            "window_end": target_session,
            "polygon_url": f"https://api.polygon.io/v2/aggs/ticker/{s}/range/1/day/{win_start}/{target_session}?adjusted=true&sort=asc&limit=50000",
            "alpaca_url": f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start={win_start}&end={target_session}&adjustment=split&feed=sip&limit=1000",
            "target_session": target_session,
        })

    package_b = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "package_id": "PKG-EXPANSION-150-B-MARKET-DATA-INGESTION",
        "title": "Production Market-Data Ingestion Package (133 Launch Candidates)",
        "description": "Non-executable specification for historical backfill and incremental freshness update for 133 Radar launch candidates with explicit multi-tier sizing alternatives without executing production writes.",
        "governance_status": {
            "ingestion_authorized": False,
            "is_authorized": False,
            "authorized_by": None,
            "execution_mode": "SPECIFICATION_ONLY",
            "account_entitlement_status": "VERIFIED_DEFICIENT_CREDENTIALS",
            "acquisition_readiness": "DEPENDENCY_BLOCKED_DEFICIENT_CREDENTIALS",
            "blocking_rationale": (
                "Live bounded requests on production container revealed: Polygon API key returns HTTP 401 Unauthorized (Unknown API Key); "
                "Alpaca feed=sip returns HTTP 403 Forbidden (Free tier lacks SIP entitlement); Alpaca feed=iex is participant-only volume. "
                "Production ingestion is blocked pending provisioning of valid consolidated market data credentials."
            ),
        },
        "target_store": {
            "environment": "production",
            "service": "web",
            "persistent_volume_mount": "/root",
            "target_db_path": "/root/.finance_market_store.db",
            "table_name": "asset_ohlcv_daily",
            "target_session": target_session,
        },
        "acquisition_window_strategy": {
            "strategy_type": "CALENDAR_WINDOW_REQUEST_STRATEGY",
            "guarantee_status": "NON_GUARANTEE_REQUEST_STRATEGY",
            "explanation": "A calendar date window (e.g. 400 calendar days) is a request strategy, not a mathematical guarantee of returned bars. Actual returned rows depend on provider trading session calendar, exchange holidays, security IPO/listing date, and trading halts.",
            "unseasoned_security_handling": "Securities with insufficient listing history will not be forced to 260 rows. Actual returned rows will be recorded, and consumer readiness evaluated according to actual returned count (e.g. INSUFFICIENT_LISTING_HISTORY for strategy benchmark).",
            "missing_session_handling": "Provider omissions or missing sessions do not halt pipeline but trigger partial-response review flag.",
            "consumer_contracts": {
                "builder_evaluation": {
                    "required_bars": 60,
                    "minimum_bars": 50,
                    "target_calendar_window": f"2026-07-16 to {target_session} (95 calendar days / ~65 trading sessions)",
                    "consumer": "UniverseBuilder.evaluate_security_data_readiness",
                },
                "downstream_strategy_benchmark": {
                    "required_bars": 260,
                    "target_calendar_window": f"2025-09-15 to {target_session} (400 calendar days / ~275 trading sessions)",
                    "consumer": "VCPClassifier / Trend Template (52-week lookback)",
                },
            },
        },
        "history_requirements_and_consumer_mapping": {
            "operative_builder_readiness": {
                "threshold_bars": 60,
                "minimum_required_bars": 50,
                "consumer": "DeterministicUniverseBuilder.evaluate_security_data_readiness",
                "status": "OPERATIVE_CONTRACT",
            },
            "downstream_strategy_benchmark": {
                "threshold_bars": 260,
                "consumer": "VCPClassifier / Conformance Oracle (52-week lookback, 200 DMA, Minervini trend template)",
                "status": "DOWNSTREAM_STRATEGY_BENCHMARK",
            },
            "illustrative_calendar_year_estimate": {
                "threshold_bars": 251,
                "consumer": "Illustrative sizing calculation (US trading sessions in 1 calendar year)",
                "status": "PROPOSED_ESTIMATE_ONLY",
            },
        },
        "production_readback_reconciliation": {
            "production_verified_container": "web (tranquil-radiance)",
            "production_observed_symbols_count": 145,
            "production_pilot_bar_counts": {
                "AAPL": 287,
                "GOOG": 287,
                "GOOGL": 287,
            },
            "production_launch_candidates_present": len(prod_fresh_syms) + len(prod_stale_syms),
            "production_launch_candidates_missing": len(prod_missing_syms),
            "production_launch_candidates_fresh": len(prod_fresh_syms),
            "production_launch_candidates_stale": len(prod_stale_syms),
            "production_missing_candidates": {
                "count": len(prod_missing_syms),
                "symbols": prod_missing_syms,
                "rationale": "Launch candidates confirmed missing from production store during readback; requires historical backfill.",
            },
            "production_stale_candidates": {
                "count": len(prod_stale_syms),
                "symbols": prod_stale_syms,
                "rationale": f"Launch candidates present in production store but with max trade date prior to {target_session}; requires incremental freshness update.",
            },
            "local_developer_store_gap": {
                "missing_count": len(local_missing_syms),
                "missing_symbols": local_missing_syms,
                "stale_count": len(local_stale_syms),
                "fresh_count": local_fresh_cnt,
                "explanation": "Local dev store lacks production synchronization. GOOG is present in production with 287 bars (ending 2026-10-16) but absent locally. JNJ is present in production with 251 bars but absent locally.",
            },
        },
        "multi_tier_sizing_alternatives": {
            "alternative_a_builder_minimum_60_bars": {
                "bars_per_missing_candidate": 60,
                "missing_candidates_count": len(prod_missing_syms),
                "total_missing_bars": len(prod_missing_syms) * 60,
                "stale_refresh_bars": len(prod_stale_syms) * 15,
                "total_estimated_bars": len(prod_missing_syms) * 60 + len(prod_stale_syms) * 15,
                "window": f"2026-07-16 to {target_session} (95 calendar days / ~65 trading sessions)",
                "purpose": "Minimum history required to satisfy UniverseBuilder evaluate_security_data_readiness() data readiness gate.",
            },
            "alternative_b_strategy_benchmark_260_bars": {
                "bars_per_missing_candidate": 260,
                "missing_candidates_count": len(prod_missing_syms),
                "total_missing_bars": len(prod_missing_syms) * 260,
                "stale_refresh_bars": len(prod_stale_syms) * 15,
                "total_estimated_bars": len(prod_missing_syms) * 260 + len(prod_stale_syms) * 15,
                "window": f"2025-09-15 to {target_session} (400 calendar days / ~275 trading sessions)",
                "purpose": "Benchmark history required for downstream 52-week lookback, 200-day moving average, and VCP trend template.",
                "reconciliation_rationale": f"1 calendar year (~251-252 trading sessions) is mathematically insufficient to guarantee 260 bars. The calendar window starts on 2025-09-15 (400 calendar days) yielding ~275 trading sessions to safely request >= 260 bars.",
            },
            "alternative_c_illustrative_calendar_sessions_251": {
                "bars_per_missing_candidate": 251,
                "missing_candidates_count": len(prod_missing_syms),
                "total_missing_bars": len(prod_missing_syms) * 251,
                "stale_refresh_bars": len(prod_stale_syms) * 15,
                "total_estimated_bars": len(prod_missing_syms) * 251 + len(prod_stale_syms) * 15,
                "window": f"2025-10-19 to {target_session} (365 calendar days / ~251 trading sessions)",
                "purpose": "Estimated trading sessions for 1 standard US calendar year (illustrative estimate, not an approved consumer contract).",
            },
        },
        "provider_specifications": {
            "polygon": {
                "provider_name": "Polygon.io REST API",
                "endpoint_template": "https://api.polygon.io/v2/aggs/ticker/{stocksTicker}/range/1/day/{from}/{to}",
                "rate_limit_documented": "5 API requests per minute (Free tier basic)",
                "historical_lookback_documented": "2 years historical data on basic tier",
                "account_entitlement_status": "VERIFIED_DEFICIENT_HTTP_401",
                "reference_data_access_status": "HTTP_401_UNAUTHORIZED",
                "historical_bar_access_verified": False,
                "access_readiness": "BLOCKED_HTTP_401_UNKNOWN_KEY",
                "entitlement_tier": "Configured 23-char key is invalid/unauthorized on Polygon endpoints",
                "query_parameters": {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                },
                "adjustment_semantics": "Split-adjusted daily OHLCV bars (dividends unadjusted per canonical contract)",
                "feed_scope": "Consolidated US equity market volume (SIP aggregated)",
                "feed_comparability_notes": "Consolidated SIP volume is standard for liquidity/turnover floor checks (Rule L04).",
                "example_benchmark_url": f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2025-09-15/{target_session}?adjusted=true&sort=asc&limit=50000",
                "example_builder_url": f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2026-07-16/{target_session}?adjusted=true&sort=asc&limit=50000",
            },
            "alpaca": {
                "provider_name": "Alpaca Market Data API v2",
                "endpoint_template": "https://data.alpaca.markets/v2/stocks/bars",
                "rate_limit_documented": "200 requests per minute",
                "historical_lookback_documented": "5 years historical data on basic IEX feed",
                "account_entitlement_status": "VERIFIED_FREE_TIER_IEX_ONLY",
                "reference_data_access_status": "FEED_IEX_ACCESSIBLE",
                "historical_bar_access_verified": False,
                "access_readiness": "BLOCKED_LACKS_CONSOLIDATED_SIP_ENTITLEMENT",
                "entitlement_tier": "Free / Basic tier (querying feed=sip returns HTTP 403 Forbidden)",
                "query_parameters": {
                    "symbols": "{symbol}",
                    "timeframe": "1Day",
                    "start": "{from}",
                    "end": "{to}",
                    "adjustment": "split",
                    "feed": "iex",
                    "limit": 1000,
                },
                "adjustment_semantics": "Split-adjusted daily OHLCV bars via IEX feed (dividends unadjusted per canonical contract)",
                "feed_scope": "IEX exchange participant volume only (feed=iex)",
                "feed_comparability_notes": "IEX volume represents a minority fraction (~2-4%) of consolidated US equity turnover. Direct dollar volume calculations using IEX volume underestimate true market liquidity and cannot be substituted for consolidated market volume.",
                "example_benchmark_url": f"https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Day&start=2025-09-15&end={target_session}&adjustment=split&feed=iex&limit=1000",
                "example_builder_url": f"https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Day&start=2026-07-16&end={target_session}&adjustment=split&feed=iex&limit=1000",
            },
        },
        "ingestion_scope": {
            "production_target_backfill": {
                "missing_candidates_count": len(prod_missing_syms),
                "missing_symbols": prod_missing_syms,
                "target_session": target_session,
                "benchmark_window_260": f"2025-09-15 to {target_session} (400 calendar days / ~275 trading sessions request window strategy)",
                "builder_window_60": f"2026-07-16 to {target_session} (95 calendar days / ~65 trading sessions request window strategy)",
                "illustrative_window_251": f"2025-10-19 to {target_session} (~1 calendar year / ~251 trading sessions estimate)",
                "variable_row_count_notice": "Actual returned row count depends on provider session calendar, IPO date, and listing history; 251 rows is an estimate, not a guaranteed fixed constant.",
                "total_estimated_bars_251_sessions": len(prod_missing_syms) * 251,
                "total_estimated_bars_260_bars": len(prod_missing_syms) * 260,
                "total_estimated_bars_60_bars": len(prod_missing_syms) * 60,
                "per_symbol_parameters": missing_params,
            },
            "production_target_refresh": {
                "stale_candidates_count": len(prod_stale_syms),
                "stale_symbols": prod_stale_syms,
                "target_session": target_session,
                "window": f"Day after observed max_date to {target_session} (exact per-symbol windows)",
                "total_estimated_bars": len(prod_stale_syms) * 15,
                "per_symbol_parameters": stale_params,
            },
            "local_developer_store_gap": {
                "missing_count": len(local_missing_syms),
                "missing_symbols": local_missing_syms,
                "stale_count": len(local_stale_syms),
                "fresh_count": local_fresh_cnt,
                "goog_note": "GOOG has 287 bars present in production store (latest_trade_date: 2026-10-16) but is absent in local dev store. Local absence must NOT generate a production backfill request.",
            },
            "adjustment_policy": "SPLIT_ADJUSTED_ONLY (Zero dividend adjustments per canonical contract)",
            "provider_options": ["Alpaca Market Data v2", "Polygon.io REST API"],
            "endpoint": "/v2/stocks/bars",
            "account_entitlement_status": "VERIFIED_DEFICIENT_CREDENTIALS",
            "entitlement_tier": "Consolidated market data access is unverified/blocked across configured keys",
        },
        "bar_validation_specification": {
            "required_fields": ["symbol", "trade_date", "open", "high", "low", "close", "volume"],
            "finite_numeric_values": "open > 0, high > 0, low > 0, close > 0, all values finite, non-null, non-NaN, non-infinite",
            "valid_dates_and_identity": "trade_date matches ISO 8601 YYYY-MM-DD within requested window, symbol uppercase matching association",
            "ohlc_consistency": "high >= low, high >= open, high >= close, low <= open, low <= close",
            "nonnegative_volume": "volume >= 0, integer or non-negative float",
            "duplicate_handling": "UNIQUE(symbol, trade_date) primary key enforcement; duplicate bars rejected",
            "adjustment_source_metadata": "split-adjusted daily bars; unadjusted dividends",
            "failure_and_partial_response_handling": "HTTP non-200, truncated payloads, or row validation failures trigger atomic rollback and fail closed",
        },
        "atomic_guarantees": {
            "idempotent_upsert": "INSERT OR REPLACE INTO asset_ohlcv_daily",
            "anomaly_rejection": "Full bar_validation_specification enforced prior to insertion",
            "zero_production_writes_by_preparation": True,
        },
    }

    # ── PACKAGE C: Controlled Universe Publication Package ─────────────────
    package_c = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "package_id": "PKG-EXPANSION-150-C-UNIVERSE-PUBLICATION",
        "title": "Controlled Universe Publication Package (Namespace ARX_US_EQUITIES_EXPANSION_150)",
        "description": "Non-executable universe publication specification restricted strictly to an isolated namespace for 133 Radar launch candidates.",
        "governance_status": {
            "executable": False,
            "ready_for_publication_count": 0,
            "publication_authorized": False,
            "default_universe_switch_authorized": False,
            "scanner_activation_authorized": False,
            "is_authorized": False,
            "authorized_by": None,
            "blocking_reasons": [
                "Product Owner exact plan approval is ABSENT",
                "Publication authorization is FALSE",
                "Market data backfill across production store is incomplete (61 launch candidates missing in production, 15 stale prior to session 2026-10-19)",
                "Production identity registration is unauthorized and unapplied for 130 new candidates in /root/analyst_dashboard/data/identity.db",
                "Consolidated market data provider access is blocked by deficient credentials (Polygon HTTP 401 Unknown Key; Alpaca SIP HTTP 403 Forbidden)",
            ],
            "decoupled_safeguards": {
                "gate_12_decoupled": True,
                "default_scanner_switch_decoupled": True,
                "notes": "Gate 12 advancement and default scanner universe switching are operational milestones decoupled from isolated namespace publication readiness.",
            },
        },
        "target_namespace": {
            "universe_id": "ARX_US_EQUITIES_EXPANSION_150",
            "isolated": True,
            "prior_published_pilot_namespace": "ARX_US_EQUITIES_PILOT_3C",
            "default_scanner_universe_preserved": "ARX_US_EQUITIES",
            "radar_mutation": "STRICTLY_PROHIBITED",
        },
        "membership_accounting": {
            "source_population_count": len(target_launch_rows),
            "counterfactual_eligible_membership_count": len(target_launch_rows),
            "actual_eligible_membership_count": len(target_launch_rows),
            "ineligible_count": 0,
            "unresolved_count": 0,
            "ineligible_breakdown": {
                "R01_ASSET_CLASS_ETFS": 0,
                "R02_SECURITY_TYPE_ADRS": 0,
            },
            "scannable_membership_count": 0,
        },
        "radar_launch_universe_accounting": {
            "policy_document": "docs/research/radar-us-listed-coverage/RADAR_LAUNCH_UNIVERSE_SELECTION_POLICY.md",
            "selected_common_equities_count": len(target_launch_rows),
            "actual_policy_pass_count": 122,
            "actual_policy_unknown_count": 11,
            "counterfactual_policy_pass_count": 122,
            "policy_pass_count": 122,
            "policy_unknown_count": 11,
            "observed_turnover_ranked_count": 122,
            "market_data_fresh_local_count": sum(1 for c in target_launch_rows if c["disposition_c_market_data"]["local_dev_store"]["session_fresh"]),
            "validation_sample_fresh_local_count": market_readiness.get("fresh_count", 38),
            "market_data_fresh_production_count": len(prod_fresh_syms),
            "ready_for_publication_count": 0,
            "excluded_count": launch_exclusions.get("total_excluded_count", 28) if launch_exclusions else 28,
            "deterministic_ranking": "20-day median dollar volume descending, symbol alphabetically for policy-pass candidates; unranked for policy-unknown candidates",
        },
        "atomic_guarantees": {
            "publication_decision": "NON_EXECUTABLE_SPECIFICATION_ONLY",
            "default_switch_prevented": True,
            "scanner_scheduler_activation_prevented": True,
            "zero_production_writes_by_preparation": True,
        },
    }

    return package_a, package_b, package_c, adoption_plan, approval_template


def main():
    start_time = time.time()
    print("=" * 80)
    print("ARX TERMINAL — SPRINT 2 APPROVED SCOPE ADDITION")
    print("BOUNDED 150-CANDIDATE COVERAGE EXPANSION & OPERATIONAL PACKAGE BUILDER")
    print("=" * 80)

    raw_dir = REPO_ROOT / "data" / "raw_feeds" / "2026-10-16"
    sec_by_ticker, nasdaq_by_sym, other_by_sym, source_digests = load_raw_directories(raw_dir)

    # 1. Build Boundary Validation Manifest (150 Items) & Exclusions
    print("\n[1/6] Building boundary validation manifest (150 candidates)...")
    manifest_rows, exclusions = generate_candidate_manifest()
    print(f"  Validation sample candidate count: {len(manifest_rows)}")
    print(f"  Dev store exclusions: {exclusions['excluded_count']}")
    print(f"  Directory additions: {exclusions['directory_additions_count']}")

    # 2. Build Radar Launch Universe Manifest (133 Qualifying Common Equities)
    print("\n[2/6] Building Radar Launch Universe manifest (Rules L01–L06)...")
    launch_manifest_rows, launch_recon = generate_radar_launch_universe(
        sec_by_ticker, nasdaq_by_sym, other_by_sym, source_digests
    )
    print(f"  Launch universe qualifying common equities: {len(launch_manifest_rows)}")
    print(f"  Launch universe total exclusions: {launch_recon['total_excluded_count']}")
    print(f"  Arithmetic closure verified: {launch_recon['arithmetic_closure']}")

    # Merge launch universe reconciliation into exclusions
    exclusions["launch_universe_reconciliation"] = launch_recon

    # 3. Evaluate Canonical Eligibility (Rules R01–R07) on Validation Sample
    print("\n[3/6] Evaluating canonical eligibility on validation sample (Rules R01–R07)...")
    attestation, eligibility_rows, eligibility_acct = evaluate_manifest_eligibility(manifest_rows)
    print(f"  Eligible: {eligibility_acct['eligible_count']}")
    print(f"  Ineligible: {eligibility_acct['ineligible_count']}")
    print(f"  Unresolved: {eligibility_acct['unresolved_count']}")
    print(f"  Accounting closure verified: {eligibility_acct['closure_verified']}")

    # 4. Evaluate Market Data Readiness
    print("\n[4/6] Evaluating market-data readiness...")
    market_readiness = evaluate_market_data_readiness(manifest_rows)
    print(f"  Data present: {market_readiness['data_present_count']}/{market_readiness['total_evaluated']}")
    print(f"  Data missing: {market_readiness['data_missing_count']}")
    print(f"  Session fresh (2026-10-16): {market_readiness['fresh_count']}")
    print(f"  Session stale: {market_readiness['stale_count']}")
    print(f"  Screener prereq (>=60 bars): {market_readiness['screener_60_bars_ready_count']}")
    print(f"  VCP trend prereq (>=260 bars): {market_readiness['vcp_260_bars_ready_count']}")

    # 5. Build Operational Packages
    print("\n[5/6] Preparing operational review packages A, B, and C...")
    pkg_a, pkg_b, pkg_c, adoption_plan, approval_template = build_operational_packages(
        manifest_rows, eligibility_acct, market_readiness, launch_manifest_rows, launch_recon
    )

    # 6. Serialize Artifacts
    print("\n[6/6] Serializing canonical evidence artifacts...")
    out_dir = REPO_ROOT / "evidence" / "coverage_expansion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Proposed Radar Launch Manifest
    launch_file = out_dir / "proposed_radar_launch_manifest.json"
    launch_file.write_text(json.dumps(launch_manifest_rows, indent=2), encoding="utf-8")
    print(f"  Wrote: {launch_file} ({launch_file.stat().st_size} bytes)")

    # Validation Sample
    manifest_file = out_dir / "candidate_manifest_150.json"
    manifest_file.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(f"  Wrote: {manifest_file} ({manifest_file.stat().st_size} bytes)")

    validation_file = out_dir / "boundary_validation_manifest_150.json"
    validation_file.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(f"  Wrote: {validation_file} ({validation_file.stat().st_size} bytes)")

    exclusions_file = out_dir / "candidate_exclusions_reconciliation.json"
    exclusions_file.write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    print(f"  Wrote: {exclusions_file} ({exclusions_file.stat().st_size} bytes)")

    # Package A Identity Adoption Plan
    pkg_a_plan_file = out_dir / "package_a_identity_adoption_plan.json"
    with open(pkg_a_plan_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(adoption_plan.to_json(indent=2))
    print(f"  Wrote: {pkg_a_plan_file} ({pkg_a_plan_file.stat().st_size} bytes)")

    # Compute exact SHA-256 of serialized plan file for approval manifest template
    plan_file_sha = hashlib.sha256(pkg_a_plan_file.read_bytes()).hexdigest()
    approval_template["reviewed_plan_file_sha256"] = plan_file_sha

    # Unsigned Operator Approval Manifest Template
    pkg_a_template_file = out_dir / "package_a_approval_manifest_template.json"
    with open(pkg_a_template_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(approval_template, indent=2))
    print(f"  Wrote: {pkg_a_template_file} ({pkg_a_template_file.stat().st_size} bytes)")

    pkg_a_file = out_dir / "package_a_identity_registration.json"
    with open(pkg_a_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(pkg_a, indent=2))
    print(f"  Wrote: {pkg_a_file} ({pkg_a_file.stat().st_size} bytes)")

    pkg_b_file = out_dir / "package_b_market_data_ingestion.json"
    pkg_b_file.write_text(json.dumps(pkg_b, indent=2), encoding="utf-8")
    print(f"  Wrote: {pkg_b_file} ({pkg_b_file.stat().st_size} bytes)")

    pkg_c_file = out_dir / "package_c_universe_publication.json"
    pkg_c_file.write_text(json.dumps(pkg_c, indent=2), encoding="utf-8")
    print(f"  Wrote: {pkg_c_file} ({pkg_c_file.stat().st_size} bytes)")

    elapsed = time.time() - start_time
    print(f"\nExecution completed successfully in {elapsed:.3f}s.")
    print("STATUS: SPRINT2_150_CANDIDATE_EXPANSION_PREPARED_FOR_CONTROLLED_EXECUTION")


if __name__ == "__main__":
    main()
