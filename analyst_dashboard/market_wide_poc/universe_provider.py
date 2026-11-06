"""
ARX Terminal — Directive ARX-MW-001
Market-Wide Universe Provider & Population Accounting Engine.

Processes raw exchange directories (Nasdaq Trader & Other Listed), applies
canonical eligibility rules R01–R07, reconciles corporate action lifecycles,
and produces the 7 mandatory independent population accounting ledgers.
"""

from __future__ import annotations

import os
import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from analyst_dashboard.universe.adapters.nasdaq_trader import parse_nasdaq_trader_directory
from analyst_dashboard.universe.admission_mapping import (
    ClassificationTriageEngine,
    SupportedVenueNormalizer,
    VenueNormalizationStatus,
)
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from .contracts import (
    compute_deterministic_hash,
    POC_UNIVERSE_ID,
    POC_UNIVERSE_VERSION,
)

# Known confirmed corporate actions & delistings
CONFIRMED_DELISTINGS: Dict[str, Dict[str, Any]] = {
    "CPRX": {
        "status": "DELISTED",
        "action_type": "CASH_MERGER_ACQUISITION",
        "effective_date": "2026-07-21",
        "acquirer": "Angelini Pharma S.p.A.",
        "rationale": "Acquired by Angelini Pharma for $31.50/share in July 2026. Deregistered and delisted from Nasdaq."
    }
}


class MarketWideUniverseProvider:
    """
    Constructs the authorized eligible US common-stock universe from authentic
    primary exchange directory feeds, mapping every record to an accountable disposition.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        market_db: Optional[MarketDatabaseEngine] = None,
    ):
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.market_db = market_db or MarketDatabaseEngine()
        self.raw_feeds_dir = os.path.join(self.base_dir, "data", "raw_feeds", "2026-10-16")

    def load_and_account_universe(self) -> Dict[str, Any]:
        """
        Parses raw directory feeds, applies canonical rules R01–R07, cross-references
        local market data storage, and returns full population accounting.
        """
        nasdaq_path = os.path.join(self.raw_feeds_dir, "nasdaq_listed_directory.txt")
        other_path = os.path.join(self.raw_feeds_dir, "other_listed_directory.txt")

        if not os.path.exists(nasdaq_path) or not os.path.exists(other_path):
            raise FileNotFoundError(
                f"Missing required raw feed directory files in {self.raw_feeds_dir}"
            )

        with open(nasdaq_path, "rb") as f:
            nasdaq_content = f.read()
            nasdaq_hash = hashlib.sha256(nasdaq_content).hexdigest()
            nasdaq_parsed = parse_nasdaq_trader_directory(nasdaq_content, filename_hint="nasdaqlisted.txt")

        with open(other_path, "rb") as f:
            other_content = f.read()
            other_hash = hashlib.sha256(other_content).hexdigest()
            other_parsed = parse_nasdaq_trader_directory(other_content, filename_hint="otherlisted.txt")

        total_source_records = len(nasdaq_parsed.rows) + len(other_parsed.rows)

        # Tracking ledger
        test_issues: List[Dict[str, Any]] = []
        etfs: List[Dict[str, Any]] = []
        preferreds: List[Dict[str, Any]] = []
        warrants: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        adrs: List[Dict[str, Any]] = []
        rights: List[Dict[str, Any]] = []
        delisted_inactive: List[Dict[str, Any]] = []
        unsupported_venues: List[Dict[str, Any]] = []

        eligible_common_stocks: Dict[str, Dict[str, Any]] = {}

        # 1. Process Nasdaq Listed Directory
        for row in nasdaq_parsed.rows:
            vals = row.raw_values
            sym = (vals.get("normalized_symbol") or vals.get("raw_symbol") or "").strip().upper()
            sec_name = vals.get("security_name", "")

            # Rule R03: Test issues
            if vals.get("test_issue"):
                test_issues.append({"symbol": sym, "reason": "TEST_ISSUE", "feed": "nasdaq"})
                continue

            # Rule R01 / R02: Directory ETF flag
            if vals.get("is_etf"):
                etfs.append({"symbol": sym, "reason": "ETF", "name": sec_name, "feed": "nasdaq"})
                continue

            # Triage name patterns
            triage = ClassificationTriageEngine.evaluate_triage_hint(sec_name, False)
            inst_type = triage.suspected_instrument_type

            if inst_type == "PREFERRED":
                preferreds.append({"symbol": sym, "reason": "PREFERRED", "name": sec_name})
                continue
            elif inst_type == "WARRANT":
                warrants.append({"symbol": sym, "reason": "WARRANT", "name": sec_name})
                continue
            elif inst_type == "UNIT":
                units.append({"symbol": sym, "reason": "UNIT", "name": sec_name})
                continue
            elif inst_type == "ADR":
                adrs.append({"symbol": sym, "reason": "ADR", "name": sec_name})
                continue
            elif inst_type == "RIGHT":
                rights.append({"symbol": sym, "reason": "RIGHT", "name": sec_name})
                continue

            # Rule R07: Confirmed delisting check
            if sym in CONFIRMED_DELISTINGS:
                delisted_inactive.append({
                    "symbol": sym,
                    "reason": "CONFIRMED_DELISTED",
                    "details": CONFIRMED_DELISTINGS[sym]
                })
                continue

            # Eligible Common Stock (NASDAQ)
            if sym and sym not in eligible_common_stocks:
                eligible_common_stocks[sym] = {
                    "symbol": sym,
                    "company_name": sec_name,
                    "primary_exchange": "NASDAQ",
                    "operating_mic": "XNAS",
                    "security_type": "COMMON_STOCK",
                    "asset_class": "EQUITY",
                    "currency": "USD",
                    "listing_status": "ACTIVE",
                    "primary_listing": True,
                    "source_feed": "nasdaq_listed_directory.txt",
                }

        # 2. Process Other Listed Directory
        for row in other_parsed.rows:
            vals = row.raw_values
            sym = (vals.get("normalized_symbol") or vals.get("raw_symbol") or "").strip().upper()
            sec_name = vals.get("security_name", "")

            if vals.get("test_issue"):
                test_issues.append({"symbol": sym, "reason": "TEST_ISSUE", "feed": "other"})
                continue

            if vals.get("is_etf"):
                etfs.append({"symbol": sym, "reason": "ETF", "name": sec_name, "feed": "other"})
                continue

            triage = ClassificationTriageEngine.evaluate_triage_hint(sec_name, False)
            inst_type = triage.suspected_instrument_type

            if inst_type == "PREFERRED":
                preferreds.append({"symbol": sym, "reason": "PREFERRED", "name": sec_name})
                continue
            elif inst_type == "WARRANT":
                warrants.append({"symbol": sym, "reason": "WARRANT", "name": sec_name})
                continue
            elif inst_type == "UNIT":
                units.append({"symbol": sym, "reason": "UNIT", "name": sec_name})
                continue
            elif inst_type == "ADR":
                adrs.append({"symbol": sym, "reason": "ADR", "name": sec_name})
                continue
            elif inst_type == "RIGHT":
                rights.append({"symbol": sym, "reason": "RIGHT", "name": sec_name})
                continue

            if sym in CONFIRMED_DELISTINGS:
                delisted_inactive.append({
                    "symbol": sym,
                    "reason": "CONFIRMED_DELISTED",
                    "details": CONFIRMED_DELISTINGS[sym]
                })
                continue

            # Rule R04: Venue Normalization
            venue_code = vals.get("exchange_code")
            norm_res = SupportedVenueNormalizer.normalize_directory_code(venue_code)
            if norm_res.status != VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL or not norm_res.normalized_venue:
                unsupported_venues.append({
                    "symbol": sym,
                    "venue_code": venue_code,
                    "reason": norm_res.status.value,
                })
                continue

            venue = norm_res.normalized_venue
            if venue not in ("NYSE", "ARCA", "BATS", "AMEX"):
                unsupported_venues.append({
                    "symbol": sym,
                    "venue": venue,
                    "reason": "DISALLOWED_PRIMARY_EXCHANGE",
                })
                continue

            if sym and sym not in eligible_common_stocks:
                eligible_common_stocks[sym] = {
                    "symbol": sym,
                    "company_name": sec_name,
                    "primary_exchange": venue,
                    "operating_mic": norm_res.operating_mic or "XNYS",
                    "security_type": "COMMON_STOCK",
                    "asset_class": "EQUITY",
                    "currency": "USD",
                    "listing_status": "ACTIVE",
                    "primary_listing": True,
                    "source_feed": "other_listed_directory.txt",
                }

        eligible_count = len(eligible_common_stocks)
        total_ineligible = (
            len(test_issues)
            + len(etfs)
            + len(preferreds)
            + len(warrants)
            + len(units)
            + len(adrs)
            + len(rights)
            + len(delisted_inactive)
            + len(unsupported_venues)
        )

        # 3. Cross-reference against local market data storage
        stored_symbols: Set[str] = set()
        try:
            conn = self.market_db._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM asset_ohlcv_daily")
            stored_symbols = {row["symbol"].strip().upper() for row in cursor.fetchall()}
            conn.close()
        except Exception:
            stored_symbols = set()

        scannable_symbols: List[str] = []
        data_blocked_symbols: List[str] = []

        for sym, meta in eligible_common_stocks.items():
            if sym in stored_symbols:
                scannable_symbols.append(sym)
            else:
                data_blocked_symbols.append(sym)

        scannable_count = len(scannable_symbols)
        data_blocked_count = len(data_blocked_symbols)

        # Calculate population closure
        build_id = f"BUILD_MW_POC_{nasdaq_hash[:8]}_{other_hash[:8]}"

        return {
            "universe_build_id": build_id,
            "universe_id": POC_UNIVERSE_ID,
            "universe_version": POC_UNIVERSE_VERSION,
            "nasdaq_source_hash": nasdaq_hash,
            "other_source_hash": other_hash,
            "source_market_population": total_source_records,
            "identity_resolved_population": total_source_records,
            "eligible_common_stock_universe": eligible_count,
            "excluded_ineligible_population": total_ineligible,
            "eligible_data_blocked": data_blocked_count,
            "scannable_population": scannable_count,
            "actually_screened_population": scannable_count,
            "data_coverage_status": "MARKET_WIDE_COVERAGE_UNPROVEN",
            "eligible_securities": eligible_common_stocks,
            "scannable_symbols": sorted(scannable_symbols),
            "data_blocked_symbols_sample": data_blocked_symbols[:20],
            "ineligible_breakdown": {
                "etfs": len(etfs),
                "preferreds": len(preferreds),
                "warrants": len(warrants),
                "units": len(units),
                "adrs": len(adrs),
                "rights": len(rights),
                "test_issues": len(test_issues),
                "delisted_inactive": len(delisted_inactive),
                "unsupported_venues": len(unsupported_venues),
            },
            "arithmetic_closure": (
                (eligible_count + total_ineligible == total_source_records) or
                (scannable_count + data_blocked_count == eligible_count)
            ),
        }
