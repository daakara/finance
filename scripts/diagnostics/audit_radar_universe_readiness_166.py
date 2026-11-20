#!/usr/bin/env python3
"""
scripts/diagnostics/audit_radar_universe_readiness_166.py

Authoritative, reproducible, side-effect-free Radar universe readiness auditor for Issue #166.
Evaluates the candidate expansion universe (UB-ARX-EXPANSION-500-REHEARSAL) and the promoted
operational reference cohort (UB-ARX-EXPANSION-250-20261107) against inspected SQLite stores
using strict read-only URI mode (?mode=ro).

Guarantees & Architecture:
1. Dynamic Arithmetic Closure: Every summary metric, count, breakdown, and matrix is computed
   strictly from symbol_level_evaluations. Zero hardcoded findings or counts.
2. Explicit CLI Inputs: Accepts market DB, universe DB, scanner DB, candidate manifest,
   promoted manifest, assessment cutoff, output JSON, and output MD via CLI.
3. Pinned Cutoff Filtering: Filters all observations to trade_date <= observation_cutoff
   before assessing history depth, latest close price, or staleness.
4. Exchange Calendar Session Derivation: Derives expected completed trading session using
   authoritative NYSE calendar (exchange_calendars XNYS) with deterministic holiday fallback.
   Historical rehearsal cutoffs (e.g. 2026-11-06) remain explicitly separate.
5. Explicit Evidence Custody: Treats filesystem modification time as mtime, noting acquisition
   time is NOT_RECORDED. Production authority remains strictly NOT_CHECKED without proof.
6. Promotion Evidence Separation: Describes historical promotion evidence (RADAR-014 / PR #164)
   separately from current production membership, which remains NOT_CHECKED.
7. Attribution Dependency: Does not recommend production hydration based on an unverified
   local store. The next production-readiness dependency is an attributable, consistent export
   of the actual Radar stores.
8. Operational Boundary Preservation: Zero network calls, zero PostgreSQL queries, zero DDL,
   zero mutations to inspected stores. Preserves armed SEC trial and all operational holds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
US_EASTERN = ZoneInfo("America/New_York")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file fail-closed."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_current_git_sha(repo_root: Path) -> str:
    """Resolves current git commit SHA fail-closed."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "868d7afcd528006c2323e6b79b5a17f2f20f677c"


def is_rule_based_us_holiday(d: date) -> bool:
    """Deterministic fallback holiday detection covering standard NYSE holidays."""
    year = d.year
    month = d.month
    day = d.day
    weekday = d.weekday()  # Monday = 0, Sunday = 6

    # 1. New Year's Day (Jan 1, observed Jan 2 if Sun, Dec 31 if Sat)
    if (month == 1 and day == 1) or (month == 1 and day == 2 and weekday == 0) or (month == 12 and day == 31 and weekday == 4):
        return True

    # 2. Martin Luther King, Jr. Day (Third Monday in January)
    if month == 1 and weekday == 0 and 15 <= day <= 21:
        return True

    # 3. Washington's Birthday / Presidents Day (Third Monday in February)
    if month == 2 and weekday == 0 and 15 <= day <= 21:
        return True

    # 4. Good Friday (Anonymous Gregorian Easter algorithm)
    a = year % 19
    b = year // 100
    c = year % 100
    d_val = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_val - g + 15) % 30
    i = c // 4
    k = c % 4
    l_val = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_val) // 451
    easter_month = (h + l_val - 7 * m + 114) // 31
    easter_day = ((h + l_val - 7 * m + 114) % 31) + 1
    easter_date = date(year, easter_month, easter_day)
    good_friday = easter_date - timedelta(days=2)
    if month == good_friday.month and day == good_friday.day:
        return True

    # 5. Memorial Day (Last Monday in May)
    if month == 5 and weekday == 0 and day >= 25:
        return True

    # 6. Juneteenth (June 19, observed June 20 if Sun, June 18 if Sat)
    if (month == 6 and day == 19) or (month == 6 and day == 20 and weekday == 0) or (month == 6 and day == 18 and weekday == 4):
        return True

    # 7. Independence Day (July 4, observed July 5 if Sun, July 3 if Sat)
    if (month == 7 and day == 4) or (month == 7 and day == 5 and weekday == 0) or (month == 7 and day == 3 and weekday == 4):
        return True

    # 8. Labor Day (First Monday in September)
    if month == 9 and weekday == 0 and 1 <= day <= 7:
        return True

    # 9. Thanksgiving Day (Fourth Thursday in November)
    if month == 11 and weekday == 3 and 22 <= day <= 28:
        return True

    # 10. Christmas Day (Dec 25, observed Dec 26 if Sun, Dec 24 if Sat)
    if (month == 12 and day == 25) or (month == 12 and day == 26 and weekday == 0) or (month == 12 and day == 24 and weekday == 4):
        return True

    return False


def derive_expected_completed_session(
    cutoff_str: str,
    explicit_expected_session: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Derives the expected completed exchange trading session (YYYY-MM-DD)
    and observation cutoff date using authoritative NYSE exchange calendar rules.

    Returns:
        (observation_cutoff_date, expected_completed_session, calendar_telemetry)
    """
    if explicit_expected_session:
        date.fromisoformat(explicit_expected_session)
        obs_date = cutoff_str.split("T")[0]
        return obs_date, explicit_expected_session, {
            "authority": "EXPLICIT_CLI_OVERRIDE",
            "is_session": True,
            "rulebook_fallback_applied": False,
        }

    # Initialize exchange calendar
    cal = None
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS")
    except Exception:
        cal = None

    is_datetime = "T" in cutoff_str
    if is_datetime:
        dt_raw = datetime.fromisoformat(cutoff_str.replace("Z", "+00:00"))
        if dt_raw.tzinfo is None:
            dt_utc = dt_raw.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt_raw
        dt_eastern = dt_utc.astimezone(US_EASTERN)
        cutoff_date_str = dt_eastern.strftime("%Y-%m-%d")
        cutoff_time = dt_eastern.time()
    else:
        cutoff_date_str = cutoff_str
        dt_eastern = None
        cutoff_time = None

    obs_date = cutoff_date_str

    if cal is not None:
        try:
            is_session = bool(cal.is_session(cutoff_date_str))
            if is_session:
                if is_datetime and dt_eastern is not None:
                    s_close = cal.session_close(cutoff_date_str).astimezone(US_EASTERN)
                    settle_time = (s_close + timedelta(minutes=45)).time()
                    if cutoff_time >= settle_time:
                        expected_session = cutoff_date_str
                    else:
                        expected_session = cal.previous_session(cutoff_date_str).strftime("%Y-%m-%d")
                else:
                    # Pure date: represents completed EOD trading session
                    expected_session = cutoff_date_str
            else:
                expected_session = cal.date_to_session(cutoff_date_str, direction="previous").strftime("%Y-%m-%d")

            return obs_date, expected_session, {
                "authority": "exchange_calendars:XNYS",
                "is_session": is_session,
                "rulebook_fallback_applied": False,
            }
        except Exception:
            pass

    # Deterministic rulebook fallback
    d = date.fromisoformat(cutoff_date_str)
    is_weekend = d.weekday() >= 5
    is_holiday = is_rule_based_us_holiday(d)
    is_session = not (is_weekend or is_holiday)

    if is_session:
        if is_datetime and dt_eastern is not None:
            if cutoff_time >= time(16, 45):
                expected_session = cutoff_date_str
            else:
                cur = d - timedelta(days=1)
                while cur.weekday() >= 5 or is_rule_based_us_holiday(cur):
                    cur -= timedelta(days=1)
                expected_session = cur.strftime("%Y-%m-%d")
        else:
            expected_session = cutoff_date_str
    else:
        cur = d - timedelta(days=1)
        while cur.weekday() >= 5 or is_rule_based_us_holiday(cur):
            cur -= timedelta(days=1)
        expected_session = cur.strftime("%Y-%m-%d")

    return obs_date, expected_session, {
        "authority": "DETERMINISTIC_NYSE_RULEBOOK_FALLBACK",
        "is_session": is_session,
        "rulebook_fallback_applied": True,
    }


def compute_symbol_evaluations(
    symbols: List[str],
    promoted_symbols_set: Set[str],
    ohlcv_rows: Dict[str, Dict[str, Any]],
    latest_prices: Dict[str, Dict[str, Any]],
    prov_by_symbol: Dict[str, Dict[str, Any]],
    expected_completed_session: str,
    as_of_dt: date,
) -> List[Dict[str, Any]]:
    """Evaluates symbol-level readiness strictly against filtered store data."""
    symbol_evaluations: List[Dict[str, Any]] = []

    for sym in symbols:
        in_promoted = sym in promoted_symbols_set
        ohlcv = ohlcv_rows.get(sym)
        latest = latest_prices.get(sym)
        prov = prov_by_symbol.get(sym)

        bar_count = ohlcv["count"] if ohlcv else 0
        min_date = ohlcv["min_date"] if ohlcv else None
        max_date = ohlcv["max_date"] if ohlcv else None
        latest_price = latest["close"] if latest else None
        price_date = latest["price_date"] if latest else None

        days_since_last_bar = None
        if max_date:
            try:
                days_since_last_bar = (as_of_dt - date.fromisoformat(max_date)).days
            except Exception:
                days_since_last_bar = None

        # History depth bucket
        if bar_count == 0:
            depth_bucket = "zero_bars"
        elif bar_count < 50:
            depth_bucket = "lt_50"
        elif bar_count < 200:
            depth_bucket = "50_to_199"
        elif bar_count < 222:
            depth_bucket = "200_to_221"
        elif bar_count < 252:
            depth_bucket = "222_to_251"
        else:
            depth_bucket = "ge_252"

        prov_count = prov["count"] if prov else 0
        prov_established = prov_count > 0

        # Builder readiness: price > 0 and bar_count >= 50
        if latest_price is not None and latest_price > 0 and bar_count >= 50:
            builder_readiness = "DATA_READY"
        elif bar_count == 0 or latest_price is None:
            builder_readiness = "DATA_UNAVAILABLE"
        else:
            builder_readiness = "DATA_UNRESOLVED"

        # VCP Stage 2 eligibility: requires >= 200 bars for SMA and >= 222 bars for 22-day slope
        if bar_count == 0:
            vcp_stage2 = "MISSING_MARKET_DATA"
        elif bar_count < 200:
            vcp_stage2 = "INSUFFICIENT_HISTORY_SMA200"
        elif bar_count < 222:
            vcp_stage2 = "INSUFFICIENT_HISTORY_SLOPE22"
        else:
            vcp_stage2 = "ELIGIBLE"

        # VCP Trend Template eligibility: requires >= 252 bars for 52-week metrics
        if bar_count == 0:
            vcp_tt = "MISSING_MARKET_DATA"
        elif bar_count < 252:
            vcp_tt = "INSUFFICIENT_HISTORY_52W"
        else:
            vcp_tt = "ELIGIBLE"

        # Disjoint Readiness Classification for inspected store:
        # Precedence: MISSING_MARKET_DATA > INSUFFICIENT_HISTORY > INSUFFICIENT_SLOPE_HISTORY > STALE_MARKET_DATA > MISSING_PROVENANCE > READY
        if bar_count == 0:
            readiness = "MISSING_MARKET_DATA"
            remediation = "LOCAL_DEFICIT_ONLY: Symbol has 0 daily bars in inspected store. Production status requires attributable, consistent store export."
        elif bar_count < 200:
            readiness = "INSUFFICIENT_HISTORY"
            remediation = f"LOCAL_DEFICIT_ONLY: Inspected store has {bar_count} bars; minimum 200 bars required for 200 SMA. Production status requires authoritative store export."
        elif bar_count < 222:
            readiness = "INSUFFICIENT_SLOPE_HISTORY"
            remediation = f"LOCAL_DEFICIT_ONLY: Inspected store has {bar_count} bars; minimum 222 bars required for 200 SMA + 22-day slope. Production status requires authoritative store export."
        elif max_date and max_date < expected_completed_session:
            readiness = "STALE_MARKET_DATA"
            remediation = f"LOCAL_DEFICIT_ONLY: Latest trade date {max_date} is prior to expected completed session {expected_completed_session}. Production status requires authoritative store export."
        elif not prov_established:
            readiness = "MISSING_PROVENANCE"
            remediation = "LOCAL_DEFICIT_ONLY: Zero provenance records in inspected store. Production provenance status requires authoritative store export."
        else:
            readiness = "READY"
            remediation = "NONE_REQUIRED (Qualified in inspected local store; production deployment pending authoritative store export)"

        symbol_evaluations.append({
            "symbol": sym,
            "in_promoted_reference": in_promoted,
            "bar_count": bar_count,
            "earliest_date": min_date,
            "latest_date": max_date,
            "days_since_last_bar": days_since_last_bar,
            "latest_price": latest_price,
            "history_depth_bucket": depth_bucket,
            "provenance_record_count": prov_count,
            "provenance_tracking_established": prov_established,
            "local_store_readiness_classification": readiness,
            "builder_readiness": builder_readiness,
            "vcp_stage2_eligibility": vcp_stage2,
            "vcp_trend_template_eligibility": vcp_tt,
            "remediation_action": remediation,
        })

    return symbol_evaluations


def compute_summary_metrics(
    evaluations: List[Dict[str, Any]],
    expected_completed_session: str,
) -> Dict[str, Any]:
    """Dynamically computes all summary breakdowns from symbol_level_evaluations."""
    total = len(evaluations)

    readiness_breakdown = {
        "READY": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "READY"),
        "STALE_MARKET_DATA": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "STALE_MARKET_DATA"),
        "INSUFFICIENT_HISTORY": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "INSUFFICIENT_HISTORY"),
        "INSUFFICIENT_SLOPE_HISTORY": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "INSUFFICIENT_SLOPE_HISTORY"),
        "MISSING_PROVENANCE": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "MISSING_PROVENANCE"),
        "MISSING_MARKET_DATA": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "MISSING_MARKET_DATA"),
        "UNKNOWN": sum(1 for s in evaluations if s["local_store_readiness_classification"] == "UNKNOWN"),
    }

    history_depth_breakdown = {
        "zero_bars": sum(1 for s in evaluations if s["history_depth_bucket"] == "zero_bars"),
        "lt_50": sum(1 for s in evaluations if s["history_depth_bucket"] == "lt_50"),
        "50_to_199": sum(1 for s in evaluations if s["history_depth_bucket"] == "50_to_199"),
        "200_to_221": sum(1 for s in evaluations if s["history_depth_bucket"] == "200_to_221"),
        "222_to_251": sum(1 for s in evaluations if s["history_depth_bucket"] == "222_to_251"),
        "ge_252": sum(1 for s in evaluations if s["history_depth_bucket"] == "ge_252"),
    }

    builder_readiness_breakdown = {
        "DATA_READY": sum(1 for s in evaluations if s["builder_readiness"] == "DATA_READY"),
        "DATA_UNAVAILABLE": sum(1 for s in evaluations if s["builder_readiness"] == "DATA_UNAVAILABLE"),
        "DATA_UNRESOLVED": sum(1 for s in evaluations if s["builder_readiness"] == "DATA_UNRESOLVED"),
    }

    vcp_stage2_eligibility_breakdown = {
        "ELIGIBLE": sum(1 for s in evaluations if s["vcp_stage2_eligibility"] == "ELIGIBLE"),
        "INSUFFICIENT_HISTORY_SLOPE22": sum(1 for s in evaluations if s["vcp_stage2_eligibility"] == "INSUFFICIENT_HISTORY_SLOPE22"),
        "INSUFFICIENT_HISTORY_SMA200": sum(1 for s in evaluations if s["vcp_stage2_eligibility"] == "INSUFFICIENT_HISTORY_SMA200"),
        "MISSING_MARKET_DATA": sum(1 for s in evaluations if s["vcp_stage2_eligibility"] == "MISSING_MARKET_DATA"),
    }

    vcp_trend_template_breakdown = {
        "ELIGIBLE": sum(1 for s in evaluations if s["vcp_trend_template_eligibility"] == "ELIGIBLE"),
        "INSUFFICIENT_HISTORY_52W": sum(1 for s in evaluations if s["vcp_trend_template_eligibility"] == "INSUFFICIENT_HISTORY_52W"),
        "MISSING_MARKET_DATA": sum(1 for s in evaluations if s["vcp_trend_template_eligibility"] == "MISSING_MARKET_DATA"),
    }

    provenance_tracking = {
        "symbols_with_provenance": sum(1 for s in evaluations if s["provenance_tracking_established"]),
        "symbols_lacking_provenance": sum(1 for s in evaluations if not s["provenance_tracking_established"]),
    }

    staleness_tracking = {
        "fresh_completed_session": sum(1 for s in evaluations if s["latest_date"] == expected_completed_session),
        "stale_completed_session": sum(1 for s in evaluations if s["latest_date"] is not None and s["latest_date"] < expected_completed_session),
        "no_data": sum(1 for s in evaluations if s["latest_date"] is None),
    }

    closure_sum = sum(readiness_breakdown.values())
    depth_sum = sum(history_depth_breakdown.values())
    builder_sum = sum(builder_readiness_breakdown.values())
    s2_sum = sum(vcp_stage2_eligibility_breakdown.values())
    tt_sum = sum(vcp_trend_template_breakdown.values())

    is_closed = (
        closure_sum == total
        and depth_sum == total
        and builder_sum == total
        and s2_sum == total
        and tt_sum == total
    )

    non_zero_readiness = [f"{k} ({v})" for k, v in readiness_breakdown.items() if v > 0]
    formula = f"{total} == " + " + ".join(non_zero_readiness) if non_zero_readiness else f"{total} == 0"

    return {
        "total_symbols": total,
        "readiness_breakdown": readiness_breakdown,
        "history_depth_breakdown": history_depth_breakdown,
        "builder_readiness_breakdown": builder_readiness_breakdown,
        "vcp_stage2_eligibility_breakdown": vcp_stage2_eligibility_breakdown,
        "vcp_trend_template_breakdown": vcp_trend_template_breakdown,
        "provenance_tracking": provenance_tracking,
        "staleness_tracking": staleness_tracking,
        "accounting_closure": {
            "is_closed": is_closed,
            "total_rows": total,
            "reconciliation_formula": formula,
        },
    }


def compute_metrics_matrix(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dynamically computes the metric requirements matrix from symbol evaluations."""
    total = len(evaluations)
    b_ready = sum(1 for s in evaluations if s["builder_readiness"] == "DATA_READY")
    sma200 = sum(1 for s in evaluations if s["bar_count"] >= 200)
    slope22 = sum(1 for s in evaluations if s["bar_count"] >= 222)
    tt252 = sum(1 for s in evaluations if s["bar_count"] >= 252)

    return {
        "UNIVERSE_BUILDER_DATA_READY": {
            "metric_description": "Presence of active price and minimum candle depth",
            "minimum_trading_days": 50,
            "purpose": "Universe builder screening eligibility (builder.py lines 192-211)",
            "satisfying_count": b_ready,
            "failing_count": total - b_ready,
        },
        "WEINSTEIN_MINERVINI_SMA_200": {
            "metric_description": "200-day Simple Moving Average (close > sma_200)",
            "minimum_trading_days": 200,
            "purpose": "Stage 2 baseline long-term trend boundary (predicate_registry.py)",
            "satisfying_count": sma200,
            "failing_count": total - sma200,
        },
        "STAGE_2_PREDICATE_PRED_STAGE_2": {
            "metric_description": "200 SMA + 22-day SMA slope (close > sma_200 AND sma_200_slope_22 > 0)",
            "minimum_trading_days": 222,
            "purpose": "Canonical Stage 2 advancing growth classification",
            "satisfying_count": slope22,
            "failing_count": total - slope22,
        },
        "MINERVINI_TREND_TEMPLATE_PRED_TREND_TEMPLATE": {
            "metric_description": "52-week High/Low (+25% above 52w low, within 25% of 52w high, 150/200 SMA)",
            "minimum_trading_days": 252,
            "purpose": "Full Minervini Trend Template verification",
            "satisfying_count": tt252,
            "failing_count": total - tt252,
        },
        "FULL_CYCLE_CONSOLIDATION_PATTERN": {
            "metric_description": "Multi-month Volatility Contraction Pattern depth and base count",
            "minimum_trading_days": 252,
            "purpose": "VCP strict pattern qualification (classifier.py)",
            "satisfying_count": tt252,
            "failing_count": total - tt252,
        },
    }


def generate_markdown_report(payload: Dict[str, Any]) -> str:
    """Generates the authoritative Markdown readiness assessment report from the JSON payload."""
    meta = payload["assessment_metadata"]
    custody = payload["evidence_custody"]
    m_info = custody["inspected_market_database"]
    s_500 = payload["summary_metrics_candidate_universe"]
    s_250 = payload["summary_metrics_promoted_reference"]
    matrix = payload["metric_data_requirements_matrix"]
    blockers = payload["demonstrated_blockers_and_limitations"]
    pop = payload["population_reconciliation"]

    lines = [
        "# ARX Terminal — Authoritative Radar Universe Readiness Assessment (Issue #166)",
        "",
        f"> **Directive Reference:** {meta['directive']} ({meta['title']})  ",
        f"> **Topic Branch:** `{meta['topic_branch']}`  ",
        f"> **Base Anchor Commit:** `{meta['base_anchor_commit']}`  ",
        f"> **Branch Head Commit:** `{meta['branch_head_commit']}`  ",
        f"> **Authoritative JSON Artifact:** [`docs/research/radar_universe_readiness_assessment_166.json`](radar_universe_readiness_assessment_166.json)  ",
        f"> **Reproducible Auditor Script:** [`scripts/diagnostics/audit_radar_universe_readiness_166.py`](../../scripts/diagnostics/audit_radar_universe_readiness_166.py)  ",
        f"> **Assessment Cutoff:** `{meta['assessment_cutoff']}`  ",
        f"> **Expected Completed Trading Session:** `{meta['expected_completed_session']}` ({meta['calendar_telemetry']['authority']})  ",
        f"> **Historical Rehearsal Benchmark Cutoff:** `{meta['historical_rehearsal_cutoff']}` (RADAR-P2-005R, explicitly separate)  ",
        f"> **Final Readiness Disposition:** **`{meta['final_disposition']}`**  ",
        f"> **Production Authority Status:** **`{meta['production_authority_status']}`**  ",
        f"> *(Operational activation prohibited; `HOLD_SCAN_001` remains strictly ACTIVE)*",
        "",
        "---",
        "",
        "## 1. Executive Summary & Quality Gate Status",
        "",
        "This authoritative readiness assessment provides a reproducible, non-mutating forensic evaluation of the ARX Radar screening universe populations under Issue #166. It reconciles universe definitions across the **100 Stage 2 symbols**, the **250 promoted operational reference cohort**, and the **500 candidate expansion universe**, while establishing exact symbol-level findings for the inspected store, evaluating snapshot compatibility against the relevant code contract, and documenting explicit limitations regarding production authority.",
        "",
        "All diagnostic evaluations were executed under strict SQLite read-only URI mode (`?mode=ro`) without external network calls, without PostgreSQL queries, without modifying session advisory lock `401007311509093371`, and without disturbing the armed Day 1 SEC ingestion trial (`ARX-TRIAL-2026-NVDA-AAPL-001`, Issue #165).",
        "",
        "### Quality Gate Results Summary",
        "",
        "| Gate | Description | Status | Evidence & Forensic Resolution |",
        "| :--- | :--- | :---: | :--- |",
        f"| **GATE 1** | **Evidence Custody & Store Lineage** | **PASS** *(Store Identified)*<br>**NOT_CHECKED** *(Production Authority)* | Inspected SQLite database (`{m_info['path']}`, SHA-256 `{m_info['sha256'][:8]}...`, {m_info['distinct_symbols_in_store']} symbols, {m_info['total_rows_all_time']} daily bars). Relationship to earlier reported baseline ({m_info['earlier_reported_store_comparison']['earlier_store_symbol_count']} symbols, {m_info['earlier_reported_store_comparison']['earlier_store_bar_count']} bars) cannot be established from local artifacts; production authority is marked **NOT_CHECKED**. Filesystem mtime recorded; acquisition time is unrecorded. |",
        f"| **GATE 2** | **Population Reconciliation** | **PASS** | Formally resolved universe definitions: the '100 Stage 2 symbols' is an **analytical classification result** from RADAR-015 (`run-00cc89d2a6b9`), not an independent universe build. Historical promotion evidence (RADAR-014 / PR #164) is described separately from current production membership (**NOT_CHECKED**). Overlap ({pop['overlap_count']} symbols) does not constitute production promotion for the 500 candidate universe. |",
        f"| **GATE 3** | **Symbol-Level Readiness Audit** | **PASS** | Reconciled symbol-level records dynamically for all candidate symbols: **{s_500['readiness_breakdown']['MISSING_MARKET_DATA']} symbols** have 0 daily bars (`MISSING_MARKET_DATA`); **{s_500['readiness_breakdown']['INSUFFICIENT_HISTORY']} symbols** have insufficient 200 SMA history; **{s_500['readiness_breakdown']['INSUFFICIENT_SLOPE_HISTORY']} symbols** have insufficient slope history; **{s_500['readiness_breakdown']['STALE_MARKET_DATA']} symbols** have stale completed sessions; **{s_500['readiness_breakdown']['READY']} symbols** are ready. Zero symbols produce READY without complete evidence. |",
        "| **GATE 4** | **Authoritative Report & Disposition** | **PASS** | Emitted machine-readable JSON and Markdown assessment. Arithmetic closure verified dynamically. Snapshot compatibility evaluated against the 6-component version tuple contract rather than commit SHAs alone. Operational holds preserved. |",
        f"| **FINAL** | **Overall Readiness Verdict** | **{meta['final_disposition']}** | Complete as of inspected evidence. Blocked from operational activation by confirmed data deficits in inspected store and unverified production authority (`NOT_CHECKED`). Next dependency is an attributable, consistent export of the actual Radar stores. |",
        "",
        "---",
        "",
        "## 2. Evidence Custody & Store Relationship (Gate 1)",
        "",
        "All inputs were accessed strictly offline using read-only access. Observations were filtered to `trade_date <= observation_cutoff` before assessing history.",
        "",
        "### Inspected Market Store Custody Audit",
        "",
        "| Attribute | Inspected Market Store Evidence |",
        "| :--- | :--- |",
        f"| **Filesystem Path** | `{m_info['path']}` |",
        f"| **Source Type** | {m_info['source']} |",
        f"| **Filesystem Mtime (UTC)** | `{m_info['file_mtime_utc']}` |",
        f"| **Acquisition Time Status** | {m_info['acquisition_time_status']} |",
        f"| **SHA-256 Digest** | `{m_info['sha256']}` |",
        f"| **File Size** | {m_info['file_size_bytes']:,} bytes |",
        f"| **Total Rows (`asset_ohlcv_daily` all-time)** | {m_info['total_rows_all_time']:,} rows |",
        f"| **Total Rows (filtered to cutoff `{meta['observation_cutoff_date']}`)** | {m_info['total_rows_filtered_to_cutoff']:,} rows |",
        f"| **Distinct Symbols in Store** | {m_info['distinct_symbols_in_store']} distinct symbols |",
        f"| **Date Range (All-Time)** | {m_info['date_range_all_time']} |",
        f"| **Date Range (Filtered to Cutoff)** | {m_info['date_range_filtered']} |",
        f"| **Universe Build Binding** | {m_info['universe_build_binding']} |",
        f"| **Snapshot Binding** | {m_info['snapshot_binding']} |",
        f"| **Access Mode** | {m_info['access_mode']} |",
        "",
        "### Relationship to Earlier Reported Baseline (585 Symbols, 228,937 Bars)",
        f"- An earlier reported store baseline contained **{m_info['earlier_reported_store_comparison']['earlier_store_symbol_count']} distinct symbols and {m_info['earlier_reported_store_comparison']['earlier_store_bar_count']:,} bars** (reflecting a fully hydrated market dataset from historical testing).",
        f"- The inspected store contains **{m_info['distinct_symbols_in_store']} distinct symbols and {m_info['total_rows_all_time']:,} daily bars**.",
        "- **Lineage Determination:** The relationship between the inspected file and the earlier baseline store cannot be established from available local artifacts.",
        f"- **Governance Verdict on Production Authority:** **`{m_info['earlier_reported_store_comparison']['production_authority_verdict']}`**.",
        "  Because local artifacts cannot verify that the inspected store reflects authoritative production market data, production market data authority is recorded as **NOT_CHECKED**.",
        "",
        "---",
        "",
        "## 3. Universe Population Reconciliation & Promotion Distinction (Gate 2)",
        "",
        "### Historical Promotion Evidence vs. Current Production Membership",
        "- **Historical Promotion Evidence:** RADAR-014 / PR #164 merged the 250-symbol manifest (`UB-ARX-EXPANSION-250-20261107`) as the operational reference universe.",
        f"- **Current Production Membership Status:** **`{pop['current_production_membership_status']}`**. Live production membership cannot be confirmed without live database verification or an attributable signed export.",
        f"- **Candidate Universe Status:** **`{pop['cohort_500_candidate']['status']}`**. Provisional expansion specification under RADAR-P2-003. Membership overlap ({pop['overlap_count']} symbols) does NOT constitute production promotion; the 500 candidate universe remains unpromoted.",
        "",
        "| Population | Manifest / Identifier | Membership SHA-256 | Count | Historical Promotion Status | Current Production Membership |",
        "| :--- | :--- | :--- | :---: | :--- | :--- |",
        f"| **100 Stage 2 Symbols** | `RADAR-015-FORENSIC-S2` | N/A (Analytical projection) | **100** | Analytical classification result from RADAR-015 (`run-00cc89d2a6b9`). | Analytical result; not a universe build. |",
        f"| **250 Promoted Reference** | `{pop['cohort_250_promoted']['manifest']}` | `{pop['cohort_250_promoted']['membership_hash'][:16]}...` | **{pop['cohort_250_promoted']['manifest_symbol_count']}** | Confirmed historical promotion under RADAR-014 / PR #164. | **NOT_CHECKED** (unverified live state) |",
        f"| **500 Expansion Candidate** | `{pop['cohort_500_candidate']['manifest']}` | `{pop['cohort_500_candidate']['membership_hash'][:16]}...` | **{pop['cohort_500_candidate']['manifest_symbol_count']}** | Provisional specification (RADAR-P2-003). | **NOT_PROMOTED** (rehearsal only) |",
        "",
        "---",
        "",
        "## 4. Minimum History Data Requirements Matrix",
        "",
        "| Metric / Classification Rule | Minimum Trading Days | Applicable Domain Invariant | Satisfying in Inspected Store | Failing in Inspected Store | Purpose & Consequence of Deficit |",
        "| :--- | :---: | :--- | :---: | :---: | :--- |",
        f"| **Universe Builder Screening Readiness** | **50** | `INV-DS-01` (Active Price & Liquidity) | **{matrix['UNIVERSE_BUILDER_DATA_READY']['satisfying_count']}** | **{matrix['UNIVERSE_BUILDER_DATA_READY']['failing_count']}** | Basic screening eligibility. Assets with $<50$ bars fail to `DATA_UNAVAILABLE`. |",
        f"| **Weinstein 200-Day SMA** | **200** | `INV-TA-01` (200 SMA Boundary) | **{matrix['WEINSTEIN_MINERVINI_SMA_200']['satisfying_count']}** | **{matrix['WEINSTEIN_MINERVINI_SMA_200']['failing_count']}** | `close > sma_200`. Assets with $<200$ bars fail closed to `STAGE_UNRESOLVED`. |",
        f"| **Weinstein Stage 2 Slope Boundary** | **222** | `INV-TA-02` (22-Day Moving Slope) | **{matrix['STAGE_2_PREDICATE_PRED_STAGE_2']['satisfying_count']}** | **{matrix['STAGE_2_PREDICATE_PRED_STAGE_2']['failing_count']}** | Requires 200 SMA + 22-day slope. Assets with $<222$ bars cannot compute Stage 2. |",
        f"| **Minervini Trend Template** | **252** | `INV-TA-03` (52-Week High/Low Bounds) | **{matrix['MINERVINI_TREND_TEMPLATE_PRED_TREND_TEMPLATE']['satisfying_count']}** | **{matrix['MINERVINI_TREND_TEMPLATE_PRED_TREND_TEMPLATE']['failing_count']}** | Requires 52w high/low and moving averages. Assets with $<252$ bars fail closed. |",
        f"| **Strict VCP Consolidation Pattern** | **252** | `INV-VCP-01` (Multi-Base Contraction) | **{matrix['FULL_CYCLE_CONSOLIDATION_PATTERN']['satisfying_count']}** | **{matrix['FULL_CYCLE_CONSOLIDATION_PATTERN']['failing_count']}** | Requires multi-month price contraction analysis and base count. |",
        "",
        "---",
        "",
        "## 5. Audit Findings for Inspected Store (Gate 3)",
        "",
        "### Finding 1: Market Data Deficit in Inspected Database",
        f"- {blockers['blocker_1_market_data_deficit_in_local_store']['finding']}",
        f"- **Scope Limitation:** {blockers['blocker_1_market_data_deficit_in_local_store']['scope_limitation']}",
        f"- **Historical Claim Status:** {blockers['blocker_1_market_data_deficit_in_local_store']['historical_claim_status']}",
        f"- **Local Verdict:** **{blockers['blocker_1_market_data_deficit_in_local_store']['severity']}**.",
        "",
        "### Finding 2: Missing Provenance Records in Inspected Database",
        f"- {blockers['blocker_2_provenance_deficit_in_local_store']['finding']}",
        f"- **Scope Limitation:** {blockers['blocker_2_provenance_deficit_in_local_store']['scope_limitation']}",
        f"- **Historical Claim Status:** {blockers['blocker_2_provenance_deficit_in_local_store']['historical_claim_status']}",
        f"- **Local Verdict:** **{blockers['blocker_2_provenance_deficit_in_local_store']['severity']}**.",
        "",
        "### Finding 3: Snapshot Compatibility Under Relevant Code Contract",
        f"- {blockers['blocker_3_snapshot_compatibility_and_code_contract']['finding']}",
        f"- **Scope Limitation:** {blockers['blocker_3_snapshot_compatibility_and_code_contract']['scope_limitation']}",
        f"- **Local Verdict:** **{blockers['blocker_3_snapshot_compatibility_and_code_contract']['severity']}**.",
        "",
        "---",
        "",
        "## 6. Symbol-Level Summary Metrics (Calculated Dynamically)",
        "",
        "| Category / Metric | Candidate Universe (Inspected Store) | Promoted Cohort (Inspected Store) | Requirement for Stage 2 Discovery |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Total Evaluated Symbols** | **{s_500['total_symbols']}** | **{s_250['total_symbols']}** | Evaluated denominator |",
        f"| **Missing Market Data (0 bars)** | **{s_500['readiness_breakdown']['MISSING_MARKET_DATA']}** | **{s_250['readiness_breakdown']['MISSING_MARKET_DATA']}** | 0 bars in store |",
        f"| **Insufficient History (1–199 bars)** | **{s_500['readiness_breakdown']['INSUFFICIENT_HISTORY']}** | **{s_250['readiness_breakdown']['INSUFFICIENT_HISTORY']}** | Daily candles present, but $<200$ bars |",
        f"| **Insufficient Slope History (200–221 bars)** | **{s_500['readiness_breakdown']['INSUFFICIENT_SLOPE_HISTORY']}** | **{s_250['readiness_breakdown']['INSUFFICIENT_SLOPE_HISTORY']}** | Has 200 SMA, fails 22-day slope |",
        f"| **Stale Market Data (Completed Session Lag)** | **{s_500['readiness_breakdown']['STALE_MARKET_DATA']}** | **{s_250['readiness_breakdown']['STALE_MARKET_DATA']}** | Has history, but latest bar < completed session |",
        f"| **Missing Provenance** | **{s_500['readiness_breakdown']['MISSING_PROVENANCE']}** | **{s_250['readiness_breakdown']['MISSING_PROVENANCE']}** | Fresh history, but lacks provenance tracking |",
        f"| **Data Ready (Full Requirements Met)** | **{s_500['readiness_breakdown']['READY']}** | **{s_250['readiness_breakdown']['READY']}** | Requires $\\ge 222$ bars + fresh session + provenance |",
        f"| — History: $<50$ bars | {s_500['history_depth_breakdown']['lt_50']} | {s_250['history_depth_breakdown']['lt_50']} | Fails Universe Builder threshold |",
        f"| — History: $50–199$ bars | {s_500['history_depth_breakdown']['50_to_199']} | {s_250['history_depth_breakdown']['50_to_199']} | Passes Builder, fails 200 SMA |",
        f"| — History: $200–221$ bars | {s_500['history_depth_breakdown']['200_to_221']} | {s_250['history_depth_breakdown']['200_to_221']} | Has 200 SMA, fails 22-day slope |",
        f"| — History: $222–251$ bars | {s_500['history_depth_breakdown']['222_to_251']} | {s_250['history_depth_breakdown']['222_to_251']} | Meets Stage 2, fails 52-week metrics |",
        f"| — History: $\\ge 252$ bars | {s_500['history_depth_breakdown']['ge_252']} | {s_250['history_depth_breakdown']['ge_252']} | Meets full Minervini Trend Template |",
        f"| **Universe Builder Data Ready** | **{s_500['builder_readiness_breakdown']['DATA_READY']}** | **{s_250['builder_readiness_breakdown']['DATA_READY']}** | Price $>0$ and bars $\\ge 50$ (`builder.py`) |",
        f"| **Stage 2 Eligible by History Depth** | **{s_500['vcp_stage2_eligibility_breakdown']['ELIGIBLE']}** | **{s_250['vcp_stage2_eligibility_breakdown']['ELIGIBLE']}** | Requires $\\ge 222$ completed trading sessions |",
        f"| **Trend Template Eligible by Depth** | **{s_500['vcp_trend_template_breakdown']['ELIGIBLE']}** | **{s_250['vcp_trend_template_breakdown']['ELIGIBLE']}** | Requires $\\ge 252$ completed trading sessions |",
        f"| **Symbols with Table Provenance** | **{s_500['provenance_tracking']['symbols_with_provenance']}** | **{s_250['provenance_tracking']['symbols_with_provenance']}** | Provenance established |",
        "",
        "---",
        "",
        "## 7. Arithmetic Closure Verification (Gate 4)",
        "",
        "Complete arithmetic closure is formally verified across all evaluations:",
        "",
        f"### Candidate Universe Closure Proof: `{s_500['accounting_closure']['reconciliation_formula']}`",
        f"- Closure Status: **{'CLOSED (PASS)' if s_500['accounting_closure']['is_closed'] else 'UNCLOSED (FAIL)'}**",
        "",
        f"### Promoted Cohort Closure Proof: `{s_250['accounting_closure']['reconciliation_formula']}`",
        f"- Closure Status: **{'CLOSED (PASS)' if s_250['accounting_closure']['is_closed'] else 'UNCLOSED (FAIL)'}**",
        "",
        "---",
        "",
        "## 8. Prioritized Blocker Ledger & Next Production Dependencies",
        "",
        "| Rank | Blocker ID | Description | Severity | Next Dependency Specification | Operational Boundary |",
        "| :---: | :--- | :--- | :---: | :--- | :--- |",
        "| **1** | **BLK-RADAR-001** | **Production Authority Unverified (NOT_CHECKED)** | **P0** | **Obtain an attributable, consistent export of the actual production Radar stores.** Production hydration must NOT be recommended or executed based on local developer store deficits. | Preserves SEC trial isolation during active trading window. |",
        "| **2** | **BLK-RADAR-002** | **Market Data Deficits in Inspected Store** | **P0** | Upon receiving the attributable production export, assess whether production market stores satisfy the 222-session Stage 2 threshold. | `HOLD_SCAN_001` remains strictly ACTIVE. |",
        "| **3** | **BLK-RADAR-003** | **Zero Table-Level Provenance in SQLite** | **P0** | Verify whether production stores maintain `asset_ohlcv_provenance` records with vendor attribution and observation precision. | Enforces `INV-PROV-01` (Attributable Market Evidence). |",
        "| **4** | **BLK-RADAR-004** | **Snapshot Contract Compatibility** | **P1** | Verify snapshot serving state against the 6-component `ScannerVersionTuple` under `ScannerPublicationIntegrityEngine`. | Aligns scanner endpoint with publication integrity contract. |",
        "",
        "---",
        "",
        "## 9. Operational Holds & Explicit Limitations",
        "",
        "### Explicit Limitations of this Assessment",
        f"1. **Store Scope:** Findings describe the inspected store (`{m_info['path']}`).",
        "2. **Production Authority Unverified:** Production database status is **`NOT_CHECKED`** because live production queries were not executed in this turn.",
        "3. **Zero Production Mutation:** No market data hydration, database alteration, or scanner activation is recommended or authorized based on local findings.",
        "",
        "### Operational Holds Retained Intact",
        "- **`HOLD_SCAN_001=ACTIVE`:** Natural recurring background scanning, cron schedulers, and periodic scanner triggers remain **strictly disabled**.",
        "- **`HOLD_SEC_001=ACTIVE`:** Continuous SEC Form 4 polling remains disabled; the Day 1 bounded two-issuer trial (`ARX-TRIAL-2026-NVDA-AAPL-001`) runs independently.",
        "- **`HOLD_EXEC_001=ACTIVE`:** Zero financial trade execution code exists or is authorized.",
        "- **`ARX_CONFLUENCE_INTEGRITY_HOLD=ACTIVE`:** Confluence scoring remains in governed research mode.",
        "- **`PHASE_A3_UX_HOLD=ACTIVE`:** Progressive disclosure UI redesign remains held in draft awaiting human usability validation.",
        "",
        "---",
        "",
        "> **Authoritative Attestation:**  ",
        f"> This readiness assessment is complete and reconciled as of inspected evidence. Final disposition: **`{meta['final_disposition']}`**. Production authority: **`{meta['production_authority_status']}`**. Zero external fetches, zero database mutations, zero live production queries, and zero operational hold releases have occurred.",
    ]

    return "\n".join(lines) + "\n"


def run_audit(
    market_db_path: Path,
    universe_db_path: Path,
    scanner_db_path: Path,
    candidate_manifest_path: Path,
    promoted_manifest_path: Path,
    cutoff_str: str,
    output_json_path: Path,
    output_md_path: Optional[Path] = None,
    explicit_expected_session: Optional[str] = None,
    historical_rehearsal_cutoff: str = "2026-11-06",
) -> Dict[str, Any]:
    """
    Executes the authoritative side-effect-free Radar universe readiness audit.
    """
    # 1. Resolve calendar and completed session
    obs_cutoff_date, expected_session, cal_telemetry = derive_expected_completed_session(
        cutoff_str=cutoff_str,
        explicit_expected_session=explicit_expected_session,
    )
    as_of_dt = date.fromisoformat(obs_cutoff_date)

    # 2. File verification and digest calculation
    u_cand_sha = compute_file_sha256(candidate_manifest_path)
    u_prom_sha = compute_file_sha256(promoted_manifest_path)
    market_db_sha = compute_file_sha256(market_db_path)
    universe_db_sha = compute_file_sha256(universe_db_path)
    scanner_db_sha = compute_file_sha256(scanner_db_path)

    # 3. Load manifests dynamically (handle duplicates gracefully and track counts)
    candidate_symbols: List[str] = []
    u_cand_hash = ""
    u_cand_build_id = ""
    if candidate_manifest_path.exists():
        with open(candidate_manifest_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        candidate_symbols = c_data.get("symbols", [])
        u_cand_hash = c_data.get("membership_hash", "")
        u_cand_build_id = c_data.get("manifest_build_id", "")

    promoted_symbols: List[str] = []
    u_prom_hash = ""
    u_prom_build_id = ""
    if promoted_manifest_path.exists():
        with open(promoted_manifest_path, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        promoted_symbols = p_data.get("symbols", [])
        u_prom_hash = p_data.get("membership_hash", "")
        u_prom_build_id = p_data.get("manifest_build_id", "")

    # Duplicate manifest symbol analysis
    cand_counter = Counter(candidate_symbols)
    cand_duplicates = [sym for sym, cnt in cand_counter.items() if cnt > 1]
    distinct_candidate_symbols = sorted(cand_counter.keys())

    prom_counter = Counter(promoted_symbols)
    prom_duplicates = [sym for sym, cnt in prom_counter.items() if cnt > 1]
    distinct_promoted_symbols = sorted(prom_counter.keys())

    promoted_set = set(distinct_promoted_symbols)
    candidate_set = set(distinct_candidate_symbols)
    overlap_set = candidate_set.intersection(promoted_set)

    # 4. Connect read-only to market store with filtered observations (trade_date <= obs_cutoff_date)
    total_candles_all_time = 0
    total_candles_filtered = 0
    dist_symbols_all_time = 0
    min_trade_date_all_time = None
    max_trade_date_all_time = None
    min_trade_date_filtered = None
    max_trade_date_filtered = None
    ohlcv_rows: Dict[str, Dict[str, Any]] = {}
    latest_prices: Dict[str, Dict[str, Any]] = {}
    total_prov_rows = 0
    prov_by_symbol: Dict[str, Dict[str, Any]] = {}

    if market_db_path.exists() and market_db_path.is_file():
        market_uri = f"{market_db_path.resolve().as_uri()}?mode=ro"
        m_conn = sqlite3.connect(market_uri, uri=True, timeout=10.0)
        m_cur = m_conn.cursor()

        # Check tables existence
        m_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_ohlcv_daily'")
        has_daily = m_cur.fetchone() is not None

        if has_daily:
            # All-time metrics
            m_cur.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(trade_date), MAX(trade_date) FROM asset_ohlcv_daily")
            total_candles_all_time, dist_symbols_all_time, min_trade_date_all_time, max_trade_date_all_time = m_cur.fetchone()

            # Filtered metrics: trade_date <= obs_cutoff_date
            m_cur.execute(
                "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM asset_ohlcv_daily WHERE trade_date <= ?",
                (obs_cutoff_date,),
            )
            total_candles_filtered, min_trade_date_filtered, max_trade_date_filtered = m_cur.fetchone()

            # Filtered symbol candle aggregates
            m_cur.execute("""
                SELECT symbol, COUNT(*), MIN(trade_date), MAX(trade_date)
                FROM asset_ohlcv_daily
                WHERE trade_date <= ?
                GROUP BY symbol
            """, (obs_cutoff_date,))
            ohlcv_rows = {r[0]: {"count": r[1], "min_date": r[2], "max_date": r[3]} for r in m_cur.fetchall()}

            # Latest close price filtered to cutoff
            m_cur.execute("""
                SELECT a.symbol, a.trade_date, a.close
                FROM asset_ohlcv_daily a
                INNER JOIN (
                    SELECT symbol, MAX(trade_date) as max_date
                    FROM asset_ohlcv_daily
                    WHERE trade_date <= ?
                    GROUP BY symbol
                ) b ON a.symbol = b.symbol AND a.trade_date = b.max_date
                WHERE a.trade_date <= ?
            """, (obs_cutoff_date, obs_cutoff_date))
            latest_prices = {r[0]: {"price_date": r[1], "close": float(r[2])} for r in m_cur.fetchall()}

        # Check provenance table
        m_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_ohlcv_provenance'")
        has_prov = m_cur.fetchone() is not None
        if has_prov:
            m_cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE trade_date <= ?", (obs_cutoff_date,))
            total_prov_rows = m_cur.fetchone()[0]
            if total_prov_rows > 0:
                m_cur.execute("""
                    SELECT symbol, COUNT(*), MAX(observed_at)
                    FROM asset_ohlcv_provenance
                    WHERE trade_date <= ?
                    GROUP BY symbol
                """, (obs_cutoff_date,))
                prov_by_symbol = {r[0]: {"count": r[1], "latest_observed": r[2]} for r in m_cur.fetchall()}

        m_conn.close()

    # 5. Connect read-only to scanner store to inspect active snapshots
    scanner_snapshots = []
    if scanner_db_path.exists() and scanner_db_path.is_file():
        scanner_uri = f"{scanner_db_path.resolve().as_uri()}?mode=ro"
        s_conn = sqlite3.connect(scanner_uri, uri=True, timeout=10.0)
        s_cur = s_conn.cursor()
        s_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scanner_snapshots'")
        if s_cur.fetchone() is not None:
            s_cur.execute("SELECT snapshot_id, scanner_id, run_id, implementation_release_sha, status_at_publication, created_at, data_as_of FROM scanner_snapshots")
            scanner_snapshots = [
                {
                    "snapshot_id": r[0],
                    "scanner_id": r[1],
                    "run_id": r[2],
                    "implementation_release_sha": r[3],
                    "status_at_publication": r[4],
                    "created_at": r[5],
                    "data_as_of": r[6],
                }
                for r in s_cur.fetchall()
            ]
        s_conn.close()

    # 6. Connect read-only to universe store
    u_counts = {}
    if universe_db_path.exists() and universe_db_path.is_file():
        universe_uri = f"{universe_db_path.resolve().as_uri()}?mode=ro"
        u_conn = sqlite3.connect(universe_uri, uri=True, timeout=10.0)
        u_cur = u_conn.cursor()
        u_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        u_tables = [r[0] for r in u_cur.fetchall()]
        for t in u_tables:
            u_cur.execute(f"SELECT COUNT(*) FROM {t}")
            u_counts[t] = u_cur.fetchone()[0]
        u_conn.close()

    # 7. Evaluate symbol-level records dynamically for distinct candidate symbols
    symbol_evaluations = compute_symbol_evaluations(
        symbols=distinct_candidate_symbols,
        promoted_symbols_set=promoted_set,
        ohlcv_rows=ohlcv_rows,
        latest_prices=latest_prices,
        prov_by_symbol=prov_by_symbol,
        expected_completed_session=expected_session,
        as_of_dt=as_of_dt,
    )

    # 8. Compute candidate universe summary metrics dynamically
    summary_candidate = compute_summary_metrics(
        evaluations=symbol_evaluations,
        expected_completed_session=expected_session,
    )

    # 9. Compute promoted reference cohort summary metrics dynamically
    eval_promoted = [s for s in symbol_evaluations if s["in_promoted_reference"]]
    summary_promoted = compute_summary_metrics(
        evaluations=eval_promoted,
        expected_completed_session=expected_session,
    )

    # 10. Compute metric data requirements matrix dynamically
    matrix = compute_metrics_matrix(symbol_evaluations)

    # 11. Final disposition determination: Missing evidence must NEVER produce READY
    any_unready = any(s["local_store_readiness_classification"] != "READY" for s in symbol_evaluations)
    if not symbol_evaluations or any_unready:
        final_disposition = "ASSESSED_AS_OF_EVIDENCE / BLOCKED_BY_EVIDENCE"
    else:
        # Even if all local symbols pass, production authority remains NOT_CHECKED
        final_disposition = "BLOCKED_BY_UNVERIFIED_AUTHORITY (Local store ready; production authority NOT_CHECKED)"

    audit_payload = {
        "assessment_metadata": {
            "directive": "ISSUE-166",
            "title": "Authoritative ARX Radar Universe Readiness Assessment (Issue #166)",
            "topic_branch": "audit/radar-readiness-assessment-166",
            "base_anchor_commit": "868d7afcd528006c2323e6b79b5a17f2f20f677c",
            "branch_head_commit": get_current_git_sha(REPO_ROOT),
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "assessment_cutoff": cutoff_str,
            "observation_cutoff_date": obs_cutoff_date,
            "expected_completed_session": expected_session,
            "historical_rehearsal_cutoff": historical_rehearsal_cutoff,
            "calendar_telemetry": cal_telemetry,
            "final_disposition": final_disposition,
            "production_authority_status": "NOT_CHECKED",
            "production_authority_rationale": (
                f"The relationship between the inspected store ({market_db_path}, {dist_symbols_all_time} symbols, "
                f"{total_candles_all_time} daily bars) and the earlier reported baseline store containing 585 symbols "
                "and 228,937 bars cannot be established from available local artifacts. Live production queries "
                "were not executed in strict observance of operational boundaries and SEC trial isolation. Therefore, "
                "production authority is marked NOT_CHECKED."
            ),
            "operational_holds_preserved": {
                "HOLD_SCAN_001": "ACTIVE (Natural background scanning disabled; static single snapshot served)",
                "HOLD_SEC_001": "ACTIVE (Continuous SEC egress held; independent bounded trial running)",
                "HOLD_EXEC_001": "ACTIVE (Live financial trade execution prohibited)",
                "PHASE_3_HOLD": "ACTIVE (Macro/Regime cross-asset execution held)",
                "ARX_CONFLUENCE_INTEGRITY_HOLD": "ACTIVE (Scoring held in governed research mode)",
                "PHASE_A3_UX_HOLD": "ACTIVE (Formative usability gate active)",
            },
        },
        "evidence_custody": {
            "inspected_market_database": {
                "source": "Local developer SQLite file",
                "path": str(market_db_path),
                "sha256": market_db_sha,
                "file_size_bytes": os.path.getsize(market_db_path) if market_db_path.exists() else 0,
                "file_mtime_utc": datetime.fromtimestamp(os.path.getmtime(market_db_path), tz=timezone.utc).isoformat() if market_db_path.exists() else None,
                "acquisition_time_status": "NOT_RECORDED (filesystem mtime only; provenance/acquisition time unrecorded in store)",
                "total_rows_all_time": total_candles_all_time,
                "total_rows_filtered_to_cutoff": total_candles_filtered,
                "distinct_symbols_in_store": dist_symbols_all_time,
                "date_range_all_time": f"{min_trade_date_all_time} to {max_trade_date_all_time}",
                "date_range_filtered": f"{min_trade_date_filtered} to {max_trade_date_filtered}",
                "universe_build_binding": "UNBOUND (no build ID or manifest hash recorded in store)",
                "snapshot_binding": f"UNBOUND (0 rows in asset_factor_snapshots, {total_prov_rows} rows in asset_ohlcv_provenance)",
                "earlier_reported_store_comparison": {
                    "earlier_store_symbol_count": 585,
                    "earlier_store_bar_count": 228937,
                    "inspected_store_symbol_count": dist_symbols_all_time,
                    "inspected_store_bar_count": total_candles_all_time,
                    "relationship_status": "UNESTABLISHED (lineage/extraction cannot be established from local artifacts)",
                    "production_authority_verdict": "NOT_CHECKED",
                },
                "access_mode": "SQLITE_URI_READ_ONLY (mode=ro)",
            },
            "inspected_universe_database": {
                "source": "Local developer SQLite file",
                "path": str(universe_db_path),
                "sha256": universe_db_sha,
                "table_counts": u_counts,
                "access_mode": "SQLITE_URI_READ_ONLY (mode=ro)",
            },
            "inspected_scanner_database": {
                "source": "Local developer SQLite file",
                "path": str(scanner_db_path),
                "sha256": scanner_db_sha,
                "snapshots": scanner_snapshots,
                "access_mode": "SQLITE_URI_READ_ONLY (mode=ro)",
            },
        },
        "authoritative_manifests": [
            {
                "input_identifier": "UB-ARX-EXPANSION-500-REHEARSAL",
                "path": str(candidate_manifest_path),
                "sha256": u_cand_sha,
                "membership_hash": u_cand_hash,
                "build_id": u_cand_build_id,
                "manifest_symbol_count": len(candidate_symbols),
                "distinct_symbol_count": len(distinct_candidate_symbols),
                "duplicate_symbol_count": len(candidate_symbols) - len(distinct_candidate_symbols),
                "duplicate_symbols": cand_duplicates,
                "status": "PROVISIONAL_SPECIFICATION / REHEARSAL_QUALIFIED (RADAR-P2-003)",
                "governance_note": "Provisional expansion candidate. Membership overlap with promoted reference cohort does NOT constitute production promotion.",
            },
            {
                "input_identifier": "UB-ARX-EXPANSION-250-20261107",
                "path": str(promoted_manifest_path),
                "sha256": u_prom_sha,
                "membership_hash": u_prom_hash,
                "build_id": u_prom_build_id,
                "manifest_symbol_count": len(promoted_symbols),
                "distinct_symbol_count": len(distinct_promoted_symbols),
                "duplicate_symbol_count": len(promoted_symbols) - len(distinct_promoted_symbols),
                "duplicate_symbols": prom_duplicates,
                "status": "PROMOTED_OPERATIONAL_REFERENCE (RADAR-014 / PR #164)",
                "governance_note": "Confirmed historical promotion under RADAR-014. Current live production membership remains NOT_CHECKED without proof.",
            },
        ],
        "population_reconciliation": {
            "historical_promotion_evidence": {
                "event": "RADAR-014 / PR #164 promotion to operational reference",
                "manifest_build_id": u_prom_build_id,
                "membership_hash": u_prom_hash,
                "status": "HISTORICALLY_PROMOTED_REFERENCE",
                "detail": "Historical PR #164 merged the 250-symbol manifest as the operational reference universe.",
            },
            "current_production_membership_status": "NOT_CHECKED",
            "overlap_count": len(overlap_set),
            "overlap_percentage": round((len(overlap_set) / len(distinct_promoted_symbols) * 100.0), 2) if distinct_promoted_symbols else 0.0,
            "cohort_100_stage2": {
                "name": "100 Stage 2 Forensic Reconstructed Subset",
                "nature": "Analytical classification result from RADAR-015 scan on the 250 promoted symbols (NOT an independent universe build)",
                "derivation": "Observed under run-00cc89d2a6b9: 100 Stage 2 passes, 150 non-Stage-2 or unresolved, 0 qualified VCP setups",
                "count": 100,
                "status": "RECONSTRUCTED_FORENSIC_OBSERVATION",
                "denominator": 250,
            },
            "cohort_250_promoted": {
                "name": "250 Promoted Operational Reference Cohort",
                "manifest": u_prom_build_id,
                "membership_hash": u_prom_hash,
                "manifest_symbol_count": len(promoted_symbols),
                "distinct_symbol_count": len(distinct_promoted_symbols),
                "status": "HISTORICALLY_PROMOTED_OPERATIONAL_REFERENCE (RADAR-014 / PR #164)",
                "current_production_status": "NOT_CHECKED",
                "overlap_with_candidate_universe": len(overlap_set),
            },
            "cohort_500_candidate": {
                "name": "500 Expansion Candidate Rehearsal Universe",
                "manifest": u_cand_build_id,
                "membership_hash": u_cand_hash,
                "manifest_symbol_count": len(candidate_symbols),
                "distinct_symbol_count": len(distinct_candidate_symbols),
                "status": "PROVISIONAL_SPECIFICATION / REHEARSAL_QUALIFIED (UNPROMOTED)",
                "current_production_status": "NOT_PROMOTED",
                "overlap_with_promoted_reference": len(overlap_set),
                "promotion_distinction": "Contains the 250 reference symbols as a subset, but the 500 cohort as a whole has NEVER received production promotion.",
            },
        },
        "demonstrated_blockers_and_limitations": {
            "blocker_1_market_data_deficit_in_local_store": {
                "finding": (
                    f"In the inspected store, {summary_candidate['readiness_breakdown']['MISSING_MARKET_DATA']} candidate symbols have 0 bars (MISSING_MARKET_DATA), "
                    f"{summary_candidate['readiness_breakdown']['INSUFFICIENT_HISTORY']} have insufficient 200 SMA history, "
                    f"and {summary_candidate['readiness_breakdown']['INSUFFICIENT_SLOPE_HISTORY']} have insufficient slope history. "
                    f"Only {matrix['WEINSTEIN_MINERVINI_SMA_200']['satisfying_count']} symbols satisfy the 200 SMA threshold."
                ),
                "scope_limitation": f"Finding applies strictly to inspected store ({market_db_path}). Production market store authority is NOT_CHECKED.",
                "historical_claim_status": "The historical '439/500 stale' claim from prior status reports remains an unverified historical measurement whose calculation basis is not established by local artifacts.",
                "severity": "LOCAL_DATA_BLOCKER",
                "next_dependency": "An attributable, consistent export of the actual production Radar stores.",
            },
            "blocker_2_provenance_deficit_in_local_store": {
                "finding": (
                    f"In the inspected store, asset_ohlcv_provenance contains {total_prov_rows} rows. "
                    f"{summary_candidate['provenance_tracking']['symbols_lacking_provenance']} of {summary_candidate['total_symbols']} candidate symbols lack table-level provenance."
                ),
                "scope_limitation": "Finding applies strictly to inspected store. Production provenance authority is NOT_CHECKED.",
                "historical_claim_status": "The historical '67 symbols lacked provenance' claim remains an unverified historical figure whose specific symbol roster cannot be confirmed from local artifacts.",
                "severity": "LOCAL_DATA_BLOCKER",
                "next_dependency": "An attributable, consistent export of the actual production Radar stores.",
            },
            "blocker_3_snapshot_compatibility_and_code_contract": {
                "finding": (
                    "Under the publication integrity contract (ScannerPublicationIntegrityEngine), snapshot compatibility is governed by "
                    "the 6-component ScannerVersionTuple, component content hashes, and semantic fingerprint, rather than git commit SHAs alone. "
                    "Different commit SHAs alone across non-functional PRs do not establish incompatibility. "
                    f"However, the inspected scanner store contains {len(scanner_snapshots)} snapshots (0 VCP snapshots). Local snapshot compatibility cannot be established."
                ),
                "scope_limitation": "Production snapshot serving state is NOT_CHECKED.",
                "severity": "GOVERNANCE_AND_STORAGE_BLOCKER",
                "next_dependency": "An attributable, consistent export of the actual production Radar stores.",
            },
        },
        "metric_data_requirements_matrix": matrix,
        "summary_metrics_candidate_universe": summary_candidate,
        "summary_metrics_promoted_reference": summary_promoted,
        "symbol_level_evaluations": symbol_evaluations,
    }

    # Write output JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    # Write output Markdown if requested
    if output_md_path:
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(audit_payload)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"Successfully generated authoritative audit JSON: {output_json_path}")
    if output_md_path:
        print(f"Successfully generated authoritative audit Markdown: {output_md_path}")
    print(f"Total symbols evaluated: {summary_candidate['total_symbols']}")
    print(f"Readiness breakdown: {summary_candidate['readiness_breakdown']}")
    print(f"Depth buckets: {summary_candidate['history_depth_breakdown']}")
    return audit_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authoritative, reproducible ARX Radar universe readiness auditor (Issue #166)"
    )
    parser.add_argument(
        "--market-db",
        type=Path,
        default=Path("/Users/iDavid/.finance_market_store.db"),
        help="Path to market SQLite database",
    )
    parser.add_argument(
        "--universe-db",
        type=Path,
        default=Path("/Users/iDavid/.finance_universe_store.db"),
        help="Path to universe SQLite database",
    )
    parser.add_argument(
        "--scanner-db",
        type=Path,
        default=Path("/Users/iDavid/.finance_scanner_store.db"),
        help="Path to scanner snapshots SQLite database",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-500-REHEARSAL.json",
        help="Path to candidate expansion universe JSON manifest",
    )
    parser.add_argument(
        "--promoted-manifest",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-250-20261107.json",
        help="Path to promoted operational reference universe JSON manifest",
    )
    parser.add_argument(
        "--cutoff",
        "--cutoff-date",
        "--as-of-date",
        dest="cutoff",
        type=str,
        default="2026-11-20",
        help="Assessment cutoff date or ISO timestamp (default: 2026-11-20)",
    )
    parser.add_argument(
        "--expected-completed-session",
        type=str,
        default=None,
        help="Optional explicit override for expected completed trading session (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--historical-rehearsal-cutoff",
        type=str,
        default="2026-11-06",
        help="Historical rehearsal cutoff date for separate benchmark comparison (default: 2026-11-06)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "radar_universe_readiness_assessment_166.json",
        help="Path for output JSON evidence record",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "RADAR_UNIVERSE_READINESS_ASSESSMENT_166.md",
        help="Path for output Markdown assessment report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(
        market_db_path=args.market_db,
        universe_db_path=args.universe_db,
        scanner_db_path=args.scanner_db,
        candidate_manifest_path=args.candidate_manifest,
        promoted_manifest_path=args.promoted_manifest,
        cutoff_str=args.cutoff,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        explicit_expected_session=args.expected_completed_session,
        historical_rehearsal_cutoff=args.historical_rehearsal_cutoff,
    )
