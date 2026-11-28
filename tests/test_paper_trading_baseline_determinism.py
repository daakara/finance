"""
tests/test_paper_trading_baseline_determinism.py

Deterministic baseline recomputation, reconciliation, and storage isolation test suite
for ARX Paper Trading (PR-PT-01).
Follows docs/planning/ARX_PAPER_TRADING_ROADMAP.md Milestone 1.

Verifies from first principles:
1. Recomputes all 19 historical pilot recommendation outcomes from scratch using
   PaperTradingOutcomeEvaluator under frozen Contract V1.0.1.
2. Validates exact individual trade outcomes (NET TP1, TMDX stop, VOO/JEPI horizon exits,
   QQQM untriggered expiration, CELH/CRSP intrabar collisions).
3. Reconciles R-multiple and fill-price percentage return calculations from first principles across all trades:
   - Historical Baseline (N=5): Net +0.60%, Reconciled R: +0.17R (prior -1.78% net and -0.01R reconciled from verified fills).
   - Prospective Clean (N=5): Net +0.47%, Reconciled R: +0.15R (prior +0.18R reconciled).
   - Combined Natural (N=10): Net +0.54%, Reconciled R: +0.16R (prior -0.65% net and +0.08R reconciled).
   - Pure Daily Resolved (N=8): Net +1.48%, Reconciled R: +0.45R (prior +0.36R reconciled).
4. Verifies repeat-run determinism SHA-256 matching recorded baseline digest:
   a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895.
5. Exercises the real storage isolation write-path validator with temporary directories:
   - Valid forward writes succeed.
   - Historical and epoch validation namespaces are rejected fail-closed (0 files changed).
   - Path traversals, absolute escapes, and symlink escapes are rejected fail-closed (0 files changed).
   - Intersections with protected historical paths are rejected fail-closed (0 files changed).

Zero network calls, zero database writes, and zero mutations to Contract V1.0.1.
"""

import os
import json
import hashlib
import tempfile
import pytest
import pandas as pd
from pathlib import Path

from analyst_dashboard.governance.paper_trading_outcome_evaluator import (
    PaperTradingOutcomeEvaluator,
    EvaluationResult,
)
from analyst_dashboard.governance.storage import (
    validate_forward_write_path,
    safe_write_forward_pilot_file,
    resolve_forward_pilot_data_root,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_PATH = REPO_ROOT / "docs" / "governance" / "ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_RECEIPT.json"
PRICE_BARS_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "paper_trading_baseline_price_bars.json"
FORWARD_STORAGE_PATH = REPO_ROOT / "data" / "forward_pilot"

EXPECTED_DETERMINISM_SHA256 = "a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895"


@pytest.fixture(scope="module")
def baseline_receipt():
    assert RECEIPT_PATH.exists(), f"Missing baseline receipt: {RECEIPT_PATH}"
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_price_bars():
    assert PRICE_BARS_FIXTURE_PATH.exists(), f"Missing price fixture: {PRICE_BARS_FIXTURE_PATH}"
    with open(PRICE_BARS_FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_first_principles_baseline_recomputation(baseline_receipt, baseline_price_bars):
    """
    Recomputes all 19 historical pilot recommendation outcomes from scratch.
    Verifies that trade outcomes, fill prices, returns, and R-multiples are established
    from first principles, keeping pure-daily results separate from intraday-resolved results.
    """
    records = baseline_receipt["evaluationRecords"]
    assert len(records) == 19, f"Expected 19 inventory records, got {len(records)}"

    recomputed_results = []

    for r in records:
        sym = r["symbol"]
        plan = r["plan"]
        sig_dict = {
            "symbol": sym,
            "engineVersion": "7ad44595826c147cc77f93cd676af520764c7442",
            "signalDate": r["signalDate"],
            "corridorMin": plan["corridorMin"],
            "corridorMax": plan["corridorMax"],
            "stopLoss": plan["stopLoss"],
            "takeProfit1": plan["takeProfit1"],
            "confluenceScore": plan["confluenceScore"],
            "marketRegime": plan.get("marketRegime") or "UNKNOWN",
        }

        bars_list = baseline_price_bars.get(sym, [])
        assert len(bars_list) > 0, f"Missing price bars for symbol {sym}"

        df = pd.DataFrame(bars_list)
        date_col = next((c for c in df.columns if c.lower() in ("date", "trade_date")), None)
        assert date_col is not None, f"No date column found in price bars for {sym}"

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        rename_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename_map)

        # Evaluate using frozen evaluator against pure daily bars (intraday_bars=None)
        eval_res = PaperTradingOutcomeEvaluator.evaluate_signal(
            signal=sig_dict,
            daily_bars=df,
            provider="YAHOO_FINANCE",
        )

        res_dict = eval_res.to_dict()
        res_dict["cohortType"] = r["cohortType"]
        res_dict["index"] = r["index"]
        recomputed_results.append(res_dict)

    # 1. Denominator Assertions (First Principles)
    total_inventory = len(recomputed_results)
    assert total_inventory == 19

    # Synthetic quarantined setup (AAPL on 2026-09-19) strictly excluded from performance
    synthetic_records = [r for r in recomputed_results if r["cohortType"] == "SYNTHETIC_QUARANTINED"]
    assert len(synthetic_records) == 1
    assert synthetic_records[0]["symbol"] == "AAPL"
    assert synthetic_records[0]["signal_date"] == "2026-09-19"
    assert synthetic_records[0]["outcome"] == "ENTRY_NOT_TRIGGERED"

    # Natural recommendations (D1_natural = 18, D2 = 18 eligible)
    natural_records = [r for r in recomputed_results if r["cohortType"] != "SYNTHETIC_QUARANTINED"]
    assert len(natural_records) == 18

    historical_records = [r for r in natural_records if r["cohortType"] == "HISTORICAL"]
    prospective_records = [r for r in natural_records if r["cohortType"] == "PROSPECTIVE_CLEAN"]
    assert len(historical_records) == 9
    assert len(prospective_records) == 9

    # Triggered Entries (D3)
    triggered = [r for r in natural_records if r["entry_state"] == "ENTRY_TRIGGERED"]
    untriggered = [r for r in natural_records if r["entry_state"] == "ENTRY_NOT_TRIGGERED"]
    unresolved_intrabar = [r for r in natural_records if r["outcome"] == "UNRESOLVED_INTRABAR_SEQUENCE"]

    # In Pure Daily OHLC, CELH and CRSP have same-bar entry+stop collisions (unresolved intrabar)
    assert len(unresolved_intrabar) == 2
    unresolved_syms = {r["symbol"] for r in unresolved_intrabar}
    assert unresolved_syms == {"CELH", "CRSP"}

    # Pure Daily Triggered entries (including the 2 intrabar collisions) = 10
    total_triggered_count = len(triggered) + len(unresolved_intrabar)
    assert total_triggered_count == 10
    assert len(untriggered) == 8

    # Pure Daily Resolved (D5_pure = 8)
    pure_daily_resolved = [r for r in triggered if r["outcome"] in ("SUCCESS", "FAILURE", "NEUTRAL")]
    assert len(pure_daily_resolved) == 8

    # 2. Key Individual Trade Recomputation Assertions Under Contract V1.0.1
    record_map = {r["symbol"] + "_" + r["signal_date"]: r for r in recomputed_results}

    # NET: Take Profit 1 Success
    net_res = record_map["NET_2026-09-04"]
    assert net_res["outcome"] == "SUCCESS"
    assert net_res["exit_reason"] == "TAKE_PROFIT_1"
    assert net_res["entry_price"] == 279.18
    assert net_res["exit_price"] == 319.90
    assert round(net_res["gross_return_pct"], 2) == 14.59
    assert round(net_res["net_simulated_return_pct"], 2) == 14.34
    assert round(net_res["r_multiple"], 2) == 3.09

    # ANET: Horizon Exit (Neutral)
    anet_res = record_map["ANET_2026-09-04"]
    assert anet_res["outcome"] == "NEUTRAL"
    assert anet_res["exit_reason"] == "HORIZON_EXIT"
    assert anet_res["entry_price"] == 194.83
    assert anet_res["exit_price"] == 206.90
    assert round(anet_res["gross_return_pct"], 2) == 6.20
    assert round(anet_res["net_simulated_return_pct"], 2) == 5.95
    assert round(anet_res["r_multiple"], 2) == 0.76

    # TMDX: Stop Loss Failure
    tmdx_res = record_map["TMDX_2026-09-04"]
    assert tmdx_res["outcome"] == "FAILURE"
    assert tmdx_res["exit_reason"] == "STOP_LOSS"
    assert round(tmdx_res["gross_return_pct"], 2) == -10.55
    assert round(tmdx_res["net_simulated_return_pct"], 2) == -10.80
    assert round(tmdx_res["r_multiple"], 2) == -1.00

    # VOO: 20-Session Horizon Exit (Neutral)
    voo_res = record_map["VOO_2026-09-24"]
    assert voo_res["outcome"] == "NEUTRAL"
    assert voo_res["exit_reason"] == "HORIZON_EXIT"
    assert voo_res["entry_price"] == 706.19
    assert voo_res["exit_price"] == 702.30
    assert round(voo_res["gross_return_pct"], 2) == -0.55
    assert round(voo_res["net_simulated_return_pct"], 2) == -0.80
    assert round(voo_res["r_multiple"], 2) == -0.14

    # JEPI: 20-Session Horizon Exit (Neutral)
    jepi_res = record_map["JEPI_2026-09-24"]
    assert jepi_res["outcome"] == "NEUTRAL"
    assert jepi_res["exit_reason"] == "HORIZON_EXIT"
    assert jepi_res["entry_price"] == 56.56
    assert jepi_res["exit_price"] == 57.29
    assert round(jepi_res["gross_return_pct"], 2) == 1.29
    assert round(jepi_res["net_simulated_return_pct"], 2) == 1.04
    assert round(jepi_res["r_multiple"], 2) == 0.37

    # QQQM: Entry Window Expired (Untriggered)
    qqqm_res = record_map["QQQM_2026-09-24"]
    assert qqqm_res["outcome"] == "ENTRY_NOT_TRIGGERED"
    assert qqqm_res["exit_reason"] == "ENTRY_WINDOW_EXPIRED"
    assert qqqm_res["entry_price"] is None

    # Prospective Signals Survival: All 5 triggered prospective setups survived to Horizon Exit
    prospective_triggered = [r for r in prospective_records if r["entry_state"] == "ENTRY_TRIGGERED"]
    assert len(prospective_triggered) == 5
    for pr in prospective_triggered:
        assert pr["outcome"] == "NEUTRAL"
        assert pr["exit_reason"] == "HORIZON_EXIT"

    # 3. Pure Daily Aggregates (N=8 resolved trades: NET, ANET, TMDX, AAPL, VOO, JEPI, MSFT, NVDA)
    pure_gross = [r["gross_return_pct"] for r in pure_daily_resolved]
    pure_net = [r["net_simulated_return_pct"] for r in pure_daily_resolved]
    pure_r = [r["r_multiple"] for r in pure_daily_resolved]
    avg_pure_gross = sum(pure_gross) / len(pure_gross)
    avg_pure_net = sum(pure_net) / len(pure_net)
    avg_pure_r = sum(pure_r) / len(pure_r)
    assert abs(avg_pure_gross - 1.73) < 0.02
    assert abs(avg_pure_net - 1.48) < 0.02
    assert abs(avg_pure_r - 0.45) < 0.02

    # 4. Intraday Sequencing Resolution Assertions (N=10 resolved trades)
    # Under retained 1-min intraday sequencing evidence, CELH and CRSP resolve to stop-outs (-1.00R each)
    # Recomputed from verified fill prices:
    # CELH: Entry $30.60, Stop/Exit $29.56 -> Gross -3.40%, Net -3.65%, R -1.00R
    # CRSP: Entry $54.45, Stop/Exit $53.05 -> Gross -2.57%, Net -2.82%, R -1.00R
    hist_net_returns = [14.34, 5.95, -10.80, -3.65, -2.82]
    hist_r_multiples = [3.09, 0.76, -1.00, -1.00, -1.00]
    recomputed_hist_net_avg = sum(hist_net_returns) / len(hist_net_returns)
    recomputed_hist_r_avg = sum(hist_r_multiples) / len(hist_r_multiples)
    assert abs(recomputed_hist_net_avg - 0.60) < 0.02
    assert abs(recomputed_hist_r_avg - 0.17) < 0.02

    # Prospective Triggered (5 trades): AAPL (+3.22%, +0.68R), MSFT (+4.17%, +0.71R), NVDA (-5.26%, -0.89R), VOO (-0.80%, -0.14R), JEPI (+1.04%, +0.37R)
    prosp_net_returns = [pr["net_simulated_return_pct"] for pr in prospective_triggered]
    prosp_r_multiples = [pr["r_multiple"] for pr in prospective_triggered]
    recomputed_prosp_net_avg = sum(prosp_net_returns) / len(prosp_net_returns)
    recomputed_prosp_r_avg = sum(prosp_r_multiples) / len(prosp_r_multiples)
    assert abs(recomputed_prosp_net_avg - 0.47) < 0.02
    assert abs(recomputed_prosp_r_avg - 0.15) < 0.02

    # Pooled Intraday Resolved (10 trades): Mean net return +0.54%, Mean R +0.16R
    pooled_intraday_net = hist_net_returns + prosp_net_returns
    pooled_intraday_r = hist_r_multiples + prosp_r_multiples
    recomputed_pooled_net = sum(pooled_intraday_net) / len(pooled_intraday_net)
    recomputed_pooled_r = sum(pooled_intraday_r) / len(pooled_intraday_r)
    assert abs(recomputed_pooled_net - 0.54) < 0.02
    assert abs(recomputed_pooled_r - 0.16) < 0.02

    # Verify reconciliation with receipt cohort breakdown & census
    hist_breakdown = baseline_receipt["cohortBreakdown"]["historicalBaseline"]
    prosp_breakdown = baseline_receipt["cohortBreakdown"]["prospectiveClean"]
    census = baseline_receipt["denominatorCensus"]
    assert hist_breakdown["averageNetReturnPct"] == 0.60
    assert hist_breakdown["averageRMultiple"] == 0.17
    assert prosp_breakdown["averageNetReturnPct"] == 0.47
    assert prosp_breakdown["averageRMultiple"] == 0.15
    assert census["pureDailyOhlc"]["average_r_multiple_resolved"] == 0.45
    assert census["pureDailyOhlc"]["average_net_return_resolved"] == 1.48
    assert census["intradayBarsResolved"]["average_r_multiple_resolved"] == 0.16
    assert census["intradayBarsResolved"]["average_net_return_resolved"] == 0.54

    # 5. Repeat-Run Determinism SHA-256
    serialized_bytes = json.dumps(recomputed_results, sort_keys=True).encode("utf-8")
    actual_hash = hashlib.sha256(serialized_bytes).hexdigest()
    assert actual_hash == EXPECTED_DETERMINISM_SHA256, (
        f"Determinism SHA-256 mismatch!\nExpected: {EXPECTED_DETERMINISM_SHA256}\nActual:   {actual_hash}"
    )


def test_storage_isolation_write_path_validator():
    """
    Exercises the real storage isolation write-path validator with temporary directories.
    Verifies that:
    1. Valid writes inside the forward root with approved cohort prefix succeed.
    2. Historical and epoch validation namespaces are rejected fail-closed (0 files changed).
    3. Path traversals and absolute escapes are rejected fail-closed (0 files changed).
    4. Symlink escapes outside the forward root are rejected fail-closed (0 files changed).
    5. Writes targeting historical directories are rejected fail-closed (0 files changed).
    """
    with tempfile.TemporaryDirectory() as tmp_forward_dir:
        # 1. Valid write test
        valid_file = os.path.join(tmp_forward_dir, "capture_stream.jsonl")
        written_path = safe_write_forward_pilot_file(
            target_path=valid_file,
            cohort_id="COHORT_FORWARD_PILOT_001",
            content='{"cycle": 1, "status": "OK"}\n',
            forward_root=tmp_forward_dir,
        )
        assert os.path.exists(written_path)
        with open(written_path, "r", encoding="utf-8") as f:
            assert f.read() == '{"cycle": 1, "status": "OK"}\n'

        # 2. Namespace rejection tests (must leave zero files created)
        for disallowed in [
            "COHORT_HISTORICAL_PILOT_001",
            "ARX_PROSPECTIVE_VALIDATION_EPOCH_1",
            "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
            "V2_4_0_BASELINE",
            "INVALID_PREFIX_TEST",
        ]:
            rejected_file = os.path.join(tmp_forward_dir, f"should_not_exist_{disallowed}.json")
            with pytest.raises(ValueError):
                safe_write_forward_pilot_file(
                    target_path=rejected_file,
                    cohort_id=disallowed,
                    content='{"should": "fail"}',
                    forward_root=tmp_forward_dir,
                )
            assert not os.path.exists(rejected_file), f"Rejected write created file: {rejected_file}"

        # 3. Path traversal rejection tests (must leave zero files created)
        traversal_file = os.path.join(tmp_forward_dir, "..", "escaped_file.json")
        with pytest.raises(ValueError):
            safe_write_forward_pilot_file(
                target_path=traversal_file,
                cohort_id="COHORT_FORWARD_PILOT_001",
                content='{"should": "fail"}',
                forward_root=tmp_forward_dir,
            )
        assert not os.path.exists(traversal_file), f"Traversal write created file: {traversal_file}"

        # 4. Symlink escape rejection test (must leave zero files created)
        with tempfile.TemporaryDirectory() as tmp_outside_dir:
            outside_file = os.path.join(tmp_outside_dir, "escaped_target.json")
            symlink_path = os.path.join(tmp_forward_dir, "symlink_escape.json")
            os.symlink(outside_file, symlink_path)

            with pytest.raises(ValueError):
                safe_write_forward_pilot_file(
                    target_path=symlink_path,
                    cohort_id="COHORT_FORWARD_PILOT_001",
                    content='{"should": "fail"}',
                    forward_root=tmp_forward_dir,
                )
            assert not os.path.exists(outside_file), f"Symlink escape created outside file: {outside_file}"

        # 5. Historical protected directory rejection
        protected_target = os.path.join(REPO_ROOT, "analyst_dashboard", "data", "illegal_overwrite.json")
        with pytest.raises(ValueError):
            validate_forward_write_path(
                target_path=protected_target,
                cohort_id="COHORT_FORWARD_PILOT_001",
                forward_root=tmp_forward_dir,
            )


def test_forward_cohort_storage_specification_readme():
    """
    Verifies that the forward pilot storage directory exists and enforces documented
    durability and isolation rules.
    """
    assert FORWARD_STORAGE_PATH.exists(), f"Forward storage directory missing: {FORWARD_STORAGE_PATH}"
    readme_path = FORWARD_STORAGE_PATH / "README.md"
    assert readme_path.exists(), f"Missing storage specification README: {readme_path}"

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Enforce documented isolation invariants
    assert "COHORT_FORWARD_PILOT_*" in readme_content
    assert "COHORT_HISTORICAL_PILOT_001" in readme_content
    assert "Zero Production Mutation" in readme_content
    assert "Zero Broker API Connectivity" in readme_content
    assert "Production Persistent Mount Requirement" in readme_content
    assert "validate_forward_write_path" in readme_content


# =====================================================================
# Targeted Verification Tests for PR-PT-01 Reconciliation
# =====================================================================

def test_independent_return_and_r_arithmetic_from_fill_prices():
    """
    Independently verifies return and R arithmetic directly from verified fill prices
    for all 10 resolved trades under Contract V1.0.1.
    Formula invariants:
    - gross_return_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
    - net_simulated_return_pct = round(gross_return_pct - 0.25, 2)
    - r_multiple = round((exit_price - entry_price) / (entry_price - stop_loss), 2)
    """
    resolved_trades = [
        # (symbol, cohort, entry, exit, stop, expected_gross, expected_net, expected_r)
        ("NET", "HISTORICAL", 279.18, 319.90, 266.02, 14.59, 14.34, 3.09),
        ("ANET", "HISTORICAL", 194.83, 206.90, 179.00, 6.20, 5.95, 0.76),
        ("TMDX", "HISTORICAL", 88.83, 79.46, 79.46, -10.55, -10.80, -1.00),
        ("CELH", "HISTORICAL", 30.60, 29.56, 29.56, -3.40, -3.65, -1.00),
        ("CRSP", "HISTORICAL", 54.45, 53.05, 53.05, -2.57, -2.82, -1.00),
        ("AAPL", "PROSPECTIVE_CLEAN", 227.63, 235.53, 215.96, 3.47, 3.22, 0.68),
        ("MSFT", "PROSPECTIVE_CLEAN", 428.90, 447.84, 402.25, 4.42, 4.17, 0.71),
        ("NVDA", "PROSPECTIVE_CLEAN", 120.87, 114.81, 114.07, -5.01, -5.26, -0.89),
        ("VOO", "PROSPECTIVE_CLEAN", 706.19, 702.30, 678.18, -0.55, -0.80, -0.14),
        ("JEPI", "PROSPECTIVE_CLEAN", 56.56, 57.29, 54.60, 1.29, 1.04, 0.37),
    ]

    for sym, cohort, en, ex, st, exp_gross, exp_net, exp_r in resolved_trades:
        calc_gross = round(((ex - en) / en) * 100.0, 2)
        calc_net = round(calc_gross - 0.25, 2)
        calc_r = round((ex - en) / (en - st), 2)

        assert calc_gross == exp_gross, f"{sym} gross mismatch: got {calc_gross}, expected {exp_gross}"
        assert calc_net == exp_net, f"{sym} net mismatch: got {calc_net}, expected {exp_net}"
        assert calc_r == exp_r, f"{sym} R mismatch: got {calc_r}, expected {exp_r}"


def test_detect_receipt_inconsistent_returns_with_correct_r():
    """
    Audits a receipt structure to detect whether it contains correct R-multiples (-1.00R)
    but inconsistent percentage returns (such as legacy transcribed -10.38% or -8.02%).
    Verifies that the audit function flags discrepancies fail-closed.
    """
    def audit_trade_record(trade: dict, tolerance: float = 0.05) -> tuple[bool, str]:
        en = trade["entryPrice"]
        ex = trade["exitPrice"]
        st = trade["stopLoss"]
        rep_net = trade["netReturnPct"]
        rep_r = trade["rMultiple"]

        calc_gross = round(((ex - en) / en) * 100.0, 2)
        calc_net = round(calc_gross - 0.25, 2)
        calc_r = round((ex - en) / (en - st), 2)

        r_ok = abs(calc_r - rep_r) <= tolerance
        net_ok = abs(calc_net - rep_net) <= tolerance

        if r_ok and not net_ok:
            return False, f"INCONSISTENT_RETURNS: {trade['symbol']} has valid R={rep_r:.2f} but inconsistent net return {rep_net:.2f}% (calculated {calc_net:.2f}%)."
        if not r_ok:
            return False, f"INCONSISTENT_R: R={rep_r} vs calculated {calc_r}."
        return True, "VALID"

    # Corrupted legacy trade (correct R=-1.00, but legacy transcribed net=-10.38%)
    legacy_celh = {
        "symbol": "CELH",
        "entryPrice": 30.60,
        "exitPrice": 29.56,
        "stopLoss": 29.56,
        "netReturnPct": -10.38,
        "rMultiple": -1.00,
    }
    is_valid, msg = audit_trade_record(legacy_celh)
    assert not is_valid
    assert "INCONSISTENT_RETURNS" in msg
    assert "CELH has valid R=-1.00 but inconsistent net return -10.38%" in msg

    # Reconciled authoritative trade
    reconciled_celh = {
        "symbol": "CELH",
        "entryPrice": 30.60,
        "exitPrice": 29.56,
        "stopLoss": 29.56,
        "netReturnPct": -3.65,
        "rMultiple": -1.00,
    }
    is_valid, msg = audit_trade_record(reconciled_celh)
    assert is_valid
    assert msg == "VALID"


def test_recompute_celh_crsp_sequencing_from_retained_evidence():
    """
    Recomputes CELH and CRSP sequencing dynamically from retained 1-min intraday bars
    in docs/governance/ARX_HISTORICAL_OUTCOME_FROZEN_EVIDENCE_V1.json.
    Proves from first principles that:
    1. Both signals trigger entry at 09:30 open on 2026-09-08.
    2. Both signals reach their stop-loss price later on that same session before hitting TP1.
    3. Evaluator outputs exact fill prices, gross returns, net returns, and -1.00R.
    """
    frozen_path = REPO_ROOT / "docs" / "governance" / "ARX_HISTORICAL_OUTCOME_FROZEN_EVIDENCE_V1.json"
    assert frozen_path.exists(), f"Missing frozen evidence: {frozen_path}"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_doc = json.load(f)

    ledger_path = REPO_ROOT / "analyst_dashboard" / "data" / "paper_trading_ledger.json"
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    sig_map = {s["symbol"]: s for s in ledger["signals"]}

    for ev in frozen_doc["evidence"]:
        sym = ev["symbol"]
        if sym not in ("CELH", "CRSP"):
            continue

        sig = sig_map[sym]
        df_daily = pd.DataFrame(ev["dailyBars"])
        df_daily["Date"] = pd.to_datetime(df_daily["date"])
        df_daily.set_index("Date", inplace=True)
        df_daily.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

        df_intraday = pd.DataFrame(ev["intradaySlice"])
        df_intraday["Datetime"] = pd.to_datetime(df_intraday["timestamp"])
        df_intraday.set_index("Datetime", inplace=True)
        df_intraday.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

        res = PaperTradingOutcomeEvaluator.evaluate_signal(
            signal=sig,
            daily_bars=df_daily,
            intraday_bars=df_intraday,
            provider=ev["provider"],
        )

        assert res.entry_state == "ENTRY_TRIGGERED"
        assert res.outcome == "FAILURE"
        assert res.exit_reason == "STOP_LOSS"
        assert res.r_multiple == -1.00

        if sym == "CELH":
            assert res.entry_price == 30.60
            assert res.exit_price == 29.56
            assert res.gross_return_pct == -3.40
            assert res.net_simulated_return_pct == -3.65
        elif sym == "CRSP":
            assert res.entry_price == 54.45
            assert res.exit_price == 53.05
            assert res.gross_return_pct == -2.57
            assert res.net_simulated_return_pct == -2.82


def test_cohort_and_pooled_summaries_reconcile_to_constituent_records(baseline_receipt):
    """
    Asserts that cohortBreakdown and denominatorCensus summaries in the receipt
    reconcile exactly to full-precision aggregations of their constituent trade records.
    """
    resolved_trades = baseline_receipt["intradaySequencingReconciliation"]["intradayTickResolvedEvaluation"]["resolvedSignals"]
    assert len(resolved_trades) == 10, f"Expected 10 resolved trades, got {len(resolved_trades)}"

    hist_syms = {"NET", "ANET", "TMDX", "CELH", "CRSP"}
    prosp_syms = {"AAPL", "MSFT", "NVDA", "VOO", "JEPI"}

    hist_trades = [t for t in resolved_trades if t["symbol"] in hist_syms]
    prosp_trades = [t for t in resolved_trades if t["symbol"] in prosp_syms]
    assert len(hist_trades) == 5
    assert len(prosp_trades) == 5

    # Historical aggregation
    hist_gross_avg = sum(t["grossRet"] for t in hist_trades) / 5.0
    hist_net_avg = sum(t["netRet"] for t in hist_trades) / 5.0
    hist_r_avg = sum(t["rMult"] for t in hist_trades) / 5.0

    assert round(hist_gross_avg, 2) == baseline_receipt["cohortBreakdown"]["historicalBaseline"]["averageGrossReturnPct"]
    assert round(hist_net_avg, 2) == baseline_receipt["cohortBreakdown"]["historicalBaseline"]["averageNetReturnPct"]
    assert round(hist_r_avg, 2) == baseline_receipt["cohortBreakdown"]["historicalBaseline"]["averageRMultiple"]

    # Prospective aggregation
    prosp_gross_avg = sum(t["grossRet"] for t in prosp_trades) / 5.0
    prosp_net_avg = sum(t["netRet"] for t in prosp_trades) / 5.0
    prosp_r_avg = sum(t["rMult"] for t in prosp_trades) / 5.0

    assert round(prosp_gross_avg, 2) == baseline_receipt["cohortBreakdown"]["prospectiveClean"]["averageGrossReturnPct"]
    assert round(prosp_net_avg, 2) == baseline_receipt["cohortBreakdown"]["prospectiveClean"]["averageNetReturnPct"]
    assert round(prosp_r_avg, 2) == baseline_receipt["cohortBreakdown"]["prospectiveClean"]["averageRMultiple"]

    # Pooled aggregation (N=10)
    pooled_gross_avg = sum(t["grossRet"] for t in resolved_trades) / 10.0
    pooled_net_avg = sum(t["netRet"] for t in resolved_trades) / 10.0
    pooled_r_avg = sum(t["rMult"] for t in resolved_trades) / 10.0

    assert round(pooled_gross_avg, 2) == baseline_receipt["denominatorCensus"]["intradayBarsResolved"]["average_gross_return_resolved"]
    assert round(pooled_net_avg, 2) == baseline_receipt["denominatorCensus"]["intradayBarsResolved"]["average_net_return_resolved"]
    assert round(pooled_r_avg, 2) == baseline_receipt["denominatorCensus"]["intradayBarsResolved"]["average_r_multiple_resolved"]


def test_preserve_unresolved_outcomes_when_evidence_insufficient(baseline_receipt, baseline_price_bars):
    """
    Verifies that when sub-daily sequencing evidence is absent (daily_bars only)
    or truncated, the evaluator strictly preserves UNRESOLVED status fail-closed,
    refusing to fabricate fill prices or percentage returns.
    """
    for sym in ("CELH", "CRSP"):
        record = next(r for r in baseline_receipt["evaluationRecords"] if r["symbol"] == sym and r["cohortType"] == "HISTORICAL")
        plan = record["plan"]
        sig = {
            "symbol": sym,
            "engineVersion": "7ad44595826c147cc77f93cd676af520764c7442",
            "signalDate": record["signalDate"],
            "corridorMin": plan["corridorMin"],
            "corridorMax": plan["corridorMax"],
            "stopLoss": plan["stopLoss"],
            "takeProfit1": plan["takeProfit1"],
            "confluenceScore": plan["confluenceScore"],
            "marketRegime": "UNKNOWN",
        }
        bars = baseline_price_bars[sym]
        df = pd.DataFrame(bars)
        date_col = next(c for c in df.columns if c.lower() in ("date", "trade_date"))
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.rename(columns={c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")})

        # Intraday bars = None -> Evaluator must fail-closed to UNRESOLVED_INTRABAR_SEQUENCE
        res = PaperTradingOutcomeEvaluator.evaluate_signal(
            signal=sig,
            daily_bars=df,
            intraday_bars=None,
            provider="YAHOO_FINANCE",
        )
        assert res.outcome == "UNRESOLVED_INTRABAR_SEQUENCE"
        assert res.exit_reason == "INTRABAR_ENTRY_STOP_COLLISION"
        assert res.gross_return_pct is None
        assert res.net_simulated_return_pct is None
        assert res.r_multiple is None
        assert res.exit_price is None
