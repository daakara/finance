"""
Behavioral verification for ARX-S4-RADAR-001:
Radar Operational Truth and Scanner-Pending Remediation.

Covers overlapping defects shared with ARX-S2-009, ARX-S3-004, and ARX-S3-005:
1. Per-candidate loop state isolation (no Stage 2 / VCP / ETF / fundamental leakage across rows).
2. Evidence-backed reason + recoveryCondition across pending, stale, rollback, hold, and error states.
3. Missing/failed scanner snapshots never appear as GENUINELY_EMPTY or PIPELINE_PENDING.
4. Served date provenance reflects actual candidate close dates (no fabricated today observed_date)
   and primary Stage-2 opportunity count is not inflated by non-Stage-2 records.
5. Candidate membership remains strictly discovery-only (isActionable=False, canSizeTrade=False).
"""

import pytest
from unittest.mock import patch
from fastapi import Response

import api.routes.screener as screener_mod
from api.routes.screener import (
    get_radar_capabilities,
    run_screener_get,
    _RADAR_RUN_CACHE,
    _RADAR_CACHE_LOCK,
)
from analyst_dashboard.analyzers.scanner_contract import FreshnessStatus, ScannerStatus
from analyst_dashboard.analyzers.scanner_runner import (
    SmartMoneyScannerRunner,
    VCPScannerRunner,
)


def _clear_radar_cache():
    with _RADAR_CACHE_LOCK:
        _RADAR_RUN_CACHE.clear()
    screener_mod._STAGE_CLASSIFICATION_CACHE.clear()


@pytest.fixture(autouse=True)
def _reset_radar_caches():
    _clear_radar_cache()
    yield
    _clear_radar_cache()


def test_loop_state_does_not_leak_across_candidates(published_screener_universe, monkeypatch):
    """
    Ensure derived loop variables (stage_class, is_vcp_qual, asset_class_str,
    fundamentalProvenance) are reset per candidate so an earlier verified Stage 2 ETF
    or equity never contaminates subsequent delisted, unverified, errored, or short-history rows.
    """
    _clear_radar_cache()
    monkeypatch.setattr(
        published_screener_universe,
        "get_scannable_membership",
        lambda _bid: ["SPY", "CPRX", "UNVERX", "ERRSYM", "SHORTHIST"],
    )

    real_eval = screener_mod.screener.evaluate_candidates
    monkeypatch.setattr(
        screener_mod.screener,
        "evaluate_candidates",
        lambda syms: [{"ticker": s, "company_name": s} for s in syms],
    )

    # Seed SPY (an ETF in KNOWN_ETFS) with full history + Stage 2 qualification,
    # UNVERX with no price, ERRSYM throwing an exception, SHORTHIST with 10 candles.
    real_get_price = screener_mod.market_db.get_latest_price
    real_get_candles = screener_mod.market_db.get_daily_candles
    real_get_factor = screener_mod.market_db.get_factor_snapshot

    sample_candles = [
        {"time": f"2026-10-{d:02d}", "open": 100.0 + d * 0.2, "high": 101.0 + d * 0.2, "low": 99.0 + d * 0.2, "close": 100.5 + d * 0.2, "volume": 1_000_000}
        for d in range(1, 26)
    ]

    def fake_latest_price(sym: str):
        if sym == "ERRSYM":
            raise RuntimeError("Simulated OHLCV read failure")
        if sym == "UNVERX":
            return None
        if sym in ("SPY", "CPRX", "SHORTHIST"):
            return {"currentPrice": 500.0 if sym == "SPY" else 20.0, "date": "2026-11-23"}
        return real_get_price(sym)

    def fake_daily_candles(sym: str, limit: int = 252):
        if sym == "SHORTHIST":
            return sample_candles[:10]
        if sym == "SPY":
            # Build 210 uptrending candles
            return [
                {
                    "time": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                    "open": 400.0 + i * 0.4,
                    "high": 402.0 + i * 0.4,
                    "low": 399.0 + i * 0.4,
                    "close": 401.0 + i * 0.4,
                    "volume": 2_000_000,
                }
                for i in range(210)
            ]
        return real_get_candles(sym, limit=limit)

    def fake_factor_snapshot(sym: str):
        if sym == "SPY":
            return {"quality_score": 85.0, "growth_score": 82.0, "valuation_score": 70.0, "as_of_date": "2026-11-20"}
        return None

    monkeypatch.setattr(screener_mod.market_db, "get_latest_price", fake_latest_price)
    monkeypatch.setattr(screener_mod.market_db, "get_daily_candles", fake_daily_candles)
    monkeypatch.setattr(screener_mod.market_db, "get_factor_snapshot", fake_factor_snapshot)
    monkeypatch.setattr(
        screener_mod,
        "compute_canonical_stage_classification",
        lambda sym, candles, eval_as_of=None: ("STAGE_2", ["TREND_OK"], True)
        if sym == "SPY"
        else ("STAGE_UNRESOLVED", ["INSUFFICIENT_HISTORY"], False),
    )

    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="diagnostic")
    assert data["status"] == "PARTIAL"
    assert any(pe["ticker"] == "ERRSYM" for pe in (data.get("partialErrors") or []))

    by_sym = {c["symbol"]: c for c in data["candidates"]}
    assert by_sym["SPY"]["stageClassification"] == "STAGE_2"
    assert by_sym["SPY"]["stageProvenance"]["isVcpQualified"] is True

    # None of the subsequent non-Stage2 / delisted / unverified / errored rows may inherit SPY's state
    for sym in ("CPRX", "UNVERX", "ERRSYM", "SHORTHIST"):
        c = by_sym[sym]
        assert c["stageClassification"] == "STAGE_UNRESOLVED", f"{sym} leaked stageClassification={c['stageClassification']}"
        assert c["stageProvenance"]["isVcpQualified"] is False, f"{sym} leaked isVcpQualified"
        assert c["fundamentalProvenance"]["isVerified"] is False, f"{sym} leaked fundamentalProvenance"
        assert c["isActionable"] is False
        assert c["canSizeTrade"] is False


def test_capability_pending_and_error_states_have_reason_and_recovery_condition(monkeypatch, tmp_path):
    """
    Verify VCP & Smart Money pending/stale/rollback/hold/error capability envelopes
    all expose non-empty reason and recoveryCondition and never fabricate 0 matches.
    """
    vcp_runner = VCPScannerRunner()
    for reason in ("NO_ACTIVE_SNAPSHOT", "RELEASE_MISMATCH", "UNSUPPORTED_LEGACY_MATCHES", "STORAGE_ERROR"):
        status = ScannerStatus.ERROR if reason == "STORAGE_ERROR" else ScannerStatus.PIPELINE_PENDING
        env = vcp_runner._to_status_envelope(status=status, reason=reason, detail=f"Test detail {reason}")
        assert env["reason"] == reason
        assert env.get("recovery_condition")
        assert env["matched_count"] is None

    # SmartMoney storage exception must fail closed to ERROR (not impersonate PIPELINE_PENDING)
    class BrokenStore:
        def get_latest_active_snapshot(self, *_args, **_kwargs):
            raise RuntimeError("Postgres connection refused")

        def get_latest_active_snapshot_id(self, *_args, **_kwargs):
            raise RuntimeError("Postgres connection refused")

    sm_runner = SmartMoneyScannerRunner(snapshot_store=BrokenStore(), backend="postgres")
    sm_err_env = sm_runner.get_active_or_latest_snapshot()
    assert sm_err_env["status"] == "ERROR"
    assert sm_err_env["reason"] == "STORAGE_ERROR"
    assert sm_err_env.get("recovery_condition")
    assert sm_err_env["matched_count"] is None

    # Control A postgres read disablement exposes POSTGRES_READ_DISABLED
    monkeypatch.setenv("ARX_SMART_MONEY_POSTGRES_READ_ENABLED", "false")
    sm_disabled_env = sm_runner.get_active_or_latest_snapshot()
    assert sm_disabled_env["status"] == "PIPELINE_PENDING"
    assert sm_disabled_env["reason"] == "POSTGRES_READ_DISABLED"
    assert sm_disabled_env.get("recovery_condition")
    monkeypatch.delenv("ARX_SMART_MONEY_POSTGRES_READ_ENABLED", raising=False)

    # Capabilities mapping checks
    class RollbackPtr:
        published_symbols = ["AAPL", "MSFT"]
        build_id = "UB-RB-1"
        universe_id = "ARX_US_EQUITIES"
        publication_digest = "sha256-rb"
        is_rollback = True
        rollback_reason = "BAD_DATA"

    rb_caps = get_radar_capabilities(publication_pointer=RollbackPtr())
    assert rb_caps["VCP"]["status"] == "PIPELINE_PENDING"
    assert rb_caps["VCP"]["reason"] == "ROLLED_BACK_PUBLICATION_POINTER"
    assert rb_caps["VCP"].get("recoveryCondition")

    stale_vcp = {"status": "STALE", "matched_count": 4, "snapshot_id": "vcp-stale", "data_as_of": "2026-10-10", "universe": {"eligible_universe_count": 200}}
    stale_sm = {"status": "AVAILABLE", "matched_count": 10, "snapshot_id": "SM-STALE", "freshness": {"status": "STALE", "data_as_of": "2026-10-10"}}
    stale_caps = get_radar_capabilities(vcp_snapshot=stale_vcp, sm_snapshot=stale_sm)
    for key in ("VCP", "SMART_MONEY"):
        assert stale_caps[key]["status"] == "PIPELINE_PENDING"
        assert stale_caps[key]["matchedCount"] is None
        assert stale_caps[key]["reason"] == "STALE_SNAPSHOT"
        assert stale_caps[key].get("recoveryCondition")

    # Preserve SNAP_SM_1792939989_48f30db6 historical 10-issuer pilot contract cleanly
    pilot_sm = {
        "status": "AVAILABLE",
        "snapshot": {"snapshot_id": "SNAP_SM_1792939989_48f30db6", "matched_count": 10, "universe_size": 10},
        "freshness": {"data_as_of": "2026-10-25", "status": "LIVE"},
    }
    pilot_caps = get_radar_capabilities(vcp_snapshot=stale_vcp, sm_snapshot=pilot_sm, sm_stage2_intersection_count=5)
    assert pilot_caps["SMART_MONEY"]["status"] == "AVAILABLE"
    assert pilot_caps["SMART_MONEY"]["snapshotId"] == "SNAP_SM_1792939989_48f30db6"
    assert pilot_caps["SMART_MONEY"]["displayedStage2IntersectionCount"] == 5
    assert pilot_caps["SMART_MONEY"]["coverageStatus"] == "PILOT_HELD"

    # Incomplete AVAILABLE SM snapshot missing matched_count never coerces None -> 0
    bad_sm = {"status": "AVAILABLE", "snapshot_id": "SM-NO-COUNT", "matched_count": None}
    bad_sm_caps = get_radar_capabilities(sm_snapshot=bad_sm)
    assert bad_sm_caps["SMART_MONEY"]["status"] == "PIPELINE_PENDING"
    assert bad_sm_caps["SMART_MONEY"]["matchedCount"] is None
    assert bad_sm_caps["SMART_MONEY"]["reason"] == "SNAPSHOT_OUTPUT_INCOMPLETE"


def test_filtered_error_and_pending_never_appear_as_genuine_zero(published_screener_universe, monkeypatch):
    """
    When filtering by 'vcp' or 'smart_money' while that scanner is in ERROR,
    run_screener_get must return ERROR / SCAN_FAILED (never SUCCESS / GENUINELY_EMPTY).
    When STALE or PIPELINE_PENDING, datasetState must be PIPELINE_PENDING and results empty.
    Genuine 0 matches returns SUCCESS / GENUINELY_EMPTY.
    """
    _clear_radar_cache()
    resp = Response()

    # 1. VCP ERROR
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "ERROR", "reason": "STORAGE_ERROR", "detail": "VCP sqlite unreadable"},
    ):
        _clear_radar_cache()
        err_vcp = run_screener_get(resp, filter_type="vcp")
        assert err_vcp["status"] == "ERROR"
        assert err_vcp["scanState"] == "FAILED"
        assert err_vcp["datasetState"] == "SCAN_FAILED"
        assert err_vcp["gemsFound"] == 0
        assert err_vcp["filterReason"] == "STORAGE_ERROR"
        assert err_vcp.get("filterRecoveryCondition")

    # 2. SMART_MONEY ERROR
    class FakeSMRunner:
        def get_active_or_latest_snapshot(self):
            return {"status": "ERROR", "reason": "STORAGE_ERROR", "detail": "Postgres down"}

    monkeypatch.setattr(screener_mod, "get_smart_money_scanner_runner", lambda: FakeSMRunner())
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "AVAILABLE", "matched_count": 0, "snapshot_id": "vcp-0", "results": [], "universe": {"eligible_universe_count": 35}},
    ):
        _clear_radar_cache()
        err_sm = run_screener_get(resp, filter_type="smart_money")
        assert err_sm["status"] == "ERROR"
        assert err_sm["scanState"] == "FAILED"
        assert err_sm["datasetState"] == "SCAN_FAILED"
        assert err_sm["gemsFound"] == 0

    # 3. STALE VCP snapshot carrying old results must still fail closed as PIPELINE_PENDING with 0 leaks
    class PendingSMRunner:
        def get_active_or_latest_snapshot(self):
            return SmartMoneyScannerRunner.get_status_envelope()

    monkeypatch.setattr(screener_mod, "get_smart_money_scanner_runner", lambda: PendingSMRunner())
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "STALE", "matched_count": 1, "snapshot_id": "vcp-old", "results": [{"symbol": "AAPL"}], "freshness": {"status": "STALE"}},
    ):
        _clear_radar_cache()
        stale_res = run_screener_get(resp, filter_type="vcp")
        assert stale_res["datasetState"] == "PIPELINE_PENDING"
        assert stale_res["gemsFound"] == 0
        assert stale_res["filterReason"] == "STALE_SNAPSHOT"

    # 4. Genuine 0 matches
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "AVAILABLE", "matched_count": 0, "snapshot_id": "vcp-zero", "results": [], "universe": {"eligible_universe_count": 35}, "freshness": {"status": "LIVE", "data_as_of": "2026-11-20"}},
    ):
        _clear_radar_cache()
        zero_res = run_screener_get(resp, filter_type="vcp")
        assert zero_res["status"] == "SUCCESS"
        assert zero_res["datasetState"] == "GENUINELY_EMPTY"
        assert zero_res["gemsFound"] == 0


def test_observed_date_and_primary_opportunity_count_reconcile(published_screener_universe, monkeypatch):
    """
    1. evidence.provenance.observed_date must reflect actual latest served priceDate (not wall-clock gmtime).
    2. Mixed candidate close dates must expose marketDataOldestAsOf & marketDataMixedDates=True.
    3. Non-Stage-2 IN_BUY_ZONE records must not inflate primary Stage-2 currentOpportunityCount.
    """
    _clear_radar_cache()
    monkeypatch.setattr(
        published_screener_universe,
        "get_scannable_membership",
        lambda _bid: ["AAPL", "MSFT"],
    )
    monkeypatch.setattr(
        screener_mod.screener,
        "evaluate_candidates",
        lambda syms: [{"ticker": s, "company_name": s} for s in syms],
    )

    real_price = screener_mod.market_db.get_latest_price
    monkeypatch.setattr(
        screener_mod.market_db,
        "get_latest_price",
        lambda sym: (
            {"currentPrice": 210.0, "date": "2026-11-23"}
            if sym == "AAPL"
            else ({"currentPrice": 210.0, "date": "2026-11-06"} if sym == "MSFT" else real_price(sym))
        ),
    )

    monkeypatch.setattr(
        screener_mod.optimal_engine,
        "calculate_trade_levels",
        lambda *_args, **_kwargs: {
            "current_price": 210.0,
            "optimal_entry_min": 205.0,
            "optimal_entry_max": 215.0,
            "stop_loss": 195.0,
            "stop_loss_pct": -7.1,
            "take_profit_1": 240.0,
            "take_profit_1_pct": 14.3,
            "take_profit_2": 260.0,
            "take_profit_2_pct": 23.8,
            "risk_reward_ratio": 2.0,
            "execution_status": "IN_BUY_ZONE",
            "setup_pattern": "Base",
            "entry_thesis": "Test",
            "invalidation_condition": "Below 195",
            "stage_phase": "Stage 2",
            "vcp_contraction_status": "N/A",
            "breakout_pivot": None,
            "atr_14": 4.0,
            "liquidity_defense": None,
        },
    )

    # AAPL is STAGE_2 IN_BUY_ZONE (served); MSFT is STAGE_1 IN_BUY_ZONE (not served in STAGE_2_PRIMARY)
    monkeypatch.setattr(
        screener_mod,
        "compute_canonical_stage_classification",
        lambda sym, *_args, **_kwargs: (
            ("STAGE_2", ["TREND_OK"], False) if sym == "AAPL" else ("STAGE_1", ["MA50_BELOW_MA200"], False)
        ),
    )

    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    assert data["status"] in ("SUCCESS", "PARTIAL")
    assert [c["symbol"] for c in data["candidates"]] == ["AAPL"]

    # Gate 4: Served dates and counts must derive from actual served rows (only AAPL on 2026-11-23)
    pop = data["population"]
    assert pop["marketDataAsOf"] == "2026-11-23"
    assert pop["marketDataOldestAsOf"] == "2026-11-23"
    assert pop["marketDataMixedDates"] is False
    assert pop["marketDataLatestCount"] == 1
    assert pop["marketDataLaggingCount"] == 0
    assert pop["universeMarketDataOldestAsOf"] == "2026-11-06"
    assert pop["universeMarketDataMixedDates"] is True
    assert data["evidence"]["provenance"]["observed_date"] == "2026-11-23"

    # Primary view currentOpportunityCount matches ONLY Stage 2 primary opportunities (1), not Stage 1 + Stage 2 (2)
    assert pop["currentOpportunityCount"] == 1
    assert pop["diagnosticOpportunityCount"] == 2
    assert data["candidates"][0]["isActionable"] is False
    assert data["candidates"][0]["canSizeTrade"] is False


def test_real_classifier_row_isolation_and_served_mixed_date_and_empty_truth(published_screener_universe, monkeypatch):
    """
    Gate 1 & Gate 4 behavioral verification without mocking VCPClassifier or OptimalExecutionEngine:
    1. Valid Stage-2 + VALUE_GARP + VCP + SMART_MONEY + ETF-like verified row before invalid/short-history/errored/Stage-4 rows
       must leak zero eligibility, categories, stageClassification, stopLoss, or provenance onto subsequent rows.
    2. Served-row mixed close dates report exact min/max/counts across served rows; genuine empty VCP reports None dates and 0 opportunities.
    """
    from datetime import date, timedelta

    _clear_radar_cache()
    monkeypatch.setattr(
        published_screener_universe,
        "get_scannable_membership",
        lambda _bid: ["AAPL", "NVDA", "CPRX", "NOPRX", "CERRX", "SHORTX", "STAGE4X"],
    )
    monkeypatch.setattr(
        screener_mod.screener,
        "evaluate_candidates",
        lambda syms: [
            (
                {
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "composite_score": 88,
                    "expert_model": "Peter Lynch GARP Compounder",
                    "peg_ratio": 0.95,
                    "roic_pct": 32.0,
                    "gross_margin_pct": 46.0,
                }
                if s == "AAPL"
                else {"ticker": s, "company_name": s}
            )
            for s in syms
        ],
    )

    def build_bars(start_price: float, step: float, count: int, end_date_str: str):
        end_dt = date.fromisoformat(end_date_str)
        out = []
        for i in range(count):
            dt = end_dt - timedelta(days=count - 1 - i)
            p = max(5.0, start_price + i * step)
            out.append({
                "time": dt.isoformat(),
                "open": p - 0.5,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "volume": 1_500_000,
            })
        return out

    aapl_bars = build_bars(100.0, 0.4, 252, "2026-11-23")
    nvda_bars = build_bars(90.0, 0.35, 252, "2026-11-06")
    stage4_bars = build_bars(260.0, -0.65, 252, "2026-11-20")
    short_bars = build_bars(50.0, 0.2, 12, "2026-11-20")

    def real_price(sym: str):
        if sym == "NOPRX":
            return None
        if sym == "NVDA":
            return {"currentPrice": nvda_bars[-1]["close"], "date": "2026-11-06"}
        if sym == "AAPL":
            return {"currentPrice": aapl_bars[-1]["close"], "date": "2026-11-23"}
        if sym in ("CPRX", "CERRX", "SHORTX", "STAGE4X"):
            return {"currentPrice": 50.0, "date": "2026-11-20"}
        return None

    def real_candles(sym: str, limit: int = 252):
        if sym == "CERRX":
            raise RuntimeError("Simulated candle read exception after latest price succeeded")
        if sym == "AAPL":
            return aapl_bars[-limit:]
        if sym == "NVDA":
            return nvda_bars[-limit:]
        if sym == "STAGE4X":
            return stage4_bars[-limit:]
        if sym == "SHORTX":
            return short_bars[-limit:]
        return []

    monkeypatch.setattr(screener_mod.market_db, "get_latest_price", real_price)
    monkeypatch.setattr(screener_mod.market_db, "get_daily_candles", real_candles)
    monkeypatch.setattr(
        screener_mod.market_db,
        "get_factor_snapshot",
        lambda sym: (
            {
                "quality_score": 90.0,
                "growth_score": 84.0,
                "valuation_score": 72.0,
                "roic": 32.0,
                "peg": 0.95,
                "source": "sec_edgar",
                "as_of_date": "2026-11-20",
                "filing_date": "2026-11-20",
                "period_end": "2026-09-30",
                "acceptance_datetime": "2026-11-20T21:00:00Z",
                "available_from": "2026-11-20",
            }
            if sym == "AAPL"
            else None
        ),
    )

    resp = Response()
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "AVAILABLE", "matched_count": 1, "snapshot_id": "vcp-1", "results": [{"symbol": "AAPL"}], "universe": {"eligible_universe_count": 7}, "freshness": {"status": "LIVE", "data_as_of": "2026-11-23"}},
    ):
        _clear_radar_cache()
        diag = run_screener_get(resp, filter_type="all", view="diagnostic")
        assert diag["status"] == "PARTIAL"
        assert [pe["ticker"] for pe in (diag.get("partialErrors") or [])] == ["CERRX"]

        by_sym = {c["symbol"]: c for c in diag["candidates"]}
        assert by_sym["AAPL"]["stageClassification"] == "STAGE_2"
        assert "VALUE_GARP" in by_sym["AAPL"]["categories"]
        assert "VCP" in by_sym["AAPL"]["categories"]
        assert by_sym["AAPL"]["fundamentalProvenance"]["isVerified"] is True
        assert by_sym["AAPL"]["stopLoss"] is not None

        for bad_sym, expected_status in (
            ("CPRX", "DELISTED_SECURITY"),
            ("NOPRX", "UNVERIFIED_ASSET"),
            ("CERRX", "UNVERIFIED_ASSET"),
            ("SHORTX", "INSUFFICIENT_HISTORY"),
        ):
            row = by_sym[bad_sym]
            assert row["executionStatus"] == expected_status
            assert row["screeningEligible"] is False
            assert row["stageClassification"] == "STAGE_UNRESOLVED"
            assert row["stageProvenance"]["isVcpQualified"] is False
            assert row["fundamentalProvenance"]["isVerified"] is False
            assert row["categories"] == []
            assert row["categoryEvidence"] == {}
            assert row["stopLoss"] is None
            assert row["atr14"] == "N/A"
            assert row["isActionable"] is False

        assert by_sym["STAGE4X"]["stageClassification"] != "STAGE_2"
        assert by_sym["STAGE4X"]["categories"] == []
        assert by_sym["STAGE4X"]["fundamentalProvenance"]["isVerified"] is False

        # Primary view serves ONLY AAPL and NVDA (Stage 2 rows spanning 2026-11-06..2026-11-23)
        _clear_radar_cache()
        primary = run_screener_get(resp, filter_type="all")
        served_syms = [c["symbol"] for c in primary["candidates"]]
        assert set(served_syms) == {"AAPL", "NVDA"}
        assert primary["population"]["marketDataAsOf"] == "2026-11-23"
        assert primary["population"]["marketDataOldestAsOf"] == "2026-11-06"
        assert primary["population"]["marketDataMixedDates"] is True
        assert primary["population"]["marketDataLatestCount"] == 1
        assert primary["population"]["marketDataLaggingCount"] == 1

        # Filtered value_garp serves ONLY AAPL (2026-11-23) -> not polluted by NVDA 2026-11-06
        _clear_radar_cache()
        vg = run_screener_get(resp, filter_type="value_garp")
        assert [c["symbol"] for c in vg["candidates"]] == ["AAPL"]
        assert vg["population"]["marketDataAsOf"] == "2026-11-23"
        assert vg["population"]["marketDataOldestAsOf"] == "2026-11-23"
        assert vg["population"]["marketDataMixedDates"] is False
        assert vg["evidence"]["provenance"]["observed_date"] == "2026-11-23"

    # Genuine empty VCP scan serves 0 rows -> must NOT carry AAPL/NVDA close dates or opportunities
    with patch.object(
        screener_mod.vcp_scanner_runner,
        "get_active_or_latest_snapshot",
        return_value={"status": "AVAILABLE", "matched_count": 0, "snapshot_id": "vcp-0", "results": [], "universe": {"eligible_universe_count": 7}, "freshness": {"status": "LIVE", "data_as_of": "2026-11-23"}},
    ):
        _clear_radar_cache()
        vcp_empty = run_screener_get(resp, filter_type="vcp")
        assert vcp_empty["status"] == "PARTIAL"
        assert [pe["ticker"] for pe in (vcp_empty.get("partialErrors") or [])] == ["CERRX"]
        assert vcp_empty["datasetState"] == "GENUINELY_EMPTY"
        assert vcp_empty["gemsFound"] == 0
        assert vcp_empty["population"]["currentOpportunityCount"] == 0
        assert vcp_empty["population"]["marketDataAsOf"] is None
        assert vcp_empty["population"]["marketDataOldestAsOf"] is None
        assert vcp_empty["population"]["marketDataMixedDates"] is False
        assert vcp_empty["evidence"]["provenance"]["observed_date"] is None

