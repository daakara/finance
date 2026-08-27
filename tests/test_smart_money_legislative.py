"""Unit & Integration Tests for Legislative Alignment, STOCK Act Staleness Decay, and Regulatory Catalysts."""

import pytest
from analyst_dashboard.analyzers.smart_money import (
    SmartMoneyEngine,
    calculate_legislative_alignment,
    compute_filing_staleness,
    CONGRESSIONAL_TRADES,
)
from analyst_dashboard.analyzers.catalysts import CatalystEngine


def test_calculate_legislative_alignment_bounds():
    """Verify legislative alignment scores are within valid quantitative bounds [35, 99]."""
    for trade in CONGRESSIONAL_TRADES:
        score = calculate_legislative_alignment(trade)
        assert 35 <= score <= 99, f"Score {score} out of bounds for trade {trade.get('id')}"


def test_calculate_legislative_alignment_jurisdiction_boost():
    """Verify trades with direct committee jurisdiction overlap receive higher alignment scores."""
    trade_defense = {
        "sector": "Defense Software & AI",
        "amount_range": "$1,000,000 - $5,000,000",
        "details": {
            "committee_assignments": ["Senate Armed Services", "Intelligence"],
            "historical_win_rate_pct": 78.0,
        },
    }
    trade_unrelated = {
        "sector": "Defense Software & AI",
        "amount_range": "$50,000 - $100,000",
        "details": {
            "committee_assignments": ["Agriculture, Nutrition & Forestry"],
            "historical_win_rate_pct": 60.0,
        },
    }
    score_defense = calculate_legislative_alignment(trade_defense)
    score_unrelated = calculate_legislative_alignment(trade_unrelated)

    assert score_defense > score_unrelated, f"Expected {score_defense} > {score_unrelated}"
    assert score_defense >= 85, f"Expected strong alignment >= 85, got {score_defense}"


def test_compute_filing_staleness_tiers():
    """Verify accurate tier classification, penalties, and warnings across filing lag windows."""
    # 1. Fresh (< 15 days)
    fresh = compute_filing_staleness(12, base_strength=95)
    assert fresh["staleness_status"] == "FRESH"
    assert fresh["staleness_penalty"] == 0
    assert fresh["effective_signal_strength"] == 95
    assert fresh["staleness_warning"] is None

    # 2. Standard (16 - 30 days)
    standard = compute_filing_staleness(20, base_strength=95)
    assert standard["staleness_status"] == "NORMAL"
    assert standard["staleness_penalty"] == 5
    assert standard["effective_signal_strength"] == 90
    assert standard["staleness_warning"] is None

    # 3. Aging (31 - 45 days)
    aging = compute_filing_staleness(38, base_strength=95)
    assert aging["staleness_status"] == "AGING"
    assert aging["staleness_penalty"] == 16
    assert aging["effective_signal_strength"] == 79
    assert "approaching statutory limit" in aging["staleness_warning"]

    # 4. Late Filer (> 45 days)
    late = compute_filing_staleness(58, base_strength=95)
    assert late["staleness_status"] == "LATE_FILER"
    assert late["staleness_penalty"] == 32
    assert late["effective_signal_strength"] == 63
    assert "exceeds statutory 45d window" in late["staleness_warning"]


def test_staleness_monotonic_decay():
    """Verify signal strength monotonically decreases as filing lag increases."""
    lag_days = [5, 15, 25, 35, 45, 55, 75]
    effective_scores = [
        compute_filing_staleness(d, base_strength=90)["effective_signal_strength"]
        for d in lag_days
    ]

    for i in range(len(effective_scores) - 1):
        assert effective_scores[i] >= effective_scores[i + 1], (
            f"Non-monotonic decay: lag {lag_days[i]}d ({effective_scores[i]}) vs "
            f"{lag_days[i+1]}d ({effective_scores[i+1]})"
        )


def test_smart_money_engine_enrichment():
    """Verify SmartMoneyEngine enriches trades with all required institutional metrics."""
    trades = SmartMoneyEngine.get_congressional_trades()
    assert len(trades) > 0

    for t in trades:
        assert "legislative_alignment_score" in t
        assert "staleness_status" in t
        assert "staleness_badge" in t
        assert "effective_signal_strength" in t
        assert "compliance_tier" in t
        assert 0 <= t["legislative_alignment_score"] <= 100
        assert t["staleness_status"] in ["FRESH", "NORMAL", "AGING", "LATE_FILER"]


def test_smart_money_overview_metrics():
    """Verify overview aggregate returns count of late filers and fresh trades."""
    overview = SmartMoneyEngine.get_smart_money_overview()
    assert "late_filers_count" in overview
    assert "fresh_trades_count" in overview
    assert overview["late_filers_count"] >= 1, "Expected at least 1 late filer test case"
    assert overview["fresh_trades_count"] >= 1, "Expected at least 1 fresh trade"


def test_catalyst_engine_regulatory_milestones():
    """Verify CatalystEngine contains legislative and regulatory milestones for key assets."""
    engine = CatalystEngine()
    nvda_report = engine.get_asset_catalyst_report("NVDA", current_price=125.0)
    assert nvda_report["symbol"] == "NVDA"
    assert len(nvda_report["upcoming_milestones"]) >= 3

    # Verify presence of legislative hearing/appropriations
    events_text = " ".join([m["event"] for m in nvda_report["upcoming_milestones"]])
    assert any(term in events_text.lower() for term in ["congressional", "export", "sovereign", "appropriations"])

    pltr_report = engine.get_asset_catalyst_report("PLTR", current_price=30.0)
    pltr_events = " ".join([m["event"] for m in pltr_report["upcoming_milestones"]])
    assert any(term in pltr_events.lower() for term in ["defense", "appropriations", "ndaa", "procurement"])
