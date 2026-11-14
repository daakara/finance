"""
Targeted Verification for ARX-CONFLUENCE-INTEGRITY-CLOSURE-006:
Provenance Wiring Across Analytics, Screener, and Decision Trace.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine
from analyst_dashboard.provenance.evidence_contract import evaluate_factor_provenance, EvidenceStatus
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


def test_decision_trace_respects_verified_provenance():
    confluence_mock = {
        "confluenceScore": 82.0,
        "verifiedPillarCount": 3,
        "fundamentalProvenance": {
            "evidenceStatus": "VERIFIED",
            "isVerified": True,
            "sourceId": "sec_edgar",
            "filingDate": "2026-10-15",
        }
    }
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="NVDA",
        current_price=130.0,
        candles=[{"close": 130.0}] * 60,
        freshness={"status": "REALTIME"},
        technicals={"rsi_14": 55.0},
        confluence=confluence_mock,
        factor_scores={"qualityScore": 88.0, "piotroskiFScore": 8},
        optimal_execution={"execution_status": "IN_BUY_ZONE"},
        asset_class="EQUITY",
    )
    assert trace["symbol"] == "NVDA"
    assert trace["decisionState"] != "EVIDENCE_INCOMPLETE"
    assert trace["trace"]["marketData"]["freshness"] == "REALTIME"


def test_decision_trace_fails_closed_on_unverified_provenance():
    confluence_mock = {
        "confluenceScore": 60.0,
        "verifiedPillarCount": 2,
        "fundamentalProvenance": {
            "evidenceStatus": "UNVERIFIED",
            "isVerified": False,
            "sourceId": "unknown",
        }
    }
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="LNTH",
        current_price=100.0,
        candles=[{"close": 100.0}] * 60,
        freshness={"status": "REALTIME"},
        technicals={"rsi_14": 55.0},
        confluence=confluence_mock,
        factor_scores={"qualityScore": 95.0, "piotroskiFScore": 8},
        optimal_execution={"execution_status": "IN_BUY_ZONE"},
        asset_class="EQUITY",
    )
    # Despite populated factor scores, unverified provenance must fail closed
    assert trace["decisionState"] == "EVIDENCE_INCOMPLETE"
    assert "fundamental" in trace["disqualificationReason"].lower() or "unverified" in trace["disqualificationReason"].lower()


def test_decision_trace_synthetic_template_rejected():
    snap = {
        "symbol": "ONON",
        "quality_score": 75.0,
        "piotroski_f": 7,
        "source": "unknown",
    }
    prov = evaluate_factor_provenance(snap, "ONON", asset_class="EQUITY")
    assert prov["isVerified"] is False
    assert prov["evidenceStatus"] == "SYNTHETIC"

    confluence_mock = {
        "confluenceScore": 50.0,
        "verifiedPillarCount": 1,
        "fundamentalProvenance": prov,
    }
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="ONON",
        current_price=50.0,
        candles=[{"close": 50.0}] * 60,
        freshness={"status": "REALTIME"},
        technicals={"rsi_14": 55.0},
        confluence=confluence_mock,
        factor_scores={"qualityScore": 75.0, "piotroskiFScore": 7},
        optimal_execution={"execution_status": "IN_BUY_ZONE"},
        asset_class="EQUITY",
        fundamental_provenance=prov,
    )
    assert trace["decisionState"] == "EVIDENCE_INCOMPLETE"


def test_decision_trace_etf_exemption_preserves_not_applicable():
    prov = evaluate_factor_provenance({}, "SPY", asset_class="ETF")
    assert prov["evidenceStatus"] == "NOT_APPLICABLE"
    assert prov["isVerified"] is True

    confluence_mock = {
        "confluenceScore": 75.0,
        "verifiedPillarCount": 3,
        "fundamentalProvenance": prov,
    }
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="SPY",
        current_price=600.0,
        candles=[{"close": 600.0}] * 60,
        freshness={"status": "REALTIME"},
        technicals={"rsi_14": 55.0},
        confluence=confluence_mock,
        factor_scores={},
        optimal_execution={"execution_status": "IN_BUY_ZONE"},
        asset_class="ETF",
        fundamental_provenance=prov,
    )
    assert trace["decisionState"] != "EVIDENCE_INCOMPLETE"


if __name__ == "__main__":
    test_decision_trace_respects_verified_provenance()
    test_decision_trace_fails_closed_on_unverified_provenance()
    test_decision_trace_synthetic_template_rejected()
    test_decision_trace_etf_exemption_preserves_not_applicable()
    print("ALL 4 CONFLUENCE PROVENANCE WIRING TESTS PASSED!")
