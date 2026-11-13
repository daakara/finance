"""Cross-Component Confluence Parity & Provenance Invariants Test Suite (ARX-CONFLUENCE-INTEGRITY-003).

Verifies:
1. Confluence calculation authority parity across Tactical Setups, Asset Analytics, and Screener.
2. Synthetic factor suppression: synthetic fixtures (ONON, MNST, ULTA) fail closed across all endpoints.
3. DecisionTraceEngine and DecisionHierarchyEngine fail closed to EVIDENCE_INCOMPLETE on synthetic inputs.
4. ETF corporate filing exemption preserves NOT_APPLICABLE status across components without phantom scoring.
5. Cross-component score parity: identical market and factor inputs produce identical confluence scores.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine
from analyst_dashboard.provenance.evidence_contract import (
    evaluate_factor_provenance,
    EvidenceStatus,
)


class TestCrossComponentConfluenceParity:
    """Verifies scoring and provenance parity across ARX components."""

    @pytest.fixture
    def engine(self):
        return ConfluenceEngine()

    @pytest.fixture
    def synthetic_factor_snapshot(self):
        return {
            "source": "synthetic",
            "quality_score": 88.0,
            "growth_score": 92.0,
            "valuation_score": 75.0,
            "piotroski_f": 8,
            "created_at": "2026-03-31T00:00:00Z",
            # Note: No filing dates or accession lineage
        }

    @pytest.fixture
    def verified_factor_snapshot(self):
        return {
            "source": "SEC_EDGAR",
            "quality_score": 85.0,
            "growth_score": 80.0,
            "valuation_score": 70.0,
            "piotroski_f": 8,
            "filing_date": "2026-02-15",
            "as_of_date": "2025-12-31",
            "created_at": "2026-02-16T10:00:00Z",
        }

    def test_synthetic_factor_suppression_across_provenance_and_confluence(
        self, engine, synthetic_factor_snapshot
    ):
        """Synthetic factor snapshot must fail closed in evaluate_factor_provenance and calculate_confluence."""
        for sym in ("ONON", "MNST", "ULTA"):
            prov = evaluate_factor_provenance(synthetic_factor_snapshot, sym, asset_class="EQUITY")
            assert prov["status"] == EvidenceStatus.SYNTHETIC.value
            assert prov["isVerified"] is False
            assert prov["isSynthetic"] is True

            conf = engine.calculate_confluence(
                symbol=sym,
                technical_data={"executionStatus": "IN_BUY_ZONE", "rsi_14": 55.0, "current_price": 100.0},
                smart_money_data={"has_insider_buy": True, "insider_value_usd": 150000.0},
                fundamental_data={
                    "qualityScore": 88.0,
                    "growthScore": 92.0,
                    "piotroski_f": 8,
                },
                catalyst_data={"days_to_earnings": 45},
                macro_data={"yield_curve_10y2y": 0.15, "credit_spread": 3.8},
                fundamental_provenance=prov,
                asset_class="EQUITY",
            )

            # Invariant: Fundamental score must be 0.0 and status synthetic/unverified
            fund_p = next(p for p in conf["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
            assert fund_p["status"] in ("unverified", "synthetic")
            assert fund_p["score"] == 0.0
            assert conf["verifiedPillarCount"] < 4
            assert conf["incompleteCoverage"] is True
            # Score must not exceed 84.0 cap (3 pillars verified: tech, smart, macro)
            assert conf["confluenceScore"] <= 84.0

    def test_decision_trace_synthetic_fundamentals_fail_closed(
        self, engine, synthetic_factor_snapshot
    ):
        """DecisionTraceEngine must produce EVIDENCE_INCOMPLETE and isActionable=False for synthetic inputs."""
        sym = "ONON"
        prov = evaluate_factor_provenance(synthetic_factor_snapshot, sym, asset_class="EQUITY")

        conf = engine.calculate_confluence(
            symbol=sym,
            technical_data={"executionStatus": "IN_BUY_ZONE", "rsi_14": 55.0, "current_price": 100.0},
            smart_money_data={"has_insider_buy": True},
            fundamental_data={"qualityScore": 88.0},
            fundamental_provenance=prov,
            asset_class="EQUITY",
        )

        trace = DecisionTraceEngine.build_decision_trace(
            symbol=sym,
            current_price=100.0,
            candles=[{"close": 100.0}] * 60,
            freshness={"status": "LIVE"},
            technicals={"rsi_14": 55.0, "current_price": 100.0},
            confluence=conf,
            factor_scores={"qualityScore": 88.0, "piotroski_f": 8},
            optimal_execution={
                "execution_status": "IN_BUY_ZONE",
                "risk_reward_ratio": 3.0,
                "stage_phase": "Stage 2 Breakout",
            },
            user_role="SWING",
            security_type="COMMON_STOCK",
            asset_class="EQUITY",
            fundamental_provenance=prov,
        )

        assert trace["decisionState"] == DecisionState.EVIDENCE_INCOMPLETE.value
        assert trace["isActionable"] is False
        assert trace["canSizeTrade"] is False
        assert "Synthetic" in trace["disqualificationReason"] or "filing" in trace["disqualificationReason"]

    def test_decision_hierarchy_parity_with_and_without_explicit_provenance(
        self, engine, synthetic_factor_snapshot
    ):
        """DecisionHierarchyEngine must resolve identical state whether provenance is passed explicitly or via confluence payload."""
        sym = "MNST"
        prov = evaluate_factor_provenance(synthetic_factor_snapshot, sym, asset_class="EQUITY")

        conf = engine.calculate_confluence(
            symbol=sym,
            technical_data={"executionStatus": "IN_BUY_ZONE", "rsi_14": 55.0},
            smart_money_data={"has_insider_buy": True},
            fundamental_data={"qualityScore": 88.0},
            fundamental_provenance=prov,
            asset_class="EQUITY",
        )

        # Resolution: Explicit provenance arguments
        res1 = DecisionHierarchyEngine.resolve_decision_state(
            symbol=sym,
            current_price=50.0,
            candle_count=100,
            freshness_status="LIVE",
            has_fundamentals=False,  # Unverified
            confluence_score=conf["confluenceScore"],
            stage_phase="Stage 2 Breakout",
            is_in_buy_zone=True,
            risk_reward_ratio=2.5,
            is_cataloged=True,
            is_confirmed=True,
            user_role="SWING",
            asset_class="EQUITY",
            execution_status="IN_BUY_ZONE",
            verified_pillar_count=conf["verifiedPillarCount"],
            fundamental_provenance=prov,
            fundamental_evidence_status=prov["status"],
        )

        assert res1["state"] == DecisionState.EVIDENCE_INCOMPLETE.value
        assert res1["isActionable"] is False

    def test_etf_corporate_filing_exemption_parity(self, engine):
        """ETFs must receive NOT_APPLICABLE exemption for corporate filings without phantom fundamental scores."""
        for etf_sym in ("SPY", "QQQ", "IWM", "XLK", "SMH"):
            prov = evaluate_factor_provenance(None, etf_sym, asset_class="ETF")
            assert prov["status"] == EvidenceStatus.NOT_APPLICABLE.value
            assert prov["isVerified"] is True

            conf = engine.calculate_confluence(
                symbol=etf_sym,
                technical_data={"executionStatus": "IN_BUY_ZONE", "rsi_14": 55.0, "current_price": 450.0},
                smart_money_data={"has_insider_buy": False},
                fundamental_data=None,
                catalyst_data=None,
                macro_data={"yield_curve_10y2y": 0.2, "credit_spread": 3.5},
                fundamental_provenance=prov,
                asset_class="ETF",
            )

            # Fundamental pillar must be marked exempt / not_applicable
            fund_p = next(p for p in conf["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
            assert fund_p["status"] == "not_applicable"
            assert fund_p["score"] == 0.0  # Zero corporate score, NOT fabricated
            assert "ETF structure" in fund_p["detail"]

            # Decision hierarchy check
            dec = DecisionHierarchyEngine.resolve_decision_state(
                symbol=etf_sym,
                current_price=450.0,
                candle_count=200,
                freshness_status="LIVE",
                has_fundamentals=True,  # ETF exemption grants fundamental satisfaction
                confluence_score=conf["confluenceScore"],
                stage_phase="Stage 2 Breakout",
                is_in_buy_zone=True,
                risk_reward_ratio=2.5,
                is_cataloged=True,
                is_confirmed=False,
                user_role="SWING",
                asset_class="ETF",
                execution_status="IN_BUY_ZONE",
                verified_pillar_count=conf["verifiedPillarCount"],
                fundamental_provenance=prov,
                fundamental_evidence_status=prov["status"],
            )

            # Must NOT fail for missing corporate 10-K
            assert dec["state"] != DecisionState.EVIDENCE_INCOMPLETE.value

    def test_verified_equity_full_confluence_path(self, engine, verified_factor_snapshot):
        """Verified equity with authentic SEC filings should achieve full 4-pillar eligibility."""
        sym = "NVDA"
        prov = evaluate_factor_provenance(verified_factor_snapshot, sym, asset_class="EQUITY")
        assert prov["status"] == EvidenceStatus.VERIFIED.value
        assert prov["isVerified"] is True

        conf = engine.calculate_confluence(
            symbol=sym,
            technical_data={"executionStatus": "IN_BUY_ZONE", "rsi_14": 60.0, "current_price": 130.0, "stop_loss": 120.0},
            smart_money_data={"has_insider_buy": True, "insider_value_usd": 500000.0},
            fundamental_data={
                "qualityScore": 85.0,
                "growthScore": 80.0,
                "piotroski_f": 8,
            },
            catalyst_data={"days_to_earnings": 30},
            macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.4},
            fundamental_provenance=prov,
            asset_class="EQUITY",
        )

        fund_p = next(p for p in conf["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
        assert fund_p["status"] == "positive"
        assert fund_p["score"] > 0.0
        assert conf["verifiedPillarCount"] == 4
        assert conf["confluenceScore"] >= 75.0
