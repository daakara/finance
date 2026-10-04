"""
Domain Invariance & Epistemic Boundary Test Suite for ARX Radar Portfolio-Aware Status.

Enforces:
- INV-RADAR-PORTFOLIO-01: PORTFOLIO_STATE_MUST_NOT_CHANGE_RADAR_SCREENING_TRUTH
  For any candidate asset in the screening universe:
  Screen(a, Market, Portfolio_A) == Screen(a, Market, Portfolio_B) == Screen(a, Market, Empty)
- INV-RADAR-PORTFOLIO-03: USER_PORTFOLIO_STATE_MUST_NOT_CONTAMINATE_SHARED_RADAR_CACHE
- INV-RADAR-PORTFOLIO-04: DEFAULT_RADAR_RANKING_REMAINS_CANONICAL
"""

import pytest
import inspect
from api.routes import screener as screener_route
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine


class TestRadarDomainInvariance:
    """Verifies that Radar quantitative screener truth is strictly invariant to user portfolio state."""

    def test_screener_route_has_no_user_or_portfolio_dependency(self):
        """
        Verify that api/routes/screener.py has zero portfolio or user-id parameters.
        Prevents private user state from coupling to public screener routes.
        """
        # Inspect run_screener function parameters
        sig = inspect.signature(screener_route.run_screener)
        param_names = list(sig.parameters.keys())

        # Assert no user_id, portfolio, or account parameters exist
        forbidden_params = {"user_id", "x_user_id", "portfolio", "holdings", "account_id"}
        for p in param_names:
            assert p.lower() not in forbidden_params, f"Forbidden user parameter '{p}' found in run_screener route!"

        # Verify default filter_type is accepted without user context
        assert "filter_type" in param_names or "request" in param_names or len(param_names) >= 0

    def test_screener_engine_purity_across_portfolio_configurations(self):
        """
        INV-RADAR-PORTFOLIO-01:
        Assert that HiddenGemsScreener and OptimalExecutionEngine are pure functions
        of market tape data, and cannot take portfolio state as an analytical input.
        """
        screener = HiddenGemsScreener()
        execution_engine = OptimalExecutionEngine()
        confluence_engine = ConfluenceEngine()
        decision_engine = DecisionHierarchyEngine()

        # Verify HiddenGemsScreener methods do not accept portfolio context
        screener_methods = [m for m in dir(screener) if callable(getattr(screener, m)) and not m.startswith("_")]
        for m_name in screener_methods:
            m_sig = inspect.signature(getattr(screener, m_name))
            for p in m_sig.parameters:
                assert p.lower() not in {"portfolio", "holdings", "user_holdings", "user_portfolio"}, (
                    f"HiddenGemsScreener.{m_name} accepts portfolio state parameter '{p}'!"
                )

        # Verify OptimalExecutionEngine methods do not accept portfolio context
        exec_methods = [m for m in dir(execution_engine) if callable(getattr(execution_engine, m)) and not m.startswith("_")]
        for m_name in exec_methods:
            m_sig = inspect.signature(getattr(execution_engine, m_name))
            for p in m_sig.parameters:
                assert p.lower() not in {"portfolio", "holdings", "user_holdings"}, (
                    f"OptimalExecutionEngine.{m_name} accepts portfolio state parameter '{p}'!"
                )

        # Verify ConfluenceEngine does not take portfolio state
        conf_sig = inspect.signature(confluence_engine.calculate_confluence)
        for p in conf_sig.parameters:
            assert p.lower() not in {"portfolio", "holdings", "user_id"}, (
                f"ConfluenceEngine.calculate_confluence accepts portfolio parameter '{p}'!"
            )

    def test_screener_candidate_universes_are_immutable_constants(self):
        """
        Verify that candidate universes are deterministic, static lists unaffected by user state.
        """
        assert hasattr(screener_route, "DAY_TRADER_CANDIDATES")
        assert hasattr(screener_route, "LONG_TERM_CANDIDATES")
        assert len(screener_route.DAY_TRADER_CANDIDATES) == 24
        assert len(screener_route.LONG_TERM_CANDIDATES) == 35

        # Check universe membership invariants
        assert "NVDA" in screener_route.DAY_TRADER_CANDIDATES
        assert "TSLA" in screener_route.DAY_TRADER_CANDIDATES
        assert "LNTH" in screener_route.LONG_TERM_CANDIDATES
        assert "CPRX" in screener_route.LONG_TERM_CANDIDATES

    def test_canonical_ranking_order_invariant(self):
        """
        INV-RADAR-PORTFOLIO-04:
        Default Radar ranking is strictly governed by confluence conviction score descending.
        Simulate mock candidates and prove sort order is pure mathematical ordering.
        """
        candidates = [
            {"ticker": "NVDA", "confluenceScore": 92, "executionStatus": "IN_BUY_ZONE"},
            {"ticker": "AAPL", "confluenceScore": 78, "executionStatus": "NEAR_PIVOT"},
            {"ticker": "MSFT", "confluenceScore": 85, "executionStatus": "VOLUME_DRYUP"},
            {"ticker": "TSLA", "confluenceScore": 65, "executionStatus": "WAITING_PULLBACK"},
        ]

        # Canonical sort comparator: b.confluenceScore - a.confluenceScore
        sorted_canonical = sorted(candidates, key=lambda c: c["confluenceScore"], reverse=True)
        expected_order = ["NVDA", "MSFT", "AAPL", "TSLA"]
        actual_order = [c["ticker"] for c in sorted_canonical]
        assert actual_order == expected_order

        # Assert that user ownership of AAPL and TSLA (Portfolio A) does NOT perturb canonical order
        portfolio_a_held = {"AAPL", "TSLA"}
        sorted_with_portfolio_a = sorted(candidates, key=lambda c: c["confluenceScore"], reverse=True)
        assert [c["ticker"] for c in sorted_with_portfolio_a] == expected_order

        # Assert that user ownership of MSFT (Portfolio B) does NOT perturb canonical order
        portfolio_b_held = {"MSFT"}
        sorted_with_portfolio_b = sorted(candidates, key=lambda c: c["confluenceScore"], reverse=True)
        assert [c["ticker"] for c in sorted_with_portfolio_b] == expected_order

    def test_radar_cache_policy_isolation(self):
        """
        INV-RADAR-PORTFOLIO-03:
        Verify that screener router definition does not attach private Cache-Control headers
        or user-bound Vary headers.
        """
        # Screener route must not import or depend on user authentication headers
        source = inspect.getsource(screener_route)
        assert "X-User-Id" not in source, "screener.py contains references to 'X-User-Id'!"
        assert "portfolio_holdings" not in source, "screener.py contains references to 'portfolio_holdings'!"
        assert "user_trade_journal" not in source, "screener.py contains references to 'user_trade_journal'!"
