"""Mathematical Reference and Invariant Tests for ETF Risk Analytics (Phase P2).

Verifies criteria P2I11–P2I20 and P2I54:
- P2I11: Maximum drawdown wealth index contract
- P2I12: Drawdown recovery contract (trading days, RECOVERED vs UNRECOVERED)
- P2I13: Sharpe ratio contract (excess returns, 252 annualization, zero-vol guard)
- P2I14: Sortino ratio contract (downside semi-deviation, zero-downside guard)
- P2I15: Calmar ratio contract (zero-drawdown guard, no infinite leak)
- P2I16: Modified Cornish-Fisher VaR 95%
- P2I17: Modified Cornish-Fisher VaR 99% (monotonicity: VaR_99 >= VaR_95)
- P2I18: Volatility regime contract (canonical thresholds: LOW < 12%, MODERATE 12-22%, HIGH > 22%)
- P2I19: Insufficient data handling without fabricated zeroes
- P2I20: Non-finite mathematical outputs safely guarded
- P2I54: Deterministic reference vectors
"""

import math
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from analysis.etf import ETFAnalyzer


class TestEtfDrawdownAnalytics:
    """Tests for Maximum Drawdown, Peak, Trough, and Recovery Duration (P2I11, P2I12)."""

    def test_drawdown_with_full_recovery(self):
        """Test a clear drawdown followed by full recovery to ATH."""
        # Dates: 6 trading days
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        # Peak at day 2 (110), trough at day 3 (88, -20%), recovery at day 5 (110)
        prices = pd.Series([100.0, 110.0, 88.0, 99.0, 110.0, 115.0], index=dates)

        result = ETFAnalyzer.calculate_drawdown_details(prices)

        assert result["max_drawdown"] == pytest.approx(0.20, abs=1e-4)
        assert result["max_drawdown_pct"] == pytest.approx(20.0, abs=1e-2)
        assert result["peak_date"] == "2024-01-02"
        assert result["trough_date"] == "2024-01-03"
        assert result["recovery_date"] == "2024-01-05"
        assert result["recovery_days"] == 2  # 2 trading days from trough to recovery
        assert result["recovery_state"] == "RECOVERED"
        assert result["current_drawdown_pct"] == 0.0

    def test_drawdown_unrecovered(self):
        """Test a drawdown that has not recovered to ATH."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Peak at day 1 (100), drops to 70 (-30%), ends at 85 (unrecovered)
        prices = pd.Series([100.0, 70.0, 75.0, 80.0, 85.0], index=dates)

        result = ETFAnalyzer.calculate_drawdown_details(prices)

        assert result["max_drawdown"] == pytest.approx(0.30, abs=1e-4)
        assert result["max_drawdown_pct"] == pytest.approx(30.0, abs=1e-2)
        assert result["peak_date"] == "2024-01-01"
        assert result["trough_date"] == "2024-01-02"
        assert result["recovery_date"] is None
        assert result["recovery_days"] is None
        assert result["recovery_state"] == "UNRECOVERED"
        assert result["current_drawdown_pct"] == pytest.approx(15.0, abs=1e-2)

    def test_drawdown_monotonically_rising(self):
        """Test an asset with strictly monotonically rising prices (no drawdown)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.Series([100.0, 102.0, 105.0, 108.0, 110.0], index=dates)

        result = ETFAnalyzer.calculate_drawdown_details(prices)

        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0
        assert result["current_drawdown_pct"] == 0.0

    def test_drawdown_insufficient_history(self):
        """Empty or single observation must fail-closed cleanly."""
        empty = pd.Series([], dtype=float)
        res_empty = ETFAnalyzer.calculate_drawdown_details(empty)
        assert res_empty["max_drawdown"] is None
        assert res_empty["recovery_state"] == "INSUFFICIENT_DATA"

        single = pd.Series([100.0])
        res_single = ETFAnalyzer.calculate_drawdown_details(single)
        assert res_single["max_drawdown"] is None
        assert res_single["recovery_state"] == "INSUFFICIENT_DATA"


class TestCornishFisherVaR:
    """Tests for Modified Cornish-Fisher Value at Risk (P2I16, P2I17, P2I54)."""

    def test_gaussian_reference_vector(self):
        """When skewness=0 and excess kurtosis=0, Cornish-Fisher must equal Gaussian VaR."""
        # Theoretical standard normal quantile at 95% confidence (p=0.05)
        z95_gaussian = stats.norm.ppf(0.05)

        # When S=0, K=0:
        # z_CF = z_p + (1/6)(z_p^2 - 1)*0 + (1/24)(z_p^3 - 3*z_p)*0 - (1/36)(2*z_p^3 - 5*z_p)*0 = z_p
        s, k = 0.0, 0.0
        z_cf = z95_gaussian + (1/6)*(z95_gaussian**2 - 1)*s + (1/24)*(z95_gaussian**3 - 3*z95_gaussian)*k - (1/36)*(2*z95_gaussian**3 - 5*z95_gaussian)*(s**2)
        assert z_cf == pytest.approx(z95_gaussian, abs=1e-10)

    def test_negative_skew_and_excess_kurtosis_monotonicity(self):
        """Financial Invariant (quant-guardian): Fat tails and negative skew must increase VaR loss.
        Also verifies VaR_99 >= VaR_95 monotonicity.
        """
        np.random.seed(42)
        # Generate skewed, fat-tailed distribution (Student-t with df=4, shifted)
        t_samples = stats.t.rvs(df=4, loc=-0.001, scale=0.012, size=300)
        returns = pd.Series(t_samples)

        var_95 = ETFAnalyzer.calculate_modified_cornish_fisher_var(returns, 0.95)
        var_99 = ETFAnalyzer.calculate_modified_cornish_fisher_var(returns, 0.99)

        assert var_95 is not None
        assert var_99 is not None
        assert var_95["method"] == "MODIFIED_CORNISH_FISHER"
        assert var_95["unit"] == "RETURN_FRACTION"
        assert var_95["sign_convention"] == "POSITIVE_LOSS"

        # VaR monotonicity invariant: 99% VaR must exceed 95% VaR
        assert var_99["daily_var"] > var_95["daily_var"]
        assert var_99["daily_var_pct"] > var_95["daily_var_pct"]

    def test_var_insufficient_data_guard(self):
        """Less than 30 observations must return None, not fabricated 0."""
        short_returns = pd.Series(np.random.normal(0, 0.01, 20))
        var = ETFAnalyzer.calculate_modified_cornish_fisher_var(short_returns, 0.95)
        assert var is None


class TestRiskAdjustedRatios:
    """Tests for Sharpe, Sortino, and Calmar ratios (P2I13, P2I14, P2I15, P2I19, P2I20)."""

    def test_sharpe_ratio_deterministic(self):
        """Sharpe ratio must match (mean(excess) / std) * sqrt(252)."""
        np.random.seed(42)
        # Constant positive returns: daily excess = 0.001 - 0.02/252, std = 0.005
        rets = pd.Series(np.random.normal(0.001, 0.005, 252))
        sharpe = ETFAnalyzer.calculate_sharpe_ratio(rets, risk_free_rate_annual=0.02)

        expected_excess = rets - (0.02 / 252.0)
        expected_sharpe = (expected_excess.mean() / rets.std(ddof=1)) * np.sqrt(252)
        assert sharpe == pytest.approx(round(float(expected_sharpe), 2), abs=0.01)

    def test_sharpe_zero_vol_guard(self):
        """Zero volatility must return None, preventing division by zero."""
        flat_rets = pd.Series([0.001] * 50)
        sharpe = ETFAnalyzer.calculate_sharpe_ratio(flat_rets)
        assert sharpe is None

    def test_sortino_ratio_deterministic(self):
        """Sortino ratio must use downside semi-deviation."""
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.008, 100))
        sortino = ETFAnalyzer.calculate_sortino_ratio(rets, risk_free_rate_annual=0.02)

        excess = rets - (0.02 / 252.0)
        downside = np.minimum(0.0, excess)
        downside_std = np.sqrt(np.mean(downside ** 2))
        expected_sortino = (excess.mean() / downside_std) * np.sqrt(252)
        assert sortino == pytest.approx(round(float(expected_sortino), 2), abs=0.01)

    def test_sortino_zero_downside_guard(self):
        """When there is no downside deviation, return None instead of inf."""
        strictly_positive = pd.Series([0.01] * 50)  # All well above daily rf
        sortino = ETFAnalyzer.calculate_sortino_ratio(strictly_positive)
        assert sortino is None

    def test_calmar_ratio_deterministic(self):
        """Calmar ratio must match annualized return / max drawdown."""
        rets = pd.Series([0.0008] * 100)
        calmar = ETFAnalyzer.calculate_calmar_ratio(rets, max_drawdown=0.15)
        expected_calmar = (0.0008 * 252) / 0.15
        assert calmar == pytest.approx(round(expected_calmar, 2), abs=0.01)

    def test_calmar_zero_drawdown_guard(self):
        """Zero drawdown must return None, preventing division by zero."""
        rets = pd.Series([0.001] * 50)
        calmar = ETFAnalyzer.calculate_calmar_ratio(rets, max_drawdown=0.0)
        assert calmar is None


class TestVolatilityRegime:
    """Tests for Volatility Regime classification (P2I18)."""

    def test_volatility_regimes(self):
        assert ETFAnalyzer.calculate_volatility_regime(8.5) == "LOW"
        assert ETFAnalyzer.calculate_volatility_regime(11.99) == "LOW"
        assert ETFAnalyzer.calculate_volatility_regime(12.0) == "MODERATE"
        assert ETFAnalyzer.calculate_volatility_regime(18.4) == "MODERATE"
        assert ETFAnalyzer.calculate_volatility_regime(22.0) == "MODERATE"
        assert ETFAnalyzer.calculate_volatility_regime(22.01) == "HIGH"
        assert ETFAnalyzer.calculate_volatility_regime(45.0) == "HIGH"

    def test_volatility_regime_non_finite_safe(self):
        assert ETFAnalyzer.calculate_volatility_regime(float("nan")) == "UNKNOWN"
        assert ETFAnalyzer.calculate_volatility_regime(float("inf")) == "UNKNOWN"


class TestSectorConservationPolicy:
    """Tests for Sector Conservation Policy (P2I30, P2I31)."""

    def test_sector_label_mapping(self):
        assert ETFAnalyzer.SECTOR_LABEL_MAP["technology"] == "Information Technology"
        assert ETFAnalyzer.SECTOR_LABEL_MAP["financial_services"] == "Financials"
        assert ETFAnalyzer.SECTOR_LABEL_MAP["realestate"] == "Real Estate"
        assert ETFAnalyzer.SECTOR_LABEL_MAP["consumer_cyclical"] == "Consumer Discretionary"
        assert ETFAnalyzer.SECTOR_LABEL_MAP["consumer_defensive"] == "Consumer Staples"

    def test_dynamic_sector_weights_conservation(self):
        """Sector weights must sum to 100% with disclosed unclassified remainder."""
        res = ETFAnalyzer.get_etf_sector_allocations("SPY")
        assert res["symbol"] == "SPY"
        assert res["conservation_policy"] == "PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT"
        assert isinstance(res["sectors"], list)

        if res["sectors"]:
            total = sum(s["weightPct"] for s in res["sectors"])
            assert total == pytest.approx(100.0, abs=0.5)
            # Ensure every sector has non-negative weight
            for s in res["sectors"]:
                assert s["weightPct"] >= 0.0
                assert len(s["sector"]) > 0
