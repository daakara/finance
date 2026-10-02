"""
ETF (Exchange-Traded Fund) analysis module.
Specialized analysis for ETFs including sector allocation, holdings, and performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import scipy.stats as stats
import yfinance as yf

from data.cache import cache_result
from data.fetchers import stock_fetcher

logger = logging.getLogger(__name__)

class ETFAnalyzer:
    """Main ETF analysis class."""

    @staticmethod
    @cache_result(ttl=3600)  # Cache for 1 hour
    def get_etf_data(
        symbol: str,
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Get comprehensive ETF data including price history and info.

        Args:
            symbol: ETF symbol
            period: Time period for analysis

        Returns:
            Dict containing ETF data and analysis
        """
        results = {}

        try:
            # Get price data
            price_data = stock_fetcher.get_stock_data(symbol, period)
            results['price_data'] = price_data

            # Get ETF info
            etf_info = stock_fetcher.get_stock_info(symbol)
            results['etf_info'] = etf_info

            # Calculate ETF-specific metrics
            if not price_data.empty:
                results['performance_metrics'] = ETFAnalyzer._calculate_etf_performance(price_data)
                results['risk_metrics'] = ETFAnalyzer._calculate_etf_risk_metrics(price_data)

            return results

        except Exception as e:
            logger.error(f"Error analyzing ETF {symbol}: {str(e)}")
            return {'error': str(e)}

    @staticmethod
    def _calculate_etf_performance(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ETF performance metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}

        close_prices = price_data['Close']

        # Calculate returns
        daily_returns = close_prices.pct_change().dropna()

        # Performance metrics
        total_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        annualized_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) ** (252 / len(close_prices)) - 1) * 100

        # Volatility
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252) * 100

        # Best and worst periods
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100

        # Rolling performance
        monthly_returns = close_prices.resample('ME').last().pct_change().dropna()
        best_month = monthly_returns.max() * 100 if not monthly_returns.empty else 0
        worst_month = monthly_returns.min() * 100 if not monthly_returns.empty else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_month': best_month,
            'worst_month': worst_month,
            'trading_days': len(daily_returns)
        }

    @staticmethod
    def _calculate_etf_risk_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ETF risk metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}

        close_prices = price_data['Close']
        daily_returns = close_prices.pct_change().dropna()

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Value at Risk (5%)
        var_5 = np.percentile(daily_returns, 5) * 100

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

        # Sortino ratio (standard full-sample downside semi-deviation)
        downside_diff = np.minimum(0.0, excess_returns)
        downside_std = np.sqrt(np.mean(downside_diff ** 2))
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Calmar ratio
        calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown / 100) if max_drawdown != 0 else 0

        return {
            'max_drawdown': abs(max_drawdown),
            'var_5_percent': var_5,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    SECTOR_LABEL_MAP: Dict[str, str] = {
        "technology": "Information Technology",
        "financial_services": "Financials",
        "healthcare": "Healthcare",
        "consumer_cyclical": "Consumer Discretionary",
        "communication_services": "Communication Services",
        "industrials": "Industrials",
        "consumer_defensive": "Consumer Staples",
        "energy": "Energy",
        "utilities": "Utilities",
        "realestate": "Real Estate",
        "basic_materials": "Materials",
    }

    @staticmethod
    def calculate_drawdown_details(close_prices: pd.Series) -> Dict[str, Any]:
        """
        Calculate detailed wealth index drawdown metrics according to P2 Quant Contract.

        Defines:
        - wealth index W_t = P_t / P_0
        - running peak M_t = max_{s <= t} W_s
        - drawdown series DD_t = (W_t - M_t) / M_t
        - maximum drawdown MDD = min_t DD_t
        - peak date, trough date, recovery date, recovery days, recovery state
        """
        if close_prices.empty or len(close_prices) < 2:
            return {
                "max_drawdown": None,
                "max_drawdown_pct": None,
                "current_drawdown": None,
                "current_drawdown_pct": None,
                "peak_date": None,
                "trough_date": None,
                "recovery_date": None,
                "recovery_days": None,
                "recovery_state": "INSUFFICIENT_DATA"
            }

        # Calculate wealth index and running peak
        wealth_index = close_prices / close_prices.iloc[0]
        running_peak = wealth_index.cummax()
        drawdown_series = (wealth_index - running_peak) / running_peak  # values <= 0

        min_dd = float(drawdown_series.min())
        trough_idx = drawdown_series.idxmin()
        trough_date_str = trough_idx.strftime("%Y-%m-%d") if hasattr(trough_idx, "strftime") else str(trough_idx)[:10]

        # Peak date: last date where drawdown_series == 0 before or at trough
        dd_up_to_trough = drawdown_series.loc[:trough_idx]
        zero_dd_dates = dd_up_to_trough[dd_up_to_trough == 0.0]
        if not zero_dd_dates.empty:
            peak_idx = zero_dd_dates.index[-1]
            peak_date_str = peak_idx.strftime("%Y-%m-%d") if hasattr(peak_idx, "strftime") else str(peak_idx)[:10]
            peak_price = float(close_prices.loc[peak_idx])
        else:
            peak_idx = close_prices.index[0]
            peak_date_str = peak_idx.strftime("%Y-%m-%d") if hasattr(peak_idx, "strftime") else str(peak_idx)[:10]
            peak_price = float(close_prices.iloc[0])

        # Recovery date: first date after trough where price >= peak_price
        prices_after_trough = close_prices.loc[trough_idx:]
        if len(prices_after_trough) > 1:
            post_trough = prices_after_trough.iloc[1:]
            recovered_series = post_trough[post_trough >= peak_price]
            if not recovered_series.empty:
                recovery_idx = recovered_series.index[0]
                recovery_date_str = recovery_idx.strftime("%Y-%m-%d") if hasattr(recovery_idx, "strftime") else str(recovery_idx)[:10]
                # Trading days count from trough to recovery
                trading_days = len(close_prices.loc[trough_idx:recovery_idx]) - 1
                recovery_days = int(trading_days)
                recovery_state = "RECOVERED"
            else:
                recovery_date_str = None
                recovery_days = None
                recovery_state = "UNRECOVERED"
        else:
            recovery_date_str = None
            recovery_days = None
            recovery_state = "UNRECOVERED"

        current_dd = float(drawdown_series.iloc[-1])

        return {
            "max_drawdown": abs(min_dd),
            "max_drawdown_pct": round(abs(min_dd) * 100.0, 2),
            "current_drawdown": abs(current_dd),
            "current_drawdown_pct": round(abs(current_dd) * 100.0, 2),
            "peak_date": peak_date_str,
            "trough_date": trough_date_str,
            "recovery_date": recovery_date_str,
            "recovery_days": recovery_days,
            "recovery_state": recovery_state,
        }

    @staticmethod
    def calculate_modified_cornish_fisher_var(
        daily_returns: pd.Series,
        confidence: float = 0.95
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate Modified Cornish-Fisher Value at Risk (VaR).

        Formula:
        p = 1 - confidence
        z_p = Phi^-1(p)
        z_CF = z_p + (1/6)(z_p^2 - 1)S + (1/24)(z_p^3 - 3*z_p)K - (1/36)(2*z_p^3 - 5*z_p)S^2
        daily_loss_var = -(mu + z_CF * sigma)
        """
        clean_returns = daily_returns.dropna()
        n = len(clean_returns)
        if n < 30:
            return None

        mu = float(clean_returns.mean())
        sigma = float(clean_returns.std(ddof=1))
        if sigma <= 0.0 or np.isnan(sigma) or np.isinf(sigma):
            return None

        # Sample skewness (unbiased)
        skew = float(stats.skew(clean_returns, bias=False))
        # Sample excess kurtosis (Fisher=True means normal distribution kurtosis = 0)
        kurt = float(stats.kurtosis(clean_returns, fisher=True, bias=False))

        if np.isnan(skew) or np.isinf(skew) or np.isnan(kurt) or np.isinf(kurt):
            return None

        p = 1.0 - confidence
        z_p = float(stats.norm.ppf(p))

        z_cf = (
            z_p
            + (1.0 / 6.0) * (z_p**2 - 1.0) * skew
            + (1.0 / 24.0) * (z_p**3 - 3.0 * z_p) * kurt
            - (1.0 / 36.0) * (2.0 * z_p**3 - 5.0 * z_p) * (skew**2)
        )

        # Return quantile: r_q = mu + z_cf * sigma
        return_quantile = mu + z_cf * sigma

        # Express as positive loss fraction: loss = -return_quantile
        daily_var_loss = -return_quantile

        return {
            "confidence": confidence,
            "method": "MODIFIED_CORNISH_FISHER",
            "daily_var": float(daily_var_loss),
            "daily_var_pct": round(float(daily_var_loss * 100.0), 2),
            "unit": "RETURN_FRACTION",
            "sign_convention": "POSITIVE_LOSS",
            "z_gaussian": round(z_p, 4),
            "z_cornish_fisher": round(z_cf, 4),
            "skewness": round(skew, 4),
            "excess_kurtosis": round(kurt, 4),
            "observations": n
        }

    @staticmethod
    def calculate_sharpe_ratio(
        daily_returns: pd.Series,
        risk_free_rate_annual: float = 0.02
    ) -> Optional[float]:
        """Calculate annualized Sharpe ratio using standard 252 trading days."""
        clean = daily_returns.dropna()
        if len(clean) < 30:
            return None
        vol = float(clean.std(ddof=1))
        if vol <= 0.0 or np.isnan(vol) or np.isinf(vol):
            return None
        rf_daily = risk_free_rate_annual / 252.0
        excess = clean - rf_daily
        sharpe = (float(excess.mean()) / vol) * np.sqrt(252)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None
        return round(float(sharpe), 2)

    @staticmethod
    def calculate_sortino_ratio(
        daily_returns: pd.Series,
        risk_free_rate_annual: float = 0.02
    ) -> Optional[float]:
        """Calculate annualized Sortino ratio using downside semi-deviation."""
        clean = daily_returns.dropna()
        if len(clean) < 30:
            return None
        rf_daily = risk_free_rate_annual / 252.0
        excess = clean - rf_daily
        downside_diff = np.minimum(0.0, excess)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_std <= 0.0 or np.isnan(downside_std) or np.isinf(downside_std):
            return None
        sortino = (float(excess.mean()) / downside_std) * np.sqrt(252)
        if np.isnan(sortino) or np.isinf(sortino):
            return None
        return round(float(sortino), 2)

    @staticmethod
    def calculate_calmar_ratio(
        daily_returns: pd.Series,
        max_drawdown: float
    ) -> Optional[float]:
        """Calculate Calmar ratio: annualized return / max drawdown."""
        clean = daily_returns.dropna()
        if len(clean) < 30 or max_drawdown <= 0.0 or np.isnan(max_drawdown) or np.isinf(max_drawdown):
            return None
        ann_return = float(clean.mean()) * 252.0
        calmar = ann_return / abs(max_drawdown)
        if np.isnan(calmar) or np.isinf(calmar):
            return None
        return round(float(calmar), 2)

    @staticmethod
    def calculate_volatility_regime(annualized_volatility_pct: float) -> str:
        """
        Determine volatility regime from annualized volatility.
        Thresholds:
        - Low: < 12.0%
        - Moderate: 12.0% to 22.0%
        - High: > 22.0%
        """
        if np.isnan(annualized_volatility_pct) or np.isinf(annualized_volatility_pct):
            return "UNKNOWN"
        if annualized_volatility_pct < 12.0:
            return "LOW"
        elif annualized_volatility_pct <= 22.0:
            return "MODERATE"
        else:
            return "HIGH"

    @staticmethod
    @cache_result(ttl=86400)  # Cache for 24 hours
    def get_etf_sector_allocations(symbol: str) -> Dict[str, Any]:
        """
        Fetch dynamic sector allocations for ETF from authoritative fund data.
        Enforces conservation policy: PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT.
        """
        clean_sym = symbol.upper().replace("-USD", "").strip()
        as_of_str = datetime.now(timezone.utc).isoformat()

        sectors: List[Dict[str, Any]] = []
        try:
            ticker = yf.Ticker(clean_sym)
            fd = getattr(ticker, "funds_data", None)
            weightings = getattr(fd, "sector_weightings", None) if fd else None

            if weightings and isinstance(weightings, dict):
                total_classified = 0.0
                for raw_k, w in weightings.items():
                    if w is not None and not np.isnan(w) and w > 0:
                        w_pct = round(float(w) * 100.0, 2)
                        norm_sector = ETFAnalyzer.SECTOR_LABEL_MAP.get(raw_k.lower(), raw_k.replace("_", " ").title())
                        sectors.append({
                            "sector": norm_sector,
                            "weightPct": w_pct,
                            "raw_sector_key": raw_k
                        })
                        total_classified += w_pct

                # Sort descending by weight
                sectors.sort(key=lambda s: s["weightPct"], reverse=True)

                # Enforce conservation: PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT
                unclassified_pct = round(max(0.0, 100.0 - total_classified), 2)
                if unclassified_pct > 0.1:
                    sectors.append({
                        "sector": "Other / Unclassified",
                        "weightPct": unclassified_pct,
                        "raw_sector_key": "unclassified"
                    })

                return {
                    "symbol": clean_sym,
                    "as_of": as_of_str,
                    "source": "YFINANCE_FUNDS_DATA",
                    "conservation_policy": "PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT",
                    "total_weight_pct": 100.0,
                    "sectors": sectors,
                    "is_dynamic": True
                }
        except Exception as e:
            logger.warning(f"Error fetching dynamic sector allocations for {clean_sym}: {e}")

        return {
            "symbol": clean_sym,
            "as_of": as_of_str,
            "source": "NONE",
            "conservation_policy": "PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT",
            "total_weight_pct": 0.0,
            "sectors": [],
            "is_dynamic": False
        }

    @staticmethod
    @cache_result(ttl=3600)  # Cache for 1 hour
    def get_etf_risk_profile(symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get complete typed institutional risk profile for ETF.
        """
        clean_sym = symbol.upper().replace("-USD", "").strip()
        as_of_str = datetime.now(timezone.utc).isoformat()

        # 1. Fetch price data
        price_data = stock_fetcher.get_stock_data(clean_sym, period)
        if price_data is None or price_data.empty or "Close" not in price_data.columns:
            return {
                "symbol": clean_sym,
                "as_of": as_of_str,
                "history_start": None,
                "history_end": None,
                "period": period,
                "observation_count": 0,
                "quality": {
                    "state": "NOT_AVAILABLE",
                    "warnings": ["No price history available for instrument"]
                },
                "drawdown": {
                    "maximum": None,
                    "maximum_pct": None,
                    "current": None,
                    "current_pct": None,
                    "peak_date": None,
                    "trough_date": None,
                    "recovery_date": None,
                    "recovery_days": None,
                    "recovery_state": "INSUFFICIENT_DATA"
                },
                "risk_adjusted_returns": {
                    "sharpe": None,
                    "sortino": None,
                    "calmar": None
                },
                "value_at_risk": {
                    "var_95": None,
                    "var_99": None,
                    "horizon": "1d",
                    "method": "MODIFIED_CORNISH_FISHER",
                    "unit": "RETURN_FRACTION"
                },
                "volatility": {
                    "realized_annualized_pct": None,
                    "regime": "UNKNOWN",
                    "history_200d": []
                },
                "max_drawdown_pct": None,
                "max_drawdown_date": None,
                "recovery_days": None,
                "current_drawdown_pct": None,
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "calmar_ratio": None,
                "var_95_daily_pct": None,
                "var_99_daily_pct": None,
                "annualized_volatility_pct": None,
                "volatility_regime": "UNKNOWN",
                "vol_history_200d": [],
                "sectors": [],
                "source_provenance": "ARX_ETF_ANALYZER_V1"
            }

        close_prices = price_data["Close"]
        daily_returns = close_prices.pct_change().dropna()
        n_obs = len(daily_returns)

        # Dates
        history_start = close_prices.index[0].strftime("%Y-%m-%d") if hasattr(close_prices.index[0], "strftime") else str(close_prices.index[0])[:10]
        history_end = close_prices.index[-1].strftime("%Y-%m-%d") if hasattr(close_prices.index[-1], "strftime") else str(close_prices.index[-1])[:10]

        warnings = []
        if n_obs < 30:
            quality_state = "INSUFFICIENT_HISTORY"
            warnings.append(f"Insufficient history ({n_obs} sessions < 30 required)")
        elif n_obs < 126:
            quality_state = "PARTIAL"
            warnings.append(f"Short history ({n_obs} sessions < 126 for established VaR confidence)")
        else:
            quality_state = "ESTABLISHED"

        # Drawdown
        drawdown_details = ETFAnalyzer.calculate_drawdown_details(close_prices)

        # Volatility
        daily_vol = float(daily_returns.std(ddof=1)) if n_obs >= 2 else 0.0
        ann_vol_pct = round(daily_vol * np.sqrt(252) * 100.0, 2) if daily_vol > 0 else None
        vol_regime = ETFAnalyzer.calculate_volatility_regime(ann_vol_pct) if ann_vol_pct is not None else "UNKNOWN"

        # Rolling 200d volatility for sparkline (last 200 points of rolling 20-day vol annualized)
        vol_history: List[float] = []
        if len(daily_returns) >= 20:
            rolling_vol = daily_returns.rolling(window=20).std(ddof=1) * np.sqrt(252) * 100.0
            rolling_vol_clean = rolling_vol.dropna()
            vol_history = [round(float(v), 2) for v in rolling_vol_clean.tail(200).tolist() if not np.isnan(v)]

        # Risk adjusted returns
        sharpe = ETFAnalyzer.calculate_sharpe_ratio(daily_returns)
        sortino = ETFAnalyzer.calculate_sortino_ratio(daily_returns)
        max_dd_val = drawdown_details.get("max_drawdown") or 0.0
        calmar = ETFAnalyzer.calculate_calmar_ratio(daily_returns, max_dd_val)

        # Value at Risk
        var_95 = ETFAnalyzer.calculate_modified_cornish_fisher_var(daily_returns, 0.95)
        var_99 = ETFAnalyzer.calculate_modified_cornish_fisher_var(daily_returns, 0.99)

        # Sectors
        sector_payload = ETFAnalyzer.get_etf_sector_allocations(clean_sym)
        sectors_list = sector_payload.get("sectors", [])

        return {
            "symbol": clean_sym,
            "as_of": as_of_str,
            "history_start": history_start,
            "history_end": history_end,
            "period": period,
            "observation_count": n_obs,
            "quality": {
                "state": quality_state,
                "warnings": warnings
            },
            "drawdown": {
                "maximum": drawdown_details["max_drawdown"],
                "maximum_pct": drawdown_details["max_drawdown_pct"],
                "current": drawdown_details["current_drawdown"],
                "current_pct": drawdown_details["current_drawdown_pct"],
                "peak_date": drawdown_details["peak_date"],
                "trough_date": drawdown_details["trough_date"],
                "recovery_date": drawdown_details["recovery_date"],
                "recovery_days": drawdown_details["recovery_days"],
                "recovery_state": drawdown_details["recovery_state"]
            },
            "risk_adjusted_returns": {
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar
            },
            "value_at_risk": {
                "var_95": var_95,
                "var_99": var_99,
                "horizon": "1d",
                "method": "MODIFIED_CORNISH_FISHER",
                "unit": "RETURN_FRACTION"
            },
            "volatility": {
                "realized_annualized_pct": ann_vol_pct,
                "regime": vol_regime,
                "history_200d": vol_history
            },
            "max_drawdown_pct": drawdown_details["max_drawdown_pct"],
            "max_drawdown_date": drawdown_details["trough_date"],
            "recovery_days": drawdown_details["recovery_days"],
            "current_drawdown_pct": drawdown_details["current_drawdown_pct"],
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "var_95_daily_pct": var_95["daily_var_pct"] if var_95 else None,
            "var_99_daily_pct": var_99["daily_var_pct"] if var_99 else None,
            "annualized_volatility_pct": ann_vol_pct,
            "volatility_regime": vol_regime,
            "vol_history_200d": vol_history,
            "sectors": sectors_list,
            "source_provenance": "ARX_ETF_ANALYZER_V1"
        }

    @staticmethod
    def get_popular_etfs() -> Dict[str, Dict[str, str]]:
        """Get list of popular ETFs by category."""
        return {
            'Broad Market': {
                'SPY': 'SPDR S&P 500 ETF',
                'VTI': 'Vanguard Total Stock Market ETF',
                'IWM': 'iShares Russell 2000 ETF',
                'QQQ': 'Invesco QQQ Trust'
            },
            'International': {
                'EFA': 'iShares MSCI EAFE ETF',
                'EEM': 'iShares MSCI Emerging Markets ETF',
                'VEA': 'Vanguard FTSE Developed Markets ETF',
                'VWO': 'Vanguard FTSE Emerging Markets ETF'
            },
            'Sector': {
                'XLK': 'Technology Select Sector SPDR Fund',
                'XLF': 'Financial Select Sector SPDR Fund',
                'XLE': 'Energy Select Sector SPDR Fund',
                'XLV': 'Health Care Select Sector SPDR Fund',
                'XLI': 'Industrial Select Sector SPDR Fund'
            },
            'Fixed Income': {
                'AGG': 'iShares Core U.S. Aggregate Bond ETF',
                'TLT': 'iShares 20+ Year Treasury Bond ETF',
                'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
                'LQD': 'iShares iBoxx Investment Grade Corporate Bond ETF'
            },
            'Commodities': {
                'GLD': 'SPDR Gold Shares',
                'SLV': 'iShares Silver Trust',
                'USO': 'United States Oil Fund',
                'DBA': 'Invesco DB Agriculture Fund'
            },
            'Thematic': {
                'ARKK': 'ARK Innovation ETF',
                'ICLN': 'iShares Global Clean Energy ETF',
                'HACK': 'ETFMG Prime Cyber Security ETF',
                'ROBO': 'ROBO Global Robotics and Automation Index ETF'
            }
        }

    @staticmethod
    def compare_etfs(
        etf_symbols: List[str],
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Compare multiple ETFs across various metrics.

        Args:
            etf_symbols: List of ETF symbols to compare
            period: Time period for comparison

        Returns:
            Dict containing comparison data
        """
        comparison_data = {}
        price_data = {}

        # Fetch data for all ETFs
        for symbol in etf_symbols:
            try:
                etf_data = ETFAnalyzer.get_etf_data(symbol, period)
                if 'price_data' in etf_data and not etf_data['price_data'].empty:
                    price_data[symbol] = etf_data['price_data']['Close']
                    comparison_data[symbol] = {
                        'performance': etf_data.get('performance_metrics', {}),
                        'risk': etf_data.get('risk_metrics', {}),
                        'info': etf_data.get('etf_info', {})
                    }
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue

        # Create comparison tables
        results = {
            'price_data': pd.DataFrame(price_data),
            'comparison_data': comparison_data
        }

        if comparison_data:
            # Performance comparison table
            perf_metrics = ['total_return', 'annualized_return', 'annualized_volatility']
            perf_table = []

            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in perf_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('performance', {}).get(metric, 0)
                perf_table.append(row)

            results['performance_comparison'] = pd.DataFrame(perf_table)

            # Risk comparison table
            risk_metrics = ['max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'var_5_percent']
            risk_table = []

            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in risk_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('risk', {}).get(metric, 0)
                risk_table.append(row)

            results['risk_comparison'] = pd.DataFrame(risk_table)

        return results

    @staticmethod
    def get_etf_expense_analysis(etf_info: Dict) -> Dict[str, Union[float, str]]:
        """
        Analyze ETF expenses and fees.

        Args:
            etf_info: ETF information dictionary

        Returns:
            Dict with expense analysis
        """
        # Extract expense ratio (this would need real data source)
        # For now, provide typical expense ratios by ETF type
        symbol = etf_info.get('symbol', '')

        typical_expenses = {
            'SPY': 0.09, 'VTI': 0.03, 'QQQ': 0.20, 'IWM': 0.19,
            'EFA': 0.32, 'EEM': 0.68, 'GLD': 0.40, 'TLT': 0.15,
            'XLK': 0.12, 'XLF': 0.12, 'ARKK': 0.75
        }

        expense_ratio = typical_expenses.get(symbol, 0.50)  # Default 0.5%

        # Calculate cost impact on $10,000 investment
        annual_cost = 10000 * (expense_ratio / 100)

        # Categorize expense level
        if expense_ratio < 0.20:
            expense_category = 'Very Low'
        elif expense_ratio < 0.50:
            expense_category = 'Low'
        elif expense_ratio < 0.75:
            expense_category = 'Moderate'
        else:
            expense_category = 'High'

        return {
            'expense_ratio': expense_ratio,
            'annual_cost_10k': annual_cost,
            'expense_category': expense_category,
            'cost_over_10_years': annual_cost * 10
        }

# Create instance for easy importing
etf_analyzer = ETFAnalyzer()
