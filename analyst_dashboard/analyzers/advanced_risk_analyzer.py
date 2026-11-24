"""
Advanced Risk Analysis Module
Sophisticated risk metrics beyond basic volatility and Sharpe ratio
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

REQUIRED_CORNISH_FISHER_OBSERVATIONS = 30

_EXCHANGE_CALENDARS: Dict[str, Any] = {}

SUPPORTED_EXCHANGE_MAP: Dict[str, Tuple[str, str]] = {
    # US Equities / ETFs
    "XNYS": ("XNYS", "America/New_York"),
    "NYSE": ("XNYS", "America/New_York"),
    "NYQ": ("XNYS", "America/New_York"),

    "XNAS": ("XNAS", "America/New_York"),
    "NASDAQ": ("XNAS", "America/New_York"),
    "NMS": ("XNAS", "America/New_York"),
    "NGS": ("XNAS", "America/New_York"),
    "NCM": ("XNAS", "America/New_York"),

    "ARCX": ("ARCX", "America/New_York"),
    "ARCA": ("ARCX", "America/New_York"),
    "PCX": ("ARCX", "America/New_York"),
    "ASE": ("ARCX", "America/New_York"),

    "BATS": ("BATS", "America/New_York"),
    "BZX": ("BATS", "America/New_York"),
    "BTS": ("BATS", "America/New_York"),

    # Crypto 24/7
    "24/7": ("24/7", "UTC"),
    "CRYPTO": ("24/7", "UTC"),
    "CCC": ("24/7", "UTC"),
    "CCY": ("24/7", "UTC"),
    "CRYPTO_24_7": ("24/7", "UTC"),
}


def get_exchange_calendar(exchange: str = "XNYS") -> Optional[Any]:
    """Cached accessor for exchange calendar instance."""
    if exchange not in _EXCHANGE_CALENDARS:
        try:
            import exchange_calendars as xcals
            _EXCHANGE_CALENDARS[exchange] = xcals.get_calendar(exchange)
        except Exception as e:
            logger.debug(f"exchange_calendars {exchange} fallback: {e}")
            _EXCHANGE_CALENDARS[exchange] = None
    return _EXCHANGE_CALENDARS.get(exchange)


def resolve_calendar_metadata(
    exchange: Optional[str] = None,
    asset_class: Optional[str] = None,
    timezone: Optional[str] = None,
    is_crypto: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolves exchange calendar code, timezone, and error reason (if unverified or unsupported).

    Returns:
        (cal_code, tz_name, error_reason, error_message)
    """
    clean_ac = str(asset_class).strip().upper() if asset_class else ""
    if is_crypto or clean_ac == "CRYPTO":
        return "24/7", "UTC", None, None

    clean_ex = str(exchange).strip().upper() if exchange else ""
    if not clean_ex or clean_ex in ("UNKNOWN", "UNVERIFIED", "NONE"):
        return None, None, "UNVERIFIED_EXCHANGE_CALENDAR", (
            f"Exchange calendar is unverified for exchange '{exchange or 'UNKNOWN'}'; "
            "Cornish-Fisher risk estimates require verified exchange calendar session alignment."
        )

    if clean_ex in SUPPORTED_EXCHANGE_MAP:
        cal_code, default_tz = SUPPORTED_EXCHANGE_MAP[clean_ex]
        tz_name = timezone or default_tz
        return cal_code, tz_name, None, None

    return None, None, "UNSUPPORTED_EXCHANGE_CALENDAR", (
        f"Exchange calendar '{exchange}' is not supported; "
        "Cornish-Fisher risk estimates require a supported exchange calendar (XNYS, XNAS, ARCX, BATS, 24/7)."
    )


def extract_valid_returns(
    price_data: Any,
    exchange: Optional[str] = None,
    asset_class: Optional[str] = None,
    timezone: Optional[str] = None,
    is_crypto: bool = False,
    require_verified_calendar: bool = False,
) -> pd.Series:
    """Extract valid finite 1-session daily return observations from price data.

    Invariants (Directive TASK-OPEN-03):
    1. Input prices must be strictly positive (> 0) and finite.
    2. Consecutive session requirement: Never forward-fill missing prices or bridge
       missing sessions. A missing or non-positive price at session t suppresses returns
       at both session t and session t+1.
    3. Authoritative exchange calendar and timezone alignment:
       - When price data has a date/datetime index, the series is aligned against the
         asset's authoritative exchange calendar (e.g. XNYS for NYSE, XNAS for NASDAQ,
         ARCX for NYSE Arca, BATS for BATS, or 24/7 daily calendar for crypto).
       - If exchange is UNKNOWN, UNVERIFIED, '', None and metadata is unverified:
         Fails closed to UNAVAILABLE with reason UNVERIFIED_EXCHANGE_CALENDAR.
       - If exchange is unsupported (e.g. LSE, TSX):
         Fails closed to UNAVAILABLE with reason UNSUPPORTED_EXCHANGE_CALENDAR.
       - Omitted trading dates (missing entirely from input) are exposed as explicit NaNs,
         preventing multi-session price jumps from being counted as 1-session returns.
       - Valid transitions across weekends and exchange holidays are preserved as consecutive 1-session returns.
    4. Synthetic non-date test fixtures (with pure integer RangeIndex):
       - Can evaluate numeric samples without calendar alignment for pure statistical unit tests
         when require_verified_calendar is False.
    5. Output contains only valid, finite returns.
    """
    if price_data is None:
        return pd.Series(dtype=float)

    if isinstance(price_data, pd.DataFrame):
        if "Close" in price_data.columns:
            prices = price_data["Close"]
        elif "close" in price_data.columns:
            prices = price_data["close"]
        else:
            prices = price_data.iloc[:, 0] if len(price_data.columns) > 0 else pd.Series(dtype=float)
    elif isinstance(price_data, pd.Series):
        prices = price_data
    else:
        try:
            prices = pd.Series(price_data, dtype=float)
        except Exception:
            return pd.Series(dtype=float)

    if len(prices) == 0:
        return pd.Series(dtype=float)

    numeric_prices = pd.to_numeric(prices, errors="coerce")
    # Strictly positive and finite prices
    valid_prices = numeric_prices.where((numeric_prices > 0) & np.isfinite(numeric_prices))

    # Check for datetime-like index
    dt_index = None
    if isinstance(valid_prices.index, pd.DatetimeIndex):
        dt_index = valid_prices.index
    elif isinstance(valid_prices.index, pd.Index) and not isinstance(valid_prices.index, (pd.RangeIndex, pd.MultiIndex)):
        try:
            parsed = pd.to_datetime(valid_prices.index, errors="coerce")
            if parsed.notna().sum() >= max(2, int(len(valid_prices) * 0.8)):
                dt_index = parsed
        except Exception:
            dt_index = None

    if dt_index is not None and len(dt_index) >= 2:
        cal_code, tz_name, cal_err, cal_msg = resolve_calendar_metadata(
            exchange=exchange,
            asset_class=asset_class,
            timezone=timezone,
            is_crypto=is_crypto,
        )
        if cal_err is not None:
            empty_s = pd.Series(dtype=float)
            empty_s.attrs["calendar_error"] = cal_err
            empty_s.attrs["calendar_message"] = cal_msg
            return empty_s

        try:
            if cal_code == "24/7":
                # 24/7/365 daily calendar for crypto
                if dt_index.tz is not None:
                    local_dates = dt_index.tz_convert("UTC").normalize().tz_localize(None)
                else:
                    local_dates = dt_index.normalize()

                p = valid_prices.copy()
                p.index = local_dates
                p = p[~p.index.duplicated(keep="last")]

                min_date = local_dates.min()
                max_date = local_dates.max()
                if min_date != max_date:
                    full_days = pd.date_range(min_date, max_date, freq="D")
                    aligned_prices = p.reindex(full_days)
                    rets = aligned_prices.pct_change(fill_method=None)
                    valid_rets = rets[np.isfinite(rets)]
                    return valid_rets
            else:
                cal = get_exchange_calendar(cal_code)
                if cal is None:
                    empty_s = pd.Series(dtype=float)
                    empty_s.attrs["calendar_error"] = "UNSUPPORTED_EXCHANGE_CALENDAR"
                    empty_s.attrs["calendar_message"] = f"Failed to load exchange calendar for '{cal_code}'."
                    return empty_s

                # Authoritative Exchange Calendar
                effective_tz = tz_name or getattr(cal, "tz", "America/New_York")
                if dt_index.tz is not None:
                    local_dates = dt_index.tz_convert(effective_tz).normalize().tz_localize(None)
                else:
                    local_dates = dt_index.normalize()

                p = valid_prices.copy()
                p.index = local_dates
                p = p[~p.index.duplicated(keep="last")]

                start_str = str(local_dates.min().date())
                end_str = str(local_dates.max().date())
                sessions = cal.sessions_in_range(start_str, end_str)
                if len(sessions) > 0:
                    sessions_idx = pd.to_datetime(sessions.date)
                    aligned_prices = p.reindex(sessions_idx)
                    rets = aligned_prices.pct_change(fill_method=None)
                    valid_rets = rets[np.isfinite(rets)]
                    return valid_rets
        except Exception as cal_err_exc:
            logger.debug(f"Calendar session alignment exception: {cal_err_exc}")

    # Fallback / Sequential Index without calendar (for pure RangeIndex synthetic test fixtures)
    if require_verified_calendar and dt_index is None:
        empty_s = pd.Series(dtype=float)
        empty_s.attrs["calendar_error"] = "UNVERIFIED_EXCHANGE_CALENDAR"
        empty_s.attrs["calendar_message"] = "Price data lacks date/datetime index and cannot be aligned against a verified exchange calendar."
        return empty_s

    # CRITICAL: Without forward-fill (fill_method=None), pct_change produces NaN at index t
    # if either valid_prices[t-1] is NaN or valid_prices[t] is NaN.
    # It NEVER forward-fills missing prices and NEVER bridges missing sessions.
    rets = valid_prices.pct_change(fill_method=None)
    valid_rets = rets[np.isfinite(rets)]
    return valid_rets


class AdvancedRiskAnalyzer:
    """Advanced risk analysis capabilities for professional-grade insights"""
    
    def analyze_comprehensive_risk(
        self,
        price_data: Any, 
        benchmark_data: Optional[Any] = None,
        exchange: Optional[str] = None,
        asset_class: Optional[str] = None,
        timezone: Optional[str] = None,
        is_crypto: bool = False,
        require_verified_calendar: bool = False,
    ) -> Dict[str, Any]:
        """Perform comprehensive risk analysis with Cornish-Fisher sufficiency gating."""
        try:
            valid_returns = extract_valid_returns(
                price_data,
                exchange=exchange,
                asset_class=asset_class,
                timezone=timezone,
                is_crypto=is_crypto,
                require_verified_calendar=require_verified_calendar,
            )
            adv_metrics = self._calculate_advanced_risk_metrics(valid_returns)
            cf_state = adv_metrics.get("cornish_fisher", {
                "status": "UNAVAILABLE",
                "reason": "INSUFFICIENT_OBSERVATIONS",
                "actual_observations": len(valid_returns),
                "required_observations": REQUIRED_CORNISH_FISHER_OBSERVATIONS,
                "message": f"Cornish-Fisher risk estimates require at least 30 valid closed daily return observations (found {len(valid_returns)})."
            })

            # Extract price series for drawdowns
            if isinstance(price_data, pd.DataFrame):
                close_series = price_data["Close"] if "Close" in price_data.columns else (
                    price_data["close"] if "close" in price_data.columns else price_data.iloc[:, 0]
                )
            elif isinstance(price_data, pd.Series):
                close_series = price_data
            else:
                close_series = pd.Series(dtype=float)

            clean_close = pd.to_numeric(close_series, errors="coerce").dropna()
            
            risk_analysis = {
                'advanced_metrics': adv_metrics,
                'drawdown_analysis': self._analyze_drawdowns(clean_close) if len(clean_close) > 0 else {},
                'regime_analysis': self._detect_market_regimes(valid_returns) if len(valid_returns) > 0 else {},
                'cornish_fisher': cf_state
            }
            
            if benchmark_data is not None:
                benchmark_returns = extract_valid_returns(
                    benchmark_data,
                    exchange=exchange,
                    asset_class=asset_class,
                    timezone=timezone,
                    is_crypto=is_crypto,
                    require_verified_calendar=require_verified_calendar,
                )
                if hasattr(self, "_analyze_relative_risk"):
                    risk_analysis['relative_risk'] = self._analyze_relative_risk(
                        valid_returns, benchmark_returns
                    )

            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk analysis: {str(e)}")
            return {
                'error': str(e),
                'advanced_metrics': {},
                'cornish_fisher': {
                    'status': 'UNAVAILABLE',
                    'reason': 'CALCULATION_ERROR',
                    'actual_observations': 0,
                    'required_observations': REQUIRED_CORNISH_FISHER_OBSERVATIONS,
                    'message': str(e)
                }
            }
    
    def _calculate_advanced_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate sophisticated risk metrics with strict Cornish-Fisher sufficiency gating (>= 30 valid returns)."""
        try:
            metrics = {}

            # 1. Clean and count valid finite return observations
            cal_err = returns.attrs.get("calendar_error") if hasattr(returns, "attrs") else None
            cal_msg = returns.attrs.get("calendar_message") if hasattr(returns, "attrs") else None

            if returns is None or len(returns) == 0:
                clean_returns = pd.Series(dtype=float)
            else:
                num_rets = pd.to_numeric(returns, errors="coerce")
                clean_returns = num_rets[np.isfinite(num_rets)]

            n_obs = len(clean_returns)

            # 2. Evaluate Cornish-Fisher Data Sufficiency and Validity Gates
            if cal_err is not None:
                cf_available = False
                cf_reason = cal_err
                cf_message = cal_msg or f"Exchange calendar error: {cal_err}"
            elif n_obs < REQUIRED_CORNISH_FISHER_OBSERVATIONS:
                cf_available = False
                cf_reason = "INSUFFICIENT_OBSERVATIONS"
                cf_message = f"Cornish-Fisher risk estimates require at least 30 valid closed daily return observations (found {n_obs})."
            else:
                # Sample variance / degeneracy check
                std_ret = float(clean_returns.std())
                if std_ret <= 0.0 or np.isnan(std_ret) or np.isinf(std_ret):
                    cf_available = False
                    cf_reason = "DEGENERATE_SAMPLE"
                    cf_message = "Return series has zero or non-finite variance; Cornish-Fisher expansion is undefined."
                else:
                    raw_skew = float(stats.skew(clean_returns))
                    raw_kurt = float(stats.kurtosis(clean_returns))
                    if np.isnan(raw_skew) or np.isnan(raw_kurt) or np.isinf(raw_skew) or np.isinf(raw_kurt):
                        cf_available = False
                        cf_reason = "NON_FINITE_DISTRIBUTION"
                        cf_message = "Skewness or kurtosis is non-finite; Cornish-Fisher expansion is undefined."
                    else:
                        cf_available = True
                        cf_reason = None
                        cf_message = None

            # 3. Empirical Percentile & Ratio Metrics (computed when sample has finite returns)
            if n_obs > 0:
                metrics['VaR_95'] = float(np.percentile(clean_returns, 5) * 100)
                metrics['VaR_99'] = float(np.percentile(clean_returns, 1) * 100)
                
                p5 = float(np.percentile(clean_returns, 5))
                p95 = float(np.percentile(clean_returns, 95))
                metrics['Tail_Ratio'] = float(abs(p95 / p5)) if p5 != 0 else 1.25
                
                tail_5 = clean_returns[clean_returns <= p5]
                metrics['CVaR_95'] = float(tail_5.mean() * 100) if len(tail_5) > 0 else metrics['VaR_95']

                p1 = float(np.percentile(clean_returns, 1))
                tail_1 = clean_returns[clean_returns <= p1]
                metrics['CVaR_99'] = float(tail_1.mean() * 100) if len(tail_1) > 0 else metrics['VaR_99']

                annual_return = float(clean_returns.mean() * 252)
                max_dd = float(self._calculate_max_drawdown(clean_returns))
                metrics['Calmar_Ratio'] = float(annual_return / max(0.001, abs(max_dd))) if max_dd != 0 else 0.0
                
                risk_free_rate = 0.0
                downside_diff = np.minimum(0.0, clean_returns - (risk_free_rate / 252.0))
                downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252))
                metrics['Downside_Deviation'] = float(downside_dev * 100)
                metrics['Sortino_Ratio'] = float((annual_return * 100) / max(0.001, downside_dev * 100)) if downside_dev != 0 else 0.0
                
                excess_returns = clean_returns
                positive_returns = float(excess_returns[excess_returns > 0].sum())
                negative_returns = float(abs(excess_returns[excess_returns < 0].sum()))
                metrics['Omega_Ratio'] = float(positive_returns / max(1e-6, negative_returns)) if negative_returns != 0 else 10.0
                
                pain_index = float(self._calculate_pain_index(clean_returns))
                metrics['Pain_Ratio'] = float(annual_return / max(0.001, pain_index)) if pain_index != 0 else 0.0
            else:
                metrics['VaR_95'] = None
                metrics['VaR_99'] = None
                metrics['CVaR_95'] = None
                metrics['CVaR_99'] = None
                metrics['Tail_Ratio'] = None
                metrics['Calmar_Ratio'] = None
                metrics['Downside_Deviation'] = None
                metrics['Sortino_Ratio'] = None
                metrics['Omega_Ratio'] = None
                metrics['Pain_Ratio'] = None

            # 4. Cornish-Fisher Expansion for Modified VaR (adjusts for non-normality)
            if cf_available:
                skew = float(np.clip(raw_skew, -3.0, 3.0))
                kurt = float(np.clip(raw_kurt, -1.0, 10.0))
                metrics['Skewness'] = raw_skew
                metrics['Kurtosis'] = raw_kurt

                # z_95 = -1.64485, z_99 = -2.32635 for left tail
                z_95 = -1.644853
                z_99 = -2.326348
                
                cf_z_95 = z_95 + (z_95**2 - 1) * (skew / 6.0) + (z_95**3 - 3 * z_95) * (kurt / 24.0) - (2 * z_95**3 - 5 * z_95) * (skew**2 / 36.0)
                cf_z_99 = z_99 + (z_99**2 - 1) * (skew / 6.0) + (z_99**3 - 3 * z_99) * (kurt / 24.0) - (2 * z_99**3 - 5 * z_99) * (skew**2 / 36.0)
                
                # Enforce monotonicity: 99% VaR tail multiplier must be strictly deeper than 95%
                if cf_z_99 > cf_z_95:
                    cf_z_99 = cf_z_95 - 0.681495  # Fallback to standard Gaussian spread
                
                mean_ret = float(clean_returns.mean())
                
                var_95_val = float((mean_ret + cf_z_95 * std_ret) * 100)
                var_99_val = float((mean_ret + cf_z_99 * std_ret) * 100)
                
                metrics['Modified_VaR_95'] = var_95_val
                metrics['Modified_VaR_99'] = min(var_99_val, var_95_val)  # Guarantee 99% VaR is more conservative
                
                # Modified CVaR using returns exceeding Modified VaR threshold
                mvar_threshold = mean_ret + cf_z_95 * std_ret
                tail_returns = clean_returns[clean_returns <= mvar_threshold]
                metrics['Modified_CVaR_95'] = float(tail_returns.mean() * 100) if len(tail_returns) > 0 else metrics.get('CVaR_95')
            else:
                metrics['Modified_VaR_95'] = None
                metrics['Modified_VaR_99'] = None
                metrics['Modified_CVaR_95'] = None
                metrics['Skewness'] = None
                metrics['Kurtosis'] = None

            # 5. Cornish-Fisher Explicit State Object
            cf_state = {
                'status': 'AVAILABLE' if cf_available else 'UNAVAILABLE',
                'reason': cf_reason,
                'actual_observations': n_obs,
                'required_observations': REQUIRED_CORNISH_FISHER_OBSERVATIONS,
                'message': cf_message,
            }
            metrics['cornish_fisher'] = cf_state

            # 6. Sanitize metrics: Preserves None, converts NaN/Inf to None, preserves dicts
            sanitized_metrics = {}
            for k, v in metrics.items():
                if k == 'cornish_fisher':
                    sanitized_metrics[k] = v
                elif v is None:
                    sanitized_metrics[k] = None
                elif isinstance(v, (float, int, np.floating)):
                    if np.isnan(v) or np.isinf(v):
                        sanitized_metrics[k] = None
                    else:
                        sanitized_metrics[k] = float(v)
                else:
                    sanitized_metrics[k] = v
            return sanitized_metrics
        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {str(e)}")
            return {
                'Modified_VaR_95': None,
                'Modified_VaR_99': None,
                'Modified_CVaR_95': None,
                'cornish_fisher': {
                    'status': 'UNAVAILABLE',
                    'reason': 'CALCULATION_ERROR',
                    'actual_observations': 0,
                    'required_observations': REQUIRED_CORNISH_FISHER_OBSERVATIONS,
                    'message': str(e)
                }
            }
    
    def _analyze_tail_risk(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze tail risk characteristics"""
        try:
            tail_analysis = {}
            
            # Extreme tail events (beyond 2.5 standard deviations)
            std_dev = returns.std()
            extreme_negative = returns[returns < -2.5 * std_dev]
            extreme_positive = returns[returns > 2.5 * std_dev]
            
            tail_analysis['extreme_negative_events'] = len(extreme_negative)
            tail_analysis['extreme_positive_events'] = len(extreme_positive)
            tail_analysis['extreme_negative_avg'] = extreme_negative.mean() * 100 if len(extreme_negative) > 0 else 0
            tail_analysis['extreme_positive_avg'] = extreme_positive.mean() * 100 if len(extreme_positive) > 0 else 0
            
            # Left tail (negative) statistics
            left_tail = returns[returns < 0]
            if len(left_tail) > 0:
                tail_analysis['left_tail_frequency'] = len(left_tail) / len(returns) * 100
                tail_analysis['left_tail_avg_loss'] = left_tail.mean() * 100
                tail_analysis['worst_single_day'] = left_tail.min() * 100
            
            # Right tail (positive) statistics
            right_tail = returns[returns > 0]
            if len(right_tail) > 0:
                tail_analysis['right_tail_frequency'] = len(right_tail) / len(returns) * 100
                tail_analysis['right_tail_avg_gain'] = right_tail.mean() * 100
                tail_analysis['best_single_day'] = right_tail.max() * 100
            
            return tail_analysis
            
        except Exception as e:
            logger.error(f"Error in tail risk analysis: {str(e)}")
            return {}
    
    def _analyze_drawdowns(self, prices: pd.Series) -> Dict[str, Any]:
        """Comprehensive drawdown analysis"""
        try:
            # Calculate drawdowns
            cumulative = (1 + prices.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = None
            
            for idx, dd in drawdowns.items():
                if dd < 0 and not in_drawdown:
                    in_drawdown = True
                    start_idx = idx
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    if start_idx is not None:
                        period_dd = drawdowns[start_idx:idx]
                        drawdown_periods.append({
                            'start': start_idx,
                            'end': idx,
                            'duration_days': (idx - start_idx).days if hasattr(idx - start_idx, 'days') else len(period_dd),
                            'max_drawdown': period_dd.min() * 100,
                            'recovery_date': idx
                        })
            
            analysis = {
                'max_drawdown': drawdowns.min() * 100,
                'avg_drawdown': drawdowns[drawdowns < 0].mean() * 100 if len(drawdowns[drawdowns < 0]) > 0 else 0,
                'drawdown_frequency': len(drawdown_periods),
                'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
                'max_drawdown_duration': max([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
                'current_drawdown': drawdowns.iloc[-1] * 100,
                'time_underwater_pct': len(drawdowns[drawdowns < 0]) / len(drawdowns) * 100
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in drawdown analysis: {str(e)}")
            return {}
    
    def _detect_market_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect different market regimes (Bull, Bear, High Vol, Low Vol)"""
        try:
            from analyst_dashboard.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
            regime_analyzer = MarketRegimeAnalyzer()
            stat_regimes = regime_analyzer._detect_statistical_regimes(returns)
            if 'error' not in stat_regimes:
                return {
                    'current_regime': stat_regimes.get('current_regime', 'Unknown'),
                    'regime_probabilities': stat_regimes.get('regime_probabilities', []),
                    'regime_statistics': stat_regimes.get('regimes', {}),
                    'model_score': stat_regimes.get('model_score', 0.0)
                }
        except Exception as e:
            logger.warning(f"GMM regime detection fallback to threshold rules: {e}")

        try:
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            rolling_returns = returns.rolling(20).mean() * 252 * 100
            vol_threshold = rolling_vol.median()
            return_threshold = 0
            
            regimes = []
            current_regime = None
            
            for i in range(len(rolling_vol)):
                if pd.isna(rolling_vol.iloc[i]) or pd.isna(rolling_returns.iloc[i]):
                    continue
                    
                vol = rolling_vol.iloc[i]
                ret = rolling_returns.iloc[i]
                
                if ret > return_threshold and vol < vol_threshold:
                    regime = "Bull_Market_Low_Vol"
                elif ret > return_threshold and vol >= vol_threshold:
                    regime = "Bull_Market_High_Vol"
                elif ret <= return_threshold and vol < vol_threshold:
                    regime = "Bear_Market_Low_Vol"
                else:
                    regime = "Bear_Market_High_Vol"
                
                if regime != current_regime:
                    regimes.append({
                        'regime': regime,
                        'start_date': rolling_vol.index[i],
                        'volatility': vol,
                        'returns': ret
                    })
                    current_regime = regime
            
            # Calculate regime statistics
            regime_stats = {}
            for regime_type in ["Bull_Market_Low_Vol", "Bull_Market_High_Vol", 
                              "Bear_Market_Low_Vol", "Bear_Market_High_Vol"]:
                regime_periods = [r for r in regimes if r['regime'] == regime_type]
                if regime_periods:
                    regime_stats[regime_type] = {
                        'frequency': len(regime_periods),
                        'avg_volatility': np.mean([r['volatility'] for r in regime_periods]),
                        'avg_returns': np.mean([r['returns'] for r in regime_periods])
                    }
            
            current_regime = regimes[-1]['regime'] if regimes else "Unknown"
            
            return {
                'current_regime': current_regime,
                'regime_history': regimes[-10:],  # Last 10 regime changes
                'regime_statistics': regime_stats,
                'regime_transitions': len(regimes)
            }
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return {}
    
    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """Calculate Pain Index (average drawdown over time)"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            pain_index = abs(drawdowns.mean())
            return pain_index
        except Exception as e:
            return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            return drawdowns.min()
        except Exception as e:
            return 0
    
    def generate_risk_insights(self, risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable risk insights"""
        insights = []
        
        try:
            if 'advanced_metrics' in risk_analysis:
                metrics = risk_analysis['advanced_metrics']
                
                # VaR insights
                var_95 = metrics.get('VaR_95', 0)
                if var_95 < -5:
                    insights.append(f"⚠️ High daily risk: 5% chance of losing more than {abs(var_95):.1f}% in a single day")
                elif var_95 < -2:
                    insights.append(f"⚠️ Moderate daily risk: 5% chance of losing more than {abs(var_95):.1f}% in a single day")
                
                # Tail ratio insights
                tail_ratio = metrics.get('Tail_Ratio', 1)
                if tail_ratio > 2:
                    insights.append("📈 Positive skew: Upside potential exceeds downside risk")
                elif tail_ratio < 0.5:
                    insights.append("📉 Negative skew: Downside risk exceeds upside potential")
                
                # Drawdown insights
                if 'drawdown_analysis' in risk_analysis:
                    dd_analysis = risk_analysis['drawdown_analysis']
                    max_dd = dd_analysis.get('max_drawdown', 0)
                    if max_dd < -20:
                        insights.append(f"🔴 High drawdown risk: Historical maximum loss of {abs(max_dd):.1f}%")
                    elif max_dd < -10:
                        insights.append(f"🟡 Moderate drawdown risk: Historical maximum loss of {abs(max_dd):.1f}%")
                
                # Regime insights
                if 'regime_analysis' in risk_analysis:
                    current_regime = risk_analysis['regime_analysis'].get('current_regime', 'Unknown')
                    if 'High_Vol' in current_regime:
                        insights.append("🌪️ Currently in high volatility regime - expect increased price swings")
                    elif 'Bull_Market' in current_regime:
                        insights.append("🐂 Currently in bullish regime - favorable risk-reward environment")
                    elif 'Bear_Market' in current_regime:
                        insights.append("🐻 Currently in bearish regime - heightened risk awareness needed")
            
        except Exception as e:
            logger.error(f"Error generating risk insights: {str(e)}")
            insights.append("⚠️ Unable to generate risk insights due to data limitations")
        
        return insights
