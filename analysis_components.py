"""
Analysis Components Module - Smaller, focused analysis functions
Breaks down large analysis engine methods into specialized components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
import warnings
import urllib3
import ssl
import requests

logger = logging.getLogger(__name__)

class DataProcessors:
    """Small, focused data processing functions."""
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, price_column: str = 'Close') -> pd.Series:
        """Calculate daily returns from price data."""
        try:
            return data[price_column].pct_change().dropna()
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        try:
            return returns.rolling(window=window).std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {e}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_drawdown(data: pd.DataFrame, price_column: str = 'Close') -> pd.Series:
        """Calculate drawdown series."""
        try:
            price_series = data[price_column]
            rolling_max = price_series.expanding().max()
            drawdown = (price_series - rolling_max) / rolling_max
            return drawdown
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def normalize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize specified columns using StandardScaler."""
        try:
            normalized_data = data.copy()
            scaler = StandardScaler()
            normalized_data[columns] = scaler.fit_transform(data[columns])
            return normalized_data
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data

class TechnicalIndicators:
    """Focused technical indicator calculations."""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = TechnicalIndicators.calculate_sma(data, window)
            std = data.rolling(window=window).std()
            
            return {
                'middle': sma,
                'upper': sma + (std * num_std),
                'lower': sma - (std * num_std)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'middle': pd.Series(), 'upper': pd.Series(), 'lower': pd.Series()}
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        try:
            ema_fast = TechnicalIndicators.calculate_ema(data, fast)
            ema_slow = TechnicalIndicators.calculate_ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

class RiskMetrics:
    """Risk calculation components."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        try:
            if len(returns) == 0:
                return 0.0
            return np.percentile(returns.dropna(), confidence_level * 100)
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            if len(returns) == 0:
                return 0.0
            var = RiskMetrics.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            excess_returns = returns.mean() * 252 - risk_free_rate
            volatility = returns.std() * np.sqrt(252)
            return excess_returns / volatility
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown metrics."""
        try:
            if len(returns) == 0:
                return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
            
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            max_dd = drawdown.min()
            
            # Calculate drawdown duration
            dd_duration = 0
            current_duration = 0
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    dd_duration = max(dd_duration, current_duration)
                else:
                    current_duration = 0
            
            return {
                'max_drawdown': max_dd,
                'max_drawdown_duration': dd_duration
            }
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        try:
            if len(asset_returns) == 0 or len(market_returns) == 0:
                return 1.0
            
            # Align data
            aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 2:
                return 1.0
            
            covariance = aligned_data.cov().iloc[0, 1]
            market_variance = aligned_data.iloc[:, 1].var()
            
            return covariance / market_variance if market_variance != 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0

class MarketDataFetchers:
    """Focused market data retrieval functions."""
    
    @staticmethod
    def fetch_economic_indicators() -> Dict[str, float]:
        """Fetch key economic indicators with improved SSL handling."""
        import warnings
        import urllib3
        
        # Suppress SSL warnings temporarily
        warnings.filterwarnings("ignore", message=".*SSL.*")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        try:
            indicators = {}
            
            # Treasury rates
            try:
                import ssl
                import requests
                
                # Configure SSL context for better compatibility
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # Use requests session with SSL config
                session = requests.Session()
                session.verify = False
                
                tnx = yf.Ticker("^TNX", session=session)
                treasury_data = tnx.history(period="5d")
                if not treasury_data.empty:
                    indicators['treasury_10y'] = treasury_data['Close'].iloc[-1]
                else:
                    indicators['treasury_10y'] = 4.5  # Default fallback
            except Exception as e:
                logger.debug(f"Treasury data fetch failed, using fallback: {e}")
                indicators['treasury_10y'] = 4.5  # Default fallback
            
            # DXY (Dollar Index)
            try:
                session = requests.Session()
                session.verify = False
                
                dxy = yf.Ticker("DX-Y.NYB", session=session)
                dxy_data = dxy.history(period="5d")
                if not dxy_data.empty:
                    indicators['dxy'] = dxy_data['Close'].iloc[-1]
                else:
                    indicators['dxy'] = 103.0  # Default fallback
            except Exception as e:
                logger.debug(f"DXY data fetch failed, using fallback: {e}")
                indicators['dxy'] = 103.0  # Default fallback
            
            # VIX
            try:
                session = requests.Session()
                session.verify = False
                
                vix = yf.Ticker("^VIX", session=session)
                vix_data = vix.history(period="5d")
                if not vix_data.empty:
                    indicators['vix'] = vix_data['Close'].iloc[-1]
                else:
                    indicators['vix'] = 20.0  # Default fallback
            except Exception as e:
                logger.debug(f"VIX data fetch failed, using fallback: {e}")
                indicators['vix'] = 20.0  # Default fallback
            
            logger.info(f"Economic indicators loaded: {indicators}")
            return indicators
            
        except Exception as e:
            logger.debug(f"Error fetching economic indicators: {e}")
            return {'treasury_10y': 4.5, 'dxy': 103.0, 'vix': 20.0}
        finally:
            # Re-enable warnings
            warnings.resetwarnings()
    
    @staticmethod
    def fetch_market_correlations(symbol: str, period: str = "1y") -> Dict[str, float]:
        """Fetch correlations with major market indices."""
        try:
            correlations = {}
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'IWM': 'Russell 2000',
                'GLD': 'Gold',
                'TLT': 'Long Bonds'
            }
            
            # Get asset data
            asset_ticker = yf.Ticker(symbol)
            asset_data = asset_ticker.history(period=period)
            
            if asset_data.empty:
                return correlations
            
            asset_returns = asset_data['Close'].pct_change().dropna()
            
            for ticker, name in indices.items():
                try:
                    index_ticker = yf.Ticker(ticker)
                    index_data = index_ticker.history(period=period)
                    
                    if not index_data.empty:
                        index_returns = index_data['Close'].pct_change().dropna()
                        
                        # Align data and calculate correlation
                        aligned_data = pd.concat([asset_returns, index_returns], axis=1).dropna()
                        if len(aligned_data) > 20:  # Need sufficient data points
                            correlation = aligned_data.corr().iloc[0, 1]
                            correlations[name.lower().replace(' ', '_')] = correlation
                
                except Exception as idx_error:
                    logger.warning(f"Could not fetch data for {ticker}: {idx_error}")
                    correlations[name.lower().replace(' ', '_')] = 0.0
            
            return correlations
        except Exception as e:
            logger.error(f"Error fetching market correlations: {e}")
            return {}

class SeasonalityAnalyzers:
    """Focused seasonality analysis components."""
    
    @staticmethod
    def analyze_monthly_patterns(data: pd.DataFrame, price_column: str = 'Close') -> Dict[str, Any]:
        """Analyze monthly seasonal patterns."""
        try:
            if data.empty:
                return {}
            
            returns = data[price_column].pct_change().dropna()
            
            # Group by month
            monthly_returns = returns.groupby(returns.index.month).agg({
                'mean': 'mean',
                'std': 'std',
                'count': 'count'
            })
            
            # Find best and worst months
            best_month = monthly_returns['mean'].idxmax()
            worst_month = monthly_returns['mean'].idxmin()
            
            # Statistical significance test
            monthly_significance = {}
            for month in range(1, 13):
                month_data = returns[returns.index.month == month]
                if len(month_data) > 5:
                    t_stat, p_value = stats.ttest_1samp(month_data, 0)
                    monthly_significance[month] = {
                        'mean_return': month_data.mean(),
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            return {
                'monthly_returns': monthly_returns.to_dict(),
                'best_month': best_month,
                'worst_month': worst_month,
                'statistical_significance': monthly_significance
            }
        except Exception as e:
            logger.error(f"Error analyzing monthly patterns: {e}")
            return {}
    
    @staticmethod
    def analyze_day_of_week_effects(data: pd.DataFrame, price_column: str = 'Close') -> Dict[str, Any]:
        """Analyze day-of-week effects."""
        try:
            if data.empty:
                return {}
            
            returns = data[price_column].pct_change().dropna()
            
            # Group by day of week
            dow_returns = returns.groupby(returns.index.day_of_week).agg({
                'mean': 'mean',
                'std': 'std',
                'count': 'count'
            })
            
            # Map to day names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_named = {}
            
            for dow, name in enumerate(day_names):
                if dow in dow_returns.index:
                    dow_named[name] = {
                        'mean_return': dow_returns.loc[dow, 'mean'],
                        'std_return': dow_returns.loc[dow, 'std'],
                        'count': dow_returns.loc[dow, 'count']
                    }
            
            # Find best and worst days
            if dow_returns['mean'].empty:
                return {}
            
            best_day_idx = dow_returns['mean'].idxmax()
            worst_day_idx = dow_returns['mean'].idxmin()
            
            best_day = day_names[best_day_idx] if best_day_idx < len(day_names) else 'Unknown'
            worst_day = day_names[worst_day_idx] if worst_day_idx < len(day_names) else 'Unknown'
            
            return {
                'day_of_week_returns': dow_named,
                'best_day': best_day,
                'worst_day': worst_day
            }
        except Exception as e:
            logger.error(f"Error analyzing day-of-week effects: {e}")
            return {}

class VolatilityModels:
    """Volatility modeling components."""
    
    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate realized volatility."""
        try:
            return returns.rolling(window=window).std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def estimate_garch_volatility(returns: pd.Series) -> Dict[str, Any]:
        """Simple GARCH-like volatility estimation."""
        try:
            if len(returns) < 50:
                return {'forecast': returns.std() * np.sqrt(252)}
            
            # Simple exponentially weighted volatility
            lambda_param = 0.94
            
            # Initialize with sample variance
            var_forecast = returns.var()
            forecasts = []
            
            for ret in returns:
                var_forecast = lambda_param * var_forecast + (1 - lambda_param) * (ret ** 2)
                forecasts.append(np.sqrt(var_forecast * 252))
            
            return {
                'forecast': forecasts[-1] if forecasts else returns.std() * np.sqrt(252),
                'series': forecasts
            }
        except Exception as e:
            logger.error(f"Error estimating GARCH volatility: {e}")
            return {'forecast': 0.20}  # Default 20% annual volatility
    
    @staticmethod
    def identify_volatility_regime(returns: pd.Series, threshold: float = 0.02) -> str:
        """Identify current volatility regime."""
        try:
            if len(returns) < 20:
                return 'Unknown'
            
            recent_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            if recent_vol > historical_vol * (1 + threshold):
                return 'High Volatility'
            elif recent_vol < historical_vol * (1 - threshold):
                return 'Low Volatility'
            else:
                return 'Normal Volatility'
        except Exception as e:
            logger.error(f"Error identifying volatility regime: {e}")
            return 'Unknown'

class SimpleForecasting:
    """Simple forecasting models."""
    
    @staticmethod
    def moving_average_forecast(data: pd.Series, window: int = 20, periods: int = 30) -> np.ndarray:
        """Simple moving average forecast."""
        try:
            if len(data) < window:
                return np.full(periods, data.iloc[-1] if not data.empty else 0)
            
            ma = data.rolling(window=window).mean().iloc[-1]
            return np.full(periods, ma)
        except Exception as e:
            logger.error(f"Error in moving average forecast: {e}")
            return np.zeros(periods)
    
    @staticmethod
    def linear_trend_forecast(data: pd.Series, periods: int = 30) -> np.ndarray:
        """Simple linear trend forecast."""
        try:
            if len(data) < 10:
                return np.full(periods, data.iloc[-1] if not data.empty else 0)
            
            # Fit linear trend to recent data
            recent_data = data.tail(min(60, len(data)))
            x = np.arange(len(recent_data))
            y = recent_data.values
            
            # Linear regression
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            # Forecast
            forecast_x = np.arange(len(recent_data), len(recent_data) + periods)
            forecast = slope * forecast_x + intercept
            
            return forecast
        except Exception as e:
            logger.error(f"Error in linear trend forecast: {e}")
            return np.zeros(periods)
    
    @staticmethod
    def random_walk_forecast(data: pd.Series, periods: int = 30) -> np.ndarray:
        """Random walk (last value) forecast."""
        try:
            if data.empty:
                return np.zeros(periods)
            
            last_value = data.iloc[-1]
            return np.full(periods, last_value)
        except Exception as e:
            logger.error(f"Error in random walk forecast: {e}")
            return np.zeros(periods)

# Export all components
__all__ = [
    'DataProcessors',
    'TechnicalIndicators', 
    'RiskMetrics',
    'MarketDataFetchers',
    'SeasonalityAnalyzers',
    'VolatilityModels',
    'SimpleForecasting'
]
