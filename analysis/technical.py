"""
Technical analysis indicators and calculations.
Implements common technical indicators with proper mathematical formulations.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import logging

from config import technical_config
from data.cache import cache_result

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Main class for technical analysis calculations."""
    
    @staticmethod
    @cache_result(ttl=300)  # Cache for 5 minutes
    def calculate_sma(
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series
            window: Period for SMA calculation
        
        Returns:
            pd.Series: SMA values
        """
        return prices.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_ema(
        prices: pd.Series,
        window: int = 20,
        alpha: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            window: Period for EMA calculation
            alpha: Smoothing factor (if None, calculated as 2/(window+1))
        
        Returns:
            pd.Series: EMA values
        """
        if alpha is None:
            alpha = 2.0 / (window + 1)
        
        return prices.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_rsi(
        prices: pd.Series,
        window: int = technical_config.RSI_PERIOD
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            window: Period for RSI calculation
        
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_macd(
        prices: pd.Series,
        fast_window: int = technical_config.MACD_FAST,
        slow_window: int = technical_config.MACD_SLOW,
        signal_window: int = technical_config.MACD_SIGNAL
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_window: Fast EMA period
            slow_window: Slow EMA period
            signal_window: Signal line EMA period
        
        Returns:
            Dict containing MACD line, signal line, and histogram
        """
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast_window)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow_window)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_window)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_bollinger_bands(
        prices: pd.Series,
        window: int = technical_config.BOLLINGER_PERIOD,
        num_std: float = technical_config.BOLLINGER_STD
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Period for calculation
            num_std: Number of standard deviations
        
        Returns:
            Dict containing upper band, middle band (SMA), and lower band
        """
        sma = TechnicalIndicators.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = technical_config.STOCH_K_PERIOD,
        d_window: int = technical_config.STOCH_D_PERIOD
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K period
            d_window: %D smoothing period
        
        Returns:
            Dict containing %K and %D lines
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = technical_config.WILLIAMS_R_PERIOD
    ) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for calculation
        
        Returns:
            pd.Series: Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for ATR calculation
        
        Returns:
            pd.Series: ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    @cache_result(ttl=300)
    def calculate_obv(
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
        
        Returns:
            pd.Series: OBV values
        """
        price_change = close.diff()
        
        # Determine volume direction
        volume_direction = pd.Series(index=close.index, dtype=float)
        volume_direction[price_change > 0] = 1
        volume_direction[price_change < 0] = -1
        volume_direction[price_change == 0] = 0
        
        # Calculate OBV
        obv = (volume * volume_direction).cumsum()
        
        return obv

class PatternRecognition:
    """Pattern recognition for technical analysis."""
    
    @staticmethod
    def identify_support_resistance(
        prices: pd.Series,
        window: int = 20,
        min_touches: int = 2
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            prices: Price series
            window: Window for local extrema detection
            min_touches: Minimum number of touches to confirm level
        
        Returns:
            Dict containing support and resistance levels
        """
        # Find local minima (potential support)
        local_mins = []
        local_maxs = []
        
        for i in range(window, len(prices) - window):
            # Check if current price is local minimum
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                local_mins.append(prices.iloc[i])
            
            # Check if current price is local maximum
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                local_maxs.append(prices.iloc[i])
        
        # Cluster similar levels
        support_levels = PatternRecognition._cluster_levels(local_mins, min_touches)
        resistance_levels = PatternRecognition._cluster_levels(local_maxs, min_touches)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], min_touches: int, tolerance: float = 0.02) -> List[float]:
        """
        Cluster price levels that are within tolerance of each other.
        
        Args:
            levels: List of price levels
            min_touches: Minimum touches to confirm level
            tolerance: Percentage tolerance for clustering
        
        Returns:
            List of confirmed levels
        """
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # Check if level is within tolerance of current cluster
            cluster_center = np.mean(current_cluster)
            if abs(level - cluster_center) / cluster_center <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster if it has enough touches
                if len(current_cluster) >= min_touches:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_touches:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    @staticmethod
    def detect_trend(
        prices: pd.Series,
        window: int = 50
    ) -> Dict[str, Union[str, float]]:
        """
        Detect overall trend using linear regression.
        
        Args:
            prices: Price series
            window: Window for trend analysis
        
        Returns:
            Dict containing trend direction, slope, and R-squared
        """
        if len(prices) < window:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
        
        # Use last 'window' periods
        recent_prices = prices.tail(window)
        x = np.arange(len(recent_prices))
        y = recent_prices.values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if slope > 0 and r_value**2 > 0.5:
            trend = 'uptrend'
        elif slope < 0 and r_value**2 > 0.5:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }

class TechnicalAnalysis:
    """Main technical analysis class combining indicators and patterns."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
    
    def full_analysis(
        self,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, Union[pd.Series, Dict, List]]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
        
        Returns:
            Dict containing all technical indicators and analysis
        """
        if ohlcv_data.empty:
            return {}
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {}
        
        close = ohlcv_data['Close']
        high = ohlcv_data['High']
        low = ohlcv_data['Low']
        volume = ohlcv_data['Volume']
        
        analysis_results = {}
        
        try:
            # Moving averages
            analysis_results['sma_20'] = self.indicators.calculate_sma(close, 20)
            analysis_results['sma_50'] = self.indicators.calculate_sma(close, 50)
            analysis_results['sma_200'] = self.indicators.calculate_sma(close, 200)
            analysis_results['ema_12'] = self.indicators.calculate_ema(close, 12)
            analysis_results['ema_26'] = self.indicators.calculate_ema(close, 26)
            
            # Momentum indicators
            analysis_results['rsi'] = self.indicators.calculate_rsi(close)
            analysis_results['macd'] = self.indicators.calculate_macd(close)
            analysis_results['stochastic'] = self.indicators.calculate_stochastic(high, low, close)
            analysis_results['williams_r'] = self.indicators.calculate_williams_r(high, low, close)
            
            # Volatility indicators
            analysis_results['bollinger_bands'] = self.indicators.calculate_bollinger_bands(close)
            analysis_results['atr'] = self.indicators.calculate_atr(high, low, close)
            
            # Volume indicators
            analysis_results['obv'] = self.indicators.calculate_obv(close, volume)
            
            # Pattern recognition
            analysis_results['support_resistance'] = self.patterns.identify_support_resistance(close)
            analysis_results['trend_analysis'] = self.patterns.detect_trend(close)
            
            logger.info("Technical analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def generate_signals(
        self,
        analysis_results: Dict
    ) -> Dict[str, str]:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            analysis_results: Results from full_analysis
        
        Returns:
            Dict containing trading signals and reasoning
        """
        signals = {}
        
        if not analysis_results or 'error' in analysis_results:
            return {'signal': 'no_data', 'reason': 'Insufficient data for analysis'}
        
        try:
            # RSI signals
            if 'rsi' in analysis_results:
                current_rsi = analysis_results['rsi'].iloc[-1]
                if current_rsi > 70:
                    signals['rsi'] = 'oversold'
                elif current_rsi < 30:
                    signals['rsi'] = 'overbought'
                else:
                    signals['rsi'] = 'neutral'
            
            # MACD signals
            if 'macd' in analysis_results:
                macd_data = analysis_results['macd']
                if len(macd_data['macd']) >= 2:
                    current_macd = macd_data['macd'].iloc[-1]
                    current_signal = macd_data['signal'].iloc[-1]
                    prev_macd = macd_data['macd'].iloc[-2]
                    prev_signal = macd_data['signal'].iloc[-2]
                    
                    if current_macd > current_signal and prev_macd <= prev_signal:
                        signals['macd'] = 'bullish_crossover'
                    elif current_macd < current_signal and prev_macd >= prev_signal:
                        signals['macd'] = 'bearish_crossover'
                    else:
                        signals['macd'] = 'neutral'
            
            # Trend signals
            if 'trend_analysis' in analysis_results:
                trend_data = analysis_results['trend_analysis']
                signals['trend'] = trend_data.get('trend', 'unknown')
            
            # Overall signal
            bullish_signals = sum(1 for signal in signals.values() 
                                if signal in ['oversold', 'bullish_crossover', 'uptrend'])
            bearish_signals = sum(1 for signal in signals.values() 
                                if signal in ['overbought', 'bearish_crossover', 'downtrend'])
            
            if bullish_signals > bearish_signals:
                signals['overall'] = 'bullish'
            elif bearish_signals > bullish_signals:
                signals['overall'] = 'bearish'
            else:
                signals['overall'] = 'neutral'
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            signals['error'] = str(e)
        
        return signals

# Create instances for easy importing
technical_indicators = TechnicalIndicators()
pattern_recognition = PatternRecognition()
technical_analysis = TechnicalAnalysis()
