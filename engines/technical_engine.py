"""
Technical Analysis Engine - Focused on technical indicators and chart patterns
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from analysis_components import TechnicalIndicators, DataProcessors

logger = logging.getLogger(__name__)

class TechnicalAnalysisEngine:
    """Lightweight technical analysis engine using modular components."""
    
    def __init__(self):
        """Initialize the technical analysis engine."""
        self.indicators = TechnicalIndicators()
        self.processors = DataProcessors()
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information (optional)
        
        Returns:
            Dictionary with technical analysis results
        """
        if price_data.empty:
            return {'error': 'No price data available'}
        
        try:
            results = {}
            
            # Calculate all technical indicators
            results['indicators'] = self._calculate_all_indicators(price_data)
            
            # Generate trading signals
            results['signals'] = self._generate_signals(results['indicators'], price_data)
            
            # Analyze trend structure
            results['trend_analysis'] = self._analyze_trends(price_data, results['indicators'])
            
            # Find support and resistance levels
            results['support_resistance'] = self._find_support_resistance(price_data)
            
            # Pattern recognition
            results['patterns'] = self._identify_patterns(price_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_all_indicators(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators."""
        indicators = {}
        close = price_data['Close']
        
        try:
            # Moving averages
            indicators['SMA_20'] = self.indicators.calculate_sma(close, 20)
            indicators['SMA_50'] = self.indicators.calculate_sma(close, 50)
            indicators['SMA_200'] = self.indicators.calculate_sma(close, 200)
            indicators['EMA_12'] = self.indicators.calculate_ema(close, 12)
            indicators['EMA_26'] = self.indicators.calculate_ema(close, 26)
            
            # Momentum indicators
            indicators['RSI'] = self.indicators.calculate_rsi(close)
            
            # Bollinger Bands
            bb_bands = self.indicators.calculate_bollinger_bands(close)
            indicators.update({
                'BB_Upper': bb_bands['upper'],
                'BB_Middle': bb_bands['middle'],
                'BB_Lower': bb_bands['lower']
            })
            
            # MACD
            macd_data = self.indicators.calculate_macd(close)
            indicators.update({
                'MACD': macd_data['macd'],
                'MACD_Signal': macd_data['signal'],
                'MACD_Histogram': macd_data['histogram']
            })
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _generate_signals(self, indicators: Dict[str, pd.Series], price_data: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals from indicators."""
        signals = {}
        
        try:
            # RSI signals
            if 'RSI' in indicators and not indicators['RSI'].empty:
                current_rsi = indicators['RSI'].iloc[-1]
                if current_rsi > 70:
                    signals['RSI'] = 'Overbought'
                elif current_rsi < 30:
                    signals['RSI'] = 'Oversold'
                else:
                    signals['RSI'] = 'Neutral'
            
            # MACD signals
            if all(k in indicators for k in ['MACD', 'MACD_Signal']):
                macd_current = indicators['MACD'].iloc[-1]
                signal_current = indicators['MACD_Signal'].iloc[-1]
                
                if macd_current > signal_current:
                    signals['MACD'] = 'Bullish'
                else:
                    signals['MACD'] = 'Bearish'
            
            # Moving average signals
            if all(k in indicators for k in ['SMA_50', 'SMA_200']):
                sma50 = indicators['SMA_50'].iloc[-1]
                sma200 = indicators['SMA_200'].iloc[-1]
                
                if sma50 > sma200:
                    signals['MA_Trend'] = 'Bullish'
                else:
                    signals['MA_Trend'] = 'Bearish'
            
            # Bollinger Band signals
            if all(k in indicators for k in ['BB_Upper', 'BB_Lower']):
                current_price = price_data['Close'].iloc[-1]
                bb_upper = indicators['BB_Upper'].iloc[-1]
                bb_lower = indicators['BB_Lower'].iloc[-1]
                
                if current_price > bb_upper:
                    signals['Bollinger'] = 'Overbought'
                elif current_price < bb_lower:
                    signals['Bollinger'] = 'Oversold'
                else:
                    signals['Bollinger'] = 'Normal'
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _analyze_trends(self, price_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze trend structure and strength."""
        try:
            close = price_data['Close']
            
            # Calculate trend using linear regression
            x = np.arange(len(close))
            slope, intercept = np.polyfit(x[-50:], close.iloc[-50:], 1)
            
            # Determine trend direction
            if slope > 0.1:
                primary_trend = 'Strong Uptrend'
                trend_strength = 'Strong'
            elif slope > 0.02:
                primary_trend = 'Uptrend'
                trend_strength = 'Moderate'
            elif slope > -0.02:
                primary_trend = 'Sideways'
                trend_strength = 'Weak'
            elif slope > -0.1:
                primary_trend = 'Downtrend'
                trend_strength = 'Moderate'
            else:
                primary_trend = 'Strong Downtrend'
                trend_strength = 'Strong'
            
            # Calculate trend consistency
            returns = close.pct_change().dropna()
            trend_consistency = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.5
            
            return {
                'primary_trend': primary_trend,
                'trend_strength': trend_strength,
                'trend_slope': slope,
                'trend_consistency': trend_consistency,
                'trend_quality': 'High' if trend_consistency > 0.6 or trend_consistency < 0.4 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'primary_trend': 'Unknown', 'trend_strength': 'Unknown'}
    
    def _find_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find key support and resistance levels."""
        try:
            highs = price_data['High']
            lows = price_data['Low']
            
            # Find local peaks and troughs
            window = 10
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(highs) - window):
                # Check for resistance (local high)
                if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                    resistance_levels.append(highs.iloc[i])
                
                # Check for support (local low)
                if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                    support_levels.append(lows.iloc[i])
            
            # Sort and take most significant levels
            resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
            support_levels = sorted(set(support_levels))[:5]
            
            return {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {'resistance_levels': [], 'support_levels': []}
    
    def _identify_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify chart patterns."""
        try:
            patterns = []
            
            # Simple pattern recognition
            close = price_data['Close']
            highs = price_data['High']
            lows = price_data['Low']
            
            # Double top pattern (simplified)
            if len(highs) >= 20:
                recent_highs = highs.iloc[-20:]
                max_high = recent_highs.max()
                high_indices = recent_highs[recent_highs > max_high * 0.98].index
                
                if len(high_indices) >= 2:
                    patterns.append('Potential Double Top')
            
            # Ascending triangle (simplified)
            if len(close) >= 20:
                recent_lows = lows.iloc[-20:]
                if recent_lows.is_monotonic_increasing:
                    patterns.append('Ascending Triangle')
            
            return {
                'identified_patterns': patterns,
                'pattern_confidence': 'Medium' if patterns else 'Low',
                'pattern_count': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return {'identified_patterns': [], 'pattern_confidence': 'Low', 'pattern_count': 0}


# Global instance
technical_engine = TechnicalAnalysisEngine()
