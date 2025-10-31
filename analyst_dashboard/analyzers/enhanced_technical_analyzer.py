"""
Enhanced Technical Analyzer - Implementation Example
Adding advanced indicators to your existing modular architecture
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class EnhancedTechnicalAnalyzer:
    """Enhanced version of your existing TechnicalAnalysisProcessor"""
    
    def calculate_advanced_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced indicators to your existing ones"""
        tech_data = price_data.copy()
        
        # Your existing indicators first
        tech_data = self._calculate_basic_indicators(tech_data)
        
        # NEW: Advanced momentum indicators
        tech_data['Stoch_RSI_K'], tech_data['Stoch_RSI_D'] = self._calculate_stochastic_rsi(
            tech_data['Close']
        )
        tech_data['Williams_R'] = self._calculate_williams_r(
            tech_data['High'], tech_data['Low'], tech_data['Close']
        )
        tech_data['CCI'] = self._calculate_cci(
            tech_data['High'], tech_data['Low'], tech_data['Close']
        )
        
        # NEW: Trend strength indicators
        tech_data['ADX'] = self._calculate_adx(
            tech_data['High'], tech_data['Low'], tech_data['Close']
        )
        
        # NEW: Volume indicators
        tech_data['MFI'] = self._calculate_money_flow_index(
            tech_data['High'], tech_data['Low'], tech_data['Close'], tech_data['Volume']
        )
        
        # NEW: Volatility indicators
        keltner = self._calculate_keltner_channels(
            tech_data['High'], tech_data['Low'], tech_data['Close']
        )
        tech_data['Keltner_Upper'] = keltner['upper']
        tech_data['Keltner_Lower'] = keltner['lower']
        tech_data['Keltner_Middle'] = keltner['middle']
        
        return tech_data
    
    def generate_advanced_signals(self, tech_data: pd.DataFrame) -> Dict[str, str]:
        """Enhanced signal generation with multi-indicator confluence"""
        signals = {}
        
        # Your existing signals first
        signals.update(self._generate_basic_signals(tech_data))
        
        # NEW: Advanced signal combinations
        signals['Trend_Strength'] = self._analyze_trend_strength(tech_data)
        signals['Volume_Confirmation'] = self._analyze_volume_confirmation(tech_data)
        signals['Volatility_Regime'] = self._analyze_volatility_regime(tech_data)
        signals['Multi_Timeframe_Alignment'] = self._check_timeframe_alignment(tech_data)
        
        # NEW: Confluence score (0-100)
        signals['Signal_Confluence'] = self._calculate_signal_confluence(signals)
        
        return signals
    
    def _calculate_stochastic_rsi(self, prices: pd.Series, rsi_period: int = 14, 
                                 stoch_period: int = 14) -> tuple:
        """Calculate Stochastic RSI - more sensitive than regular RSI"""
        # Calculate RSI first
        rsi = self._calculate_rsi(prices, rsi_period)
        
        # Apply Stochastic formula to RSI
        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        
        stoch_rsi_k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
        stoch_rsi_d = stoch_rsi_k.rolling(3).mean()
        
        return stoch_rsi_k, stoch_rsi_d
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        
        # Mean deviation
        mad = typical_price.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (trend strength)"""
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = (high - high.shift()).where(
            (high - high.shift()) > (low.shift() - low), 0
        ).where((high - high.shift()) > 0, 0)
        
        dm_minus = (low.shift() - low).where(
            (low.shift() - low) > (high - high.shift()), 0
        ).where((low.shift() - low) > 0, 0)
        
        # Smooth the values
        atr = tr.rolling(period).mean()
        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
        
        # ADX calculation
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _analyze_trend_strength(self, tech_data: pd.DataFrame) -> str:
        """Analyze overall trend strength using multiple indicators"""
        try:
            adx = tech_data['ADX'].iloc[-1]
            
            if adx > 40:
                return "Very Strong Trend"
            elif adx > 25:
                return "Strong Trend"
            elif adx > 15:
                return "Moderate Trend"
            else:
                return "Weak/Sideways"
        except:
            return "Unknown"
    
    def _analyze_volume_confirmation(self, tech_data: pd.DataFrame) -> str:
        """Analyze if volume confirms price action"""
        try:
            # Compare recent volume to average
            recent_volume = tech_data['Volume'].iloc[-5:].mean()
            avg_volume = tech_data['Volume'].iloc[-50:].mean()
            
            # Compare price change to volume
            price_change_pct = (tech_data['Close'].iloc[-1] / tech_data['Close'].iloc[-2] - 1) * 100
            volume_ratio = recent_volume / avg_volume
            
            if abs(price_change_pct) > 2 and volume_ratio > 1.5:
                return "Strong Volume Confirmation"
            elif volume_ratio > 1.2:
                return "Volume Confirmation"
            elif volume_ratio < 0.8:
                return "Volume Divergence Warning"
            else:
                return "Normal Volume"
        except:
            return "Unknown"
    
    def _calculate_signal_confluence(self, signals: Dict[str, str]) -> int:
        """Calculate signal confluence score (0-100)"""
        bullish_signals = [
            'Bullish', 'Strong Bullish', 'Oversold', 'Below Lower Band',
            'Bullish Crossover', 'Strong Trend', 'Strong Volume Confirmation'
        ]
        
        bearish_signals = [
            'Bearish', 'Strong Bearish', 'Overbought', 'Above Upper Band',
            'Bearish Crossover'
        ]
        
        total_signals = 0
        bullish_count = 0
        
        for signal_value in signals.values():
            if isinstance(signal_value, str):
                total_signals += 1
                if any(bullish in signal_value for bullish in bullish_signals):
                    bullish_count += 1
        
        if total_signals == 0:
            return 50  # Neutral
        
        confluence_score = int((bullish_count / total_signals) * 100)
        return confluence_score
