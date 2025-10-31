"""
Technical Analysis Processor - Handles technical indicator calculations and analysis
Focused on technical analysis computations and signal generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class TechnicalAnalysisProcessor:
    """Processes technical analysis indicators and generates trading signals."""
    
    def analyze_technical_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis."""
        try:
            if price_data.empty:
                return {'error': 'No price data available'}
            
            # Calculate all technical indicators
            tech_data = self.calculate_technical_indicators(price_data)
            
            # Generate trading signals
            signals = self.generate_trading_signals(tech_data)
            
            # Analyze trends
            trend_analysis = self.analyze_trends(tech_data)
            
            return {
                'indicators': tech_data,
                'signals': signals,
                'trend_analysis': trend_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {'error': str(e)}
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators including Priority 1 enhancements."""
        tech_data = price_data.copy()
        
        # Basic Moving averages
        tech_data['SMA_50'] = tech_data['Close'].rolling(50).mean()
        tech_data['SMA_200'] = tech_data['Close'].rolling(200).mean()
        tech_data['EMA_12'] = tech_data['Close'].ewm(span=12).mean()
        tech_data['EMA_26'] = tech_data['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        tech_data['BB_Middle'] = tech_data['Close'].rolling(bb_period).mean()
        bb_std_dev = tech_data['Close'].rolling(bb_period).std()
        tech_data['BB_Upper'] = tech_data['BB_Middle'] + (bb_std_dev * bb_std)
        tech_data['BB_Lower'] = tech_data['BB_Middle'] - (bb_std_dev * bb_std)
        
        # Basic RSI
        tech_data['RSI'] = self._calculate_rsi(tech_data['Close'])
        
        # MACD
        tech_data['MACD'] = tech_data['EMA_12'] - tech_data['EMA_26']
        tech_data['MACD_Signal'] = tech_data['MACD'].ewm(span=9).mean()
        tech_data['MACD_Histogram'] = tech_data['MACD'] - tech_data['MACD_Signal']
        
        # Stochastic Oscillator
        stoch_data = self._calculate_stochastic(tech_data)
        tech_data['Stoch_K'] = stoch_data['%K']
        tech_data['Stoch_D'] = stoch_data['%D']
        
        # Average True Range (ATR)
        tech_data['ATR'] = self._calculate_atr(tech_data)
        
        # PRIORITY 1 ENHANCEMENTS - Advanced Technical Indicators
        
        # 1. Stochastic RSI (Advanced Momentum)
        tech_data['Stoch_RSI'] = self._calculate_stochastic_rsi(tech_data['RSI'])
        
        # 2. Williams %R (Advanced Momentum)
        tech_data['Williams_R'] = self._calculate_williams_r(tech_data)
        
        # 3. Commodity Channel Index (CCI) - Advanced Momentum
        tech_data['CCI'] = self._calculate_cci(tech_data)
        
        # 4. Average Directional Index (ADX) - Trend Strength
        adx_data = self._calculate_adx(tech_data)
        tech_data['ADX'] = adx_data['ADX']
        tech_data['DI_Plus'] = adx_data['DI_Plus']
        tech_data['DI_Minus'] = adx_data['DI_Minus']
        
        # 5. Money Flow Index (MFI) - Volume Confirmation
        tech_data['MFI'] = self._calculate_mfi(tech_data)
        
        # 6. Chaikin Money Flow (CMF) - Volume Confirmation
        tech_data['CMF'] = self._calculate_chaikin_money_flow(tech_data)
        
        # 7. Rate of Change (ROC) - Momentum
        tech_data['ROC'] = self._calculate_roc(tech_data['Close'])
        
        # 8. Parabolic SAR - Trend Following
        tech_data['PSAR'] = self._calculate_parabolic_sar(tech_data)
        
        return tech_data
    
    def generate_trading_signals(self, tech_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading signals with confluence scoring."""
        signals = {}
        signal_scores = {}
        
        try:
            current_price = tech_data['Close'].iloc[-1]
            
            # Basic RSI Signal
            current_rsi = tech_data['RSI'].iloc[-1]
            if current_rsi > 70:
                signals['RSI'] = 'Overbought'
                signal_scores['RSI'] = -30  # Bearish
            elif current_rsi < 30:
                signals['RSI'] = 'Oversold'
                signal_scores['RSI'] = 30   # Bullish
            else:
                signals['RSI'] = 'Neutral'
                signal_scores['RSI'] = 0    # Neutral
            
            # PRIORITY 1 ENHANCEMENT - Advanced Signal Generation with Confluence
            
            # Stochastic RSI Signal (More sensitive)
            stoch_rsi = tech_data['Stoch_RSI'].iloc[-1]
            if stoch_rsi > 80:
                signals['Stoch_RSI'] = 'Overbought'
                signal_scores['Stoch_RSI'] = -25
            elif stoch_rsi < 20:
                signals['Stoch_RSI'] = 'Oversold'
                signal_scores['Stoch_RSI'] = 25
            else:
                signals['Stoch_RSI'] = 'Neutral'
                signal_scores['Stoch_RSI'] = 0
            
            # Williams %R Signal
            williams_r = tech_data['Williams_R'].iloc[-1]
            if williams_r > -20:
                signals['Williams_R'] = 'Overbought'
                signal_scores['Williams_R'] = -20
            elif williams_r < -80:
                signals['Williams_R'] = 'Oversold'
                signal_scores['Williams_R'] = 20
            else:
                signals['Williams_R'] = 'Neutral'
                signal_scores['Williams_R'] = 0
            
            # CCI Signal
            cci = tech_data['CCI'].iloc[-1]
            if cci > 100:
                signals['CCI'] = 'Overbought'
                signal_scores['CCI'] = -20
            elif cci < -100:
                signals['CCI'] = 'Oversold'
                signal_scores['CCI'] = 20
            else:
                signals['CCI'] = 'Neutral'
                signal_scores['CCI'] = 0
            
            # ADX Trend Strength Signal
            adx = tech_data['ADX'].iloc[-1]
            di_plus = tech_data['DI_Plus'].iloc[-1]
            di_minus = tech_data['DI_Minus'].iloc[-1]
            
            if adx > 25:  # Strong trend
                if di_plus > di_minus:
                    signals['ADX'] = 'Strong Uptrend'
                    signal_scores['ADX'] = 25
                else:
                    signals['ADX'] = 'Strong Downtrend'
                    signal_scores['ADX'] = -25
            else:
                signals['ADX'] = 'Weak Trend'
                signal_scores['ADX'] = 0
            
            # Money Flow Index Signal
            mfi = tech_data['MFI'].iloc[-1]
            if mfi > 80:
                signals['MFI'] = 'Overbought'
                signal_scores['MFI'] = -15
            elif mfi < 20:
                signals['MFI'] = 'Oversold'
                signal_scores['MFI'] = 15
            else:
                signals['MFI'] = 'Neutral'
                signal_scores['MFI'] = 0
            
            # Chaikin Money Flow Signal
            cmf = tech_data['CMF'].iloc[-1]
            if cmf > 0.1:
                signals['CMF'] = 'Buying Pressure'
                signal_scores['CMF'] = 15
            elif cmf < -0.1:
                signals['CMF'] = 'Selling Pressure'
                signal_scores['CMF'] = -15
            else:
                signals['CMF'] = 'Neutral'
                signal_scores['CMF'] = 0
            
            # MACD Signal (Enhanced)
            current_macd = tech_data['MACD'].iloc[-1]
            current_signal = tech_data['MACD_Signal'].iloc[-1]
            prev_macd = tech_data['MACD'].iloc[-2] if len(tech_data) > 1 else current_macd
            prev_signal = tech_data['MACD_Signal'].iloc[-2] if len(tech_data) > 1 else current_signal
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                signals['MACD'] = 'Bullish Crossover'
                signal_scores['MACD'] = 35
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signals['MACD'] = 'Bearish Crossover'
                signal_scores['MACD'] = -35
            elif current_macd > current_signal:
                signals['MACD'] = 'Bullish'
                signal_scores['MACD'] = 15
            else:
                signals['MACD'] = 'Bearish'
                signal_scores['MACD'] = -15
            
            # Bollinger Bands Signal
            bb_upper = tech_data['BB_Upper'].iloc[-1]
            bb_lower = tech_data['BB_Lower'].iloc[-1]
            
            if current_price > bb_upper:
                signals['Bollinger'] = 'Above Upper Band'
                signal_scores['Bollinger'] = -10
            elif current_price < bb_lower:
                signals['Bollinger'] = 'Below Lower Band'
                signal_scores['Bollinger'] = 10
            else:
                signals['Bollinger'] = 'Within Bands'
                signal_scores['Bollinger'] = 0
            
            # Moving Average Signal
            sma_50 = tech_data['SMA_50'].iloc[-1]
            sma_200 = tech_data['SMA_200'].iloc[-1]
            
            if current_price > sma_50 > sma_200:
                signals['Moving_Average'] = 'Strong Bullish'
                signal_scores['Moving_Average'] = 30
            elif current_price > sma_50:
                signals['Moving_Average'] = 'Bullish'
                signal_scores['Moving_Average'] = 15
            elif current_price < sma_50 < sma_200:
                signals['Moving_Average'] = 'Strong Bearish'
                signal_scores['Moving_Average'] = -30
            else:
                signals['Moving_Average'] = 'Mixed'
                signal_scores['Moving_Average'] = 0
            
            # CONFLUENCE SCORING SYSTEM
            confluence_analysis = self._calculate_signal_confluence(signal_scores)
            signals.update(confluence_analysis)
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            signals['Error'] = 'Signal generation failed'
        
        return signals
    
    def _calculate_signal_confluence(self, signal_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate signal confluence and generate high-confidence signals."""
        confluence = {}
        
        try:
            # Calculate weighted confluence score
            total_score = sum(signal_scores.values())
            signal_count = len([s for s in signal_scores.values() if s != 0])
            
            # Normalize score to 0-100 scale
            max_possible_score = sum([abs(score) for score in signal_scores.values()])
            if max_possible_score > 0:
                confluence_score = ((total_score + max_possible_score) / (2 * max_possible_score)) * 100
            else:
                confluence_score = 50
            
            # Generate confluence signal
            if confluence_score >= 75 and signal_count >= 4:
                confluence['signal'] = 'STRONG_BUY'
                confluence['confidence'] = min(95, 60 + (confluence_score - 75) * 1.4)
            elif confluence_score >= 65 and signal_count >= 3:
                confluence['signal'] = 'BUY'
                confluence['confidence'] = min(85, 50 + (confluence_score - 65) * 1.5)
            elif confluence_score <= 25 and signal_count >= 4:
                confluence['signal'] = 'STRONG_SELL'
                confluence['confidence'] = min(95, 60 + (25 - confluence_score) * 1.4)
            elif confluence_score <= 35 and signal_count >= 3:
                confluence['signal'] = 'SELL'
                confluence['confidence'] = min(85, 50 + (35 - confluence_score) * 1.5)
            else:
                confluence['signal'] = 'HOLD'
                confluence['confidence'] = 50
            
            confluence['score'] = round(confluence_score, 1)
            confluence['active_signals'] = signal_count
            confluence['total_indicators'] = len(signal_scores)
            
            # Signal strength breakdown
            bullish_signals = len([s for s in signal_scores.values() if s > 0])
            bearish_signals = len([s for s in signal_scores.values() if s < 0])
            
            confluence['bullish_count'] = bullish_signals
            confluence['bearish_count'] = bearish_signals
            confluence['neutral_count'] = len(signal_scores) - bullish_signals - bearish_signals
            
            # High confidence flag
            confluence['high_confidence'] = confluence['confidence'] >= 80
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {str(e)}")
            confluence['error'] = str(e)
        
        return confluence
    
    def analyze_trends(self, tech_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics."""
        trend_analysis = {}
        
        try:
            current_price = tech_data['Close'].iloc[-1]
            sma_50 = tech_data['SMA_50'].iloc[-1]
            sma_200 = tech_data['SMA_200'].iloc[-1]
            
            # Primary trend determination
            if current_price > sma_50 > sma_200:
                primary_trend = 'Strong Uptrend'
                trend_strength = 'Strong'
            elif current_price > sma_50:
                primary_trend = 'Uptrend'
                trend_strength = 'Moderate'
            elif current_price < sma_50 < sma_200:
                primary_trend = 'Strong Downtrend'
                trend_strength = 'Strong'
            elif current_price < sma_50:
                primary_trend = 'Downtrend'
                trend_strength = 'Moderate'
            else:
                primary_trend = 'Sideways'
                trend_strength = 'Weak'
            
            trend_analysis['primary_trend'] = primary_trend
            trend_analysis['trend_strength'] = trend_strength
            
            # Momentum analysis
            rsi = tech_data['RSI'].iloc[-1]
            if rsi > 60:
                momentum = 'Strong Bullish'
            elif rsi > 50:
                momentum = 'Bullish'
            elif rsi < 40:
                momentum = 'Strong Bearish'
            elif rsi < 50:
                momentum = 'Bearish'
            else:
                momentum = 'Neutral'
            
            trend_analysis['momentum'] = momentum
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            trend_analysis['error'] = str(e)
        
        return trend_analysis
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = data['Low'].rolling(k_period).min()
        high_max = data['High'].rolling(k_period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(d_period).mean()
        
        return {'%K': k_percent, '%D': d_percent}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    # PRIORITY 1 ENHANCEMENT METHODS
    
    def _calculate_stochastic_rsi(self, rsi: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI - more sensitive momentum indicator."""
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        return stoch_rsi
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R - momentum oscillator."""
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index - momentum oscillator."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_deviation = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index and Directional Indicators."""
        # True Range calculation
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift())
        tr3 = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement calculation
        dm_plus = data['High'].diff()
        dm_minus = -data['Low'].diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        atr = true_range.rolling(period).mean()
        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return {
            'ADX': adx,
            'DI_Plus': di_plus,
            'DI_Minus': di_minus
        }
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index - volume-weighted RSI."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_flow_sum = positive_flow.rolling(period).sum()
        negative_flow_sum = negative_flow.rolling(period).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _calculate_chaikin_money_flow(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow - measures buying/selling pressure."""
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        clv = clv.fillna(0)  # Handle division by zero
        
        money_flow_volume = clv * data['Volume']
        cmf = money_flow_volume.rolling(period).sum() / data['Volume'].rolling(period).sum()
        
        return cmf
    
    def _calculate_roc(self, close: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change - momentum indicator."""
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame, initial_af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR - trend following indicator."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Initialize arrays
        psar = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)
        af = pd.Series(index=data.index, dtype=float)
        ep = pd.Series(index=data.index, dtype=float)
        
        # Initial values
        psar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        af.iloc[0] = initial_af
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(data)):
            if trend.iloc[i-1] == 1:  # Uptrend
                psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                if low.iloc[i] <= psar.iloc[i]:
                    # Trend reversal
                    trend.iloc[i] = -1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = initial_af
                else:
                    # Continue uptrend
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + initial_af, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                psar.iloc[i] = psar.iloc[i-1] - af.iloc[i-1] * (psar.iloc[i-1] - ep.iloc[i-1])
                
                if high.iloc[i] >= psar.iloc[i]:
                    # Trend reversal
                    trend.iloc[i] = 1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = initial_af
                else:
                    # Continue downtrend
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + initial_af, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return psar
