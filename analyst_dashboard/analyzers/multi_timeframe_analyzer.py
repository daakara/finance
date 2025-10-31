"""
Multi-Timeframe Analysis Module
Analyzes assets across multiple timeframes for better market context and signal confluence.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Analyzes assets across multiple timeframes to provide comprehensive market context.
    Implements timeframe alignment scoring and signal confluence analysis.
    """
    
    def __init__(self):
        """Initialize the multi-timeframe analyzer."""
        self.timeframe_weights = {
            '1mo': 1.0,   # Short-term weight
            '3mo': 1.5,   # Medium-term weight  
            '1y': 2.0     # Long-term weight (highest importance)
        }
        
        self.timeframe_labels = {
            '1mo': 'Short-term (1M)',
            '3mo': 'Medium-term (3M)', 
            '1y': 'Long-term (1Y)'
        }
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """
        Perform comprehensive multi-timeframe analysis.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary containing multi-timeframe analysis results
        """
        try:
            logger.info(f"Starting multi-timeframe analysis for {symbol}")
            
            # Fetch data for all timeframes
            timeframe_data = self._fetch_timeframe_data(symbol)
            
            if not timeframe_data:
                return {'error': 'No data available for multi-timeframe analysis'}
            
            # Analyze each timeframe
            timeframe_analysis = {}
            for period, data in timeframe_data.items():
                timeframe_analysis[period] = self._analyze_single_timeframe(data, period)
            
            # Calculate alignment scores
            alignment_scores = self._calculate_alignment_scores(timeframe_analysis)
            
            # Generate confluence signals
            confluence_signals = self._generate_confluence_signals(timeframe_analysis, alignment_scores)
            
            # Create summary
            summary = self._create_analysis_summary(timeframe_analysis, alignment_scores, confluence_signals)
            
            return {
                'symbol': symbol,
                'timeframe_data': timeframe_data,
                'timeframe_analysis': timeframe_analysis,
                'alignment_scores': alignment_scores,
                'confluence_signals': confluence_signals,
                'summary': summary,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _fetch_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all timeframes."""
        timeframe_data = {}
        
        for period in self.timeframe_weights.keys():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    timeframe_data[period] = data
                    logger.info(f"Fetched {len(data)} data points for {symbol} - {period}")
                else:
                    logger.warning(f"No data available for {symbol} - {period}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} - {period}: {str(e)}")
                continue
        
        return timeframe_data
    
    def _analyze_single_timeframe(self, data: pd.DataFrame, period: str) -> Dict:
        """Analyze a single timeframe for trend, momentum, and signals."""
        analysis = {}
        
        try:
            # Calculate technical indicators
            analysis['indicators'] = self._calculate_timeframe_indicators(data)
            
            # Determine trend direction and strength
            analysis['trend'] = self._analyze_trend(data, analysis['indicators'])
            
            # Analyze momentum
            analysis['momentum'] = self._analyze_momentum(analysis['indicators'])
            
            # Generate signals
            analysis['signals'] = self._generate_timeframe_signals(analysis['indicators'], analysis['trend'], analysis['momentum'])
            
            # Calculate confidence score
            analysis['confidence'] = self._calculate_timeframe_confidence(analysis['signals'])
            
            analysis['period'] = period
            analysis['period_label'] = self.timeframe_labels[period]
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {period}: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for a timeframe."""
        indicators = {}
        
        try:
            close = data['Close']
            
            # Moving averages
            indicators['sma_20'] = close.rolling(20).mean()
            indicators['sma_50'] = close.rolling(50).mean()
            indicators['sma_200'] = close.rolling(200).mean()
            
            # Exponential moving averages
            indicators['ema_12'] = close.ewm(span=12).mean()
            indicators['ema_26'] = close.ewm(span=26).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            indicators['bb_upper'] = bb_middle + (bb_std * 2)
            indicators['bb_lower'] = bb_middle - (bb_std * 2)
            indicators['bb_middle'] = bb_middle
            
            # Price position relative to BBs
            indicators['bb_position'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Volatility
            indicators['volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            indicators['error'] = str(e)
        
        return indicators
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze trend direction and strength."""
        trend = {}
        
        try:
            current_price = data['Close'].iloc[-1]
            
            # Moving average analysis
            sma_20_current = indicators['sma_20'].iloc[-1]
            sma_50_current = indicators['sma_50'].iloc[-1]
            sma_200_current = indicators['sma_200'].iloc[-1]
            
            # Trend direction based on MA hierarchy
            if current_price > sma_20_current > sma_50_current > sma_200_current:
                trend['direction'] = 'strong_uptrend'
                trend['score'] = 100
            elif current_price > sma_20_current > sma_50_current:
                trend['direction'] = 'uptrend'
                trend['score'] = 75
            elif current_price > sma_20_current:
                trend['direction'] = 'weak_uptrend'  
                trend['score'] = 60
            elif current_price < sma_20_current < sma_50_current < sma_200_current:
                trend['direction'] = 'strong_downtrend'
                trend['score'] = 0
            elif current_price < sma_20_current < sma_50_current:
                trend['direction'] = 'downtrend'
                trend['score'] = 25
            elif current_price < sma_20_current:
                trend['direction'] = 'weak_downtrend'
                trend['score'] = 40
            else:
                trend['direction'] = 'sideways'
                trend['score'] = 50
            
            # Trend strength based on MA slopes
            sma_20_slope = self._calculate_slope(indicators['sma_20'], 5)
            sma_50_slope = self._calculate_slope(indicators['sma_50'], 10)
            
            trend['strength'] = abs(sma_20_slope) + abs(sma_50_slope)
            trend['sma_20_slope'] = sma_20_slope
            trend['sma_50_slope'] = sma_50_slope
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            trend['error'] = str(e)
        
        return trend
    
    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """Analyze momentum indicators."""
        momentum = {}
        
        try:
            # RSI analysis
            current_rsi = indicators['rsi'].iloc[-1]
            rsi_trend = self._calculate_slope(indicators['rsi'], 5)
            
            if current_rsi > 70:
                momentum['rsi_signal'] = 'overbought'
                momentum['rsi_score'] = 20
            elif current_rsi < 30:
                momentum['rsi_signal'] = 'oversold'
                momentum['rsi_score'] = 80
            else:
                momentum['rsi_signal'] = 'neutral'
                momentum['rsi_score'] = 50
            
            momentum['rsi_value'] = current_rsi
            momentum['rsi_trend'] = rsi_trend
            
            # MACD analysis
            macd_current = indicators['macd'].iloc[-1]
            macd_signal_current = indicators['macd_signal'].iloc[-1]
            macd_histogram_current = indicators['macd_histogram'].iloc[-1]
            
            if macd_current > macd_signal_current and macd_histogram_current > 0:
                momentum['macd_signal'] = 'bullish'
                momentum['macd_score'] = 80
            elif macd_current < macd_signal_current and macd_histogram_current < 0:
                momentum['macd_signal'] = 'bearish'
                momentum['macd_score'] = 20
            else:
                momentum['macd_signal'] = 'neutral'
                momentum['macd_score'] = 50
            
            momentum['macd_value'] = macd_current
            momentum['macd_histogram'] = macd_histogram_current
            
            # Overall momentum score
            momentum['overall_score'] = (momentum['rsi_score'] + momentum['macd_score']) / 2
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}")
            momentum['error'] = str(e)
        
        return momentum
    
    def _generate_timeframe_signals(self, indicators: Dict, trend: Dict, momentum: Dict) -> Dict:
        """Generate trading signals for a timeframe."""
        signals = {}
        
        try:
            # Trend signals
            trend_score = trend.get('score', 50)
            if trend_score >= 75:
                signals['trend_signal'] = 'strong_buy'
            elif trend_score >= 60:
                signals['trend_signal'] = 'buy'
            elif trend_score <= 25:
                signals['trend_signal'] = 'strong_sell'
            elif trend_score <= 40:
                signals['trend_signal'] = 'sell'
            else:
                signals['trend_signal'] = 'hold'
            
            # Momentum signals
            momentum_score = momentum.get('overall_score', 50)
            if momentum_score >= 70:
                signals['momentum_signal'] = 'strong_buy'
            elif momentum_score >= 60:
                signals['momentum_signal'] = 'buy'
            elif momentum_score <= 30:
                signals['momentum_signal'] = 'strong_sell'
            elif momentum_score <= 40:
                signals['momentum_signal'] = 'sell'
            else:
                signals['momentum_signal'] = 'hold'
            
            # Combined signal
            combined_score = (trend_score + momentum_score) / 2
            if combined_score >= 75:
                signals['combined_signal'] = 'strong_buy'
            elif combined_score >= 60:
                signals['combined_signal'] = 'buy'
            elif combined_score <= 25:
                signals['combined_signal'] = 'strong_sell'
            elif combined_score <= 40:
                signals['combined_signal'] = 'sell'
            else:
                signals['combined_signal'] = 'hold'
            
            signals['combined_score'] = combined_score
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            signals['error'] = str(e)
        
        return signals
    
    def _calculate_timeframe_confidence(self, signals: Dict) -> float:
        """Calculate confidence score for timeframe analysis."""
        try:
            # Base confidence on signal alignment
            trend_signal = signals.get('trend_signal', 'hold')
            momentum_signal = signals.get('momentum_signal', 'hold')
            
            # High confidence when trend and momentum align
            if trend_signal == momentum_signal:
                if trend_signal in ['strong_buy', 'strong_sell']:
                    return 0.95
                elif trend_signal in ['buy', 'sell']:
                    return 0.80
                else:
                    return 0.60
            else:
                # Lower confidence when signals conflict
                return 0.40
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.50
    
    def _calculate_alignment_scores(self, timeframe_analysis: Dict) -> Dict:
        """Calculate alignment scores across timeframes.""" 
        alignment = {}
        
        try:
            # Extract signals from each timeframe
            timeframe_signals = {}
            for period, analysis in timeframe_analysis.items():
                if 'signals' in analysis:
                    timeframe_signals[period] = analysis['signals'].get('combined_score', 50)
            
            # Calculate weighted alignment score
            total_weight = sum(self.timeframe_weights.values())
            weighted_score = 0
            
            for period, score in timeframe_signals.items():
                weight = self.timeframe_weights.get(period, 1.0)
                weighted_score += (score * weight)
            
            alignment['weighted_score'] = weighted_score / total_weight
            
            # Calculate consensus strength
            scores = list(timeframe_signals.values())
            if len(scores) > 1:
                alignment['consensus_strength'] = 1 - (np.std(scores) / 50)  # Normalize by max possible std
            else:
                alignment['consensus_strength'] = 1.0
            
            # Overall alignment rating
            if alignment['consensus_strength'] > 0.8 and alignment['weighted_score'] > 70:
                alignment['rating'] = 'strong_bullish_alignment'
            elif alignment['consensus_strength'] > 0.8 and alignment['weighted_score'] < 30:
                alignment['rating'] = 'strong_bearish_alignment'
            elif alignment['consensus_strength'] > 0.6:
                alignment['rating'] = 'moderate_alignment'
            else:
                alignment['rating'] = 'mixed_signals'
            
            alignment['individual_scores'] = timeframe_signals
            
        except Exception as e:
            logger.error(f"Error calculating alignment: {str(e)}")
            alignment['error'] = str(e)
        
        return alignment
    
    def _generate_confluence_signals(self, timeframe_analysis: Dict, alignment_scores: Dict) -> Dict:
        """Generate high-confidence confluence signals."""
        confluence = {}
        
        try:
            weighted_score = alignment_scores.get('weighted_score', 50)
            consensus_strength = alignment_scores.get('consensus_strength', 0)
            
            # High confidence thresholds
            if consensus_strength > 0.8 and weighted_score > 75:
                confluence['signal'] = 'STRONG_BUY'
                confluence['confidence'] = min(0.95, consensus_strength + 0.1)
                confluence['description'] = 'All timeframes strongly aligned bullish'
            elif consensus_strength > 0.8 and weighted_score < 25:
                confluence['signal'] = 'STRONG_SELL'
                confluence['confidence'] = min(0.95, consensus_strength + 0.1)
                confluence['description'] = 'All timeframes strongly aligned bearish'
            elif consensus_strength > 0.6 and weighted_score > 65:
                confluence['signal'] = 'BUY'
                confluence['confidence'] = min(0.80, consensus_strength)
                confluence['description'] = 'Most timeframes aligned bullish'
            elif consensus_strength > 0.6 and weighted_score < 35:
                confluence['signal'] = 'SELL'
                confluence['confidence'] = min(0.80, consensus_strength)
                confluence['description'] = 'Most timeframes aligned bearish'
            else:
                confluence['signal'] = 'HOLD'
                confluence['confidence'] = 0.50
                confluence['description'] = 'Mixed signals across timeframes'
            
            # Add supporting details
            confluence['weighted_score'] = weighted_score
            confluence['consensus_strength'] = consensus_strength
            confluence['timeframe_count'] = len(timeframe_analysis)
            
        except Exception as e:
            logger.error(f"Error generating confluence signals: {str(e)}")
            confluence['error'] = str(e)
        
        return confluence
    
    def _create_analysis_summary(self, timeframe_analysis: Dict, alignment_scores: Dict, confluence_signals: Dict) -> Dict:
        """Create a comprehensive analysis summary."""
        summary = {}
        
        try:
            # Overall assessment
            summary['confluence_signal'] = confluence_signals.get('signal', 'HOLD')
            summary['confidence_level'] = confluence_signals.get('confidence', 0.5)
            summary['alignment_rating'] = alignment_scores.get('rating', 'mixed_signals')
            
            # Timeframe breakdown
            summary['timeframes'] = {}
            for period, analysis in timeframe_analysis.items():
                summary['timeframes'][period] = {
                    'trend': analysis.get('trend', {}).get('direction', 'unknown'),
                    'signal': analysis.get('signals', {}).get('combined_signal', 'hold'),
                    'confidence': analysis.get('confidence', 0.5)
                }
            
            # Key insights
            summary['insights'] = []
            
            if alignment_scores.get('consensus_strength', 0) > 0.8:
                summary['insights'].append('Strong consensus across all timeframes')
            
            if confluence_signals.get('confidence', 0) > 0.85:
                summary['insights'].append('High-confidence trading opportunity')
            
            # Risk assessment
            summary['risk_level'] = self._assess_risk_level(timeframe_analysis, alignment_scores)
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def _assess_risk_level(self, timeframe_analysis: Dict, alignment_scores: Dict) -> str:
        """Assess overall risk level based on analysis."""
        try:
            consensus = alignment_scores.get('consensus_strength', 0)
            
            if consensus > 0.8:
                return 'low'  # High consensus = lower risk
            elif consensus > 0.6:
                return 'medium'
            else:
                return 'high'  # Mixed signals = higher risk
                
        except Exception:
            return 'medium'
    
    def _calculate_slope(self, series: pd.Series, periods: int) -> float:
        """Calculate the slope of a series over the last N periods."""
        try:
            if len(series) < periods:
                return 0.0
            
            recent_data = series.iloc[-periods:].dropna()
            if len(recent_data) < 2:
                return 0.0
            
            x = np.arange(len(recent_data))
            y = recent_data.values
            
            # Calculate slope using least squares
            slope = np.polyfit(x, y, 1)[0]
            return slope
            
        except Exception:
            return 0.0

    def get_timeframe_summary(self, analysis_result: Dict) -> str:
        """Get a human-readable summary of the multi-timeframe analysis."""
        if 'error' in analysis_result:
            return f"Analysis Error: {analysis_result['error']}"
        
        try:
            summary = analysis_result.get('summary', {})
            confluence = analysis_result.get('confluence_signals', {})
            
            signal = confluence.get('signal', 'HOLD')
            confidence = confluence.get('confidence', 0.5)
            description = confluence.get('description', 'No clear signal')
            
            return f"Multi-Timeframe Signal: {signal} (Confidence: {confidence:.1%}) - {description}"
            
        except Exception as e:
            return f"Summary Error: {str(e)}"
