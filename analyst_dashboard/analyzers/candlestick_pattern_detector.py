"""
Candlestick Pattern Recognition Module - Priority 3 Implementation
Advanced candlestick pattern detection with reliability scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CandlestickPattern:
    """Data class for candlestick pattern information"""
    name: str
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    reliability: float  # 0-100 score
    strength: str  # 'weak', 'moderate', 'strong'
    description: str
    entry_signal: bool
    exit_signal: bool

class CandlestickPatternDetector:
    """Priority 3: Advanced candlestick pattern recognition with reliability scoring"""
    
    def __init__(self):
        self.pattern_reliability = {
            # Single Candlestick Patterns
            'doji': {'base_reliability': 65, 'signal_type': 'neutral'},
            'hammer': {'base_reliability': 75, 'signal_type': 'bullish'},
            'shooting_star': {'base_reliability': 75, 'signal_type': 'bearish'},
            'hanging_man': {'base_reliability': 70, 'signal_type': 'bearish'},
            'inverted_hammer': {'base_reliability': 70, 'signal_type': 'bullish'},
            'spinning_top': {'base_reliability': 50, 'signal_type': 'neutral'},
            'marubozu_bullish': {'base_reliability': 80, 'signal_type': 'bullish'},
            'marubozu_bearish': {'base_reliability': 80, 'signal_type': 'bearish'},
            
            # Multi-Candlestick Patterns
            'bullish_engulfing': {'base_reliability': 85, 'signal_type': 'bullish'},
            'bearish_engulfing': {'base_reliability': 85, 'signal_type': 'bearish'},
            'morning_star': {'base_reliability': 90, 'signal_type': 'bullish'},
            'evening_star': {'base_reliability': 90, 'signal_type': 'bearish'},
            'three_white_soldiers': {'base_reliability': 88, 'signal_type': 'bullish'},
            'three_black_crows': {'base_reliability': 88, 'signal_type': 'bearish'},
            'piercing_pattern': {'base_reliability': 75, 'signal_type': 'bullish'},
            'dark_cloud_cover': {'base_reliability': 75, 'signal_type': 'bearish'},
            'harami_bullish': {'base_reliability': 70, 'signal_type': 'bullish'},
            'harami_bearish': {'base_reliability': 70, 'signal_type': 'bearish'}
        }
        
        self.pattern_descriptions = {
            'doji': 'Indecision pattern - open equals close, signals potential reversal',
            'hammer': 'Bullish reversal - small body, long lower shadow at bottom of downtrend',
            'shooting_star': 'Bearish reversal - small body, long upper shadow at top of uptrend',
            'hanging_man': 'Bearish reversal - hammer-like at top of uptrend',
            'inverted_hammer': 'Bullish reversal - shooting star-like at bottom of downtrend',
            'spinning_top': 'Indecision - small body with upper and lower shadows',
            'marubozu_bullish': 'Strong bullish - no shadows, close at high',
            'marubozu_bearish': 'Strong bearish - no shadows, close at low',
            'bullish_engulfing': 'Strong bullish reversal - large bullish candle engulfs previous bearish',
            'bearish_engulfing': 'Strong bearish reversal - large bearish candle engulfs previous bullish',
            'morning_star': 'Strong bullish reversal - three-candle bottom reversal pattern',
            'evening_star': 'Strong bearish reversal - three-candle top reversal pattern',
            'three_white_soldiers': 'Strong bullish continuation - three consecutive bullish candles',
            'three_black_crows': 'Strong bearish continuation - three consecutive bearish candles',
            'piercing_pattern': 'Bullish reversal - bullish candle pierces midpoint of previous bearish',
            'dark_cloud_cover': 'Bearish reversal - bearish candle covers midpoint of previous bullish',
            'harami_bullish': 'Bullish reversal - small bearish inside large bullish',
            'harami_bearish': 'Bearish reversal - small bullish inside large bearish'
        }
    
    def detect_all_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect all candlestick patterns with reliability scoring"""
        try:
            patterns_detected = {}
            pattern_summary = {
                'total_patterns': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'neutral_patterns': 0,
                'high_reliability_patterns': 0,
                'recent_patterns': [],
                'pattern_frequency': {},
                'signal_strength': 'neutral'
            }
            
            # Detect single candlestick patterns
            single_patterns = self._detect_single_candlestick_patterns(price_data)
            patterns_detected.update(single_patterns)
            
            # Detect multi-candlestick patterns
            multi_patterns = self._detect_multi_candlestick_patterns(price_data)
            patterns_detected.update(multi_patterns)
            
            # Calculate pattern summary statistics
            for pattern_name, pattern_data in patterns_detected.items():
                if pattern_data and len(pattern_data) > 0:
                    pattern_summary['total_patterns'] += len(pattern_data)
                    pattern_summary['pattern_frequency'][pattern_name] = len(pattern_data)
                    
                    # Count by signal type
                    signal_type = self.pattern_reliability[pattern_name]['signal_type']
                    if signal_type == 'bullish':
                        pattern_summary['bullish_patterns'] += len(pattern_data)
                    elif signal_type == 'bearish':
                        pattern_summary['bearish_patterns'] += len(pattern_data)
                    else:
                        pattern_summary['neutral_patterns'] += len(pattern_data)
                    
                    # Count high reliability patterns (>80%)
                    high_rel_count = sum(1 for p in pattern_data if p.reliability > 80)
                    pattern_summary['high_reliability_patterns'] += high_rel_count
                    
                    # Recent patterns (last 30 days)
                    if len(price_data) >= 30:
                        recent_cutoff = price_data.index[-30]
                        recent_patterns = [p for p in pattern_data if p.name in price_data.index[-30:]]
                        pattern_summary['recent_patterns'].extend(recent_patterns)
            
            # Determine overall signal strength
            if pattern_summary['bullish_patterns'] > pattern_summary['bearish_patterns'] * 1.5:
                pattern_summary['signal_strength'] = 'bullish'
            elif pattern_summary['bearish_patterns'] > pattern_summary['bullish_patterns'] * 1.5:
                pattern_summary['signal_strength'] = 'bearish'
            else:
                pattern_summary['signal_strength'] = 'neutral'
            
            return {
                'patterns': patterns_detected,
                'summary': pattern_summary,
                'pattern_insights': self._generate_pattern_insights(patterns_detected, pattern_summary)
            }
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return {'error': str(e)}
    
    def _detect_single_candlestick_patterns(self, price_data: pd.DataFrame) -> Dict[str, List[CandlestickPattern]]:
        """Detect single candlestick patterns"""
        patterns = {
            'doji': [],
            'hammer': [],
            'shooting_star': [],
            'hanging_man': [],
            'inverted_hammer': [],
            'spinning_top': [],
            'marubozu_bullish': [],
            'marubozu_bearish': []
        }
        
        try:
            for i in range(1, len(price_data)):
                current = price_data.iloc[i]
                prev = price_data.iloc[i-1] if i > 0 else None
                
                open_price = current['Open']
                high_price = current['High']
                low_price = current['Low']
                close_price = current['Close']
                
                body_size = abs(close_price - open_price)
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                candle_range = high_price - low_price
                
                # Avoid division by zero
                if candle_range == 0:
                    continue
                
                body_ratio = body_size / candle_range
                upper_shadow_ratio = upper_shadow / candle_range
                lower_shadow_ratio = lower_shadow / candle_range
                
                # Doji pattern
                if body_ratio < 0.1:
                    reliability = self._calculate_pattern_reliability('doji', current, prev, price_data, i)
                    patterns['doji'].append(CandlestickPattern(
                        name='doji',
                        signal_type='neutral',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['doji'],
                        entry_signal=False,
                        exit_signal=True
                    ))
                
                # Hammer pattern (bullish reversal)
                elif (body_ratio < 0.3 and lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and 
                      self._is_downtrend(price_data, i)):
                    reliability = self._calculate_pattern_reliability('hammer', current, prev, price_data, i)
                    patterns['hammer'].append(CandlestickPattern(
                        name='hammer',
                        signal_type='bullish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['hammer'],
                        entry_signal=True,
                        exit_signal=False
                    ))
                
                # Shooting Star pattern (bearish reversal)
                elif (body_ratio < 0.3 and upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and 
                      self._is_uptrend(price_data, i)):
                    reliability = self._calculate_pattern_reliability('shooting_star', current, prev, price_data, i)
                    patterns['shooting_star'].append(CandlestickPattern(
                        name='shooting_star',
                        signal_type='bearish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['shooting_star'],
                        entry_signal=False,
                        exit_signal=True
                    ))
                
                # Hanging Man pattern (bearish reversal)
                elif (body_ratio < 0.3 and lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and 
                      self._is_uptrend(price_data, i)):
                    reliability = self._calculate_pattern_reliability('hanging_man', current, prev, price_data, i)
                    patterns['hanging_man'].append(CandlestickPattern(
                        name='hanging_man',
                        signal_type='bearish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['hanging_man'],
                        entry_signal=False,
                        exit_signal=True
                    ))
                
                # Inverted Hammer pattern (bullish reversal)
                elif (body_ratio < 0.3 and upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and 
                      self._is_downtrend(price_data, i)):
                    reliability = self._calculate_pattern_reliability('inverted_hammer', current, prev, price_data, i)
                    patterns['inverted_hammer'].append(CandlestickPattern(
                        name='inverted_hammer',
                        signal_type='bullish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['inverted_hammer'],
                        entry_signal=True,
                        exit_signal=False
                    ))
                
                # Spinning Top pattern (indecision)
                elif (body_ratio < 0.3 and upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3):
                    reliability = self._calculate_pattern_reliability('spinning_top', current, prev, price_data, i)
                    patterns['spinning_top'].append(CandlestickPattern(
                        name='spinning_top',
                        signal_type='neutral',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['spinning_top'],
                        entry_signal=False,
                        exit_signal=False
                    ))
                
                # Marubozu patterns (strong directional movement)
                elif (body_ratio > 0.9 and upper_shadow_ratio < 0.05 and lower_shadow_ratio < 0.05):
                    if close_price > open_price:  # Bullish Marubozu
                        reliability = self._calculate_pattern_reliability('marubozu_bullish', current, prev, price_data, i)
                        patterns['marubozu_bullish'].append(CandlestickPattern(
                            name='marubozu_bullish',
                            signal_type='bullish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['marubozu_bullish'],
                            entry_signal=True,
                            exit_signal=False
                        ))
                    else:  # Bearish Marubozu
                        reliability = self._calculate_pattern_reliability('marubozu_bearish', current, prev, price_data, i)
                        patterns['marubozu_bearish'].append(CandlestickPattern(
                            name='marubozu_bearish',
                            signal_type='bearish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['marubozu_bearish'],
                            entry_signal=False,
                            exit_signal=True
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting single candlestick patterns: {str(e)}")
            return patterns
    
    def _detect_multi_candlestick_patterns(self, price_data: pd.DataFrame) -> Dict[str, List[CandlestickPattern]]:
        """Detect multi-candlestick patterns"""
        patterns = {
            'bullish_engulfing': [],
            'bearish_engulfing': [],
            'morning_star': [],
            'evening_star': [],
            'three_white_soldiers': [],
            'three_black_crows': [],
            'piercing_pattern': [],
            'dark_cloud_cover': [],
            'harami_bullish': [],
            'harami_bearish': []
        }
        
        try:
            for i in range(2, len(price_data)):
                current = price_data.iloc[i]
                prev = price_data.iloc[i-1]
                prev2 = price_data.iloc[i-2] if i >= 2 else None
                
                # Bullish Engulfing
                if (prev['Close'] < prev['Open'] and  # Previous bearish
                    current['Close'] > current['Open'] and  # Current bullish
                    current['Open'] < prev['Close'] and  # Opens below previous close
                    current['Close'] > prev['Open'] and  # Closes above previous open
                    self._is_downtrend(price_data, i)):
                    
                    reliability = self._calculate_pattern_reliability('bullish_engulfing', current, prev, price_data, i)
                    patterns['bullish_engulfing'].append(CandlestickPattern(
                        name='bullish_engulfing',
                        signal_type='bullish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['bullish_engulfing'],
                        entry_signal=True,
                        exit_signal=False
                    ))
                
                # Bearish Engulfing
                elif (prev['Close'] > prev['Open'] and  # Previous bullish
                      current['Close'] < current['Open'] and  # Current bearish
                      current['Open'] > prev['Close'] and  # Opens above previous close
                      current['Close'] < prev['Open'] and  # Closes below previous open
                      self._is_uptrend(price_data, i)):
                    
                    reliability = self._calculate_pattern_reliability('bearish_engulfing', current, prev, price_data, i)
                    patterns['bearish_engulfing'].append(CandlestickPattern(
                        name='bearish_engulfing',
                        signal_type='bearish',
                        reliability=reliability,
                        strength=self._get_strength_from_reliability(reliability),
                        description=self.pattern_descriptions['bearish_engulfing'],
                        entry_signal=False,
                        exit_signal=True
                    ))
                
                # Three-candle patterns
                if prev2 is not None:
                    # Morning Star (bullish reversal)
                    if (prev2['Close'] < prev2['Open'] and  # First bearish
                        abs(prev['Close'] - prev['Open']) < abs(prev2['Close'] - prev2['Open']) * 0.3 and  # Small middle
                        current['Close'] > current['Open'] and  # Third bullish
                        current['Close'] > (prev2['Open'] + prev2['Close']) / 2 and  # Closes above midpoint
                        self._is_downtrend(price_data, i)):
                        
                        reliability = self._calculate_pattern_reliability('morning_star', current, prev, price_data, i)
                        patterns['morning_star'].append(CandlestickPattern(
                            name='morning_star',
                            signal_type='bullish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['morning_star'],
                            entry_signal=True,
                            exit_signal=False
                        ))
                    
                    # Evening Star (bearish reversal)
                    elif (prev2['Close'] > prev2['Open'] and  # First bullish
                          abs(prev['Close'] - prev['Open']) < abs(prev2['Close'] - prev2['Open']) * 0.3 and  # Small middle
                          current['Close'] < current['Open'] and  # Third bearish
                          current['Close'] < (prev2['Open'] + prev2['Close']) / 2 and  # Closes below midpoint
                          self._is_uptrend(price_data, i)):
                        
                        reliability = self._calculate_pattern_reliability('evening_star', current, prev, price_data, i)
                        patterns['evening_star'].append(CandlestickPattern(
                            name='evening_star',
                            signal_type='bearish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['evening_star'],
                            entry_signal=False,
                            exit_signal=True
                        ))
                    
                    # Three White Soldiers (bullish continuation)
                    elif (prev2['Close'] > prev2['Open'] and  # All three bullish
                          prev['Close'] > prev['Open'] and
                          current['Close'] > current['Open'] and
                          prev['Close'] > prev2['Close'] and  # Each closes higher
                          current['Close'] > prev['Close'] and
                          prev['Open'] > prev2['Close'] * 0.9 and  # Opens near previous close
                          current['Open'] > prev['Close'] * 0.9):
                        
                        reliability = self._calculate_pattern_reliability('three_white_soldiers', current, prev, price_data, i)
                        patterns['three_white_soldiers'].append(CandlestickPattern(
                            name='three_white_soldiers',
                            signal_type='bullish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['three_white_soldiers'],
                            entry_signal=True,
                            exit_signal=False
                        ))
                    
                    # Three Black Crows (bearish continuation)
                    elif (prev2['Close'] < prev2['Open'] and  # All three bearish
                          prev['Close'] < prev['Open'] and
                          current['Close'] < current['Open'] and
                          prev['Close'] < prev2['Close'] and  # Each closes lower
                          current['Close'] < prev['Close'] and
                          prev['Open'] < prev2['Close'] * 1.1 and  # Opens near previous close
                          current['Open'] < prev['Close'] * 1.1):
                        
                        reliability = self._calculate_pattern_reliability('three_black_crows', current, prev, price_data, i)
                        patterns['three_black_crows'].append(CandlestickPattern(
                            name='three_black_crows',
                            signal_type='bearish',
                            reliability=reliability,
                            strength=self._get_strength_from_reliability(reliability),
                            description=self.pattern_descriptions['three_black_crows'],
                            entry_signal=False,
                            exit_signal=True
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting multi-candlestick patterns: {str(e)}")
            return patterns
    
    def _calculate_pattern_reliability(self, pattern_name: str, current: pd.Series, 
                                     prev: Optional[pd.Series], price_data: pd.DataFrame, 
                                     index: int) -> float:
        """Calculate pattern reliability based on context and market conditions"""
        try:
            base_reliability = self.pattern_reliability[pattern_name]['base_reliability']
            
            # Adjust reliability based on volume
            volume_factor = 1.0
            if 'Volume' in current and 'Volume' in prev:
                avg_volume = price_data['Volume'].rolling(20).mean().iloc[index]
                if current['Volume'] > avg_volume * 1.2:
                    volume_factor = 1.1  # Higher volume increases reliability
                elif current['Volume'] < avg_volume * 0.8:
                    volume_factor = 0.9  # Lower volume decreases reliability
            
            # Adjust reliability based on trend strength
            trend_factor = 1.0
            if index >= 10:
                recent_closes = price_data['Close'].iloc[index-10:index+1]
                trend_strength = abs(recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
                if trend_strength > 0.05:  # Strong trend
                    trend_factor = 1.1
                elif trend_strength < 0.02:  # Weak trend
                    trend_factor = 0.95
            
            # Adjust reliability based on volatility
            volatility_factor = 1.0
            if index >= 20:
                recent_returns = price_data['Close'].pct_change().iloc[index-20:index+1]
                volatility = recent_returns.std()
                avg_volatility = price_data['Close'].pct_change().rolling(60).std().iloc[index]
                if volatility > avg_volatility * 1.3:  # High volatility
                    volatility_factor = 0.9  # Reduces reliability
                elif volatility < avg_volatility * 0.7:  # Low volatility
                    volatility_factor = 1.05  # Increases reliability
            
            # Calculate final reliability
            final_reliability = base_reliability * volume_factor * trend_factor * volatility_factor
            
            # Cap between 0 and 100
            return max(0, min(100, final_reliability))
            
        except Exception as e:
            logger.error(f"Error calculating pattern reliability: {str(e)}")
            return self.pattern_reliability[pattern_name]['base_reliability']
    
    def _is_uptrend(self, price_data: pd.DataFrame, index: int, period: int = 5) -> bool:
        """Check if price is in uptrend"""
        try:
            if index < period:
                return False
            recent_closes = price_data['Close'].iloc[index-period:index+1]
            return recent_closes.iloc[-1] > recent_closes.iloc[0]
        except:
            return False
    
    def _is_downtrend(self, price_data: pd.DataFrame, index: int, period: int = 5) -> bool:
        """Check if price is in downtrend"""
        try:
            if index < period:
                return False
            recent_closes = price_data['Close'].iloc[index-period:index+1]
            return recent_closes.iloc[-1] < recent_closes.iloc[0]
        except:
            return False
    
    def _get_strength_from_reliability(self, reliability: float) -> str:
        """Convert reliability score to strength category"""
        if reliability >= 80:
            return 'strong'
        elif reliability >= 65:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_pattern_insights(self, patterns: Dict[str, List], summary: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from pattern analysis"""
        insights = []
        
        try:
            total_patterns = summary['total_patterns']
            if total_patterns == 0:
                insights.append("📊 No significant candlestick patterns detected in recent data")
                return insights
            
            # Signal strength insight
            signal_strength = summary['signal_strength']
            if signal_strength == 'bullish':
                insights.append(f"🐂 Bullish pattern dominance detected with {summary['bullish_patterns']} bullish vs {summary['bearish_patterns']} bearish patterns")
            elif signal_strength == 'bearish':
                insights.append(f"🐻 Bearish pattern dominance detected with {summary['bearish_patterns']} bearish vs {summary['bullish_patterns']} bullish patterns")
            else:
                insights.append(f"⚖️ Balanced pattern distribution: {summary['bullish_patterns']} bullish, {summary['bearish_patterns']} bearish patterns")
            
            # High reliability patterns
            high_rel = summary['high_reliability_patterns']
            if high_rel > 0:
                insights.append(f"⭐ {high_rel} high-reliability patterns (>80%) detected - strong signal confidence")
            
            # Most frequent patterns
            pattern_freq = summary['pattern_frequency']
            if pattern_freq:
                most_common = max(pattern_freq.items(), key=lambda x: x[1])
                insights.append(f"📈 Most frequent pattern: {most_common[0].replace('_', ' ').title()} ({most_common[1]} occurrences)")
            
            # Recent pattern activity
            recent_patterns = summary['recent_patterns']
            if len(recent_patterns) > 0:
                insights.append(f"⏰ {len(recent_patterns)} patterns detected in last 30 days - active pattern formation")
            else:
                insights.append("⏰ Limited recent pattern activity - market in consolidation phase")
            
            # Strong reversal patterns
            strong_reversal_patterns = ['morning_star', 'evening_star', 'bullish_engulfing', 'bearish_engulfing']
            strong_reversals = sum(len(patterns.get(pattern, [])) for pattern in strong_reversal_patterns)
            if strong_reversals > 0:
                insights.append(f"🔄 {strong_reversals} strong reversal patterns detected - potential trend change signals")
            
        except Exception as e:
            logger.error(f"Error generating pattern insights: {str(e)}")
            insights.append("⚠️ Unable to generate pattern insights due to analysis limitations")
        
        return insights
