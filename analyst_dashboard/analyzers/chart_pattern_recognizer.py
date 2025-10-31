"""
Chart Pattern Recognition Module - Priority 3 Implementation
Advanced chart pattern detection including Head & Shoulders, Double Tops/Bottoms, Triangles, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt

logger = logging.getLogger(__name__)

@dataclass
class ChartPattern:
    """Data class for chart pattern information"""
    name: str
    pattern_type: str  # 'reversal', 'continuation', 'breakout'
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-100 score
    strength: str  # 'weak', 'moderate', 'strong'
    start_date: str
    end_date: str
    key_levels: Dict[str, float]  # Support, resistance, targets
    description: str
    entry_signal: bool
    target_price: Optional[float]

class ChartPatternRecognizer:
    """Priority 3: Advanced chart pattern recognition with breakout detection"""
    
    def __init__(self):
        self.pattern_definitions = {
            'head_and_shoulders': {
                'type': 'reversal',
                'base_confidence': 85,
                'description': 'Bearish reversal pattern with three peaks, middle highest'
            },
            'inverse_head_and_shoulders': {
                'type': 'reversal', 
                'base_confidence': 85,
                'description': 'Bullish reversal pattern with three troughs, middle lowest'
            },
            'double_top': {
                'type': 'reversal',
                'base_confidence': 80,
                'description': 'Bearish reversal pattern with two equal peaks'
            },
            'double_bottom': {
                'type': 'reversal',
                'base_confidence': 80,
                'description': 'Bullish reversal pattern with two equal troughs'
            },
            'ascending_triangle': {
                'type': 'continuation',
                'base_confidence': 75,
                'description': 'Bullish continuation with horizontal resistance and rising support'
            },
            'descending_triangle': {
                'type': 'continuation',
                'base_confidence': 75,
                'description': 'Bearish continuation with horizontal support and declining resistance'
            },
            'symmetrical_triangle': {
                'type': 'breakout',
                'base_confidence': 70,
                'description': 'Neutral pattern with converging support and resistance'
            },
            'flag_bullish': {
                'type': 'continuation',
                'base_confidence': 78,
                'description': 'Bullish continuation pattern after strong upward move'
            },
            'flag_bearish': {
                'type': 'continuation',
                'base_confidence': 78,
                'description': 'Bearish continuation pattern after strong downward move'
            },
            'pennant_bullish': {
                'type': 'continuation',
                'base_confidence': 76,
                'description': 'Bullish continuation with converging trendlines after upward move'
            },
            'pennant_bearish': {
                'type': 'continuation',
                'base_confidence': 76,
                'description': 'Bearish continuation with converging trendlines after downward move'
            },
            'wedge_rising': {
                'type': 'reversal',
                'base_confidence': 72,
                'description': 'Bearish reversal with rising support and resistance, narrowing range'
            },
            'wedge_falling': {
                'type': 'reversal',
                'base_confidence': 72,
                'description': 'Bullish reversal with falling support and resistance, narrowing range'
            }
        }
    
    def detect_all_patterns(self, price_data: pd.DataFrame, lookback_period: int = 100) -> Dict[str, Any]:
        """Detect all chart patterns with confidence scoring"""
        try:
            # Ensure we have enough data
            if len(price_data) < 50:
                return {'error': 'Insufficient data for pattern recognition'}
            
            # Limit analysis to recent data for performance
            analysis_data = price_data.tail(min(len(price_data), lookback_period))
            
            patterns_detected = {}
            pattern_summary = {
                'total_patterns': 0,
                'reversal_patterns': 0,
                'continuation_patterns': 0,
                'breakout_patterns': 0,
                'high_confidence_patterns': 0,
                'support_resistance_levels': [],
                'trend_analysis': {},
                'pattern_insights': []
            }
            
            # Find peaks and troughs for pattern analysis
            peaks, troughs = self._find_peaks_and_troughs(analysis_data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(analysis_data, peaks, troughs)
            patterns_detected.update(reversal_patterns)
            
            # Detect continuation patterns
            continuation_patterns = self._detect_continuation_patterns(analysis_data, peaks, troughs)
            patterns_detected.update(continuation_patterns)
            
            # Detect triangle patterns
            triangle_patterns = self._detect_triangle_patterns(analysis_data, peaks, troughs)
            patterns_detected.update(triangle_patterns)
            
            # Calculate support and resistance levels
            support_resistance = self._calculate_support_resistance(analysis_data, peaks, troughs)
            pattern_summary['support_resistance_levels'] = support_resistance
            
            # Analyze trend context
            trend_analysis = self._analyze_trend_context(analysis_data)
            pattern_summary['trend_analysis'] = trend_analysis
            
            # Calculate summary statistics
            for pattern_name, pattern_list in patterns_detected.items():
                if pattern_list and len(pattern_list) > 0:
                    pattern_summary['total_patterns'] += len(pattern_list)
                    
                    pattern_type = self.pattern_definitions[pattern_name]['type']
                    if pattern_type == 'reversal':
                        pattern_summary['reversal_patterns'] += len(pattern_list)
                    elif pattern_type == 'continuation':
                        pattern_summary['continuation_patterns'] += len(pattern_list)
                    else:
                        pattern_summary['breakout_patterns'] += len(pattern_list)
                    
                    # Count high confidence patterns (>80%)
                    high_conf_count = sum(1 for p in pattern_list if p.confidence > 80)
                    pattern_summary['high_confidence_patterns'] += high_conf_count
            
            # Generate pattern insights
            pattern_summary['pattern_insights'] = self._generate_pattern_insights(
                patterns_detected, pattern_summary, analysis_data
            )
            
            return {
                'patterns': patterns_detected,
                'summary': pattern_summary,
                'analysis_period': f"{len(analysis_data)} days",
                'key_levels': support_resistance
            }
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {str(e)}")
            return {'error': str(e)}
    
    def _find_peaks_and_troughs(self, price_data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs in price data"""
        try:
            highs = price_data['High'].values
            lows = price_data['Low'].values
            
            # Find peaks in highs (local maxima)
            peak_indices, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            
            # Find troughs in lows (local minima) 
            trough_indices, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            return list(peak_indices), list(trough_indices)
            
        except Exception as e:
            logger.error(f"Error finding peaks and troughs: {str(e)}")
            return [], []
    
    def _detect_reversal_patterns(self, price_data: pd.DataFrame, 
                                peaks: List[int], troughs: List[int]) -> Dict[str, List[ChartPattern]]:
        """Detect reversal patterns (Head & Shoulders, Double Tops/Bottoms)"""
        patterns = {
            'head_and_shoulders': [],
            'inverse_head_and_shoulders': [],
            'double_top': [],
            'double_bottom': []
        }
        
        try:
            # Head and Shoulders pattern
            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left_peak = peaks[i]
                    head = peaks[i + 1]
                    right_peak = peaks[i + 2]
                    
                    left_high = price_data['High'].iloc[left_peak]
                    head_high = price_data['High'].iloc[head]
                    right_high = price_data['High'].iloc[right_peak]
                    
                    # Check H&S criteria: head higher than shoulders, shoulders roughly equal
                    if (head_high > left_high * 1.02 and head_high > right_high * 1.02 and
                        abs(left_high - right_high) / max(left_high, right_high) < 0.05):
                        
                        # Find neckline (troughs between peaks)
                        relevant_troughs = [t for t in troughs if left_peak < t < right_peak]
                        if len(relevant_troughs) >= 1:
                            neckline_level = min(price_data['Low'].iloc[t] for t in relevant_troughs)
                            
                            confidence = self._calculate_pattern_confidence(
                                'head_and_shoulders', price_data, left_peak, right_peak
                            )
                            
                            patterns['head_and_shoulders'].append(ChartPattern(
                                name='head_and_shoulders',
                                pattern_type='reversal',
                                signal_type='bearish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[left_peak].date()),
                                end_date=str(price_data.index[right_peak].date()),
                                key_levels={
                                    'neckline': neckline_level,
                                    'head': head_high,
                                    'left_shoulder': left_high,
                                    'right_shoulder': right_high
                                },
                                description=self.pattern_definitions['head_and_shoulders']['description'],
                                entry_signal=True,
                                target_price=neckline_level - (head_high - neckline_level)
                            ))
            
            # Inverse Head and Shoulders pattern
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left_trough = troughs[i]
                    head = troughs[i + 1]
                    right_trough = troughs[i + 2]
                    
                    left_low = price_data['Low'].iloc[left_trough]
                    head_low = price_data['Low'].iloc[head]
                    right_low = price_data['Low'].iloc[right_trough]
                    
                    # Check inverse H&S criteria
                    if (head_low < left_low * 0.98 and head_low < right_low * 0.98 and
                        abs(left_low - right_low) / max(left_low, right_low) < 0.05):
                        
                        # Find neckline (peaks between troughs)
                        relevant_peaks = [p for p in peaks if left_trough < p < right_trough]
                        if len(relevant_peaks) >= 1:
                            neckline_level = max(price_data['High'].iloc[p] for p in relevant_peaks)
                            
                            confidence = self._calculate_pattern_confidence(
                                'inverse_head_and_shoulders', price_data, left_trough, right_trough
                            )
                            
                            patterns['inverse_head_and_shoulders'].append(ChartPattern(
                                name='inverse_head_and_shoulders',
                                pattern_type='reversal',
                                signal_type='bullish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[left_trough].date()),
                                end_date=str(price_data.index[right_trough].date()),
                                key_levels={
                                    'neckline': neckline_level,
                                    'head': head_low,
                                    'left_shoulder': left_low,
                                    'right_shoulder': right_low
                                },
                                description=self.pattern_definitions['inverse_head_and_shoulders']['description'],
                                entry_signal=True,
                                target_price=neckline_level + (neckline_level - head_low)
                            ))
            
            # Double Top pattern
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    first_peak = peaks[i]
                    second_peak = peaks[i + 1]
                    
                    first_high = price_data['High'].iloc[first_peak]
                    second_high = price_data['High'].iloc[second_peak]
                    
                    # Check double top criteria: roughly equal peaks
                    if abs(first_high - second_high) / max(first_high, second_high) < 0.03:
                        # Find valley between peaks
                        valley_troughs = [t for t in troughs if first_peak < t < second_peak]
                        if valley_troughs:
                            valley_level = min(price_data['Low'].iloc[t] for t in valley_troughs)
                            
                            confidence = self._calculate_pattern_confidence(
                                'double_top', price_data, first_peak, second_peak
                            )
                            
                            patterns['double_top'].append(ChartPattern(
                                name='double_top',
                                pattern_type='reversal',
                                signal_type='bearish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[first_peak].date()),
                                end_date=str(price_data.index[second_peak].date()),
                                key_levels={
                                    'first_peak': first_high,
                                    'second_peak': second_high,
                                    'valley': valley_level
                                },
                                description=self.pattern_definitions['double_top']['description'],
                                entry_signal=True,
                                target_price=valley_level - (first_high - valley_level)
                            ))
            
            # Double Bottom pattern
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    first_trough = troughs[i]
                    second_trough = troughs[i + 1]
                    
                    first_low = price_data['Low'].iloc[first_trough]
                    second_low = price_data['Low'].iloc[second_trough]
                    
                    # Check double bottom criteria: roughly equal troughs
                    if abs(first_low - second_low) / max(first_low, second_low) < 0.03:
                        # Find peak between troughs
                        peak_between = [p for p in peaks if first_trough < p < second_trough]
                        if peak_between:
                            peak_level = max(price_data['High'].iloc[p] for p in peak_between)
                            
                            confidence = self._calculate_pattern_confidence(
                                'double_bottom', price_data, first_trough, second_trough
                            )
                            
                            patterns['double_bottom'].append(ChartPattern(
                                name='double_bottom',
                                pattern_type='reversal',
                                signal_type='bullish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[first_trough].date()),
                                end_date=str(price_data.index[second_trough].date()),
                                key_levels={
                                    'first_bottom': first_low,
                                    'second_bottom': second_low,
                                    'peak': peak_level
                                },
                                description=self.pattern_definitions['double_bottom']['description'],
                                entry_signal=True,
                                target_price=peak_level + (peak_level - first_low)
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {str(e)}")
            return patterns
    
    def _detect_continuation_patterns(self, price_data: pd.DataFrame,
                                    peaks: List[int], troughs: List[int]) -> Dict[str, List[ChartPattern]]:
        """Detect continuation patterns (Flags, Pennants, Wedges)"""
        patterns = {
            'flag_bullish': [],
            'flag_bearish': [],
            'pennant_bullish': [],
            'pennant_bearish': [],
            'wedge_rising': [],
            'wedge_falling': []
        }
        
        try:
            # Analyze recent price action for flags and pennants
            if len(price_data) >= 20:
                for window_start in range(10, len(price_data) - 10):
                    window_data = price_data.iloc[window_start:window_start + 15]
                    
                    if len(window_data) < 10:
                        continue
                    
                    # Check for strong move before pattern (flagpole)
                    pre_move_data = price_data.iloc[max(0, window_start - 10):window_start]
                    if len(pre_move_data) < 5:
                        continue
                    
                    move_strength = abs(pre_move_data['Close'].iloc[-1] - pre_move_data['Close'].iloc[0]) / pre_move_data['Close'].iloc[0]
                    
                    if move_strength > 0.05:  # Significant move (>5%)
                        # Analyze consolidation pattern
                        highs = window_data['High']
                        lows = window_data['Low']
                        
                        # Calculate trend lines
                        x = np.arange(len(highs))
                        high_slope, _, high_r_value, _, _ = stats.linregress(x, highs)
                        low_slope, _, low_r_value, _, _ = stats.linregress(x, lows)
                        
                        # Flag pattern criteria
                        if (abs(high_slope) < 0.1 and abs(low_slope) < 0.1 and  # Horizontal consolidation
                            abs(high_r_value) > 0.7 and abs(low_r_value) > 0.7):  # Good trend line fit
                            
                            if pre_move_data['Close'].iloc[-1] > pre_move_data['Close'].iloc[0]:  # Bullish flag
                                confidence = self._calculate_pattern_confidence(
                                    'flag_bullish', window_data, 0, len(window_data) - 1
                                )
                                
                                patterns['flag_bullish'].append(ChartPattern(
                                    name='flag_bullish',
                                    pattern_type='continuation',
                                    signal_type='bullish',
                                    confidence=confidence,
                                    strength=self._get_strength_from_confidence(confidence),
                                    start_date=str(window_data.index[0].date()),
                                    end_date=str(window_data.index[-1].date()),
                                    key_levels={
                                        'upper_flag': highs.max(),
                                        'lower_flag': lows.min(),
                                        'flagpole_height': abs(pre_move_data['Close'].iloc[-1] - pre_move_data['Close'].iloc[0])
                                    },
                                    description=self.pattern_definitions['flag_bullish']['description'],
                                    entry_signal=True,
                                    target_price=highs.max() + abs(pre_move_data['Close'].iloc[-1] - pre_move_data['Close'].iloc[0])
                                ))
                            
                            else:  # Bearish flag
                                confidence = self._calculate_pattern_confidence(
                                    'flag_bearish', window_data, 0, len(window_data) - 1
                                )
                                
                                patterns['flag_bearish'].append(ChartPattern(
                                    name='flag_bearish',
                                    pattern_type='continuation',
                                    signal_type='bearish',
                                    confidence=confidence,
                                    strength=self._get_strength_from_confidence(confidence),
                                    start_date=str(window_data.index[0].date()),
                                    end_date=str(window_data.index[-1].date()),
                                    key_levels={
                                        'upper_flag': highs.max(),
                                        'lower_flag': lows.min(),
                                        'flagpole_height': abs(pre_move_data['Close'].iloc[-1] - pre_move_data['Close'].iloc[0])
                                    },
                                    description=self.pattern_definitions['flag_bearish']['description'],
                                    entry_signal=True,
                                    target_price=lows.min() - abs(pre_move_data['Close'].iloc[-1] - pre_move_data['Close'].iloc[0])
                                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting continuation patterns: {str(e)}")
            return patterns
    
    def _detect_triangle_patterns(self, price_data: pd.DataFrame,
                                peaks: List[int], troughs: List[int]) -> Dict[str, List[ChartPattern]]:
        """Detect triangle patterns (Ascending, Descending, Symmetrical)"""
        patterns = {
            'ascending_triangle': [],
            'descending_triangle': [],
            'symmetrical_triangle': []
        }
        
        try:
            # Need at least 4 touch points (2 peaks, 2 troughs) for triangles
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Sort by index for chronological analysis
                all_points = sorted([(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs])
                
                # Look for triangle patterns in sliding windows
                for i in range(len(all_points) - 3):
                    window_points = all_points[i:i+4]
                    
                    # Extract peaks and troughs in this window
                    window_peaks = [p[0] for p in window_points if p[1] == 'peak']
                    window_troughs = [p[0] for p in window_points if p[1] == 'trough']
                    
                    if len(window_peaks) >= 2 and len(window_troughs) >= 2:
                        # Calculate trend lines
                        peak_highs = [price_data['High'].iloc[p] for p in window_peaks]
                        peak_indices = list(range(len(peak_highs)))
                        
                        trough_lows = [price_data['Low'].iloc[t] for t in window_troughs]
                        trough_indices = list(range(len(trough_lows)))
                        
                        # Resistance trend line (peaks)
                        if len(peak_highs) >= 2:
                            resistance_slope, _, resistance_r, _, _ = stats.linregress(peak_indices, peak_highs)
                        else:
                            continue
                        
                        # Support trend line (troughs)
                        if len(trough_lows) >= 2:
                            support_slope, _, support_r, _, _ = stats.linregress(trough_indices, trough_lows)
                        else:
                            continue
                        
                        # Pattern classification
                        pattern_start = min(window_peaks + window_troughs)
                        pattern_end = max(window_peaks + window_troughs)
                        
                        # Ascending Triangle: horizontal resistance, rising support
                        if (abs(resistance_slope) < 0.1 and support_slope > 0.1 and 
                            abs(resistance_r) > 0.6 and abs(support_r) > 0.6):
                            
                            confidence = self._calculate_pattern_confidence(
                                'ascending_triangle', price_data, pattern_start, pattern_end
                            )
                            
                            patterns['ascending_triangle'].append(ChartPattern(
                                name='ascending_triangle',
                                pattern_type='continuation',
                                signal_type='bullish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[pattern_start].date()),
                                end_date=str(price_data.index[pattern_end].date()),
                                key_levels={
                                    'resistance': max(peak_highs),
                                    'support_start': min(trough_lows),
                                    'support_end': trough_lows[-1] if len(trough_lows) > 1 else trough_lows[0]
                                },
                                description=self.pattern_definitions['ascending_triangle']['description'],
                                entry_signal=True,
                                target_price=max(peak_highs) + (max(peak_highs) - min(trough_lows)) * 0.5
                            ))
                        
                        # Descending Triangle: declining resistance, horizontal support
                        elif (resistance_slope < -0.1 and abs(support_slope) < 0.1 and
                              abs(resistance_r) > 0.6 and abs(support_r) > 0.6):
                            
                            confidence = self._calculate_pattern_confidence(
                                'descending_triangle', price_data, pattern_start, pattern_end
                            )
                            
                            patterns['descending_triangle'].append(ChartPattern(
                                name='descending_triangle',
                                pattern_type='continuation',
                                signal_type='bearish',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[pattern_start].date()),
                                end_date=str(price_data.index[pattern_end].date()),
                                key_levels={
                                    'support': min(trough_lows),
                                    'resistance_start': max(peak_highs),
                                    'resistance_end': peak_highs[-1] if len(peak_highs) > 1 else peak_highs[0]
                                },
                                description=self.pattern_definitions['descending_triangle']['description'],
                                entry_signal=True,
                                target_price=min(trough_lows) - (max(peak_highs) - min(trough_lows)) * 0.5
                            ))
                        
                        # Symmetrical Triangle: converging trend lines
                        elif (resistance_slope < -0.05 and support_slope > 0.05 and
                              abs(resistance_r) > 0.6 and abs(support_r) > 0.6):
                            
                            confidence = self._calculate_pattern_confidence(
                                'symmetrical_triangle', price_data, pattern_start, pattern_end
                            )
                            
                            patterns['symmetrical_triangle'].append(ChartPattern(
                                name='symmetrical_triangle',
                                pattern_type='breakout',
                                signal_type='neutral',
                                confidence=confidence,
                                strength=self._get_strength_from_confidence(confidence),
                                start_date=str(price_data.index[pattern_start].date()),
                                end_date=str(price_data.index[pattern_end].date()),
                                key_levels={
                                    'apex_resistance': peak_highs[-1] if len(peak_highs) > 1 else peak_highs[0],
                                    'apex_support': trough_lows[-1] if len(trough_lows) > 1 else trough_lows[0],
                                    'base_width': max(peak_highs) - min(trough_lows)
                                },
                                description=self.pattern_definitions['symmetrical_triangle']['description'],
                                entry_signal=False,
                                target_price=None
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {str(e)}")
            return patterns
    
    def _calculate_pattern_confidence(self, pattern_name: str, price_data: pd.DataFrame,
                                    start_idx: int, end_idx: int) -> float:
        """Calculate pattern confidence based on various factors"""
        try:
            base_confidence = self.pattern_definitions[pattern_name]['base_confidence']
            
            # Volume confirmation factor
            volume_factor = 1.0
            if 'Volume' in price_data.columns:
                pattern_volume = price_data['Volume'].iloc[start_idx:end_idx+1].mean()
                avg_volume = price_data['Volume'].rolling(20).mean().iloc[end_idx]
                if pattern_volume > avg_volume * 1.1:
                    volume_factor = 1.1
                elif pattern_volume < avg_volume * 0.9:
                    volume_factor = 0.95
            
            # Time factor (longer patterns generally more reliable)
            time_span = end_idx - start_idx
            time_factor = 1.0
            if time_span > 20:  # > 20 days
                time_factor = 1.05
            elif time_span < 10:  # < 10 days
                time_factor = 0.95
            
            # Price action quality factor
            pattern_data = price_data.iloc[start_idx:end_idx+1]
            volatility = pattern_data['Close'].pct_change().std()
            avg_volatility = price_data['Close'].pct_change().rolling(30).std().iloc[end_idx]
            
            volatility_factor = 1.0
            if volatility < avg_volatility * 0.8:  # Lower volatility = cleaner pattern
                volatility_factor = 1.05
            elif volatility > avg_volatility * 1.3:  # Higher volatility = messier pattern
                volatility_factor = 0.9
            
            # Calculate final confidence
            final_confidence = base_confidence * volume_factor * time_factor * volatility_factor
            
            return max(0, min(100, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {str(e)}")
            return self.pattern_definitions[pattern_name]['base_confidence']
    
    def _calculate_support_resistance(self, price_data: pd.DataFrame,
                                    peaks: List[int], troughs: List[int]) -> List[Dict[str, Any]]:
        """Calculate key support and resistance levels"""
        try:
            levels = []
            
            # Resistance levels from peaks
            if peaks:
                peak_prices = [price_data['High'].iloc[p] for p in peaks]
                # Group similar levels
                resistance_levels = self._group_similar_levels(peak_prices)
                for level in resistance_levels:
                    levels.append({
                        'type': 'resistance',
                        'price': level['price'],
                        'strength': level['count'],
                        'confidence': min(90, level['count'] * 20)
                    })
            
            # Support levels from troughs
            if troughs:
                trough_prices = [price_data['Low'].iloc[t] for t in troughs]
                # Group similar levels
                support_levels = self._group_similar_levels(trough_prices)
                for level in support_levels:
                    levels.append({
                        'type': 'support',
                        'price': level['price'],
                        'strength': level['count'],
                        'confidence': min(90, level['count'] * 20)
                    })
            
            # Sort by confidence
            levels.sort(key=lambda x: x['confidence'], reverse=True)
            return levels[:10]  # Return top 10 levels
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return []
    
    def _group_similar_levels(self, prices: List[float], tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """Group similar price levels together"""
        try:
            if not prices:
                return []
            
            groups = []
            sorted_prices = sorted(prices)
            
            i = 0
            while i < len(sorted_prices):
                group_prices = [sorted_prices[i]]
                j = i + 1
                
                # Find similar prices within tolerance
                while j < len(sorted_prices):
                    if abs(sorted_prices[j] - sorted_prices[i]) / sorted_prices[i] <= tolerance:
                        group_prices.append(sorted_prices[j])
                        j += 1
                    else:
                        break
                
                # Create group
                groups.append({
                    'price': np.mean(group_prices),
                    'count': len(group_prices),
                    'min_price': min(group_prices),
                    'max_price': max(group_prices)
                })
                
                i = j
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping similar levels: {str(e)}")
            return []
    
    def _analyze_trend_context(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend context"""
        try:
            closes = price_data['Close']
            
            # Short, medium, long term trends
            short_trend = (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10] if len(closes) >= 10 else 0
            medium_trend = (closes.iloc[-1] - closes.iloc[-30]) / closes.iloc[-30] if len(closes) >= 30 else 0
            long_trend = (closes.iloc[-1] - closes.iloc[-60]) / closes.iloc[-60] if len(closes) >= 60 else 0
            
            # Trend strength
            short_strength = 'strong' if abs(short_trend) > 0.1 else 'moderate' if abs(short_trend) > 0.05 else 'weak'
            medium_strength = 'strong' if abs(medium_trend) > 0.2 else 'moderate' if abs(medium_trend) > 0.1 else 'weak'
            long_strength = 'strong' if abs(long_trend) > 0.3 else 'moderate' if abs(long_trend) > 0.15 else 'weak'
            
            return {
                'short_term': {
                    'direction': 'bullish' if short_trend > 0 else 'bearish',
                    'strength': short_strength,
                    'change_pct': short_trend * 100
                },
                'medium_term': {
                    'direction': 'bullish' if medium_trend > 0 else 'bearish',
                    'strength': medium_strength,
                    'change_pct': medium_trend * 100
                },
                'long_term': {
                    'direction': 'bullish' if long_trend > 0 else 'bearish',
                    'strength': long_strength,
                    'change_pct': long_trend * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend context: {str(e)}")
            return {}
    
    def _get_strength_from_confidence(self, confidence: float) -> str:
        """Convert confidence score to strength category"""
        if confidence >= 85:
            return 'strong'
        elif confidence >= 70:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_pattern_insights(self, patterns: Dict[str, List], summary: Dict[str, Any],
                                 price_data: pd.DataFrame) -> List[str]:
        """Generate actionable insights from pattern analysis"""
        insights = []
        
        try:
            if summary['total_patterns'] == 0:
                insights.append("📊 No significant chart patterns detected in current analysis period")
                return insights
            
            # Pattern type distribution
            reversal_count = summary['reversal_patterns']
            continuation_count = summary['continuation_patterns']
            breakout_count = summary['breakout_patterns']
            
            if reversal_count > continuation_count + breakout_count:
                insights.append(f"🔄 {reversal_count} reversal patterns detected - potential trend change signals")
            elif continuation_count > reversal_count + breakout_count:
                insights.append(f"➡️ {continuation_count} continuation patterns detected - trend likely to persist")
            elif breakout_count > 0:
                insights.append(f"📈 {breakout_count} breakout patterns detected - prepare for directional move")
            
            # High confidence patterns
            high_conf = summary['high_confidence_patterns']
            if high_conf > 0:
                insights.append(f"⭐ {high_conf} high-confidence patterns (>80%) - strong technical signals")
            
            # Support/resistance insights
            sr_levels = summary.get('support_resistance_levels', [])
            if sr_levels:
                strongest_level = max(sr_levels, key=lambda x: x['confidence'])
                level_type = strongest_level['type']
                level_price = strongest_level['price']
                current_price = price_data['Close'].iloc[-1]
                distance_pct = abs(current_price - level_price) / current_price * 100
                
                insights.append(f"🎯 Strong {level_type} at ${level_price:.2f} ({distance_pct:.1f}% from current price)")
            
            # Trend context insights
            trend_analysis = summary.get('trend_analysis', {})
            if trend_analysis:
                short_term = trend_analysis.get('short_term', {})
                if short_term:
                    direction = short_term.get('direction', 'neutral')
                    strength = short_term.get('strength', 'weak')
                    insights.append(f"📊 Short-term trend: {strength} {direction} momentum")
            
        except Exception as e:
            logger.error(f"Error generating pattern insights: {str(e)}")
            insights.append("⚠️ Unable to generate pattern insights due to analysis limitations")
        
        return insights
