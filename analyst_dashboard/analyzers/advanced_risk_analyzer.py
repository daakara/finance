"""
Advanced Risk Analysis Module
Sophisticated risk metrics beyond basic volatility and Sharpe ratio
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedRiskAnalyzer:
    """Advanced risk analysis capabilities for professional-grade insights"""
    
    def analyze_comprehensive_risk(self, price_data: pd.DataFrame, 
                                 benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            risk_analysis = {
                'basic_metrics': self._calculate_basic_risk_metrics(returns),
                'advanced_metrics': self._calculate_advanced_risk_metrics(returns),
                'tail_risk': self._analyze_tail_risk(returns),
                'drawdown_analysis': self._analyze_drawdowns(price_data['Close']),
                'regime_analysis': self._detect_market_regimes(returns),
                'risk_attribution': self._attribute_risk_sources(returns)
            }
            
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                risk_analysis['relative_risk'] = self._analyze_relative_risk(
                    returns, benchmark_returns
                )
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_advanced_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate sophisticated risk metrics"""
        try:
            metrics = {}
            
            # Value at Risk (VaR) at different confidence levels
            metrics['VaR_95'] = np.percentile(returns, 5) * 100
            metrics['VaR_99'] = np.percentile(returns, 1) * 100
            
            # Conditional Value at Risk (Expected Shortfall)
            metrics['CVaR_95'] = returns[returns <= np.percentile(returns, 5)].mean() * 100
            metrics['CVaR_99'] = returns[returns <= np.percentile(returns, 1)].mean() * 100
            
            # Tail Ratio (95th percentile / 5th percentile)
            metrics['Tail_Ratio'] = abs(np.percentile(returns, 95) / np.percentile(returns, 5))
            
            # Skewness and Kurtosis
            metrics['Skewness'] = stats.skew(returns.dropna())
            metrics['Kurtosis'] = stats.kurtosis(returns.dropna())
            
            # Calmar Ratio (Annual Return / Max Drawdown)
            annual_return = returns.mean() * 252
            max_dd = self._calculate_max_drawdown(returns)
            metrics['Calmar_Ratio'] = annual_return / abs(max_dd) if max_dd != 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            metrics['Sortino_Ratio'] = (annual_return * 100) / (downside_std * 100) if downside_std != 0 else 0
            
            # Omega Ratio
            threshold = 0  # Risk-free rate
            excess_returns = returns - threshold
            positive_returns = excess_returns[excess_returns > 0].sum()
            negative_returns = abs(excess_returns[excess_returns < 0].sum())
            metrics['Omega_Ratio'] = positive_returns / negative_returns if negative_returns != 0 else np.inf
            
            # Pain Ratio
            pain_index = self._calculate_pain_index(returns)
            metrics['Pain_Ratio'] = annual_return / pain_index if pain_index != 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {str(e)}")
            return {}
    
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
            # Rolling volatility
            rolling_vol = returns.rolling(30).std() * np.sqrt(252)
            
            # Rolling returns
            rolling_returns = returns.rolling(30).mean() * 252
            
            # Define regime thresholds
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
        except:
            return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            return drawdowns.min()
        except:
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
