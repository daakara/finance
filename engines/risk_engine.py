"""
Risk Analysis Engine - Focused on risk metrics and portfolio risk assessment
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from analysis_components import RiskMetrics, DataProcessors

logger = logging.getLogger(__name__)

class RiskAnalysisEngine:
    """Lightweight risk analysis engine using modular components."""
    
    def __init__(self):
        """Initialize the risk analysis engine."""
        self.risk_metrics = RiskMetrics()
        self.processors = DataProcessors()
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information (optional)
        
        Returns:
            Dictionary with risk analysis results
        """
        if price_data.empty:
            return {'error': 'No price data available'}
        
        try:
            results = {}
            
            # Calculate returns
            returns = self.processors.calculate_returns(price_data)
            
            # Volatility analysis
            results['volatility_metrics'] = self._analyze_volatility(returns, price_data)
            
            # Drawdown analysis
            results['drawdown_analysis'] = self._analyze_drawdowns(price_data)
            
            # Value at Risk analysis
            results['var_analysis'] = self._analyze_var(returns)
            
            # Risk-adjusted returns
            results['risk_adjusted_returns'] = self._calculate_risk_adjusted_returns(returns)
            
            # Tail risk metrics
            results['tail_risk_metrics'] = self._analyze_tail_risk(returns)
            
            # Overall risk assessment
            results['risk_assessment'] = self._assess_overall_risk(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Risk analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_volatility(self, returns: pd.Series, price_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze various volatility metrics."""
        try:
            # Basic volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252) * 100
            
            # Rolling volatility
            rolling_vol = self.processors.calculate_rolling_volatility(returns, 20)
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else annual_vol
            
            # Volatility percentiles
            vol_percentile_25 = rolling_vol.quantile(0.25) if not rolling_vol.empty else annual_vol * 0.8
            vol_percentile_75 = rolling_vol.quantile(0.75) if not rolling_vol.empty else annual_vol * 1.2
            
            # Volatility regime
            if current_vol > vol_percentile_75:
                vol_regime = 'High Volatility'
            elif current_vol < vol_percentile_25:
                vol_regime = 'Low Volatility'
            else:
                vol_regime = 'Normal Volatility'
            
            return {
                'daily_volatility': daily_vol * 100,
                'annual_volatility': annual_vol,
                'current_volatility': current_vol,
                'volatility_regime': vol_regime,
                'vol_25th_percentile': vol_percentile_25,
                'vol_75th_percentile': vol_percentile_75
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {'annual_volatility': 20.0, 'volatility_regime': 'Unknown'}
    
    def _analyze_drawdowns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown metrics."""
        try:
            prices = price_data['Close']
            drawdown_series = self.processors.calculate_drawdown(price_data)
            
            # Maximum drawdown
            max_dd_info = self.risk_metrics.calculate_max_drawdown(self.processors.calculate_returns(price_data))
            max_drawdown = max_dd_info['max_drawdown'] * 100
            max_dd_duration = max_dd_info['max_drawdown_duration']
            
            # Current drawdown
            current_drawdown = drawdown_series.iloc[-1] * 100 if not drawdown_series.empty else 0
            
            # Average drawdown
            negative_drawdowns = drawdown_series[drawdown_series < 0]
            avg_drawdown = negative_drawdowns.mean() * 100 if not negative_drawdowns.empty else 0
            
            # Recovery analysis
            is_in_drawdown = current_drawdown < -1  # More than 1% drawdown
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_dd_duration,
                'current_drawdown': current_drawdown,
                'average_drawdown': avg_drawdown,
                'is_in_drawdown': is_in_drawdown,
                'drawdown_frequency': len(negative_drawdowns) / len(drawdown_series) if len(drawdown_series) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {e}")
            return {'max_drawdown': -15.0, 'current_drawdown': 0.0}
    
    def _analyze_var(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze Value at Risk metrics."""
        try:
            # Different confidence levels
            var_1 = self.risk_metrics.calculate_var(returns, 0.01) * 100
            var_5 = self.risk_metrics.calculate_var(returns, 0.05) * 100
            
            # Conditional VaR (Expected Shortfall)
            cvar_5 = self.risk_metrics.calculate_cvar(returns, 0.05) * 100
            
            return {
                'var_1_percent': var_1,
                'var_5_percent': var_5,
                'cvar_5_percent': cvar_5,
                'var_interpretation': self._interpret_var(var_5)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing VaR: {e}")
            return {'var_5_percent': -5.0, 'cvar_5_percent': -8.0}
    
    def _calculate_risk_adjusted_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        try:
            # Sharpe ratio
            sharpe_ratio = self.risk_metrics.calculate_sharpe_ratio(returns)
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            annual_return = returns.mean() * 252
            sortino_ratio = (annual_return - 0.02) / downside_std if downside_std > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            max_dd_info = self.risk_metrics.calculate_max_drawdown(returns)
            max_dd = abs(max_dd_info['max_drawdown'])
            calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'annual_return': annual_return * 100,
                'risk_adjusted_quality': self._assess_risk_adjusted_quality(sharpe_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {'sharpe_ratio': 0.5, 'annual_return': 8.0}
    
    def _analyze_tail_risk(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze tail risk characteristics."""
        try:
            # Skewness and kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Tail ratio (95th percentile / 5th percentile)
            p95 = returns.quantile(0.95)
            p5 = returns.quantile(0.05)
            tail_ratio = abs(p95 / p5) if p5 != 0 else 1
            
            # Extreme value analysis
            extreme_losses = returns[returns < returns.quantile(0.01)]
            extreme_gains = returns[returns > returns.quantile(0.99)]
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_ratio': tail_ratio,
                'extreme_loss_avg': extreme_losses.mean() * 100 if not extreme_losses.empty else 0,
                'extreme_gain_avg': extreme_gains.mean() * 100 if not extreme_gains.empty else 0,
                'tail_risk_assessment': self._assess_tail_risk(skewness, kurtosis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tail risk: {e}")
            return {'skewness': 0, 'kurtosis': 3, 'tail_risk_assessment': 'Normal'}
    
    def _assess_overall_risk(self, risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall risk assessment."""
        try:
            # Extract key metrics
            volatility = risk_results.get('volatility_metrics', {}).get('annual_volatility', 20)
            max_drawdown = abs(risk_results.get('drawdown_analysis', {}).get('max_drawdown', 15))
            sharpe_ratio = risk_results.get('risk_adjusted_returns', {}).get('sharpe_ratio', 0.5)
            var_5 = abs(risk_results.get('var_analysis', {}).get('var_5_percent', 5))
            
            # Risk scoring (0-100, lower is better)
            vol_score = min(volatility / 30 * 100, 100)  # 30% vol = 100 points
            dd_score = min(max_drawdown / 50 * 100, 100)  # 50% DD = 100 points
            sharpe_score = max(0, 100 - sharpe_ratio * 50)  # Higher Sharpe = lower risk score
            var_score = min(var_5 * 10, 100)  # 10% VaR = 100 points
            
            overall_score = (vol_score + dd_score + sharpe_score + var_score) / 4
            
            # Risk level classification
            if overall_score < 30:
                risk_level = 'Low'
            elif overall_score < 60:
                risk_level = 'Moderate'
            elif overall_score < 80:
                risk_level = 'High'
            else:
                risk_level = 'Very High'
            
            return {
                'overall_risk_score': overall_score,
                'risk_level': risk_level,
                'key_risk_factors': self._identify_key_risk_factors(risk_results),
                'risk_recommendation': self._generate_risk_recommendation(risk_level, overall_score)
            }
            
        except Exception as e:
            logger.error(f"Error in overall risk assessment: {e}")
            return {'risk_level': 'Moderate', 'overall_risk_score': 50}
    
    def _interpret_var(self, var_5: float) -> str:
        """Interpret VaR values."""
        if var_5 > -2:
            return 'Low risk - small potential losses'
        elif var_5 > -5:
            return 'Moderate risk - manageable potential losses'
        elif var_5 > -10:
            return 'High risk - significant potential losses'
        else:
            return 'Very high risk - severe potential losses'
    
    def _assess_risk_adjusted_quality(self, sharpe_ratio: float) -> str:
        """Assess quality of risk-adjusted returns."""
        if sharpe_ratio > 1.5:
            return 'Excellent'
        elif sharpe_ratio > 1.0:
            return 'Good'
        elif sharpe_ratio > 0.5:
            return 'Fair'
        else:
            return 'Poor'
    
    def _assess_tail_risk(self, skewness: float, kurtosis: float) -> str:
        """Assess tail risk based on distribution characteristics."""
        if abs(skewness) > 1 or kurtosis > 5:
            return 'High tail risk'
        elif abs(skewness) > 0.5 or kurtosis > 3.5:
            return 'Moderate tail risk'
        else:
            return 'Normal tail risk'
    
    def _identify_key_risk_factors(self, risk_results: Dict[str, Any]) -> List[str]:
        """Identify the main risk factors."""
        risk_factors = []
        
        try:
            # Check volatility
            vol_regime = risk_results.get('volatility_metrics', {}).get('volatility_regime')
            if vol_regime == 'High Volatility':
                risk_factors.append('High volatility environment')
            
            # Check drawdowns
            current_dd = risk_results.get('drawdown_analysis', {}).get('current_drawdown', 0)
            if current_dd < -10:
                risk_factors.append('Currently in significant drawdown')
            
            # Check Sharpe ratio
            sharpe = risk_results.get('risk_adjusted_returns', {}).get('sharpe_ratio', 0.5)
            if sharpe < 0.5:
                risk_factors.append('Poor risk-adjusted returns')
            
            # Check tail risk
            tail_assessment = risk_results.get('tail_risk_metrics', {}).get('tail_risk_assessment')
            if 'High' in str(tail_assessment):
                risk_factors.append('Elevated tail risk')
            
            return risk_factors if risk_factors else ['No major risk factors identified']
            
        except Exception as e:
            return ['Risk factor analysis incomplete']
    
    def _generate_risk_recommendation(self, risk_level: str, risk_score: float) -> str:
        """Generate risk management recommendation."""
        recommendations = {
            'Low': 'Risk profile is manageable. Continue monitoring key metrics.',
            'Moderate': 'Balanced risk profile. Consider position sizing and diversification.',
            'High': 'Elevated risk detected. Implement risk management strategies and consider reducing exposure.',
            'Very High': 'Significant risk concerns. Immediate risk reduction measures recommended.'
        }
        
        base_rec = recommendations.get(risk_level, 'Monitor risk metrics regularly.')
        
        if risk_score > 70:
            base_rec += ' Consider defensive positioning and hedging strategies.'
        
        return base_rec


# Global instance
risk_engine = RiskAnalysisEngine()
