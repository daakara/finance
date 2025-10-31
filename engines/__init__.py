"""
Engine Module - Lightweight engine orchestrator
Imports and exposes all analysis engines with global instances
"""

import numpy as np
from .technical_engine import technical_engine, TechnicalAnalysisEngine
from .risk_engine import risk_engine, RiskAnalysisEngine  
from .fundamental_engine import fundamental_engine, FundamentalAnalysisEngine

# Create additional lightweight engines for compatibility
class PerformanceAnalysisEngine:
    """Simple performance analysis engine."""
    
    def analyze(self, price_data, asset_info=None):
        """Calculate basic performance metrics."""
        try:
            if price_data.empty:
                return {'error': 'No price data available'}
            
            returns = price_data['Close'].pct_change().dropna()
            
            # Basic performance metrics
            total_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0] - 1) * 100
            annualized_return = ((1 + total_return/100) ** (252/len(price_data)) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': annualized_return / volatility if volatility > 0 else 0,
                'max_drawdown': ((price_data['Close'] / price_data['Close'].expanding().max()) - 1).min() * 100
            }
            
        except Exception as e:
            return {'error': f'Performance analysis failed: {e}'}

class MacroeconomicAnalysisEngine:
    """Simple macro analysis engine."""
    
    def analyze(self, price_data, asset_info=None):
        """Basic macro context analysis."""
        try:
            from analysis_components import MarketDataFetchers
            
            # Get economic indicators
            indicators = MarketDataFetchers.fetch_economic_indicators()
            
            # Simple correlation analysis
            returns = price_data['Close'].pct_change().dropna()
            
            return {
                'economic_indicators': indicators,
                'interest_rate_environment': 'Rising' if indicators.get('treasury_10y', 4) > 4 else 'Stable',
                'macro_sentiment': 'Risk-On' if indicators.get('vix', 20) < 20 else 'Risk-Off',
                'currency_strength': 'Strong USD' if indicators.get('dxy', 103) > 105 else 'Weak USD'
            }
            
        except Exception as e:
            return {'error': f'Macro analysis failed: {e}'}

class PortfolioStrategyEngine:
    """Simple portfolio strategy engine."""
    
    def analyze(self, price_data, asset_info=None):
        """Basic portfolio allocation suggestions."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Simple allocation based on volatility
            if volatility < 0.15:
                risk_profile = 'Low Risk'
                suggested_allocation = {'stocks': 0.8, 'bonds': 0.2}
            elif volatility < 0.25:
                risk_profile = 'Medium Risk'
                suggested_allocation = {'stocks': 0.6, 'bonds': 0.3, 'alternatives': 0.1}
            else:
                risk_profile = 'High Risk'
                suggested_allocation = {'stocks': 0.4, 'bonds': 0.4, 'cash': 0.2}
            
            return {
                'risk_profile': risk_profile,
                'suggested_allocation': suggested_allocation,
                'rebalancing_frequency': 'Quarterly',
                'portfolio_volatility': volatility * 100
            }
            
        except Exception as e:
            return {'error': f'Portfolio analysis failed: {e}'}

class ForecastingEngine:
    """Simple forecasting engine."""
    
    def analyze(self, price_data, asset_info=None):
        """Basic price forecasting."""
        try:
            from analysis_components import SimpleForecasting
            
            prices = price_data['Close']
            
            # Generate different forecasts
            ma_forecast = SimpleForecasting.moving_average_forecast(prices)
            trend_forecast = SimpleForecasting.linear_trend_forecast(prices)
            
            return {
                'forecast_horizon': '30 days',
                'moving_average_forecast': ma_forecast[-1],
                'trend_forecast': trend_forecast[-1],
                'confidence_level': 'Medium',
                'forecast_accuracy': 'Historical accuracy varies with market conditions'
            }
            
        except Exception as e:
            return {'error': f'Forecasting failed: {e}'}

# Create global instances
performance_engine = PerformanceAnalysisEngine()
macro_engine = MacroeconomicAnalysisEngine()
portfolio_engine = PortfolioStrategyEngine()
forecasting_engine = ForecastingEngine()

# Export all engines
__all__ = [
    'technical_engine', 'TechnicalAnalysisEngine',
    'risk_engine', 'RiskAnalysisEngine', 
    'fundamental_engine', 'FundamentalAnalysisEngine',
    'performance_engine', 'PerformanceAnalysisEngine',
    'macro_engine', 'MacroeconomicAnalysisEngine',
    'portfolio_engine', 'PortfolioStrategyEngine',
    'forecasting_engine', 'ForecastingEngine'
]
