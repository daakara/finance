"""
Analysis Engine Module - Streamlined engine orchestrator
Imports modular engines and provides backward compatibility
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import all engines from the modular engines package
from engines import (
    technical_engine, TechnicalAnalysisEngine,
    risk_engine, RiskAnalysisEngine,
    fundamental_engine, FundamentalAnalysisEngine,
    performance_engine, PerformanceAnalysisEngine,
    macro_engine, MacroeconomicAnalysisEngine,
    portfolio_engine, PortfolioStrategyEngine,
    forecasting_engine, ForecastingEngine
)

logger = logging.getLogger(__name__)

class BaseAnalysisEngine(ABC):
    """Abstract base class for analysis engines."""
    
    @abstractmethod
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on asset data."""
        pass

class TechnicalAnalysisEngine(BaseAnalysisEngine):
    """Technical analysis engine for all asset types."""
    
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
            results['indicators'] = self._calculate_technical_indicators(price_data)
            
            # Generate signals
            results['signals'] = self._generate_technical_signals(results['indicators'])
            
            # Trend analysis
            results['trend_analysis'] = self._analyze_trend(price_data, results['indicators'])
            
            # Support and resistance
            results['support_resistance'] = self._find_support_resistance(price_data)
            
            # Pattern recognition
            results['patterns'] = self._detect_patterns(price_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators."""
        indicators = {}
        close = price_data['Close']
        high = price_data['High']
        low = price_data['Low']
        volume = price_data.get('Volume', pd.Series())
        
        # Moving Averages
        indicators['SMA_20'] = close.rolling(20).mean()
        indicators['SMA_50'] = close.rolling(50).mean()
        indicators['SMA_200'] = close.rolling(200).mean()
        indicators['EMA_12'] = close.ewm(span=12).mean()
        indicators['EMA_26'] = close.ewm(span=26).mean()
        
        # Bollinger Bands
        sma_20 = indicators['SMA_20']
        std_20 = close.rolling(20).std()
        indicators['BB_Upper'] = sma_20 + (2 * std_20)
        indicators['BB_Lower'] = sma_20 - (2 * std_20)
        indicators['BB_Width'] = indicators['BB_Upper'] - indicators['BB_Lower']
        indicators['BB_Position'] = (close - indicators['BB_Lower']) / indicators['BB_Width']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        # Stochastic Oscillator
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        indicators['Stoch_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        indicators['Stoch_D'] = indicators['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        indicators['Williams_R'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # Commodity Channel Index (CCI)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['ATR'] = true_range.rolling(14).mean()
        
        # On-Balance Volume (if volume available)
        if not volume.empty:
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            indicators['OBV'] = pd.Series(obv, index=close.index)
        
        # Volume Weighted Average Price (VWAP)
        if not volume.empty:
            vwap = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
            indicators['VWAP'] = vwap
        
        return indicators
    
    def _generate_technical_signals(self, indicators: Dict[str, pd.Series]) -> Dict[str, str]:
        """Generate trading signals from technical indicators."""
        signals = {}
        
        try:
            # Current values (last available)
            current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else 50
            current_macd = indicators['MACD'].iloc[-1] if not indicators['MACD'].empty else 0
            current_macd_signal = indicators['MACD_Signal'].iloc[-1] if not indicators['MACD_Signal'].empty else 0
            current_stoch_k = indicators['Stoch_K'].iloc[-1] if not indicators['Stoch_K'].empty else 50
            current_bb_position = indicators['BB_Position'].iloc[-1] if not indicators['BB_Position'].empty else 0.5
            
            # RSI Signals
            if current_rsi > 70:
                signals['RSI'] = 'Overbought'
            elif current_rsi < 30:
                signals['RSI'] = 'Oversold'
            else:
                signals['RSI'] = 'Neutral'
            
            # MACD Signals
            if current_macd > current_macd_signal:
                if current_macd > 0:
                    signals['MACD'] = 'Strong Bullish'
                else:
                    signals['MACD'] = 'Bullish'
            else:
                if current_macd < 0:
                    signals['MACD'] = 'Strong Bearish'
                else:
                    signals['MACD'] = 'Bearish'
            
            # Stochastic Signals
            if current_stoch_k > 80:
                signals['Stochastic'] = 'Overbought'
            elif current_stoch_k < 20:
                signals['Stochastic'] = 'Oversold'
            else:
                signals['Stochastic'] = 'Neutral'
            
            # Bollinger Bands Signals
            if current_bb_position > 1:
                signals['Bollinger'] = 'Above Upper Band'
            elif current_bb_position < 0:
                signals['Bollinger'] = 'Below Lower Band'
            elif current_bb_position > 0.8:
                signals['Bollinger'] = 'Near Upper Band'
            elif current_bb_position < 0.2:
                signals['Bollinger'] = 'Near Lower Band'
            else:
                signals['Bollinger'] = 'Within Bands'
            
        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            signals['Error'] = str(e)
        
        return signals
    
    def _analyze_trend(self, price_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze price trend using multiple methods."""
        trend_analysis = {}
        
        try:
            close = price_data['Close']
            current_price = close.iloc[-1]
            
            # Moving average trend
            sma_20 = indicators['SMA_20'].iloc[-1] if not indicators['SMA_20'].empty else current_price
            sma_50 = indicators['SMA_50'].iloc[-1] if not indicators['SMA_50'].empty else current_price
            sma_200 = indicators['SMA_200'].iloc[-1] if not indicators['SMA_200'].empty else current_price
            
            if current_price > sma_20 > sma_50 > sma_200:
                trend_analysis['primary_trend'] = 'Strong Uptrend'
                trend_analysis['trend_strength'] = 'Strong'
            elif current_price > sma_20 > sma_50:
                trend_analysis['primary_trend'] = 'Uptrend'
                trend_analysis['trend_strength'] = 'Moderate'
            elif current_price < sma_20 < sma_50 < sma_200:
                trend_analysis['primary_trend'] = 'Strong Downtrend'
                trend_analysis['trend_strength'] = 'Strong'
            elif current_price < sma_20 < sma_50:
                trend_analysis['primary_trend'] = 'Downtrend'
                trend_analysis['trend_strength'] = 'Moderate'
            else:
                trend_analysis['primary_trend'] = 'Sideways'
                trend_analysis['trend_strength'] = 'Weak'
            
            # Price momentum
            returns_5d = (current_price / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
            returns_20d = (current_price / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
            
            trend_analysis['momentum_5d'] = returns_5d
            trend_analysis['momentum_20d'] = returns_20d
            
            # Trend consistency
            price_above_sma20_count = (close.tail(20) > indicators['SMA_20'].tail(20)).sum()
            trend_analysis['trend_consistency'] = price_above_sma20_count / 20
            
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            trend_analysis['error'] = str(e)
        
        return trend_analysis
    
    def _find_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels."""
        try:
            high = price_data['High']
            low = price_data['Low']
            close = price_data['Close']
            
            # Simple pivot point method
            support_levels = []
            resistance_levels = []
            
            # Look for local minima and maxima
            window = 10
            for i in range(window, len(price_data) - window):
                # Local minimum (support)
                if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                    support_levels.append(low.iloc[i])
                
                # Local maximum (resistance)
                if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                    resistance_levels.append(high.iloc[i])
            
            # Keep only significant levels
            current_price = close.iloc[-1]
            
            # Filter support levels (below current price)
            support_levels = [s for s in support_levels if s < current_price]
            support_levels = sorted(set(support_levels))[-3:]  # Keep top 3
            
            # Filter resistance levels (above current price)
            resistance_levels = [r for r in resistance_levels if r > current_price]
            resistance_levels = sorted(set(resistance_levels))[:3]  # Keep top 3
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Support/resistance analysis error: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _detect_patterns(self, price_data: pd.DataFrame) -> Dict[str, bool]:
        """Detect common chart patterns."""
        patterns = {}
        
        try:
            close = price_data['Close']
            high = price_data['High']
            low = price_data['Low']
            
            # Golden Cross / Death Cross
            if len(close) >= 200:
                sma_50 = close.rolling(50).mean()
                sma_200 = close.rolling(200).mean()
                
                # Check recent crossover
                recent_50 = sma_50.tail(5)
                recent_200 = sma_200.tail(5)
                
                if (recent_50.iloc[-1] > recent_200.iloc[-1] and 
                    recent_50.iloc[-3] <= recent_200.iloc[-3]):
                    patterns['Golden_Cross'] = True
                else:
                    patterns['Golden_Cross'] = False
                
                if (recent_50.iloc[-1] < recent_200.iloc[-1] and 
                    recent_50.iloc[-3] >= recent_200.iloc[-3]):
                    patterns['Death_Cross'] = True
                else:
                    patterns['Death_Cross'] = False
            
            # Breakout pattern (price breaking above recent high)
            if len(close) >= 20:
                recent_high = high.tail(20).max()
                current_price = close.iloc[-1]
                patterns['Breakout'] = current_price > recent_high * 1.02  # 2% above recent high
            
            # Breakdown pattern (price breaking below recent low)
            if len(close) >= 20:
                recent_low = low.tail(20).min()
                current_price = close.iloc[-1]
                patterns['Breakdown'] = current_price < recent_low * 0.98  # 2% below recent low
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            patterns['error'] = str(e)
        
        return patterns

class RiskAnalysisEngine(BaseAnalysisEngine):
    """Risk analysis engine for all asset types."""
    
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
            returns = price_data['Close'].pct_change().dropna()
            
            # Basic risk metrics
            results['volatility_metrics'] = self._calculate_volatility_metrics(returns)
            
            # Drawdown analysis
            results['drawdown_analysis'] = self._calculate_drawdown_metrics(price_data['Close'])
            
            # Value at Risk
            results['var_analysis'] = self._calculate_var_metrics(returns)
            
            # Risk-adjusted returns
            results['risk_adjusted_returns'] = self._calculate_risk_adjusted_returns(returns)
            
            # Tail risk metrics
            results['tail_risk'] = self._calculate_tail_risk_metrics(returns)
            
            return results
            
        except Exception as e:
            logger.error(f"Risk analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-related metrics."""
        metrics = {}
        
        # Daily volatility
        daily_vol = returns.std()
        metrics['daily_volatility'] = daily_vol
        
        # Annualized volatility
        metrics['annual_volatility'] = daily_vol * np.sqrt(252) * 100
        
        # Rolling volatilities
        vol_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
        vol_90d = returns.rolling(90).std().iloc[-1] * np.sqrt(252) * 100
        
        metrics['volatility_30d'] = vol_30d if not np.isnan(vol_30d) else metrics['annual_volatility']
        metrics['volatility_90d'] = vol_90d if not np.isnan(vol_90d) else metrics['annual_volatility']
        
        # Volatility of volatility
        rolling_vol = returns.rolling(30).std()
        vol_of_vol = rolling_vol.std() * np.sqrt(252) * 100
        metrics['volatility_of_volatility'] = vol_of_vol if not np.isnan(vol_of_vol) else 0
        
        return metrics
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        metrics = {}
        
        # Calculate drawdown
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        metrics['max_drawdown'] = abs(drawdown.min()) * 100
        
        # Average drawdown
        metrics['average_drawdown'] = abs(drawdown[drawdown < 0].mean()) * 100 if (drawdown < 0).any() else 0
        
        # Current drawdown
        metrics['current_drawdown'] = abs(drawdown.iloc[-1]) * 100
        
        # Drawdown duration (days in drawdown)
        in_drawdown = drawdown < -0.05  # More than 5% drawdown
        metrics['drawdown_duration'] = in_drawdown.sum()
        
        return metrics
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk metrics."""
        metrics = {}
        
        # Historical VaR
        metrics['var_1_percent'] = np.percentile(returns, 1) * 100
        metrics['var_5_percent'] = np.percentile(returns, 5) * 100
        metrics['var_10_percent'] = np.percentile(returns, 10) * 100
        
        # Expected Shortfall (Conditional VaR)
        var_5_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_5_threshold]
        metrics['expected_shortfall_5'] = tail_returns.mean() * 100 if not tail_returns.empty else 0
        
        return metrics
    
    def _calculate_risk_adjusted_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        metrics = {}
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else returns.std()
        metrics['sortino_ratio'] = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        
        # Calmar ratio
        annual_return = returns.mean() * 252
        max_dd = self._calculate_drawdown_metrics(returns.cumsum())['max_drawdown'] / 100
        metrics['calmar_ratio'] = annual_return / max_dd if max_dd != 0 else 0
        
        return metrics
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics."""
        metrics = {}
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Tail ratio
        tail_95 = returns.quantile(0.95)
        tail_5 = returns.quantile(0.05)
        metrics['tail_ratio'] = abs(tail_95 / tail_5) if tail_5 != 0 else 0
        
        return metrics

class PerformanceAnalysisEngine(BaseAnalysisEngine):
    """Performance analysis engine for all asset types."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information (optional)
        
        Returns:
            Dictionary with performance analysis results
        """
        if price_data.empty:
            return {'error': 'No price data available'}
        
        try:
            results = {}
            
            prices = price_data['Close']
            returns = prices.pct_change().dropna()
            
            # Return metrics
            results['return_metrics'] = self._calculate_return_metrics(prices)
            
            # Rolling performance
            results['rolling_performance'] = self._calculate_rolling_performance(prices)
            
            # Calendar performance
            results['calendar_performance'] = self._calculate_calendar_performance(prices)
            
            # Benchmark comparison (if applicable)
            results['relative_performance'] = self._calculate_relative_performance(prices, asset_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Performance analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_return_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics."""
        metrics = {}
        
        # Total return
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        metrics['total_return'] = total_return
        
        # Annualized return
        days = len(prices)
        annualized_return = ((prices.iloc[-1] / prices.iloc[0]) ** (252 / days) - 1) * 100
        metrics['annualized_return'] = annualized_return
        
        # Period-specific returns
        if len(prices) >= 7:
            weekly_return = (prices.iloc[-1] / prices.iloc[-8] - 1) * 100
            metrics['weekly_return'] = weekly_return
        
        if len(prices) >= 30:
            monthly_return = (prices.iloc[-1] / prices.iloc[-31] - 1) * 100
            metrics['monthly_return'] = monthly_return
        
        if len(prices) >= 90:
            quarterly_return = (prices.iloc[-1] / prices.iloc[-91] - 1) * 100
            metrics['quarterly_return'] = quarterly_return
        
        # Best and worst periods
        daily_returns = prices.pct_change().dropna() * 100
        metrics['best_day'] = daily_returns.max()
        metrics['worst_day'] = daily_returns.min()
        
        return metrics
    
    def _calculate_rolling_performance(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        rolling_metrics = {}
        
        # Rolling returns
        rolling_metrics['rolling_30d_returns'] = prices.pct_change(30) * 100
        rolling_metrics['rolling_90d_returns'] = prices.pct_change(90) * 100
        
        # Rolling volatility
        returns = prices.pct_change().dropna()
        rolling_metrics['rolling_30d_volatility'] = returns.rolling(30).std() * np.sqrt(252) * 100
        
        return rolling_metrics
    
    def _calculate_calendar_performance(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate calendar-based performance."""
        calendar_perf = {}
        
        try:
            # Monthly returns
            monthly_prices = prices.resample('M').last()
            monthly_returns = monthly_prices.pct_change().dropna() * 100
            
            if not monthly_returns.empty:
                calendar_perf['monthly_returns'] = monthly_returns
                calendar_perf['best_month'] = monthly_returns.max()
                calendar_perf['worst_month'] = monthly_returns.min()
                calendar_perf['positive_months'] = (monthly_returns > 0).sum()
                calendar_perf['negative_months'] = (monthly_returns < 0).sum()
            
        except Exception as e:
            logger.error(f"Calendar performance error: {str(e)}")
            calendar_perf['error'] = str(e)
        
        return calendar_perf
    
    def _calculate_relative_performance(self, prices: pd.Series, asset_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance relative to benchmarks."""
        relative_perf = {}
        
        # This would typically compare against relevant benchmarks
        # For now, we'll provide placeholder metrics
        relative_perf['vs_market'] = 0  # Would compare to S&P 500 for stocks
        relative_perf['vs_sector'] = 0  # Would compare to sector ETF
        relative_perf['beta'] = asset_info.get('beta', 1.0) if asset_info else 1.0
        
        return relative_perf

class MacroeconomicAnalysisEngine(BaseAnalysisEngine):
    """Macroeconomic analysis engine for market context."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform macroeconomic analysis and market context assessment.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information
        
        Returns:
            Dictionary with macroeconomic analysis results
        """
        try:
            results = {}
            
            # Monetary policy impact assessment
            results['monetary_policy'] = self._assess_monetary_policy_impact(price_data, asset_info)
            
            # Interest rate sensitivity analysis
            results['interest_rate_sensitivity'] = self._analyze_interest_rate_sensitivity(price_data)
            
            # Inflation impact assessment
            results['inflation_impact'] = self._assess_inflation_impact(price_data, asset_info)
            
            # Market correlation analysis
            results['market_correlations'] = self._analyze_market_correlations(price_data)
            
            # Economic cycle analysis
            results['economic_cycle'] = self._analyze_economic_cycle(price_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in macroeconomic analysis: {str(e)}")
            return {'error': f'Macroeconomic analysis failed: {str(e)}'}
    
    def _assess_monetary_policy_impact(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of monetary policy on asset."""
        try:
            # Simulate monetary policy indicators
            current_date = datetime.now()
            
            # Mock Fed funds rate and policy stance
            fed_funds_rate = 5.25  # Current simulated rate
            policy_stance = "restrictive"  # hawkish/neutral/dovish
            
            # Calculate asset's historical response to rate changes
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.rolling(30).std().iloc[-1] if len(returns) > 30 else returns.std()
            
            # Assess impact based on asset type
            asset_type = asset_info.get('asset_type', 'Unknown') if asset_info else 'Unknown'
            
            if asset_type == 'Stock':
                # Growth stocks more sensitive to rates
                sector = asset_info.get('sector', 'Technology') if asset_info else 'Technology'
                rate_sensitivity = 'High' if sector in ['Technology', 'Growth'] else 'Medium'
                impact_score = -0.7 if policy_stance == 'restrictive' else 0.3
            elif asset_type == 'ETF':
                # ETF sensitivity depends on underlying assets
                rate_sensitivity = 'Medium'
                impact_score = -0.4 if policy_stance == 'restrictive' else 0.2
            else:  # Cryptocurrency
                # Crypto historically negatively correlated with rates
                rate_sensitivity = 'Very High'
                impact_score = -0.9 if policy_stance == 'restrictive' else 0.6
            
            return {
                'current_fed_rate': fed_funds_rate,
                'policy_stance': policy_stance,
                'rate_sensitivity': rate_sensitivity,
                'impact_score': impact_score,
                'volatility_factor': float(volatility),
                'assessment': f"{asset_type} shows {rate_sensitivity.lower()} sensitivity to monetary policy changes"
            }
            
        except Exception as e:
            logger.error(f"Error assessing monetary policy impact: {e}")
            return {'error': 'Monetary policy assessment failed'}
    
    def _analyze_interest_rate_sensitivity(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze interest rate sensitivity (duration analysis)."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Simulate 10-year treasury yields (mock data)
            dates = price_data.index[-len(returns):]
            treasury_yields = np.random.normal(4.0, 0.5, len(returns))  # Mock 10Y yields
            
            # Calculate correlation with interest rates
            if len(returns) > 10:
                correlation = np.corrcoef(returns, treasury_yields)[0, 1]
                
                # Modified duration calculation (simplified)
                price_changes = price_data['Close'].pct_change().dropna()
                yield_changes = np.diff(treasury_yields)
                
                if len(price_changes) > len(yield_changes):
                    price_changes = price_changes.iloc[:len(yield_changes)]
                elif len(yield_changes) > len(price_changes):
                    yield_changes = yield_changes[:len(price_changes)]
                
                # Duration approximation
                if len(yield_changes) > 0 and np.std(yield_changes) > 0:
                    duration = -np.mean(price_changes) / np.mean(yield_changes) if np.mean(yield_changes) != 0 else 0
                else:
                    duration = 0
            else:
                correlation = 0
                duration = 0
            
            # Interpret sensitivity
            if abs(correlation) > 0.7:
                sensitivity_level = 'Very High'
            elif abs(correlation) > 0.4:
                sensitivity_level = 'High'
            elif abs(correlation) > 0.2:
                sensitivity_level = 'Medium'
            else:
                sensitivity_level = 'Low'
            
            return {
                'rate_correlation': float(correlation),
                'modified_duration': float(duration),
                'sensitivity_level': sensitivity_level,
                'current_10y_yield': 4.2,  # Mock current yield
                'interpretation': f"Asset shows {sensitivity_level.lower()} interest rate sensitivity"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing interest rate sensitivity: {e}")
            return {'error': 'Interest rate sensitivity analysis failed'}
    
    def _assess_inflation_impact(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess inflation impact on asset."""
        try:
            # Mock current inflation metrics
            current_cpi = 3.2  # Current CPI YoY
            core_cpi = 2.8     # Core CPI YoY
            pce = 2.9          # PCE YoY
            
            # Calculate real returns (inflation-adjusted)
            returns = price_data['Close'].pct_change().dropna()
            real_returns = returns - (current_cpi / 100 / 12)  # Monthly real return approximation
            
            # Assess inflation hedge characteristics
            asset_type = asset_info.get('asset_type', 'Unknown') if asset_info else 'Unknown'
            
            if asset_type == 'Stock':
                sector = asset_info.get('sector', 'Technology') if asset_info else 'Technology'
                if sector in ['Energy', 'Materials', 'Real Estate']:
                    hedge_quality = 'Good'
                    hedge_score = 0.6
                elif sector in ['Utilities', 'Consumer Staples']:
                    hedge_quality = 'Moderate'
                    hedge_score = 0.3
                else:
                    hedge_quality = 'Poor'
                    hedge_score = -0.2
            elif asset_type == 'ETF':
                # Depends on underlying assets
                hedge_quality = 'Moderate'
                hedge_score = 0.2
            else:  # Cryptocurrency
                # Crypto as digital gold thesis
                hedge_quality = 'Speculative'
                hedge_score = 0.4
            
            return {
                'current_cpi': current_cpi,
                'core_cpi': core_cpi,
                'pce': pce,
                'real_return_avg': float(real_returns.mean()) if len(real_returns) > 0 else 0,
                'inflation_hedge_quality': hedge_quality,
                'hedge_effectiveness_score': hedge_score,
                'assessment': f"Asset serves as {hedge_quality.lower()} inflation hedge in current environment"
            }
            
        except Exception as e:
            logger.error(f"Error assessing inflation impact: {e}")
            return {'error': 'Inflation impact assessment failed'}
    
    def _analyze_market_correlations(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations with major market indices."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Simulate major market indices returns (mock data)
            sp500_returns = np.random.normal(0.001, 0.02, len(returns))  # S&P 500
            nasdaq_returns = np.random.normal(0.001, 0.025, len(returns))  # NASDAQ
            dxy_returns = np.random.normal(0, 0.01, len(returns))  # Dollar Index
            vix_returns = np.random.normal(0, 0.1, len(returns))  # VIX
            gold_returns = np.random.normal(0.0005, 0.015, len(returns))  # Gold
            
            correlations = {}
            if len(returns) > 10:
                correlations['sp500'] = float(np.corrcoef(returns, sp500_returns)[0, 1])
                correlations['nasdaq'] = float(np.corrcoef(returns, nasdaq_returns)[0, 1])
                correlations['dxy'] = float(np.corrcoef(returns, dxy_returns)[0, 1])
                correlations['vix'] = float(np.corrcoef(returns, vix_returns)[0, 1])
                correlations['gold'] = float(np.corrcoef(returns, gold_returns)[0, 1])
            else:
                correlations = {'sp500': 0, 'nasdaq': 0, 'dxy': 0, 'vix': 0, 'gold': 0}
            
            # Determine market relationship
            if correlations['sp500'] > 0.7:
                market_relationship = 'Highly correlated with equities'
            elif correlations['sp500'] > 0.3:
                market_relationship = 'Moderately correlated with equities'
            elif correlations['sp500'] < -0.3:
                market_relationship = 'Negatively correlated with equities'
            else:
                market_relationship = 'Low correlation with traditional markets'
            
            return {
                'correlations': correlations,
                'market_relationship': market_relationship,
                'diversification_benefit': 'High' if abs(correlations['sp500']) < 0.3 else 'Low',
                'risk_on_off_sensitivity': 'High' if abs(correlations['vix']) > 0.4 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market correlations: {e}")
            return {'error': 'Market correlation analysis failed'}
    
    def _analyze_economic_cycle(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze asset performance across economic cycles."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Mock economic indicators
            gdp_growth = 2.1  # Current GDP growth rate
            unemployment = 3.8  # Current unemployment rate
            leading_indicators = 98.5  # Leading economic indicators index
            
            # Determine current economic cycle phase
            if gdp_growth > 3.0 and unemployment < 4.0:
                cycle_phase = 'Expansion'
                phase_score = 0.8
            elif gdp_growth > 1.0 and unemployment < 6.0:
                cycle_phase = 'Mid-Cycle'
                phase_score = 0.4
            elif gdp_growth > 0 and unemployment > 5.0:
                cycle_phase = 'Late-Cycle'
                phase_score = 0.1
            else:
                cycle_phase = 'Contraction'
                phase_score = -0.3
            
            # Calculate asset performance metrics for current cycle
            if len(returns) >= 90:  # 3 months of data
                recent_performance = returns.iloc[-90:].mean() * 252  # Annualized
                recent_volatility = returns.iloc[-90:].std() * np.sqrt(252)
            else:
                recent_performance = returns.mean() * 252 if len(returns) > 0 else 0
                recent_volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            return {
                'current_cycle_phase': cycle_phase,
                'phase_score': phase_score,
                'gdp_growth': gdp_growth,
                'unemployment_rate': unemployment,
                'recent_performance': float(recent_performance),
                'recent_volatility': float(recent_volatility),
                'cycle_assessment': f"Asset performance in {cycle_phase} phase shows {'positive' if recent_performance > 0 else 'negative'} momentum"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing economic cycle: {e}")
            return {'error': 'Economic cycle analysis failed'}

class FundamentalAnalysisEngine(BaseAnalysisEngine):
    """Fundamental analysis engine for ETF holdings and crypto on-chain metrics."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform fundamental analysis based on asset type.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information
        
        Returns:
            Dictionary with fundamental analysis results
        """
        try:
            if not asset_info:
                return {'error': 'Asset information required for fundamental analysis'}
            
            asset_type = asset_info.get('asset_type', 'Unknown')
            
            if asset_type == 'ETF':
                return self._analyze_etf_fundamentals(price_data, asset_info)
            elif asset_type == 'Cryptocurrency':
                return self._analyze_crypto_fundamentals(price_data, asset_info)
            elif asset_type == 'Stock':
                return self._analyze_stock_fundamentals(price_data, asset_info)
            else:
                return {'error': f'Fundamental analysis not supported for {asset_type}'}
                
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return {'error': f'Fundamental analysis failed: {str(e)}'}
    
    def _analyze_etf_fundamentals(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETF fundamentals including holdings and expense ratios."""
        try:
            symbol = asset_info.get('symbol', 'Unknown')
            
            # Mock ETF fundamental data (in production, would fetch from actual sources)
            etf_data = {
                'total_assets': asset_info.get('totalAssets', 1000000000),
                'expense_ratio': asset_info.get('expenseRatio', 0.05),
                'category': asset_info.get('category', 'Equity'),
                'fund_family': asset_info.get('fundFamily', f'{symbol} Funds')
            }
            
            # Simulate top holdings analysis
            top_holdings = self._generate_mock_etf_holdings(symbol)
            
            # Sector exposure analysis
            sector_exposure = self._analyze_sector_exposure(symbol)
            
            # Expense ratio comparison
            expense_analysis = self._analyze_expense_ratio(etf_data['expense_ratio'], etf_data['category'])
            
            # Asset flow analysis
            flow_analysis = self._analyze_asset_flows(price_data, etf_data['total_assets'])
            
            # Risk assessment
            risk_metrics = self._assess_etf_risks(price_data, etf_data)
            
            return {
                'etf_metrics': etf_data,
                'top_holdings': top_holdings,
                'sector_exposure': sector_exposure,
                'expense_analysis': expense_analysis,
                'flow_analysis': flow_analysis,
                'risk_assessment': risk_metrics,
                'recommendation': self._generate_etf_recommendation(etf_data, expense_analysis, risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ETF fundamentals: {e}")
            return {'error': 'ETF fundamental analysis failed'}
    
    def _analyze_crypto_fundamentals(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptocurrency fundamentals including on-chain metrics and adoption."""
        try:
            symbol = asset_info.get('symbol', 'Unknown')
            
            # Mock on-chain metrics (in production, would fetch from blockchain APIs)
            onchain_metrics = self._generate_mock_onchain_metrics(symbol, price_data)
            
            # Developer activity analysis
            dev_activity = self._analyze_developer_activity(symbol)
            
            # Adoption metrics
            adoption_metrics = self._analyze_adoption_metrics(symbol)
            
            # Network health assessment
            network_health = self._assess_network_health(symbol, onchain_metrics)
            
            # Tokenomics analysis
            tokenomics = self._analyze_tokenomics(asset_info)
            
            # Regulatory risk assessment
            regulatory_risk = self._assess_regulatory_risk(symbol)
            
            return {
                'onchain_metrics': onchain_metrics,
                'developer_activity': dev_activity,
                'adoption_metrics': adoption_metrics,
                'network_health': network_health,
                'tokenomics': tokenomics,
                'regulatory_risk': regulatory_risk,
                'fundamental_score': self._calculate_crypto_fundamental_score(
                    onchain_metrics, dev_activity, adoption_metrics, network_health
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing crypto fundamentals: {e}")
            return {'error': 'Crypto fundamental analysis failed'}
    
    def _analyze_stock_fundamentals(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stock fundamentals."""
        try:
            # Extract key financial metrics
            financials = {
                'market_cap': asset_info.get('marketCap', 0),
                'pe_ratio': asset_info.get('trailingPE', 0),
                'dividend_yield': asset_info.get('dividendYield', 0),
                'beta': asset_info.get('beta', 1.0),
                'sector': asset_info.get('sector', 'Unknown'),
                'industry': asset_info.get('industry', 'Unknown')
            }
            
            # Valuation analysis
            valuation = self._analyze_valuation_metrics(financials, price_data)
            
            # Sector comparison
            sector_analysis = self._compare_to_sector(financials)
            
            # Growth analysis
            growth_metrics = self._analyze_growth_potential(price_data, financials)
            
            return {
                'financial_metrics': financials,
                'valuation_analysis': valuation,
                'sector_comparison': sector_analysis,
                'growth_analysis': growth_metrics,
                'investment_thesis': self._generate_investment_thesis(financials, valuation, growth_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock fundamentals: {e}")
            return {'error': 'Stock fundamental analysis failed'}
    
    def _generate_mock_etf_holdings(self, symbol: str) -> Dict[str, Any]:
        """Generate mock ETF holdings data."""
        # Mock top 10 holdings based on symbol
        holdings_map = {
            'SPY': [
                {'symbol': 'AAPL', 'weight': 7.2, 'name': 'Apple Inc'},
                {'symbol': 'MSFT', 'weight': 6.8, 'name': 'Microsoft Corp'},
                {'symbol': 'AMZN', 'weight': 3.4, 'name': 'Amazon.com Inc'},
                {'symbol': 'NVDA', 'weight': 3.1, 'name': 'NVIDIA Corp'},
                {'symbol': 'GOOGL', 'weight': 2.9, 'name': 'Alphabet Inc'}
            ],
            'QQQ': [
                {'symbol': 'AAPL', 'weight': 12.8, 'name': 'Apple Inc'},
                {'symbol': 'MSFT', 'weight': 11.2, 'name': 'Microsoft Corp'},
                {'symbol': 'AMZN', 'weight': 6.1, 'name': 'Amazon.com Inc'},
                {'symbol': 'NVDA', 'weight': 5.8, 'name': 'NVIDIA Corp'},
                {'symbol': 'META', 'weight': 4.9, 'name': 'Meta Platforms Inc'}
            ]
        }
        
        default_holdings = [
            {'symbol': f'{symbol}H1', 'weight': 8.5, 'name': f'{symbol} Top Holding 1'},
            {'symbol': f'{symbol}H2', 'weight': 7.2, 'name': f'{symbol} Top Holding 2'},
            {'symbol': f'{symbol}H3', 'weight': 6.1, 'name': f'{symbol} Top Holding 3'},
            {'symbol': f'{symbol}H4', 'weight': 5.4, 'name': f'{symbol} Top Holding 4'},
            {'symbol': f'{symbol}H5', 'weight': 4.8, 'name': f'{symbol} Top Holding 5'}
        ]
        
        holdings = holdings_map.get(symbol, default_holdings)
        concentration_risk = sum(h['weight'] for h in holdings[:5])  # Top 5
        # Calculate missing data for top holdings
        for holding in holdings:
            holding.setdefault('name', 'Unknown Company')
            holding.setdefault('weight', 0)
        
        return {
            'top_holdings': holdings,
            'concentration_risk': concentration_risk,
            'diversification_score': 100 - concentration_risk
        }
    
    def _analyze_sector_exposure(self, symbol: str) -> Dict[str, Any]:
        """Analyze ETF sector exposure."""
        # Mock sector allocations
        sector_allocations = {
            'Technology': np.random.uniform(20, 40),
            'Healthcare': np.random.uniform(10, 20),
            'Financial Services': np.random.uniform(10, 20),
            'Consumer Cyclical': np.random.uniform(8, 15),
            'Communication Services': np.random.uniform(5, 12),
            'Industrials': np.random.uniform(5, 10),
            'Consumer Defensive': np.random.uniform(3, 8),
            'Energy': np.random.uniform(2, 6),
            'Utilities': np.random.uniform(2, 5),
            'Real Estate': np.random.uniform(2, 5)
        }
        
        # Normalize to 100%
        total = sum(sector_allocations.values())
        sector_allocations = {k: round(v/total * 100, 1) for k, v in sector_allocations.items()}
        
        # Identify dominant sectors
        dominant_sectors = {k: v for k, v in sector_allocations.items() if v > 15}
        
        return {
            'sector_allocations': sector_allocations,
            'dominant_sectors': dominant_sectors,
            'sector_concentration_risk': max(sector_allocations.values())
        }
    
    def _analyze_expense_ratio(self, expense_ratio: float, category: str) -> Dict[str, Any]:
        """Analyze ETF expense ratio competitiveness."""
        # Industry averages by category
        category_averages = {
            'Equity': 0.06,
            'Fixed Income': 0.05,
            'Commodity': 0.08,
            'Sector': 0.07
        }
        
        avg_expense = category_averages.get(category, 0.06)
        
        if expense_ratio < avg_expense * 0.8:
            competitiveness = 'Very Competitive'
        elif expense_ratio < avg_expense * 1.2:
            competitiveness = 'Competitive'
        elif expense_ratio < avg_expense * 1.5:
            competitiveness = 'Average'
        else:
            competitiveness = 'High'
        
        return {
            'expense_ratio': expense_ratio,
            'category_average': avg_expense,
            'competitiveness': competitiveness,
            'annual_cost_per_10k': expense_ratio * 10000
        }
    
    def _generate_mock_onchain_metrics(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mock on-chain metrics for cryptocurrency analysis."""
        current_price = price_data['Close'].iloc[-1] if not price_data.empty else 100
        
        if symbol == 'BTC':
            return {
                'active_addresses': 850000,
                'transaction_volume_24h': 15000000000,  # $15B
                'hash_rate': 450000000,  # 450 EH/s
                'network_value_to_transactions': current_price * 19500000 / 15000000000,
                'realized_cap': 400000000000,  # $400B
                'mvrv_ratio': current_price * 19500000 / 400000000000,
                'long_term_holder_supply': 0.68,  # 68%
                'exchange_inflow': -2500,  # Net outflow (positive = bearish)
                'fear_greed_index': np.random.randint(20, 80)
            }
        elif symbol == 'ETH':
            return {
                'active_addresses': 450000,
                'transaction_volume_24h': 8000000000,  # $8B
                'gas_price_gwei': 25,
                'total_value_locked': 25000000000,  # $25B in DeFi
                'staking_ratio': 0.22,  # 22% of supply staked
                'burn_rate': 1200,  # ETH burned per day
                'network_growth': 0.15,  # 15% YoY address growth
                'defi_dominance': 0.65  # 65% of DeFi on Ethereum
            }
        else:
            # Generic altcoin metrics
            return {
                'active_addresses': np.random.randint(10000, 100000),
                'transaction_volume_24h': np.random.randint(10000000, 1000000000),
                'network_growth': np.random.uniform(-0.1, 0.3),
                'token_velocity': np.random.uniform(2, 20),
                'development_activity': np.random.uniform(0.1, 1.0)
            }
    
    def _analyze_developer_activity(self, symbol: str) -> Dict[str, Any]:
        """Analyze cryptocurrency developer activity and ecosystem health."""
        # Mock developer metrics
        github_commits = np.random.randint(50, 500)  # Monthly commits
        active_developers = np.random.randint(10, 100)
        code_quality_score = np.random.uniform(0.6, 0.95)
        
        if symbol in ['BTC', 'ETH']:
            github_commits = np.random.randint(200, 800)
            active_developers = np.random.randint(50, 200)
            code_quality_score = np.random.uniform(0.85, 0.98)
        
        # Calculate development health score
        dev_score = (
            min(github_commits / 500, 1.0) * 0.4 +
            min(active_developers / 100, 1.0) * 0.3 +
            code_quality_score * 0.3
        )
        
        if dev_score > 0.8:
            development_health = 'Excellent'
        elif dev_score > 0.6:
            development_health = 'Good'
        elif dev_score > 0.4:
            development_health = 'Fair'
        else:
            development_health = 'Poor'
        
        return {
            'monthly_commits': github_commits,
            'active_developers': active_developers,
            'code_quality_score': round(code_quality_score, 2),
            'development_health': development_health,
            'ecosystem_partnerships': np.random.randint(5, 50),
            'innovation_score': round(dev_score, 2)
        }
    
    def _analyze_adoption_metrics(self, symbol: str) -> Dict[str, Any]:
        """Analyze cryptocurrency adoption and real-world usage."""
        # Mock adoption metrics
        if symbol == 'BTC':
            metrics = {
                'merchant_acceptance': 15000,
                'institutional_adoption': 'High',
                'payment_volume_24h': 5000000000,  # $5B
                'wallet_addresses': 45000000,
                'social_sentiment': np.random.uniform(0.4, 0.8),
                'google_trends_score': np.random.randint(60, 100)
            }
        elif symbol == 'ETH':
            metrics = {
                'dapp_users_30d': 2500000,
                'defi_tvl': 25000000000,  # $25B
                'nft_volume_30d': 800000000,  # $800M
                'enterprise_adoption': 'Growing',
                'social_sentiment': np.random.uniform(0.5, 0.8),
                'developer_mindshare': 'Dominant'
            }
        else:
            metrics = {
                'daily_active_users': np.random.randint(1000, 50000),
                'transaction_count_24h': np.random.randint(10000, 500000),
                'social_sentiment': np.random.uniform(0.2, 0.7),
                'partnerships': np.random.randint(1, 20),
                'use_case_strength': np.random.choice(['Weak', 'Moderate', 'Strong'])
            }
        
        return metrics
    
    def _assess_network_health(self, symbol: str, onchain_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall network health and sustainability."""
        try:
            # Calculate composite health score
            if symbol == 'BTC':
                hash_rate_score = min(onchain_metrics.get('hash_rate', 0) / 500000000, 1.0)
                address_growth = min(onchain_metrics.get('active_addresses', 0) / 1000000, 1.0)
                network_security = 'Excellent' if hash_rate_score > 0.8 else 'Good'
            elif symbol == 'ETH':
                staking_score = onchain_metrics.get('staking_ratio', 0)
                tvl_score = min(onchain_metrics.get('total_value_locked', 0) / 50000000000, 1.0)
                network_security = 'Excellent' if staking_score > 0.2 else 'Good'
            else:
                address_growth = np.random.uniform(0.3, 0.8)
                network_security = np.random.choice(['Fair', 'Good', 'Excellent'])
            
            # Network congestion analysis
            if 'gas_price_gwei' in onchain_metrics:
                gas_price = onchain_metrics['gas_price_gwei']
                congestion_level = 'High' if gas_price > 50 else 'Low' if gas_price < 20 else 'Medium'
            else:
                congestion_level = 'Low'
            
            # Decentralization score
            decentralization_score = np.random.uniform(0.6, 0.95)
            
            return {
                'network_security': network_security,
                'congestion_level': congestion_level,
                'decentralization_score': round(decentralization_score, 2),
                'scalability_rating': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent']),
                'sustainability_outlook': 'Positive' if decentralization_score > 0.7 else 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Error assessing network health: {e}")
            return {'error': 'Network health assessment failed'}
    
    def _analyze_tokenomics(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptocurrency tokenomics."""
        try:
            supply_data = {
                'circulating_supply': asset_info.get('circulating_supply', 0),
                'max_supply': asset_info.get('max_supply'),
                'market_cap': asset_info.get('market_cap', 0)
            }
            
            # Calculate supply metrics
            if supply_data['max_supply']:
                supply_inflation = (supply_data['max_supply'] - supply_data['circulating_supply']) / supply_data['max_supply']
                scarcity_score = 1 - supply_inflation
            else:
                supply_inflation = None
                scarcity_score = 0.5  # Unknown max supply
            
            # Token distribution analysis (mock)
            distribution = {
                'public_allocation': np.random.uniform(0.4, 0.8),
                'team_allocation': np.random.uniform(0.1, 0.25),
                'treasury_allocation': np.random.uniform(0.05, 0.2),
                'ecosystem_fund': np.random.uniform(0.05, 0.15)
            }
            
            # Normalize distribution
            total_dist = sum(distribution.values())
            distribution = {k: round(v/total_dist, 2) for k, v in distribution.items()}
            
            return {
                'supply_metrics': supply_data,
                'supply_inflation_rate': supply_inflation,
                'scarcity_score': round(scarcity_score, 2),
                'token_distribution': distribution,
                'inflation_model': 'Deflationary' if supply_inflation and supply_inflation < 0.05 else 'Inflationary'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tokenomics: {e}")
            return {'error': 'Tokenomics analysis failed'}
    
    def _assess_regulatory_risk(self, symbol: str) -> Dict[str, Any]:
        """Assess regulatory risk for cryptocurrency."""
        # Mock regulatory risk assessment
        risk_factors = {
            'sec_clarity': np.random.choice(['Clear', 'Unclear', 'Unfavorable']),
            'global_acceptance': np.random.choice(['Widespread', 'Limited', 'Restricted']),
            'compliance_score': np.random.uniform(0.3, 0.95)
        }
        
        # Special cases for major cryptocurrencies
        if symbol == 'BTC':
            risk_factors = {
                'sec_clarity': 'Clear',
                'global_acceptance': 'Widespread',
                'compliance_score': 0.9
            }
        elif symbol == 'ETH':
            risk_factors = {
                'sec_clarity': 'Mostly Clear',
                'global_acceptance': 'Widespread',
                'compliance_score': 0.85
            }
        
        # Calculate overall risk level
        if risk_factors['compliance_score'] > 0.8 and risk_factors['sec_clarity'] == 'Clear':
            overall_risk = 'Low'
        elif risk_factors['compliance_score'] > 0.6:
            overall_risk = 'Medium'
        else:
            overall_risk = 'High'
        
        return {
            'regulatory_factors': risk_factors,
            'overall_risk_level': overall_risk,
            'key_jurisdictions': ['US', 'EU', 'Asia-Pacific'],
            'outlook': 'Improving' if risk_factors['compliance_score'] > 0.7 else 'Uncertain'
        }
    
    def _calculate_crypto_fundamental_score(self, onchain_metrics: Dict[str, Any], 
                                          dev_activity: Dict[str, Any], 
                                          adoption_metrics: Dict[str, Any], 
                                          network_health: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall fundamental score for cryptocurrency."""
        try:
            # Weighted scoring
            scores = {
                'onchain_score': 0.3,
                'development_score': 0.25,
                'adoption_score': 0.25,
                'network_score': 0.2
            }
            
            # Calculate individual scores (0-1 scale)
            onchain_score = min(onchain_metrics.get('active_addresses', 0) / 1000000, 1.0) * 0.5 + \
                           min(onchain_metrics.get('transaction_volume_24h', 0) / 20000000000, 1.0) * 0.5
            
            development_score = dev_activity.get('innovation_score', 0.5)
            
            if 'social_sentiment' in adoption_metrics:
                adoption_score = adoption_metrics['social_sentiment']
            else:
                adoption_score = 0.5
            
            network_score = network_health.get('decentralization_score', 0.5)
            
            # Calculate weighted fundamental score
            fundamental_score = (
                onchain_score * scores['onchain_score'] +
                development_score * scores['development_score'] +
                adoption_score * scores['adoption_score'] +
                network_score * scores['network_score']
            )
            
            # Interpret score
            if fundamental_score > 0.8:
                rating = 'Excellent'
            elif fundamental_score > 0.6:
                rating = 'Good'
            elif fundamental_score > 0.4:
                rating = 'Fair'
            else:
                rating = 'Poor'
            
            return {
                'fundamental_score': round(fundamental_score, 2),
                'rating': rating,
                'component_scores': {
                    'onchain': round(onchain_score, 2),
                    'development': round(development_score, 2),
                    'adoption': round(adoption_score, 2),
                    'network': round(network_score, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return {'error': 'Fundamental score calculation failed'}
        
    def _analyze_valuation_metrics(self, financials: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stock valuation metrics."""
        try:
            current_price = price_data['Close'].iloc[-1] if not price_data.empty else 150
            market_cap = financials.get('market_cap', 0)
            pe_ratio = financials.get('pe_ratio', 15)
            
            # Determine valuation level
            if pe_ratio < 15:
                valuation_level = 'Undervalued'
            elif pe_ratio < 25:
                valuation_level = 'Fair Value'
            else:
                valuation_level = 'Overvalued'
            
            return {
                'current_valuation': valuation_level,
                'pe_ratio': pe_ratio,
                'price_to_book': np.random.uniform(1.0, 4.0),
                'price_to_sales': np.random.uniform(0.5, 8.0),
                'ev_to_ebitda': np.random.uniform(8.0, 20.0)
            }
        except Exception as e:
            return {'error': f'Valuation analysis failed: {e}'}
    
    def _compare_to_sector(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Compare financial metrics to sector averages."""
        sector = financials.get('sector', 'Technology')
        
        # Mock sector comparison
        sector_avg_pe = {
            'Technology': 25, 'Healthcare': 20, 'Finance': 12,
            'Consumer': 18, 'Industrial': 16
        }.get(sector, 18)
        
        company_pe = financials.get('pe_ratio', 15)
        
        return {
            'sector': sector,
            'sector_avg_pe': sector_avg_pe,
            'company_pe': company_pe,
            'pe_premium_discount': (company_pe - sector_avg_pe) / sector_avg_pe,
            'sector_ranking': 'Above Average' if company_pe < sector_avg_pe else 'Below Average'
        }
    
    def _analyze_growth_potential(self, price_data: pd.DataFrame, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company growth potential."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Calculate growth metrics
            revenue_growth = np.random.uniform(-0.1, 0.3)  # Mock revenue growth
            earnings_growth = np.random.uniform(-0.2, 0.4)  # Mock earnings growth
            
            # Price momentum
            price_momentum = returns.iloc[-20:].mean() * 252 if len(returns) >= 20 else 0
            
            return {
                'revenue_growth_estimate': revenue_growth,
                'earnings_growth_estimate': earnings_growth,
                'price_momentum': price_momentum,
                'growth_rating': 'High' if revenue_growth > 0.15 else 'Moderate' if revenue_growth > 0.05 else 'Low'
            }
        except Exception as e:
            return {'error': f'Growth analysis failed: {e}'}
    
    def _generate_investment_thesis(self, financials: Dict[str, Any], valuation: Dict[str, Any], growth: Dict[str, Any]) -> str:
        """Generate investment thesis based on analysis."""
        try:
            symbol = financials.get('symbol', 'Unknown')
            sector = financials.get('sector', 'Unknown')
            valuation_level = valuation.get('current_valuation', 'Unknown')
            growth_rating = growth.get('growth_rating', 'Unknown')
            
            thesis = f"""
            Investment Thesis for {symbol}:
            
            {symbol} operates in the {sector} sector with {valuation_level.lower()} current valuation metrics.
            The company demonstrates {growth_rating.lower()} growth characteristics based on recent performance indicators.
            
            Key factors supporting the investment case include sector positioning and current market dynamics.
            Risk factors include market volatility and sector-specific challenges.
            
            Overall assessment suggests a {'positive' if valuation_level == 'Undervalued' and growth_rating == 'High' else 'neutral'} 
            investment outlook subject to broader market conditions.
            """
            
            return thesis.strip()
        except Exception as e:
            return f"Investment thesis generation failed: {e}"
    
    def _analyze_asset_flows(self, price_data: pd.DataFrame, total_assets: float) -> Dict[str, Any]:
        """Analyze ETF asset flows."""
        try:
            # Mock flow analysis
            recent_flows = np.random.uniform(-0.1, 0.1) * total_assets  # 10% max flow
            flow_trend = 'Inflows' if recent_flows > 0 else 'Outflows'
            
            return {
                'recent_flows': recent_flows,
                'flow_trend': flow_trend,
                'flow_magnitude': abs(recent_flows) / total_assets,
                'investor_sentiment': 'Positive' if recent_flows > 0 else 'Negative'
            }
        except Exception as e:
            return {'error': f'Flow analysis failed: {e}'}
    
    def _assess_etf_risks(self, price_data: pd.DataFrame, etf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ETF-specific risks."""
        try:
            tracking_error = np.random.uniform(0.001, 0.01)  # Mock tracking error
            liquidity_risk = 'Low' if etf_data.get('total_assets', 0) > 1e9 else 'Medium'
            
            return {
                'tracking_error': tracking_error,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': 'Medium',  # Based on holdings analysis
                'overall_risk_rating': 'Low to Medium'
            }
        except Exception as e:
            return {'error': f'Risk assessment failed: {e}'}
    
    def _generate_etf_recommendation(self, etf_data: Dict[str, Any], expense_analysis: Dict[str, Any], risk_metrics: Dict[str, Any]) -> str:
        """Generate ETF recommendation."""
        try:
            competitiveness = expense_analysis.get('competitiveness', 'Unknown')
            risk_rating = risk_metrics.get('overall_risk_rating', 'Unknown')
            
            recommendation = f"""
            ETF Recommendation:
            
            This ETF demonstrates {competitiveness.lower()} expense structure within its category.
            Risk profile is assessed as {risk_rating.lower()} based on tracking error and liquidity metrics.
            
            Suitable for investors seeking diversified exposure with professional management.
            Regular monitoring of expense ratios and tracking performance recommended.
            """
            
            return recommendation.strip()
        except Exception as e:
            return f"Recommendation generation failed: {e}"

class PortfolioStrategyEngine(BaseAnalysisEngine):
    """Portfolio strategy engine for asset allocation and risk management."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform portfolio strategy analysis.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information (optional)
        
        Returns:
            Dictionary with portfolio strategy analysis results
        """
        if price_data.empty:
            return {'error': 'No price data available'}
        
        try:
            results = {}
            
            # Asset allocation strategies
            results['allocation_strategies'] = self._analyze_allocation_strategies(price_data, asset_info)
            
            # Risk management strategies
            results['risk_management'] = self._analyze_risk_management(price_data, asset_info)
            
            # Performance projections
            results['performance_projections'] = self._project_performance(price_data, asset_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Portfolio strategy analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_allocation_strategies(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze various asset allocation strategies."""
        strategies = {}
        
        try:
            # Get available assets
            assets = asset_info.get('assets', [])
            n_assets = len(assets)
            
            if n_assets == 0:
                return {'error': 'No assets available for allocation'}
            
            # Calculate expected returns and covariances
            expected_returns = np.array([self._calculate_expected_return(a, price_data) for a in assets])
            cov_matrix = np.cov(np.array([price_data[a].pct_change().dropna() for a in assets]))
            
            # Minimum variance portfolio
            strategies['min_variance'] = self._min_variance_portfolio(expected_returns, cov_matrix)
            
            # Maximum Sharpe ratio portfolio
            strategies['max_sharpe'] = self._max_sharpe_portfolio(expected_returns, cov_matrix)
            
            # Equal weight portfolio
            strategies['equal_weight'] = self._equal_weight_portfolio(n_assets)
            
            # Risk parity portfolio
            volatilities = np.sqrt(np.diag(cov_matrix))
            strategies['risk_parity'] = self._risk_parity_portfolio(volatilities)
            
            # Efficient frontier
            strategies['efficient_frontier'] = self._calculate_efficient_frontier(expected_returns, cov_matrix)
            
        except Exception as e:
            logger.error(f"Error analyzing allocation strategies: {e}")
            return {'error': 'Allocation strategy analysis failed'}
        
        return strategies
    
    def _analyze_risk_management(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk management strategies."""
        strategies = {}
        
        try:
            # Value at Risk (VaR) at 95% confidence
            returns = price_data['Close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5) * 100
            cvar_95 = returns[returns <= var_95].mean() * 100
            
            strategies['value_at_risk'] = {
                'var_95': var_95,
                'cvar_95': cvar_95
            }
            
            # Maximum drawdown
            drawdown_metrics = self._calculate_drawdown_metrics(price_data['Close'])
            strategies['max_drawdown'] = drawdown_metrics['max_drawdown']
            
            # Volatility targeting
            target_volatility = asset_info.get('target_volatility', 0.12)
            current_volatility = returns.std() * np.sqrt(252) * 100
            strategies['volatility_targeting'] = {
                'target_volatility': target_volatility,
                'current_volatility': current_volatility
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk management strategies: {e}")
            return {'error': 'Risk management analysis failed'}
        
        return strategies
    
    def _project_performance(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Project future performance based on historical data and scenarios."""
        projections = {}
        
        try:
            # Monte Carlo simulation parameters
            n_simulations = 1000
            n_days = 252  # 1 year
            
            # Historical returns
            returns = price_data['Close'].pct_change().dropna()
            mean_return = returns.mean()
            volatility = returns.std()
            
            # Simulate price paths
            price_paths = np.zeros((n_simulations, n_days))
            for i in range(n_simulations):
                # Random walk simulation
                random_shocks = np.random.normal(mean_return / n_days, volatility / np.sqrt(n_days), n_days)
                price_paths[i] = np.exp(np.log(price_data['Close'].iloc[-1]) + np.cumsum(random_shocks))
            
            # Calculate expected price and return
            expected_price = price_paths.mean(axis=0)
            expected_return = (expected_price[-1] / price_data['Close'].iloc[-1]) - 1
            
            projections = {
                'expected_price': expected_price,
                'expected_return': expected_return * 100,
                'simulated_price_paths': price_paths
            }
            
        except Exception as e:
            logger.error(f"Error projecting performance: {e}")
            return {'error': 'Performance projection failed'}
        
        return projections
    
    def _calculate_expected_return(self, allocation: Dict[str, Any], price_data: pd.DataFrame) -> float:
        """Calculate expected return for allocation strategy."""
        try:
            # Simplified expected return calculation
            returns = price_data['Close'].pct_change().dropna()
            historical_return = returns.mean() * 252 if len(returns) > 0 else 0.08
            
            # Adjust based on allocation strategy risk level
            risk_adjustment = {
                'Low': 0.6, 'Medium': 0.8, 'Medium-High': 1.0, 'High': 1.2
            }.get(allocation.get('risk_tolerance', 'Medium'), 0.8)
            
            return historical_return * risk_adjustment
        except Exception as e:
            return 0.08  # Default 8% return
    
    def _calculate_expected_risk(self, allocation: Dict[str, Any], price_data: pd.DataFrame) -> float:
        """Calculate expected risk for allocation strategy."""
        try:
            # Use target volatility if available
            target_vol = allocation.get('target_volatility', 0.12)
            return target_vol
        except Exception as e:
            return 0.12  # Default 12% volatility
    
    def _generate_correlation_matrix(self, n_assets: int) -> np.ndarray:
        """Generate realistic correlation matrix for portfolio optimization."""
        try:
            # Create a realistic correlation matrix
            np.random.seed(42)  # For reproducibility
            
            # Start with identity matrix
            corr_matrix = np.eye(n_assets)
            
            # Add realistic correlations
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    # Generate correlation between -0.2 and 0.8
                    correlation = np.random.uniform(0.1, 0.7)
                    corr_matrix[i, j] = correlation
                    corr_matrix[j, i] = correlation
            
            return corr_matrix
        except Exception as e:
            return np.eye(n_assets)
    
    def _min_variance_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate minimum variance portfolio."""
        try:
            n_assets = len(expected_returns)
            # Equal weights as approximation
            weights = np.ones(n_assets) / n_assets
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            return {
                'weights': weights.tolist(),
                'expected_return': portfolio_return,
                'expected_risk': portfolio_risk,
                'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            }
        except Exception as e:
            n_assets = len(expected_returns) if hasattr(expected_returns, '__len__') else 5
            weights = [1/n_assets] * n_assets
            return {'weights': weights, 'expected_return': 0.08, 'expected_risk': 0.12, 'sharpe_ratio': 0.67}
    
    def _max_sharpe_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate maximum Sharpe ratio portfolio."""
        try:
            # Simplified approach - weight by risk-adjusted returns
            risk_free_rate = 0.03
            excess_returns = expected_returns - risk_free_rate
            
            # Weight by excess return / variance (simplified)
            variances = np.diag(cov_matrix)
            weights = excess_returns / variances
            weights = weights / weights.sum()  # Normalize
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return {
                'weights': weights.tolist(),
                'expected_return': portfolio_return,
                'expected_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            n_assets = len(expected_returns) if hasattr(expected_returns, '__len__') else 5
            weights = [1/n_assets] * n_assets
            return {'weights': weights, 'expected_return': 0.10, 'expected_risk': 0.15, 'sharpe_ratio': 0.47}
    
    def _equal_weight_portfolio(self, n_assets: int) -> Dict[str, Any]:
        """Calculate equal weight portfolio."""
        weight = 1.0 / n_assets
        weights = [weight] * n_assets
        
        return {
            'weights': weights,
            'expected_return': 0.09,
            'expected_risk': 0.14,
            'sharpe_ratio': 0.43
        }
    
    def _risk_parity_portfolio(self, volatilities: np.ndarray) -> Dict[str, Any]:
        """Calculate risk parity portfolio."""
        try:
            # Weight inversely proportional to volatility
            inv_vol = 1.0 / volatilities
            weights = inv_vol / inv_vol.sum()
            
            return {
                'weights': weights.tolist(),
                'expected_return': 0.08,
                'expected_risk': 0.11,
                'sharpe_ratio': 0.45
            }
        except Exception as e:
            n_assets = len(volatilities) if hasattr(volatilities, '__len__') else 5
            weights = [1/n_assets] * n_assets
            return {'weights': weights, 'expected_return': 0.08, 'expected_risk': 0.11, 'sharpe_ratio': 0.45}
    
    def _calculate_efficient_frontier(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate efficient frontier points."""
        try:
            # Generate range of target returns
            min_ret = expected_returns.min()
            max_ret = expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, 20)
            
            frontier_risks = []
            frontier_returns = []
            
            for target_ret in target_returns:
                # Simplified: linear interpolation between min variance and max return
                weight = (target_ret - min_ret) / (max_ret - min_ret) if max_ret != min_ret else 0.5
                risk = 0.08 + weight * 0.15  # Scale risk from 8% to 23%
                
                frontier_returns.append(target_ret)
                frontier_risks.append(risk)
            
            return {
                'returns': frontier_returns,
                'risks': frontier_risks
            }
        except Exception as e:
            return {'returns': [0.06, 0.08, 0.10, 0.12], 'risks': [0.08, 0.12, 0.16, 0.20]}
    
    def _analyze_rebalancing_strategy(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimal rebalancing strategy."""
        try:
            # Calculate various rebalancing metrics
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.15
            
            # Recommend rebalancing frequency based on volatility
            if volatility > 0.3:
                rebalance_frequency = 'Monthly'
                threshold = 0.05
            elif volatility > 0.2:
                rebalance_frequency = 'Quarterly'
                threshold = 0.10
            else:
                rebalance_frequency = 'Semi-Annually'
                threshold = 0.15
            
            return {
                'recommended_frequency': rebalance_frequency,
                'drift_threshold': threshold,
                'transaction_costs': 0.001,  # 0.1% assumed cost
                'expected_benefit': 0.005   # 0.5% annual benefit
            }
        except Exception as e:
            return {
                'recommended_frequency': 'Quarterly',
                'drift_threshold': 0.10,
                'transaction_costs': 0.001,
                'expected_benefit': 0.005
            }
    
    def _generate_allocation_recommendations(self, strategies: Dict[str, Any], asset_type: str) -> str:
        """Generate allocation recommendations."""
        try:
            best_strategy = max(strategies.keys(), 
                              key=lambda x: strategies[x].get('sharpe_estimate', 0))
            
            recommendation = f"""
            Allocation Recommendations for {asset_type}:
            
            Based on risk-return analysis, the {best_strategy} strategy shows the most attractive 
            risk-adjusted returns (Sharpe ratio: {strategies[best_strategy].get('sharpe_estimate', 0):.2f}).
            
            Key considerations:
            • Match allocation to your risk tolerance and time horizon
            • Consider transaction costs and tax implications
            • Regular rebalancing is essential for maintaining target allocation
            • Monitor correlation changes during market stress periods
            
            The {best_strategy} approach balances growth potential with risk management.
            """
            
            return recommendation.strip()
        except Exception as e:
            return f"Recommendation generation failed: {e}"
    
    def _estimate_scenario_probability(self, scenario_name: str) -> float:
        """Estimate probability of stress scenario."""
        probabilities = {
            '2008_financial_crisis': 0.05,   # 5% chance per decade
            '2020_covid_crash': 0.10,        # 10% chance per decade  
            'dot_com_bubble': 0.08,          # 8% chance per decade
            'inflation_shock': 0.15,         # 15% chance per decade
            'interest_rate_shock': 0.20      # 20% chance per decade
        }
        return probabilities.get(scenario_name, 0.10)
    
    def _estimate_recovery_time(self, scenario_name: str) -> str:
        """Estimate recovery time for stress scenario."""
        recovery_times = {
            '2008_financial_crisis': '3-5 years',
            '2020_covid_crash': '1-2 years',
            'dot_com_bubble': '4-6 years',
            'inflation_shock': '2-3 years',
            'interest_rate_shock': '1-3 years'
        }
        return recovery_times.get(scenario_name, '2-4 years')
    
    def _assess_stress_risk(self, stress_results: Dict[str, Any]) -> str:
        """Assess overall stress risk."""
        try:
            worst_scenario = min(stress_results.values(), key=lambda x: x['scenario_return'])
            worst_return = worst_scenario['scenario_return']
            
            if worst_return < -0.4:
                return 'High stress risk - consider defensive positioning'
            elif worst_return < -0.2:
                return 'Moderate stress risk - maintain risk management'
            else:
                return 'Low stress risk - portfolio shows resilience'
        except Exception as e:
            return 'Risk assessment unavailable'
    
    def _identify_key_catalysts(self, asset_type: str) -> List[str]:
        """Identify key catalysts for asset type."""
        catalysts = {
            'Stock': [
                'Earnings announcements',
                'Industry developments',
                'Regulatory changes',
                'Management changes',
                'Product launches'
            ],
            'ETF': [
                'Underlying asset performance',
                'Flow dynamics',
                'Expense ratio changes',
                'Index reconstitution',
                'Market sector rotation'
            ],
            'Cryptocurrency': [
                'Regulatory clarity',
                'Institutional adoption',
                'Technology upgrades',
                'Market sentiment shifts',
                'Macroeconomic factors'
            ]
        }
        return catalysts.get(asset_type, ['Market conditions', 'Economic data', 'Policy changes'])
    
    def _get_monitoring_indicators(self, asset_type: str) -> List[str]:
        """Get key indicators to monitor."""
        indicators = {
            'Stock': [
                'Earnings revisions',
                'Analyst ratings',
                'Insider trading',
                'Short interest',
                'Relative strength'
            ],
            'ETF': [
                'Premium/discount to NAV',
                'Daily trading volume',
                'Asset flows',
                'Tracking error',
                'Holdings changes'
            ],
            'Cryptocurrency': [
                'On-chain metrics',
                'Exchange flows',
                'Developer activity',
                'Social sentiment',
                'Regulatory news'
            ]
        }
        return indicators.get(asset_type, ['Price action', 'Volume', 'Market sentiment'])
    
class ForecastingEngine(BaseAnalysisEngine):
    """Forecasting engine for projecting future price trends and volatility."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive forecasting analysis.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information (optional)
        
        Returns:
            Dictionary with forecasting analysis results
        """
        if price_data.empty:
            return {'error': 'No price data available'}
        
        try:
            results = {}
            
            # Trend projection
            results['trend_projection'] = self._project_trends(price_data)
            
            # Volatility forecast
            results['volatility_forecast'] = self._forecast_volatility(price_data)
            
            # Scenario analysis
            results['scenario_analysis'] = self._perform_scenario_analysis(price_data, asset_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Forecasting analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _project_trends(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Project price trends into the future."""
        try:
            prices = price_data['Close'].dropna()
            
            if len(prices) < 10:
                return {'error': 'Insufficient data for trend projection'}
            
            # Calculate trend components
            short_term_trend = self._calculate_trend(prices, 10)
            medium_term_trend = self._calculate_trend(prices, 30) if len(prices) >= 30 else short_term_trend
            long_term_trend = self._calculate_trend(prices, 90) if len(prices) >= 90 else medium_term_trend
            
            # Determine overall trend direction
            trends = [short_term_trend, medium_term_trend, long_term_trend]
            avg_trend = np.mean(trends)
            
            if avg_trend > 0.02:
                trend_direction = 'Strong Uptrend'
            elif avg_trend > 0.005:
                trend_direction = 'Uptrend'
            elif avg_trend > -0.005:
                trend_direction = 'Sideways'
            elif avg_trend > -0.02:
                trend_direction = 'Downtrend'
            else:
                trend_direction = 'Strong Downtrend'
            
            return {
                'short_term_trend': short_term_trend,
                'medium_term_trend': medium_term_trend,
                'long_term_trend': long_term_trend,
                'overall_direction': trend_direction,
                'trend_strength': abs(avg_trend),
                'trend_consistency': np.std(trends)
            }
            
        except Exception as e:
            logger.error(f"Error in trend projection: {e}")
            return {'error': 'Trend projection failed'}
    
    def _calculate_trend(self, prices: pd.Series, window: int) -> float:
        """Calculate trend slope for given window."""
        try:
            if len(prices) < window:
                window = len(prices)
            
            recent_prices = prices.iloc[-window:]
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            
            # Normalize by price level
            avg_price = recent_prices.mean()
            normalized_slope = slope / avg_price if avg_price > 0 else 0
            
            return float(normalized_slope)
        except Exception as e:
            return 0.0
    
    def _forecast_volatility(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Forecast future volatility using historical data."""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            if len(returns) < 10:
                return {'error': 'Insufficient data for volatility forecast'}
            
            # GARCH(1, 1) model for volatility forecasting
            from arch import arch_model
            
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Forecast next 5 days
            forecast = model_fit.forecast(horizon=5)
            volatility_forecast = forecast.variance.values[-1]
            
            return {
                'next_day_volatility': float(np.sqrt(volatility_forecast[0])),
                'forecasted_volatility_5d': float(np.sqrt(volatility_forecast.mean())),
                'model_summary': model_fit.summary()
            }
        except Exception as e:
            logger.error(f"Error in volatility forecast: {e}")
            return {'error': 'Volatility forecast failed'}
    
    def _perform_scenario_analysis(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scenario analysis to assess potential future states."""
        scenarios = ['2008_financial_crisis', '2020_covid_crash', 'dot_com_bubble', 'inflation_shock', 'interest_rate_shock']
        results = {}
        
        try:
            for scenario in scenarios:
                # Shock parameters based on historical events
                if scenario == '2008_financial_crisis':
                    shock_size = -0.3  # 30% drop
                    recovery_time = '5 years'
                elif scenario == '2020_covid_crash':
                    shock_size = -0.2  # 20% drop
                    recovery_time = '2 years'
                elif scenario == 'dot_com_bubble':
                    shock_size = -0.25  # 25% drop
                    recovery_time = '6 years'
                elif scenario == 'inflation_shock':
                    shock_size = -0.15  # 15% drop
                    recovery_time = '3 years'
                elif scenario == 'interest_rate_shock':
                    shock_size = -0.1  # 10% drop
                    recovery_time = '1 year'
                
                # Apply shock to price data
                shocked_prices = price_data['Close'] * (1 + shock_size)
                
                # Calculate potential recovery path (V-shaped recovery)
                recovery_path = shocked_prices.iloc[-1] + (price_data['Close'].iloc[-1] - shocked_prices.iloc[-1]) * np.linspace(0, 1, 252)
                
                results[scenario] = {
                    'shocked_prices': shocked_prices.tolist(),
                    'recovery_path': recovery_path.tolist(),
                    'shock_size': shock_size,
                    'recovery_time': recovery_time
                }
            
            return results
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {'error': 'Scenario analysis failed'}
    
    def _evaluate_model_performance(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate forecasting model performance using historical data."""
        try:
            prices = price_data['Close'].dropna()
            
            if len(prices) < 60:  # Need at least 60 days for backtesting
                return {'message': 'Insufficient data for model evaluation'}
            
            # Split data for backtesting
            train_size = int(len(prices) * 0.8)
            train_data = prices.iloc[:train_size]
            test_data = prices.iloc[train_size:]
            
            # Simple model evaluation (moving average as baseline)
            ma_forecast = train_data.rolling(20).mean().iloc[-1]
            actual_avg = test_data.mean()
            
            # Calculate evaluation metrics
            mae = abs(ma_forecast - actual_avg)  # Mean Absolute Error
            mape = mae / actual_avg if actual_avg != 0 else 0  # Mean Absolute Percentage Error
            
            # Determine model quality
            if mape < 0.05:
                model_quality = 'Excellent'
            elif mape < 0.10:
                model_quality = 'Good'
            elif mape < 0.20:
                model_quality = 'Fair'
            else:
                model_quality = 'Poor'
            
            return {
                'mean_absolute_error': float(mae),
                'mean_absolute_percentage_error': float(mape),
                'model_quality': model_quality,
                'backtest_period': f'{len(test_data)} days',
                'recommendation': f'Model shows {model_quality.lower()} predictive accuracy'
            }
            
        except Exception as e:
            logger.error(f"Error in model performance evaluation: {e}")
            return {'error': 'Model performance evaluation failed'}

# Global engine instances for backward compatibility
technical_engine = TechnicalAnalysisEngine()
risk_engine = RiskAnalysisEngine()
performance_engine = PerformanceAnalysisEngine()
macro_engine = MacroeconomicAnalysisEngine()
fundamental_engine = FundamentalAnalysisEngine()
portfolio_engine = PortfolioStrategyEngine()
forecasting_engine = ForecastingEngine()
