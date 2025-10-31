"""
Volatility Forecasting Module - Priority 3 Implementation
GARCH models and advanced volatility prediction with regime transitions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import GARCH model - if not available, provide fallback
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("arch package not available - using simplified volatility models")

logger = logging.getLogger(__name__)

@dataclass
class VolatilityForecast:
    """Data class for volatility forecast information"""
    forecast_horizon: int
    current_volatility: float
    forecasted_volatility: List[float]
    confidence_intervals: Dict[str, List[float]]
    regime_probability: Dict[str, float]
    model_type: str
    forecast_accuracy: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'

class VolatilityForecaster:
    """Priority 3: Advanced volatility forecasting with GARCH models and regime detection"""
    
    def __init__(self):
        self.volatility_regimes = {
            'low': {'threshold': 0.15, 'description': 'Low volatility regime - stable market conditions'},
            'medium': {'threshold': 0.25, 'description': 'Medium volatility regime - normal market fluctuations'},
            'high': {'threshold': 0.40, 'description': 'High volatility regime - increased market stress'},
            'extreme': {'threshold': float('inf'), 'description': 'Extreme volatility regime - market crisis conditions'}
        }
        
        self.model_params = {
            'garch': {'p': 1, 'q': 1},
            'egarch': {'p': 1, 'o': 1, 'q': 1},
            'gjr_garch': {'p': 1, 'o': 1, 'q': 1}
        }
    
    def generate_volatility_forecast(self, price_data: pd.DataFrame, 
                                   forecast_horizon: int = 30) -> Dict[str, Any]:
        """Generate comprehensive volatility forecast"""
        try:
            if len(price_data) < 100:
                return {'error': 'Insufficient data for volatility forecasting (minimum 100 observations required)'}
            
            # Calculate returns
            returns = price_data['Close'].pct_change().dropna() * 100  # Convert to percentage
            
            # Current volatility metrics
            current_metrics = self._calculate_current_volatility_metrics(returns)
            
            # Generate forecasts using multiple models
            forecasts = {}
            
            if ARCH_AVAILABLE:
                # GARCH model forecast
                garch_forecast = self._fit_garch_model(returns, forecast_horizon)
                forecasts['garch'] = garch_forecast
                
                # EGARCH model forecast (captures asymmetric effects)
                egarch_forecast = self._fit_egarch_model(returns, forecast_horizon)
                forecasts['egarch'] = egarch_forecast
                
                # GJR-GARCH model forecast (threshold effects)
                gjr_forecast = self._fit_gjr_garch_model(returns, forecast_horizon)
                forecasts['gjr_garch'] = gjr_forecast
            
            # Historical volatility models (always available)
            historical_forecast = self._historical_volatility_forecast(returns, forecast_horizon)
            forecasts['historical'] = historical_forecast
            
            ewma_forecast = self._ewma_volatility_forecast(returns, forecast_horizon)
            forecasts['ewma'] = ewma_forecast
            
            # Regime analysis
            regime_analysis = self._analyze_volatility_regimes(returns, current_metrics)
            
            # Ensemble forecast (combine multiple models)
            ensemble_forecast = self._create_ensemble_forecast(forecasts, forecast_horizon)
            
            # Volatility clustering analysis
            clustering_analysis = self._analyze_volatility_clustering(returns)
            
            # Generate insights
            forecast_insights = self._generate_forecast_insights(
                ensemble_forecast, regime_analysis, clustering_analysis, current_metrics
            )
            
            return {
                'current_metrics': current_metrics,
                'individual_forecasts': forecasts,
                'ensemble_forecast': ensemble_forecast,
                'regime_analysis': regime_analysis,
                'clustering_analysis': clustering_analysis,
                'forecast_insights': forecast_insights,
                'forecast_horizon_days': forecast_horizon,
                'model_availability': {
                    'garch_models': ARCH_AVAILABLE,
                    'historical_models': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating volatility forecast: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_current_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate current volatility metrics"""
        try:
            # Annualized volatility (252 trading days)
            current_vol = returns.std() * np.sqrt(252)
            
            # Rolling volatilities
            vol_5d = returns.rolling(5).std().iloc[-1] * np.sqrt(252)
            vol_10d = returns.rolling(10).std().iloc[-1] * np.sqrt(252)
            vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            vol_60d = returns.rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # Volatility of volatility
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            vol_of_vol = rolling_vol.std()
            
            # Volatility persistence (autocorrelation)
            squared_returns = returns ** 2
            persistence = squared_returns.autocorr(lag=1)
            
            # Volatility skewness (asymmetry)
            vol_skew = rolling_vol.skew()
            
            return {
                'current_volatility': current_vol,
                'vol_5d': vol_5d,
                'vol_10d': vol_10d,
                'vol_20d': vol_20d,
                'vol_60d': vol_60d,
                'volatility_of_volatility': vol_of_vol,
                'volatility_persistence': persistence,
                'volatility_skewness': vol_skew,
                'volatility_percentile': self._calculate_volatility_percentile(returns, current_vol)
            }
            
        except Exception as e:
            logger.error(f"Error calculating current volatility metrics: {str(e)}")
            return {}
    
    def _fit_garch_model(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Fit GARCH(1,1) model and generate forecast"""
        try:
            if not ARCH_AVAILABLE:
                return self._create_fallback_forecast(returns, horizon, 'garch_fallback')
            
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Generate forecast
            forecast = fitted_model.forecast(horizon=horizon)
            
            # Extract forecasted volatility (annualized)
            forecasted_vol = np.sqrt(forecast.variance.iloc[-1].values * 252)
            
            # Calculate confidence intervals (approximate)
            std_error = np.sqrt(np.diag(fitted_model.cov)) if hasattr(fitted_model, 'cov') else None
            
            confidence_intervals = {}
            if std_error is not None:
                ci_95_lower = forecasted_vol - 1.96 * std_error[0] * np.sqrt(252)
                ci_95_upper = forecasted_vol + 1.96 * std_error[0] * np.sqrt(252)
                confidence_intervals['95%'] = [ci_95_lower.tolist(), ci_95_upper.tolist()]
            
            # Determine volatility trend
            recent_vol = returns.rolling(20).std().iloc[-5:].mean() * np.sqrt(252)
            vol_trend = 'increasing' if forecasted_vol[0] > recent_vol else 'decreasing' if forecasted_vol[0] < recent_vol * 0.95 else 'stable'
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=returns.std() * np.sqrt(252),
                forecasted_volatility=forecasted_vol.tolist(),
                confidence_intervals=confidence_intervals,
                regime_probability={},  # Will be filled by regime analysis
                model_type='GARCH(1,1)',
                forecast_accuracy=fitted_model.aic if hasattr(fitted_model, 'aic') else 0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            return self._create_fallback_forecast(returns, horizon, 'garch_error')
    
    def _fit_egarch_model(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Fit EGARCH model and generate forecast (captures asymmetric effects)"""
        try:
            if not ARCH_AVAILABLE:
                return self._create_fallback_forecast(returns, horizon, 'egarch_fallback')
            
            # Fit EGARCH model
            model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Generate forecast
            forecast = fitted_model.forecast(horizon=horizon)
            forecasted_vol = np.sqrt(forecast.variance.iloc[-1].values * 252)
            
            # Determine volatility trend
            recent_vol = returns.rolling(20).std().iloc[-5:].mean() * np.sqrt(252)
            vol_trend = 'increasing' if forecasted_vol[0] > recent_vol else 'decreasing' if forecasted_vol[0] < recent_vol * 0.95 else 'stable'
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=returns.std() * np.sqrt(252),
                forecasted_volatility=forecasted_vol.tolist(),
                confidence_intervals={},
                regime_probability={},
                model_type='EGARCH(1,1,1)',
                forecast_accuracy=fitted_model.aic if hasattr(fitted_model, 'aic') else 0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error fitting EGARCH model: {str(e)}")
            return self._create_fallback_forecast(returns, horizon, 'egarch_error')
    
    def _fit_gjr_garch_model(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Fit GJR-GARCH model and generate forecast (threshold effects)"""
        try:
            if not ARCH_AVAILABLE:
                return self._create_fallback_forecast(returns, horizon, 'gjr_fallback')
            
            # Fit GJR-GARCH model
            model = arch_model(returns, vol='GARCH', p=1, o=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Generate forecast
            forecast = fitted_model.forecast(horizon=horizon)
            forecasted_vol = np.sqrt(forecast.variance.iloc[-1].values * 252)
            
            # Determine volatility trend
            recent_vol = returns.rolling(20).std().iloc[-5:].mean() * np.sqrt(252)
            vol_trend = 'increasing' if forecasted_vol[0] > recent_vol else 'decreasing' if forecasted_vol[0] < recent_vol * 0.95 else 'stable'
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=returns.std() * np.sqrt(252),
                forecasted_volatility=forecasted_vol.tolist(),
                confidence_intervals={},
                regime_probability={},
                model_type='GJR-GARCH(1,1,1)',
                forecast_accuracy=fitted_model.aic if hasattr(fitted_model, 'aic') else 0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error fitting GJR-GARCH model: {str(e)}")
            return self._create_fallback_forecast(returns, horizon, 'gjr_error')
    
    def _historical_volatility_forecast(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Generate forecast using historical volatility"""
        try:
            # Calculate rolling volatilities
            vol_windows = [5, 10, 20, 60]
            historical_vols = []
            
            for window in vol_windows:
                if len(returns) >= window:
                    vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
                    historical_vols.append(vol)
            
            # Average of different windows
            avg_historical_vol = np.mean(historical_vols) if historical_vols else returns.std() * np.sqrt(252)
            
            # Simple persistence forecast (assume volatility remains constant)
            forecasted_vol = [avg_historical_vol] * horizon
            
            # Determine trend based on recent volatility changes
            if len(returns) >= 40:
                recent_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                older_vol = returns.rolling(20).std().iloc[-21] * np.sqrt(252)
                vol_trend = 'increasing' if recent_vol > older_vol * 1.05 else 'decreasing' if recent_vol < older_vol * 0.95 else 'stable'
            else:
                vol_trend = 'stable'
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=returns.std() * np.sqrt(252),
                forecasted_volatility=forecasted_vol,
                confidence_intervals={},
                regime_probability={},
                model_type='Historical Volatility',
                forecast_accuracy=0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error in historical volatility forecast: {str(e)}")
            return self._create_fallback_forecast(returns, horizon, 'historical_error')
    
    def _ewma_volatility_forecast(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Generate forecast using Exponentially Weighted Moving Average"""
        try:
            # EWMA with decay factor (common choice: 0.94)
            lambda_decay = 0.94
            
            # Calculate EWMA volatility
            squared_returns = returns ** 2
            ewma_var = squared_returns.ewm(alpha=1-lambda_decay).mean().iloc[-1]
            ewma_vol = np.sqrt(ewma_var * 252)
            
            # Simple persistence forecast
            forecasted_vol = [ewma_vol] * horizon
            
            # Determine trend
            if len(returns) >= 20:
                recent_ewma = np.sqrt(squared_returns.ewm(alpha=1-lambda_decay).mean().iloc[-5:].mean() * 252)
                vol_trend = 'increasing' if ewma_vol > recent_ewma * 1.05 else 'decreasing' if ewma_vol < recent_ewma * 0.95 else 'stable'
            else:
                vol_trend = 'stable'
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=returns.std() * np.sqrt(252),
                forecasted_volatility=forecasted_vol,
                confidence_intervals={},
                regime_probability={},
                model_type='EWMA',
                forecast_accuracy=0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error in EWMA volatility forecast: {str(e)}")
            return self._create_fallback_forecast(returns, horizon, 'ewma_error')
    
    def _analyze_volatility_regimes(self, returns: pd.Series, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze volatility regimes and transition probabilities"""
        try:
            current_vol = current_metrics.get('current_volatility', 0)
            
            # Determine current regime
            current_regime = 'low'
            for regime, info in self.volatility_regimes.items():
                if current_vol <= info['threshold']:
                    current_regime = regime
                    break
            
            # Calculate historical regime frequencies
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            regime_history = []
            
            for vol in rolling_vol.dropna():
                for regime, info in self.volatility_regimes.items():
                    if vol <= info['threshold']:
                        regime_history.append(regime)
                        break
            
            # Calculate regime frequencies
            regime_freq = {}
            total_periods = len(regime_history)
            
            for regime in self.volatility_regimes.keys():
                count = regime_history.count(regime)
                regime_freq[regime] = count / total_periods if total_periods > 0 else 0
            
            # Calculate transition probabilities (simple Markov chain)
            transition_probs = self._calculate_regime_transitions(regime_history)
            
            # Persistence analysis
            regime_persistence = self._calculate_regime_persistence(regime_history)
            
            return {
                'current_regime': current_regime,
                'regime_description': self.volatility_regimes[current_regime]['description'],
                'regime_frequencies': regime_freq,
                'transition_probabilities': transition_probs,
                'regime_persistence': regime_persistence,
                'volatility_percentile': current_metrics.get('volatility_percentile', 50)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility regimes: {str(e)}")
            return {}
    
    def _calculate_regime_transitions(self, regime_history: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities"""
        try:
            transitions = {}
            
            for regime in self.volatility_regimes.keys():
                transitions[regime] = {target: 0 for target in self.volatility_regimes.keys()}
            
            # Count transitions
            for i in range(len(regime_history) - 1):
                current = regime_history[i]
                next_regime = regime_history[i + 1]
                transitions[current][next_regime] += 1
            
            # Convert to probabilities
            for current_regime in transitions:
                total_transitions = sum(transitions[current_regime].values())
                if total_transitions > 0:
                    for target_regime in transitions[current_regime]:
                        transitions[current_regime][target_regime] /= total_transitions
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error calculating regime transitions: {str(e)}")
            return {}
    
    def _calculate_regime_persistence(self, regime_history: List[str]) -> Dict[str, float]:
        """Calculate how long each regime typically persists"""
        try:
            persistence = {}
            
            for regime in self.volatility_regimes.keys():
                regime_runs = []
                current_run = 0
                
                for hist_regime in regime_history:
                    if hist_regime == regime:
                        current_run += 1
                    else:
                        if current_run > 0:
                            regime_runs.append(current_run)
                        current_run = 0
                
                # Don't forget the last run
                if current_run > 0:
                    regime_runs.append(current_run)
                
                persistence[regime] = np.mean(regime_runs) if regime_runs else 0
            
            return persistence
            
        except Exception as e:
            logger.error(f"Error calculating regime persistence: {str(e)}")
            return {}
    
    def _analyze_volatility_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility clustering patterns"""
        try:
            # Calculate absolute returns (proxy for volatility)
            abs_returns = np.abs(returns)
            
            # Autocorrelation of absolute returns (volatility clustering)
            clustering_lags = [1, 2, 3, 5, 10, 20]
            autocorrelations = {}
            
            for lag in clustering_lags:
                autocorr = abs_returns.autocorr(lag=lag)
                autocorrelations[f'lag_{lag}'] = autocorr if pd.notna(autocorr) else 0
            
            # ARCH effect test (simplified)
            squared_returns = returns ** 2
            arch_lags = [5, 10]
            arch_effects = {}
            
            for lag in arch_lags:
                if len(squared_returns) > lag:
                    arch_autocorr = squared_returns.autocorr(lag=lag)
                    arch_effects[f'lag_{lag}'] = arch_autocorr if pd.notna(arch_autocorr) else 0
            
            # Volatility clustering strength
            avg_autocorr = np.mean(list(autocorrelations.values()))
            clustering_strength = 'strong' if avg_autocorr > 0.1 else 'moderate' if avg_autocorr > 0.05 else 'weak'
            
            return {
                'volatility_autocorrelations': autocorrelations,
                'arch_effects': arch_effects,
                'clustering_strength': clustering_strength,
                'average_clustering': avg_autocorr
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility clustering: {str(e)}")
            return {}
    
    def _create_ensemble_forecast(self, forecasts: Dict[str, VolatilityForecast], 
                                horizon: int) -> VolatilityForecast:
        """Create ensemble forecast combining multiple models"""
        try:
            if not forecasts:
                return self._create_fallback_forecast(pd.Series([0]), horizon, 'no_forecasts')
            
            # Weights for different models (can be optimized based on historical accuracy)
            model_weights = {
                'garch': 0.3,
                'egarch': 0.25,
                'gjr_garch': 0.25,
                'historical': 0.1,
                'ewma': 0.1
            }
            
            # Adjust weights based on available models
            available_models = list(forecasts.keys())
            adjusted_weights = {}
            total_weight = 0
            
            for model in available_models:
                weight = model_weights.get(model, 0.1)
                adjusted_weights[model] = weight
                total_weight += weight
            
            # Normalize weights
            for model in adjusted_weights:
                adjusted_weights[model] /= total_weight
            
            # Combine forecasts
            ensemble_vol = []
            for i in range(horizon):
                weighted_vol = 0
                for model, forecast in forecasts.items():
                    if i < len(forecast.forecasted_volatility):
                        weighted_vol += adjusted_weights[model] * forecast.forecasted_volatility[i]
                ensemble_vol.append(weighted_vol)
            
            # Determine ensemble trend
            if len(ensemble_vol) > 1:
                trend_slope = (ensemble_vol[-1] - ensemble_vol[0]) / len(ensemble_vol)
                if trend_slope > 0.001:
                    vol_trend = 'increasing'
                elif trend_slope < -0.001:
                    vol_trend = 'decreasing'
                else:
                    vol_trend = 'stable'
            else:
                vol_trend = 'stable'
            
            # Get current volatility from first available forecast
            current_vol = list(forecasts.values())[0].current_volatility
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=current_vol,
                forecasted_volatility=ensemble_vol,
                confidence_intervals={},
                regime_probability={},
                model_type='Ensemble',
                forecast_accuracy=0.0,
                volatility_trend=vol_trend
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {str(e)}")
            return self._create_fallback_forecast(pd.Series([0]), horizon, 'ensemble_error')
    
    def _create_fallback_forecast(self, returns: pd.Series, horizon: int, 
                                fallback_type: str) -> VolatilityForecast:
        """Create fallback forecast when models fail"""
        try:
            if len(returns) > 0:
                current_vol = returns.std() * np.sqrt(252)
                forecasted_vol = [current_vol] * horizon
            else:
                current_vol = 0.20  # Default 20% volatility
                forecasted_vol = [current_vol] * horizon
            
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=current_vol,
                forecasted_volatility=forecasted_vol,
                confidence_intervals={},
                regime_probability={},
                model_type=f'Fallback ({fallback_type})',
                forecast_accuracy=0.0,
                volatility_trend='stable'
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback forecast: {str(e)}")
            return VolatilityForecast(
                forecast_horizon=horizon,
                current_volatility=0.20,
                forecasted_volatility=[0.20] * horizon,
                confidence_intervals={},
                regime_probability={},
                model_type='Error Fallback',
                forecast_accuracy=0.0,
                volatility_trend='stable'
            )
    
    def _calculate_volatility_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """Calculate percentile rank of current volatility"""
        try:
            if len(returns) < 60:
                return 50.0  # Default to median if insufficient data
            
            # Calculate rolling volatilities
            rolling_vols = returns.rolling(20).std() * np.sqrt(252)
            rolling_vols = rolling_vols.dropna()
            
            if len(rolling_vols) == 0:
                return 50.0
            
            # Calculate percentile
            percentile = (rolling_vols < current_vol).mean() * 100
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {str(e)}")
            return 50.0
    
    def _generate_forecast_insights(self, ensemble_forecast: VolatilityForecast,
                                  regime_analysis: Dict[str, Any],
                                  clustering_analysis: Dict[str, Any],
                                  current_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable insights from volatility forecast"""
        insights = []
        
        try:
            current_vol = current_metrics.get('current_volatility', 0)
            vol_percentile = current_metrics.get('volatility_percentile', 50)
            
            # Current volatility level insight
            if vol_percentile > 80:
                insights.append(f"⚡ Current volatility ({current_vol:.1%}) is at {vol_percentile:.0f}th percentile - unusually high levels")
            elif vol_percentile < 20:
                insights.append(f"😌 Current volatility ({current_vol:.1%}) is at {vol_percentile:.0f}th percentile - unusually calm conditions")
            else:
                insights.append(f"📊 Current volatility ({current_vol:.1%}) is at {vol_percentile:.0f}th percentile - normal range")
            
            # Forecast trend insight
            vol_trend = ensemble_forecast.volatility_trend
            if vol_trend == 'increasing':
                insights.append("📈 Volatility forecast trend: INCREASING - expect higher price swings ahead")
            elif vol_trend == 'decreasing':
                insights.append("📉 Volatility forecast trend: DECREASING - expect calmer market conditions")
            else:
                insights.append("➡️ Volatility forecast trend: STABLE - expect similar volatility levels")
            
            # Regime analysis insight
            current_regime = regime_analysis.get('current_regime', 'medium')
            regime_desc = regime_analysis.get('regime_description', '')
            if current_regime == 'extreme':
                insights.append(f"🚨 Current regime: {current_regime.upper()} - {regime_desc}")
            elif current_regime == 'high':
                insights.append(f"⚠️ Current regime: {current_regime.upper()} - {regime_desc}")
            elif current_regime == 'low':
                insights.append(f"✅ Current regime: {current_regime.upper()} - {regime_desc}")
            else:
                insights.append(f"📊 Current regime: {current_regime.upper()} - {regime_desc}")
            
            # Volatility clustering insight
            clustering_strength = clustering_analysis.get('clustering_strength', 'moderate')
            if clustering_strength == 'strong':
                insights.append("🔗 Strong volatility clustering detected - high volatility periods tend to persist")
            elif clustering_strength == 'weak':
                insights.append("🔀 Weak volatility clustering - volatility changes are more random")
            
            # Model availability insight
            if ARCH_AVAILABLE:
                insights.append("🎯 Advanced GARCH models available - high-quality volatility forecasts")
            else:
                insights.append("📈 Using historical models - consider installing 'arch' package for advanced forecasts")
            
        except Exception as e:
            logger.error(f"Error generating forecast insights: {str(e)}")
            insights.append("⚠️ Unable to generate forecast insights due to analysis limitations")
        
        return insights
