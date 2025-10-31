"""
Market Regime Detection and Analysis
Identify market conditions and adapt analysis accordingly
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer:
    """Advanced market regime detection and analysis"""
    
    def __init__(self):
        self.regime_model = None
        self.scaler = StandardScaler()
        
    def analyze_market_regimes(self, price_data: pd.DataFrame, 
                             market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """Comprehensive market regime analysis"""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            regime_analysis = {
                'statistical_regimes': self._detect_statistical_regimes(returns),
                'volatility_regimes': self._detect_volatility_regimes(returns),
                'trend_regimes': self._detect_trend_regimes(price_data['Close']),
                'market_cycle_position': self._analyze_market_cycle_position(returns),
                'regime_transitions': self._analyze_regime_transitions(returns),
                'current_regime_characteristics': self._characterize_current_regime(returns)
            }
            
            if market_data:
                regime_analysis['macro_regime_context'] = self._analyze_macro_context(
                    returns, market_data
                )
            
            # Generate regime-specific recommendations
            regime_analysis['regime_recommendations'] = self._generate_regime_recommendations(
                regime_analysis
            )
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {str(e)}")
            return {'error': str(e)}
    
    def _detect_statistical_regimes(self, returns: pd.Series, n_regimes: int = 3) -> Dict[str, Any]:
        """Use Gaussian Mixture Models to detect statistical regimes"""
        try:
            # Prepare features for regime detection
            features = self._prepare_regime_features(returns)
            
            if len(features) < 50:  # Need sufficient data
                return {'error': 'Insufficient data for regime detection'}
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            regime_labels = gmm.fit_predict(features)
            
            # Analyze each regime
            regimes = {}
            for regime_id in range(n_regimes):
                regime_mask = regime_labels == regime_id
                regime_returns = returns.iloc[len(returns) - len(regime_labels):][regime_mask]
                
                if len(regime_returns) > 0:
                    regimes[f'Regime_{regime_id}'] = {
                        'frequency': regime_mask.sum() / len(regime_mask) * 100,
                        'avg_return': regime_returns.mean() * 252 * 100,  # Annualized %
                        'volatility': regime_returns.std() * np.sqrt(252) * 100,
                        'sharpe_ratio': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() != 0 else 0,
                        'max_drawdown': self._calculate_regime_max_drawdown(regime_returns),
                        'regime_type': self._classify_regime_type(regime_returns)
                    }
            
            # Determine current regime
            current_regime = regime_labels[-1] if len(regime_labels) > 0 else None
            
            return {
                'regimes': regimes,
                'current_regime': f'Regime_{current_regime}' if current_regime is not None else 'Unknown',
                'regime_probabilities': gmm.predict_proba(features[-1:]).flatten().tolist() if len(features) > 0 else [],
                'model_score': gmm.score(features)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical regime detection: {str(e)}")
            return {'error': str(e)}
    
    def _detect_volatility_regimes(self, returns: pd.Series, window: int = 30) -> Dict[str, Any]:
        """Detect high/low volatility regimes"""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            
            # Define volatility thresholds (percentiles)
            low_vol_threshold = rolling_vol.quantile(0.33)
            high_vol_threshold = rolling_vol.quantile(0.67)
            
            # Classify volatility regimes
            vol_regimes = []
            for vol in rolling_vol:
                if pd.isna(vol):
                    vol_regimes.append('Unknown')
                elif vol <= low_vol_threshold:
                    vol_regimes.append('Low_Volatility')
                elif vol >= high_vol_threshold:
                    vol_regimes.append('High_Volatility')
                else:
                    vol_regimes.append('Normal_Volatility')
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in ['Low_Volatility', 'Normal_Volatility', 'High_Volatility']:
                regime_periods = [i for i, r in enumerate(vol_regimes) if r == regime]
                if regime_periods:
                    regime_returns = returns.iloc[regime_periods]
                    regime_stats[regime] = {
                        'frequency': len(regime_periods) / len(vol_regimes) * 100,
                        'avg_return': regime_returns.mean() * 252 * 100,
                        'avg_volatility': rolling_vol.iloc[regime_periods].mean(),
                        'avg_duration': self._calculate_avg_regime_duration(vol_regimes, regime)
                    }
            
            current_vol_regime = vol_regimes[-1] if vol_regimes else 'Unknown'
            
            return {
                'current_regime': current_vol_regime,
                'regime_statistics': regime_stats,
                'volatility_trend': self._analyze_volatility_trend(rolling_vol),
                'regime_persistence': self._calculate_regime_persistence(vol_regimes)
            }
            
        except Exception as e:
            logger.error(f"Error in volatility regime detection: {str(e)}")
            return {'error': str(e)}
    
    def _detect_trend_regimes(self, prices: pd.Series, short_window: int = 50, 
                            long_window: int = 200) -> Dict[str, Any]:
        """Detect trend-based market regimes"""
        try:
            # Calculate moving averages
            sma_short = prices.rolling(short_window).mean()
            sma_long = prices.rolling(long_window).mean()
            
            # Determine trend regimes
            trend_regimes = []
            for i in range(len(prices)):
                if pd.isna(sma_short.iloc[i]) or pd.isna(sma_long.iloc[i]):
                    trend_regimes.append('Unknown')
                elif prices.iloc[i] > sma_short.iloc[i] > sma_long.iloc[i]:
                    trend_regimes.append('Strong_Uptrend')
                elif prices.iloc[i] > sma_short.iloc[i]:
                    trend_regimes.append('Uptrend')
                elif prices.iloc[i] < sma_short.iloc[i] < sma_long.iloc[i]:
                    trend_regimes.append('Strong_Downtrend')
                elif prices.iloc[i] < sma_short.iloc[i]:
                    trend_regimes.append('Downtrend')
                else:
                    trend_regimes.append('Sideways')
            
            # Calculate trend regime statistics
            returns = prices.pct_change().dropna()
            regime_stats = {}
            
            for regime in ['Strong_Uptrend', 'Uptrend', 'Sideways', 'Downtrend', 'Strong_Downtrend']:
                regime_indices = [i for i, r in enumerate(trend_regimes) if r == regime]
                if regime_indices and len(regime_indices) > 1:
                    regime_returns = returns.iloc[regime_indices]
                    regime_stats[regime] = {
                        'frequency': len(regime_indices) / len(trend_regimes) * 100,
                        'avg_return': regime_returns.mean() * 252 * 100,
                        'win_rate': (regime_returns > 0).mean() * 100,
                        'avg_duration': self._calculate_avg_regime_duration(trend_regimes, regime)
                    }
            
            current_trend_regime = trend_regimes[-1] if trend_regimes else 'Unknown'
            
            return {
                'current_regime': current_trend_regime,
                'regime_statistics': regime_stats,
                'trend_strength': self._calculate_trend_strength(prices, sma_short, sma_long),
                'regime_history': trend_regimes[-20:]  # Last 20 periods
            }
            
        except Exception as e:
            logger.error(f"Error in trend regime detection: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_market_cycle_position(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze position in market cycle (Early Bull, Late Bull, Early Bear, Late Bear)"""
        try:
            # Calculate multiple timeframe indicators
            short_momentum = returns.rolling(30).mean() * 252 * 100
            long_momentum = returns.rolling(120).mean() * 252 * 100
            volatility = returns.rolling(30).std() * np.sqrt(252) * 100
            
            current_short_mom = short_momentum.iloc[-1] if not short_momentum.empty else 0
            current_long_mom = long_momentum.iloc[-1] if not long_momentum.empty else 0
            current_vol = volatility.iloc[-1] if not volatility.empty else 0
            
            # Determine cycle position
            if current_short_mom > 0 and current_long_mom > 0:
                if current_vol < volatility.median():
                    cycle_position = "Early_Bull_Market"
                else:
                    cycle_position = "Late_Bull_Market"
            elif current_short_mom < 0 and current_long_mom < 0:
                if current_vol > volatility.median():
                    cycle_position = "Early_Bear_Market"
                else:
                    cycle_position = "Late_Bear_Market"
            else:
                cycle_position = "Transition_Phase"
            
            return {
                'current_position': cycle_position,
                'short_term_momentum': current_short_mom,
                'long_term_momentum': current_long_mom,
                'volatility_percentile': (current_vol > volatility).mean() * 100,
                'cycle_confidence': self._calculate_cycle_confidence(
                    current_short_mom, current_long_mom, current_vol, volatility.median()
                )
            }
            
        except Exception as e:
            logger.error(f"Error in market cycle analysis: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_regime_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for regime detection"""
        try:
            # Calculate various features
            features_df = pd.DataFrame(index=returns.index)
            
            # Return-based features
            features_df['returns'] = returns
            features_df['volatility'] = returns.rolling(20).std()
            features_df['skewness'] = returns.rolling(60).skew()
            features_df['kurtosis'] = returns.rolling(60).kurt()
            
            # Momentum features
            features_df['momentum_5'] = returns.rolling(5).mean()
            features_df['momentum_20'] = returns.rolling(20).mean()
            features_df['momentum_60'] = returns.rolling(60).mean()
            
            # Volatility clustering
            features_df['vol_regime'] = (features_df['volatility'] > features_df['volatility'].rolling(100).median()).astype(int)
            
            # Drop NaN values and scale features
            features_clean = features_df.dropna()
            if len(features_clean) > 0:
                features_scaled = self.scaler.fit_transform(features_clean)
                return features_scaled
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"Error preparing regime features: {str(e)}")
            return np.array([])
    
    def _generate_regime_recommendations(self, regime_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on regime analysis"""
        recommendations = []
        
        try:
            # Volatility regime recommendations
            if 'volatility_regimes' in regime_analysis:
                vol_regime = regime_analysis['volatility_regimes'].get('current_regime', 'Unknown')
                
                if vol_regime == 'High_Volatility':
                    recommendations.append("🌪️ High volatility regime detected - Consider reducing position sizes and increasing cash allocation")
                    recommendations.append("⚠️ Use wider stop-losses and expect more frequent false signals")
                elif vol_regime == 'Low_Volatility':
                    recommendations.append("😴 Low volatility regime - Consider increasing position sizes cautiously")
                    recommendations.append("📈 Good environment for trend-following strategies")
            
            # Trend regime recommendations
            if 'trend_regimes' in regime_analysis:
                trend_regime = regime_analysis['trend_regimes'].get('current_regime', 'Unknown')
                
                if 'Strong_Uptrend' in trend_regime:
                    recommendations.append("🚀 Strong uptrend - Momentum strategies favored, avoid contrarian plays")
                elif 'Strong_Downtrend' in trend_regime:
                    recommendations.append("📉 Strong downtrend - Focus on risk management, consider defensive positioning")
                elif 'Sideways' in trend_regime:
                    recommendations.append("↔️ Sideways market - Range-bound strategies, mean reversion approaches favored")
            
            # Market cycle recommendations
            if 'market_cycle_position' in regime_analysis:
                cycle_pos = regime_analysis['market_cycle_position'].get('current_position', 'Unknown')
                
                if 'Early_Bull' in cycle_pos:
                    recommendations.append("🐂 Early bull market - Aggressive growth positioning, sector rotation opportunities")
                elif 'Late_Bull' in cycle_pos:
                    recommendations.append("🚨 Late bull market - Take profits, reduce risk, prepare for volatility")
                elif 'Early_Bear' in cycle_pos:
                    recommendations.append("🐻 Early bear market - Capital preservation priority, avoid catching falling knives")
                elif 'Late_Bear' in cycle_pos:
                    recommendations.append("💰 Late bear market - Prepare for opportunities, quality at discount prices")
            
        except Exception as e:
            logger.error(f"Error generating regime recommendations: {str(e)}")
            recommendations.append("⚠️ Unable to generate regime-specific recommendations")
        
        return recommendations
    
    def _classify_regime_type(self, returns: pd.Series) -> str:
        """Classify regime based on return characteristics"""
        try:
            avg_return = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
            
            if avg_return > 0.05 and volatility < 0.15:
                return "Low_Risk_Growth"
            elif avg_return > 0.05 and volatility >= 0.15:
                return "High_Risk_Growth"
            elif avg_return <= 0.05 and volatility < 0.15:
                return "Low_Risk_Stable"
            else:
                return "High_Risk_Volatile"
        except:
            return "Unknown"
    
    def _calculate_avg_regime_duration(self, regime_list: List[str], regime: str) -> float:
        """Calculate average duration of a specific regime"""
        try:
            durations = []
            current_duration = 0
            
            for r in regime_list:
                if r == regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            return np.mean(durations) if durations else 0
        except:
            return 0
