"""
Historical Pattern Analysis for Hidden Gems Scanner
Analyzes similarity to historical multi-bagger patterns for pattern matching
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import warnings

logger = logging.getLogger(__name__)

@dataclass
class HistoricalPattern:
    """Structure for historical multi-bagger pattern data"""
    ticker: str
    name: str
    sector: str
    period_start: str
    period_end: str
    entry_price: float
    peak_price: float
    gain_percent: float
    pattern_duration_months: int
    
    # Pattern characteristics
    base_formation_months: int
    breakout_volume_multiplier: float
    sector_tailwinds: str
    fundamental_catalyst: str
    technical_setup: str
    
    # Market conditions during pattern
    market_environment: str
    interest_rate_environment: str
    sector_rotation: str
    
    # Key metrics at pattern start
    market_cap_at_entry: float
    revenue_growth_at_entry: float
    institutional_ownership_at_entry: float
    analyst_coverage_at_entry: int
    
    # Pattern validation
    pattern_confidence: float  # 0-1 score
    replication_difficulty: str  # Easy, Medium, Hard

class HistoricalPatternAnalyzer:
    """Analyzes current opportunities against historical multi-bagger patterns"""
    
    def __init__(self):
        """Initialize with historical pattern database"""
        self.historical_patterns = self._build_pattern_database()
        self.scaler = StandardScaler()
        
    def _build_pattern_database(self) -> List[HistoricalPattern]:
        """Build database of historical multi-bagger patterns"""
        
        patterns = [
            # IREN - Bitcoin Mining Infrastructure
            HistoricalPattern(
                ticker="IREN",
                name="Iris Energy Ltd",
                sector="Bitcoin Mining",
                period_start="2023-01-01",
                period_end="2024-03-01",
                entry_price=3.0,
                peak_price=18.0,
                gain_percent=500.0,
                pattern_duration_months=14,
                base_formation_months=6,
                breakout_volume_multiplier=3.5,
                sector_tailwinds="Bitcoin halving cycle, institutional adoption",
                fundamental_catalyst="Mining capacity expansion, operational efficiency gains",
                technical_setup="Cup and handle, accumulation base, volume breakout",
                market_environment="Bull market recovery",
                interest_rate_environment="Peak rates, dovish pivot",
                sector_rotation="Into risk assets, crypto adoption",
                market_cap_at_entry=150e6,
                revenue_growth_at_entry=1.5,  # 150% growth
                institutional_ownership_at_entry=0.25,
                analyst_coverage_at_entry=8,
                pattern_confidence=0.95,
                replication_difficulty="Medium"
            ),
            
            # NVDA - AI Revolution (2022-2024)
            HistoricalPattern(
                ticker="NVDA",
                name="NVIDIA Corporation",
                sector="Semiconductors",
                period_start="2022-10-01",
                period_end="2024-06-01",
                entry_price=110.0,
                peak_price=950.0,
                gain_percent=763.0,
                pattern_duration_months=20,
                base_formation_months=8,
                breakout_volume_multiplier=2.8,
                sector_tailwinds="AI revolution, ChatGPT launch, enterprise AI adoption",
                fundamental_catalyst="Data center revenue explosion, AI chip monopoly",
                technical_setup="Double bottom, momentum breakout, parabolic advance",
                market_environment="Bear to bull transition",
                interest_rate_environment="Rising to peak rates",
                sector_rotation="Into AI/tech leadership",
                market_cap_at_entry=270e9,
                revenue_growth_at_entry=0.02,  # 2% (cyclical trough)
                institutional_ownership_at_entry=0.65,
                analyst_coverage_at_entry=45,
                pattern_confidence=0.98,
                replication_difficulty="Hard"
            ),
            
            # TSLA - EV Adoption (2019-2021)
            HistoricalPattern(
                ticker="TSLA",
                name="Tesla Inc",
                sector="Electric Vehicles",
                period_start="2019-06-01",
                period_end="2021-01-01",
                entry_price=35.0,  # Split-adjusted
                peak_price=414.0,  # Split-adjusted
                gain_percent=1083.0,
                pattern_duration_months=19,
                base_formation_months=12,
                breakout_volume_multiplier=4.2,
                sector_tailwinds="EV adoption acceleration, regulatory support",
                fundamental_catalyst="Model 3 production ramp, profitability achieved",
                technical_setup="Multi-year base, momentum breakout, parabolic run",
                market_environment="Bull market, pandemic recovery",
                interest_rate_environment="Zero interest rates",
                sector_rotation="Into growth, ESG themes",
                market_cap_at_entry=63e9,
                revenue_growth_at_entry=0.15,  # 15%
                institutional_ownership_at_entry=0.42,
                analyst_coverage_at_entry=32,
                pattern_confidence=0.92,
                replication_difficulty="Hard"
            ),
            
            # SHOP - E-commerce Platform (2016-2021)
            HistoricalPattern(
                ticker="SHOP",
                name="Shopify Inc",
                sector="E-commerce Software",
                period_start="2016-05-01",
                period_end="2021-11-01",
                entry_price=28.0,
                peak_price=176.0,
                gain_percent=529.0,
                pattern_duration_months=66,
                base_formation_months=3,
                breakout_volume_multiplier=2.1,
                sector_tailwinds="E-commerce adoption, SMB digitization",
                fundamental_catalyst="Platform expansion, subscription growth",
                technical_setup="Early stage growth, consistent uptrend",
                market_environment="Bull market, low rates",
                interest_rate_environment="Ultra-low rates",
                sector_rotation="Into growth tech",
                market_cap_at_entry=3.2e9,
                revenue_growth_at_entry=0.90,  # 90%
                institutional_ownership_at_entry=0.35,
                analyst_coverage_at_entry=18,
                pattern_confidence=0.88,
                replication_difficulty="Medium"
            ),
            
            # MSTR - Bitcoin Treasury Strategy (2020-2021)
            HistoricalPattern(
                ticker="MSTR",
                name="MicroStrategy Inc",
                sector="Business Intelligence/Bitcoin",
                period_start="2020-08-01",
                period_end="2021-11-01",
                entry_price=135.0,
                peak_price=1315.0,
                gain_percent=874.0,
                pattern_duration_months=15,
                base_formation_months=4,
                breakout_volume_multiplier=5.8,
                sector_tailwinds="Bitcoin adoption, corporate treasury diversification",
                fundamental_catalyst="Bitcoin treasury strategy, leverage play",
                technical_setup="Breakout from base, momentum acceleration",
                market_environment="Pandemic liquidity, risk-on",
                interest_rate_environment="Zero rates, QE",
                sector_rotation="Into alternative assets",
                market_cap_at_entry=1.4e9,
                revenue_growth_at_entry=-0.02,  # -2% (legacy business declining)
                institutional_ownership_at_entry=0.58,
                analyst_coverage_at_entry=12,
                pattern_confidence=0.85,
                replication_difficulty="Hard"
            ),
            
            # ROKU - Streaming Wars (2017-2021)
            HistoricalPattern(
                ticker="ROKU",
                name="Roku Inc",
                sector="Streaming/Connected TV",
                period_start="2017-09-01",
                period_end="2021-02-01",
                entry_price=14.0,
                peak_price=180.0,
                gain_percent=1186.0,
                pattern_duration_months=41,
                base_formation_months=6,
                breakout_volume_multiplier=3.1,
                sector_tailwinds="Cord cutting, streaming adoption, pandemic viewing",
                fundamental_catalyst="Platform monetization, advertising growth",
                technical_setup="IPO base, growth breakout, pandemic acceleration",
                market_environment="Bull market, pandemic winners",
                interest_rate_environment="Ultra-low rates",
                sector_rotation="Into secular growth themes",
                market_cap_at_entry=1.5e9,
                revenue_growth_at_entry=0.45,  # 45%
                institutional_ownership_at_entry=0.28,
                analyst_coverage_at_entry=15,
                pattern_confidence=0.90,
                replication_difficulty="Medium"
            )
        ]
        
        return patterns
    
    def find_similar_patterns(self, current_ticker: str, current_data: Dict[str, Any], 
                            top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find historical patterns most similar to current opportunity.
        
        Args:
            current_ticker: Current ticker being analyzed
            current_data: Comprehensive data for current ticker
            top_n: Number of top similar patterns to return
            
        Returns:
            List of similar patterns with similarity scores
        """
        try:
            # Extract current characteristics
            current_features = self._extract_pattern_features(current_ticker, current_data)
            
            if not current_features:
                logger.warning(f"Could not extract features for {current_ticker}")
                return []
            
            # Calculate similarity to each historical pattern
            similarities = []
            
            for pattern in self.historical_patterns:
                try:
                    # Extract historical pattern features
                    historical_features = self._extract_historical_features(pattern)
                    
                    # Calculate similarity score
                    similarity_score = self._calculate_pattern_similarity(
                        current_features, historical_features
                    )
                    
                    # Create result entry
                    similar_pattern = {
                        'historical_ticker': pattern.ticker,
                        'historical_name': pattern.name,
                        'similarity_score': similarity_score,
                        'gain_achieved': pattern.gain_percent,
                        'pattern_duration_months': pattern.pattern_duration_months,
                        'sector_tailwinds': pattern.sector_tailwinds,
                        'fundamental_catalyst': pattern.fundamental_catalyst,
                        'technical_setup': pattern.technical_setup,
                        'pattern_confidence': pattern.pattern_confidence,
                        'replication_difficulty': pattern.replication_difficulty,
                        'key_similarities': self._identify_key_similarities(
                            current_features, historical_features, pattern
                        ),
                        'pattern_differences': self._identify_pattern_differences(
                            current_features, historical_features, pattern
                        )
                    }
                    
                    similarities.append(similar_pattern)
                    
                except Exception as e:
                    logger.error(f"Error comparing with pattern {pattern.ticker}: {e}")
                    continue
            
            # Sort by similarity score and return top N
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar patterns for {current_ticker}: {e}")
            return []
    
    def _extract_pattern_features(self, ticker: str, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract pattern features from current ticker data"""
        try:
            # Market data
            market_data = data.get('market_data', {})
            fundamental_data = data.get('fundamental_data', {})
            price_data = data.get('price_data', pd.DataFrame())
            info = data.get('info', {})
            
            # Extract key features for pattern matching
            features = {
                # Market characteristics
                'market_cap': market_data.get('market_cap', 0),
                'institutional_ownership': market_data.get('held_percent_institutions', 0),
                'analyst_coverage': 10,  # Default estimate
                'beta': market_data.get('beta', 1.0),
                
                # Fundamental characteristics
                'revenue_growth': fundamental_data.get('revenue_growth_yoy', 0),
                'gross_margin': fundamental_data.get('gross_margin', 0),
                'operating_margin': fundamental_data.get('operating_margin', 0),
                'debt_to_equity': fundamental_data.get('debt_to_equity', 0) / 100 if fundamental_data.get('debt_to_equity') else 0,
                'pe_ratio': fundamental_data.get('pe_ratio', 0),
                
                # Technical characteristics
                'volatility': 0.0,
                'momentum_6m': 0.0,
                'relative_strength': 1.0,
                'volume_trend': 1.0
            }
            
            # Calculate technical features from price data
            if not price_data.empty and len(price_data) >= 126:  # 6 months of data
                returns = price_data['Close'].pct_change().dropna()
                
                # Volatility (annualized)
                features['volatility'] = returns.std() * np.sqrt(252)
                
                # 6-month momentum
                if len(price_data) >= 126:
                    price_6m_ago = price_data['Close'].iloc[-126]
                    current_price = price_data['Close'].iloc[-1]
                    features['momentum_6m'] = (current_price - price_6m_ago) / price_6m_ago
                
                # Volume trend (recent vs historical average)
                if 'Volume' in price_data.columns:
                    recent_volume = price_data['Volume'].tail(30).mean()
                    historical_volume = price_data['Volume'].mean()
                    features['volume_trend'] = recent_volume / historical_volume if historical_volume > 0 else 1.0
            
            # Normalize market cap for comparison (log scale)
            if features['market_cap'] > 0:
                features['log_market_cap'] = np.log10(features['market_cap'])
            else:
                features['log_market_cap'] = 8.0  # ~$100M default
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features for {ticker}: {e}")
            return None
    
    def _extract_historical_features(self, pattern: HistoricalPattern) -> Dict[str, float]:
        """Extract features from historical pattern for comparison"""
        try:
            features = {
                # Market characteristics at entry
                'log_market_cap': np.log10(pattern.market_cap_at_entry),
                'institutional_ownership': pattern.institutional_ownership_at_entry,
                'analyst_coverage': pattern.analyst_coverage_at_entry,
                'beta': 1.2,  # Estimated for growth stocks
                
                # Fundamental characteristics
                'revenue_growth': pattern.revenue_growth_at_entry,
                'gross_margin': 0.35,  # Estimated average
                'operating_margin': 0.15,  # Estimated average
                'debt_to_equity': 0.3,  # Estimated average
                'pe_ratio': 50.0,  # Estimated for growth stocks
                
                # Technical characteristics (estimated based on pattern)
                'volatility': 0.6,  # High volatility for multi-baggers
                'momentum_6m': 0.2,  # Positive momentum before breakout
                'relative_strength': 1.3,  # Outperforming market
                'volume_trend': pattern.breakout_volume_multiplier / 2.0  # Normalized
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting historical features for {pattern.ticker}: {e}")
            return {}
    
    def _calculate_pattern_similarity(self, current_features: Dict[str, float], 
                                    historical_features: Dict[str, float]) -> float:
        """Calculate similarity score between current and historical patterns"""
        try:
            # Define feature weights (importance for pattern matching)
            weights = {
                'log_market_cap': 0.15,      # Market cap similarity
                'revenue_growth': 0.20,      # Growth similarity
                'institutional_ownership': 0.10,  # Ownership structure
                'analyst_coverage': 0.05,    # Coverage level
                'volatility': 0.10,          # Risk profile
                'momentum_6m': 0.15,         # Momentum similarity
                'gross_margin': 0.10,        # Business quality
                'operating_margin': 0.05,    # Profitability
                'debt_to_equity': 0.05,      # Financial health
                'volume_trend': 0.05         # Volume pattern
            }
            
            # Calculate weighted similarity
            total_similarity = 0.0
            total_weight = 0.0
            
            for feature, weight in weights.items():
                if feature in current_features and feature in historical_features:
                    current_val = current_features[feature]
                    historical_val = historical_features[feature]
                    
                    # Calculate feature similarity (inverse of normalized difference)
                    if feature == 'log_market_cap':
                        # Market cap: similarity decreases with log difference
                        diff = abs(current_val - historical_val)
                        similarity = max(0, 1 - (diff / 3.0))  # 3 orders of magnitude max
                    elif feature in ['revenue_growth', 'momentum_6m']:
                        # Growth metrics: similarity based on relative difference
                        if historical_val != 0:
                            diff = abs((current_val - historical_val) / abs(historical_val))
                            similarity = max(0, 1 - min(diff, 2.0) / 2.0)
                        else:
                            similarity = 1.0 if current_val == 0 else 0.5
                    elif feature in ['institutional_ownership', 'gross_margin', 'operating_margin']:
                        # Percentage metrics: direct comparison
                        diff = abs(current_val - historical_val)
                        similarity = max(0, 1 - diff)
                    elif feature == 'analyst_coverage':
                        # Analyst coverage: logarithmic similarity
                        log_current = np.log10(max(current_val, 1))
                        log_historical = np.log10(max(historical_val, 1))
                        diff = abs(log_current - log_historical)
                        similarity = max(0, 1 - diff / 2.0)
                    else:
                        # Other metrics: normalized difference
                        max_val = max(abs(current_val), abs(historical_val), 1.0)
                        diff = abs(current_val - historical_val) / max_val
                        similarity = max(0, 1 - diff)
                    
                    total_similarity += similarity * weight
                    total_weight += weight
            
            # Return normalized similarity score (0-1)
            return total_similarity / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _identify_key_similarities(self, current_features: Dict[str, float], 
                                 historical_features: Dict[str, float], 
                                 pattern: HistoricalPattern) -> List[str]:
        """Identify key similarities between current and historical patterns"""
        similarities = []
        
        try:
            # Market cap similarity
            if 'log_market_cap' in current_features and 'log_market_cap' in historical_features:
                current_mcap = 10 ** current_features['log_market_cap']
                historical_mcap = 10 ** historical_features['log_market_cap']
                
                if abs(np.log10(current_mcap / historical_mcap)) < 0.5:  # Within ~3x
                    similarities.append(f"Similar market cap range (${current_mcap/1e9:.1f}B vs ${historical_mcap/1e9:.1f}B)")
            
            # Growth similarity
            if 'revenue_growth' in current_features and 'revenue_growth' in historical_features:
                current_growth = current_features['revenue_growth'] * 100
                historical_growth = historical_features['revenue_growth'] * 100
                
                if abs(current_growth - historical_growth) < 20:  # Within 20%
                    similarities.append(f"Similar revenue growth profile ({current_growth:.0f}% vs {historical_growth:.0f}%)")
            
            # Momentum similarity
            if 'momentum_6m' in current_features and 'momentum_6m' in historical_features:
                current_momentum = current_features['momentum_6m'] * 100
                historical_momentum = historical_features['momentum_6m'] * 100
                
                if abs(current_momentum - historical_momentum) < 15:  # Within 15%
                    similarities.append(f"Similar momentum profile ({current_momentum:.0f}% vs {historical_momentum:.0f}%)")
            
            # Always include sector context
            similarities.append(f"Operating in emerging/high-growth sector ({pattern.sector})")
            
            # Pattern-specific similarities
            if pattern.base_formation_months >= 6:
                similarities.append("Extended base formation period suggests accumulation")
            
            if pattern.breakout_volume_multiplier >= 3.0:
                similarities.append("Strong volume expansion on breakout indicates institutional interest")
            
        except Exception as e:
            logger.error(f"Error identifying similarities: {e}")
        
        return similarities[:5]  # Return top 5 similarities
    
    def _identify_pattern_differences(self, current_features: Dict[str, float], 
                                    historical_features: Dict[str, float], 
                                    pattern: HistoricalPattern) -> List[str]:
        """Identify key differences between current and historical patterns"""
        differences = []
        
        try:
            # Market environment differences
            differences.append(f"Different market environment (current vs {pattern.market_environment})")
            
            # Interest rate environment
            differences.append(f"Different rate environment (current vs {pattern.interest_rate_environment})")
            
            # Market cap differences
            if 'log_market_cap' in current_features and 'log_market_cap' in historical_features:
                current_mcap = 10 ** current_features['log_market_cap']
                historical_mcap = 10 ** historical_features['log_market_cap']
                
                ratio = current_mcap / historical_mcap
                if ratio > 3:
                    differences.append(f"Larger market cap may limit upside potential ({ratio:.1f}x larger)")
                elif ratio < 0.33:
                    differences.append(f"Smaller market cap may increase volatility risk ({ratio:.1f}x smaller)")
            
            # Growth differences
            if 'revenue_growth' in current_features and 'revenue_growth' in historical_features:
                current_growth = current_features['revenue_growth'] * 100
                historical_growth = historical_features['revenue_growth'] * 100
                
                growth_diff = current_growth - historical_growth
                if abs(growth_diff) > 25:
                    if growth_diff > 0:
                        differences.append(f"Higher current growth rate may indicate different stage ({current_growth:.0f}% vs {historical_growth:.0f}%)")
                    else:
                        differences.append(f"Lower current growth rate may limit upside ({current_growth:.0f}% vs {historical_growth:.0f}%)")
            
            # Replication difficulty
            if pattern.replication_difficulty == "Hard":
                differences.append("Historical pattern may be difficult to replicate due to unique circumstances")
            
        except Exception as e:
            logger.error(f"Error identifying differences: {e}")
        
        return differences[:4]  # Return top 4 differences
    
    def analyze_pattern_replication_probability(self, current_ticker: str, 
                                              similar_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze probability of replicating similar historical patterns.
        
        Args:
            current_ticker: Current ticker being analyzed
            similar_patterns: List of similar historical patterns
            
        Returns:
            Analysis of replication probability and risk factors
        """
        try:
            if not similar_patterns:
                return {
                    'replication_probability': 0.0,
                    'confidence_level': 'Low',
                    'risk_factors': ['No similar historical patterns found'],
                    'success_factors': [],
                    'recommended_monitoring': []
                }
            
            # Calculate weighted replication probability
            total_weight = 0.0
            weighted_probability = 0.0
            
            replication_difficulties = []
            pattern_confidences = []
            
            for pattern in similar_patterns:
                similarity = pattern['similarity_score']
                confidence = pattern['pattern_confidence']
                
                # Adjust probability based on replication difficulty
                difficulty_multiplier = {
                    'Easy': 1.0,
                    'Medium': 0.7,
                    'Hard': 0.4
                }.get(pattern['replication_difficulty'], 0.5)
                
                pattern_probability = similarity * confidence * difficulty_multiplier
                weighted_probability += pattern_probability * similarity  # Weight by similarity
                total_weight += similarity
                
                replication_difficulties.append(pattern['replication_difficulty'])
                pattern_confidences.append(confidence)
            
            # Calculate final probability
            replication_probability = weighted_probability / total_weight if total_weight > 0 else 0.0
            
            # Determine confidence level
            avg_confidence = np.mean(pattern_confidences)
            avg_similarity = np.mean([p['similarity_score'] for p in similar_patterns])
            
            if replication_probability >= 0.7 and avg_confidence >= 0.8 and avg_similarity >= 0.6:
                confidence_level = 'High'
            elif replication_probability >= 0.5 and avg_confidence >= 0.6:
                confidence_level = 'Medium'
            else:
                confidence_level = 'Low'
            
            # Identify risk factors
            risk_factors = []
            if 'Hard' in replication_difficulties:
                risk_factors.append('Some similar patterns had unique circumstances that may be hard to replicate')
            
            if avg_similarity < 0.5:
                risk_factors.append('Limited similarity to historical patterns increases uncertainty')
            
            current_market_risks = [
                'Current market conditions may differ from historical pattern periods',
                'Interest rate environment may impact growth stock valuations',
                'Increased competition in emerging sectors'
            ]
            risk_factors.extend(current_market_risks[:2])
            
            # Identify success factors
            success_factors = []
            for pattern in similar_patterns[:3]:  # Top 3 patterns
                if pattern['gain_achieved'] > 300:  # 3x+ gains
                    success_factors.append(f"Similar to {pattern['historical_ticker']}: {pattern['fundamental_catalyst']}")
            
            # Monitoring recommendations
            monitoring_factors = [
                'Volume expansion on price breakouts',
                'Institutional ownership changes (13F filings)',
                'Revenue growth acceleration',
                'Sector rotation and capital flows',
                'Technical pattern completion'
            ]
            
            return {
                'replication_probability': replication_probability,
                'confidence_level': confidence_level,
                'risk_factors': risk_factors,
                'success_factors': success_factors,
                'recommended_monitoring': monitoring_factors,
                'similar_pattern_count': len(similar_patterns),
                'avg_historical_gain': np.mean([p['gain_achieved'] for p in similar_patterns]),
                'avg_pattern_duration': np.mean([p['pattern_duration_months'] for p in similar_patterns])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing replication probability for {current_ticker}: {e}")
            return {
                'replication_probability': 0.0,
                'confidence_level': 'Low',
                'error': str(e)
            }
    
    def get_pattern_insights(self, current_ticker: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive pattern analysis insights for a ticker.
        
        Args:
            current_ticker: Ticker to analyze
            current_data: Comprehensive ticker data
            
        Returns:
            Complete pattern analysis with insights and recommendations
        """
        try:
            # Find similar patterns
            similar_patterns = self.find_similar_patterns(current_ticker, current_data)
            
            # Analyze replication probability
            replication_analysis = self.analyze_pattern_replication_probability(
                current_ticker, similar_patterns
            )
            
            # Generate insights
            insights = {
                'ticker': current_ticker,
                'analysis_date': datetime.now().isoformat(),
                'similar_patterns': similar_patterns,
                'replication_analysis': replication_analysis,
                'pattern_summary': self._generate_pattern_summary(similar_patterns),
                'actionable_insights': self._generate_actionable_insights(
                    current_ticker, similar_patterns, replication_analysis
                )
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting pattern insights for {current_ticker}: {e}")
            return {
                'ticker': current_ticker,
                'error': str(e),
                'analysis_date': datetime.now().isoformat()
            }
    
    def _generate_pattern_summary(self, similar_patterns: List[Dict[str, Any]]) -> str:
        """Generate a summary of similar patterns"""
        if not similar_patterns:
            return "No similar historical patterns identified."
        
        try:
            top_pattern = similar_patterns[0]
            avg_gain = np.mean([p['gain_achieved'] for p in similar_patterns])
            avg_duration = np.mean([p['pattern_duration_months'] for p in similar_patterns])
            
            summary = f"""
            Most similar to {top_pattern['historical_ticker']} ({top_pattern['similarity_score']:.1%} similarity), 
            which achieved {top_pattern['gain_achieved']:.0f}% gains over {top_pattern['pattern_duration_months']} months.
            
            Average gains across {len(similar_patterns)} similar patterns: {avg_gain:.0f}% 
            over {avg_duration:.0f} months.
            
            Key pattern characteristics: {top_pattern['technical_setup']}
            Primary catalyst similarity: {top_pattern['fundamental_catalyst']}
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating pattern summary: {e}")
            return "Pattern summary generation failed."
    
    def _generate_actionable_insights(self, ticker: str, similar_patterns: List[Dict[str, Any]], 
                                    replication_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on pattern analysis"""
        insights = []
        
        try:
            if not similar_patterns:
                insights.append("Consider expanding analysis to include more diverse historical patterns")
                return insights
            
            # Probability-based insights
            prob = replication_analysis.get('replication_probability', 0)
            if prob >= 0.7:
                insights.append(f"High probability of pattern replication ({prob:.1%}) suggests strong multi-bagger potential")
            elif prob >= 0.4:
                insights.append(f"Moderate probability of pattern replication ({prob:.1%}) warrants position sizing appropriate to conviction")
            else:
                insights.append(f"Lower probability of pattern replication ({prob:.1%}) suggests speculative position sizing")
            
            # Duration insights
            avg_duration = replication_analysis.get('avg_pattern_duration', 0)
            if avg_duration > 0:
                insights.append(f"Historical patterns suggest {avg_duration:.0f}-month holding period for maximum gains")
            
            # Gain potential insights
            avg_gain = replication_analysis.get('avg_historical_gain', 0)
            if avg_gain > 500:
                insights.append(f"Similar patterns achieved average gains of {avg_gain:.0f}%, suggesting 5x+ potential")
            elif avg_gain > 200:
                insights.append(f"Similar patterns achieved average gains of {avg_gain:.0f}%, suggesting 2-3x potential")
            
            # Risk management insights
            risk_factors = replication_analysis.get('risk_factors', [])
            if len(risk_factors) > 2:
                insights.append("Multiple risk factors identified - consider gradual position building")
            
            # Monitoring insights
            monitoring = replication_analysis.get('recommended_monitoring', [])
            if monitoring:
                insights.append(f"Key monitoring points: {', '.join(monitoring[:3])}")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating actionable insights: {e}")
            return ["Pattern analysis insights generation failed"]


# Example usage and testing
if __name__ == "__main__":
    # Initialize pattern analyzer
    analyzer = HistoricalPatternAnalyzer()
    
    print("📊 Historical Pattern Analyzer - Testing")
    print("=" * 50)
    
    # Test pattern matching (would use real data in practice)
    sample_data = {
        'market_data': {
            'market_cap': 500e6,  # $500M
            'held_percent_institutions': 0.3,  # 30%
            'beta': 1.5
        },
        'fundamental_data': {
            'revenue_growth_yoy': 0.4,  # 40%
            'gross_margin': 0.3,  # 30%
            'operating_margin': 0.1,  # 10%
        },
        'price_data': pd.DataFrame(),  # Empty for test
        'info': {'sector': 'Technology'}
    }
    
    # Test pattern analysis
    insights = analyzer.get_pattern_insights('TEST', sample_data)
    
    print(f"Analysis Results for TEST:")
    print(f"Similar Patterns Found: {len(insights.get('similar_patterns', []))}")
    
    if insights.get('similar_patterns'):
        top_match = insights['similar_patterns'][0]
        print(f"Top Match: {top_match['historical_ticker']} ({top_match['similarity_score']:.1%} similarity)")
        print(f"Historical Gain: {top_match['gain_achieved']:.0f}%")
    
    replication = insights.get('replication_analysis', {})
    if replication:
        print(f"Replication Probability: {replication.get('replication_probability', 0):.1%}")
        print(f"Confidence Level: {replication.get('confidence_level', 'Unknown')}")
    
    print("\n✅ Pattern analysis test completed")
