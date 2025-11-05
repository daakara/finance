"""
Hidden Gems Scanner - Advanced Multi-Asset Discovery System
Core screening engine for identifying undervalued/overlooked assets with 10x+ potential
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import yfinance as yf
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class GemCriteria:
    """Configuration for hidden gem screening criteria"""
    # Market cap constraints
    min_market_cap: float = 50e6      # $50M minimum
    max_market_cap: float = 15e9      # $15B maximum (increased for better results)
    
    # Fundamental thresholds
    min_revenue_growth: float = 0.15   # 15% YoY (decreased for better results)
    min_gross_margin: float = 0.20     # 20% (decreased for better results)
    max_debt_equity: float = 0.5       # Debt/Equity < 0.5
    min_insider_ownership: float = 0.10 # 10%
    
    # Visibility constraints
    max_analyst_coverage: int = 20     # Increased to 20
    min_institutional_ownership: float = 0.10  # 10%
    max_institutional_ownership: float = 0.40  # 40%
    
    # Technical criteria
    min_base_formation_days: int = 90   # 3 months consolidation
    max_drawdown_from_high: float = 0.30 # 30% max drawdown
    min_relative_strength: float = 1.0   # Outperform market

@dataclass
class GemScore:
    """Scoring results for a potential hidden gem"""
    ticker: str
    composite_score: float
    sector_score: float
    fundamental_score: float
    technical_score: float
    visibility_score: float
    catalyst_score: float
    smart_money_score: float
    sustainability_score: float  # NEW: ESG/Impact score
    risk_rating: str
    investment_thesis: str
    primary_catalyst: str
    action_plan: Dict[str, Any]
    sustainability_data: Optional[Any] = None  # NEW: Detailed sustainability metrics

class HiddenGemScreener:
    """Multi-factor screening system for identifying hidden gem opportunities"""
    
    def __init__(self, criteria: Optional[GemCriteria] = None):
        """Initialize with screening criteria"""
        self.criteria = criteria or GemCriteria()
        
        # Import sustainability analyzer
        try:
            from analyst_dashboard.analyzers.sustainability_analyzer import SustainabilityAnalyzer
            self.sustainability_analyzer = SustainabilityAnalyzer()
            self.use_sustainability = True
        except ImportError:
            logger.warning("Sustainability analyzer not available")
            self.sustainability_analyzer = None
            self.use_sustainability = False
        
        # Emerging sectors to monitor
        self.emerging_sectors = {
            'blockchain_infrastructure': ['Bitcoin Mining', 'Blockchain Infrastructure', 'Crypto Services'],
            'ai_ml': ['Artificial Intelligence', 'Machine Learning', 'AI Hardware'],
            'quantum_computing': ['Quantum Computing', 'Quantum Software'],
            'clean_energy': ['Solar', 'Wind', 'Energy Storage', 'Battery Technology'],
            'space_technology': ['Satellite Technology', 'Space Exploration'],
            'biotechnology': ['Gene Therapy', 'Longevity Research', 'Precision Medicine'],
            'cybersecurity': ['Data Privacy', 'Cybersecurity', 'Identity Management'],
            'fintech': ['Digital Payments', 'DeFi', 'Neobanking'],
            'robotics': ['Industrial Automation', 'Service Robotics'],
            'edge_computing': ['Edge Computing', '5G Infrastructure', 'IoT']
        }
        
        # Historical multi-bagger patterns for comparison
        self.historical_patterns = {
            'IREN': {
                'sector': 'Bitcoin Mining',
                'entry_price': 3.0,
                'peak_price': 18.0,
                'gain_percent': 600.0,
                'pattern_characteristics': {
                    'base_formation_months': 6,
                    'breakout_volume_increase': 3.5,
                    'sector_tailwinds': 'Bitcoin halving cycle',
                    'fundamental_catalyst': 'Capacity expansion'
                }
            }
        }
    
    def calculate_emerging_sector_score(self, ticker: str, sector_data: Dict[str, Any]) -> float:
        """
        Analyze asset's exposure to emerging high-growth sectors.
        
        Args:
            ticker: Asset ticker
            sector_data: Sector classification and exposure data
            
        Returns:
            Score from 0-100 based on emerging sector exposure
        """
        try:
            score = 0.0
            
            # Primary sector classification
            primary_sector = sector_data.get('primary_sector', '')
            for sector_group, sectors in self.emerging_sectors.items():
                if any(sector.lower() in primary_sector.lower() for sector in sectors):
                    score += 30.0  # Base score for being in emerging sector
                    break
            
            # Revenue exposure to emerging technologies
            revenue_breakdown = sector_data.get('revenue_segments', {})
            for segment, percentage in revenue_breakdown.items():
                for sector_group, sectors in self.emerging_sectors.items():
                    if any(sector.lower() in segment.lower() for sector in sectors):
                        score += percentage * 0.5  # Weight by revenue exposure
            
            # Recent strategic moves toward emerging tech
            strategic_moves = sector_data.get('strategic_initiatives', [])
            emerging_moves = sum(1 for move in strategic_moves 
                               if any(keyword in move.lower() 
                                     for keywords in self.emerging_sectors.values()
                                     for keyword in keywords))
            score += min(emerging_moves * 10, 20)  # Max 20 points for strategic moves
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating sector score for {ticker}: {e}")
            return 0.0
    
    def calculate_visibility_score(self, ticker: str, market_data: Dict[str, Any]) -> float:
        """
        Calculate how 'under the radar' an asset is.
        Lower score = more hidden (which is what we want)
        
        Args:
            ticker: Asset ticker
            market_data: Market and coverage data
            
        Returns:
            Score from 0-100 where lower is better (more hidden)
        """
        try:
            visibility_score = 0.0
            
            # Market cap factor (sweet spot: $50M - $2B)
            market_cap = market_data.get('market_cap', 0)
            if market_cap > 0:
                if market_cap < self.criteria.min_market_cap:
                    visibility_score += 40  # Too small, liquidity risk
                elif market_cap > self.criteria.max_market_cap:
                    visibility_score += 60  # Too large, likely discovered
                else:
                    # Sweet spot scoring
                    ratio = (market_cap - self.criteria.min_market_cap) / (self.criteria.max_market_cap - self.criteria.min_market_cap)
                    visibility_score += 10 + (ratio * 20)  # 10-30 range
            
            # Analyst coverage
            analyst_count = market_data.get('analyst_coverage', 0)
            visibility_score += min(analyst_count * 5, 30)  # Max 30 points
            
            # News mentions (last 30 days)
            news_mentions = market_data.get('news_mentions_30d', 0)
            visibility_score += min(news_mentions * 2, 20)  # Max 20 points
            
            # Social media volume
            social_volume = market_data.get('social_volume_score', 0)
            visibility_score += min(social_volume, 15)  # Max 15 points
            
            # Institutional ownership (10-40% is ideal)
            inst_ownership = market_data.get('institutional_ownership', 0)
            if inst_ownership < 0.10:
                visibility_score += 10  # Too little institutional interest
            elif inst_ownership > 0.40:
                visibility_score += 25  # Too crowded
            else:
                visibility_score += 5   # Good range
            
            return min(visibility_score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating visibility score for {ticker}: {e}")
            return 50.0  # Default to moderate visibility
    
    def calculate_fundamental_score(self, ticker: str, fundamental_data: Dict[str, Any]) -> float:
        """
        Evaluate fundamental strength of the asset.
        
        Args:
            ticker: Asset ticker
            fundamental_data: Financial and operational metrics
            
        Returns:
            Score from 0-100 based on fundamental strength
        """
        try:
            score = 0.0
            criteria_met = 0
            total_criteria = 8
            
            # Revenue growth (25%+ YoY)
            revenue_growth = fundamental_data.get('revenue_growth_yoy', 0)
            if revenue_growth >= self.criteria.min_revenue_growth:
                score += 15
                criteria_met += 1
            
            # Gross margins (30%+)
            gross_margin = fundamental_data.get('gross_margin', 0)
            if gross_margin >= self.criteria.min_gross_margin:
                score += 15
                criteria_met += 1
            
            # Cash runway (2+ years)
            cash_runway = fundamental_data.get('cash_runway_years', 0)
            if cash_runway >= 2.0:
                score += 10
                criteria_met += 1
            
            # Debt-to-equity (<0.5)
            debt_equity = fundamental_data.get('debt_to_equity', float('inf'))
            if debt_equity <= self.criteria.max_debt_equity:
                score += 10
                criteria_met += 1
            
            # Operating leverage (improving margins)
            margin_trend = fundamental_data.get('operating_margin_trend', 0)
            if margin_trend > 0:  # Positive trend
                score += 10
                criteria_met += 1
            
            # Insider ownership (10%+)
            insider_ownership = fundamental_data.get('insider_ownership', 0)
            if insider_ownership >= self.criteria.min_insider_ownership:
                score += 15
                criteria_met += 1
            
            # Recent insider buying
            insider_buying = fundamental_data.get('insider_buying_6m', False)
            if insider_buying:
                score += 15
                criteria_met += 1
            
            # Business model quality
            business_quality = fundamental_data.get('business_quality_score', 0)
            if business_quality >= 70:  # High quality business model
                score += 10
                criteria_met += 1
            
            # Bonus for meeting most criteria
            if criteria_met >= 6:
                score += 10  # Consistency bonus
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score for {ticker}: {e}")
            return 0.0
    
    def detect_accumulation_pattern(self, ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify institutional accumulation patterns in price and volume data.
        
        Args:
            ticker: Asset ticker
            price_data: OHLCV price data
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            if len(price_data) < 60:  # Need at least 60 days of data
                return {'pattern_detected': False, 'reason': 'Insufficient data'}
            
            analysis = {
                'pattern_detected': False,
                'accumulation_score': 0.0,
                'base_formation': False,
                'volume_analysis': {},
                'technical_indicators': {},
                'support_resistance': {}
            }
            
            # Calculate technical indicators
            price_data = price_data.copy()
            price_data['MA_20'] = price_data['Close'].rolling(20).mean()
            price_data['MA_50'] = price_data['Close'].rolling(50).mean()
            price_data['Volume_MA_20'] = price_data['Volume'].rolling(20).mean()
            
            # On-Balance Volume (OBV)
            price_data['OBV'] = (price_data['Volume'] * 
                               np.where(price_data['Close'] > price_data['Close'].shift(1), 1,
                                       np.where(price_data['Close'] < price_data['Close'].shift(1), -1, 0))).cumsum()
            
            # Base formation analysis (3-12 months consolidation)
            recent_90d = price_data.tail(90)
            price_range = (recent_90d['High'].max() - recent_90d['Low'].min()) / recent_90d['Close'].mean()
            
            if price_range < 0.40:  # Consolidating within 40% range
                analysis['base_formation'] = True
                analysis['accumulation_score'] += 25
            
            # Volume dry-up during consolidation
            recent_volume_avg = recent_90d['Volume'].mean()
            historical_volume_avg = price_data['Volume'].mean()
            volume_ratio = recent_volume_avg / historical_volume_avg
            
            if 0.7 <= volume_ratio <= 1.3:  # Volume normalizing, not spiking
                analysis['accumulation_score'] += 20
            
            # OBV trend analysis
            obv_recent = price_data['OBV'].tail(30).mean()
            obv_historical = price_data['OBV'].head(30).mean()
            
            if obv_recent > obv_historical:  # Accumulation
                analysis['accumulation_score'] += 25
            
            # Price vs moving averages
            current_price = price_data['Close'].iloc[-1]
            ma_50 = price_data['MA_50'].iloc[-1]
            
            if not pd.isna(ma_50) and current_price >= ma_50 * 0.95:  # Near or above 50-day MA
                analysis['accumulation_score'] += 15
            
            # Support level identification
            lows = price_data['Low'].tail(60)
            support_level = lows.quantile(0.1)  # Bottom 10% of lows
            current_distance_from_support = (current_price - support_level) / support_level
            
            if 0.05 <= current_distance_from_support <= 0.20:  # 5-20% above strong support
                analysis['accumulation_score'] += 15
            
            analysis['volume_analysis'] = {
                'recent_vs_historical_ratio': volume_ratio,
                'volume_trend': 'normalizing' if 0.7 <= volume_ratio <= 1.3 else 'high' if volume_ratio > 1.3 else 'low'
            }
            
            analysis['technical_indicators'] = {
                'obv_trend': 'accumulation' if obv_recent > obv_historical else 'distribution',
                'price_vs_ma50': current_price / ma_50 if not pd.isna(ma_50) else 0,
                'distance_from_support': current_distance_from_support
            }
            
            # Pattern detected if score >= 60
            if analysis['accumulation_score'] >= 60:
                analysis['pattern_detected'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error detecting accumulation pattern for {ticker}: {e}")
            return {'pattern_detected': False, 'error': str(e)}
    
    def calculate_technical_score(self, ticker: str, price_data: pd.DataFrame) -> float:
        """
        Calculate technical analysis score for accumulation setup.
        
        Args:
            ticker: Asset ticker
            price_data: OHLCV price data
            
        Returns:
            Score from 0-100 based on technical setup quality
        """
        try:
            accumulation_analysis = self.detect_accumulation_pattern(ticker, price_data)
            
            base_score = accumulation_analysis.get('accumulation_score', 0)
            
            # Additional technical factors
            if len(price_data) >= 252:  # Need 1 year of data for relative strength
                # Relative strength vs market (assume SPY as benchmark)
                returns_6m = price_data['Close'].pct_change().tail(126).mean() * 252  # Annualized
                # For now, use absolute performance as proxy
                if returns_6m > 0.10:  # 10%+ performance
                    base_score += 10
            
            # Volume confirmation
            volume_analysis = accumulation_analysis.get('volume_analysis', {})
            if volume_analysis.get('volume_trend') == 'normalizing':
                base_score += 10
            
            return min(base_score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating technical score for {ticker}: {e}")
            return 0.0
    
    def scan_upcoming_catalysts(self, ticker: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify potential near-term and long-term catalysts.
        
        Args:
            ticker: Asset ticker
            company_data: Company information and events
            
        Returns:
            Dictionary with catalyst analysis
        """
        try:
            catalysts = {
                'near_term': [],     # 0-6 months
                'long_term': [],     # 6-24 months
                'catalyst_score': 0.0
            }
            
            # Earnings-related catalysts
            earnings_data = company_data.get('earnings', {})
            if earnings_data.get('estimate_revisions_trend') == 'up':
                catalysts['near_term'].append({
                    'type': 'earnings_surprise',
                    'description': 'Analyst estimates trending upward',
                    'probability': 0.7,
                    'impact': 'medium'
                })
                catalysts['catalyst_score'] += 15
            
            # Product/regulatory catalysts
            pipeline = company_data.get('product_pipeline', [])
            for product in pipeline:
                if product.get('expected_approval_date'):
                    months_to_approval = product.get('months_to_approval', 12)
                    catalyst_item = {
                        'type': 'product_approval',
                        'description': product.get('description', ''),
                        'probability': product.get('approval_probability', 0.5),
                        'impact': product.get('market_impact', 'medium')
                    }
                    
                    if months_to_approval <= 6:
                        catalysts['near_term'].append(catalyst_item)
                        catalysts['catalyst_score'] += 20
                    else:
                        catalysts['long_term'].append(catalyst_item)
                        catalysts['catalyst_score'] += 10
            
            # Strategic partnerships
            recent_partnerships = company_data.get('recent_partnerships', [])
            for partnership in recent_partnerships:
                if partnership.get('strategic_value', 0) > 50:  # High value partnerships
                    catalysts['near_term'].append({
                        'type': 'strategic_partnership',
                        'description': partnership.get('description', ''),
                        'probability': 0.8,
                        'impact': 'high'
                    })
                    catalysts['catalyst_score'] += 25
            
            # Insider activity
            if company_data.get('insider_buying_trend') == 'increasing':
                catalysts['near_term'].append({
                    'type': 'insider_confidence',
                    'description': 'Increasing insider purchases',
                    'probability': 0.9,
                    'impact': 'medium'
                })
                catalysts['catalyst_score'] += 15
            
            # Sector tailwinds
            sector = company_data.get('sector', '')
            for sector_group, sectors in self.emerging_sectors.items():
                if any(s.lower() in sector.lower() for s in sectors):
                    catalysts['long_term'].append({
                        'type': 'sector_tailwinds',
                        'description': f'Emerging sector growth: {sector_group}',
                        'probability': 0.6,
                        'impact': 'high'
                    })
                    catalysts['catalyst_score'] += 20
                    break
            
            return catalysts
            
        except Exception as e:
            logger.error(f"Error scanning catalysts for {ticker}: {e}")
            return {'near_term': [], 'long_term': [], 'catalyst_score': 0.0}
    
    def track_smart_money_flows(self, ticker: str, institutional_data: Dict[str, Any]) -> float:
        """
        Analyze institutional position changes and smart money activity.
        
        Args:
            ticker: Asset ticker
            institutional_data: 13F filings and institutional data
            
        Returns:
            Score from 0-100 based on smart money activity
        """
        try:
            score = 0.0
            
            # New 13F positions
            new_positions = institutional_data.get('new_positions_count', 0)
            score += min(new_positions * 5, 25)  # Max 25 points
            
            # Increased positions (>20% increase)
            increased_positions = institutional_data.get('increased_positions_count', 0)
            score += min(increased_positions * 3, 20)  # Max 20 points
            
            # Notable investor entries
            notable_investors = institutional_data.get('notable_new_investors', [])
            notable_count = len([inv for inv in notable_investors if inv.get('aum', 0) > 1e9])  # $1B+ AUM
            score += min(notable_count * 10, 30)  # Max 30 points
            
            # Options activity
            options_data = institutional_data.get('options_activity', {})
            unusual_call_activity = options_data.get('unusual_call_volume', False)
            if unusual_call_activity:
                score += 15
            
            # ETF inflows for thematic exposure
            etf_flows = institutional_data.get('thematic_etf_flows', 0)
            if etf_flows > 0:  # Positive flows into relevant thematic ETFs
                score += 10
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error tracking smart money flows for {ticker}: {e}")
            return 0.0
    
    def calculate_composite_score(self, ticker: str, all_data: Dict[str, Any]) -> GemScore:
        """
        Calculate weighted composite score and generate full analysis.
        
        Args:
            ticker: Asset ticker
            all_data: Complete data dictionary for the asset
            
        Returns:
            GemScore object with complete analysis
        """
        try:
            # Extract data components
            sector_data = all_data.get('sector_data', {})
            market_data = all_data.get('market_data', {})
            fundamental_data = all_data.get('fundamental_data', {})
            price_data = all_data.get('price_data', pd.DataFrame())
            company_data = all_data.get('company_data', {})
            institutional_data = all_data.get('institutional_data', {})
            
            # Calculate individual scores
            sector_score = self.calculate_emerging_sector_score(ticker, sector_data)
            visibility_score = self.calculate_visibility_score(ticker, market_data)
            fundamental_score = self.calculate_fundamental_score(ticker, fundamental_data)
            technical_score = self.calculate_technical_score(ticker, price_data)
            
            catalyst_analysis = self.scan_upcoming_catalysts(ticker, company_data)
            catalyst_score = catalyst_analysis.get('catalyst_score', 0)
            
            smart_money_score = self.track_smart_money_flows(ticker, institutional_data)
            
            # Weighted composite scoring
            # Calculate sustainability score if available
            sustainability_score = 50.0  # Default neutral score
            sustainability_data = None
            if self.use_sustainability and self.sustainability_analyzer:
                try:
                    # Get company info from all_data
                    company_info = all_data.get('info', {})
                    sustainability_data = self.sustainability_analyzer.analyze_sustainability(
                        ticker, company_info
                    )
                    sustainability_score = sustainability_data.overall_score
                except Exception as e:
                    logger.warning(f"Could not calculate sustainability score for {ticker}: {e}")
            
            weights = {
                'sector': 0.22,        # 22% - Sector tailwinds (reduced)
                'fundamental': 0.18,   # 18% - Fundamental strength (reduced)
                'technical': 0.18,     # 18% - Technical setup (reduced)
                'visibility': 0.14,    # 14% - Under-radar (reduced)
                'catalyst': 0.14,      # 14% - Catalyst potential (reduced)
                'smart_money': 0.05,   # 5% - Smart money flow
                'sustainability': 0.09 # 9% - ESG/Impact score (NEW)
            }
            
            # Invert visibility score (lower visibility is better for hidden gems)
            adjusted_visibility_score = 100 - visibility_score
            
            composite_score = (
                sector_score * weights['sector'] +
                fundamental_score * weights['fundamental'] +
                technical_score * weights['technical'] +
                adjusted_visibility_score * weights['visibility'] +
                catalyst_score * weights['catalyst'] +
                smart_money_score * weights['smart_money'] +
                sustainability_score * weights['sustainability']
            )
            
            # Risk rating
            if composite_score >= 80:
                risk_rating = "High Conviction"
            elif composite_score >= 65:
                risk_rating = "Medium-High"
            elif composite_score >= 50:
                risk_rating = "Medium"
            else:
                risk_rating = "High Risk"
            
            # Generate investment thesis
            investment_thesis = self._generate_investment_thesis(
                ticker, sector_score, fundamental_score, catalyst_analysis
            )
            
            # Primary catalyst
            primary_catalyst = "N/A"
            if catalyst_analysis['near_term']:
                primary_catalyst = catalyst_analysis['near_term'][0].get('description', 'Near-term catalyst')
            elif catalyst_analysis['long_term']:
                primary_catalyst = catalyst_analysis['long_term'][0].get('description', 'Long-term opportunity')
            
            # Action plan
            action_plan = self._generate_action_plan(
                ticker, price_data, composite_score, technical_score
            )
            
            return GemScore(
                ticker=ticker,
                composite_score=composite_score,
                sector_score=sector_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                visibility_score=visibility_score,
                catalyst_score=catalyst_score,
                smart_money_score=smart_money_score,
                sustainability_score=sustainability_score,
                risk_rating=risk_rating,
                investment_thesis=investment_thesis,
                primary_catalyst=primary_catalyst,
                action_plan=action_plan,
                sustainability_data=sustainability_data
            )
            
        except Exception as e:
            logger.error(f"Error calculating composite score for {ticker}: {e}")
            return GemScore(
                ticker=ticker,
                composite_score=0.0,
                sector_score=0.0,
                fundamental_score=0.0,
                technical_score=0.0,
                visibility_score=100.0,  # High visibility (not hidden)
                catalyst_score=0.0,
                smart_money_score=0.0,
                sustainability_score=50.0,  # Neutral
                risk_rating="High Risk",
                investment_thesis="Analysis failed",
                primary_catalyst="None identified",
                action_plan={}
            )
    
    def _generate_investment_thesis(self, ticker: str, sector_score: float, 
                                  fundamental_score: float, catalyst_analysis: Dict) -> str:
        """Generate investment thesis based on scoring results"""
        try:
            thesis_parts = []
            
            if sector_score >= 70:
                thesis_parts.append(f"{ticker} is positioned in a high-growth emerging sector with significant tailwinds.")
            
            if fundamental_score >= 70:
                thesis_parts.append("The company demonstrates strong fundamental metrics including revenue growth, margin expansion, and insider confidence.")
            
            if catalyst_analysis.get('catalyst_score', 0) >= 60:
                near_term = len(catalyst_analysis.get('near_term', []))
                if near_term > 0:
                    thesis_parts.append(f"Multiple near-term catalysts ({near_term}) could drive significant revaluation.")
            
            if len(thesis_parts) == 0:
                return f"{ticker} shows mixed signals across fundamental and technical analysis."
            
            return " ".join(thesis_parts)
            
        except Exception:
            return f"Investment thesis analysis pending for {ticker}."
    
    def _generate_action_plan(self, ticker: str, price_data: pd.DataFrame, 
                             composite_score: float, technical_score: float) -> Dict[str, Any]:
        """Generate actionable investment plan"""
        try:
            if price_data.empty:
                return {'status': 'Insufficient price data'}
            
            current_price = price_data['Close'].iloc[-1]
            
            # Entry range (5-10% below current for better risk/reward)
            entry_low = current_price * 0.90
            entry_high = current_price * 0.95
            
            # Stop loss (15-20% below entry)
            stop_loss = entry_low * 0.80
            
            # Price targets based on composite score
            if composite_score >= 80:
                target_12m = current_price * 2.0   # 100% target
                target_24m = current_price * 3.0   # 200% target
            elif composite_score >= 65:
                target_12m = current_price * 1.5   # 50% target
                target_24m = current_price * 2.0   # 100% target
            else:
                target_12m = current_price * 1.25  # 25% target
                target_24m = current_price * 1.5   # 50% target
            
            # Position sizing based on conviction
            if composite_score >= 80:
                position_size = "2-3% of portfolio (high conviction)"
            elif composite_score >= 65:
                position_size = "1-2% of portfolio (medium conviction)"
            else:
                position_size = "0.5-1% of portfolio (speculative)"
            
            return {
                'entry_range': {'low': entry_low, 'high': entry_high},
                'stop_loss': stop_loss,
                'targets': {'12_month': target_12m, '24_month': target_24m},
                'position_sizing': position_size,
                'monitoring_frequency': 'Weekly' if composite_score >= 70 else 'Monthly'
            }
            
        except Exception as e:
            logger.error(f"Error generating action plan for {ticker}: {e}")
            return {'error': str(e)}
    
    def screen_universe(self, tickers: List[str]) -> List[GemScore]:
        """
        Screen a universe of tickers for hidden gem opportunities.
        
        Args:
            tickers: List of ticker symbols to screen
            
        Returns:
            List of GemScore objects for qualifying opportunities
        """
        results = []
        
        for ticker in tickers:
            try:
                logger.info(f"Screening {ticker} for hidden gem potential...")
                
                # This would normally fetch real data from multiple sources
                # For now, we'll create a placeholder structure
                all_data = self._fetch_comprehensive_data(ticker)
                
                if all_data.get('error'):
                    logger.warning(f"Skipping {ticker}: {all_data['error']}")
                    continue
                
                gem_score = self.calculate_composite_score(ticker, all_data)
                
                # Only include if meets minimum criteria
                if gem_score.composite_score >= 50:  # Minimum threshold
                    results.append(gem_score)
                    
            except Exception as e:
                logger.error(f"Error screening {ticker}: {e}")
                continue
        
        # Sort by composite score (highest first)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        return results
    
    def _fetch_comprehensive_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive data for a ticker from multiple sources.
        This is a placeholder that would integrate with real data sources.
        """
        try:
            # In a real implementation, this would fetch from:
            # - yfinance for basic data
            # - Alpha Vantage for fundamentals
            # - Custom APIs for institutional data
            # - News APIs for sentiment
            # - SEC filings for insider activity
            
            # For now, return sample data structure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="1y")
                    ticker_info = info.info
                except:
                    # Fallback sample data
                    hist = pd.DataFrame()
                    ticker_info = {}
            
            return {
                'sector_data': {
                    'primary_sector': ticker_info.get('sector', 'Unknown'),
                    'industry': ticker_info.get('industry', 'Unknown'),
                    'revenue_segments': {},
                    'strategic_initiatives': []
                },
                'market_data': {
                    'market_cap': ticker_info.get('marketCap', 0),
                    'analyst_coverage': 5,  # Sample
                    'news_mentions_30d': 3,  # Sample
                    'social_volume_score': 20,  # Sample
                    'institutional_ownership': 0.25  # Sample
                },
                'fundamental_data': {
                    'revenue_growth_yoy': 0.30,  # Sample
                    'gross_margin': 0.35,  # Sample
                    'cash_runway_years': 3.0,  # Sample
                    'debt_to_equity': 0.3,  # Sample
                    'operating_margin_trend': 0.05,  # Sample
                    'insider_ownership': 0.15,  # Sample
                    'insider_buying_6m': True,  # Sample
                    'business_quality_score': 75  # Sample
                },
                'price_data': hist,
                'company_data': {
                    'sector': ticker_info.get('sector', 'Unknown'),
                    'earnings': {'estimate_revisions_trend': 'up'},
                    'product_pipeline': [],
                    'recent_partnerships': [],
                    'insider_buying_trend': 'stable'
                },
                'institutional_data': {
                    'new_positions_count': 2,  # Sample
                    'increased_positions_count': 3,  # Sample
                    'notable_new_investors': [],
                    'options_activity': {'unusual_call_volume': False},
                    'thematic_etf_flows': 1000000  # Sample
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {'error': str(e)}
    
    def scan_blockchain_infrastructure_gems(self) -> List[GemScore]:
        """
        Specific scanner for blockchain infrastructure opportunities (IREN-style).
        
        Returns:
            List of blockchain infrastructure hidden gems
        """
        # Sample blockchain/mining tickers - in practice, this would be a comprehensive list
        blockchain_tickers = ['IREN', 'RIOT', 'MARA', 'CLSK', 'BITF', 'HUT', 'CORZ', 'BTBT']
        
        # Apply blockchain-specific filtering
        blockchain_criteria = GemCriteria(
            min_market_cap=100e6,      # $100M minimum for mining companies
            max_market_cap=5e9,        # $5B maximum (larger cap acceptable for miners)
            min_revenue_growth=0.30,   # 30% YoY for crypto exposure
            min_gross_margin=0.20,     # Lower margins acceptable for mining
            max_debt_equity=0.8,       # Higher debt tolerance for infrastructure
            min_insider_ownership=0.05  # 5% minimum (public companies)
        )
        
        # Temporarily update criteria
        original_criteria = self.criteria
        self.criteria = blockchain_criteria
        
        try:
            results = self.screen_universe(blockchain_tickers)
            return results
        finally:
            # Restore original criteria
            self.criteria = original_criteria


# Example usage and testing
if __name__ == "__main__":
    # Initialize screener
    screener = HiddenGemScreener()
    
    # Test with sample tickers
    sample_tickers = ['AAPL', 'NVDA', 'IREN', 'MSTR']
    
    print("🔍 Hidden Gems Scanner - Testing Mode")
    print("=" * 50)
    
    results = screener.screen_universe(sample_tickers)
    
    for result in results:
        print(f"\n📊 {result.ticker}")
        print(f"Composite Score: {result.composite_score:.1f}/100")
        print(f"Risk Rating: {result.risk_rating}")
        print(f"Primary Catalyst: {result.primary_catalyst}")
        print(f"Investment Thesis: {result.investment_thesis}")
        print("-" * 30)
