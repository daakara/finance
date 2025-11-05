"""
Sustainability Analyzer - ESG and Impact Scoring System
Evaluates environmental, social, and governance factors for hidden gems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class SustainabilityScore:
    """ESG and sustainability scoring results"""
    ticker: str
    overall_score: float  # 0-100
    environmental_score: float  # 0-100
    social_score: float  # 0-100
    governance_score: float  # 0-100
    controversy_score: float  # 0-100 (higher is better)
    
    # Detailed metrics
    carbon_intensity: Optional[float] = None
    renewable_energy_usage: Optional[float] = None
    waste_reduction_score: float = 0.0
    
    diversity_score: float = 0.0
    employee_satisfaction: Optional[float] = None
    community_impact: float = 0.0
    
    board_independence: Optional[float] = None
    executive_compensation_ratio: Optional[float] = None
    transparency_score: float = 0.0
    
    # Impact categories
    impact_categories: List[str] = None
    sustainability_initiatives: List[str] = None
    
    # Risk factors
    environmental_risks: List[str] = None
    social_risks: List[str] = None
    governance_risks: List[str] = None
    
    # Investment alignment
    sdg_alignment: List[int] = None  # UN Sustainable Development Goals
    impact_thesis: str = ""
    
    def __post_init__(self):
        if self.impact_categories is None:
            self.impact_categories = []
        if self.sustainability_initiatives is None:
            self.sustainability_initiatives = []
        if self.environmental_risks is None:
            self.environmental_risks = []
        if self.social_risks is None:
            self.social_risks = []
        if self.governance_risks is None:
            self.governance_risks = []
        if self.sdg_alignment is None:
            self.sdg_alignment = []


class SustainabilityAnalyzer:
    """Comprehensive ESG and sustainability analysis system"""
    
    def __init__(self):
        """Initialize sustainability analyzer"""
        # UN Sustainable Development Goals mapping
        self.sdg_categories = {
            1: "No Poverty",
            3: "Good Health and Well-being",
            4: "Quality Education",
            5: "Gender Equality",
            6: "Clean Water and Sanitation",
            7: "Affordable and Clean Energy",
            8: "Decent Work and Economic Growth",
            9: "Industry, Innovation and Infrastructure",
            10: "Reduced Inequalities",
            11: "Sustainable Cities and Communities",
            12: "Responsible Consumption and Production",
            13: "Climate Action",
            14: "Life Below Water",
            15: "Life on Land",
            16: "Peace, Justice and Strong Institutions",
            17: "Partnerships for the Goals"
        }
        
        # Impact sector classifications
        self.impact_sectors = {
            'renewable_energy': {
                'keywords': ['solar', 'wind', 'renewable', 'clean energy', 'sustainable energy'],
                'sdgs': [7, 13],
                'weight': 1.5
            },
            'clean_tech': {
                'keywords': ['battery', 'energy storage', 'electric vehicle', 'ev', 'emissions'],
                'sdgs': [9, 11, 13],
                'weight': 1.4
            },
            'circular_economy': {
                'keywords': ['recycling', 'waste reduction', 'circular', 'sustainable materials'],
                'sdgs': [12, 14, 15],
                'weight': 1.3
            },
            'healthcare_innovation': {
                'keywords': ['biotech', 'healthcare', 'medical', 'health tech', 'telemedicine'],
                'sdgs': [3],
                'weight': 1.3
            },
            'education_tech': {
                'keywords': ['edtech', 'education technology', 'learning platform', 'online education'],
                'sdgs': [4],
                'weight': 1.2
            },
            'financial_inclusion': {
                'keywords': ['fintech', 'digital banking', 'microfinance', 'financial inclusion'],
                'sdgs': [1, 8, 10],
                'weight': 1.2
            },
            'sustainable_agriculture': {
                'keywords': ['agtech', 'precision agriculture', 'sustainable farming', 'vertical farming'],
                'sdgs': [2, 12, 13],
                'weight': 1.3
            },
            'water_tech': {
                'keywords': ['water treatment', 'water purification', 'desalination', 'water management'],
                'sdgs': [6, 14],
                'weight': 1.4
            }
        }
        
    def analyze_sustainability(self, ticker: str, company_info: Dict[str, Any] = None) -> SustainabilityScore:
        """
        Comprehensive sustainability analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            company_info: Optional pre-fetched company information
            
        Returns:
            SustainabilityScore with detailed ESG metrics
        """
        try:
            # Fetch company data if not provided
            if company_info is None:
                stock = yf.Ticker(ticker)
                company_info = stock.info
            
            # Calculate individual scores
            env_score = self._calculate_environmental_score(ticker, company_info)
            social_score = self._calculate_social_score(ticker, company_info)
            gov_score = self._calculate_governance_score(ticker, company_info)
            controversy_score = self._calculate_controversy_score(ticker, company_info)
            
            # Overall score (weighted average)
            overall = (env_score * 0.35 + social_score * 0.25 + 
                      gov_score * 0.25 + controversy_score * 0.15)
            
            # Identify impact categories and SDG alignment
            impact_cats, sdgs = self._identify_impact_alignment(ticker, company_info)
            
            # Extract detailed metrics
            carbon_intensity = company_info.get('carbonIntensity')
            renewable_usage = self._estimate_renewable_energy_usage(company_info)
            
            # Board metrics
            board_independence = self._calculate_board_independence(company_info)
            
            # Generate impact thesis
            impact_thesis = self._generate_impact_thesis(ticker, company_info, impact_cats)
            
            # Identify risks
            env_risks = self._identify_environmental_risks(company_info)
            social_risks = self._identify_social_risks(company_info)
            gov_risks = self._identify_governance_risks(company_info)
            
            return SustainabilityScore(
                ticker=ticker,
                overall_score=overall,
                environmental_score=env_score,
                social_score=social_score,
                governance_score=gov_score,
                controversy_score=controversy_score,
                carbon_intensity=carbon_intensity,
                renewable_energy_usage=renewable_usage,
                board_independence=board_independence,
                impact_categories=impact_cats,
                sdg_alignment=sdgs,
                impact_thesis=impact_thesis,
                environmental_risks=env_risks,
                social_risks=social_risks,
                governance_risks=gov_risks,
                sustainability_initiatives=self._extract_sustainability_initiatives(company_info)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sustainability for {ticker}: {e}")
            return SustainabilityScore(
                ticker=ticker,
                overall_score=50.0,
                environmental_score=50.0,
                social_score=50.0,
                governance_score=50.0,
                controversy_score=50.0
            )
    
    def _calculate_environmental_score(self, ticker: str, info: Dict[str, Any]) -> float:
        """Calculate environmental impact score"""
        score = 50.0  # Base neutral score
        
        try:
            # Check for ESG scores from yfinance
            if 'esgScores' in info and info['esgScores']:
                env_score = info['esgScores'].get('environmentScore')
                if env_score:
                    score = env_score
                    return score
            
            # Sector-based heuristics
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            
            # Positive environmental impact sectors
            positive_keywords = ['renewable', 'solar', 'wind', 'clean energy', 'electric vehicle', 
                               'battery', 'recycling', 'sustainable', 'environmental']
            if any(keyword in sector + ' ' + industry for keyword in positive_keywords):
                score += 30
            
            # Negative impact sectors
            negative_keywords = ['oil', 'coal', 'mining', 'fossil', 'petroleum']
            if any(keyword in sector + ' ' + industry for keyword in negative_keywords):
                score -= 20
            
            # Carbon intensity
            carbon_intensity = info.get('carbonIntensity')
            if carbon_intensity:
                if carbon_intensity < 50:
                    score += 10
                elif carbon_intensity > 200:
                    score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating environmental score: {e}")
            return 50.0
    
    def _calculate_social_score(self, ticker: str, info: Dict[str, Any]) -> float:
        """Calculate social impact score"""
        score = 50.0
        
        try:
            # Check for ESG scores
            if 'esgScores' in info and info['esgScores']:
                social_score = info['esgScores'].get('socialScore')
                if social_score:
                    return social_score
            
            # Industry impact
            industry = info.get('industry', '').lower()
            
            # Positive social impact industries
            positive_keywords = ['healthcare', 'education', 'biotech', 'medical', 
                               'telemedicine', 'health tech', 'social impact']
            if any(keyword in industry for keyword in positive_keywords):
                score += 20
            
            # Employee metrics
            full_time_employees = info.get('fullTimeEmployees', 0)
            if full_time_employees > 1000:
                score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating social score: {e}")
            return 50.0
    
    def _calculate_governance_score(self, ticker: str, info: Dict[str, Any]) -> float:
        """Calculate governance score"""
        score = 50.0
        
        try:
            # Check for ESG scores
            if 'esgScores' in info and info['esgScores']:
                gov_score = info['esgScores'].get('governanceScore')
                if gov_score:
                    return gov_score
            
            # Board metrics
            board_independence = self._calculate_board_independence(info)
            if board_independence and board_independence > 0.5:
                score += 15
            
            # Audit quality
            audit_risk = info.get('auditRisk')
            if audit_risk:
                if audit_risk < 5:
                    score += 10
                elif audit_risk > 8:
                    score -= 10
            
            # Compensation alignment
            compensation_risk = info.get('compensationRisk')
            if compensation_risk:
                if compensation_risk < 5:
                    score += 10
                elif compensation_risk > 8:
                    score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating governance score: {e}")
            return 50.0
    
    def _calculate_controversy_score(self, ticker: str, info: Dict[str, Any]) -> float:
        """Calculate controversy score (higher is better - fewer controversies)"""
        score = 100.0  # Start with perfect score
        
        try:
            # Overall risk rating
            overall_risk = info.get('overallRisk')
            if overall_risk:
                score = max(0, 100 - overall_risk * 5)
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating controversy score: {e}")
            return 75.0
    
    def _calculate_board_independence(self, info: Dict[str, Any]) -> Optional[float]:
        """Estimate board independence ratio"""
        try:
            board_risk = info.get('boardRisk')
            if board_risk:
                # Lower risk suggests better independence
                return max(0, min(1, 1 - (board_risk / 10)))
            return None
        except:
            return None
    
    def _estimate_renewable_energy_usage(self, info: Dict[str, Any]) -> Optional[float]:
        """Estimate renewable energy usage percentage"""
        try:
            industry = info.get('industry', '').lower()
            sector = info.get('sector', '').lower()
            
            # High renewable usage industries
            if any(kw in industry + sector for kw in ['solar', 'wind', 'renewable', 'clean energy']):
                return 0.90
            elif 'technology' in sector:
                return 0.30
            
            return None
        except:
            return None
    
    def _identify_impact_alignment(self, ticker: str, info: Dict[str, Any]) -> tuple:
        """Identify impact categories and SDG alignment"""
        categories = []
        sdgs = set()
        
        try:
            description = (info.get('longBusinessSummary', '') + ' ' + 
                          info.get('industry', '') + ' ' + 
                          info.get('sector', '')).lower()
            
            for category, details in self.impact_sectors.items():
                if any(keyword in description for keyword in details['keywords']):
                    categories.append(category)
                    sdgs.update(details['sdgs'])
            
            return categories, sorted(list(sdgs))
            
        except Exception as e:
            logger.error(f"Error identifying impact alignment: {e}")
            return [], []
    
    def _generate_impact_thesis(self, ticker: str, info: Dict[str, Any], 
                               impact_categories: List[str]) -> str:
        """Generate sustainability/impact investment thesis"""
        try:
            if not impact_categories:
                return "Standard corporate sustainability practices with potential for improvement."
            
            category_descriptions = {
                'renewable_energy': "Advancing clean energy transition and climate action",
                'clean_tech': "Enabling sustainable transportation and emissions reduction",
                'circular_economy': "Promoting resource efficiency and waste reduction",
                'healthcare_innovation': "Improving healthcare access and outcomes",
                'education_tech': "Democratizing education and skills development",
                'financial_inclusion': "Expanding financial access to underserved populations",
                'sustainable_agriculture': "Transforming food systems for sustainability",
                'water_tech': "Ensuring clean water access and conservation"
            }
            
            impacts = [category_descriptions.get(cat, cat) for cat in impact_categories[:3]]
            
            if len(impacts) == 1:
                return f"Impact focus: {impacts[0]}."
            elif len(impacts) == 2:
                return f"Dual impact: {impacts[0]} and {impacts[1]}."
            else:
                return f"Multi-impact opportunity: {', '.join(impacts[:-1])}, and {impacts[-1]}."
                
        except:
            return "Sustainability alignment under evaluation."
    
    def _extract_sustainability_initiatives(self, info: Dict[str, Any]) -> List[str]:
        """Extract known sustainability initiatives"""
        initiatives = []
        
        try:
            description = info.get('longBusinessSummary', '').lower()
            
            initiative_keywords = {
                'carbon_neutral': ['carbon neutral', 'net zero', 'carbon negative'],
                'renewable_energy': ['100% renewable', 'renewable energy commitment'],
                'circular_economy': ['circular economy', 'product recycling', 'zero waste'],
                'sustainable_supply_chain': ['sustainable sourcing', 'ethical supply chain'],
                'diversity_commitment': ['diversity and inclusion', 'equal opportunity'],
            }
            
            for initiative, keywords in initiative_keywords.items():
                if any(keyword in description for keyword in keywords):
                    initiatives.append(initiative)
            
            return initiatives
            
        except:
            return []
    
    def _identify_environmental_risks(self, info: Dict[str, Any]) -> List[str]:
        """Identify environmental risk factors"""
        risks = []
        
        try:
            # High carbon intensity
            carbon = info.get('carbonIntensity')
            if carbon and carbon > 200:
                risks.append("High carbon intensity operations")
            
            # Environmental risk score
            env_risk = info.get('environmentRisk')
            if env_risk and env_risk > 7:
                risks.append("Elevated environmental risk profile")
            
            # Sector-based risks
            sector = info.get('sector', '').lower()
            if any(kw in sector for kw in ['oil', 'mining', 'chemical']):
                risks.append("Industry with inherent environmental challenges")
                
        except:
            pass
        
        return risks if risks else ["No significant environmental risks identified"]
    
    def _identify_social_risks(self, info: Dict[str, Any]) -> List[str]:
        """Identify social risk factors"""
        risks = []
        
        try:
            social_risk = info.get('socialRisk')
            if social_risk and social_risk > 7:
                risks.append("Elevated social risk indicators")
                
        except:
            pass
        
        return risks if risks else ["No significant social risks identified"]
    
    def _identify_governance_risks(self, info: Dict[str, Any]) -> List[str]:
        """Identify governance risk factors"""
        risks = []
        
        try:
            gov_risk = info.get('governanceRisk')
            if gov_risk and gov_risk > 7:
                risks.append("Governance structure concerns")
            
            board_risk = info.get('boardRisk')
            if board_risk and board_risk > 7:
                risks.append("Board composition or independence issues")
            
            compensation_risk = info.get('compensationRisk')
            if compensation_risk and compensation_risk > 7:
                risks.append("Executive compensation alignment concerns")
                
        except:
            pass
        
        return risks if risks else ["No significant governance risks identified"]
    
    def calculate_impact_premium(self, sustainability_score: SustainabilityScore) -> float:
        """
        Calculate impact premium - additional value for sustainability leaders.
        
        Returns:
            Premium multiplier (1.0 = no premium, 1.2 = 20% premium, etc.)
        """
        try:
            base_premium = 1.0
            
            # Overall score bonus
            if sustainability_score.overall_score > 80:
                base_premium += 0.15
            elif sustainability_score.overall_score > 70:
                base_premium += 0.10
            elif sustainability_score.overall_score > 60:
                base_premium += 0.05
            
            # Impact category bonus
            impact_bonus = len(sustainability_score.impact_categories) * 0.02
            base_premium += min(impact_bonus, 0.10)
            
            # SDG alignment bonus
            sdg_bonus = len(sustainability_score.sdg_alignment) * 0.01
            base_premium += min(sdg_bonus, 0.05)
            
            return min(base_premium, 1.30)  # Cap at 30% premium
            
        except:
            return 1.0
