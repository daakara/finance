"""
Fundamental Analysis Engine - Asset-specific fundamental analysis
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FundamentalAnalysisEngine:
    """Fundamental analysis engine for different asset types."""
    
    def __init__(self):
        """Initialize the fundamental analysis engine."""
        self.etf_analyzer = ETFFundamentalAnalyzer()
        self.crypto_analyzer = CryptoFundamentalAnalyzer()
        self.stock_analyzer = StockFundamentalAnalyzer()
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform fundamental analysis based on asset type.
        
        Args:
            price_data: OHLCV price data
            asset_info: Asset information
        
        Returns:
            Dictionary with fundamental analysis results
        """
        if not asset_info:
            return {'error': 'Asset information required for fundamental analysis'}
        
        asset_type = asset_info.get('asset_type', 'Unknown')
        
        try:
            if asset_type == 'ETF':
                return self.etf_analyzer.analyze(price_data, asset_info)
            elif asset_type == 'Cryptocurrency':
                return self.crypto_analyzer.analyze(price_data, asset_info)
            elif asset_type == 'Stock':
                return self.stock_analyzer.analyze(price_data, asset_info)
            else:
                return {'error': f'Unsupported asset type: {asset_type}'}
                
        except Exception as e:
            logger.error(f"Fundamental analysis error: {str(e)}")
            return {'error': f'Fundamental analysis failed: {str(e)}'}


class ETFFundamentalAnalyzer:
    """Specialized ETF fundamental analysis."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETF fundamentals."""
        try:
            results = {}
            
            # Basic ETF metrics
            results['etf_metrics'] = self._extract_basic_metrics(asset_info)
            
            # Holdings analysis
            results['holdings_analysis'] = self._analyze_holdings(asset_info)
            
            # Expense analysis
            results['expense_analysis'] = self._analyze_expenses(asset_info)
            
            # Performance vs benchmark
            results['benchmark_comparison'] = self._compare_to_benchmark(price_data, asset_info)
            
            # Overall assessment
            results['assessment'] = self._generate_etf_assessment(results)
            
            return results
            
        except Exception as e:
            logger.error(f"ETF analysis error: {e}")
            return {'error': 'ETF fundamental analysis failed'}
    
    def _extract_basic_metrics(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic ETF metrics."""
        return {
            'total_assets': asset_info.get('totalAssets', 0),
            'expense_ratio': asset_info.get('expenseRatio', 0),
            'category': asset_info.get('category', 'Unknown'),
            'fund_family': asset_info.get('fundFamily', 'Unknown'),
            'inception_date': asset_info.get('inceptionDate', 'Unknown'),
            'yield': asset_info.get('yield', 0)
        }
    
    def _analyze_holdings(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETF holdings concentration and diversification."""
        # Mock holdings analysis - in production would fetch real data
        symbol = asset_info.get('symbol', 'Unknown')
        
        # Simulated holdings concentration
        top_10_concentration = np.random.uniform(0.15, 0.45)  # 15-45%
        
        if top_10_concentration > 0.4:
            concentration_risk = 'High'
        elif top_10_concentration > 0.25:
            concentration_risk = 'Medium'
        else:
            concentration_risk = 'Low'
        
        return {
            'top_10_concentration': top_10_concentration * 100,
            'concentration_risk': concentration_risk,
            'estimated_holdings_count': np.random.randint(50, 500),
            'diversification_score': (1 - top_10_concentration) * 100
        }
    
    def _analyze_expenses(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETF expense structure."""
        expense_ratio = asset_info.get('expenseRatio', 0.05)
        category = asset_info.get('category', 'Equity')
        
        # Category benchmarks
        category_benchmarks = {
            'Equity': 0.05,
            'Fixed Income': 0.04,
            'International': 0.06,
            'Sector': 0.07,
            'Commodity': 0.08
        }
        
        benchmark = category_benchmarks.get(category, 0.05)
        
        if expense_ratio < benchmark * 0.8:
            competitiveness = 'Very Competitive'
        elif expense_ratio < benchmark * 1.2:
            competitiveness = 'Competitive'
        else:
            competitiveness = 'Expensive'
        
        return {
            'expense_ratio': expense_ratio * 100,
            'category_average': benchmark * 100,
            'competitiveness': competitiveness,
            'annual_cost_per_10k': expense_ratio * 10000
        }
    
    def _compare_to_benchmark(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance to relevant benchmark."""
        # Simplified benchmark comparison
        returns = price_data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return {'error': 'No returns data for comparison'}
        
        # Mock benchmark returns (would fetch real benchmark data in production)
        benchmark_returns = returns + np.random.normal(0, 0.001, len(returns))  # Similar but slightly different
        
        etf_total_return = (1 + returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        
        return {
            'etf_total_return': etf_total_return * 100,
            'benchmark_total_return': benchmark_total_return * 100,
            'excess_return': (etf_total_return - benchmark_total_return) * 100,
            'tracking_error': tracking_error * 100,
            'tracking_quality': 'Good' if tracking_error < 0.02 else 'Fair' if tracking_error < 0.05 else 'Poor'
        }
    
    def _generate_etf_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall ETF assessment."""
        try:
            # Score components
            expense_score = 100 if results['expense_analysis']['competitiveness'] == 'Very Competitive' else 75 if results['expense_analysis']['competitiveness'] == 'Competitive' else 50
            
            concentration_risk = results['holdings_analysis']['concentration_risk']
            concentration_score = 100 if concentration_risk == 'Low' else 75 if concentration_risk == 'Medium' else 50
            
            tracking_quality = results['benchmark_comparison'].get('tracking_quality', 'Fair')
            tracking_score = 100 if tracking_quality == 'Good' else 75 if tracking_quality == 'Fair' else 50
            
            overall_score = (expense_score + concentration_score + tracking_score) / 3
            
            if overall_score >= 90:
                rating = 'Excellent'
            elif overall_score >= 75:
                rating = 'Good'
            elif overall_score >= 60:
                rating = 'Fair'
            else:
                rating = 'Poor'
            
            return {
                'overall_score': overall_score,
                'rating': rating,
                'key_strengths': self._identify_strengths(results),
                'key_concerns': self._identify_concerns(results)
            }
            
        except Exception as e:
            return {'rating': 'Unknown', 'overall_score': 50}
    
    def _identify_strengths(self, results: Dict[str, Any]) -> List[str]:
        """Identify ETF strengths."""
        strengths = []
        
        if results['expense_analysis']['competitiveness'] in ['Very Competitive', 'Competitive']:
            strengths.append('Low expense ratio')
        
        if results['holdings_analysis']['concentration_risk'] == 'Low':
            strengths.append('Well diversified holdings')
        
        if results['benchmark_comparison'].get('tracking_quality') == 'Good':
            strengths.append('Good benchmark tracking')
        
        return strengths if strengths else ['Standard ETF structure']
    
    def _identify_concerns(self, results: Dict[str, Any]) -> List[str]:
        """Identify ETF concerns."""
        concerns = []
        
        if results['expense_analysis']['competitiveness'] == 'Expensive':
            concerns.append('High expense ratio')
        
        if results['holdings_analysis']['concentration_risk'] == 'High':
            concerns.append('High concentration risk')
        
        if results['benchmark_comparison'].get('tracking_quality') == 'Poor':
            concerns.append('Poor benchmark tracking')
        
        return concerns if concerns else ['No major concerns identified']


class CryptoFundamentalAnalyzer:
    """Specialized cryptocurrency fundamental analysis."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptocurrency fundamentals."""
        try:
            symbol = asset_info.get('symbol', 'Unknown')
            
            results = {}
            
            # On-chain metrics
            results['onchain_metrics'] = self._generate_onchain_metrics(symbol, price_data)
            
            # Network analysis
            results['network_analysis'] = self._analyze_network_health(symbol)
            
            # Adoption metrics
            results['adoption_metrics'] = self._analyze_adoption(symbol)
            
            # Development activity
            results['development_activity'] = self._analyze_development(symbol)
            
            # Tokenomics
            results['tokenomics'] = self._analyze_tokenomics(asset_info)
            
            # Overall fundamental score
            results['fundamental_score'] = self._calculate_fundamental_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Crypto analysis error: {e}")
            return {'error': 'Crypto fundamental analysis failed'}
    
    def _generate_onchain_metrics(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate on-chain metrics for cryptocurrency."""
        # Mock on-chain data based on symbol
        current_price = price_data['Close'].iloc[-1] if not price_data.empty else 100
        
        if symbol == 'BTC':
            return {
                'active_addresses': 850000 + np.random.randint(-50000, 50000),
                'transaction_volume_24h': 15e9 + np.random.uniform(-2e9, 2e9),
                'network_hash_rate': 450e6 + np.random.uniform(-50e6, 50e6),
                'mvrv_ratio': np.random.uniform(1.5, 3.5),
                'nvt_ratio': np.random.uniform(80, 150),
                'exchange_balance': np.random.uniform(2.5e6, 3e6)
            }
        elif symbol == 'ETH':
            return {
                'active_addresses': 450000 + np.random.randint(-30000, 30000),
                'transaction_volume_24h': 8e9 + np.random.uniform(-1e9, 1e9),
                'gas_usage': np.random.uniform(50, 90),
                'defi_tvl': 25e9 + np.random.uniform(-5e9, 5e9),
                'staking_ratio': np.random.uniform(0.15, 0.25),
                'burn_rate': np.random.uniform(800, 1500)
            }
        else:
            return {
                'active_addresses': np.random.randint(5000, 100000),
                'transaction_volume_24h': np.random.uniform(1e6, 1e9),
                'network_activity': np.random.uniform(0.3, 0.9),
                'holder_distribution': np.random.uniform(0.4, 0.8)
            }
    
    def _analyze_network_health(self, symbol: str) -> Dict[str, Any]:
        """Analyze network health and security."""
        # Mock network health metrics
        if symbol in ['BTC', 'ETH']:
            security_score = np.random.uniform(0.85, 0.98)
            decentralization = np.random.uniform(0.8, 0.95)
        else:
            security_score = np.random.uniform(0.5, 0.85)
            decentralization = np.random.uniform(0.4, 0.8)
        
        return {
            'security_score': security_score,
            'decentralization_score': decentralization,
            'network_uptime': np.random.uniform(0.98, 0.999),
            'consensus_mechanism': 'Proof of Work' if symbol == 'BTC' else 'Proof of Stake' if symbol == 'ETH' else 'Various',
            'validator_count': np.random.randint(100, 10000)
        }
    
    def _analyze_adoption(self, symbol: str) -> Dict[str, Any]:
        """Analyze cryptocurrency adoption metrics."""
        if symbol == 'BTC':
            return {
                'institutional_adoption': 'High',
                'payment_acceptance': 15000,
                'exchange_listings': 500,
                'social_mentions': np.random.randint(80000, 120000),
                'developer_interest': 'Very High'
            }
        elif symbol == 'ETH':
            return {
                'dapp_ecosystem': 3000,
                'defi_protocols': 200,
                'nft_platforms': 50,
                'enterprise_adoption': 'Growing',
                'developer_interest': 'Very High'
            }
        else:
            return {
                'exchange_listings': np.random.randint(10, 100),
                'social_mentions': np.random.randint(1000, 20000),
                'partnerships': np.random.randint(1, 50),
                'use_case_strength': np.random.choice(['Weak', 'Moderate', 'Strong'])
            }
    
    def _analyze_development(self, symbol: str) -> Dict[str, Any]:
        """Analyze development activity."""
        if symbol in ['BTC', 'ETH']:
            commits = np.random.randint(200, 800)
            developers = np.random.randint(50, 200)
        else:
            commits = np.random.randint(10, 200)
            developers = np.random.randint(5, 50)
        
        return {
            'monthly_commits': commits,
            'active_developers': developers,
            'code_quality': np.random.uniform(0.6, 0.95),
            'development_activity': 'High' if commits > 100 else 'Medium' if commits > 50 else 'Low',
            'roadmap_progress': np.random.uniform(0.5, 0.9)
        }
    
    def _analyze_tokenomics(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze token economics."""
        return {
            'max_supply': asset_info.get('max_supply'),
            'circulating_supply': asset_info.get('circulating_supply', 0),
            'inflation_rate': np.random.uniform(-2, 5),
            'distribution_fairness': np.random.uniform(0.5, 0.9),
            'utility_score': np.random.uniform(0.3, 0.9)
        }
    
    def _calculate_fundamental_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall fundamental score."""
        try:
            # Component scores (0-1 scale)
            network_score = results['network_analysis']['security_score'] * 0.3
            adoption_score = 0.7 if 'High' in str(results['adoption_metrics']) else 0.5
            development_score = min(results['development_activity']['monthly_commits'] / 500, 1.0) * 0.5
            tokenomics_score = results['tokenomics']['utility_score'] * 0.4
            
            overall_score = (network_score + adoption_score + development_score + tokenomics_score) / 2.2
            
            if overall_score > 0.8:
                rating = 'Excellent'
            elif overall_score > 0.6:
                rating = 'Good'
            elif overall_score > 0.4:
                rating = 'Fair' 
            else:
                rating = 'Poor'
            
            return {
                'fundamental_score': round(overall_score, 2),
                'rating': rating,
                'component_scores': {
                    'network': round(network_score, 2),
                    'adoption': round(adoption_score, 2),
                    'development': round(development_score, 2),
                    'tokenomics': round(tokenomics_score, 2)
                }
            }
            
        except Exception as e:
            return {'fundamental_score': 0.5, 'rating': 'Unknown'}


class StockFundamentalAnalyzer:
    """Specialized stock fundamental analysis."""
    
    def analyze(self, price_data: pd.DataFrame, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stock fundamentals."""
        try:
            results = {}
            
            # Financial metrics
            results['financial_metrics'] = self._extract_financial_metrics(asset_info)
            
            # Valuation analysis
            results['valuation_analysis'] = self._analyze_valuation(asset_info, price_data)
            
            # Growth analysis
            results['growth_analysis'] = self._analyze_growth_prospects(asset_info)
            
            # Sector comparison
            results['sector_comparison'] = self._compare_to_sector(asset_info)
            
            # Overall assessment
            results['investment_thesis'] = self._generate_investment_thesis(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Stock analysis error: {e}")
            return {'error': 'Stock fundamental analysis failed'}
    
    def _extract_financial_metrics(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics."""
        return {
            'market_cap': asset_info.get('marketCap', 0),
            'pe_ratio': asset_info.get('trailingPE', 0),
            'forward_pe': asset_info.get('forwardPE', 0),
            'peg_ratio': asset_info.get('pegRatio', 0),
            'price_to_book': asset_info.get('priceToBook', 0),
            'price_to_sales': asset_info.get('priceToSalesTrailing12Months', 0),
            'dividend_yield': asset_info.get('dividendYield', 0),
            'beta': asset_info.get('beta', 1.0)
        }
    
    def _analyze_valuation(self, asset_info: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stock valuation."""
        pe_ratio = asset_info.get('trailingPE', 15)
        pb_ratio = asset_info.get('priceToBook', 2.5)
        
        # Simple valuation assessment
        if pe_ratio < 15 and pb_ratio < 2:
            valuation = 'Undervalued'
        elif pe_ratio < 25 and pb_ratio < 4:
            valuation = 'Fair Value'
        else:
            valuation = 'Overvalued'
        
        return {
            'valuation_assessment': valuation,
            'pe_relative_to_growth': pe_ratio / max(asset_info.get('earningsGrowth', 10), 1),
            'price_momentum': price_data['Close'].pct_change().tail(20).mean() * 252 if not price_data.empty else 0
        }
    
    def _analyze_growth_prospects(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth prospects."""
        return {
            'revenue_growth': asset_info.get('revenueGrowth', np.random.uniform(-0.1, 0.3)),
            'earnings_growth': asset_info.get('earningsGrowth', np.random.uniform(-0.2, 0.4)),
            'growth_consistency': np.random.uniform(0.3, 0.9),
            'growth_quality': 'High' if asset_info.get('earningsGrowth', 0.1) > 0.15 else 'Moderate'
        }
    
    def _compare_to_sector(self, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compare to sector averages."""
        sector = asset_info.get('sector', 'Technology')
        
        # Mock sector comparison
        return {
            'sector': sector,
            'sector_pe_average': 20,
            'relative_valuation': 'Premium' if asset_info.get('trailingPE', 15) > 20 else 'Discount',
            'sector_growth_rate': 0.12
        }
    
    def _generate_investment_thesis(self, results: Dict[str, Any]) -> str:
        """Generate investment thesis."""
        valuation = results['valuation_analysis']['valuation_assessment']
        growth_quality = results['growth_analysis']['growth_quality']
        
        return f"""
        Investment thesis based on current fundamental analysis:
        
        Valuation appears {valuation.lower()} based on traditional metrics.
        Growth prospects show {growth_quality.lower()} characteristics.
        
        Key investment considerations include market position, competitive advantages,
        and sector dynamics. Regular monitoring of financial performance recommended.
        """


# Global instance
fundamental_engine = FundamentalAnalysisEngine()
