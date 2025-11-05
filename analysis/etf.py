"""
ETF (Exchange-Traded Fund) analysis module.
Specialized analysis for ETFs including sector allocation, holdings, and performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data.cache import cache_result
from data.fetchers import stock_fetcher

logger = logging.getLogger(__name__)

class ETFAnalyzer:
    """Main ETF analysis class."""
    
    @staticmethod
    @cache_result(ttl=3600)  # Cache for 1 hour
    def get_etf_data(
        symbol: str,
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Get comprehensive ETF data including price history and info.
        
        Args:
            symbol: ETF symbol
            period: Time period for analysis
        
        Returns:
            Dict containing ETF data and analysis
        """
        results = {}
        
        try:
            # Get price data
            price_data = stock_fetcher.get_stock_data(symbol, period)
            results['price_data'] = price_data
            
            # Get ETF info
            etf_info = stock_fetcher.get_stock_info(symbol)
            results['etf_info'] = etf_info
            
            # Calculate ETF-specific metrics
            if not price_data.empty:
                results['performance_metrics'] = ETFAnalyzer._calculate_etf_performance(price_data)
                results['risk_metrics'] = ETFAnalyzer._calculate_etf_risk_metrics(price_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing ETF {symbol}: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def _calculate_etf_performance(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ETF performance metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        
        # Calculate returns
        daily_returns = close_prices.pct_change().dropna()
        
        # Performance metrics
        total_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        annualized_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) ** (252 / len(close_prices)) - 1) * 100
        
        # Volatility
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252) * 100
        
        # Best and worst periods
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100
        
        # Rolling performance
        monthly_returns = close_prices.resample('ME').last().pct_change().dropna()
        best_month = monthly_returns.max() * 100 if not monthly_returns.empty else 0
        worst_month = monthly_returns.min() * 100 if not monthly_returns.empty else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_month': best_month,
            'worst_month': worst_month,
            'trading_days': len(daily_returns)
        }
    
    @staticmethod
    def _calculate_etf_risk_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ETF risk metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        daily_returns = close_prices.pct_change().dropna()
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Value at Risk (5%)
        var_5 = np.percentile(daily_returns, 5) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else daily_returns.std()
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        
        # Calmar ratio
        calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'var_5_percent': var_5,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    @staticmethod
    def get_popular_etfs() -> Dict[str, Dict[str, str]]:
        """Get list of popular ETFs by category."""
        return {
            'Broad Market': {
                'SPY': 'SPDR S&P 500 ETF',
                'VTI': 'Vanguard Total Stock Market ETF',
                'IWM': 'iShares Russell 2000 ETF',
                'QQQ': 'Invesco QQQ Trust'
            },
            'International': {
                'EFA': 'iShares MSCI EAFE ETF',
                'EEM': 'iShares MSCI Emerging Markets ETF',
                'VEA': 'Vanguard FTSE Developed Markets ETF',
                'VWO': 'Vanguard FTSE Emerging Markets ETF'
            },
            'Sector': {
                'XLK': 'Technology Select Sector SPDR Fund',
                'XLF': 'Financial Select Sector SPDR Fund',
                'XLE': 'Energy Select Sector SPDR Fund',
                'XLV': 'Health Care Select Sector SPDR Fund',
                'XLI': 'Industrial Select Sector SPDR Fund'
            },
            'Fixed Income': {
                'AGG': 'iShares Core U.S. Aggregate Bond ETF',
                'TLT': 'iShares 20+ Year Treasury Bond ETF',
                'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
                'LQD': 'iShares iBoxx Investment Grade Corporate Bond ETF'
            },
            'Commodities': {
                'GLD': 'SPDR Gold Shares',
                'SLV': 'iShares Silver Trust',
                'USO': 'United States Oil Fund',
                'DBA': 'Invesco DB Agriculture Fund'
            },
            'Thematic': {
                'ARKK': 'ARK Innovation ETF',
                'ICLN': 'iShares Global Clean Energy ETF',
                'HACK': 'ETFMG Prime Cyber Security ETF',
                'ROBO': 'ROBO Global Robotics and Automation Index ETF'
            }
        }
    
    @staticmethod
    def compare_etfs(
        etf_symbols: List[str],
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Compare multiple ETFs across various metrics.
        
        Args:
            etf_symbols: List of ETF symbols to compare
            period: Time period for comparison
        
        Returns:
            Dict containing comparison data
        """
        comparison_data = {}
        price_data = {}
        
        # Fetch data for all ETFs
        for symbol in etf_symbols:
            try:
                etf_data = ETFAnalyzer.get_etf_data(symbol, period)
                if 'price_data' in etf_data and not etf_data['price_data'].empty:
                    price_data[symbol] = etf_data['price_data']['Close']
                    comparison_data[symbol] = {
                        'performance': etf_data.get('performance_metrics', {}),
                        'risk': etf_data.get('risk_metrics', {}),
                        'info': etf_data.get('etf_info', {})
                    }
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        # Create comparison tables
        results = {
            'price_data': pd.DataFrame(price_data),
            'comparison_data': comparison_data
        }
        
        if comparison_data:
            # Performance comparison table
            perf_metrics = ['total_return', 'annualized_return', 'annualized_volatility']
            perf_table = []
            
            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in perf_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('performance', {}).get(metric, 0)
                perf_table.append(row)
            
            results['performance_comparison'] = pd.DataFrame(perf_table)
            
            # Risk comparison table
            risk_metrics = ['max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'var_5_percent']
            risk_table = []
            
            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in risk_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('risk', {}).get(metric, 0)
                risk_table.append(row)
            
            results['risk_comparison'] = pd.DataFrame(risk_table)
        
        return results
    
    @staticmethod
    def get_etf_expense_analysis(etf_info: Dict) -> Dict[str, Union[float, str]]:
        """
        Analyze ETF expenses and fees.
        
        Args:
            etf_info: ETF information dictionary
        
        Returns:
            Dict with expense analysis
        """
        # Extract expense ratio (this would need real data source)
        # For now, provide typical expense ratios by ETF type
        symbol = etf_info.get('symbol', '')
        
        typical_expenses = {
            'SPY': 0.09, 'VTI': 0.03, 'QQQ': 0.20, 'IWM': 0.19,
            'EFA': 0.32, 'EEM': 0.68, 'GLD': 0.40, 'TLT': 0.15,
            'XLK': 0.12, 'XLF': 0.12, 'ARKK': 0.75
        }
        
        expense_ratio = typical_expenses.get(symbol, 0.50)  # Default 0.5%
        
        # Calculate cost impact on $10,000 investment
        annual_cost = 10000 * (expense_ratio / 100)
        
        # Categorize expense level
        if expense_ratio < 0.20:
            expense_category = 'Very Low'
        elif expense_ratio < 0.50:
            expense_category = 'Low'
        elif expense_ratio < 0.75:
            expense_category = 'Moderate'
        else:
            expense_category = 'High'
        
        return {
            'expense_ratio': expense_ratio,
            'annual_cost_10k': annual_cost,
            'expense_category': expense_category,
            'cost_over_10_years': annual_cost * 10
        }

# Create instance for easy importing
etf_analyzer = ETFAnalyzer()
