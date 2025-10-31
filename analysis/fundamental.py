"""
Fundamental analysis calculations and financial metrics.
Implements key financial ratios and valuation models.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from data.cache import cache_result

logger = logging.getLogger(__name__)

class FinancialRatios:
    """Calculate financial ratios and metrics."""
    
    @staticmethod
    def calculate_pe_ratio(price: float, eps: float) -> Optional[float]:
        """
        Calculate Price-to-Earnings ratio.
        
        Args:
            price: Current stock price
            eps: Earnings per share
        
        Returns:
            P/E ratio or None if invalid
        """
        if eps and eps != 0:
            return price / eps
        return None
    
    @staticmethod
    def calculate_peg_ratio(pe_ratio: float, growth_rate: float) -> Optional[float]:
        """
        Calculate Price/Earnings to Growth ratio.
        
        Args:
            pe_ratio: P/E ratio
            growth_rate: Earnings growth rate (as percentage)
        
        Returns:
            PEG ratio or None if invalid
        """
        if pe_ratio and growth_rate and growth_rate != 0:
            return pe_ratio / growth_rate
        return None
    
    @staticmethod
    def calculate_pb_ratio(price: float, book_value_per_share: float) -> Optional[float]:
        """
        Calculate Price-to-Book ratio.
        
        Args:
            price: Current stock price
            book_value_per_share: Book value per share
        
        Returns:
            P/B ratio or None if invalid
        """
        if book_value_per_share and book_value_per_share != 0:
            return price / book_value_per_share
        return None
    
    @staticmethod
    def calculate_ps_ratio(market_cap: float, revenue: float) -> Optional[float]:
        """
        Calculate Price-to-Sales ratio.
        
        Args:
            market_cap: Market capitalization
            revenue: Total revenue
        
        Returns:
            P/S ratio or None if invalid
        """
        if revenue and revenue != 0:
            return market_cap / revenue
        return None
    
    @staticmethod
    def calculate_debt_to_equity(total_debt: float, total_equity: float) -> Optional[float]:
        """
        Calculate Debt-to-Equity ratio.
        
        Args:
            total_debt: Total debt
            total_equity: Total equity
        
        Returns:
            D/E ratio or None if invalid
        """
        if total_equity and total_equity != 0:
            return total_debt / total_equity
        return None
    
    @staticmethod
    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
        """
        Calculate Current ratio.
        
        Args:
            current_assets: Current assets
            current_liabilities: Current liabilities
        
        Returns:
            Current ratio or None if invalid
        """
        if current_liabilities and current_liabilities != 0:
            return current_assets / current_liabilities
        return None
    
    @staticmethod
    def calculate_quick_ratio(
        current_assets: float,
        inventory: float,
        current_liabilities: float
    ) -> Optional[float]:
        """
        Calculate Quick ratio (Acid-test ratio).
        
        Args:
            current_assets: Current assets
            inventory: Inventory value
            current_liabilities: Current liabilities
        
        Returns:
            Quick ratio or None if invalid
        """
        if current_liabilities and current_liabilities != 0:
            quick_assets = current_assets - inventory
            return quick_assets / current_liabilities
        return None
    
    @staticmethod
    def calculate_roe(net_income: float, shareholders_equity: float) -> Optional[float]:
        """
        Calculate Return on Equity.
        
        Args:
            net_income: Net income
            shareholders_equity: Total shareholders' equity
        
        Returns:
            ROE as percentage or None if invalid
        """
        if shareholders_equity and shareholders_equity != 0:
            return (net_income / shareholders_equity) * 100
        return None
    
    @staticmethod
    def calculate_roa(net_income: float, total_assets: float) -> Optional[float]:
        """
        Calculate Return on Assets.
        
        Args:
            net_income: Net income
            total_assets: Total assets
        
        Returns:
            ROA as percentage or None if invalid
        """
        if total_assets and total_assets != 0:
            return (net_income / total_assets) * 100
        return None
    
    @staticmethod
    def calculate_roic(
        net_income: float,
        dividends: float,
        total_debt: float,
        total_equity: float
    ) -> Optional[float]:
        """
        Calculate Return on Invested Capital.
        
        Args:
            net_income: Net income
            dividends: Dividends paid
            total_debt: Total debt
            total_equity: Total equity
        
        Returns:
            ROIC as percentage or None if invalid
        """
        invested_capital = total_debt + total_equity
        if invested_capital and invested_capital != 0:
            nopat = net_income - dividends  # Simplified NOPAT calculation
            return (nopat / invested_capital) * 100
        return None

class ValuationModels:
    """Implement various valuation models."""
    
    @staticmethod
    def discounted_cash_flow(
        free_cash_flows: List[float],
        terminal_value: float,
        discount_rate: float,
        growth_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate intrinsic value using DCF model.
        
        Args:
            free_cash_flows: List of projected free cash flows
            terminal_value: Terminal value
            discount_rate: Discount rate (WACC)
            growth_rate: Terminal growth rate
        
        Returns:
            Dict containing DCF valuation components
        """
        if not free_cash_flows or discount_rate <= 0:
            return {'error': 'Invalid inputs for DCF calculation'}
        
        # Calculate present value of cash flows
        pv_cash_flows = []
        for i, fcf in enumerate(free_cash_flows, 1):
            pv = fcf / ((1 + discount_rate) ** i)
            pv_cash_flows.append(pv)
        
        # Calculate present value of terminal value
        terminal_years = len(free_cash_flows)
        pv_terminal = terminal_value / ((1 + discount_rate) ** terminal_years)
        
        # Total enterprise value
        enterprise_value = sum(pv_cash_flows) + pv_terminal
        
        return {
            'enterprise_value': enterprise_value,
            'pv_cash_flows': sum(pv_cash_flows),
            'pv_terminal_value': pv_terminal,
            'terminal_value': terminal_value
        }
    
    @staticmethod
    def dividend_discount_model(
        current_dividend: float,
        growth_rate: float,
        required_return: float
    ) -> Optional[float]:
        """
        Calculate fair value using Gordon Growth Model.
        
        Args:
            current_dividend: Current annual dividend
            growth_rate: Expected dividend growth rate
            required_return: Required rate of return
        
        Returns:
            Fair value or None if invalid
        """
        if required_return <= growth_rate or required_return <= 0:
            return None
        
        next_dividend = current_dividend * (1 + growth_rate)
        fair_value = next_dividend / (required_return - growth_rate)
        
        return fair_value
    
    @staticmethod
    def calculate_wacc(
        market_value_equity: float,
        market_value_debt: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital.
        
        Args:
            market_value_equity: Market value of equity
            market_value_debt: Market value of debt
            cost_of_equity: Cost of equity
            cost_of_debt: Cost of debt
            tax_rate: Corporate tax rate
        
        Returns:
            WACC
        """
        total_value = market_value_equity + market_value_debt
        
        if total_value == 0:
            return 0
        
        equity_weight = market_value_equity / total_value
        debt_weight = market_value_debt / total_value
        
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc

class FundamentalAnalysis:
    """Main fundamental analysis class."""
    
    def __init__(self):
        self.ratios = FinancialRatios()
        self.valuation = ValuationModels()
        
    def analyze_fundamentals(self, stock_info: Dict) -> Dict[str, Union[float, str, None]]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            stock_info: Dictionary containing stock fundamental data
        
        Returns:
            Dict containing fundamental analysis results
        """
        if not stock_info:
            return {'error': 'No stock information provided'}
        
        analysis = {}
        
        try:
            # Basic information
            analysis['symbol'] = stock_info.get('symbol', 'N/A')
            analysis['company_name'] = stock_info.get('company_name', 'N/A')
            analysis['sector'] = stock_info.get('sector', 'N/A')
            analysis['industry'] = stock_info.get('industry', 'N/A')
            
            # Valuation ratios
            current_price = stock_info.get('current_price')
            analysis['current_price'] = current_price
            analysis['pe_ratio'] = stock_info.get('pe_ratio')
            analysis['forward_pe'] = stock_info.get('forward_pe')
            analysis['peg_ratio'] = stock_info.get('peg_ratio')
            analysis['price_to_book'] = stock_info.get('price_to_book')
            
            # Financial health
            analysis['debt_to_equity'] = stock_info.get('debt_to_equity')
            analysis['current_ratio'] = stock_info.get('current_ratio')
            analysis['quick_ratio'] = stock_info.get('quick_ratio')
            
            # Profitability
            analysis['roe'] = stock_info.get('roe')
            analysis['roa'] = stock_info.get('roa')
            analysis['profit_margin'] = stock_info.get('profit_margin')
            
            # Market metrics
            analysis['market_cap'] = stock_info.get('market_cap')
            analysis['beta'] = stock_info.get('beta')
            analysis['dividend_yield'] = stock_info.get('dividend_yield')
            
            # Price ranges
            analysis['52_week_high'] = stock_info.get('52_week_high')
            analysis['52_week_low'] = stock_info.get('52_week_low')
            
            # Calculate additional metrics if data is available
            if current_price and analysis['52_week_high'] and analysis['52_week_low']:
                week_52_range = analysis['52_week_high'] - analysis['52_week_low']
                if week_52_range > 0:
                    analysis['position_in_52_week_range'] = (
                        (current_price - analysis['52_week_low']) / week_52_range
                    ) * 100
            
            # Quality score calculation
            analysis['quality_score'] = self._calculate_quality_score(analysis)
            
            # Value score calculation
            analysis['value_score'] = self._calculate_value_score(analysis)
            
            # Overall recommendation
            analysis['recommendation'] = self._generate_recommendation(analysis)
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """
        Calculate a quality score based on financial metrics.
        
        Args:
            analysis: Analysis results dictionary
        
        Returns:
            Quality score (0-100)
        """
        score = 0
        max_score = 0
        
        # ROE score (higher is better)
        if analysis.get('roe') is not None:
            roe = analysis['roe']
            if roe > 15:
                score += 20
            elif roe > 10:
                score += 15
            elif roe > 5:
                score += 10
            max_score += 20
        
        # Debt-to-Equity score (lower is better)
        if analysis.get('debt_to_equity') is not None:
            de_ratio = analysis['debt_to_equity']
            if de_ratio < 0.3:
                score += 20
            elif de_ratio < 0.6:
                score += 15
            elif de_ratio < 1.0:
                score += 10
            max_score += 20
        
        # Profit margin score (higher is better)
        if analysis.get('profit_margin') is not None:
            margin = analysis['profit_margin'] * 100  # Convert to percentage
            if margin > 20:
                score += 20
            elif margin > 10:
                score += 15
            elif margin > 5:
                score += 10
            max_score += 20
        
        # Current ratio score (around 2 is ideal)
        if analysis.get('current_ratio') is not None:
            current_ratio = analysis['current_ratio']
            if 1.5 <= current_ratio <= 3.0:
                score += 20
            elif 1.0 <= current_ratio < 1.5 or 3.0 < current_ratio <= 4.0:
                score += 15
            elif current_ratio >= 1.0:
                score += 10
            max_score += 20
        
        # Beta score (lower volatility preferred for quality)
        if analysis.get('beta') is not None:
            beta = analysis['beta']
            if 0.5 <= beta <= 1.2:
                score += 20
            elif beta < 0.5 or (1.2 < beta <= 1.5):
                score += 15
            elif beta > 0:
                score += 10
            max_score += 20
        
        return (score / max_score * 100) if max_score > 0 else 0
    
    def _calculate_value_score(self, analysis: Dict) -> float:
        """
        Calculate a value score based on valuation metrics.
        
        Args:
            analysis: Analysis results dictionary
        
        Returns:
            Value score (0-100)
        """
        score = 0
        max_score = 0
        
        # P/E ratio score (lower is better for value)
        if analysis.get('pe_ratio') is not None:
            pe = analysis['pe_ratio']
            if pe < 15:
                score += 25
            elif pe < 20:
                score += 20
            elif pe < 25:
                score += 15
            elif pe < 30:
                score += 10
            max_score += 25
        
        # PEG ratio score (closer to 1 is better)
        if analysis.get('peg_ratio') is not None:
            peg = analysis['peg_ratio']
            if 0.5 <= peg <= 1.5:
                score += 25
            elif 0.3 <= peg < 0.5 or 1.5 < peg <= 2.0:
                score += 20
            elif peg > 0:
                score += 10
            max_score += 25
        
        # Price-to-Book score (lower is better for value)
        if analysis.get('price_to_book') is not None:
            pb = analysis['price_to_book']
            if pb < 1.0:
                score += 25
            elif pb < 2.0:
                score += 20
            elif pb < 3.0:
                score += 15
            elif pb < 5.0:
                score += 10
            max_score += 25
        
        # Dividend yield score (higher is better for value)
        if analysis.get('dividend_yield') is not None:
            div_yield = analysis['dividend_yield'] * 100  # Convert to percentage
            if div_yield > 4:
                score += 25
            elif div_yield > 2:
                score += 20
            elif div_yield > 1:
                score += 15
            elif div_yield > 0:
                score += 10
            max_score += 25
        
        return (score / max_score * 100) if max_score > 0 else 0
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """
        Generate investment recommendation based on analysis.
        
        Args:
            analysis: Analysis results dictionary
        
        Returns:
            Investment recommendation
        """
        quality_score = analysis.get('quality_score', 0)
        value_score = analysis.get('value_score', 0)
        
        if quality_score >= 70 and value_score >= 70:
            return 'Strong Buy'
        elif quality_score >= 60 and value_score >= 60:
            return 'Buy'
        elif quality_score >= 40 and value_score >= 40:
            return 'Hold'
        elif quality_score < 30 or value_score < 30:
            return 'Sell'
        else:
            return 'Weak Hold'

# Create instances for easy importing
financial_ratios = FinancialRatios()
valuation_models = ValuationModels()
fundamental_analysis = FundamentalAnalysis()
