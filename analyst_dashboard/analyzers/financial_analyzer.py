"""
Financial Metrics Analyzer - Handles fundamental analysis calculations
Focused on financial ratios, valuation metrics, and company fundamentals
"""

import pandas as pd
import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class FinancialMetricsAnalyzer:
    """Analyzes financial metrics and fundamental data."""
    
    def analyze_financials(self, info_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive financial analysis."""
        try:
            # Extract key financial metrics
            key_metrics = self.extract_key_metrics(info_data)
            
            # Calculate financial ratios
            ratios = self.calculate_financial_ratios(info_data)
            
            # Analyze valuation metrics
            valuation = self.analyze_valuation_metrics(info_data)
            
            # Assess financial health
            health_score = self.calculate_health_score(info_data)
            
            return {
                'key_metrics': key_metrics,
                'ratios': ratios,
                'valuation': valuation,
                'health_score': health_score
            }
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {str(e)}")
            return {'error': str(e)}
    
    def extract_key_metrics(self, info_data: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """Extract key financial metrics from info data."""
        metrics = {}
        
        try:
            # Revenue metrics
            metrics['Revenue'] = info_data.get('totalRevenue', 'N/A')
            metrics['Revenue Growth'] = info_data.get('revenueGrowth', 'N/A')
            
            # Profitability metrics
            metrics['Net Income'] = info_data.get('netIncomeToCommon', 'N/A')
            metrics['Profit Margin'] = info_data.get('profitMargins', 'N/A')
            metrics['Operating Margin'] = info_data.get('operatingMargins', 'N/A')
            metrics['Gross Margin'] = info_data.get('grossMargins', 'N/A')
            
            # Per-share metrics
            metrics['EPS'] = info_data.get('trailingEps', info_data.get('forwardEps', 'N/A'))
            metrics['Book Value/Share'] = info_data.get('bookValue', 'N/A')
            metrics['Revenue/Share'] = info_data.get('revenuePerShare', 'N/A')
            
            # Dividend metrics
            metrics['Dividend Yield'] = info_data.get('dividendYield', 'N/A')
            metrics['Dividend Rate'] = info_data.get('dividendRate', 'N/A')
            metrics['Payout Ratio'] = info_data.get('payoutRatio', 'N/A')
            
            # Cash flow metrics
            metrics['Operating Cash Flow'] = info_data.get('operatingCashflow', 'N/A')
            metrics['Free Cash Flow'] = info_data.get('freeCashflow', 'N/A')
            
            # Balance sheet metrics
            metrics['Total Cash'] = info_data.get('totalCash', 'N/A')
            metrics['Total Debt'] = info_data.get('totalDebt', 'N/A')
            metrics['Total Assets'] = info_data.get('totalAssets', 'N/A')
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def calculate_financial_ratios(self, info_data: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """Calculate important financial ratios."""
        ratios = {}
        
        try:
            # Valuation ratios
            ratios['P/E Ratio'] = info_data.get('trailingPE', info_data.get('forwardPE', 'N/A'))
            ratios['P/B Ratio'] = info_data.get('priceToBook', 'N/A')
            ratios['P/S Ratio'] = info_data.get('priceToSalesTrailing12Months', 'N/A')
            ratios['PEG Ratio'] = info_data.get('pegRatio', 'N/A')
            ratios['EV/Revenue'] = info_data.get('enterpriseToRevenue', 'N/A')
            ratios['EV/EBITDA'] = info_data.get('enterpriseToEbitda', 'N/A')
            
            # Profitability ratios
            ratios['Return on Assets'] = info_data.get('returnOnAssets', 'N/A')
            ratios['Return on Equity'] = info_data.get('returnOnEquity', 'N/A')
            
            # Efficiency ratios
            ratios['Current Ratio'] = info_data.get('currentRatio', 'N/A')
            ratios['Quick Ratio'] = info_data.get('quickRatio', 'N/A')
            
            # Leverage ratios
            total_debt = info_data.get('totalDebt')
            total_cash = info_data.get('totalCash')
            if total_debt and total_cash:
                ratios['Net Debt'] = total_debt - total_cash
                ratios['Debt-to-Cash'] = total_debt / total_cash if total_cash != 0 else 'N/A'
            
            ratios['Debt-to-Equity'] = info_data.get('debtToEquity', 'N/A')
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}")
            ratios['error'] = str(e)
        
        return ratios
    
    def analyze_valuation_metrics(self, info_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics and provide interpretation."""
        valuation = {}
        
        try:
            pe_ratio = info_data.get('trailingPE')
            pb_ratio = info_data.get('priceToBook')
            ps_ratio = info_data.get('priceToSalesTrailing12Months')
            peg_ratio = info_data.get('pegRatio')
            
            # P/E Analysis
            if pe_ratio:
                if pe_ratio < 15:
                    pe_analysis = "Potentially undervalued"
                elif pe_ratio < 25:
                    pe_analysis = "Fairly valued"
                else:
                    pe_analysis = "Potentially overvalued"
                valuation['PE_Analysis'] = f"{pe_ratio:.2f} - {pe_analysis}"
            else:
                valuation['PE_Analysis'] = "N/A"
            
            # P/B Analysis
            if pb_ratio:
                if pb_ratio < 1:
                    pb_analysis = "Trading below book value"
                elif pb_ratio < 3:
                    pb_analysis = "Reasonable valuation"
                else:
                    pb_analysis = "High premium to book"
                valuation['PB_Analysis'] = f"{pb_ratio:.2f} - {pb_analysis}"
            else:
                valuation['PB_Analysis'] = "N/A"
            
            # PEG Analysis
            if peg_ratio:
                if peg_ratio < 1:
                    peg_analysis = "Growth at reasonable price"
                elif peg_ratio < 2:
                    peg_analysis = "Fair growth valuation"
                else:
                    peg_analysis = "Expensive relative to growth"
                valuation['PEG_Analysis'] = f"{peg_ratio:.2f} - {peg_analysis}"
            else:
                valuation['PEG_Analysis'] = "N/A"
            
            # Overall valuation assessment
            valuation_score = 0
            metrics_count = 0
            
            if pe_ratio and pe_ratio < 25:
                valuation_score += 1
            if pe_ratio:
                metrics_count += 1
                
            if pb_ratio and pb_ratio < 3:
                valuation_score += 1
            if pb_ratio:
                metrics_count += 1
                
            if peg_ratio and peg_ratio < 2:
                valuation_score += 1
            if peg_ratio:
                metrics_count += 1
            
            if metrics_count > 0:
                overall_score = valuation_score / metrics_count
                if overall_score >= 0.8:
                    overall_assessment = "Attractive Valuation"
                elif overall_score >= 0.5:
                    overall_assessment = "Fair Valuation"
                else:
                    overall_assessment = "Expensive Valuation"
                    
                valuation['Overall_Assessment'] = overall_assessment
            else:
                valuation['Overall_Assessment'] = "Insufficient data"
                
        except Exception as e:
            logger.error(f"Error analyzing valuation metrics: {str(e)}")
            valuation['error'] = str(e)
        
        return valuation
    
    def calculate_health_score(self, info_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial health score."""
        health_score = {}
        
        try:
            score = 0
            max_score = 0
            details = []
            
            # Profitability check
            profit_margin = info_data.get('profitMargins')
            if profit_margin is not None:
                max_score += 20
                if profit_margin > 0.15:
                    score += 20
                    details.append("✓ Strong profit margins (>15%)")
                elif profit_margin > 0.05:
                    score += 10
                    details.append("⚠ Moderate profit margins (5-15%)")
                else:
                    details.append("✗ Low profit margins (<5%)")
            
            # Debt management
            current_ratio = info_data.get('currentRatio')
            if current_ratio is not None:
                max_score += 20
                if current_ratio > 2:
                    score += 20
                    details.append("✓ Strong liquidity (Current Ratio >2)")
                elif current_ratio > 1:
                    score += 10
                    details.append("⚠ Adequate liquidity (Current Ratio >1)")
                else:
                    details.append("✗ Poor liquidity (Current Ratio <1)")
            
            # Growth
            revenue_growth = info_data.get('revenueGrowth')
            if revenue_growth is not None:
                max_score += 20
                if revenue_growth > 0.15:
                    score += 20
                    details.append("✓ Strong revenue growth (>15%)")
                elif revenue_growth > 0.05:
                    score += 10
                    details.append("⚠ Moderate revenue growth (5-15%)")
                else:
                    details.append("✗ Low revenue growth (<5%)")
            
            # Return on Equity
            roe = info_data.get('returnOnEquity')
            if roe is not None:
                max_score += 20
                if roe > 0.15:
                    score += 20
                    details.append("✓ Strong return on equity (>15%)")
                elif roe > 0.10:
                    score += 10
                    details.append("⚠ Moderate return on equity (10-15%)")
                else:
                    details.append("✗ Low return on equity (<10%)")
            
            # Cash position
            total_cash = info_data.get('totalCash')
            total_debt = info_data.get('totalDebt')
            if total_cash is not None and total_debt is not None:
                max_score += 20
                if total_cash > total_debt:
                    score += 20
                    details.append("✓ Net cash positive")
                elif total_cash > total_debt * 0.5:
                    score += 10
                    details.append("⚠ Adequate cash reserves")
                else:
                    details.append("✗ High debt relative to cash")
            
            # Calculate final score
            if max_score > 0:
                final_score = (score / max_score) * 100
                
                if final_score >= 80:
                    health_rating = "Excellent"
                elif final_score >= 60:
                    health_rating = "Good"
                elif final_score >= 40:
                    health_rating = "Fair"
                else:
                    health_rating = "Poor"
                    
                health_score['score'] = final_score
                health_score['rating'] = health_rating
                health_score['details'] = details
            else:
                health_score['score'] = 0
                health_score['rating'] = "Insufficient data"
                health_score['details'] = ["No financial data available"]
                
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            health_score['error'] = str(e)
        
        return health_score
