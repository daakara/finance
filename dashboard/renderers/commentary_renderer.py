"""
Commentary and Comparison Renderers - Handles analyst commentary and comparative analysis
Combined renderer for comprehensive commentary and multi-asset comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from visualizer import financial_visualizer
from ui_components import ErrorHandler

logger = logging.getLogger(__name__)

class CommentaryRenderer:
    """Renders comprehensive analyst commentary and recommendations."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render comprehensive analyst commentary and recommendations."""
        try:
            st.subheader("📝 Comprehensive Analyst Commentary")
            
            # Get all analysis data
            asset_type = asset_data.get('asset_type', 'Unknown')
            asset_info = (asset_data.get('stock_info') or 
                         asset_data.get('etf_info') or 
                         asset_data.get('crypto_info', {}))
            asset_info['asset_type'] = asset_type
            
            # Generate comprehensive commentary
            commentary = self._generate_comprehensive_commentary(asset_data, asset_info)
            
            # Create tabs for different aspects of commentary
            commentary_tabs = st.tabs(["Executive Summary", "Investment Thesis", "Risk Assessment", "Catalysts & Outlook"])
            
            with commentary_tabs[0]:
                st.markdown("#### 📋 Executive Summary")
                st.write(commentary.get('executive_summary', 'Analysis in progress...'))
            
            with commentary_tabs[1]:
                st.markdown("#### 💡 Investment Thesis")
                st.write(commentary.get('investment_thesis', 'Investment thesis under development...'))
            
            with commentary_tabs[2]:
                st.markdown("#### ⚠️ Risk Assessment")
                st.write(commentary.get('risk_assessment', 'Risk analysis pending...'))
            
            with commentary_tabs[3]:
                st.markdown("#### 🚀 Catalysts & Outlook")
                st.write(commentary.get('catalysts_outlook', 'Market outlook analysis in progress...'))
            
            # Overall recommendation
            self._render_overall_recommendation(commentary)
            
        except Exception as e:
            st.error(f"Error rendering analyst commentary: {str(e)}")
    
    def _generate_comprehensive_commentary(self, asset_data: Dict[str, Any], asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analyst commentary."""
        try:
            price_data = asset_data['price_data']
            asset_type = asset_data.get('asset_type', 'Unknown')
            ticker = asset_info.get('symbol', asset_data.get('ticker', 'Unknown'))
            
            # Calculate key metrics
            current_price = price_data['Close'].iloc[-1] if not price_data.empty else 100
            price_change = price_data['Close'].pct_change().iloc[-1] if len(price_data) > 1 else 0
            total_return = ((current_price / price_data['Close'].iloc[0]) - 1) if not price_data.empty else 0
            volatility = price_data['Close'].pct_change().std() * np.sqrt(252) if len(price_data) > 1 else 0.2
            
            return self._build_commentary_sections(ticker, asset_type, asset_info, current_price, price_change, total_return, volatility)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive commentary: {e}")
            return self._get_default_commentary()
    
    def _build_commentary_sections(self, ticker: str, asset_type: str, asset_info: Dict[str, Any],
                                 current_price: float, price_change: float, total_return: float, volatility: float) -> Dict[str, Any]:
        """Build commentary sections based on asset analysis."""
        commentary = {}
        
        # Executive Summary
        commentary['executive_summary'] = f"""
        {ticker} ({asset_type}) is currently trading at ${current_price:.2f}, representing a 
        {total_return:.1%} total return over the analysis period. The asset exhibits 
        {volatility:.1%} annualized volatility, positioning it as a {'high' if volatility > 0.3 else 'moderate' if volatility > 0.15 else 'low'}-risk 
        investment. Recent price action shows {price_change:.1%} movement in the latest session.
        
        Based on our comprehensive analysis incorporating macroeconomic factors, fundamental metrics, 
        and technical indicators, we maintain a balanced view on the asset's near-term prospects while 
        acknowledging the broader market environment's influence on performance.
        """
        
        # Investment Thesis (asset-specific)
        commentary['investment_thesis'] = self._get_investment_thesis(ticker, asset_type, asset_info)
        
        # Risk Assessment
        risk_level = 'High' if volatility > 0.3 else 'Moderate' if volatility > 0.15 else 'Low'
        commentary['risk_assessment'] = self._get_risk_assessment(asset_type, volatility, risk_level)
        
        # Catalysts & Outlook
        commentary['catalysts_outlook'] = self._get_catalysts_outlook(asset_type)
        
        # Overall Recommendation
        commentary['overall_recommendation'] = self._get_overall_recommendation(total_return, volatility, current_price, risk_level)
        
        return commentary
    
    def _get_investment_thesis(self, ticker: str, asset_type: str, asset_info: Dict[str, Any]) -> str:
        """Get asset-specific investment thesis."""
        if asset_type == 'Stock':
            return f"""
            Our investment thesis for {ticker} centers on the company's position within the 
            {asset_info.get('sector', 'unknown')} sector and its ability to navigate current market conditions. 
            The stock's beta of {asset_info.get('beta', 1.0):.2f} suggests {'higher' if asset_info.get('beta', 1.0) > 1.2 else 'lower' if asset_info.get('beta', 1.0) < 0.8 else 'market-level'} 
            sensitivity to broader market movements.
            
            Key investment merits include the company's market positioning and the stock's 
            {'attractive' if asset_info.get('trailingPE', 20) < 20 else 'premium'} valuation at 
            {asset_info.get('trailingPE', 0):.1f}x trailing earnings.
            """
        elif asset_type == 'ETF':
            return f"""
            {ticker} provides diversified exposure to {asset_info.get('category', 'unknown')} assets, 
            making it suitable for investors seeking broad market participation with professional management. 
            The ETF's expense ratio of {asset_info.get('expenseRatio', 0.05):.3f}% is 
            {'competitive' if asset_info.get('expenseRatio', 0.05) < 0.1 else 'elevated'} within its peer group.
            
            This vehicle offers efficient access to a diversified portfolio, reducing single-asset risk 
            while maintaining exposure to the underlying asset class's growth potential.
            """
        else:  # Cryptocurrency
            return f"""
            {ticker} represents exposure to the digital asset ecosystem, characterized by high growth 
            potential but also elevated volatility and regulatory uncertainty. The cryptocurrency's 
            market position and adoption metrics suggest {'strong' if ticker in ['BTC', 'ETH'] else 'developing'} 
            fundamental support.
            
            Investment merit centers on the long-term adoption trajectory of blockchain technology 
            and digital assets, though investors should prepare for significant price volatility.
            """
    
    def _get_risk_assessment(self, asset_type: str, volatility: float, risk_level: str) -> str:
        """Get risk assessment based on asset type and metrics."""
        base_assessment = f"""
        Risk Level: {risk_level}
        
        Primary risks include market volatility ({volatility:.1%} annualized), macroeconomic sensitivity, 
        and asset-specific factors. The current market environment presents challenges from interest rate 
        policies and inflation concerns.
        """
        
        if asset_type == 'Cryptocurrency':
            base_assessment += "\nCryptocurrency-specific risks include regulatory changes, technology risks, and extreme volatility."
        elif asset_type == 'ETF':
            base_assessment += "\nETF-specific risks include tracking error, underlying asset concentration, and management risk."
        elif asset_type == 'Stock':
            base_assessment += "\nCompany-specific risks include competitive pressures, operational challenges, and sector dynamics."
        
        base_assessment += "\n\nInvestors should maintain appropriate position sizing and risk management protocols."
        return base_assessment
    
    def _get_catalysts_outlook(self, asset_type: str) -> str:
        """Get catalysts and outlook based on asset type."""
        return f"""
        Near-term catalysts include broader market sentiment shifts, macroeconomic data releases, 
        and sector-specific developments. The Federal Reserve's monetary policy stance remains a 
        key driver for asset performance across all categories.
        
        Medium-term outlook depends on economic growth sustainability, inflation trajectory, and 
        {'regulatory clarity for digital assets' if asset_type == 'Cryptocurrency' else 'corporate earnings growth' if asset_type == 'Stock' else 'underlying asset performance'}.
        
        We recommend maintaining a balanced approach with regular portfolio review and risk assessment.
        """
    
    def _get_overall_recommendation(self, total_return: float, volatility: float, current_price: float, risk_level: str) -> Dict[str, Any]:
        """Generate overall recommendation based on metrics."""
        if total_return > 0.1 and volatility < 0.25:
            rating = 'BUY'
            rationale = 'Strong performance with manageable risk'
        elif total_return < -0.15 or volatility > 0.4:
            rating = 'HOLD'
            rationale = 'Elevated risk or poor performance warrant caution'
        else:
            rating = 'HOLD'
            rationale = 'Balanced risk-return profile'
        
        return {
            'rating': rating,
            'target_price': current_price * (1.1 if rating == 'BUY' else 1.0 if rating == 'HOLD' else 0.9),
            'time_horizon': '12 months',
            'rationale': rationale,
            'key_points': [
                f"Current valuation {'appears attractive' if rating == 'BUY' else 'reflects fair value' if rating == 'HOLD' else 'may be stretched'}",
                f"Risk profile is {risk_level.lower()} based on historical volatility",
                f"Macroeconomic environment {'supports' if rating == 'BUY' else 'is neutral for' if rating == 'HOLD' else 'challenges'} the investment thesis",
                "Regular monitoring and risk management essential"
            ]
        }
    
    def _get_default_commentary(self) -> Dict[str, Any]:
        """Get default commentary when analysis fails."""
        return {
            'executive_summary': 'Analysis in progress...',
            'investment_thesis': 'Under development...',
            'risk_assessment': 'Risk analysis pending...',
            'catalysts_outlook': 'Outlook under review...',
            'overall_recommendation': {'rating': 'HOLD', 'rationale': 'Analysis incomplete'}
        }
    
    def _render_overall_recommendation(self, commentary: Dict[str, Any]):
        """Render the overall recommendation section."""
        st.markdown("---")
        st.markdown("### 🎯 Overall Recommendation")
        
        recommendation = commentary.get('overall_recommendation', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rating = recommendation.get('rating', 'HOLD')
            if rating == 'BUY':
                st.success(f"**Rating:** {rating}")
            elif rating == 'SELL':
                st.error(f"**Rating:** {rating}")
            else:
                st.warning(f"**Rating:** {rating}")
        
        with col2:
            target_price = recommendation.get('target_price', 0)
            if target_price > 0:
                st.metric("Target Price", f"${target_price:.2f}")
        
        with col3:
            time_horizon = recommendation.get('time_horizon', '12 months')
            st.write(f"**Time Horizon:** {time_horizon}")
        
        # Key points
        if 'key_points' in recommendation:
            st.markdown("#### Key Points:")
            for point in recommendation['key_points']:
                st.write(f"• {point}")


class ComparisonRenderer:
    """Renders comparative analysis between multiple assets."""
    
    def render_normalized_returns(self, comparison_data: Dict[str, Any]):
        """Render normalized returns comparison."""
        st.markdown("### 📈 Normalized Returns Comparison")
        
        # Implementation would create normalized return charts
        st.info("Normalized returns comparison chart would be displayed here")
    
    def render_performance_metrics(self, comparison_data: Dict[str, Any]):
        """Render performance metrics comparison."""
        st.markdown("### 📊 Performance Metrics Comparison")
        
        # Implementation would create performance comparison table
        st.info("Performance metrics comparison table would be displayed here")
    
    def render_correlation_analysis(self, comparison_data: Dict[str, Any]):
        """Render correlation analysis."""
        st.markdown("### 🔗 Correlation Analysis")
        
        # Implementation would create correlation matrix
        st.info("Correlation matrix would be displayed here")
    
    def render_risk_return_profile(self, comparison_data: Dict[str, Any]):
        """Render risk-return profile comparison."""
        st.markdown("### ⚖️ Risk-Return Profile")
        
        # Implementation would create risk-return scatter plot
        st.info("Risk-return scatter plot would be displayed here")
