"""
Fundamental Analysis Renderer - Handles asset-specific fundamental analysis rendering
Focused on displaying fundamental metrics for stocks, ETFs, and cryptocurrencies
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from analysis_engine import fundamental_engine
from analysis_renderers import FundamentalRenderer as ModularFundamentalRenderer
from ui_components import ErrorHandler

logger = logging.getLogger(__name__)

class FundamentalRenderer:
    """Renders fundamental analysis using modular renderers."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render fundamental analysis using modular renderers."""
        try:
            st.subheader("🔍 Fundamental Analysis")
            
            # Get asset info
            asset_type = asset_data.get('asset_type', 'Unknown')
            asset_info = (asset_data.get('stock_info') or 
                         asset_data.get('etf_info') or 
                         asset_data.get('crypto_info', {}))
            asset_info['asset_type'] = asset_type
            
            # Perform fundamental analysis
            with st.spinner("Performing fundamental analysis..."):
                fundamental_analysis = fundamental_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in fundamental_analysis:
                ErrorHandler.handle_analysis_error(fundamental_analysis['error'], "Fundamental Analysis")
                return
            
            # Use modular renderers for each asset type
            if asset_type == 'ETF':
                ModularFundamentalRenderer.render_etf_fundamentals(fundamental_analysis)
            elif asset_type == 'Cryptocurrency':
                ModularFundamentalRenderer.render_crypto_fundamentals(fundamental_analysis)
            elif asset_type == 'Stock':
                self._render_stock_fundamentals(fundamental_analysis)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Fundamental Analysis")
    
    def _render_stock_fundamentals(self, analysis: Dict[str, Any]):
        """Render stock-specific fundamental analysis."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💼 Financial Metrics")
            if 'financial_metrics' in analysis:
                metrics = analysis['financial_metrics']
                st.metric("Market Cap", self._format_large_number(metrics.get('market_cap', 0)))
                st.metric("P/E Ratio", f"{metrics.get('pe_ratio', 0):.2f}")
                st.metric("Dividend Yield", f"{metrics.get('dividend_yield', 0):.2f}%")
                st.metric("Beta", f"{metrics.get('beta', 0):.2f}")
        
        with col2:
            st.markdown("### 📊 Valuation Analysis")
            if 'valuation_analysis' in analysis:
                valuation = analysis['valuation_analysis']
                for key, value in valuation.items():
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    def _format_large_number(self, number: float) -> str:
        """Format large numbers with appropriate suffixes."""
        if number >= 1e12:
            return f"${number/1e12:.2f}T"
        elif number >= 1e9:
            return f"${number/1e9:.2f}B"
        elif number >= 1e6:
            return f"${number/1e6:.2f}M"
        elif number >= 1e3:
            return f"${number/1e3:.2f}K"
        else:
            return f"${number:.2f}"
