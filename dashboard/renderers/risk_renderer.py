"""
Risk Analysis Renderer - Handles risk and volatility analysis rendering
Focused on displaying risk metrics, volatility analysis, and risk charts
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from analysis_engine import risk_engine
from visualizer import financial_visualizer
from ui_components import ErrorHandler

logger = logging.getLogger(__name__)

class RiskRenderer:
    """Renders risk and volatility analysis section."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render risk and volatility analysis section."""
        st.subheader("⚠️ Risk & Volatility Profile (The Edge)")
        
        price_data = asset_data['price_data']
        if price_data.empty:
            st.warning("No price data available")
            return
        
        try:
            # Perform risk analysis
            with st.spinner("Analyzing risk metrics..."):
                asset_type = asset_data.get('asset_type', 'Unknown')
                asset_info = asset_data.get(f'{asset_type.lower()}_info')
                risk_analysis = risk_engine.analyze(price_data, asset_info)
            
            if 'error' in risk_analysis:
                st.error(f"Risk analysis error: {risk_analysis['error']}")
                return
            
            # Render risk analysis components
            self._render_risk_metrics(risk_analysis)
            self._render_risk_charts(price_data)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Risk Analysis")
    
    def _render_risk_metrics(self, risk_analysis: Dict[str, Any]):
        """Render risk metrics summary."""
        st.markdown("**Risk Metrics Summary**")
        
        vol_metrics = risk_analysis.get('volatility_metrics', {})
        drawdown_metrics = risk_analysis.get('drawdown_analysis', {})
        var_metrics = risk_analysis.get('var_analysis', {})
        risk_adjusted = risk_analysis.get('risk_adjusted_returns', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_vol = vol_metrics.get('annual_volatility', 0)
            st.metric(
                "Annual Volatility",
                f"{annual_vol:.2f}%",
                help="Annualized standard deviation of returns"
            )
        
        with col2:
            max_dd = drawdown_metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                help="Maximum peak-to-trough decline"
            )
        
        with col3:
            sharpe = risk_adjusted.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            var_5 = var_metrics.get('var_5_percent', 0)
            st.metric(
                "VaR (5%)",
                f"{var_5:.2f}%",
                help="Value at Risk (5% confidence level)"
            )
    
    def _render_risk_charts(self, price_data):
        """Render volatility and drawdown charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**20-Day Rolling Volatility**")
            returns = price_data['Close'].pct_change().dropna()
            vol_chart = financial_visualizer.create_volatility_chart(
                returns,
                title="Rolling Volatility Analysis"
            )
            st.plotly_chart(vol_chart, use_container_width=True)
        
        with col2:
            st.markdown("**Drawdown Analysis**")
            dd_chart = financial_visualizer.create_drawdown_chart(
                price_data['Close'],
                title="Drawdown Analysis"
            )
            st.plotly_chart(dd_chart, use_container_width=True)
