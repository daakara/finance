"""
Macroeconomic Analysis Renderer - Handles macro context rendering
Focused on displaying macroeconomic factors and market environment analysis
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from analysis_engine import macro_engine
from analysis_renderers import MacroeconomicRenderer
from ui_components import ErrorHandler, MetricsDisplayManager

logger = logging.getLogger(__name__)

class MacroRenderer:
    """Renders macroeconomic context and market environment analysis."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render macroeconomic context using modular renderer."""
        try:
            st.subheader("🌍 Macroeconomic Context & Market Environment")
            
            # Get asset info for analysis
            asset_type = asset_data.get('asset_type', 'Unknown')
            asset_info = (asset_data.get('stock_info') or 
                         asset_data.get('etf_info') or 
                         asset_data.get('crypto_info', {}))
            asset_info['asset_type'] = asset_type
            
            # Perform macroeconomic analysis
            with st.spinner("Analyzing macroeconomic factors..."):
                macro_analysis = macro_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in macro_analysis:
                ErrorHandler.handle_analysis_error(macro_analysis['error'], "Macroeconomic Analysis")
                return
            
            # Use modular renderers for each section
            self._render_macro_sections(macro_analysis)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Macroeconomic Analysis")
    
    def _render_macro_sections(self, macro_analysis: Dict[str, Any]):
        """Render macroeconomic analysis sections."""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'monetary_policy' in macro_analysis:
                MacroeconomicRenderer.render_monetary_policy_section(macro_analysis['monetary_policy'])
            
            if 'inflation_impact' in macro_analysis:
                MacroeconomicRenderer.render_inflation_section(macro_analysis['inflation_impact'])
        
        with col2:
            if 'market_correlations' in macro_analysis:
                MacroeconomicRenderer.render_market_correlations_section(macro_analysis['market_correlations'])
            
            # Economic cycle section
            if 'economic_cycle' in macro_analysis:
                self._render_economic_cycle(macro_analysis['economic_cycle'])
        
        # Interest rate sensitivity chart
        if 'interest_rate_sensitivity' in macro_analysis:
            self._render_interest_rate_sensitivity(macro_analysis['interest_rate_sensitivity'])
    
    def _render_economic_cycle(self, cycle_data: Dict[str, Any]):
        """Render economic cycle information."""
        st.markdown("### 📊 Economic Cycle")
        
        metrics_manager = MetricsDisplayManager()
        cycle_metrics = {
            'GDP Growth': {'value': f"{cycle_data.get('gdp_growth', 0):.1f}%"},
            'Unemployment': {'value': f"{cycle_data.get('unemployment_rate', 0):.1f}%"}
        }
        metrics_manager.display_metrics_row(cycle_metrics)
        
        st.write(f"**Current Phase:** {cycle_data.get('current_cycle_phase', 'Unknown')}")
        st.write(f"**Recent Performance:** {cycle_data.get('recent_performance', 0):.1f}%")
    
    def _render_interest_rate_sensitivity(self, irs_data: Dict[str, Any]):
        """Render interest rate sensitivity analysis."""
        st.markdown("### 📈 Interest Rate Sensitivity Analysis")
        
        sensitivity_metrics = {
            'Rate Correlation': {'value': f"{irs_data.get('rate_correlation', 0):.3f}"},
            'Modified Duration': {'value': f"{irs_data.get('modified_duration', 0):.2f}"},
            'Sensitivity Level': {'value': irs_data.get('sensitivity_level', 'Unknown')}
        }
        
        metrics_manager = MetricsDisplayManager()
        metrics_manager.display_metrics_row(sensitivity_metrics)
