"""
Comparative Analysis Workflow - Handles multi-asset comparative analysis
Focused on orchestrating comparative analysis between multiple assets
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any

from data_fetcher import unified_fetcher
from dashboard.renderers.commentary_renderer import ComparisonRenderer

logger = logging.getLogger(__name__)

class ComparativeWorkflow:
    """Orchestrates comparative analysis workflow for multiple assets."""
    
    def __init__(self):
        """Initialize workflow with comparison renderer."""
        self.comparison_renderer = ComparisonRenderer()
    
    def execute(self, main_ticker: str, comparison_tickers: List[str], 
                asset_type: str, period: str):
        """Execute the comparative analysis workflow."""
        try:
            st.subheader("🔍 Comparative Analysis")
            
            all_tickers = [main_ticker] + comparison_tickers
            st.markdown(f"**Comparing {len(all_tickers)} assets: {', '.join(all_tickers)}**")
            
            # Load comparison data
            comparison_data = self._load_comparison_data(all_tickers, asset_type, period)
            if len(comparison_data) < 2:
                st.error("Need at least 2 assets with valid data for comparison")
                return
            
            # Execute comparative analysis sections
            self._execute_comparative_analysis(comparison_data)
            
        except Exception as e:
            st.error(f"Error in comparative analysis: {str(e)}")
            logger.error(f"Comparative analysis error: {str(e)}")
    
    def _load_comparison_data(self, all_tickers: List[str], asset_type: str, 
                            period: str) -> Dict[str, Any]:
        """Load data for all assets in comparison."""
        comparison_data = {}
        progress_bar = st.progress(0)
        sample_data_count = 0
        
        for i, ticker in enumerate(all_tickers):
            try:
                with st.spinner(f"Loading {ticker}..."):
                    data = unified_fetcher.get_data(ticker, asset_type, period)
                    if 'error' not in data:
                        comparison_data[ticker] = data
                        if data.get('data_source') == 'sample':
                            sample_data_count += 1
                progress_bar.progress((i + 1) / len(all_tickers))
            except Exception as e:
                st.warning(f"Could not load {ticker}: {str(e)}")
        
        progress_bar.empty()
        
        # Show data source summary
        self._show_data_source_summary(sample_data_count, len(comparison_data))
        
        return comparison_data
    
    def _show_data_source_summary(self, sample_count: int, total_count: int):
        """Show summary of data sources used."""
        if sample_count > 0:
            if sample_count == total_count:
                st.info("📊 **Demo Mode**: All data is sample data due to network issues.")
            else:
                st.warning(f"⚠️ **Mixed Data**: {sample_count} out of {total_count} assets using sample data.")
        else:
            st.success("📡 **Live Data**: All assets using real-time market data.")
    
    def _execute_comparative_analysis(self, comparison_data: Dict[str, Any]):
        """Execute comparative analysis sections."""
        
        # 1. Normalized Returns Comparison
        self.comparison_renderer.render_normalized_returns(comparison_data)
        st.markdown("---")
        
        # 2. Performance Metrics Comparison  
        self.comparison_renderer.render_performance_metrics(comparison_data)
        st.markdown("---")
        
        # 3. Correlation Matrix
        self.comparison_renderer.render_correlation_analysis(comparison_data)
        st.markdown("---")
        
        # 4. Risk-Return Profile
        self.comparison_renderer.render_risk_return_profile(comparison_data)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the comparative workflow."""
        return {
            'workflow_type': 'Comparative Analysis',
            'sections': [
                'Normalized Returns Comparison',
                'Performance Metrics Comparison',
                'Correlation Analysis', 
                'Risk-Return Profile'
            ],
            'renderers': ['ComparisonRenderer']
        }
