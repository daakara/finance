"""
Single Asset Analysis Workflow - Handles single asset analysis orchestration
Focused on coordinating the analyst workflow for individual assets
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from data_fetcher import unified_fetcher
from dashboard.renderers.summary_renderer import SummaryRenderer
from dashboard.renderers.technical_renderer import TechnicalRenderer  
from dashboard.renderers.risk_renderer import RiskRenderer
from dashboard.renderers.macro_renderer import MacroRenderer
from dashboard.renderers.fundamental_renderer import FundamentalRenderer
from dashboard.renderers.portfolio_renderer import PortfolioRenderer, ForecastingRenderer
from dashboard.renderers.commentary_renderer import CommentaryRenderer

logger = logging.getLogger(__name__)

class SingleAssetWorkflow:
    """Orchestrates single asset analysis workflow following analyst methodology."""
    
    def __init__(self):
        """Initialize workflow with specialized renderers."""
        self.summary_renderer = SummaryRenderer()
        self.technical_renderer = TechnicalRenderer()
        self.risk_renderer = RiskRenderer()
        self.macro_renderer = MacroRenderer()
        self.fundamental_renderer = FundamentalRenderer()
        self.portfolio_renderer = PortfolioRenderer()
        self.forecasting_renderer = ForecastingRenderer()
        self.commentary_renderer = CommentaryRenderer()
    
    def execute(self, ticker: str, asset_type: str, period: str, 
                show_volume: bool = True, show_indicators: bool = True):
        """Execute the complete single asset analysis workflow."""
        try:
            # Load asset data
            asset_data = self._load_asset_data(ticker, asset_type, period)
            if asset_data is None:
                return
            
            # Execute analyst workflow sections in order
            self._execute_analyst_workflow(asset_data, show_volume, show_indicators)
            
        except Exception as e:
            st.error(f"Error in single asset workflow: {str(e)}")
            logger.error(f"Single asset workflow error: {str(e)}")
    
    def _load_asset_data(self, ticker: str, asset_type: str, period: str) -> Optional[Dict[str, Any]]:
        """Load and validate asset data."""
        with st.spinner("Loading asset data..."):
            asset_data = unified_fetcher.get_data(ticker, asset_type, period)
        
        if 'error' in asset_data:
            st.error(f"Error loading data: {asset_data['error']}")
            return None
        
        # Show data source indicator
        if asset_data.get('data_source') == 'sample':
            st.info("📊 **Demo Mode**: Using sample data due to network connectivity issues. All analysis features are fully functional.")
        else:
            st.success("📡 **Live Data**: Using real-time market data.")
        
        return asset_data
    
    def _execute_analyst_workflow(self, asset_data: Dict[str, Any], 
                                show_volume: bool, show_indicators: bool):
        """Execute the complete analyst workflow."""
        
        # 1. HIGH-LEVEL SUMMARY (The Snapshot)
        self.summary_renderer.render(asset_data)
        st.markdown("---")
        
        # 2. PRICE ACTION & TECHNICAL ANALYSIS (The Trend)
        self.technical_renderer.render(asset_data, show_volume, show_indicators)
        st.markdown("---")
        
        # 3. RISK & VOLATILITY PROFILE (The Edge)
        self.risk_renderer.render(asset_data)
        st.markdown("---")
        
        # 4. MACROECONOMIC CONTEXT & MARKET ENVIRONMENT
        self.macro_renderer.render(asset_data)
        st.markdown("---")
        
        # 5. FUNDAMENTAL ANALYSIS (Asset-Specific)
        self.fundamental_renderer.render(asset_data)
        st.markdown("---")
        
        # 6. PORTFOLIO STRATEGY & ALLOCATION
        self.portfolio_renderer.render(asset_data)
        st.markdown("---")
        
        # 7. FORECASTING & FORWARD-LOOKING OUTLOOK
        self.forecasting_renderer.render(asset_data)
        st.markdown("---")
        
        # 8. ANALYST COMMENTARY & RECOMMENDATIONS
        self.commentary_renderer.render(asset_data)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow structure."""
        return {
            'workflow_type': 'Single Asset Analysis',
            'sections': [
                'High-Level Summary',
                'Price Action & Technical Analysis', 
                'Risk & Volatility Profile',
                'Macroeconomic Context',
                'Fundamental Analysis',
                'Portfolio Strategy & Allocation',
                'Forecasting & Outlook',
                'Analyst Commentary'
            ],
            'renderers': [
                'SummaryRenderer',
                'TechnicalRenderer',
                'RiskRenderer', 
                'MacroRenderer',
                'FundamentalRenderer',
                'PortfolioRenderer',
                'ForecastingRenderer',
                'CommentaryRenderer'
            ]
        }
