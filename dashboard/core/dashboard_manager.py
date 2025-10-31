"""
Core Dashboard Manager - Handles dashboard setup and coordination
Focused on dashboard initialization, configuration, and state management
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from ui_components import SidebarManager
from dashboard.workflows.single_asset_workflow import SingleAssetWorkflow
from dashboard.workflows.comparative_workflow import ComparativeWorkflow

logger = logging.getLogger(__name__)

class DashboardManager:
    """Core dashboard manager handling setup and workflow coordination."""
    
    def __init__(self):
        """Initialize the dashboard manager."""
        self.sidebar_manager = SidebarManager()
        self.single_asset_workflow = SingleAssetWorkflow()
        self.comparative_workflow = ComparativeWorkflow()
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup dashboard configuration and sidebar controls."""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Financial Analyst Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Setup sidebar controls
        self._setup_sidebar_controls()
    
    def _setup_sidebar_controls(self):
        """Setup sidebar controls using modular components."""
        # Get all sidebar configuration from SidebarManager
        sidebar_config = self.sidebar_manager.setup_sidebar()
        
        # Extract configuration values
        self.asset_type = sidebar_config['asset_type']
        self.ticker = sidebar_config['ticker']
        self.period = sidebar_config['period']
        self.analysis_mode = sidebar_config['analysis_mode']
        self.comparison_tickers = sidebar_config.get('comparison_tickers', [])
        self.show_volume = sidebar_config.get('show_volume', True)
        self.show_indicators = sidebar_config.get('show_indicators', True)
    
    def run_dashboard(self):
        """Run the main dashboard application."""
        try:
            # Main title
            st.title("📊 Financial Analyst Dashboard")
            st.markdown(f"**Analyzing {self.asset_type}: {self.ticker} | Period: {self.period}**")
            
            # Route to appropriate workflow
            if self.analysis_mode == "Single Asset Analysis":
                self.single_asset_workflow.execute(
                    ticker=self.ticker,
                    asset_type=self.asset_type,
                    period=self.period,
                    show_volume=self.show_volume,
                    show_indicators=self.show_indicators
                )
            else:
                self.comparative_workflow.execute(
                    main_ticker=self.ticker,
                    comparison_tickers=self.comparison_tickers,
                    asset_type=self.asset_type,
                    period=self.period
                )
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state for debugging or testing."""
        return {
            'asset_type': self.asset_type,
            'ticker': self.ticker,
            'period': self.period,
            'analysis_mode': self.analysis_mode,
            'comparison_tickers': self.comparison_tickers,
            'show_volume': self.show_volume,
            'show_indicators': self.show_indicators
        }
