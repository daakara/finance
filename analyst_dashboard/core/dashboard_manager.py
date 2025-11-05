"""
Analyst Dashboard Core Manager - Handles application setup and coordination
Focused on dashboard initialization, configuration, and state management
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from analyst_dashboard.workflows.single_asset_workflow import SingleAssetWorkflow
from analyst_dashboard.workflows.comparative_workflow import ComparativeAnalysisWorkflow

logger = logging.getLogger(__name__)

class AnalystDashboardManager:
    """Core analyst dashboard manager handling setup and workflow coordination."""
    
    def __init__(self):
        """Initialize the analyst dashboard manager."""
        self._setup_page_config()
        self._setup_sidebar()
        
        # Initialize workflows
        self.single_asset_workflow = SingleAssetWorkflow()
        self.comparative_workflow = ComparativeAnalysisWorkflow()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Financial Analyst Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _setup_sidebar(self):
        """Setup sidebar controls and capture user inputs."""
        st.sidebar.title("📊 Financial Analyst Dashboard")
        st.sidebar.markdown("---")
        
        # Asset type selection
        self.asset_type = st.sidebar.selectbox(
            "Asset Type:",
            ["Stock", "ETF", "Cryptocurrency"],
            help="Select the type of asset to analyze"
        )
        
        # Ticker input with smart defaults
        default_tickers = {
            "Stock": "AAPL",
            "ETF": "SPY",
            "Cryptocurrency": "BTC/USDT"
        }
        
        self.ticker = st.sidebar.text_input(
            "Ticker Symbol:",
            value=default_tickers[self.asset_type],
            help="Enter the ticker symbol"
        ).upper()
        
        # Time period selection
        self.period = st.sidebar.selectbox(
            "Analysis Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select time period for analysis"
        )
        
        # Analysis mode selection
        st.sidebar.markdown("---")
        self.analysis_mode = st.sidebar.radio(
            "Analysis Mode:",
            ["Single Asset Analysis", "Comparative Analysis"],
            help="Choose between single asset deep dive or multi-asset comparison"
        )
        
        # Comparative analysis settings
        if self.analysis_mode == "Comparative Analysis":
            self._setup_comparison_settings()
        
        # Additional controls
        st.sidebar.markdown("---")
        self.refresh_data = st.sidebar.button("🔄 Refresh Data", help="Refresh all data")
        
        # Market status
        self._show_market_status()
    
    def _setup_comparison_settings(self):
        """Setup comparative analysis settings."""
        default_comparisons = {
            "Stock": "MSFT,GOOGL,TSLA",
            "ETF": "QQQ,IWM,EFA",
            "Cryptocurrency": "ETH,ADA,SOL"
        }
        
        comparison_input = st.sidebar.text_area(
            "Comparison Tickers:",
            value=default_comparisons[self.asset_type],
            help="Enter comma-separated ticker symbols for comparison"
        ).upper()
        
        self.comparison_tickers = [t.strip() for t in comparison_input.split(",") if t.strip()]
    
    def _show_market_status(self):
        """Show current market status in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Market Status**")
        
        # Market status logic
        market_open = datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16
        
        if market_open:
            st.sidebar.success("🟢 Market Open")
        else:
            st.sidebar.info("🔴 Market Closed")
        
        st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    def run_dashboard(self):
        """Run the main analyst dashboard application."""
        try:
            # Main title
            st.title("📊 Financial Analyst Dashboard")
            st.markdown(f"**Analyzing {self.asset_type}: {self.ticker}**")
            
            # Route to appropriate workflow
            if self.analysis_mode == "Single Asset Analysis":
                self.single_asset_workflow.run_complete_analysis(
                    symbol=self.ticker,
                    asset_type=self.asset_type.lower(),
                    time_period=self.period
                )
            else:
                # For comparative analysis, use single asset workflow for now
                # (Comparative workflow can be implemented in Priority 4)
                st.info("🚧 Comparative analysis will be available in Priority 4. Showing single asset analysis for now.")
                self.single_asset_workflow.run_complete_analysis(
                    symbol=self.ticker,
                    asset_type=self.asset_type.lower(),
                    time_period=self.period
                )
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state for debugging."""
        return {
            'asset_type': self.asset_type,
            'ticker': self.ticker,
            'period': self.period,
            'analysis_mode': self.analysis_mode,
            'comparison_tickers': getattr(self, 'comparison_tickers', []),
            'refresh_data': self.refresh_data
        }
