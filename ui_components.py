"""
UI Components Module - Reusable Streamlit UI components
Breaks down large dashboard functions into smaller, focused components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class SidebarManager:
    """Manages sidebar controls and user inputs."""
    
    def __init__(self):
        """Initialize sidebar manager."""
        self.asset_type = None
        self.ticker = None
        self.period = None
        self.analysis_mode = None
        self.comparison_tickers = []
        
    def setup_sidebar(self) -> Dict[str, Any]:
        """Setup sidebar controls and return configuration."""
        st.sidebar.title("📊 Financial Analyst Dashboard")
        st.sidebar.markdown("---")
        
        config = {}
        
        # Asset type selection
        config['asset_type'] = st.sidebar.selectbox(
            "Asset Type:",
            ["Stock", "ETF", "Cryptocurrency"],
            help="Select the type of asset to analyze"
        )
        
        # Ticker input with smart defaults
        config.update(self._setup_ticker_input(config['asset_type']))
        
        # Time period selection
        config['period'] = self._setup_period_selection()
        
        # Analysis mode
        config.update(self._setup_analysis_mode())
        
        # Technical analysis settings
        config.update(self._setup_technical_settings())
        
        return config
    
    def _setup_ticker_input(self, asset_type: str) -> Dict[str, str]:
        """Setup ticker input with smart defaults."""
        default_tickers = {
            "Stock": "AAPL",
            "ETF": "SPY", 
            "Cryptocurrency": "BTC"
        }
        
        ticker = st.sidebar.text_input(
            "Ticker Symbol:",
            value=default_tickers[asset_type],
            help="Enter the ticker symbol"
        ).upper()
        
        return {'ticker': ticker}
    
    def _setup_period_selection(self) -> str:
        """Setup time period selection."""
        return st.sidebar.selectbox(
            "Analysis Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select time period for analysis"
        )
    
    def _setup_analysis_mode(self) -> Dict[str, Any]:
        """Setup analysis mode selection."""
        st.sidebar.markdown("---")
        analysis_mode = st.sidebar.radio(
            "Analysis Mode:",
            ["Single Asset Analysis", "Comparative Analysis"],
            help="Choose between single asset analysis or multi-asset comparison"
        )
        
        config = {'analysis_mode': analysis_mode}
        
        # Comparative analysis settings
        if analysis_mode == "Comparative Analysis":
            config.update(self._setup_comparison_settings())
        
        return config
    
    def _setup_comparison_settings(self) -> Dict[str, List[str]]:
        """Setup comparative analysis settings."""
        default_comparisons = {
            "Stock": "MSFT,GOOGL,TSLA",
            "ETF": "QQQ,IWM,EFA",
            "Cryptocurrency": "ETH,ADA,SOL"
        }
        
        asset_type = st.session_state.get('asset_type', 'Stock')
        comparison_input = st.sidebar.text_area(
            "Comparison Tickers:",
            value=default_comparisons.get(asset_type, ""),
            help="Enter comma-separated ticker symbols"
        ).upper()
        
        comparison_tickers = [t.strip() for t in comparison_input.split(",") if t.strip()]
        
        return {'comparison_tickers': comparison_tickers}
    
    def _setup_technical_settings(self) -> Dict[str, Any]:
        """Setup technical analysis settings."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Technical Analysis Settings**")
        
        return {
            'show_volume': st.sidebar.checkbox("Show Volume", value=True),
            'show_indicators': st.sidebar.checkbox("Show Technical Indicators", value=True),
            'show_patterns': st.sidebar.checkbox("Show Chart Patterns", value=False)
        }

class MetricsDisplayManager:
    """Manages display of key metrics and KPIs."""
    
    @staticmethod
    def display_key_metrics(asset_data: Dict[str, Any], columns: int = 4):
        """Display key financial metrics in columns."""
        try:
            price_data = asset_data.get('price_data', pd.DataFrame())
            if price_data.empty:
                st.warning("No price data available for metrics display")
                return
            
            # Calculate key metrics
            metrics = MetricsDisplayManager._calculate_key_metrics(price_data)
            
            # Display in columns
            cols = st.columns(columns)
            metric_items = list(metrics.items())
            
            for i, (label, value) in enumerate(metric_items):
                col_idx = i % columns
                with cols[col_idx]:
                    if isinstance(value, dict):
                        st.metric(label, value['value'], value.get('delta'))
                    else:
                        st.metric(label, value)
                        
        except Exception as e:
            logger.error(f"Error displaying key metrics: {e}")
            st.error("Unable to display key metrics")
    
    @staticmethod
    def _calculate_key_metrics(price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key financial metrics from price data."""
        try:
            current_price = price_data['Close'].iloc[-1]
            previous_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
            
            # Calculate additional metrics
            high_52w = price_data['High'].max()
            low_52w = price_data['Low'].min()
            avg_volume = price_data['Volume'].mean() if 'Volume' in price_data.columns else 0
            
            return {
                "Current Price": {
                    'value': f"${current_price:.2f}",
                    'delta': f"{price_change_pct:+.2f}%"
                },
                "52W High": f"${high_52w:.2f}",
                "52W Low": f"${low_52w:.2f}",
                "Avg Volume": f"{avg_volume:,.0f}" if avg_volume > 0 else "N/A"
            }
            
        except Exception as e:
            logger.error(f"Error calculating key metrics: {e}")
            return {"Error": "Unable to calculate metrics"}

class DataSourceIndicator:
    """Manages data source indicators and status displays."""
    
    @staticmethod
    def show_data_source_status(asset_data: Dict[str, Any]):
        """Display data source status indicator."""
        try:
            data_source = asset_data.get('data_source', 'live')
            
            if data_source == 'sample':
                st.info("📊 **Demo Mode**: Using sample data due to network connectivity issues. All analysis features are fully functional.")
            else:
                st.success("📡 **Live Data**: Using real-time market data.")
                
        except Exception as e:
            logger.error(f"Error displaying data source status: {e}")
            st.warning("⚠️ **Unknown Data Source**: Unable to determine data source status.")
    
    @staticmethod
    def show_multi_asset_status(comparison_data: Dict[str, Any]):
        """Display status for multi-asset comparison."""
        try:
            sample_data_count = sum(1 for data in comparison_data.values() 
                                  if data.get('data_source') == 'sample')
            total_assets = len(comparison_data)
            
            if sample_data_count > 0:
                if sample_data_count == total_assets:
                    st.info("📊 **Demo Mode**: All data is sample data due to network issues.")
                else:
                    st.warning(f"⚠️ **Mixed Data**: {sample_data_count} out of {total_assets} assets using sample data.")
            else:
                st.success("📡 **Live Data**: All assets using real-time market data.")
                
        except Exception as e:
            logger.error(f"Error displaying multi-asset status: {e}")
            st.warning("⚠️ **Status Unknown**: Unable to determine data source status.")

class ProgressManager:
    """Manages progress bars and loading indicators."""
    
    @staticmethod
    def show_data_loading_progress(tickers: List[str], data_loader_func) -> Dict[str, Any]:
        """Show progress while loading multiple assets."""
        try:
            comparison_data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(tickers):
                try:
                    status_text.text(f"Loading {ticker}...")
                    data = data_loader_func(ticker)
                    
                    if 'error' not in data:
                        comparison_data[ticker] = data
                    
                    progress_bar.progress((i + 1) / len(tickers))
                    
                except Exception as e:
                    st.warning(f"Could not load {ticker}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error in progress management: {e}")
            return {}

class ChartManager:
    """Manages chart display and configuration."""
    
    @staticmethod
    def display_chart_with_config(chart_func, data, title: str, **kwargs):
        """Display chart with configuration options."""
        try:
            # Chart configuration options
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Chart Options**")
                height = st.slider("Height", 300, 800, 500, key=f"height_{title}")
                show_legend = st.checkbox("Show Legend", True, key=f"legend_{title}")
            
            with col1:
                # Generate and display chart
                fig = chart_func(data, title=title, height=height, **kwargs)
                
                if hasattr(fig, 'update_layout'):
                    fig.update_layout(showlegend=show_legend)
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error displaying chart {title}: {e}")
            st.error(f"Unable to display {title}")

class AnalysisTabManager:
    """Manages analysis tabs and sections."""
    
    @staticmethod
    def create_analysis_tabs(tab_names: List[str]) -> List:
        """Create analysis tabs with consistent styling."""
        return st.tabs([f"📊 {name}" for name in tab_names])
    
    @staticmethod
    def display_analysis_section(title: str, content_func, *args, **kwargs):
        """Display an analysis section with consistent formatting."""
        try:
            st.subheader(title)
            
            with st.spinner(f"Loading {title.lower()}..."):
                content_func(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in analysis section {title}: {e}")
            st.error(f"Error loading {title}: {str(e)}")

class ErrorHandler:
    """Centralized error handling for UI components."""
    
    @staticmethod
    def handle_data_error(error_msg: str, context: str = ""):
        """Handle data-related errors with user-friendly messages."""
        logger.error(f"Data error in {context}: {error_msg}")
        st.error(f"Data Error: {error_msg}")
        
        with st.expander("Troubleshooting"):
            st.write("**Possible solutions:**")
            st.write("- Check ticker symbol spelling")
            st.write("- Try a different time period")
            st.write("- Verify internet connection")
            st.write("- Switch to demo mode if available")
    
    @staticmethod
    def handle_analysis_error(error_msg: str, analysis_type: str = ""):
        """Handle analysis-related errors."""
        logger.error(f"Analysis error in {analysis_type}: {error_msg}")
        st.warning(f"Analysis Warning: Unable to complete {analysis_type} analysis")
        
        with st.expander("Error Details"):
            st.code(error_msg)
    
    @staticmethod
    def handle_visualization_error(error_msg: str, chart_type: str = ""):
        """Handle visualization-related errors."""
        logger.error(f"Visualization error in {chart_type}: {error_msg}")
        st.error(f"Chart Error: Unable to display {chart_type}")
        
        # Provide fallback message
        st.info("Please try refreshing the page or selecting different parameters.")

# Export all UI component classes
__all__ = [
    'SidebarManager',
    'MetricsDisplayManager', 
    'DataSourceIndicator',
    'ProgressManager',
    'ChartManager',
    'AnalysisTabManager',
    'ErrorHandler'
]
