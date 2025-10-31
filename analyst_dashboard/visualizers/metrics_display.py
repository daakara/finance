"""
Metrics Display Manager - Handles formatting and display of financial metrics
Focused on clean presentation of financial data and ratios
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class MetricsDisplayManager:
    """Manages the display and formatting of financial metrics."""
    
    def display_key_metrics(self, metrics: Dict[str, Any], title: str = "Key Financial Metrics"):
        """Display key financial metrics in organized columns."""
        try:
            st.subheader(title)
            
            if 'error' in metrics:
                st.error(f"Error loading metrics: {metrics['error']}")
                return
            
            # Create columns for different metric categories
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Profitability**")
                self._display_metric("Net Income", metrics.get('Net Income'))
                self._display_metric("Profit Margin", metrics.get('Profit Margin'), is_percentage=True)
                self._display_metric("Operating Margin", metrics.get('Operating Margin'), is_percentage=True)
                self._display_metric("Gross Margin", metrics.get('Gross Margin'), is_percentage=True)
            
            with col2:
                st.write("**Per Share Metrics**")
                self._display_metric("EPS", metrics.get('EPS'))
                self._display_metric("Book Value/Share", metrics.get('Book Value/Share'))
                self._display_metric("Revenue/Share", metrics.get('Revenue/Share'))
                
                st.write("**Dividends**")
                self._display_metric("Dividend Yield", metrics.get('Dividend Yield'), is_percentage=True)
                self._display_metric("Dividend Rate", metrics.get('Dividend Rate'))
            
            with col3:
                st.write("**Financial Position**")
                self._display_metric("Total Cash", metrics.get('Total Cash'), is_currency=True)
                self._display_metric("Total Debt", metrics.get('Total Debt'), is_currency=True)
                self._display_metric("Free Cash Flow", metrics.get('Free Cash Flow'), is_currency=True)
                self._display_metric("Operating Cash Flow", metrics.get('Operating Cash Flow'), is_currency=True)
                
        except Exception as e:
            logger.error(f"Error displaying key metrics: {str(e)}")
            st.error("Error displaying metrics")
    
    def display_financial_ratios(self, ratios: Dict[str, Any], title: str = "Financial Ratios"):
        """Display financial ratios in organized sections."""
        try:
            st.subheader(title)
            
            if 'error' in ratios:
                st.error(f"Error loading ratios: {ratios['error']}")
                return
            
            # Create tabs for different ratio categories
            val_tab, prof_tab, liq_tab = st.tabs(["Valuation", "Profitability", "Liquidity"])
            
            with val_tab:
                col1, col2 = st.columns(2)
                with col1:
                    self._display_metric("P/E Ratio", ratios.get('P/E Ratio'))
                    self._display_metric("P/B Ratio", ratios.get('P/B Ratio'))
                    self._display_metric("P/S Ratio", ratios.get('P/S Ratio'))
                with col2:
                    self._display_metric("PEG Ratio", ratios.get('PEG Ratio'))
                    self._display_metric("EV/Revenue", ratios.get('EV/Revenue'))
                    self._display_metric("EV/EBITDA", ratios.get('EV/EBITDA'))
            
            with prof_tab:
                col1, col2 = st.columns(2)
                with col1:
                    self._display_metric("Return on Assets", ratios.get('Return on Assets'), is_percentage=True)
                    self._display_metric("Return on Equity", ratios.get('Return on Equity'), is_percentage=True)
                with col2:
                    self._display_metric("Debt-to-Equity", ratios.get('Debt-to-Equity'))
                    if 'Net Debt' in ratios:
                        self._display_metric("Net Debt", ratios.get('Net Debt'), is_currency=True)
            
            with liq_tab:
                col1, col2 = st.columns(2)
                with col1:
                    self._display_metric("Current Ratio", ratios.get('Current Ratio'))
                    self._display_metric("Quick Ratio", ratios.get('Quick Ratio'))
                with col2:
                    if 'Debt-to-Cash' in ratios:
                        self._display_metric("Debt-to-Cash", ratios.get('Debt-to-Cash'))
                
        except Exception as e:
            logger.error(f"Error displaying financial ratios: {str(e)}")
            st.error("Error displaying ratios")
    
    def display_valuation_analysis(self, valuation: Dict[str, Any], title: str = "Valuation Analysis"):
        """Display valuation analysis with interpretations."""
        try:
            st.subheader(title)
            
            if 'error' in valuation:
                st.error(f"Error loading valuation analysis: {valuation['error']}")
                return
            
            # Overall assessment
            if 'Overall_Assessment' in valuation:
                assessment = valuation['Overall_Assessment']
                if "Attractive" in assessment:
                    st.success(f"🎯 **Overall Assessment:** {assessment}")
                elif "Fair" in assessment:
                    st.info(f"⚖️ **Overall Assessment:** {assessment}")
                else:
                    st.warning(f"⚠️ **Overall Assessment:** {assessment}")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                if 'PE_Analysis' in valuation and valuation['PE_Analysis'] != 'N/A':
                    st.write("**P/E Ratio Analysis**")
                    st.write(valuation['PE_Analysis'])
                
                if 'PEG_Analysis' in valuation and valuation['PEG_Analysis'] != 'N/A':
                    st.write("**PEG Ratio Analysis**")
                    st.write(valuation['PEG_Analysis'])
            
            with col2:
                if 'PB_Analysis' in valuation and valuation['PB_Analysis'] != 'N/A':
                    st.write("**P/B Ratio Analysis**")
                    st.write(valuation['PB_Analysis'])
                
        except Exception as e:
            logger.error(f"Error displaying valuation analysis: {str(e)}")
            st.error("Error displaying valuation analysis")
    
    def display_health_score(self, health_data: Dict[str, Any], title: str = "Financial Health Score"):
        """Display financial health score with detailed breakdown."""
        try:
            st.subheader(title)
            
            if 'error' in health_data:
                st.error(f"Error loading health score: {health_data['error']}")
                return
            
            # Main score display
            score = health_data.get('score', 0)
            rating = health_data.get('rating', 'Unknown')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Color-coded score display
                if score >= 80:
                    st.success(f"**Score: {score:.1f}/100**")
                    st.success(f"**Rating: {rating}**")
                elif score >= 60:
                    st.info(f"**Score: {score:.1f}/100**")
                    st.info(f"**Rating: {rating}**")
                elif score >= 40:
                    st.warning(f"**Score: {score:.1f}/100**")
                    st.warning(f"**Rating: {rating}**")
                else:
                    st.error(f"**Score: {score:.1f}/100**")
                    st.error(f"**Rating: {rating}**")
                
                # Progress bar
                st.progress(score / 100)
            
            with col2:
                # Detailed breakdown
                details = health_data.get('details', [])
                if details:
                    st.write("**Health Check Details:**")
                    for detail in details:
                        if detail.startswith('✓'):
                            st.success(detail)
                        elif detail.startswith('⚠'):
                            st.warning(detail)
                        elif detail.startswith('✗'):
                            st.error(detail)
                        else:
                            st.write(detail)
                
        except Exception as e:
            logger.error(f"Error displaying health score: {str(e)}")
            st.error("Error displaying health score")
    
    def display_trading_signals(self, signals: Dict[str, str], title: str = "Trading Signals"):
        """Display trading signals with appropriate styling."""
        try:
            st.subheader(title)
            
            if 'Error' in signals:
                st.error(f"Error generating signals: {signals['Error']}")
                return
            
            # Create columns for different signal types
            col1, col2 = st.columns(2)
            
            with col1:
                for signal_name, signal_value in list(signals.items())[:len(signals)//2]:
                    self._display_signal(signal_name, signal_value)
            
            with col2:
                for signal_name, signal_value in list(signals.items())[len(signals)//2:]:
                    self._display_signal(signal_name, signal_value)
                
        except Exception as e:
            logger.error(f"Error displaying trading signals: {str(e)}")
            st.error("Error displaying trading signals")
    
    def display_comparison_table(self, comparison_data: Dict[str, Dict], title: str = "Asset Comparison"):
        """Display comparison data in a formatted table."""
        try:
            st.subheader(title)
            
            if not comparison_data:
                st.warning("No comparison data available")
                return
            
            # Convert to DataFrame for better display
            df_data = {}
            for asset, metrics in comparison_data.items():
                df_data[asset] = metrics
            
            df = pd.DataFrame(df_data).T
            
            # Format numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_columns] = df[numeric_columns].round(4)
            
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error displaying comparison table: {str(e)}")
            st.error("Error displaying comparison table")
    
    def _display_metric(self, label: str, value: Any, is_percentage: bool = False, 
                       is_currency: bool = False):
        """Helper method to display a single metric with appropriate formatting."""
        try:
            if value == 'N/A' or value is None:
                st.write(f"**{label}:** N/A")
                return
            
            if isinstance(value, (int, float)):
                if is_percentage:
                    formatted_value = f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"
                elif is_currency:
                    if abs(value) >= 1e9:
                        formatted_value = f"${value / 1e9:.2f}B"
                    elif abs(value) >= 1e6:
                        formatted_value = f"${value / 1e6:.2f}M"
                    elif abs(value) >= 1e3:
                        formatted_value = f"${value / 1e3:.2f}K"
                    else:
                        formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else:
                formatted_value = str(value)
            
            st.write(f"**{label}:** {formatted_value}")
            
        except Exception as e:
            logger.error(f"Error formatting metric {label}: {str(e)}")
            st.write(f"**{label}:** Error")
    
    def _display_signal(self, signal_name: str, signal_value: str):
        """Helper method to display a trading signal with appropriate styling."""
        try:
            # Color-code based on signal sentiment
            bullish_signals = ['Bullish', 'Strong Bullish', 'Oversold', 'Below Lower Band', 'Bullish Crossover']
            bearish_signals = ['Bearish', 'Strong Bearish', 'Overbought', 'Above Upper Band', 'Bearish Crossover']
            
            if any(bullish in signal_value for bullish in bullish_signals):
                st.success(f"**{signal_name}:** {signal_value}")
            elif any(bearish in signal_value for bearish in bearish_signals):
                st.error(f"**{signal_name}:** {signal_value}")
            else:
                st.info(f"**{signal_name}:** {signal_value}")
                
        except Exception as e:
            logger.error(f"Error displaying signal {signal_name}: {str(e)}")
            st.write(f"**{signal_name}:** {signal_value}")
