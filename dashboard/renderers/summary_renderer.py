"""
Summary Renderer - Handles high-level summary and snapshot rendering
Focused on displaying key metrics and asset overview information
"""

import streamlit as st
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from ui_components import MetricsDisplayManager, ErrorHandler

logger = logging.getLogger(__name__)

class SummaryRenderer:
    """Renders high-level summary section using modular components."""
    
    def __init__(self):
        """Initialize the summary renderer."""
        self.metrics_manager = MetricsDisplayManager()
    
    def render(self, asset_data: Dict[str, Any]):
        """Render high-level summary section."""
        st.subheader("📋 High-Level Summary (The Snapshot)")
        
        if 'price_data' not in asset_data or asset_data['price_data'].empty:
            ErrorHandler.handle_data_error("No price data available")
            return
        
        try:
            # Calculate and display price metrics
            price_metrics = self._calculate_price_metrics(asset_data)
            
            # Get asset-specific metrics
            asset_metrics = self._get_asset_specific_metrics(asset_data)
            price_metrics.update(asset_metrics)
            
            # Display metrics using Streamlit columns directly
            self._display_custom_metrics(price_metrics)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "High-Level Summary")
    
    def _calculate_price_metrics(self, asset_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Calculate basic price metrics."""
        price_data = asset_data['price_data']
        current_price = price_data['Close'].iloc[-1]
        prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
        daily_change = current_price - prev_price
        daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0
        
        return {
            'Current Price': {
                'value': f"${current_price:.2f}",
                'delta': f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
            }
        }
    
    def _get_asset_specific_metrics(self, asset_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get asset-specific metrics for display."""
        asset_type = asset_data.get('asset_type', 'Unknown')
        asset_info = asset_data.get(f'{asset_type.lower()}_info', {})
        metrics = {}
        
        if asset_type == "Stock":
            metrics.update(self._get_stock_metrics(asset_info))
        elif asset_type == "ETF":
            metrics.update(self._get_etf_metrics(asset_info))
        elif asset_type == "Cryptocurrency":
            metrics.update(self._get_crypto_metrics(asset_info))
        
        return metrics
    
    def _get_stock_metrics(self, asset_info: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get stock-specific metrics."""
        metrics = {}
        
        market_cap = asset_info.get('marketCap', 0)
        metrics['Market Cap'] = {
            'value': self._format_large_number(market_cap) if market_cap else "N/A"
        }
        
        pe_ratio = asset_info.get('trailingPE', 0)
        metrics['P/E Ratio'] = {
            'value': f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        }
        
        div_yield = asset_info.get('dividendYield', 0)
        metrics['Dividend Yield'] = {
            'value': f"{div_yield*100:.2f}%" if div_yield else "0.00%"
        }
        
        return metrics
    
    def _get_etf_metrics(self, asset_info: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get ETF-specific metrics."""
        metrics = {}
        
        total_assets = asset_info.get('totalAssets', 0)
        metrics['Total Assets'] = {
            'value': self._format_large_number(total_assets) if total_assets else "N/A"
        }
        
        expense_ratio = asset_info.get('expenseRatio', 0)
        metrics['Expense Ratio'] = {
            'value': f"{expense_ratio:.2f}%" if expense_ratio else "N/A"
        }
        
        category = asset_info.get('category', 'N/A')
        metrics['Category'] = {'value': category}
        
        return metrics
    
    def _get_crypto_metrics(self, asset_info: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get cryptocurrency-specific metrics."""
        metrics = {}
        
        market_cap_rank = asset_info.get('market_cap_rank', 0)
        metrics['Market Cap Rank'] = {
            'value': f"#{market_cap_rank}" if market_cap_rank else "N/A"
        }
        
        market_cap = asset_info.get('market_cap', 0)
        metrics['Market Cap'] = {
            'value': self._format_large_number(market_cap) if market_cap else "N/A"
        }
        
        circ_supply = asset_info.get('circulating_supply', 0)
        metrics['Circulating Supply'] = {
            'value': self._format_large_number(circ_supply) if circ_supply else "N/A"
        }
        
        return metrics
    
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
    
    def _display_custom_metrics(self, metrics: Dict[str, Dict[str, str]]):
        """Display custom metrics in columns."""
        cols = st.columns(len(metrics))
        
        for i, (label, metric_data) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(metric_data, dict):
                    st.metric(label, metric_data['value'], metric_data.get('delta'))
                else:
                    st.metric(label, str(metric_data))
