"""
Technical Analysis Renderer - Handles price action and technical analysis rendering
Focused on displaying technical indicators, charts, and trading signals
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

from analysis_engine import technical_engine
from visualizer import financial_visualizer
from ui_components import ErrorHandler

logger = logging.getLogger(__name__)

class TechnicalRenderer:
    """Renders price action and technical analysis section."""
    
    def render(self, asset_data: Dict[str, Any], show_volume: bool = True, 
               show_indicators: bool = True):
        """Render price action and technical analysis section."""
        st.subheader("📈 Price Action & Technical Analysis (The Trend)")
        
        price_data = asset_data['price_data']
        if price_data.empty:
            st.warning("No price data available")
            return
        
        try:
            # Perform technical analysis
            with st.spinner("Analyzing technical indicators..."):
                asset_type = asset_data.get('asset_type', 'Unknown')
                asset_info = asset_data.get(f'{asset_type.lower()}_info')
                tech_analysis = technical_engine.analyze(price_data, asset_info)
            
            if 'error' in tech_analysis:
                st.error(f"Technical analysis error: {tech_analysis['error']}")
                return
            
            # Render technical analysis components
            self._render_price_chart(asset_data, tech_analysis, show_volume, show_indicators)
            self._render_technical_signals(tech_analysis)
            self._render_support_resistance(tech_analysis)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Technical Analysis")
    
    def _render_price_chart(self, asset_data: Dict[str, Any], tech_analysis: Dict[str, Any],
                           show_volume: bool, show_indicators: bool):
        """Render interactive price chart with technical indicators."""
        st.markdown("**Interactive Price Chart with Technical Indicators**")
        
        ticker = asset_data.get('ticker', 'Unknown')
        price_data = asset_data['price_data']
        indicators = tech_analysis.get('indicators', {}) if show_indicators else {}
        
        chart = financial_visualizer.create_advanced_candlestick_chart(
            price_data,
            indicators,
            title=f"{ticker} Technical Analysis",
            height=800
        )
        st.plotly_chart(chart, width='stretch')
    
    def _render_technical_signals(self, tech_analysis: Dict[str, Any]):
        """Render technical signals summary."""
        st.markdown("**Technical Signals Summary**")
        
        signals = tech_analysis.get('signals', {})
        trend_analysis = tech_analysis.get('trend_analysis', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            primary_trend = trend_analysis.get('primary_trend', 'Unknown')
            trend_strength = trend_analysis.get('trend_strength', 'Unknown')
            st.metric("Primary Trend", primary_trend, help=f"Strength: {trend_strength}")
        
        with col2:
            rsi_signal = signals.get('RSI', 'Unknown')
            st.metric("RSI Signal", rsi_signal)
        
        with col3:
            macd_signal = signals.get('MACD', 'Unknown')
            st.metric("MACD Signal", macd_signal)
        
        with col4:
            bb_signal = signals.get('Bollinger', 'Unknown')
            st.metric("Bollinger Bands", bb_signal)
    
    def _render_support_resistance(self, tech_analysis: Dict[str, Any]):
        """Render support and resistance levels."""
        support_resistance = tech_analysis.get('support_resistance', {})
        if not support_resistance:
            return
            
        st.markdown("**Key Levels**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            support_levels = support_resistance.get('support_levels', [])
            if support_levels:
                st.write("**Support Levels:**")
                for level in support_levels[-3:]:  # Show top 3
                    st.write(f"• ${level:.2f}")
            else:
                st.write("**Support Levels:** None identified")
        
        with col2:
            resistance_levels = support_resistance.get('resistance_levels', [])
            if resistance_levels:
                st.write("**Resistance Levels:**")
                for level in resistance_levels[:3]:  # Show top 3
                    st.write(f"• ${level:.2f}")
            else:
                st.write("**Resistance Levels:** None identified")
