"""
Advanced Chart Visualizer - Priority 1 Enhancement
Creates sophisticated visualizations for advanced technical indicators and multi-timeframe analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedChartVisualizer:
    """
    Creates advanced chart visualizations for Priority 1 enhancements.
    Handles multi-timeframe charts, confluence dashboards, and advanced indicators.
    """
    
    def __init__(self):
        """Initialize the advanced chart visualizer."""
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'background': '#1e1e1e',
            'grid': '#333333',
            'text': '#ffffff',
            'strong_signal': '#ffaa00',
            'weak_signal': '#666666'
        }
    
    def create_enhanced_technical_chart(self, price_data: pd.DataFrame, tech_data: pd.DataFrame, 
                                      signals: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive technical analysis chart with Priority 1 enhancements.
        """
        try:
            # Create subplots with enhanced layout
            fig = make_subplots(
                rows=5, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=(
                    'Price Action & Trend Indicators',
                    'Advanced Momentum Oscillators',
                    'Volume-Based Indicators', 
                    'Trend Strength (ADX)',
                    'Signal Confluence Dashboard'
                ),
                row_heights=[0.4, 0.2, 0.15, 0.15, 0.1]
            )
            
            # 1. Main Price Chart with Enhanced Indicators
            self._add_price_chart(fig, price_data, tech_data, row=1)
            
            # 2. Advanced Momentum Oscillators
            self._add_momentum_oscillators(fig, tech_data, row=2)
            
            # 3. Volume-Based Indicators
            self._add_volume_indicators(fig, tech_data, row=3)
            
            # 4. Trend Strength (ADX)
            self._add_trend_strength(fig, tech_data, row=4)
            
            # 5. Signal Confluence Dashboard
            self._add_confluence_dashboard(fig, signals, row=5)
            
            # Update layout
            self._update_layout(fig, "Enhanced Technical Analysis Dashboard")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating enhanced technical chart: {str(e)}")
            return self._create_error_chart(str(e))
    
    def create_multi_timeframe_chart(self, timeframe_data: Dict[str, pd.DataFrame],
                                   timeframe_analysis: Dict[str, Dict]) -> go.Figure:
        """Create multi-timeframe analysis visualization."""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                shared_xaxes=False,
                subplot_titles=[
                    'Long-term (1Y)', 'Medium-term (3M)', 
                    'Short-term (1M)', 'Timeframe Alignment',
                    'Confluence Scores', 'Signal Summary'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"type": "bar"}, {"type": "table"}]
                ]
            )
            
            # Add timeframe charts
            timeframes = ['1y', '3mo', '1mo']
            positions = [(1, 1), (1, 2), (2, 1)]
            
            for i, (period, pos) in enumerate(zip(timeframes, positions)):
                if period in timeframe_data:
                    self._add_timeframe_subplot(
                        fig, timeframe_data[period], 
                        timeframe_analysis.get(period, {}),
                        row=pos[0], col=pos[1]
                    )
            
            # Add alignment visualization
            self._add_alignment_chart(fig, timeframe_analysis, row=2, col=2)
            
            # Add confluence scores
            self._add_timeframe_confluence_scores(fig, timeframe_analysis, row=3, col=1)
            
            # Add signal summary table
            self._add_signal_summary_table(fig, timeframe_analysis, row=3, col=2)
            
            self._update_layout(fig, "Multi-Timeframe Analysis Dashboard")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-timeframe chart: {str(e)}")
            return self._create_error_chart(str(e))
    
    def create_confluence_meter(self, signals: Dict[str, Any]) -> go.Figure:
        """Create a visual confluence strength meter."""
        try:
            fig = go.Figure()
            
            confluence_score = signals.get('score', 50)
            confidence = signals.get('confidence', 50)
            signal = signals.get('signal', 'HOLD')
            
            # Create gauge chart
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=confluence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Signal Confluence<br><span style='font-size:0.8em;color:gray'>Signal: {signal}</span>"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_signal_color(confluence_score)},
                    'steps': [
                        {'range': [0, 25], 'color': "#ff4444"},
                        {'range': [25, 35], 'color': "#ff8844"},
                        {'range': [35, 65], 'color': "#888888"},
                        {'range': [65, 75], 'color': "#88ff44"},
                        {'range': [75, 100], 'color': "#00ff88"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': confluence_score
                    }
                }
            ))
            
            # Add confidence annotation
            fig.add_annotation(
                text=f"Confidence: {confidence:.0f}%",
                xref="paper", yref="paper",
                x=0.5, y=0.15,
                showarrow=False,
                font=dict(size=14, color=self.colors['text'])
            )
            
            fig.update_layout(
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font={'color': self.colors['text']},
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confluence meter: {str(e)}")
            return self._create_error_chart(str(e))
    
    def _add_price_chart(self, fig: go.Figure, price_data: pd.DataFrame, 
                        tech_data: pd.DataFrame, row: int):
        """Add enhanced price chart with multiple indicators."""
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name="Price",
                showlegend=False
            ),
            row=row, col=1
        )
        
        # Moving averages
        for ma_name, color in [('SMA_50', 'orange'), ('SMA_200', 'red')]:
            if ma_name in tech_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tech_data.index,
                        y=tech_data[ma_name],
                        name=ma_name,
                        line=dict(color=color, width=1),
                        showlegend=False
                    ),
                    row=row, col=1
                )
        
        # Parabolic SAR
        if 'PSAR' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['PSAR'],
                    mode='markers',
                    name='PSAR',
                    marker=dict(size=3, color='yellow'),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        # Bollinger Bands
        if all(col in tech_data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['BB_Upper'],
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False,
                    name='BB Upper'
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['BB_Lower'],
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False,
                    name='BB Lower'
                ),
                row=row, col=1
            )
    
    def _add_momentum_oscillators(self, fig: go.Figure, tech_data: pd.DataFrame, row: int):
        """Add advanced momentum oscillators."""
        # RSI and Stochastic RSI
        if 'RSI' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        if 'Stoch_RSI' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Stoch_RSI'],
                    name='Stoch RSI',
                    line=dict(color='orange', width=1, dash='dot'),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        if 'Williams_R' in tech_data.columns:
            # Normalize Williams %R to 0-100 scale for display
            williams_r_norm = (tech_data['Williams_R'] + 100)
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=williams_r_norm,
                    name='Williams %R',
                    line=dict(color='cyan', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        # Add reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1, opacity=0.5)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1, opacity=0.3)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1, opacity=0.3)
    
    def _add_volume_indicators(self, fig: go.Figure, tech_data: pd.DataFrame, row: int):
        """Add volume-based indicators."""
        if 'MFI' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MFI'],
                    name='Money Flow Index',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        if 'CMF' in tech_data.columns:
            # Normalize CMF for display
            cmf_norm = (tech_data['CMF'] + 1) * 50  # Convert -1 to 1 range to 0-100
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=cmf_norm,
                    name='Chaikin Money Flow',
                    line=dict(color='green', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        # Reference lines for volume indicators
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1, opacity=0.5)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=row, col=1, opacity=0.3)
    
    def _add_trend_strength(self, fig: go.Figure, tech_data: pd.DataFrame, row: int):
        """Add ADX trend strength indicator."""
        if 'ADX' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['ADX'],
                    name='ADX',
                    line=dict(color='red', width=3),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        if 'DI_Plus' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['DI_Plus'],
                    name='DI+',
                    line=dict(color='green', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        if 'DI_Minus' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['DI_Minus'],
                    name='DI-',
                    line=dict(color='red', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
        
        # ADX threshold line
        fig.add_hline(y=25, line_dash="dash", line_color="yellow", row=row, col=1, opacity=0.7)
    
    def _add_confluence_dashboard(self, fig: go.Figure, signals: Dict[str, Any], row: int):
        """Add signal confluence visualization."""
        try:
            confluence_score = signals.get('score', 50)
            confidence = signals.get('confidence', 50)
            signal = signals.get('signal', 'HOLD')
            
            # Create a simple bar showing confluence strength
            fig.add_trace(
                go.Bar(
                    x=[confluence_score],
                    y=['Confluence'],
                    orientation='h',
                    marker_color=self._get_signal_color(confluence_score),
                    showlegend=False,
                    text=[f"{signal} ({confidence:.0f}%)"],
                    textposition="middle center"
                ),
                row=row, col=1
            )
            
            # Update x-axis for confluence bar
            fig.update_xaxes(range=[0, 100], row=row, col=1)
            
        except Exception as e:
            logger.error(f"Error adding confluence dashboard: {str(e)}")
    
    def _add_timeframe_subplot(self, fig: go.Figure, data: pd.DataFrame, 
                             analysis: Dict, row: int, col: int):
        """Add individual timeframe subplot."""
        try:
            # Simple price line with trend
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f'Price',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add trend line if available
            if 'indicators' in analysis and 'sma_50' in analysis['indicators']:
                sma_50 = analysis['indicators']['sma_50']
                fig.add_trace(
                    go.Scatter(
                        x=sma_50.index,
                        y=sma_50,
                        name='SMA 50',
                        line=dict(color='orange', width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.error(f"Error adding timeframe subplot: {str(e)}")
    
    def _add_alignment_chart(self, fig: go.Figure, timeframe_analysis: Dict, row: int, col: int):
        """Add timeframe alignment visualization."""
        try:
            periods = list(timeframe_analysis.keys())
            scores = []
            
            for period in periods:
                analysis = timeframe_analysis.get(period, {})
                signals = analysis.get('signals', {})
                score = signals.get('combined_score', 50)
                scores.append(score)
            
            colors = [self._get_signal_color(score) for score in scores]
            
            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=scores,
                    marker_color=colors,
                    showlegend=False,
                    name='Alignment Scores'
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(range=[0, 100], row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding alignment chart: {str(e)}")
    
    def _add_timeframe_confluence_scores(self, fig: go.Figure, timeframe_analysis: Dict, 
                                       row: int, col: int):
        """Add confluence scores bar chart."""
        try:
            periods = list(timeframe_analysis.keys())
            scores = []
            
            for period in periods:
                analysis = timeframe_analysis.get(period, {})
                confidence = analysis.get('confidence', 0.5) * 100
                scores.append(confidence)
            
            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=scores,
                    marker_color='lightblue',
                    showlegend=False,
                    name='Confidence Scores'
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(range=[0, 100], title="Confidence %", row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding confluence scores: {str(e)}")
    
    def _add_signal_summary_table(self, fig: go.Figure, timeframe_analysis: Dict, 
                                row: int, col: int):
        """Add signal summary table."""
        try:
            periods = []
            trends = []
            signals = []
            confidences = []
            
            for period, analysis in timeframe_analysis.items():
                periods.append(period.upper())
                trend = analysis.get('trend', {}).get('direction', 'Unknown')
                trends.append(trend.replace('_', ' ').title())
                signal = analysis.get('signals', {}).get('combined_signal', 'Hold')
                signals.append(signal.replace('_', ' ').title())
                confidence = analysis.get('confidence', 0.5)
                confidences.append(f"{confidence*100:.0f}%")
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Period', 'Trend', 'Signal', 'Confidence'],
                        fill_color='lightblue',
                        align='center'
                    ),
                    cells=dict(
                        values=[periods, trends, signals, confidences],
                        fill_color='white',
                        align='center'
                    )
                ),
                row=row, col=col
            )
            
        except Exception as e:
            logger.error(f"Error adding signal summary table: {str(e)}")
    
    def _get_signal_color(self, score: float) -> str:
        """Get color based on signal score."""
        if score >= 75:
            return self.colors['bullish']
        elif score >= 65:
            return '#88ff44'
        elif score <= 25:
            return self.colors['bearish']
        elif score <= 35:
            return '#ff8844'
        else:
            return self.colors['neutral']
    
    def _update_layout(self, fig: go.Figure, title: str):
        """Update figure layout with consistent styling."""
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=self.colors['text']),
                x=0.5
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=False,
            height=800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update all axes
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            zeroline=False
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            zeroline=False
        )
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when visualization fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Chart Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=400
        )
        
        return fig
