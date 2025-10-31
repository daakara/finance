"""
Chart Visualizer - Handles all chart generation and plotting
Focused on creating interactive financial charts with technical indicators
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class ChartVisualizer:
    """Creates interactive financial charts and visualizations."""
    
    def create_candlestick_chart(self, price_data: pd.DataFrame, symbol: str, 
                                tech_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create an interactive candlestick chart with technical indicators."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Technical Indicators'),
                row_width=[0.2, 0.1, 0.7]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add technical indicators if available
            if tech_data is not None:
                # Moving averages
                if 'SMA_50' in tech_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=tech_data.index,
                            y=tech_data['SMA_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='orange', width=1)
                        ),
                        row=1, col=1
                    )
                
                if 'SMA_200' in tech_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=tech_data.index,
                            y=tech_data['SMA_200'],
                            mode='lines',
                            name='SMA 200',
                            line=dict(color='red', width=1)
                        ),
                        row=1, col=1
                    )
                
                # Bollinger Bands
                if all(col in tech_data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                    fig.add_trace(
                        go.Scatter(
                            x=tech_data.index,
                            y=tech_data['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=tech_data.index,
                            y=tech_data['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # RSI in third subplot
                if 'RSI' in tech_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=tech_data.index,
                            y=tech_data['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=3, col=1
                    )
                    
                    # RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                 annotation_text="Overbought", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Oversold", row=3, col=1)
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Technical Analysis Chart',
                yaxis_title='Price ($)',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            # Return empty figure on error
            return go.Figure()
    
    def create_comparison_chart(self, comparison_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create a comparison chart for multiple assets."""
        try:
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, (symbol, data) in enumerate(comparison_data.items()):
                if not data.empty and 'Close' in data.columns:
                    # Normalize to percentage change from first value
                    normalized_data = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=normalized_data,
                            mode='lines',
                            name=symbol,
                            line=dict(color=colors[i % len(colors)], width=2)
                        )
                    )
            
            fig.update_layout(
                title='Asset Performance Comparison (% Change)',
                xaxis_title='Date',
                yaxis_title='Percentage Change (%)',
                height=600,
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
            return go.Figure()
    
    def create_volume_analysis_chart(self, price_data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volume analysis chart with price overlay."""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price', 'Volume Analysis'),
                row_heights=[0.7, 0.3]
            )
            
            # Price line chart
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Volume with color coding
            volume_colors = []
            for i in range(len(price_data)):
                if i == 0:
                    volume_colors.append('gray')
                else:
                    if price_data['Close'].iloc[i] > price_data['Close'].iloc[i-1]:
                        volume_colors.append('green')
                    else:
                        volume_colors.append('red')
            
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add volume moving average
            volume_ma = price_data['Volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=volume_ma,
                    mode='lines',
                    name='Volume MA(20)',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{symbol} - Price and Volume Analysis',
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volume analysis chart: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, correlation_data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for asset comparison."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_data.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Asset Correlation Matrix',
                xaxis_title='Assets',
                yaxis_title='Assets',
                height=500,
                width=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_performance_metrics_chart(self, metrics_data: Dict[str, Dict]) -> go.Figure:
        """Create performance metrics comparison chart."""
        try:
            # Extract metrics for comparison
            symbols = list(metrics_data.keys())
            metrics = ['Return', 'Volatility', 'Sharpe_Ratio', 'Max_Drawdown']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Returns (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set3
            
            for i, metric in enumerate(metrics):
                row = i // 2 + 1
                col = i % 2 + 1
                
                values = []
                for symbol in symbols:
                    if symbol in metrics_data and metric in metrics_data[symbol]:
                        values.append(metrics_data[symbol][metric])
                    else:
                        values.append(0)
                
                fig.add_trace(
                    go.Bar(
                        x=symbols,
                        y=values,
                        name=metric,
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Performance Metrics Comparison',
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance metrics chart: {str(e)}")
            return go.Figure()
