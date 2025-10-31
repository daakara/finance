"""
Visualizer Module - Advanced financial data visualization
Creates professional charts for technical analysis, risk assessment, and comparative analysis
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from config import config

logger = logging.getLogger(__name__)

class FinancialVisualizer:
    """Main visualizer class for financial charts."""
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        self.color_palette = [
            config.COLORS['bullish'],    # Green
            config.COLORS['bearish'],    # Red  
            config.COLORS['neutral'],    # Blue
            '#FFD700',                   # Gold
            '#FF6B35',                   # Orange
            '#9B59B6'                    # Purple
        ]
        
        self.default_layout = dict(
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background'],
            hovermode='x unified'
        )
    
    def create_advanced_candlestick_chart(self, 
                                        price_data: pd.DataFrame,
                                        indicators: Dict[str, pd.Series],
                                        title: str = "Price Chart",
                                        height: int = 800) -> go.Figure:
        """
        Create advanced candlestick chart with technical indicators.
        
        Args:
            price_data: OHLCV price data
            indicators: Technical indicators
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if price_data.empty:
            return self._create_empty_chart("No price data available")
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{title} - Price & Indicators', 'RSI', 'MACD'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Main candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name="Price",
                    increasing_line_color=config.COLORS['bullish'],
                    decreasing_line_color=config.COLORS['bearish']
                ),
                row=1, col=1
            )
            
            # Moving averages
            if 'SMA_50' in indicators and not indicators['SMA_50'].empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicators['SMA_50'].index,
                        y=indicators['SMA_50'],
                        name='SMA 50',
                        line=dict(color=self.color_palette[1], width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            if 'SMA_200' in indicators and not indicators['SMA_200'].empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicators['SMA_200'].index,
                        y=indicators['SMA_200'],
                        name='SMA 200',
                        line=dict(color=self.color_palette[2], width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(k in indicators for k in ['BB_Upper', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=indicators['BB_Upper'].index,
                        y=indicators['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicators['BB_Lower'].index,
                        y=indicators['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # RSI subplot
            if 'RSI' in indicators and not indicators['RSI'].empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicators['RSI'].index,
                        y=indicators['RSI'],
                        name='RSI',
                        line=dict(color=self.color_palette[3], width=2)
                    ),
                    row=2, col=1
                )
                
                # RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color=config.COLORS['bearish'], 
                             row=2, col=1, opacity=0.5)
                fig.add_hline(y=30, line_dash="dash", line_color=config.COLORS['bullish'], 
                             row=2, col=1, opacity=0.5)
                fig.add_hline(y=50, line_dash="dot", line_color=config.COLORS['text'], 
                             row=2, col=1, opacity=0.3)
            
            # MACD subplot
            if all(k in indicators for k in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig.add_trace(
                    go.Scatter(
                        x=indicators['MACD'].index,
                        y=indicators['MACD'],
                        name='MACD',
                        line=dict(color=self.color_palette[4], width=2)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicators['MACD_Signal'].index,
                        y=indicators['MACD_Signal'],
                        name='MACD Signal',
                        line=dict(color=self.color_palette[5], width=1)
                    ),
                    row=3, col=1
                )
                
                # MACD histogram
                colors = ['red' if val < 0 else 'green' for val in indicators['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(
                        x=indicators['MACD_Histogram'].index,
                        y=indicators['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=20)),
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            # Remove rangeslider
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_volatility_chart(self, 
                               returns: pd.Series,
                               title: str = "Rolling Volatility Analysis",
                               height: int = 400) -> go.Figure:
        """
        Create rolling volatility chart.
        
        Args:
            returns: Return series
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if returns.empty:
            return self._create_empty_chart("No return data available")
        
        try:
            # Calculate rolling volatilities
            vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100
            vol_60d = returns.rolling(60).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            
            # 20-day volatility
            fig.add_trace(
                go.Scatter(
                    x=vol_20d.index,
                    y=vol_20d,
                    name='20-Day Volatility',
                    line=dict(color=self.color_palette[0], width=2),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
                )
            )
            
            # 60-day volatility
            if not vol_60d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=vol_60d.index,
                        y=vol_60d,
                        name='60-Day Volatility',
                        line=dict(color=self.color_palette[1], width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
                    )
                )
            
            # Add average volatility line
            avg_vol = vol_20d.mean()
            fig.add_hline(
                y=avg_vol,
                line_dash="dot",
                line_color=config.COLORS['text'],
                opacity=0.5,
                annotation_text=f"Average: {avg_vol:.1f}%"
            )
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                yaxis_title="Volatility (%)",
                xaxis_title="Date",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volatility chart: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_drawdown_chart(self, 
                             prices: pd.Series,
                             title: str = "Drawdown Analysis",
                             height: int = 400) -> go.Figure:
        """
        Create drawdown chart.
        
        Args:
            prices: Price series
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if prices.empty:
            return self._create_empty_chart("No price data available")
        
        try:
            # Calculate drawdown
            returns = prices.pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            fig = go.Figure()
            
            # Drawdown area
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=config.COLORS['bearish'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(220, 53, 69, 0.3)',
                    hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                )
            )
            
            # Max drawdown line
            max_dd = drawdown.min()
            fig.add_hline(
                y=max_dd,
                line_dash="dash",
                line_color=config.COLORS['bearish'],
                annotation_text=f"Max DD: {max_dd:.2f}%"
            )
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                yaxis_title="Drawdown (%)",
                xaxis_title="Date",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_comparison_chart(self, 
                               price_data: Dict[str, pd.Series],
                               title: str = "Asset Comparison",
                               normalize: bool = True,
                               height: int = 500) -> go.Figure:
        """
        Create multi-asset comparison chart.
        
        Args:
            price_data: Dictionary of price series
            title: Chart title
            normalize: Whether to normalize prices
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if not price_data:
            return self._create_empty_chart("No comparison data available")
        
        try:
            fig = go.Figure()
            
            for i, (symbol, prices) in enumerate(price_data.items()):
                if prices.empty:
                    continue
                
                # Normalize if requested
                if normalize:
                    normalized_prices = prices / prices.iloc[0]
                    y_data = normalized_prices
                    y_title = "Normalized Price"
                else:
                    y_data = prices
                    y_title = "Price ($)"
                
                color = self.color_palette[i % len(self.color_palette)]
                
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=y_data,
                        name=symbol,
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>{y_title}: %{{y:.3f}}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                yaxis_title=y_title,
                xaxis_title="Date",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_correlation_heatmap(self, 
                                  correlation_matrix: pd.DataFrame,
                                  title: str = "Correlation Matrix",
                                  height: int = 500) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if correlation_matrix.empty:
            return self._create_empty_chart("No correlation data available")
        
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdYlBu_r',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_risk_return_scatter(self, 
                                  risk_return_data: pd.DataFrame,
                                  title: str = "Risk-Return Profile",
                                  height: int = 500) -> go.Figure:
        """
        Create risk-return scatter plot.
        
        Args:
            risk_return_data: DataFrame with Risk, Return, and Ticker columns
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if risk_return_data.empty:
            return self._create_empty_chart("No risk-return data available")
        
        try:
            fig = px.scatter(
                risk_return_data,
                x='Risk',
                y='Return',
                text='Ticker',
                title=title,
                labels={'Risk': 'Risk (Volatility %)', 'Return': 'Return (%)'},
                hover_data={'Risk': ':.2f', 'Return': ':.2f'}
            )
            
            fig.update_traces(
                textposition="top center",
                marker=dict(size=12, color=self.color_palette[0])
            )
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="dash", line_color=config.COLORS['text'], opacity=0.3)
            fig.add_vline(x=risk_return_data['Risk'].median(), line_dash="dash", 
                         line_color=config.COLORS['text'], opacity=0.3)
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk-return scatter: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def create_performance_bar_chart(self, 
                                    performance_data: pd.DataFrame,
                                    title: str = "Performance Comparison",
                                    height: int = 400) -> go.Figure:
        """
        Create performance comparison bar chart.
        
        Args:
            performance_data: DataFrame with performance metrics
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure
        """
        if performance_data.empty:
            return self._create_empty_chart("No performance data available")
        
        try:
            fig = go.Figure()
            
            # Assume first column is ticker/symbol, rest are metrics
            symbols = performance_data.iloc[:, 0]
            
            for i, col in enumerate(performance_data.columns[1:]):
                values = performance_data[col]
                color = self.color_palette[i % len(self.color_palette)]
                
                fig.add_trace(
                    go.Bar(
                        x=symbols,
                        y=values,
                        name=col,
                        marker_color=color,
                        hovertemplate=f'<b>%{{x}}</b><br>{col}: %{{y}}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                **self.default_layout,
                title=dict(text=title, font=dict(size=16)),
                height=height,
                xaxis_title="Assets",
                yaxis_title="Value",
                barmode='group',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance bar chart: {str(e)}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=config.COLORS['text'])
        )
        fig.update_layout(**self.default_layout, height=400)
        return fig

# Global instance
financial_visualizer = FinancialVisualizer()
