"""
Chart creation functions for financial data visualization.
Professional-grade interactive charts using Plotly.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from config import config

logger = logging.getLogger(__name__)

class PriceCharts:
    """Create price-related charts."""
    
    @staticmethod
    def create_candlestick_chart(
        ohlcv_data: pd.DataFrame,
        title: str = "Stock Price",
        height: int = config.CHART_HEIGHT,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create an interactive candlestick chart.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            title: Chart title
            height: Chart height in pixels
            show_volume: Whether to show volume subplot
        
        Returns:
            Plotly figure object
        """
        if ohlcv_data.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplots
        if show_volume and 'Volume' in ohlcv_data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, 'Volume')
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=ohlcv_data.index,
            open=ohlcv_data['Open'],
            high=ohlcv_data['High'],
            low=ohlcv_data['Low'],
            close=ohlcv_data['Close'],
            name="Price",
            increasing_line_color=config.COLORS['bullish'],
            decreasing_line_color=config.COLORS['bearish']
        )
        
        if show_volume and 'Volume' in ohlcv_data.columns:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Add volume chart
            colors = [config.COLORS['bullish'] if close >= open else config.COLORS['bearish'] 
                     for close, open in zip(ohlcv_data['Close'], ohlcv_data['Open'])]
            
            volume_bar = go.Bar(
                x=ohlcv_data.index,
                y=ohlcv_data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            )
            fig.add_trace(volume_bar, row=2, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background']
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=config.COLORS['grid'],
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor=config.COLORS['grid'],
            showgrid=True
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(
        price_data: Union[pd.Series, pd.DataFrame],
        title: str = "Price Chart",
        height: int = config.CHART_HEIGHT,
        color_column: Optional[str] = None
    ) -> go.Figure:
        """
        Create a line chart for price data.
        
        Args:
            price_data: Price data (Series or DataFrame)
            title: Chart title
            height: Chart height
            color_column: Column to use for color mapping (if DataFrame)
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        if isinstance(price_data, pd.Series):
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode='lines',
                name='Price',
                line=dict(color=config.COLORS['neutral'], width=2)
            ))
        elif isinstance(price_data, pd.DataFrame):
            for column in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background'],
            xaxis=dict(gridcolor=config.COLORS['grid']),
            yaxis=dict(gridcolor=config.COLORS['grid'])
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(
        price_data_dict: Dict[str, pd.Series],
        title: str = "Price Comparison",
        normalize: bool = True,
        height: int = config.CHART_HEIGHT
    ) -> go.Figure:
        """
        Create a comparison chart for multiple assets.
        
        Args:
            price_data_dict: Dictionary of symbol -> price series
            title: Chart title
            normalize: Whether to normalize prices to 100
            height: Chart height
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for symbol, prices in price_data_dict.items():
            if prices.empty:
                continue
            
            y_data = prices
            if normalize:
                # Normalize to starting value of 100
                y_data = (prices / prices.iloc[0]) * 100
            
            fig.add_trace(go.Scatter(
                x=prices.index,
                y=y_data,
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        y_title = "Normalized Price (Base=100)" if normalize else "Price"
        
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background'],
            xaxis=dict(title="Date", gridcolor=config.COLORS['grid']),
            yaxis=dict(title=y_title, gridcolor=config.COLORS['grid']),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_multi_line_chart(
        data: pd.DataFrame,
        title: str = "Multi-Asset Comparison",
        y_title: str = "Price",
        height: int = config.CHART_HEIGHT
    ) -> go.Figure:
        """
        Create a multi-line chart for comparing multiple assets.
        
        Args:
            data: DataFrame with time series data (columns are different assets)
            title: Chart title
            y_title: Y-axis title
            height: Chart height in pixels
        
        Returns:
            Plotly figure object
        """
        if data.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Color palette for different lines
        colors = [
            config.COLORS['primary'],
            config.COLORS['secondary'],
            config.COLORS['success'],
            config.COLORS['warning'],
            config.COLORS['info'],
            config.COLORS['danger']
        ]
        
        # Add a line for each column
        for i, column in enumerate(data.columns):
            if data[column].notna().any():  # Only plot if there's data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2
                    ),
                    hovertemplate=f"<b>{column}</b><br>" +
                                "Date: %{x}<br>" +
                                "Value: %{y:.2f}<br>" +
                                "<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=config.COLORS['text'])
            ),
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background'],
            height=height,
            xaxis=dict(
                title="Date",
                gridcolor=config.COLORS['grid'],
                showgrid=True
            ),
            yaxis=dict(
                title=y_title,
                gridcolor=config.COLORS['grid'],
                showgrid=True
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

class TechnicalCharts:
    """Create technical analysis charts."""
    
    @staticmethod
    def create_technical_indicators_chart(
        ohlcv_data: pd.DataFrame,
        indicators: Dict[str, Union[pd.Series, Dict]],
        title: str = "Technical Analysis",
        height: int = 800
    ) -> go.Figure:
        """
        Create comprehensive technical analysis chart.
        
        Args:
            ohlcv_data: OHLCV data
            indicators: Dictionary of technical indicators
            title: Chart title
            height: Chart height
        
        Returns:
            Plotly figure with multiple subplots
        """
        # Create subplots
        subplot_titles = ["Price & Moving Averages", "Volume", "RSI", "MACD"]
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.175, 0.175],
            subplot_titles=subplot_titles
        )
        
        # Main price chart (candlestick)
        if not ohlcv_data.empty:
            fig.add_trace(
                go.Candlestick(
                    x=ohlcv_data.index,
                    open=ohlcv_data['Open'],
                    high=ohlcv_data['High'],
                    low=ohlcv_data['Low'],
                    close=ohlcv_data['Close'],
                    name="Price",
                    increasing_line_color=config.COLORS['bullish'],
                    decreasing_line_color=config.COLORS['bearish']
                ),
                row=1, col=1
            )
        
        # Add moving averages
        ma_indicators = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']
        colors = ['orange', 'blue', 'red', 'green', 'purple']
        
        for i, (ma_name, color) in enumerate(zip(ma_indicators, colors)):
            if ma_name in indicators and not indicators[ma_name].empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicators[ma_name].index,
                        y=indicators[ma_name],
                        mode='lines',
                        name=ma_name.upper(),
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if 'bollinger_bands' in indicators:
            bb_data = indicators['bollinger_bands']
            for band_name, color, fill in [('upper', 'rgba(173,204,255,0.2)', None), 
                                          ('middle', 'blue', 'tonexty'), 
                                          ('lower', 'rgba(173,204,255,0.2)', 'tonexty')]:
                if band_name in bb_data:
                    fig.add_trace(
                        go.Scatter(
                            x=bb_data[band_name].index,
                            y=bb_data[band_name],
                            mode='lines',
                            name=f'BB {band_name.title()}',
                            line=dict(color=color if band_name == 'middle' else 'lightblue', width=1),
                            fill=fill,
                            fillcolor='rgba(173,204,255,0.1)' if fill else None
                        ),
                        row=1, col=1
                    )
        
        # Volume chart
        if 'Volume' in ohlcv_data.columns:
            colors = [config.COLORS['bullish'] if close >= open else config.COLORS['bearish'] 
                     for close, open in zip(ohlcv_data['Close'], ohlcv_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=ohlcv_data.index,
                    y=ohlcv_data['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI chart
        if 'rsi' in indicators and not indicators['rsi'].empty:
            fig.add_trace(
                go.Scatter(
                    x=indicators['rsi'].index,
                    y=indicators['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD chart
        if 'macd' in indicators:
            macd_data = indicators['macd']
            
            # MACD line
            if 'macd' in macd_data:
                fig.add_trace(
                    go.Scatter(
                        x=macd_data['macd'].index,
                        y=macd_data['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=4, col=1
                )
            
            # Signal line
            if 'signal' in macd_data:
                fig.add_trace(
                    go.Scatter(
                        x=macd_data['signal'].index,
                        y=macd_data['signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=4, col=1
                )
            
            # Histogram
            if 'histogram' in macd_data:
                colors = [config.COLORS['bullish'] if val >= 0 else config.COLORS['bearish'] 
                         for val in macd_data['histogram']]
                
                fig.add_trace(
                    go.Bar(
                        x=macd_data['histogram'].index,
                        y=macd_data['histogram'],
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background'],
            xaxis_rangeslider_visible=False
        )
        
        # Update all axes
        for i in range(1, 5):
            fig.update_xaxes(gridcolor=config.COLORS['grid'], row=i, col=1)
            fig.update_yaxes(gridcolor=config.COLORS['grid'], row=i, col=1)
        
        return fig
    
    @staticmethod
    def create_support_resistance_chart(
        ohlcv_data: pd.DataFrame,
        support_levels: List[float],
        resistance_levels: List[float],
        title: str = "Support & Resistance Levels"
    ) -> go.Figure:
        """
        Create chart showing support and resistance levels.
        
        Args:
            ohlcv_data: OHLCV data
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = PriceCharts.create_candlestick_chart(ohlcv_data, title, show_volume=False)
        
        # Add support levels
        for level in support_levels:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: ${level:.2f}",
                annotation_position="left"
            )
        
        # Add resistance levels
        for level in resistance_levels:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: ${level:.2f}",
                annotation_position="left"
            )
        
        return fig

class PortfolioCharts:
    """Create portfolio-related charts."""
    
    @staticmethod
    def create_portfolio_composition_pie(
        weights: Dict[str, float],
        title: str = "Portfolio Composition"
    ) -> go.Figure:
        """
        Create pie chart showing portfolio composition.
        
        Args:
            weights: Dictionary of asset weights
            title: Chart title
        
        Returns:
            Plotly figure
        """
        labels = list(weights.keys())
        values = list(weights.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background']
        )
        
        return fig
    
    @staticmethod
    def create_efficient_frontier(
        risk_return_points: List[Tuple[float, float]],
        optimal_portfolio: Optional[Tuple[float, float]] = None,
        title: str = "Efficient Frontier"
    ) -> go.Figure:
        """
        Create efficient frontier chart.
        
        Args:
            risk_return_points: List of (risk, return) tuples
            optimal_portfolio: Optimal portfolio point
            title: Chart title
        
        Returns:
            Plotly figure
        """
        if not risk_return_points:
            return go.Figure()
        
        risks, returns = zip(*risk_return_points)
        
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color=config.COLORS['neutral'], width=3),
            marker=dict(size=6)
        ))
        
        # Add optimal portfolio if provided
        if optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio[0]],
                y=[optimal_portfolio[1]],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color=config.COLORS['bullish'],
                    symbol='star'
                )
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background']
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix"
    ) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background']
        )
        
        return fig
    
    @staticmethod
    def create_drawdown_chart(
        returns: pd.Series,
        title: str = "Portfolio Drawdown"
    ) -> go.Figure:
        """
        Create underwater (drawdown) chart.
        
        Args:
            returns: Return series
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns and drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        # Add drawdown area chart
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,  # Convert to percentage
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color=config.COLORS['bearish']),
            fillcolor='rgba(255, 68, 68, 0.3)'  # Red with transparency
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_color="white", line_width=1)
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            font=dict(color=config.COLORS['text']),
            paper_bgcolor=config.COLORS['background'],
            plot_bgcolor=config.COLORS['background']
        )
        
        return fig

# Create instances for easy importing
price_charts = PriceCharts()
technical_charts = TechnicalCharts()
portfolio_charts = PortfolioCharts()
