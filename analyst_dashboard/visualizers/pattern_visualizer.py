"""
Pattern Visualization Module - Priority 3 Implementation  
Advanced visualizations for candlestick patterns, chart patterns, and volatility forecasts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PatternVisualizer:
    """Priority 3: Advanced pattern recognition visualizations"""
    
    def __init__(self):
        self.pattern_colors = {
            'bullish': '#26a69a',      # Teal
            'bearish': '#ef5350',       # Red
            'neutral': '#66bb6a',       # Light Green
            'reversal': '#ff7043',      # Deep Orange
            'continuation': '#42a5f5',  # Blue
            'breakout': '#ab47bc'       # Purple
        }
        
        self.candlestick_colors = {
            'bullish_patterns': '#00e676',  # Bright Green
            'bearish_patterns': '#ff1744',  # Bright Red
            'neutral_patterns': '#ffeb3b'   # Yellow
        }
    
    def create_candlestick_pattern_chart(self, price_data: pd.DataFrame, 
                                       pattern_analysis: Dict[str, Any]) -> go.Figure:
        """Create candlestick chart with pattern annotations"""
        try:
            fig = go.Figure()
            
            # Base candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                )
            )
            
            # Add pattern annotations
            if 'patterns' in pattern_analysis:
                self._add_candlestick_pattern_annotations(fig, price_data, pattern_analysis['patterns'])
            
            # Add pattern summary box
            if 'summary' in pattern_analysis:
                self._add_pattern_summary_box(fig, pattern_analysis['summary'])
            
            fig.update_layout(
                title="Candlestick Patterns Detection",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                showlegend=True,
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick pattern chart: {str(e)}")
            return self._create_error_chart("Candlestick Pattern Chart Error")
    
    def create_chart_pattern_visualization(self, price_data: pd.DataFrame,
                                         pattern_analysis: Dict[str, Any]) -> go.Figure:
        """Create chart pattern visualization with trend lines"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=('Price with Chart Patterns', 'Pattern Confidence Levels'),
                vertical_spacing=0.1,
                row_heights=[0.8, 0.2]
            )
            
            # Price chart
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
            
            # Add chart pattern overlays
            if 'patterns' in pattern_analysis:
                self._add_chart_pattern_overlays(fig, price_data, pattern_analysis['patterns'])
            
            # Add support/resistance levels
            if 'key_levels' in pattern_analysis:
                self._add_support_resistance_lines(fig, pattern_analysis['key_levels'])
            
            # Pattern confidence timeline
            if 'summary' in pattern_analysis:
                self._add_pattern_confidence_timeline(fig, price_data, pattern_analysis)
            
            fig.update_layout(
                title="Chart Pattern Recognition Analysis",
                height=700,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart pattern visualization: {str(e)}")
            return self._create_error_chart("Chart Pattern Visualization Error")
    
    def create_volatility_forecast_chart(self, price_data: pd.DataFrame,
                                       forecast_analysis: Dict[str, Any]) -> go.Figure:
        """Create comprehensive volatility forecast visualization"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    'Price Movement',
                    'Historical & Forecasted Volatility',
                    'Volatility Regime Analysis'
                ),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.4, 0.2]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            # Historical volatility
            returns = price_data['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name='Historical Volatility (20d)',
                    line=dict(color='#A23B72', width=2)
                ),
                row=2, col=1
            )
            
            # Add volatility forecast
            if 'ensemble_forecast' in forecast_analysis:
                self._add_volatility_forecast(fig, price_data, forecast_analysis['ensemble_forecast'])
            
            # Add regime analysis
            if 'regime_analysis' in forecast_analysis:
                self._add_volatility_regime_visualization(fig, rolling_vol, forecast_analysis['regime_analysis'])
            
            # Add current metrics annotations
            if 'current_metrics' in forecast_analysis:
                self._add_volatility_metrics_annotations(fig, forecast_analysis['current_metrics'])
            
            fig.update_layout(
                title="Volatility Forecasting Analysis",
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            fig.update_yaxes(title_text="Regime", row=3, col=1, showticklabels=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volatility forecast chart: {str(e)}")
            return self._create_error_chart("Volatility Forecast Chart Error")
    
    def create_pattern_summary_dashboard(self, candlestick_analysis: Dict[str, Any],
                                       chart_pattern_analysis: Dict[str, Any],
                                       volatility_analysis: Dict[str, Any]) -> go.Figure:
        """Create comprehensive pattern analysis dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Candlestick Patterns', 'Chart Patterns', 'Volatility Regime',
                    'Pattern Signals', 'Confidence Levels', 'Forecast Trend'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "pie"}, {"type": "indicator"}],
                    [{"type": "bar"}, {"type": "bar"}, {"type": "indicator"}]
                ]
            )
            
            # Candlestick pattern distribution
            if 'summary' in candlestick_analysis:
                cs_summary = candlestick_analysis['summary']
                fig.add_trace(
                    go.Pie(
                        labels=['Bullish', 'Bearish', 'Neutral'],
                        values=[
                            cs_summary.get('bullish_patterns', 0),
                            cs_summary.get('bearish_patterns', 0),
                            cs_summary.get('neutral_patterns', 0)
                        ],
                        name="Candlestick Patterns",
                        marker_colors=[self.pattern_colors['bullish'], 
                                     self.pattern_colors['bearish'],
                                     self.pattern_colors['neutral']]
                    ),
                    row=1, col=1
                )
            
            # Chart pattern distribution
            if 'summary' in chart_pattern_analysis:
                cp_summary = chart_pattern_analysis['summary']
                fig.add_trace(
                    go.Pie(
                        labels=['Reversal', 'Continuation', 'Breakout'],
                        values=[
                            cp_summary.get('reversal_patterns', 0),
                            cp_summary.get('continuation_patterns', 0),
                            cp_summary.get('breakout_patterns', 0)
                        ],
                        name="Chart Patterns",
                        marker_colors=[self.pattern_colors['reversal'],
                                     self.pattern_colors['continuation'],
                                     self.pattern_colors['breakout']]
                    ),
                    row=1, col=2
                )
            
            # Volatility regime indicator
            if 'regime_analysis' in volatility_analysis:
                current_regime = volatility_analysis['regime_analysis'].get('current_regime', 'medium')
                regime_colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'extreme': 'red'}
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=self._regime_to_numeric(current_regime),
                        title={'text': f"Volatility Regime<br>{current_regime.title()}"},
                        gauge={
                            'axis': {'range': [0, 4]},
                            'bar': {'color': regime_colors.get(current_regime, 'yellow')},
                            'steps': [
                                {'range': [0, 1], 'color': 'lightgreen'},
                                {'range': [1, 2], 'color': 'yellow'},
                                {'range': [2, 3], 'color': 'orange'},
                                {'range': [3, 4], 'color': 'red'}
                            ]
                        }
                    ),
                    row=1, col=3
                )
            
            # Pattern signal strength
            signal_data = self._aggregate_pattern_signals(candlestick_analysis, chart_pattern_analysis)
            if signal_data:
                fig.add_trace(
                    go.Bar(
                        x=list(signal_data.keys()),
                        y=list(signal_data.values()),
                        name="Signal Strength",
                        marker_color=[self.pattern_colors['bullish'] if v > 0 else self.pattern_colors['bearish'] 
                                    for v in signal_data.values()]
                    ),
                    row=2, col=1
                )
            
            # Confidence levels
            confidence_data = self._aggregate_confidence_levels(candlestick_analysis, chart_pattern_analysis)
            if confidence_data:
                fig.add_trace(
                    go.Bar(
                        x=list(confidence_data.keys()),
                        y=list(confidence_data.values()),
                        name="Confidence Levels",
                        marker_color='#42a5f5'
                    ),
                    row=2, col=2
                )
            
            # Volatility forecast trend
            if 'ensemble_forecast' in volatility_analysis:
                forecast = volatility_analysis['ensemble_forecast']
                trend_value = self._trend_to_numeric(forecast.volatility_trend)
                trend_colors = {'increasing': 'red', 'stable': 'yellow', 'decreasing': 'green'}
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=trend_value,
                        title={'text': f"Vol Forecast<br>{forecast.volatility_trend.title()}"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': trend_colors.get(forecast.volatility_trend, 'yellow')},
                            'steps': [
                                {'range': [-1, -0.3], 'color': 'lightgreen'},
                                {'range': [-0.3, 0.3], 'color': 'lightyellow'},
                                {'range': [0.3, 1], 'color': 'lightcoral'}
                            ]
                        }
                    ),
                    row=2, col=3
                )
            
            fig.update_layout(
                title="Pattern Recognition & Forecasting Dashboard",
                height=700,
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern summary dashboard: {str(e)}")
            return self._create_error_chart("Pattern Summary Dashboard Error")
    
    def _add_candlestick_pattern_annotations(self, fig: go.Figure, price_data: pd.DataFrame,
                                           patterns: Dict[str, List]) -> None:
        """Add candlestick pattern annotations to chart"""
        try:
            for pattern_name, pattern_list in patterns.items():
                if not pattern_list:
                    continue
                
                for pattern in pattern_list[-10:]:  # Show last 10 patterns
                    # Find the approximate date for the pattern
                    pattern_date = price_data.index[-len(pattern_list)]  # Simplified approximation
                    pattern_price = price_data['High'].loc[pattern_date]
                    
                    # Determine color based on signal type
                    color = self.candlestick_colors.get(f'{pattern.signal_type}_patterns', '#ffeb3b')
                    
                    fig.add_annotation(
                        x=pattern_date,
                        y=pattern_price,
                        text=f"{pattern_name.replace('_', ' ').title()}<br>({pattern.reliability:.0f}%)",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        bgcolor=color,
                        bordercolor="white",
                        borderwidth=2,
                        font=dict(size=10, color="white")
                    )
                    
        except Exception as e:
            logger.error(f"Error adding candlestick pattern annotations: {str(e)}")
    
    def _add_chart_pattern_overlays(self, fig: go.Figure, price_data: pd.DataFrame,
                                  patterns: Dict[str, List]) -> None:
        """Add chart pattern overlays (trend lines, shapes)"""
        try:
            for pattern_name, pattern_list in patterns.items():
                if not pattern_list:
                    continue
                
                for pattern in pattern_list:
                    color = self.pattern_colors.get(pattern.signal_type, '#42a5f5')
                    
                    # Add pattern rectangle
                    fig.add_shape(
                        type="rect",
                        x0=pattern.start_date,
                        x1=pattern.end_date,
                        y0=min(price_data['Low']),
                        y1=max(price_data['High']),
                        fillcolor=color,
                        opacity=0.1,
                        line=dict(color=color, width=2),
                        row=1, col=1
                    )
                    
                    # Add pattern label
                    fig.add_annotation(
                        x=pattern.start_date,
                        y=max(price_data['High']) * 0.95,
                        text=f"{pattern_name.replace('_', ' ').title()}<br>Conf: {pattern.confidence:.0f}%",
                        showarrow=False,
                        bgcolor=color,
                        bordercolor="white",
                        font=dict(size=9, color="white"),
                        row=1, col=1
                    )
                    
        except Exception as e:
            logger.error(f"Error adding chart pattern overlays: {str(e)}")
    
    def _add_support_resistance_lines(self, fig: go.Figure, levels: List[Dict[str, Any]]) -> None:
        """Add support and resistance lines"""
        try:
            for level in levels[:5]:  # Show top 5 levels
                level_type = level['type']
                price = level['price']
                confidence = level['confidence']
                
                color = '#26a69a' if level_type == 'support' else '#ef5350'
                line_style = 'solid' if confidence > 70 else 'dash'
                
                fig.add_hline(
                    y=price,
                    line_dash=line_style,
                    line_color=color,
                    opacity=0.7,
                    annotation_text=f"{level_type.title()}: ${price:.2f} ({confidence:.0f}%)",
                    annotation_position="top right",
                    row=1, col=1
                )
                
        except Exception as e:
            logger.error(f"Error adding support/resistance lines: {str(e)}")
    
    def _add_volatility_forecast(self, fig: go.Figure, price_data: pd.DataFrame,
                               forecast: Any) -> None:
        """Add volatility forecast to chart"""
        try:
            # Create future dates for forecast
            last_date = price_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast.forecast_horizon,
                freq='D'
            )
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast.forecasted_volatility,
                    mode='lines',
                    name='Volatility Forecast',
                    line=dict(color='#ff7043', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            # Add current volatility level line
            fig.add_hline(
                y=forecast.current_volatility * 100,
                line_dash="dot",
                line_color="#666666",
                annotation_text=f"Current: {forecast.current_volatility*100:.1f}%",
                row=2, col=1
            )
            
        except Exception as e:
            logger.error(f"Error adding volatility forecast: {str(e)}")
    
    def _add_volatility_regime_visualization(self, fig: go.Figure, rolling_vol: pd.Series,
                                           regime_analysis: Dict[str, Any]) -> None:
        """Add volatility regime visualization"""
        try:
            current_regime = regime_analysis.get('current_regime', 'medium')
            regime_colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'extreme': 'red'}
            
            # Create regime timeline
            regime_timeline = []
            for vol in rolling_vol.dropna():
                if vol <= 15:
                    regime_timeline.append(1)  # Low
                elif vol <= 25:
                    regime_timeline.append(2)  # Medium
                elif vol <= 40:
                    regime_timeline.append(3)  # High
                else:
                    regime_timeline.append(4)  # Extreme
            
            if regime_timeline:
                fig.add_trace(
                    go.Bar(
                        x=rolling_vol.dropna().index,
                        y=[1] * len(regime_timeline),
                        marker_color=[regime_colors['low'] if r == 1 else 
                                    regime_colors['medium'] if r == 2 else
                                    regime_colors['high'] if r == 3 else 
                                    regime_colors['extreme'] for r in regime_timeline],
                        name='Volatility Regime',
                        opacity=0.7
                    ),
                    row=3, col=1
                )
                
        except Exception as e:
            logger.error(f"Error adding volatility regime visualization: {str(e)}")
    
    def _add_volatility_metrics_annotations(self, fig: go.Figure, metrics: Dict[str, float]) -> None:
        """Add volatility metrics annotations"""
        try:
            vol_percentile = metrics.get('volatility_percentile', 50)
            current_vol = metrics.get('current_volatility', 0)
            
            annotation_text = f"Current Vol: {current_vol*100:.1f}%<br>Percentile: {vol_percentile:.0f}%"
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=annotation_text,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                font=dict(size=12)
            )
            
        except Exception as e:
            logger.error(f"Error adding volatility metrics annotations: {str(e)}")
    
    def _add_pattern_summary_box(self, fig: go.Figure, summary: Dict[str, Any]) -> None:
        """Add pattern summary information box"""
        try:
            total_patterns = summary.get('total_patterns', 0)
            signal_strength = summary.get('signal_strength', 'neutral')
            high_reliability = summary.get('high_reliability_patterns', 0)
            
            summary_text = (f"Total Patterns: {total_patterns}<br>"
                          f"Signal: {signal_strength.title()}<br>"
                          f"High Reliability: {high_reliability}")
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=summary_text,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                font=dict(size=11)
            )
            
        except Exception as e:
            logger.error(f"Error adding pattern summary box: {str(e)}")
    
    def _add_pattern_confidence_timeline(self, fig: go.Figure, price_data: pd.DataFrame,
                                       pattern_analysis: Dict[str, Any]) -> None:
        """Add pattern confidence timeline"""
        try:
            # Create simplified confidence timeline
            confidence_data = []
            dates = price_data.index[-30:]  # Last 30 days
            
            for date in dates:
                # Simplified confidence calculation (in real implementation, this would be more sophisticated)
                base_confidence = 50
                confidence_data.append(base_confidence)
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=confidence_data,
                    mode='lines+markers',
                    name='Pattern Confidence',
                    line=dict(color='#42a5f5', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(66, 165, 245, 0.2)'
                ),
                row=2, col=1
            )
            
        except Exception as e:
            logger.error(f"Error adding pattern confidence timeline: {str(e)}")
    
    def _regime_to_numeric(self, regime: str) -> float:
        """Convert regime to numeric value for gauge"""
        regime_values = {'low': 0.5, 'medium': 1.5, 'high': 2.5, 'extreme': 3.5}
        return regime_values.get(regime, 1.5)
    
    def _trend_to_numeric(self, trend: str) -> float:
        """Convert trend to numeric value for gauge"""
        trend_values = {'decreasing': -0.7, 'stable': 0, 'increasing': 0.7}
        return trend_values.get(trend, 0)
    
    def _aggregate_pattern_signals(self, candlestick_analysis: Dict[str, Any],
                                 chart_pattern_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate pattern signals for visualization"""
        try:
            signals = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
            
            # Candlestick patterns
            if 'summary' in candlestick_analysis:
                cs_summary = candlestick_analysis['summary']
                signals['Bullish'] += cs_summary.get('bullish_patterns', 0)
                signals['Bearish'] += cs_summary.get('bearish_patterns', 0)
                signals['Neutral'] += cs_summary.get('neutral_patterns', 0)
            
            # Chart patterns
            if 'summary' in chart_pattern_analysis:
                cp_summary = chart_pattern_analysis['summary']
                # Simplified mapping
                signals['Bullish'] += cp_summary.get('continuation_patterns', 0) * 0.5
                signals['Bearish'] += cp_summary.get('reversal_patterns', 0) * 0.5
                signals['Neutral'] += cp_summary.get('breakout_patterns', 0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error aggregating pattern signals: {str(e)}")
            return {}
    
    def _aggregate_confidence_levels(self, candlestick_analysis: Dict[str, Any],
                                   chart_pattern_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate confidence levels for visualization"""
        try:
            confidence = {'High (>80%)': 0, 'Medium (60-80%)': 0, 'Low (<60%)': 0}
            
            # Candlestick patterns
            if 'summary' in candlestick_analysis:
                high_rel = candlestick_analysis['summary'].get('high_reliability_patterns', 0)
                confidence['High (>80%)'] += high_rel
                confidence['Medium (60-80%)'] += max(0, candlestick_analysis['summary'].get('total_patterns', 0) - high_rel)
            
            # Chart patterns  
            if 'summary' in chart_pattern_analysis:
                high_conf = chart_pattern_analysis['summary'].get('high_confidence_patterns', 0)
                confidence['High (>80%)'] += high_conf
                confidence['Medium (60-80%)'] += max(0, chart_pattern_analysis['summary'].get('total_patterns', 0) - high_conf)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error aggregating confidence levels: {str(e)}")
            return {}
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart placeholder"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: {error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            title=error_message,
            height=400,
            template='plotly_white'
        )
        return fig
