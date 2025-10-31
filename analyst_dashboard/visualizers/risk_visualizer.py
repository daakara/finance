"""
Risk Visualization Module - Priority 2 Implementation
Advanced risk analysis visualizations for sophisticated risk metrics and regime detection
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RiskVisualizer:
    """Priority 2: Advanced risk analysis visualizations"""
    
    def __init__(self):
        self.risk_colors = {
            'low': '#00cc44',      # Green
            'medium': '#ffaa00',    # Orange  
            'high': '#ff4444',      # Red
            'extreme': '#8b0000'    # Dark Red
        }
        self.regime_colors = {
            'Bull_Market_Low_Vol': '#228B22',     # Forest Green
            'Bull_Market_High_Vol': '#32CD32',    # Lime Green
            'Bear_Market_Low_Vol': '#DC143C',     # Crimson
            'Bear_Market_High_Vol': '#8B0000'     # Dark Red
        }
    
    def create_var_analysis_chart(self, price_data: pd.DataFrame, 
                                risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create Value at Risk analysis visualization"""
        try:
            # Calculate rolling VaR
            returns = price_data['Close'].pct_change().dropna()
            rolling_var_95 = returns.rolling(30).quantile(0.05) * 100
            rolling_var_99 = returns.rolling(30).quantile(0.01) * 100
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=('Price Movement', 'Rolling Value at Risk'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    name='Price',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            # VaR charts
            fig.add_trace(
                go.Scatter(
                    x=rolling_var_95.index,
                    y=rolling_var_95,
                    name='VaR 95%',
                    line=dict(color='#A23B72', width=2),
                    fill='tonexty' if len(fig.data) > 0 else None,
                    fillcolor='rgba(162, 59, 114, 0.2)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_var_99.index,
                    y=rolling_var_99,
                    name='VaR 99%',
                    line=dict(color='#F18F01', width=2)
                ),
                row=2, col=1
            )
            
            # Add current VaR levels as horizontal lines
            if 'advanced_metrics' in risk_metrics:
                current_var_95 = risk_metrics['advanced_metrics'].get('VaR_95', 0)
                current_var_99 = risk_metrics['advanced_metrics'].get('VaR_99', 0)
                
                fig.add_hline(
                    y=current_var_95,
                    line_dash="dash",
                    line_color="#A23B72",
                    annotation_text=f"Current VaR 95%: {current_var_95:.2f}%",
                    row=2, col=1
                )
                
                fig.add_hline(
                    y=current_var_99,
                    line_dash="dash", 
                    line_color="#F18F01",
                    annotation_text=f"Current VaR 99%: {current_var_99:.2f}%",
                    row=2, col=1
                )
            
            fig.update_layout(
                title="Value at Risk Analysis",
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="VaR (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating VaR analysis chart: {str(e)}")
            return self._create_error_chart("VaR Analysis Error")
    
    def create_drawdown_analysis_chart(self, price_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive drawdown analysis visualization"""
        try:
            # Calculate drawdowns
            returns = price_data['Close'].pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max * 100
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=('Cumulative Returns vs Peak', 'Drawdown Analysis'),
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4]
            )
            
            # Cumulative returns and rolling max
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative * 100,
                    name='Cumulative Returns',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_max.index,
                    y=rolling_max * 100,
                    name='Peak Value',
                    line=dict(color='#A23B72', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Drawdown chart with color coding
            drawdown_colors = []
            for dd in drawdowns:
                if dd <= -20:
                    drawdown_colors.append(self.risk_colors['extreme'])
                elif dd <= -10:
                    drawdown_colors.append(self.risk_colors['high'])
                elif dd <= -5:
                    drawdown_colors.append(self.risk_colors['medium'])
                else:
                    drawdown_colors.append(self.risk_colors['low'])
            
            fig.add_trace(
                go.Bar(
                    x=drawdowns.index,
                    y=drawdowns,
                    name='Drawdown',
                    marker_color=drawdown_colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(
                title="Drawdown Analysis",
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown analysis chart: {str(e)}")
            return self._create_error_chart("Drawdown Analysis Error")
    
    def create_regime_detection_chart(self, price_data: pd.DataFrame, 
                                    regime_analysis: Dict[str, Any]) -> go.Figure:
        """Create market regime detection visualization"""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Calculate rolling metrics for regime detection
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
            rolling_returns = returns.rolling(30).mean() * 252 * 100
            
            # Create regime classification
            vol_threshold = rolling_vol.median()
            regime_colors_mapped = []
            regime_labels = []
            
            for i in range(len(rolling_vol)):
                if pd.isna(rolling_vol.iloc[i]) or pd.isna(rolling_returns.iloc[i]):
                    regime_colors_mapped.append('#cccccc')
                    regime_labels.append('Unknown')
                    continue
                    
                vol = rolling_vol.iloc[i]
                ret = rolling_returns.iloc[i]
                
                if ret > 0 and vol < vol_threshold:
                    regime = "Bull_Market_Low_Vol"
                elif ret > 0 and vol >= vol_threshold:
                    regime = "Bull_Market_High_Vol"
                elif ret <= 0 and vol < vol_threshold:
                    regime = "Bear_Market_Low_Vol"
                else:
                    regime = "Bear_Market_High_Vol"
                
                regime_colors_mapped.append(self.regime_colors[regime])
                regime_labels.append(regime.replace('_', ' '))
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    'Price Movement', 
                    'Rolling Volatility (30-day)', 
                    'Market Regime Classification'
                ),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    name='Price',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            # Rolling volatility
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    name='30-Day Volatility',
                    line=dict(color='#F18F01', width=2)
                ),
                row=2, col=1
            )
            
            # Add volatility threshold line
            fig.add_hline(
                y=vol_threshold,
                line_dash="dash",
                line_color="#A23B72",
                annotation_text=f"Vol Threshold: {vol_threshold:.1f}%",
                row=2, col=1
            )
            
            # Regime visualization
            fig.add_trace(
                go.Bar(
                    x=rolling_returns.index,
                    y=[1] * len(rolling_returns),  # Fixed height bars
                    name='Market Regime',
                    marker_color=regime_colors_mapped,
                    opacity=0.8,
                    hovertemplate='<b>%{x}</b><br>Regime: %{text}<extra></extra>',
                    text=regime_labels
                ),
                row=3, col=1
            )
            
            # Add regime legend
            unique_regimes = list(set(regime_labels))
            for regime in unique_regimes:
                if regime != 'Unknown':
                    regime_key = regime.replace(' ', '_')
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=self.regime_colors.get(regime_key, '#cccccc')
                            ),
                            name=regime,
                            showlegend=True
                        )
                    )
            
            fig.update_layout(
                title="Market Regime Detection Analysis",
                height=700,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            fig.update_yaxes(title_text="Regime", row=3, col=1, showticklabels=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regime detection chart: {str(e)}")
            return self._create_error_chart("Regime Detection Error")
    
    def create_tail_risk_analysis_chart(self, price_data: pd.DataFrame, 
                                      tail_risk: Dict[str, Any]) -> go.Figure:
        """Create tail risk analysis visualization"""
        try:
            returns = price_data['Close'].pct_change().dropna() * 100
            
            # Create histogram of returns with tail highlighting
            fig = go.Figure()
            
            # Main histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color='rgba(46, 134, 171, 0.7)',
                    opacity=0.7
                )
            )
            
            # Highlight extreme tails (beyond 2.5 std dev)
            std_dev = returns.std()
            mean_ret = returns.mean()
            
            # Left tail (extreme negative)
            left_tail_threshold = mean_ret - 2.5 * std_dev
            extreme_negative = returns[returns < left_tail_threshold]
            
            if len(extreme_negative) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=extreme_negative,
                        nbinsx=10,
                        name=f'Extreme Negative ({len(extreme_negative)} events)',
                        marker_color=self.risk_colors['extreme'],
                        opacity=0.8
                    )
                )
            
            # Right tail (extreme positive)
            right_tail_threshold = mean_ret + 2.5 * std_dev
            extreme_positive = returns[returns > right_tail_threshold]
            
            if len(extreme_positive) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=extreme_positive,
                        nbinsx=10,
                        name=f'Extreme Positive ({len(extreme_positive)} events)',
                        marker_color=self.risk_colors['low'],
                        opacity=0.8
                    )
                )
            
            # Add VaR lines
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            fig.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="#A23B72",
                annotation_text=f"VaR 95%: {var_95:.2f}%"
            )
            
            fig.add_vline(
                x=var_99,
                line_dash="dash",
                line_color="#F18F01", 
                annotation_text=f"VaR 99%: {var_99:.2f}%"
            )
            
            # Add normal distribution overlay for comparison
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = (len(returns) * (returns.max() - returns.min()) / 50) * \
                    (1/np.sqrt(2*np.pi*std_dev**2)) * \
                    np.exp(-0.5*((x_norm - mean_ret)/std_dev)**2)
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2, dash='dot'),
                    opacity=0.7
                )
            )
            
            fig.update_layout(
                title="Tail Risk Analysis - Return Distribution",
                xaxis_title="Daily Returns (%)",
                yaxis_title="Frequency",
                height=500,
                showlegend=True,
                template='plotly_white',
                bargap=0.1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating tail risk analysis chart: {str(e)}")
            return self._create_error_chart("Tail Risk Analysis Error")
    
    def create_risk_metrics_dashboard(self, risk_analysis: Dict[str, Any]) -> go.Figure:
        """Create comprehensive risk metrics dashboard"""
        try:
            if 'advanced_metrics' not in risk_analysis:
                return self._create_error_chart("Risk Metrics Not Available")
                
            metrics = risk_analysis['advanced_metrics']
            
            # Create gauge charts for key risk metrics
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'VaR 95%', 'Calmar Ratio', 'Sortino Ratio',
                    'Tail Ratio', 'Omega Ratio', 'Pain Ratio'
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
                ]
            )
            
            # VaR 95% gauge (risk - lower is better)
            var_95 = abs(metrics.get('VaR_95', 0))
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=var_95,
                    title={'text': "VaR 95% (%)"},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': self._get_risk_color(var_95, [0, 2, 5, 10])},
                        'steps': [
                            {'range': [0, 2], 'color': self.risk_colors['low']},
                            {'range': [2, 5], 'color': self.risk_colors['medium']},
                            {'range': [5, 10], 'color': self.risk_colors['high']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 5
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Calmar Ratio gauge (higher is better)
            calmar = metrics.get('Calmar_Ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=calmar,
                    title={'text': "Calmar Ratio"},
                    gauge={
                        'axis': {'range': [0, 2]},
                        'bar': {'color': self._get_performance_color(calmar, [0, 0.5, 1, 2])},
                        'steps': [
                            {'range': [0, 0.5], 'color': self.risk_colors['high']},
                            {'range': [0.5, 1], 'color': self.risk_colors['medium']},
                            {'range': [1, 2], 'color': self.risk_colors['low']}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            # Sortino Ratio gauge (higher is better)
            sortino = metrics.get('Sortino_Ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sortino,
                    title={'text': "Sortino Ratio"},
                    gauge={
                        'axis': {'range': [0, 3]},
                        'bar': {'color': self._get_performance_color(sortino, [0, 1, 2, 3])},
                        'steps': [
                            {'range': [0, 1], 'color': self.risk_colors['high']},
                            {'range': [1, 2], 'color': self.risk_colors['medium']},
                            {'range': [2, 3], 'color': self.risk_colors['low']}
                        ]
                    }
                ),
                row=1, col=3
            )
            
            # Tail Ratio gauge (around 1 is neutral, >1 positive skew)
            tail_ratio = metrics.get('Tail_Ratio', 1)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=tail_ratio,
                    title={'text': "Tail Ratio"},
                    gauge={
                        'axis': {'range': [0, 3]},
                        'bar': {'color': self._get_tail_ratio_color(tail_ratio)},
                        'steps': [
                            {'range': [0, 0.5], 'color': self.risk_colors['high']},
                            {'range': [0.5, 1.5], 'color': self.risk_colors['medium']},
                            {'range': [1.5, 3], 'color': self.risk_colors['low']}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # Omega Ratio gauge (higher is better)
            omega = metrics.get('Omega_Ratio', 1)
            omega_display = min(omega, 5) if omega != np.inf else 5  # Cap for display
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=omega_display,
                    title={'text': "Omega Ratio"},
                    gauge={
                        'axis': {'range': [0, 5]},
                        'bar': {'color': self._get_performance_color(omega_display, [0, 1, 2, 5])},
                        'steps': [
                            {'range': [0, 1], 'color': self.risk_colors['high']},
                            {'range': [1, 2], 'color': self.risk_colors['medium']},
                            {'range': [2, 5], 'color': self.risk_colors['low']}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            # Pain Ratio gauge (higher is better)
            pain_ratio = metrics.get('Pain_Ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=pain_ratio,
                    title={'text': "Pain Ratio"},
                    gauge={
                        'axis': {'range': [0, 2]},
                        'bar': {'color': self._get_performance_color(pain_ratio, [0, 0.5, 1, 2])},
                        'steps': [
                            {'range': [0, 0.5], 'color': self.risk_colors['high']},
                            {'range': [0.5, 1], 'color': self.risk_colors['medium']},
                            {'range': [1, 2], 'color': self.risk_colors['low']}
                        ]
                    }
                ),
                row=2, col=3
            )
            
            fig.update_layout(
                title="Risk Metrics Dashboard",
                height=600,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk metrics dashboard: {str(e)}")
            return self._create_error_chart("Risk Metrics Dashboard Error")
    
    def _get_risk_color(self, value: float, thresholds: List[float]) -> str:
        """Get color based on risk level (lower is better)"""
        if value <= thresholds[1]:
            return self.risk_colors['low']
        elif value <= thresholds[2]:
            return self.risk_colors['medium']
        else:
            return self.risk_colors['high']
    
    def _get_performance_color(self, value: float, thresholds: List[float]) -> str:
        """Get color based on performance level (higher is better)"""
        if value >= thresholds[2]:
            return self.risk_colors['low']
        elif value >= thresholds[1]:
            return self.risk_colors['medium']
        else:
            return self.risk_colors['high']
    
    def _get_tail_ratio_color(self, tail_ratio: float) -> str:
        """Get color for tail ratio (1 is neutral, >1 is positive skew)"""
        if tail_ratio > 1.5:
            return self.risk_colors['low']  # Good positive skew
        elif tail_ratio > 0.7:
            return self.risk_colors['medium']  # Neutral
        else:
            return self.risk_colors['high']  # Negative skew
    
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
