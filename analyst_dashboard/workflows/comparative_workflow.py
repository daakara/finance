"""
Comparative Analysis Workflow - Handles multi-asset comparison and analysis
Orchestrates comparative metrics, correlation analysis, and portfolio insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from ..core.asset_data_manager import AssetDataManager
from ..analyzers.technical_analyzer import TechnicalAnalysisProcessor
from ..visualizers.chart_visualizer import ChartVisualizer
from ..visualizers.metrics_display import MetricsDisplayManager

logger = logging.getLogger(__name__)

class ComparativeAnalysisWorkflow:
    """Orchestrates comparative analysis workflow for multiple assets."""
    
    def __init__(self):
        self.data_manager = AssetDataManager()
        self.technical_analyzer = TechnicalAnalysisProcessor()
        self.chart_visualizer = ChartVisualizer()
        self.metrics_display = MetricsDisplayManager()
    
    def run_comparative_analysis(self, symbols: List[str], asset_types: List[str], 
                               time_period: str = '1y') -> Dict[str, Any]:
        """Run comparative analysis workflow for multiple assets."""
        try:
            st.write(f"📊 **Comparative Analysis** ({len(symbols)} assets)")
            
            # Step 1: Fetch data for all assets
            with st.spinner("Fetching data for all assets..."):
                assets_data = self._fetch_multi_asset_data(symbols, asset_types, time_period)
            
            if 'error' in assets_data:
                st.error(f"Data fetch failed: {assets_data['error']}")
                return assets_data
            
            # Step 2: Perform comparative analysis
            comparative_results = self._perform_comparative_analysis(assets_data)
            
            # Step 3: Display results
            self._display_comparative_results(comparative_results, symbols)
            
            return comparative_results
            
        except Exception as e:
            logger.error(f"Error in comparative analysis workflow: {str(e)}")
            st.error(f"Comparative analysis workflow failed: {str(e)}")
            return {'error': str(e)}
    
    def _fetch_multi_asset_data(self, symbols: List[str], asset_types: List[str], 
                               time_period: str) -> Dict[str, Any]:
        """Fetch data for multiple assets."""
        try:
            assets_data = {}
            failed_symbols = []
            
            for i, symbol in enumerate(symbols):
                try:
                    asset_type = asset_types[i] if i < len(asset_types) else 'stock'
                    
                    # Fetch price data based on asset type
                    if asset_type == 'stock':
                        price_data = self.data_manager.fetch_stock_data(symbol, time_period)
                        info_data = self.data_manager.fetch_stock_info(symbol)
                    elif asset_type == 'etf':
                        price_data = self.data_manager.fetch_etf_data(symbol, time_period)
                        info_data = self.data_manager.fetch_etf_info(symbol)
                    elif asset_type == 'crypto':
                        price_data = self.data_manager.fetch_crypto_data(symbol, time_period)
                        info_data = self.data_manager.fetch_crypto_info(symbol)
                    else:
                        failed_symbols.append(f"{symbol} (unsupported type: {asset_type})")
                        continue
                    
                    if not price_data.empty:
                        assets_data[symbol] = {
                            'price_data': price_data,
                            'info_data': info_data,
                            'asset_type': asset_type
                        }
                    else:
                        failed_symbols.append(f"{symbol} (no price data)")
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    failed_symbols.append(f"{symbol} (fetch error)")
            
            if not assets_data:
                return {'error': 'No valid asset data could be fetched'}
            
            if failed_symbols:
                st.warning(f"Failed to fetch data for: {', '.join(failed_symbols)}")
            
            return {
                'assets_data': assets_data,
                'time_period': time_period,
                'successful_symbols': list(assets_data.keys()),
                'failed_symbols': failed_symbols
            }
            
        except Exception as e:
            logger.error(f"Error fetching multi-asset data: {str(e)}")
            return {'error': str(e)}
    
    def _perform_comparative_analysis(self, multi_asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis on multiple assets."""
        try:
            assets_data = multi_asset_data['assets_data']
            results = {
                'symbols': multi_asset_data['successful_symbols'],
                'time_period': multi_asset_data['time_period']
            }
            
            # Extract price data for all assets
            price_comparison_data = {}
            for symbol, data in assets_data.items():
                price_comparison_data[symbol] = data['price_data']
            
            # Performance comparison
            with st.spinner("Calculating performance metrics..."):
                performance_comparison = self._calculate_comparative_performance(price_comparison_data)
                results['performance_comparison'] = performance_comparison
            
            # Correlation analysis
            with st.spinner("Performing correlation analysis..."):
                correlation_analysis = self._calculate_correlation_analysis(price_comparison_data)
                results['correlation_analysis'] = correlation_analysis
            
            # Risk-Return analysis
            with st.spinner("Analyzing risk-return profiles..."):
                risk_return_analysis = self._calculate_risk_return_analysis(price_comparison_data)
                results['risk_return_analysis'] = risk_return_analysis
            
            # Technical signals comparison
            with st.spinner("Comparing technical signals..."):
                technical_comparison = self._compare_technical_signals(price_comparison_data)
                results['technical_comparison'] = technical_comparison
            
            # Store raw price data for charts
            results['price_data'] = price_comparison_data
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing comparative analysis: {str(e)}")
            return {'error': str(e)}
    
    def _display_comparative_results(self, results: Dict[str, Any], symbols: List[str]):
        """Display comparative analysis results."""
        try:
            # Overview section
            st.subheader("📈 Performance Comparison Overview")
            self._display_performance_comparison(results)
            
            # Detailed analysis tabs
            perf_tab, corr_tab, risk_tab, tech_tab, charts_tab = st.tabs([
                "Performance Metrics", "Correlation Analysis", "Risk-Return", 
                "Technical Signals", "Interactive Charts"
            ])
            
            with perf_tab:
                if 'performance_comparison' in results:
                    self._display_detailed_performance(results['performance_comparison'])
            
            with corr_tab:
                if 'correlation_analysis' in results:
                    self._display_correlation_analysis(results['correlation_analysis'])
            
            with risk_tab:
                if 'risk_return_analysis' in results:
                    self._display_risk_return_analysis(results['risk_return_analysis'])
            
            with tech_tab:
                if 'technical_comparison' in results:
                    self._display_technical_comparison(results['technical_comparison'])
            
            with charts_tab:
                if 'price_data' in results:
                    self._display_comparative_charts(results['price_data'], symbols)
                    
        except Exception as e:
            logger.error(f"Error displaying comparative results: {str(e)}")
            st.error("Error displaying comparative analysis results")
    
    def _calculate_comparative_performance(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Calculate performance metrics for comparison."""
        try:
            performance_metrics = {}
            
            for symbol, data in price_data.items():
                if data.empty or 'Close' not in data.columns:
                    continue
                
                close_prices = data['Close']
                daily_returns = close_prices.pct_change().dropna()
                
                # Calculate metrics
                total_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100
                volatility = daily_returns.std() * (252 ** 0.5) * 100
                avg_return = daily_returns.mean() * 252 * 100
                
                # Sharpe ratio
                sharpe_ratio = avg_return / volatility if volatility != 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdowns.min() * 100
                
                performance_metrics[symbol] = {
                    'Return': total_return,
                    'Volatility': volatility,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown,
                    'Avg_Daily_Return': avg_return
                }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating comparative performance: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_correlation_analysis(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate correlation analysis between assets."""
        try:
            # Extract close prices for all assets
            close_prices = {}
            for symbol, data in price_data.items():
                if not data.empty and 'Close' in data.columns:
                    close_prices[symbol] = data['Close']
            
            if len(close_prices) < 2:
                return {'error': 'Need at least 2 assets for correlation analysis'}
            
            # Create DataFrame with aligned data
            correlation_df = pd.DataFrame(close_prices)
            correlation_df = correlation_df.dropna()
            
            # Calculate correlation matrix
            correlation_matrix = correlation_df.corr()
            
            # Calculate daily returns correlation
            returns_df = correlation_df.pct_change().dropna()
            returns_correlation = returns_df.corr()
            
            return {
                'price_correlation': correlation_matrix,
                'returns_correlation': returns_correlation,
                'correlation_data': correlation_df
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_risk_return_analysis(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate risk-return analysis for portfolio insights."""
        try:
            risk_return_data = {}
            
            for symbol, data in price_data.items():
                if data.empty or 'Close' not in data.columns:
                    continue
                
                close_prices = data['Close']
                daily_returns = close_prices.pct_change().dropna()
                
                # Annualized metrics
                annual_return = daily_returns.mean() * 252 * 100
                annual_volatility = daily_returns.std() * (252 ** 0.5) * 100
                
                # Risk-adjusted metrics
                sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
                
                # Downside deviation (for Sortino ratio)
                downside_returns = daily_returns[daily_returns < 0]
                downside_deviation = downside_returns.std() * (252 ** 0.5) * 100 if len(downside_returns) > 0 else 0
                sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else 0
                
                risk_return_data[symbol] = {
                    'Annual_Return': annual_return,
                    'Annual_Volatility': annual_volatility,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Sortino_Ratio': sortino_ratio,
                    'Return_Volatility_Ratio': annual_return / annual_volatility if annual_volatility != 0 else 0
                }
            
            return risk_return_data
            
        except Exception as e:
            logger.error(f"Error calculating risk-return analysis: {str(e)}")
            return {'error': str(e)}
    
    def _compare_technical_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Compare technical signals across assets."""
        try:
            technical_comparison = {}
            
            for symbol, data in price_data.items():
                if data.empty:
                    continue
                
                # Get technical analysis for this asset
                tech_analysis = self.technical_analyzer.analyze_technical_data(data)
                
                if 'signals' in tech_analysis:
                    technical_comparison[symbol] = tech_analysis['signals']
            
            return technical_comparison
            
        except Exception as e:
            logger.error(f"Error comparing technical signals: {str(e)}")
            return {'error': str(e)}
    
    def _display_performance_comparison(self, results: Dict[str, Any]):
        """Display performance comparison overview."""
        try:
            if 'performance_comparison' in results:
                performance_data = results['performance_comparison']
                
                # Create metrics summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Best Performer (Return)**")
                    best_return = max(performance_data.items(), key=lambda x: x[1].get('Return', -float('inf')))
                    st.success(f"{best_return[0]}: {best_return[1]['Return']:.2f}%")
                
                with col2:
                    st.write("**Lowest Volatility**")
                    lowest_vol = min(performance_data.items(), key=lambda x: x[1].get('Volatility', float('inf')))
                    st.info(f"{lowest_vol[0]}: {lowest_vol[1]['Volatility']:.2f}%")
                
                with col3:
                    st.write("**Best Sharpe Ratio**")
                    best_sharpe = max(performance_data.items(), key=lambda x: x[1].get('Sharpe_Ratio', -float('inf')))
                    st.success(f"{best_sharpe[0]}: {best_sharpe[1]['Sharpe_Ratio']:.2f}")
                    
        except Exception as e:
            logger.error(f"Error displaying performance comparison: {str(e)}")
    
    def _display_detailed_performance(self, performance_data: Dict[str, Dict]):
        """Display detailed performance metrics table."""
        try:
            self.metrics_display.display_comparison_table(
                performance_data, "Detailed Performance Metrics"
            )
            
            # Performance metrics chart
            perf_chart = self.chart_visualizer.create_performance_metrics_chart(performance_data)
            st.plotly_chart(perf_chart, width='stretch')
            
        except Exception as e:
            logger.error(f"Error displaying detailed performance: {str(e)}")
    
    def _display_correlation_analysis(self, correlation_data: Dict[str, Any]):
        """Display correlation analysis results."""
        try:
            if 'returns_correlation' in correlation_data:
                st.write("**Returns Correlation Matrix**")
                correlation_heatmap = self.chart_visualizer.create_correlation_heatmap(
                    correlation_data['returns_correlation']
                )
                st.plotly_chart(correlation_heatmap, width='stretch')
                
                # Display correlation insights
                corr_matrix = correlation_data['returns_correlation']
                st.write("**Correlation Insights:**")
                
                # Find highest and lowest correlations (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                masked_corr = corr_matrix.where(mask)
                
                # Highest correlation
                max_corr_idx = masked_corr.stack().idxmax()
                max_corr_val = masked_corr.stack().max()
                st.write(f"• Highest correlation: {max_corr_idx[0]} - {max_corr_idx[1]} ({max_corr_val:.3f})")
                
                # Lowest correlation
                min_corr_idx = masked_corr.stack().idxmin()
                min_corr_val = masked_corr.stack().min()
                st.write(f"• Lowest correlation: {min_corr_idx[0]} - {min_corr_idx[1]} ({min_corr_val:.3f})")
                
        except Exception as e:
            logger.error(f"Error displaying correlation analysis: {str(e)}")
    
    def _display_risk_return_analysis(self, risk_return_data: Dict[str, Dict]):
        """Display risk-return analysis."""
        try:
            self.metrics_display.display_comparison_table(
                risk_return_data, "Risk-Return Analysis"
            )
            
            # Create risk-return scatter plot
            if len(risk_return_data) > 1:
                symbols = list(risk_return_data.keys())
                returns = [risk_return_data[s]['Annual_Return'] for s in symbols]
                volatilities = [risk_return_data[s]['Annual_Volatility'] for s in symbols]
                
                import plotly.express as px
                
                fig = px.scatter(
                    x=volatilities, y=returns, text=symbols,
                    labels={'x': 'Annual Volatility (%)', 'y': 'Annual Return (%)'},
                    title='Risk-Return Profile'
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, width='stretch')
                
        except Exception as e:
            logger.error(f"Error displaying risk-return analysis: {str(e)}")
    
    def _display_technical_comparison(self, technical_data: Dict[str, Dict]):
        """Display technical signals comparison."""
        try:
            if not technical_data:
                st.warning("No technical signals data available")
                return
            
            # Create comparison table for technical signals
            st.write("**Technical Signals Comparison**")
            
            # Get all unique signal types
            all_signals = set()
            for signals in technical_data.values():
                all_signals.update(signals.keys())
            
            # Create comparison DataFrame
            comparison_data = {}
            for symbol, signals in technical_data.items():
                comparison_data[symbol] = {signal: signals.get(signal, 'N/A') for signal in all_signals}
            
            comparison_df = pd.DataFrame(comparison_data).T
            
            # Display with color coding
            def highlight_signals(val):
                if 'Bullish' in str(val) or 'Oversold' in str(val):
                    return 'background-color: lightgreen'
                elif 'Bearish' in str(val) or 'Overbought' in str(val):
                    return 'background-color: lightcoral'
                else:
                    return ''
            
            styled_df = comparison_df.style.applymap(highlight_signals)
            st.dataframe(styled_df, width='stretch')
            
        except Exception as e:
            logger.error(f"Error displaying technical comparison: {str(e)}")
    
    def _display_comparative_charts(self, price_data: Dict[str, pd.DataFrame], symbols: List[str]):
        """Display comparative charts."""
        try:
            # Comparison chart
            comparison_chart = self.chart_visualizer.create_comparison_chart(price_data)
            st.plotly_chart(comparison_chart, width='stretch')
            
        except Exception as e:
            logger.error(f"Error displaying comparative charts: {str(e)}")
            st.error("Error generating comparative charts")
