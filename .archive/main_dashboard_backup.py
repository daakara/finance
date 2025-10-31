"""
Financial Analyst Dashboard - Streamlit Main Application
Integrated modular financial analysis platform following analyst workflow
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any

# Configure page
st.set_page_config(
    page_title="Financial Analyst Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
try:
    from data_fetcher import unified_fetcher
    from analysis_engine import (
        technical_engine, risk_engine, performance_engine,
        macro_engine, fundamental_engine, portfolio_engine, forecasting_engine,
        MacroeconomicAnalysisEngine, FundamentalAnalysisEngine,
        PortfolioStrategyEngine, ForecastingEngine
    )
    from visualizer import financial_visualizer
    from config import config
    
    # Import new modular UI components
    from ui_components import (
        SidebarManager,
        MetricsDisplayManager,
        DataSourceIndicator,
        ProgressManager,
        ChartManager,
        AnalysisTabManager,
        ErrorHandler
    )
    from analysis_renderers import (
        MacroeconomicRenderer,
        FundamentalRenderer,
        PortfolioRenderer,
        ForecastingRenderer,
        CommentaryRenderer
    )
    from analysis_components import (
        DataProcessors,
        TechnicalIndicators,
        RiskMetrics,
        MarketDataFetchers
    )
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

# Analysis engines are imported as global instances from analysis_engine module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalystDashboard:
    """Main Financial Analyst Dashboard following analyst workflow."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.sidebar_manager = SidebarManager()
        self.setup_sidebar()
        
    def setup_sidebar(self):
        """Setup sidebar controls using modular components."""
        # Set up asset selection controls
        asset_controls = self.sidebar_manager.create_asset_selection_controls()
        self.asset_type = asset_controls['asset_type']
        self.ticker = asset_controls['ticker']
        self.period = asset_controls['period']
        
        # Set up analysis mode controls
        analysis_controls = self.sidebar_manager.create_analysis_mode_controls(self.asset_type)
        self.analysis_mode = analysis_controls['analysis_mode']
        self.comparison_tickers = analysis_controls.get('comparison_tickers', [])
        
        # Set up technical analysis controls
        tech_controls = self.sidebar_manager.create_technical_analysis_controls()
        self.show_volume = tech_controls['show_volume']
        self.show_indicators = tech_controls['show_indicators']
        
        # Show market status
        self.sidebar_manager.show_market_status()
    
    def run(self):
        """Run the main dashboard application."""
        try:
            # Main title
            st.title("📊 Financial Analyst Dashboard")
            st.markdown(f"**Analyzing {self.asset_type}: {self.ticker} | Period: {self.period}**")
            
            if self.analysis_mode == "Single Asset Analysis":
                self.single_asset_workflow()
            else:
                self.comparative_analysis_workflow()
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    def single_asset_workflow(self):
        """Single asset analysis following analyst workflow."""
        try:
            # Load data
            with st.spinner("Loading asset data..."):
                asset_data = unified_fetcher.get_data(self.ticker, self.asset_type, self.period)
            
            if 'error' in asset_data:
                st.error(f"Error loading data: {asset_data['error']}")
                return
            
            # Show data source indicator
            if asset_data.get('data_source') == 'sample':
                st.info("📊 **Demo Mode**: Using sample data due to network connectivity issues. All analysis features are fully functional.")
            else:
                st.success("📡 **Live Data**: Using real-time market data.")
            
            # Analyst Workflow Sections
            
            # 1. HIGH-LEVEL SUMMARY (The Snapshot)
            self.render_high_level_summary(asset_data)
            
            st.markdown("---")
            
            # 2. PRICE ACTION & TECHNICAL ANALYSIS (The Trend)
            self.render_price_action_analysis(asset_data)
            
            st.markdown("---")
            
            # 3. RISK & VOLATILITY PROFILE (The Edge)
            self.render_risk_volatility_profile(asset_data)
            
            st.markdown("---")
            
            # 4. MACROECONOMIC CONTEXT & MARKET ENVIRONMENT
            self.render_macroeconomic_analysis(asset_data)
            
            st.markdown("---")
            
            # 5. FUNDAMENTAL ANALYSIS (Asset-Specific)
            self.render_fundamental_analysis(asset_data)
            
            st.markdown("---")
            
            # 6. PORTFOLIO STRATEGY & ALLOCATION
            self.render_portfolio_strategy_analysis(asset_data)
            
            st.markdown("---")
            
            # 7. FORECASTING & FORWARD-LOOKING OUTLOOK
            self.render_forecasting_analysis(asset_data)
            
            st.markdown("---")
            
            # 8. ANALYST COMMENTARY & RECOMMENDATIONS
            self.render_comprehensive_analyst_commentary(asset_data)
            
        except Exception as e:
            st.error(f"Error in single asset analysis: {str(e)}")
            logger.error(f"Single asset analysis error: {str(e)}")
    
    def comparative_analysis_workflow(self):
        """Comparative analysis workflow."""
        try:
            st.subheader("🔍 Comparative Analysis")
            
            all_tickers = [self.ticker] + self.comparison_tickers
            st.markdown(f"**Comparing {len(all_tickers)} assets: {', '.join(all_tickers)}**")
            
            # Load data for all assets
            comparison_data = {}
            progress_bar = st.progress(0)
            
            sample_data_count = 0
            
            for i, ticker in enumerate(all_tickers):
                try:
                    with st.spinner(f"Loading {ticker}..."):
                        data = unified_fetcher.get_data(ticker, self.asset_type, self.period)
                        if 'error' not in data:
                            comparison_data[ticker] = data
                            if data.get('data_source') == 'sample':
                                sample_data_count += 1
                    progress_bar.progress((i + 1) / len(all_tickers))
                except Exception as e:
                    st.warning(f"Could not load {ticker}: {str(e)}")
            
            progress_bar.empty()
            
            # Show data source summary
            if sample_data_count > 0:
                if sample_data_count == len(comparison_data):
                    st.info("📊 **Demo Mode**: All data is sample data due to network issues.")
                else:
                    st.warning(f"⚠️ **Mixed Data**: {sample_data_count} out of {len(comparison_data)} assets using sample data.")
            else:
                st.success("📡 **Live Data**: All assets using real-time market data.")
            
            if len(comparison_data) < 2:
                st.error("Need at least 2 assets with valid data for comparison")
                return
            
            # COMPARATIVE ANALYSIS SECTIONS
            
            # 1. Normalized Returns Comparison
            self.render_normalized_returns_comparison(comparison_data)
            
            st.markdown("---")
            
            # 2. Performance Metrics Comparison
            self.render_performance_metrics_comparison(comparison_data)
            
            st.markdown("---")
            
            # 3. Correlation Matrix
            self.render_correlation_analysis(comparison_data)
            
            st.markdown("---")
            
            # 4. Risk-Return Profile
            self.render_risk_return_analysis(comparison_data)
            
        except Exception as e:
            st.error(f"Error in comparative analysis: {str(e)}")
            logger.error(f"Comparative analysis error: {str(e)}")
    
    def render_high_level_summary(self, asset_data: Dict[str, Any]):
        """Render high-level summary section using modular components."""
        st.subheader("📋 High-Level Summary (The Snapshot)")
        
        if 'price_data' not in asset_data or asset_data['price_data'].empty:
            ErrorHandler.handle_data_error("No price data available")
            return
        
        try:
            # Use MetricsDisplayManager for consistent metric display
            metrics_manager = MetricsDisplayManager()
            
            # Calculate price metrics
            price_data = asset_data['price_data']
            current_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
            daily_change = current_price - prev_price
            daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Display price metrics
            price_metrics = {
                'Current Price': {
                    'value': f"${current_price:.2f}",
                    'delta': f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
                }
            }
            
            # Get asset-specific metrics
            asset_metrics = self._get_asset_specific_metrics(asset_data)
            price_metrics.update(asset_metrics)
            
            # Render all metrics using the modular display manager
            metrics_manager.display_key_metrics(price_metrics)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "High-Level Summary")
    
    def _get_asset_specific_metrics(self, asset_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get asset-specific metrics for display."""
        asset_info = asset_data.get(f'{self.asset_type.lower()}_info', {})
        metrics = {}
        
        if self.asset_type == "Stock":
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
        
        elif self.asset_type == "ETF":
            total_assets = asset_info.get('totalAssets', 0)
            metrics['Total Assets'] = {
                'value': self._format_large_number(total_assets) if total_assets else "N/A"
            }
            
            expense_ratio = asset_info.get('expenseRatio', 0)
            metrics['Expense Ratio'] = {
                'value': f"{expense_ratio:.2f}%" if expense_ratio else "N/A"
            }
            
            category = asset_info.get('category', 'N/A')
            metrics['Category'] = {
                'value': category
            }
        
        elif self.asset_type == "Cryptocurrency":
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
    
    def render_price_action_analysis(self, asset_data: Dict[str, Any]):
        """Render price action and technical analysis section."""
        st.subheader("📈 Price Action & Technical Analysis (The Trend)")
        
        price_data = asset_data['price_data']
        if price_data.empty:
            st.warning("No price data available")
            return
        
        # Perform technical analysis
        with st.spinner("Analyzing technical indicators..."):
            tech_analysis = technical_engine.analyze(price_data, asset_data.get(f'{self.asset_type.lower()}_info'))
        
        if 'error' in tech_analysis:
            st.error(f"Technical analysis error: {tech_analysis['error']}")
            return
        
        # Interactive candlestick chart with technical indicators
        st.markdown("**Interactive Price Chart with Technical Indicators**")
        
        chart = financial_visualizer.create_advanced_candlestick_chart(
            price_data,
            tech_analysis.get('indicators', {}),
            title=f"{self.ticker} Technical Analysis",
            height=800
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # Technical signals summary
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
        
        # Support and resistance levels
        support_resistance = tech_analysis.get('support_resistance', {})
        if support_resistance:
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
    
    def render_risk_volatility_profile(self, asset_data: Dict[str, Any]):
        """Render risk and volatility analysis section."""
        st.subheader("⚠️ Risk & Volatility Profile (The Edge)")
        
        price_data = asset_data['price_data']
        if price_data.empty:
            st.warning("No price data available")
            return
        
        # Perform risk analysis
        with st.spinner("Analyzing risk metrics..."):
            risk_analysis = risk_engine.analyze(price_data, asset_data.get(f'{self.asset_type.lower()}_info'))
        
        if 'error' in risk_analysis:
            st.error(f"Risk analysis error: {risk_analysis['error']}")
            return
        
        # Risk metrics summary
        st.markdown("**Risk Metrics Summary**")
        
        vol_metrics = risk_analysis.get('volatility_metrics', {})
        drawdown_metrics = risk_analysis.get('drawdown_analysis', {})
        var_metrics = risk_analysis.get('var_analysis', {})
        risk_adjusted = risk_analysis.get('risk_adjusted_returns', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_vol = vol_metrics.get('annual_volatility', 0)
            st.metric(
                "Annual Volatility",
                f"{annual_vol:.2f}%",
                help="Annualized standard deviation of returns"
            )
        
        with col2:
            max_dd = drawdown_metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                help="Maximum peak-to-trough decline"
            )
        
        with col3:
            sharpe = risk_adjusted.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            var_5 = var_metrics.get('var_5_percent', 0)
            st.metric(
                "VaR (5%)",
                f"{var_5:.2f}%",
                help="Value at Risk (5% confidence level)"
            )
        
        # Volatility and drawdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**20-Day Rolling Volatility**")
            returns = price_data['Close'].pct_change().dropna()
            vol_chart = financial_visualizer.create_volatility_chart(
                returns,
                title="Rolling Volatility Analysis"
            )
            st.plotly_chart(vol_chart, use_container_width=True)
        
        with col2:
            st.markdown("**Drawdown Analysis**")
            dd_chart = financial_visualizer.create_drawdown_chart(
                price_data['Close'],
                title="Drawdown Analysis"
            )
            st.plotly_chart(dd_chart, use_container_width=True)
    
    def render_macroeconomic_analysis(self, asset_data: Dict[str, Any]):
        """Render macroeconomic context using modular renderer."""
        try:
            st.subheader("🌍 Macroeconomic Context & Market Environment")
            
            # Get asset info for analysis
            asset_info = asset_data.get('stock_info') or asset_data.get('etf_info') or asset_data.get('crypto_info', {})
            asset_info['asset_type'] = asset_data.get('asset_type', 'Unknown')
            
            # Perform macroeconomic analysis
            with st.spinner("Analyzing macroeconomic factors..."):
                macro_analysis = macro_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in macro_analysis:
                ErrorHandler.handle_analysis_error(macro_analysis['error'], "Macroeconomic Analysis")
                return
            
            # Use modular renderers for each section
            col1, col2 = st.columns(2)
            
            with col1:
                if 'monetary_policy' in macro_analysis:
                    MacroeconomicRenderer.render_monetary_policy_section(macro_analysis['monetary_policy'])
                
                if 'inflation_impact' in macro_analysis:
                    MacroeconomicRenderer.render_inflation_section(macro_analysis['inflation_impact'])
            
            with col2:
                if 'market_correlations' in macro_analysis:
                    MacroeconomicRenderer.render_market_correlations_section(macro_analysis['market_correlations'])
                
                # Economic cycle section (keeping original for now)
                if 'economic_cycle' in macro_analysis:
                    st.markdown("### 📊 Economic Cycle")
                    cycle = macro_analysis['economic_cycle']
                    
                    metrics_manager = MetricsDisplayManager()
                    cycle_metrics = {
                        'GDP Growth': {'value': f"{cycle.get('gdp_growth', 0):.1f}%"},
                        'Unemployment': {'value': f"{cycle.get('unemployment_rate', 0):.1f}%"}
                    }
                    metrics_manager.display_metrics_row(cycle_metrics)
                    
                    st.write(f"**Current Phase:** {cycle.get('current_cycle_phase', 'Unknown')}")
                    st.write(f"**Recent Performance:** {cycle.get('recent_performance', 0):.1f}%")
            
            # Interest rate sensitivity chart (keep as is for now)
            if 'interest_rate_sensitivity' in macro_analysis:
                st.markdown("### 📈 Interest Rate Sensitivity Analysis")
                irs = macro_analysis['interest_rate_sensitivity']
                
                sensitivity_metrics = {
                    'Rate Correlation': {'value': f"{irs.get('rate_correlation', 0):.3f}"},
                    'Modified Duration': {'value': f"{irs.get('modified_duration', 0):.2f}"},
                    'Sensitivity Level': {'value': irs.get('sensitivity_level', 'Unknown')}
                }
                
                metrics_manager = MetricsDisplayManager()
                metrics_manager.display_metrics_row(sensitivity_metrics)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Macroeconomic Analysis")
    
    def render_fundamental_analysis(self, asset_data: Dict[str, Any]):
        """Render fundamental analysis using modular renderers."""
        try:
            st.subheader("🔍 Fundamental Analysis")
            
            # Get asset info
            asset_info = asset_data.get('stock_info') or asset_data.get('etf_info') or asset_data.get('crypto_info', {})
            asset_info['asset_type'] = asset_data.get('asset_type', 'Unknown')
            
            # Perform fundamental analysis
            with st.spinner("Performing fundamental analysis..."):
                fundamental_analysis = fundamental_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in fundamental_analysis:
                ErrorHandler.handle_analysis_error(fundamental_analysis['error'], "Fundamental Analysis")
                return
            
            asset_type = asset_data.get('asset_type', 'Unknown')
            
            # Use modular renderers for each asset type
            if asset_type == 'ETF':
                FundamentalRenderer.render_etf_fundamentals(fundamental_analysis)
            elif asset_type == 'Cryptocurrency':
                FundamentalRenderer.render_crypto_fundamentals(fundamental_analysis)
            elif asset_type == 'Stock':
                # Stock fundamental rendering can be added to FundamentalRenderer if needed
                self._render_stock_fundamentals(fundamental_analysis)
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Fundamental Analysis")
    
    def _render_stock_fundamentals(self, analysis: Dict[str, Any]):
        """Render stock-specific fundamental analysis."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💼 Financial Metrics")
            if 'financial_metrics' in analysis:
                metrics = analysis['financial_metrics']
                st.metric("Market Cap", self.format_large_number(metrics.get('market_cap', 0)))
                st.metric("P/E Ratio", f"{metrics.get('pe_ratio', 0):.2f}")
                st.metric("Dividend Yield", f"{metrics.get('dividend_yield', 0):.2f}%")
                st.metric("Beta", f"{metrics.get('beta', 0):.2f}")
        
        with col2:
            st.markdown("### 📊 Valuation Analysis")
            if 'valuation_analysis' in analysis:
                valuation = analysis['valuation_analysis']
                for key, value in valuation.items():
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    def render_portfolio_strategy_analysis(self, asset_data: Dict[str, Any]):
        """Render portfolio strategy and allocation analysis."""
        try:
            st.subheader("💼 Portfolio Strategy & Allocation")
            
            # Get asset info
            asset_info = asset_data.get('stock_info') or asset_data.get('etf_info') or asset_data.get('crypto_info', {})
            asset_info['asset_type'] = asset_data.get('asset_type', 'Unknown')
            
            # Perform portfolio analysis
            with st.spinner("Analyzing portfolio strategies..."):
                portfolio_analysis = portfolio_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in portfolio_analysis:
                st.error(f"Portfolio analysis error: {portfolio_analysis['error']}")
                return
            
            # Allocation strategies
            if 'allocation_strategies' in portfolio_analysis:
                st.markdown("### 🎯 Asset Allocation Strategies")
                
                strategies = portfolio_analysis['allocation_strategies']['strategies']
                
                # Create tabs for different strategies
                strategy_tabs = st.tabs(["Conservative", "Moderate", "Aggressive", "Institutional"])
                
                for i, (strategy_name, tab) in enumerate(zip(strategies.keys(), strategy_tabs)):
                    with tab:
                        strategy = strategies[strategy_name]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Risk Tolerance:** {strategy.get('risk_tolerance', 'Unknown')}")
                            st.write(f"**Time Horizon:** {strategy.get('time_horizon', 'Unknown')}")
                            st.metric("Target Volatility", f"{strategy.get('target_volatility', 0):.1%}")
                        
                        with col2:
                            st.metric("Expected Return", f"{strategy.get('expected_return', 0):.2%}")
                            st.metric("Expected Risk", f"{strategy.get('expected_risk', 0):.2%}")
                            st.metric("Sharpe Estimate", f"{strategy.get('sharpe_estimate', 0):.2f}")
                        
                        # Show allocation breakdown
                        allocation_data = {k: v for k, v in strategy.items() 
                                         if isinstance(v, (int, float)) and k not in 
                                         ['target_volatility', 'expected_return', 'expected_risk', 'sharpe_estimate']}
                        
                        if allocation_data:
                            allocation_df = pd.DataFrame(list(allocation_data.items()), 
                                                       columns=['Asset Class', 'Allocation'])
                            st.bar_chart(allocation_df.set_index('Asset Class'))
            
            # Stress testing results
            if 'stress_testing' in portfolio_analysis:
                st.markdown("### ⚠️ Stress Testing & Risk Scenarios")
                
                stress_results = portfolio_analysis['stress_testing']['stress_scenarios']
                
                # Create a comparison table
                stress_df = pd.DataFrame(stress_results).T
                stress_df['scenario_return'] = stress_df['scenario_return'].apply(lambda x: f"{x:.2%}")
                stress_df['probability'] = stress_df['probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(stress_df[['scenario_return', 'probability', 'recovery_time_estimate']], 
                           use_container_width=True)
                
                worst_case = portfolio_analysis['stress_testing']['worst_case_scenario']
                st.warning(f"**Worst Case Scenario:** {worst_case}")
            
            # Scenario analysis
            if 'scenario_analysis' in portfolio_analysis:
                st.markdown("### 🔮 Forward-Looking Scenarios")
                
                scenarios = portfolio_analysis['scenario_analysis']['scenarios']
                weighted_return = portfolio_analysis['scenario_analysis']['weighted_expected_return']
                
                st.metric("Scenario-Weighted Expected Return", f"{weighted_return:.1%}")
                
                # Show key scenarios
                scenario_df = pd.DataFrame(scenarios).T
                scenario_df['probability'] = scenario_df['probability'].apply(lambda x: f"{x:.1%}")
                scenario_df['expected_return'] = scenario_df['expected_return'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(scenario_df[['probability', 'expected_return', 'duration_months']], 
                           use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering portfolio strategy analysis: {str(e)}")
    
    def render_forecasting_analysis(self, asset_data: Dict[str, Any]):
        """Render forecasting and forward-looking analysis."""
        try:
            st.subheader("🔮 Forecasting & Forward-Looking Outlook")
            
            # Perform forecasting analysis
            with st.spinner("Generating forecasts and projections..."):
                forecast_analysis = forecasting_engine.analyze(asset_data['price_data'])
            
            if 'error' in forecast_analysis:
                st.error(f"Forecasting analysis error: {forecast_analysis['error']}")
                return
            
            # Price forecasting
            if 'price_forecasting' in forecast_analysis:
                st.markdown("### 📈 Price Forecasting")
                
                forecasting = forecast_analysis['price_forecasting']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Forecasting Horizon:** {forecasting.get('forecasting_horizon', '30 days')}")
                    st.write(f"**Recommended Model:** {forecasting.get('model_recommendation', 'Ensemble')}")
                
                with col2:
                    if 'ensemble_forecast' in forecasting:
                        ensemble = forecasting['ensemble_forecast']
                        if 'forecast' in ensemble and len(ensemble['forecast']) > 0:
                            current_price = asset_data['price_data']['Close'].iloc[-1] if not asset_data['price_data'].empty else 100
                            forecast_price = ensemble['forecast'][-1]  # 30-day forecast
                            price_change = (forecast_price - current_price) / current_price
                            
                            st.metric("30-Day Price Forecast", 
                                    f"${forecast_price:.2f}", 
                                    f"{price_change:.1%}")
                
                # Show model comparison
                if 'models' in forecasting:
                    st.markdown("#### 🎯 Model Comparison")
                    models = forecasting['models']
                    
                    model_comparison = []
                    for model_name, model_data in models.items():
                        if 'forecast' in model_data and len(model_data['forecast']) > 0:
                            final_forecast = model_data['forecast'][-1]
                            model_comparison.append({
                                'Model': model_name.replace('_', ' ').title(),
                                'Type': model_data.get('model_type', 'Unknown'),
                                '30-Day Forecast': f"${final_forecast:.2f}"
                            })
                    
                    if model_comparison:
                        model_df = pd.DataFrame(model_comparison)
                        st.dataframe(model_df, use_container_width=True)
            
            # Volatility forecasting
            if 'volatility_forecasting' in forecast_analysis:
                st.markdown("### 📊 Volatility Forecasting")
                
                vol_forecast = forecast_analysis['volatility_forecasting']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Volatility", f"{vol_forecast.get('current_volatility', 0):.2%}")
                with col2:
                    st.metric("Historical Average", f"{vol_forecast.get('historical_average', 0):.2%}")
                with col3:
                    st.metric("Volatility Regime", vol_forecast.get('volatility_regime', 'Unknown'))
            
            # Seasonality analysis
            if 'seasonality_analysis' in forecast_analysis:
                st.markdown("### 📅 Seasonal Patterns")
                
                seasonality = forecast_analysis['seasonality_analysis']
                
                if 'monthly_seasonality' in seasonality:
                    monthly = seasonality['monthly_seasonality']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Month", f"Month {monthly.get('best_month', 0)}")
                        st.metric("Worst Month", f"Month {monthly.get('worst_month', 0)}")
                    
                if 'day_of_week_effects' in seasonality:
                    dow = seasonality['day_of_week_effects']
                    
                    with col2:
                        st.metric("Best Day", dow.get('best_day', 'Unknown'))
                        st.metric("Worst Day", dow.get('worst_day', 'Unknown'))
            
        except Exception as e:
            st.error(f"Error rendering forecasting analysis: {str(e)}")
    
    def render_comprehensive_analyst_commentary(self, asset_data: Dict[str, Any]):
        """Render comprehensive analyst commentary and recommendations."""
        try:
            st.subheader("📝 Comprehensive Analyst Commentary")
            
            # Get all analysis data
            asset_info = asset_data.get('stock_info') or asset_data.get('etf_info') or asset_data.get('crypto_info', {})
            asset_info['asset_type'] = asset_data.get('asset_type', 'Unknown')
            
            # Generate comprehensive commentary
            commentary = self._generate_comprehensive_commentary(asset_data, asset_info)
            
            # Create tabs for different aspects of commentary
            commentary_tabs = st.tabs(["Executive Summary", "Investment Thesis", "Risk Assessment", "Catalysts & Outlook"])
            
            with commentary_tabs[0]:
                st.markdown("#### 📋 Executive Summary")
                st.write(commentary.get('executive_summary', 'Analysis in progress...'))
            
            with commentary_tabs[1]:
                st.markdown("#### 💡 Investment Thesis")
                st.write(commentary.get('investment_thesis', 'Investment thesis under development...'))
            
            with commentary_tabs[2]:
                st.markdown("#### ⚠️ Risk Assessment")
                st.write(commentary.get('risk_assessment', 'Risk analysis pending...'))
            
            with commentary_tabs[3]:
                st.markdown("#### 🚀 Catalysts & Outlook")
                st.write(commentary.get('catalysts_outlook', 'Market outlook analysis in progress...'))
            
            # Overall recommendation
            st.markdown("---")
            st.markdown("### 🎯 Overall Recommendation")
            
            recommendation = commentary.get('overall_recommendation', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rating = recommendation.get('rating', 'HOLD')
                if rating == 'BUY':
                    st.success(f"**Rating:** {rating}")
                elif rating == 'SELL':
                    st.error(f"**Rating:** {rating}")
                else:
                    st.warning(f"**Rating:** {rating}")
            
            with col2:
                target_price = recommendation.get('target_price', 0)
                if target_price > 0:
                    st.metric("Target Price", f"${target_price:.2f}")
            
            with col3:
                time_horizon = recommendation.get('time_horizon', '12 months')
                st.write(f"**Time Horizon:** {time_horizon}")
            
            # Key points
            if 'key_points' in recommendation:
                st.markdown("#### Key Points:")
                for point in recommendation['key_points']:
                    st.write(f"• {point}")
            
        except Exception as e:
            st.error(f"Error rendering analyst commentary: {str(e)}")
    
    def _generate_comprehensive_commentary(self, asset_data: Dict[str, Any], asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analyst commentary."""
        try:
            price_data = asset_data['price_data']
            asset_type = asset_data.get('asset_type', 'Unknown')
            ticker = asset_info.get('symbol', self.ticker)
            
            # Calculate key metrics
            current_price = price_data['Close'].iloc[-1] if not price_data.empty else 100
            price_change = price_data['Close'].pct_change().iloc[-1] if len(price_data) > 1 else 0
            total_return = ((current_price / price_data['Close'].iloc[0]) - 1) if not price_data.empty else 0
            volatility = price_data['Close'].pct_change().std() * np.sqrt(252) if len(price_data) > 1 else 0.2
            
            commentary = {}
            
            # Executive Summary
            commentary['executive_summary'] = f"""
            {ticker} ({asset_type}) is currently trading at ${current_price:.2f}, representing a 
            {total_return:.1%} total return over the analysis period. The asset exhibits 
            {volatility:.1%} annualized volatility, positioning it as a {'high' if volatility > 0.3 else 'moderate' if volatility > 0.15 else 'low'}-risk 
            investment. Recent price action shows {price_change:.1%} movement in the latest session.
            
            Based on our comprehensive analysis incorporating macroeconomic factors, fundamental metrics, 
            and technical indicators, we maintain a balanced view on the asset's near-term prospects while 
            acknowledging the broader market environment's influence on performance.
            """
            
            # Investment Thesis
            if asset_type == 'Stock':
                commentary['investment_thesis'] = f"""
                Our investment thesis for {ticker} centers on the company's position within the 
                {asset_info.get('sector', 'unknown')} sector and its ability to navigate current market conditions. 
                The stock's beta of {asset_info.get('beta', 1.0):.2f} suggests {'higher' if asset_info.get('beta', 1.0) > 1.2 else 'lower' if asset_info.get('beta', 1.0) < 0.8 else 'market-level'} 
                sensitivity to broader market movements.
                
                Key investment merits include the company's market positioning and the stock's 
                {'attractive' if asset_info.get('trailingPE', 20) < 20 else 'premium'} valuation at 
                {asset_info.get('trailingPE', 0):.1f}x trailing earnings.
                """
            elif asset_type == 'ETF':
                commentary['investment_thesis'] = f"""
                {ticker} provides diversified exposure to {asset_info.get('category', 'unknown')} assets, 
                making it suitable for investors seeking broad market participation with professional management. 
                The ETF's expense ratio of {asset_info.get('expenseRatio', 0.05):.3f}% is 
                {'competitive' if asset_info.get('expenseRatio', 0.05) < 0.1 else 'elevated'} within its peer group.
                
                This vehicle offers efficient access to a diversified portfolio, reducing single-asset risk 
                while maintaining exposure to the underlying asset class's growth potential.
                """
            else:  # Cryptocurrency
                commentary['investment_thesis'] = f"""
                {ticker} represents exposure to the digital asset ecosystem, characterized by high growth 
                potential but also elevated volatility and regulatory uncertainty. The cryptocurrency's 
                market position and adoption metrics suggest {'strong' if ticker in ['BTC', 'ETH'] else 'developing'} 
                fundamental support.
                
                Investment merit centers on the long-term adoption trajectory of blockchain technology 
                and digital assets, though investors should prepare for significant price volatility.
                """
            
            # Risk Assessment
            risk_level = 'High' if volatility > 0.3 else 'Moderate' if volatility > 0.15 else 'Low'
            commentary['risk_assessment'] = f"""
            Risk Level: {risk_level}
            
            Primary risks include market volatility ({volatility:.1%} annualized), macroeconomic sensitivity, 
            and asset-specific factors. The current market environment presents challenges from interest rate 
            policies and inflation concerns.
            
            {'Cryptocurrency-specific risks include regulatory changes, technology risks, and extreme volatility.' if asset_type == 'Cryptocurrency' else ''}
            {'ETF-specific risks include tracking error, underlying asset concentration, and management risk.' if asset_type == 'ETF' else ''}
            {'Company-specific risks include competitive pressures, operational challenges, and sector dynamics.' if asset_type == 'Stock' else ''}
            
            Investors should maintain appropriate position sizing and risk management protocols.
            """
            
            # Catalysts & Outlook
            commentary['catalysts_outlook'] = f"""
            Near-term catalysts include broader market sentiment shifts, macroeconomic data releases, 
            and sector-specific developments. The Federal Reserve's monetary policy stance remains a 
            key driver for asset performance across all categories.
            
            Medium-term outlook depends on economic growth sustainability, inflation trajectory, and 
            {'regulatory clarity for digital assets' if asset_type == 'Cryptocurrency' else 'corporate earnings growth' if asset_type == 'Stock' else 'underlying asset performance'}.
            
            We recommend maintaining a balanced approach with regular portfolio review and risk assessment.
            """
            
            # Overall Recommendation
            if total_return > 0.1 and volatility < 0.25:
                rating = 'BUY'
                rationale = 'Strong performance with manageable risk'
            elif total_return < -0.15 or volatility > 0.4:
                rating = 'HOLD'
                rationale = 'Elevated risk or poor performance warrant caution'
            else:
                rating = 'HOLD'
                rationale = 'Balanced risk-return profile'
            
            commentary['overall_recommendation'] = {
                'rating': rating,
                'target_price': current_price * (1.1 if rating == 'BUY' else 1.0 if rating == 'HOLD' else 0.9),
                'time_horizon': '12 months',
                'rationale': rationale,
                'key_points': [
                    f"Current valuation {'appears attractive' if rating == 'BUY' else 'reflects fair value' if rating == 'HOLD' else 'may be stretched'}",
                    f"Risk profile is {risk_level.lower()} based on historical volatility",
                    f"Macroeconomic environment {'supports' if rating == 'BUY' else 'is neutral for' if rating == 'HOLD' else 'challenges'} the investment thesis",
                    "Regular monitoring and risk management essential"
                ]
            }
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating comprehensive commentary: {e}")
            return {
                'executive_summary': 'Analysis in progress...',
                'investment_thesis': 'Under development...',
                'risk_assessment': 'Risk analysis pending...',
                'catalysts_outlook': 'Outlook under review...',
                'overall_recommendation': {'rating': 'HOLD', 'rationale': 'Analysis incomplete'}
            }
    
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
    
    def format_large_number(self, number: float) -> str:
        """Format large numbers with appropriate suffixes (public method for backward compatibility)."""
        return self._format_large_number(number)

def main():
    """Main application entry point."""
    try:
        dashboard = FinancialAnalystDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
