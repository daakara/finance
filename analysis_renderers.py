"""
Analysis Renderers Module - Specialized rendering components for different analysis types
Breaks down large analysis rendering functions into focused, reusable components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

from ui_components import MetricsDisplayManager, ChartManager, ErrorHandler
from visualizer import financial_visualizer

logger = logging.getLogger(__name__)

class MacroeconomicRenderer:
    """Renders macroeconomic analysis components."""
    
    @staticmethod
    def render_monetary_policy_section(monetary_policy: Dict[str, Any]):
        """Render monetary policy impact section."""
        try:
            st.markdown("### 📈 Monetary Policy Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Fed Rate", f"{monetary_policy.get('current_fed_rate', 0):.2f}%")
                st.write(f"**Policy Stance:** {monetary_policy.get('policy_stance', 'Unknown')}")
            
            with col2:
                st.write(f"**Rate Sensitivity:** {monetary_policy.get('rate_sensitivity', 'Unknown')}")
                impact_score = monetary_policy.get('impact_score', 0)
                st.write(f"**Impact Score:** {impact_score:.2f}")
                
                # Progress bar for impact score
                normalized_score = (impact_score + 1) / 2  # Convert -1,1 to 0,1
                st.progress(normalized_score)
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Monetary Policy")
    
    @staticmethod
    def render_inflation_section(inflation_impact: Dict[str, Any]):
        """Render inflation impact section."""
        try:
            st.markdown("### 💰 Inflation Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current CPI", f"{inflation_impact.get('current_cpi', 0):.1f}%")
                st.metric("Core CPI", f"{inflation_impact.get('core_cpi', 0):.1f}%")
            
            with col2:
                st.write(f"**Hedge Quality:** {inflation_impact.get('inflation_hedge_quality', 'Unknown')}")
                st.write(f"**Real Return (Avg):** {inflation_impact.get('real_return_avg', 0):.3f}")
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Inflation Impact")
    
    @staticmethod
    def render_market_correlations_section(correlations: Dict[str, Any]):
        """Render market correlations section."""
        try:
            st.markdown("### 🔗 Market Correlations")
            
            correlation_data = correlations.get('correlations', {})
            
            # Display correlation metrics in a grid
            cols = st.columns(len(correlation_data))
            
            for i, (market, correlation) in enumerate(correlation_data.items()):
                with cols[i]:
                    st.metric(
                        f"{market.upper()}",
                        f"{correlation:.3f}",
                        help=f"Correlation with {market} returns"
                    )
            
            # Market relationship summary
            st.write(f"**Market Relationship:** {correlations.get('market_relationship', 'Unknown')}")
            st.write(f"**Diversification Benefit:** {correlations.get('diversification_benefit', 'Unknown')}")
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Market Correlations")

class FundamentalRenderer:
    """Renders fundamental analysis components."""
    
    @staticmethod
    def render_etf_fundamentals(analysis: Dict[str, Any]):
        """Render ETF-specific fundamental analysis."""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                FundamentalRenderer._render_etf_metrics(analysis.get('etf_metrics', {}))
                FundamentalRenderer._render_expense_analysis(analysis.get('expense_analysis', {}))
            
            with col2:
                FundamentalRenderer._render_etf_holdings(analysis.get('top_holdings', {}))
                FundamentalRenderer._render_sector_exposure(analysis.get('sector_exposure', {}))
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "ETF Fundamentals")
    
    @staticmethod
    def _render_etf_metrics(metrics: Dict[str, Any]):
        """Render ETF metrics subsection."""
        st.markdown("### 📊 ETF Metrics")
        
        if metrics:
            st.metric("Total Assets", FundamentalRenderer._format_large_number(metrics.get('total_assets', 0)))
            st.metric("Expense Ratio", f"{metrics.get('expense_ratio', 0):.3f}%")
            st.write(f"**Category:** {metrics.get('category', 'Unknown')}")
            st.write(f"**Fund Family:** {metrics.get('fund_family', 'Unknown')}")
    
    @staticmethod
    def _render_expense_analysis(expense_data: Dict[str, Any]):
        """Render expense analysis subsection."""
        st.markdown("### 💰 Expense Analysis")
        
        if expense_data:
            st.metric("Competitiveness", expense_data.get('competitiveness', 'Unknown'))
            st.metric("Annual Cost per $10K", f"${expense_data.get('annual_cost_per_10k', 0):.0f}")
    
    @staticmethod
    def _render_etf_holdings(holdings_data: Dict[str, Any]):
        """Render ETF holdings subsection."""
        st.markdown("### 🏢 Top Holdings")
        
        if holdings_data and 'top_holdings' in holdings_data:
            holdings_df = pd.DataFrame(holdings_data['top_holdings'])
            st.dataframe(holdings_df, width='stretch')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Concentration Risk", f"{holdings_data.get('concentration_risk', 0):.1f}%")
            with col2:
                st.metric("Diversification Score", f"{holdings_data.get('diversification_score', 0):.1f}")
    
    @staticmethod
    def _render_sector_exposure(sector_data: Dict[str, Any]):
        """Render sector exposure subsection."""
        st.markdown("### 📈 Sector Exposure")
        
        if sector_data and 'sector_allocations' in sector_data:
            sector_df = pd.DataFrame(
                list(sector_data['sector_allocations'].items()),
                columns=['Sector', 'Allocation %']
            )
            st.bar_chart(sector_df.set_index('Sector'))
    
    @staticmethod
    def render_crypto_fundamentals(analysis: Dict[str, Any]):
        """Render cryptocurrency-specific fundamental analysis."""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                FundamentalRenderer._render_onchain_metrics(analysis.get('onchain_metrics', {}))
                FundamentalRenderer._render_developer_activity(analysis.get('developer_activity', {}))
            
            with col2:
                FundamentalRenderer._render_adoption_metrics(analysis.get('adoption_metrics', {}))
                FundamentalRenderer._render_network_health(analysis.get('network_health', {}))
            
            # Fundamental score
            if 'fundamental_score' in analysis:
                FundamentalRenderer._render_fundamental_score(analysis['fundamental_score'])
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Crypto Fundamentals")
    
    @staticmethod
    def _render_onchain_metrics(metrics: Dict[str, Any]):
        """Render on-chain metrics subsection."""
        st.markdown("### ⛓️ On-Chain Metrics")
        
        if metrics:
            st.metric("Active Addresses", f"{metrics.get('active_addresses', 0):,}")
            st.metric("24h Volume", FundamentalRenderer._format_large_number(metrics.get('transaction_volume_24h', 0)))
            
            if 'hash_rate' in metrics:
                st.metric("Hash Rate", f"{metrics['hash_rate']:,} EH/s")
            if 'mvrv_ratio' in metrics:
                st.metric("MVRV Ratio", f"{metrics['mvrv_ratio']:.2f}")
    
    @staticmethod
    def _render_developer_activity(dev_data: Dict[str, Any]):
        """Render developer activity subsection."""
        st.markdown("### 👨‍💻 Developer Activity")
        
        if dev_data:
            st.metric("Monthly Commits", f"{dev_data.get('monthly_commits', 0):,}")
            st.metric("Active Developers", f"{dev_data.get('active_developers', 0):,}")
            st.metric("Development Health", dev_data.get('development_health', 'Unknown'))
            st.metric("Innovation Score", f"{dev_data.get('innovation_score', 0):.2f}")
    
    @staticmethod
    def _render_adoption_metrics(adoption: Dict[str, Any]):
        """Render adoption metrics subsection."""
        st.markdown("### 📈 Adoption Metrics")
        
        for key, value in adoption.items():
            if isinstance(value, (int, float)):
                if value > 1000000:
                    st.metric(key.replace('_', ' ').title(), f"{value:,.0f}")
                else:
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            else:
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    @staticmethod
    def _render_network_health(health: Dict[str, Any]):
        """Render network health subsection."""
        st.markdown("### 🏛️ Network Health")
        
        if health:
            st.metric("Network Security", health.get('network_security', 'Unknown'))
            st.metric("Congestion Level", health.get('congestion_level', 'Unknown'))
            st.metric("Decentralization Score", f"{health.get('decentralization_score', 0):.2f}")
    
    @staticmethod
    def _render_fundamental_score(score_data: Dict[str, Any]):
        """Render fundamental score subsection."""
        st.markdown("### 🎯 Overall Fundamental Score")
        
        score = score_data.get('fundamental_score', 0)
        rating = score_data.get('rating', 'Unknown')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Fundamental Score", f"{score:.2f}", rating)
        with col2:
            st.progress(score)
            st.write(f"**Rating:** {rating}")
    
    @staticmethod
    def _format_large_number(number: float) -> str:
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

class PortfolioRenderer:
    """Renders portfolio strategy and allocation components."""
    
    @staticmethod
    def render_allocation_strategies(strategies: Dict[str, Any]):
        """Render asset allocation strategies."""
        try:
            st.markdown("### 🎯 Asset Allocation Strategies")
            
            strategy_data = strategies.get('strategies', {})
            
            # Create tabs for different strategies
            strategy_names = list(strategy_data.keys())
            if strategy_names:
                tabs = st.tabs([name.title() for name in strategy_names])
                
                for tab, (strategy_name, strategy) in zip(tabs, strategy_data.items()):
                    with tab:
                        PortfolioRenderer._render_single_strategy(strategy_name, strategy)
                        
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Allocation Strategies")
    
    @staticmethod
    def _render_single_strategy(strategy_name: str, strategy: Dict[str, Any]):
        """Render a single allocation strategy."""
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
            allocation_df = pd.DataFrame(
                list(allocation_data.items()), 
                columns=['Asset Class', 'Allocation']
            )
            st.bar_chart(allocation_df.set_index('Asset Class'))
    
    @staticmethod
    def render_stress_testing(stress_data: Dict[str, Any]):
        """Render stress testing results."""
        try:
            st.markdown("### ⚠️ Stress Testing & Risk Scenarios")
            
            stress_scenarios = stress_data.get('stress_scenarios', {})
            
            if stress_scenarios:
                # Create comparison table
                stress_df = pd.DataFrame(stress_scenarios).T
                stress_df['scenario_return'] = stress_df['scenario_return'].apply(lambda x: f"{x:.2%}")
                stress_df['probability'] = stress_df['probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    stress_df[['scenario_return', 'probability', 'recovery_time_estimate']], 
                    width='stretch'
                )
                
                # Highlight worst case
                worst_case = stress_data.get('worst_case_scenario', 'Unknown')
                st.warning(f"**Worst Case Scenario:** {worst_case}")
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Stress Testing")

class ForecastingRenderer:
    """Renders forecasting and time series analysis components."""
    
    @staticmethod
    def render_price_forecasting(forecasting: Dict[str, Any], current_price: float):
        """Render price forecasting section."""
        try:
            st.markdown("### 📈 Price Forecasting")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Forecasting Horizon:** {forecasting.get('forecasting_horizon', '30 days')}")
                st.write(f"**Recommended Model:** {forecasting.get('model_recommendation', 'Ensemble')}")
            
            with col2:
                ForecastingRenderer._render_ensemble_forecast(forecasting, current_price)
            
            # Model comparison
            if 'models' in forecasting:
                ForecastingRenderer._render_model_comparison(forecasting['models'])
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Price Forecasting")
    
    @staticmethod
    def _render_ensemble_forecast(forecasting: Dict[str, Any], current_price: float):
        """Render ensemble forecast metrics."""
        if 'ensemble_forecast' in forecasting:
            ensemble = forecasting['ensemble_forecast']
            if 'forecast' in ensemble and len(ensemble['forecast']) > 0:
                forecast_price = ensemble['forecast'][-1]  # 30-day forecast
                price_change = (forecast_price - current_price) / current_price
                
                st.metric(
                    "30-Day Price Forecast", 
                    f"${forecast_price:.2f}", 
                    f"{price_change:.1%}"
                )
    
    @staticmethod
    def _render_model_comparison(models: Dict[str, Any]):
        """Render model comparison table."""
        st.markdown("#### 🎯 Model Comparison")
        
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
            st.dataframe(model_df, width='stretch')
    
    @staticmethod
    def render_volatility_forecasting(vol_forecast: Dict[str, Any]):
        """Render volatility forecasting section."""
        try:
            st.markdown("### 📊 Volatility Forecasting")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Volatility", f"{vol_forecast.get('current_volatility', 0):.2%}")
            with col2:
                st.metric("Historical Average", f"{vol_forecast.get('historical_average', 0):.2%}")
            with col3:
                st.metric("Volatility Regime", vol_forecast.get('volatility_regime', 'Unknown'))
                
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Volatility Forecasting")
    
    @staticmethod
    def render_seasonality_analysis(seasonality: Dict[str, Any]):
        """Render seasonality analysis section."""
        try:
            st.markdown("### 📅 Seasonal Patterns")
            
            col1, col2 = st.columns(2)
            
            if 'monthly_seasonality' in seasonality:
                monthly = seasonality['monthly_seasonality']
                
                with col1:
                    st.metric("Best Month", f"Month {monthly.get('best_month', 0)}")
                    st.metric("Worst Month", f"Month {monthly.get('worst_month', 0)}")
            
            if 'day_of_week_effects' in seasonality:
                dow = seasonality['day_of_week_effects']
                
                with col2:
                    st.metric("Best Day", dow.get('best_day', 'Unknown'))
                    st.metric("Worst Day", dow.get('worst_day', 'Unknown'))
                    
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Seasonality Analysis")

class CommentaryRenderer:
    """Renders analyst commentary and recommendations."""
    
    @staticmethod
    def render_comprehensive_commentary(commentary: Dict[str, Any]):
        """Render comprehensive analyst commentary."""
        try:
            # Create tabs for different aspects
            tabs = st.tabs(["Executive Summary", "Investment Thesis", "Risk Assessment", "Catalysts & Outlook"])
            
            with tabs[0]:
                st.markdown("#### 📋 Executive Summary")
                st.write(commentary.get('executive_summary', 'Analysis in progress...'))
            
            with tabs[1]:
                st.markdown("#### 💡 Investment Thesis")  
                st.write(commentary.get('investment_thesis', 'Investment thesis under development...'))
            
            with tabs[2]:
                st.markdown("#### ⚠️ Risk Assessment")
                st.write(commentary.get('risk_assessment', 'Risk analysis pending...'))
            
            with tabs[3]:
                st.markdown("#### 🚀 Catalysts & Outlook")
                st.write(commentary.get('catalysts_outlook', 'Market outlook analysis in progress...'))
            
            # Overall recommendation
            CommentaryRenderer._render_recommendation(commentary.get('overall_recommendation', {}))
            
        except Exception as e:
            ErrorHandler.handle_analysis_error(str(e), "Analyst Commentary")
    
    @staticmethod
    def _render_recommendation(recommendation: Dict[str, Any]):
        """Render overall recommendation section."""
        st.markdown("---")
        st.markdown("### 🎯 Overall Recommendation")
        
        col1, col2, col3 = st.columns(3)
        
        # Rating display
        with col1:
            rating = recommendation.get('rating', 'HOLD')
            if rating == 'BUY':
                st.success(f"**Rating:** {rating}")
            elif rating == 'SELL':
                st.error(f"**Rating:** {rating}")
            else:
                st.warning(f"**Rating:** {rating}")
        
        # Target price
        with col2:
            target_price = recommendation.get('target_price', 0)
            if target_price > 0:
                st.metric("Target Price", f"${target_price:.2f}")
        
        # Time horizon
        with col3:
            time_horizon = recommendation.get('time_horizon', '12 months')
            st.write(f"**Time Horizon:** {time_horizon}")
        
        # Key points
        if 'key_points' in recommendation:
            st.markdown("#### Key Points:")
            for point in recommendation['key_points']:
                st.write(f"• {point}")

# Export all renderer classes
__all__ = [
    'MacroeconomicRenderer',
    'FundamentalRenderer', 
    'PortfolioRenderer',
    'ForecastingRenderer',
    'CommentaryRenderer'
]
