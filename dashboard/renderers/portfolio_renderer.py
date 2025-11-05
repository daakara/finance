"""
Portfolio and Forecasting Renderers - Handles portfolio strategy and forecasting rendering
Combined renderer for portfolio allocation and forecasting analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from analysis_engine import portfolio_engine, forecasting_engine
from ui_components import ErrorHandler

# ForecastingRenderer will be defined at the end of the file

logger = logging.getLogger(__name__)

class PortfolioRenderer:
    """Renders portfolio strategy and allocation analysis."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render portfolio strategy and allocation analysis."""
        try:
            st.subheader("💼 Portfolio Strategy & Allocation")
            
            # Get asset info
            asset_type = asset_data.get('asset_type', 'Unknown')
            asset_info = (asset_data.get('stock_info') or 
                         asset_data.get('etf_info') or 
                         asset_data.get('crypto_info', {}))
            asset_info['asset_type'] = asset_type
            
            # Perform portfolio analysis
            with st.spinner("Analyzing portfolio strategies..."):
                portfolio_analysis = portfolio_engine.analyze(asset_data['price_data'], asset_info)
            
            if 'error' in portfolio_analysis:
                st.error(f"Portfolio analysis error: {portfolio_analysis['error']}")
                return
            
            # Render portfolio analysis sections
            self._render_allocation_strategies(portfolio_analysis)
            self._render_stress_testing(portfolio_analysis)
            self._render_scenario_analysis(portfolio_analysis)
            
        except Exception as e:
            st.error(f"Error rendering portfolio strategy analysis: {str(e)}")
    
    def _render_allocation_strategies(self, portfolio_analysis: Dict[str, Any]):
        """Render allocation strategies section."""
        if 'allocation_strategies' not in portfolio_analysis:
            return
            
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
    
    def _render_stress_testing(self, portfolio_analysis: Dict[str, Any]):
        """Render stress testing results."""
        if 'stress_testing' not in portfolio_analysis:
            return
            
        st.markdown("### ⚠️ Stress Testing & Risk Scenarios")
        
        stress_results = portfolio_analysis['stress_testing']['stress_scenarios']
        
        # Create a comparison table
        stress_df = pd.DataFrame(stress_results).T
        stress_df['scenario_return'] = stress_df['scenario_return'].apply(lambda x: f"{x:.2%}")
        stress_df['probability'] = stress_df['probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(stress_df[['scenario_return', 'probability', 'recovery_time_estimate']], 
                   width='stretch')
        
        worst_case = portfolio_analysis['stress_testing']['worst_case_scenario']
        st.warning(f"**Worst Case Scenario:** {worst_case}")
    
    def _render_scenario_analysis(self, portfolio_analysis: Dict[str, Any]):
        """Render scenario analysis."""
        if 'scenario_analysis' not in portfolio_analysis:
            return
            
        st.markdown("### 🔮 Forward-Looking Scenarios")
        
        scenarios = portfolio_analysis['scenario_analysis']['scenarios']
        weighted_return = portfolio_analysis['scenario_analysis']['weighted_expected_return']
        
        st.metric("Scenario-Weighted Expected Return", f"{weighted_return:.1%}")
        
        # Show key scenarios
        scenario_df = pd.DataFrame(scenarios).T
        scenario_df['probability'] = scenario_df['probability'].apply(lambda x: f"{x:.1%}")
        scenario_df['expected_return'] = scenario_df['expected_return'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(scenario_df[['probability', 'expected_return', 'duration_months']], 
                   width='stretch')


# Redefine ForecastingRenderer properly now that dependencies are resolved
class ForecastingRenderer:
    """Renders forecasting and forward-looking analysis."""
    
    def render(self, asset_data: Dict[str, Any]):
        """Render forecasting and forward-looking analysis."""
        try:
            st.subheader("🔮 Forecasting & Forward-Looking Outlook")
            
            # Perform forecasting analysis
            with st.spinner("Generating forecasts and projections..."):
                forecast_analysis = forecasting_engine.analyze(asset_data['price_data'])
            
            if 'error' in forecast_analysis:
                st.error(f"Forecasting analysis error: {forecast_analysis['error']}")
                return
            
            # Render forecasting sections
            self._render_price_forecasting(forecast_analysis, asset_data)
            self._render_volatility_forecasting(forecast_analysis)
            self._render_seasonality_analysis(forecast_analysis)
            
        except Exception as e:
            st.error(f"Error rendering forecasting analysis: {str(e)}")
    
    def _render_price_forecasting(self, forecast_analysis: Dict[str, Any], asset_data: Dict[str, Any]):
        """Render price forecasting section."""
        if 'price_forecasting' not in forecast_analysis:
            return
            
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
                st.dataframe(model_df, width='stretch')
    
    def _render_volatility_forecasting(self, forecast_analysis: Dict[str, Any]):
        """Render volatility forecasting section."""
        if 'volatility_forecasting' not in forecast_analysis:
            return
            
        st.markdown("### 📊 Volatility Forecasting")
        
        vol_forecast = forecast_analysis['volatility_forecasting']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Volatility", f"{vol_forecast.get('current_volatility', 0):.2%}")
        with col2:
            st.metric("Historical Average", f"{vol_forecast.get('historical_average', 0):.2%}")
        with col3:
            st.metric("Volatility Regime", vol_forecast.get('volatility_regime', 'Unknown'))
    
    def _render_seasonality_analysis(self, forecast_analysis: Dict[str, Any]):
        """Render seasonality analysis section."""
        if 'seasonality_analysis' not in forecast_analysis:
            return
            
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
