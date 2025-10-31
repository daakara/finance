"""
Single Asset Analysis Workflow - Handles complete analysis of individual assets
Orchestrates technical analysis, fundamental analysis, and comprehensive reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from ..core.asset_data_manager import AssetDataManager
from ..analyzers.technical_analyzer import TechnicalAnalysisProcessor
from ..analyzers.financial_analyzer import FinancialMetricsAnalyzer
from ..analyzers.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from ..analyzers.candlestick_pattern_detector import CandlestickPatternDetector
from ..analyzers.chart_pattern_recognizer import ChartPatternRecognizer
from ..analyzers.volatility_forecaster import VolatilityForecaster
from ..visualizers.chart_visualizer import ChartVisualizer
from ..visualizers.advanced_chart_visualizer import AdvancedChartVisualizer
from ..visualizers.risk_visualizer import RiskVisualizer
from ..visualizers.pattern_visualizer import PatternVisualizer
from ..visualizers.metrics_display import MetricsDisplayManager

logger = logging.getLogger(__name__)

class SingleAssetWorkflow:
    """Orchestrates comprehensive analysis workflow for a single asset with Priority 1 enhancements."""
    
    def __init__(self):
        self.data_manager = AssetDataManager()
        self.technical_analyzer = TechnicalAnalysisProcessor()
        self.financial_analyzer = FinancialMetricsAnalyzer()
        self.chart_visualizer = ChartVisualizer()
        self.metrics_display = MetricsDisplayManager()
        
        # PRIORITY 1 ENHANCEMENTS
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        self.advanced_risk_analyzer = AdvancedRiskAnalyzer()
        self.advanced_chart_visualizer = AdvancedChartVisualizer()
        
        # PRIORITY 2 ENHANCEMENTS
        self.risk_visualizer = RiskVisualizer()
        
        # PRIORITY 3 ENHANCEMENTS
        self.candlestick_pattern_detector = CandlestickPatternDetector()
        self.chart_pattern_recognizer = ChartPatternRecognizer()
        self.volatility_forecaster = VolatilityForecaster()
        self.pattern_visualizer = PatternVisualizer()
        
        # PRIORITY 4 ENHANCEMENTS - Hidden Gems Scanner
        from ..analyzers.gem_screener import HiddenGemScreener
        from ..analyzers.historical_patterns import HistoricalPatternAnalyzer
        from ..data.gem_fetchers import MultiAssetDataPipeline
        
        self.gem_screener = HiddenGemScreener()
        self.historical_pattern_analyzer = HistoricalPatternAnalyzer()
        self.gem_data_pipeline = MultiAssetDataPipeline()
    
    def run_complete_analysis(self, symbol: str, asset_type: str = 'stock', 
                            time_period: str = '1y') -> Dict[str, Any]:
        """Run complete analysis workflow for a single asset."""
        try:
            st.write(f"📊 **Analyzing {symbol}** ({asset_type.upper()})")
            
            # Step 1: Fetch all required data
            with st.spinner("Fetching asset data..."):
                asset_data = self._fetch_comprehensive_data(symbol, asset_type, time_period)
            
            if 'error' in asset_data:
                st.error(f"Data fetch failed: {asset_data['error']}")
                return asset_data
            
            # Step 2: Perform analyses
            analysis_results = self._perform_comprehensive_analysis(asset_data, symbol)
            
            # Step 3: Generate visualizations and display results
            self._display_analysis_results(analysis_results, symbol, asset_type)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in single asset analysis workflow: {str(e)}")
            st.error(f"Analysis workflow failed: {str(e)}")
            return {'error': str(e)}
    
    def _fetch_comprehensive_data(self, symbol: str, asset_type: str, 
                                time_period: str) -> Dict[str, Any]:
        """Fetch all required data for comprehensive analysis."""
        try:
            # Fetch price and info data based on asset type
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
                return {'error': f'Unsupported asset type: {asset_type}'}
            
            # Validate data
            if price_data.empty:
                return {'error': 'No price data available'}
            
            return {
                'price_data': price_data,
                'info_data': info_data,
                'asset_type': asset_type,
                'time_period': time_period
            }
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data: {str(e)}")
            return {'error': str(e)}
    
    def _perform_comprehensive_analysis(self, asset_data: Dict[str, Any], 
                                      symbol: str) -> Dict[str, Any]:
        """Perform all types of analysis on the asset data with Priority 1 enhancements."""
        try:
            results = {
                'symbol': symbol,
                'asset_data': asset_data
            }
            
            price_data = asset_data['price_data']
            info_data = asset_data['info_data']
            
            # Enhanced Technical Analysis with Advanced Indicators
            with st.spinner("Performing enhanced technical analysis..."):
                technical_results = self.technical_analyzer.analyze_technical_data(price_data)
                results['technical_analysis'] = technical_results
            
            # PRIORITY 1 ENHANCEMENT: Multi-Timeframe Analysis
            with st.spinner("Analyzing multiple timeframes..."):
                multi_timeframe_results = self.multi_timeframe_analyzer.analyze_multi_timeframe(symbol)
                results['multi_timeframe_analysis'] = multi_timeframe_results
            
            # PRIORITY 2 ENHANCEMENT: Advanced Risk Analysis
            with st.spinner("Calculating advanced risk metrics..."):
                advanced_risk_results = self.advanced_risk_analyzer.analyze_comprehensive_risk(price_data)
                results['advanced_risk_analysis'] = advanced_risk_results
            
            # PRIORITY 3 ENHANCEMENTS: Pattern Recognition & Forecasting
            with st.spinner("Detecting candlestick patterns..."):
                candlestick_patterns = self.candlestick_pattern_detector.detect_all_patterns(price_data)
                results['candlestick_patterns'] = candlestick_patterns
            
            with st.spinner("Recognizing chart patterns..."):
                chart_patterns = self.chart_pattern_recognizer.detect_all_patterns(price_data)
                results['chart_patterns'] = chart_patterns
            
            with st.spinner("Generating volatility forecasts..."):
                volatility_forecast = self.volatility_forecaster.generate_volatility_forecast(price_data)
                results['volatility_forecast'] = volatility_forecast
            
            # Financial Analysis (for stocks and ETFs)
            if asset_data['asset_type'] in ['stock', 'etf'] and info_data:
                with st.spinner("Performing fundamental analysis..."):
                    financial_results = self.financial_analyzer.analyze_financials(info_data)
                    results['financial_analysis'] = financial_results
            
            # Enhanced performance metrics
            with st.spinner("Calculating performance metrics..."):
                performance_metrics = self._calculate_performance_metrics(price_data)
                results['performance_metrics'] = performance_metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing comprehensive analysis: {str(e)}")
            return {'error': str(e)}
    
    def _display_analysis_results(self, results: Dict[str, Any], symbol: str, asset_type: str):
        """Display all analysis results with Priority 1 enhancements."""
        try:
            # Asset Information Header
            self._display_asset_header(results, symbol, asset_type)
            
            # PRIORITY 1 ENHANCEMENT: Multi-Timeframe Analysis Summary
            if 'multi_timeframe_analysis' in results and 'error' not in results['multi_timeframe_analysis']:
                st.subheader("🎯 Multi-Timeframe Signal Confluence")
                self._display_multi_timeframe_analysis(results['multi_timeframe_analysis'])
            
            # Performance Overview
            if 'performance_metrics' in results:
                st.subheader("📈 Performance Overview")
                self._display_performance_overview(results['performance_metrics'])
            
            # PRIORITY 1 ENHANCEMENT: Advanced Risk Analysis
            if 'advanced_risk_analysis' in results and 'error' not in results['advanced_risk_analysis']:
                st.subheader("⚠️ Advanced Risk Analysis")
                self._display_advanced_risk_analysis(results['advanced_risk_analysis'])
            
            # Enhanced Technical Analysis Section
            if 'technical_analysis' in results and 'error' not in results['technical_analysis']:
                st.subheader("🔧 Enhanced Technical Analysis")
                self._display_enhanced_technical_analysis(results['technical_analysis'], symbol)
            
            # Financial Analysis Section (for stocks/ETFs)
            if 'financial_analysis' in results and 'error' not in results['financial_analysis']:
                st.subheader("💰 Fundamental Analysis")
                self._display_financial_analysis(results['financial_analysis'])
            
            # PRIORITY 3 ENHANCEMENTS: Pattern Recognition & Forecasting
            if 'candlestick_patterns' in results and 'error' not in results['candlestick_patterns']:
                st.subheader("🕯️ Candlestick Pattern Recognition")
                self._display_candlestick_patterns(results['candlestick_patterns'])
            
            if 'chart_patterns' in results and 'error' not in results['chart_patterns']:
                st.subheader("📈 Chart Pattern Recognition")
                self._display_chart_patterns(results['chart_patterns'])
            
            if 'volatility_forecast' in results and 'error' not in results['volatility_forecast']:
                st.subheader("🔮 Volatility Forecasting")
                self._display_volatility_forecast(results['volatility_forecast'])
            
            # PRIORITY 1-3 ENHANCEMENT: Advanced Charts Section
            st.subheader("📊 Advanced Interactive Charts")
            self._display_enhanced_charts(results, symbol)
            
            # PRIORITY 4 ENHANCEMENT: Hidden Gems Analysis
            if 'gem_score' in results and 'error' not in results['gem_score']:
                st.subheader("💎 Hidden Gems Analysis")
                self._display_hidden_gems_analysis(results['gem_score'], symbol)
            
        except Exception as e:
            logger.error(f"Error displaying analysis results: {str(e)}")
            st.error("Error displaying analysis results")
    
    def _display_asset_header(self, results: Dict[str, Any], symbol: str, asset_type: str):
        """Display asset information header."""
        try:
            asset_data = results.get('asset_data', {})
            info_data = asset_data.get('info_data', {})
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"## {symbol}")
                if info_data and 'longName' in info_data:
                    st.write(f"**{info_data['longName']}**")
                elif info_data and 'shortName' in info_data:
                    st.write(f"**{info_data['shortName']}**")
                
                if info_data and 'sector' in info_data:
                    st.write(f"Sector: {info_data['sector']}")
                if info_data and 'industry' in info_data:
                    st.write(f"Industry: {info_data['industry']}")
            
            with col2:
                price_data = asset_data.get('price_data', pd.DataFrame())
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
            
            with col3:
                if info_data and 'marketCap' in info_data:
                    market_cap = info_data['marketCap']
                    if market_cap >= 1e9:
                        cap_display = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        cap_display = f"${market_cap/1e6:.2f}M"
                    else:
                        cap_display = f"${market_cap:,.0f}"
                    st.metric("Market Cap", cap_display)
                    
        except Exception as e:
            logger.error(f"Error displaying asset header: {str(e)}")
    
    def _display_performance_overview(self, performance_metrics: Dict[str, Any]):
        """Display performance metrics overview."""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'total_return' in performance_metrics:
                    return_val = performance_metrics['total_return']
                    st.metric("Total Return", f"{return_val:.2f}%")
            
            with col2:
                if 'volatility' in performance_metrics:
                    vol_val = performance_metrics['volatility']
                    st.metric("Volatility", f"{vol_val:.2f}%")
            
            with col3:
                if 'sharpe_ratio' in performance_metrics:
                    sharpe_val = performance_metrics['sharpe_ratio']
                    st.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
            
            with col4:
                if 'max_drawdown' in performance_metrics:
                    dd_val = performance_metrics['max_drawdown']
                    st.metric("Max Drawdown", f"{dd_val:.2f}%")
                    
        except Exception as e:
            logger.error(f"Error displaying performance overview: {str(e)}")
    
    def _display_multi_timeframe_analysis(self, multi_tf_results: Dict[str, Any]):
        """Display multi-timeframe analysis with confluence signals."""
        try:
            # Display confluence signal prominently
            if 'confluence_signals' in multi_tf_results:
                confluence = multi_tf_results['confluence_signals']
                signal = confluence.get('signal', 'HOLD')
                confidence = confluence.get('confidence', 0.5)
                description = confluence.get('description', 'Mixed signals')
                
                # Create columns for the main signal display
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Signal color based on type
                    if signal in ['STRONG_BUY', 'BUY']:
                        st.success(f"🚀 **{signal}**")
                    elif signal in ['STRONG_SELL', 'SELL']:
                        st.error(f"📉 **{signal}**")
                    else:
                        st.info(f"⏸️ **{signal}**")
                
                with col2:
                    st.write(f"**Confidence:** {confidence:.1%}")
                    st.write(f"*{description}*")
                
                with col3:
                    # Display timeframe count
                    tf_count = confluence.get('timeframe_count', 0)
                    st.metric("Timeframes", f"{tf_count}/3")
            
            # Display individual timeframe signals
            if 'summary' in multi_tf_results and 'timeframes' in multi_tf_results['summary']:
                st.write("**Individual Timeframe Signals:**")
                
                timeframes = multi_tf_results['summary']['timeframes']
                tf_data = []
                
                for period, tf_info in timeframes.items():
                    tf_data.append({
                        'Timeframe': period.upper(),
                        'Trend': tf_info.get('trend', 'Unknown').replace('_', ' ').title(),
                        'Signal': tf_info.get('signal', 'Hold').replace('_', ' ').title(),
                        'Confidence': f"{tf_info.get('confidence', 0.5):.1%}"
                    })
                
                if tf_data:
                    df = pd.DataFrame(tf_data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    
        except Exception as e:
            logger.error(f"Error displaying multi-timeframe analysis: {str(e)}")
    
    def _display_advanced_risk_analysis(self, risk_results: Dict[str, Any]):
        """Display Priority 2 advanced risk analysis results."""
        try:
            if 'error' in risk_results:
                st.warning(f"Risk analysis error: {risk_results['error']}")
                return
            
            # PRIORITY 2: Advanced Risk Metrics Display
            st.markdown("### 📊 Value at Risk (VaR) & Tail Risk")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'advanced_metrics' in risk_results and 'VaR_95' in risk_results['advanced_metrics']:
                    var_95 = risk_results['advanced_metrics']['VaR_95']
                    st.metric("VaR (95%)", f"{var_95:.2f}%", help="Daily Value at Risk at 95% confidence")
            
            with col2:
                if 'advanced_metrics' in risk_results and 'VaR_99' in risk_results['advanced_metrics']:
                    var_99 = risk_results['advanced_metrics']['VaR_99']
                    st.metric("VaR (99%)", f"{var_99:.2f}%", help="Daily Value at Risk at 99% confidence")
            
            with col3:
                if 'advanced_metrics' in risk_results and 'CVaR_95' in risk_results['advanced_metrics']:
                    cvar_95 = risk_results['advanced_metrics']['CVaR_95']
                    st.metric("CVaR (95%)", f"{cvar_95:.2f}%", help="Conditional Value at Risk")
            
            with col4:
                if 'advanced_metrics' in risk_results and 'Tail_Ratio' in risk_results['advanced_metrics']:
                    tail_ratio = risk_results['advanced_metrics']['Tail_Ratio']
                    st.metric("Tail Ratio", f"{tail_ratio:.2f}", help="Right tail to left tail ratio")
            
            # PRIORITY 2: Advanced Risk Ratios
            st.markdown("### 📈 Advanced Risk Ratios")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'advanced_metrics' in risk_results and 'Calmar_Ratio' in risk_results['advanced_metrics']:
                    calmar = risk_results['advanced_metrics']['Calmar_Ratio']
                    st.metric("Calmar Ratio", f"{calmar:.3f}", help="Return to max drawdown ratio")
            
            with col2:
                if 'advanced_metrics' in risk_results and 'Sortino_Ratio' in risk_results['advanced_metrics']:
                    sortino = risk_results['advanced_metrics']['Sortino_Ratio']
                    st.metric("Sortino Ratio", f"{sortino:.3f}", help="Return to downside deviation ratio")
            
            with col3:
                if 'advanced_metrics' in risk_results and 'Omega_Ratio' in risk_results['advanced_metrics']:
                    omega = risk_results['advanced_metrics']['Omega_Ratio']
                    if omega == np.inf:
                        st.metric("Omega Ratio", "∞", help="Positive to negative return ratio")
                    else:
                        st.metric("Omega Ratio", f"{omega:.3f}", help="Positive to negative return ratio")
            
            with col4:
                if 'advanced_metrics' in risk_results and 'Pain_Ratio' in risk_results['advanced_metrics']:
                    pain = risk_results['advanced_metrics']['Pain_Ratio']
                    st.metric("Pain Ratio", f"{pain:.3f}", help="Return to pain index ratio")
            
            # PRIORITY 2: Drawdown Analysis
            if 'drawdown_analysis' in risk_results:
                dd_analysis = risk_results['drawdown_analysis']
                st.markdown("### 📉 Drawdown Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    max_dd = dd_analysis.get('max_drawdown', 0)
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                
                with col2:
                    avg_dd = dd_analysis.get('avg_drawdown', 0)
                    st.metric("Avg Drawdown", f"{avg_dd:.2f}%")
                
                with col3:
                    dd_freq = dd_analysis.get('drawdown_frequency', 0)
                    st.metric("Drawdown Events", f"{dd_freq}")
                
                with col4:
                    time_underwater = dd_analysis.get('time_underwater_pct', 0)
                    st.metric("Time Underwater", f"{time_underwater:.1f}%")
            
            # PRIORITY 2: Market Regime Analysis
            if 'regime_analysis' in risk_results:
                regime_analysis = risk_results['regime_analysis']
                st.markdown("### 🎯 Market Regime Detection")
                
                current_regime = regime_analysis.get('current_regime', 'Unknown')
                regime_display = current_regime.replace('_', ' ').title()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Bull_Market' in current_regime:
                        st.success(f"🐂 **Current Regime:** {regime_display}")
                    elif 'Bear_Market' in current_regime:
                        st.error(f"🐻 **Current Regime:** {regime_display}")
                    else:
                        st.info(f"📊 **Current Regime:** {regime_display}")
                
                with col2:
                    transitions = regime_analysis.get('regime_transitions', 0)
                    st.metric("Regime Changes", f"{transitions}")
                
                # Display regime statistics
                if 'regime_statistics' in regime_analysis:
                    regime_stats = regime_analysis['regime_statistics']
                    st.write("**Regime Statistics:**")
                    
                    regime_data = []
                    for regime_type, stats in regime_stats.items():
                        regime_data.append({
                            'Regime': regime_type.replace('_', ' ').title(),
                            'Frequency': stats.get('frequency', 0),
                            'Avg Volatility': f"{stats.get('avg_volatility', 0)*100:.1f}%",
                            'Avg Returns': f"{stats.get('avg_returns', 0)*100:.1f}%"
                        })
                    
                    if regime_data:
                        regime_df = pd.DataFrame(regime_data)
                        st.dataframe(regime_df, hide_index=True, use_container_width=True)
            
            # PRIORITY 2: Tail Risk Analysis
            if 'tail_risk' in risk_results:
                tail_risk = risk_results['tail_risk']
                st.markdown("### ⚡ Tail Risk Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    extreme_neg = tail_risk.get('extreme_negative_events', 0)
                    st.metric("Extreme Negative Events", f"{extreme_neg}")
                
                with col2:
                    extreme_pos = tail_risk.get('extreme_positive_events', 0)
                    st.metric("Extreme Positive Events", f"{extreme_pos}")
                
                with col3:
                    worst_day = tail_risk.get('worst_single_day', 0)
                    st.metric("Worst Single Day", f"{worst_day:.2f}%")
                
                with col4:
                    best_day = tail_risk.get('best_single_day', 0)
                    st.metric("Best Single Day", f"{best_day:.2f}%")
            
            # Generate and display risk insights
            risk_insights = self.advanced_risk_analyzer.generate_risk_insights(risk_results)
            if risk_insights:
                st.markdown("### 💡 Risk Insights")
                for insight in risk_insights[:5]:  # Show top 5 insights
                    st.info(insight)
                    
        except Exception as e:
            logger.error(f"Error displaying advanced risk analysis: {str(e)}")
            st.error("Error displaying risk analysis")
    
    def _display_enhanced_technical_analysis(self, technical_results: Dict[str, Any], symbol: str):
        """Display enhanced technical analysis with confluence scoring."""
        try:
            # Display confluence signal prominently
            if 'score' in technical_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confluence_score = technical_results['score']
                    signal = technical_results.get('signal', 'HOLD')
                    
                    if signal in ['STRONG_BUY', 'BUY']:
                        st.success(f"**Signal: {signal}**")
                    elif signal in ['STRONG_SELL', 'SELL']:
                        st.error(f"**Signal: {signal}**")
                    else:
                        st.info(f"**Signal: {signal}**")
                
                with col2:
                    st.metric("Confluence Score", f"{confluence_score}/100")
                
                with col3:
                    confidence = technical_results.get('confidence', 50)
                    st.metric("Confidence", f"{confidence:.0f}%")
            
            # Display individual indicator signals
            if 'signals' in technical_results:
                signals = technical_results['signals']
                
                st.write("**Individual Indicator Signals:**")
                
                # Create a more organized display of signals
                indicator_groups = {
                    'Momentum': ['RSI', 'Stoch_RSI', 'Williams_R', 'CCI', 'MFI'],
                    'Trend': ['Moving_Average', 'MACD', 'ADX', 'PSAR'],
                    'Volume': ['CMF'],
                    'Volatility': ['Bollinger']
                }
                
                for group_name, indicators in indicator_groups.items():
                    group_signals = {k: v for k, v in signals.items() if k in indicators}
                    if group_signals:
                        st.write(f"**{group_name} Indicators:**")
                        cols = st.columns(min(len(group_signals), 4))
                        
                        for i, (indicator, signal) in enumerate(group_signals.items()):
                            with cols[i % len(cols)]:
                                # Color code based on signal
                                if 'bullish' in signal.lower() or 'buy' in signal.lower() or 'oversold' in signal.lower():
                                    st.success(f"{indicator}: {signal}")
                                elif 'bearish' in signal.lower() or 'sell' in signal.lower() or 'overbought' in signal.lower():
                                    st.error(f"{indicator}: {signal}")
                                else:
                                    st.info(f"{indicator}: {signal}")
            
            # Display traditional trend analysis
            if 'trend_analysis' in technical_results:
                st.write("**Trend Analysis:**")
                trend_data = technical_results['trend_analysis']
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'primary_trend' in trend_data:
                        st.write(f"Primary Trend: **{trend_data['primary_trend']}**")
                
                with col2:
                    if 'momentum' in trend_data:
                        st.write(f"Momentum: **{trend_data['momentum']}**")
                        
        except Exception as e:
            logger.error(f"Error displaying enhanced technical analysis: {str(e)}")
    
    def _display_enhanced_charts(self, results: Dict[str, Any], symbol: str):
        """Display enhanced charts with Priority 1 visualizations."""
        try:
            asset_data = results.get('asset_data', {})
            price_data = asset_data.get('price_data', pd.DataFrame())
            
            if price_data.empty:
                st.warning("No price data available for charts")
                return
            
            # Enhanced Chart tabs - Priority 1-4 with pattern recognition, forecasting, and hidden gems analysis
            (tech_chart_tab, confluence_tab, multi_tf_tab, risk_dashboard_tab, var_tab, drawdown_tab, 
             regime_tab, tail_risk_tab, candlestick_tab, chart_patterns_tab, volatility_forecast_tab, pattern_dashboard_tab, hidden_gems_tab) = st.tabs([
                "Enhanced Technical Chart", 
                "Signal Confluence", 
                "Multi-Timeframe",
                "Risk Dashboard",
                "VaR Analysis",
                "Drawdown Analysis", 
                "Market Regimes",
                "Tail Risk",
                "Candlestick Patterns",
                "Chart Patterns", 
                "Volatility Forecast",
                "Pattern Dashboard",
                "💎 Hidden Gems Analysis"
            ])
            
            with tech_chart_tab:
                # Enhanced technical chart with all advanced indicators
                tech_data = None
                signals = {}
                
                if 'technical_analysis' in results and 'indicators' in results['technical_analysis']:
                    tech_data = results['technical_analysis']['indicators']
                    signals = results['technical_analysis'].get('signals', {})
                
                if tech_data is not None:
                    enhanced_chart = self.advanced_chart_visualizer.create_enhanced_technical_chart(
                        price_data, tech_data, signals
                    )
                    st.plotly_chart(enhanced_chart, use_container_width=True)
                else:
                    # Fallback to basic chart
                    basic_chart = self.chart_visualizer.create_candlestick_chart(
                        price_data, symbol, tech_data
                    )
                    st.plotly_chart(basic_chart, use_container_width=True)
            
            with confluence_tab:
                # Signal confluence meter
                if 'technical_analysis' in results:
                    confluence_meter = self.advanced_chart_visualizer.create_confluence_meter(
                        results['technical_analysis']
                    )
                    st.plotly_chart(confluence_meter, use_container_width=True)
                else:
                    st.info("Technical analysis required for confluence visualization")
            
            with multi_tf_tab:
                # Multi-timeframe analysis chart
                if 'multi_timeframe_analysis' in results and 'timeframe_data' in results['multi_timeframe_analysis']:
                    multi_tf_data = results['multi_timeframe_analysis']['timeframe_data']
                    multi_tf_analysis = results['multi_timeframe_analysis']['timeframe_analysis']
                    
                    multi_tf_chart = self.advanced_chart_visualizer.create_multi_timeframe_chart(
                        multi_tf_data, multi_tf_analysis
                    )
                    st.plotly_chart(multi_tf_chart, use_container_width=True)
                else:
                    st.info("Multi-timeframe data not available")
            
            # PRIORITY 2: Advanced Risk Analysis Tabs
            with risk_dashboard_tab:
                # Risk metrics dashboard
                if 'advanced_risk_analysis' in results:
                    risk_dashboard = self.risk_visualizer.create_risk_metrics_dashboard(
                        results['advanced_risk_analysis']
                    )
                    st.plotly_chart(risk_dashboard, use_container_width=True)
                else:
                    st.info("Advanced risk analysis required for risk dashboard")
            
            with var_tab:
                # Value at Risk analysis
                if 'advanced_risk_analysis' in results:
                    var_chart = self.risk_visualizer.create_var_analysis_chart(
                        price_data, results['advanced_risk_analysis']
                    )
                    st.plotly_chart(var_chart, use_container_width=True)
                else:
                    st.info("Advanced risk analysis required for VaR visualization")
            
            with drawdown_tab:
                # Drawdown analysis
                drawdown_chart = self.risk_visualizer.create_drawdown_analysis_chart(price_data)
                st.plotly_chart(drawdown_chart, use_container_width=True)
            
            with regime_tab:
                # Market regime detection
                if 'advanced_risk_analysis' in results and 'regime_analysis' in results['advanced_risk_analysis']:
                    regime_chart = self.risk_visualizer.create_regime_detection_chart(
                        price_data, results['advanced_risk_analysis']['regime_analysis']
                    )
                    st.plotly_chart(regime_chart, use_container_width=True)
                else:
                    st.info("Regime analysis data required for regime visualization")
            
            with tail_risk_tab:
                # Tail risk analysis
                if 'advanced_risk_analysis' in results and 'tail_risk' in results['advanced_risk_analysis']:
                    tail_risk_chart = self.risk_visualizer.create_tail_risk_analysis_chart(
                        price_data, results['advanced_risk_analysis']['tail_risk']
                    )
                    st.plotly_chart(tail_risk_chart, use_container_width=True)
                else:
                    st.info("Tail risk analysis required for tail risk visualization")
            
            # PRIORITY 3: Pattern Recognition & Forecasting Charts
            with candlestick_tab:
                # Candlestick patterns visualization
                if 'candlestick_patterns' in results and 'error' not in results['candlestick_patterns']:
                    candlestick_chart = self.pattern_visualizer.create_candlestick_pattern_chart(
                        price_data, results['candlestick_patterns']
                    )
                    st.plotly_chart(candlestick_chart, use_container_width=True)
                else:
                    st.info("Candlestick pattern analysis required for pattern visualization")
            
            with chart_patterns_tab:
                # Chart patterns visualization
                if 'chart_patterns' in results and 'error' not in results['chart_patterns']:
                    chart_pattern_viz = self.pattern_visualizer.create_chart_pattern_visualization(
                        price_data, results['chart_patterns']
                    )
                    st.plotly_chart(chart_pattern_viz, use_container_width=True)
                else:
                    st.info("Chart pattern analysis required for pattern visualization")
            
            with volatility_forecast_tab:
                # Volatility forecasting visualization
                if 'volatility_forecast' in results and 'error' not in results['volatility_forecast']:
                    volatility_chart = self.pattern_visualizer.create_volatility_forecast_chart(
                        price_data, results['volatility_forecast']
                    )
                    st.plotly_chart(volatility_chart, use_container_width=True)
                else:
                    st.info("Volatility forecast analysis required for forecast visualization")
            
            with pattern_dashboard_tab:
                # Comprehensive pattern dashboard
                candlestick_data = results.get('candlestick_patterns', {})
                chart_pattern_data = results.get('chart_patterns', {})
                volatility_data = results.get('volatility_forecast', {})
                
                if (candlestick_data and 'error' not in candlestick_data and
                    chart_pattern_data and 'error' not in chart_pattern_data and
                    volatility_data and 'error' not in volatility_data):
                    
                    pattern_dashboard = self.pattern_visualizer.create_pattern_summary_dashboard(
                        candlestick_data, chart_pattern_data, volatility_data
                    )
                    st.plotly_chart(pattern_dashboard, use_container_width=True)
                else:
                    st.info("Complete pattern analysis required for pattern dashboard")
            
            with hidden_gems_tab:
                # Priority 4: Hidden Gems Analysis
                st.subheader("💎 Hidden Gems Analysis")
                st.markdown("*Advanced multi-bagger opportunity assessment*")
                
                try:
                    # Run Hidden Gems analysis
                    gems_analysis = self._analyze_hidden_gem_potential(symbol, results)
                    
                    if gems_analysis and 'error' not in gems_analysis:
                        self._display_hidden_gems_analysis(gems_analysis, symbol)
                    else:
                        error_msg = gems_analysis.get('error', 'Unknown error') if gems_analysis else 'Analysis failed'
                        st.error(f"Hidden Gems analysis failed: {error_msg}")
                        
                except Exception as e:
                    logger.error(f"Error in Hidden Gems analysis for {symbol}: {e}")
                    st.error("Hidden Gems analysis temporarily unavailable")
                
        except Exception as e:
            logger.error(f"Error displaying enhanced charts: {str(e)}")
            st.error("Error generating enhanced charts")
    
    def _display_financial_analysis(self, financial_results: Dict[str, Any]):
        """Display financial analysis results."""
        try:
            # Display in tabs for better organization
            key_tab, ratios_tab, val_tab, health_tab = st.tabs([
                "Key Metrics", "Financial Ratios", "Valuation", "Health Score"
            ])
            
            with key_tab:
                if 'key_metrics' in financial_results:
                    self.metrics_display.display_key_metrics(financial_results['key_metrics'])
            
            with ratios_tab:
                if 'ratios' in financial_results:
                    self.metrics_display.display_financial_ratios(financial_results['ratios'])
            
            with val_tab:
                if 'valuation' in financial_results:
                    self.metrics_display.display_valuation_analysis(financial_results['valuation'])
            
            with health_tab:
                if 'health_score' in financial_results:
                    self.metrics_display.display_health_score(financial_results['health_score'])
                    
        except Exception as e:
            logger.error(f"Error displaying financial analysis: {str(e)}")
    
    def _display_charts(self, results: Dict[str, Any], symbol: str):
        """Display interactive charts."""
        try:
            asset_data = results.get('asset_data', {})
            price_data = asset_data.get('price_data', pd.DataFrame())
            
            if price_data.empty:
                st.warning("No price data available for charts")
                return
            
            # Chart tabs
            tech_chart_tab, volume_tab = st.tabs(["Technical Chart", "Volume Analysis"])
            
            with tech_chart_tab:
                tech_data = None
                if 'technical_analysis' in results and 'indicators' in results['technical_analysis']:
                    tech_data = results['technical_analysis']['indicators']
                
                candlestick_chart = self.chart_visualizer.create_candlestick_chart(
                    price_data, symbol, tech_data
                )
                st.plotly_chart(candlestick_chart, use_container_width=True)
            
            with volume_tab:
                volume_chart = self.chart_visualizer.create_volume_analysis_chart(
                    price_data, symbol
                )
                st.plotly_chart(volume_chart, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error displaying charts: {str(e)}")
            st.error("Error generating charts")
    
    def _calculate_performance_metrics(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        try:
            if price_data.empty or 'Close' not in price_data.columns:
                return {'error': 'Insufficient price data'}
            
            close_prices = price_data['Close']
            
            # Total return
            total_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100
            
            # Daily returns for other calculations
            daily_returns = close_prices.pct_change().dropna()
            
            # Volatility (annualized)
            volatility = daily_returns.std() * (252 ** 0.5) * 100
            
            # Sharpe ratio (assuming 0% risk-free rate)
            avg_return = daily_returns.mean() * 252
            sharpe_ratio = avg_return / (daily_returns.std() * (252 ** 0.5)) if daily_returns.std() != 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}
    
    def _display_candlestick_patterns(self, pattern_results: Dict[str, Any]):
        """Display candlestick pattern analysis results"""
        try:
            if 'error' in pattern_results:
                st.warning(f"Candlestick pattern analysis error: {pattern_results['error']}")
                return
            
            # Pattern summary
            if 'summary' in pattern_results:
                summary = pattern_results['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total = summary.get('total_patterns', 0)
                    st.metric("Total Patterns", f"{total}")
                
                with col2:
                    bullish = summary.get('bullish_patterns', 0)
                    st.metric("Bullish Patterns", f"{bullish}", delta=None)
                
                with col3:
                    bearish = summary.get('bearish_patterns', 0)
                    st.metric("Bearish Patterns", f"{bearish}", delta=None)
                
                with col4:
                    high_rel = summary.get('high_reliability_patterns', 0)
                    st.metric("High Reliability", f"{high_rel}")
                
                # Signal strength
                signal_strength = summary.get('signal_strength', 'neutral')
                if signal_strength == 'bullish':
                    st.success(f"🐂 **Overall Signal: {signal_strength.upper()}** - Bullish pattern dominance")
                elif signal_strength == 'bearish':
                    st.error(f"🐻 **Overall Signal: {signal_strength.upper()}** - Bearish pattern dominance")
                else:
                    st.info(f"⚖️ **Overall Signal: {signal_strength.upper()}** - Balanced pattern distribution")
            
            # Pattern insights
            if 'pattern_insights' in pattern_results:
                st.write("**Pattern Insights:**")
                for insight in pattern_results['pattern_insights'][:5]:
                    st.info(insight)
            
            # Recent patterns
            if 'summary' in pattern_results and 'recent_patterns' in pattern_results['summary']:
                recent = pattern_results['summary']['recent_patterns']
                if recent:
                    st.write("**Recent Patterns (Last 30 Days):**")
                    for pattern in recent[:5]:
                        reliability_emoji = "⭐" if pattern.reliability > 80 else "🔸" if pattern.reliability > 65 else "🔹"
                        signal_emoji = "🟢" if pattern.signal_type == 'bullish' else "🔴" if pattern.signal_type == 'bearish' else "🟡"
                        st.write(f"{reliability_emoji} {signal_emoji} **{pattern.name.replace('_', ' ').title()}** - {pattern.reliability:.0f}% reliability")
                        
        except Exception as e:
            logger.error(f"Error displaying candlestick patterns: {str(e)}")
    
    def _display_chart_patterns(self, pattern_results: Dict[str, Any]):
        """Display chart pattern analysis results"""
        try:
            if 'error' in pattern_results:
                st.warning(f"Chart pattern analysis error: {pattern_results['error']}")
                return
            
            # Pattern summary
            if 'summary' in pattern_results:
                summary = pattern_results['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total = summary.get('total_patterns', 0)
                    st.metric("Total Patterns", f"{total}")
                
                with col2:
                    reversal = summary.get('reversal_patterns', 0)
                    st.metric("Reversal Patterns", f"{reversal}")
                
                with col3:
                    continuation = summary.get('continuation_patterns', 0)
                    st.metric("Continuation Patterns", f"{continuation}")
                
                with col4:
                    high_conf = summary.get('high_confidence_patterns', 0)
                    st.metric("High Confidence", f"{high_conf}")
            
            # Support and resistance levels
            if 'key_levels' in pattern_results:
                levels = pattern_results['key_levels']
                if levels:
                    st.write("**Key Support & Resistance Levels:**")
                    
                    level_data = []
                    for level in levels[:5]:
                        level_data.append({
                            'Type': level['type'].title(),
                            'Price': f"${level['price']:.2f}",
                            'Strength': level['strength'],
                            'Confidence': f"{level['confidence']:.0f}%"
                        })
                    
                    if level_data:
                        df = pd.DataFrame(level_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Pattern insights
            if 'summary' in pattern_results and 'pattern_insights' in pattern_results['summary']:
                insights = pattern_results['summary']['pattern_insights']
                if insights:
                    st.write("**Chart Pattern Insights:**")
                    for insight in insights[:5]:
                        st.info(insight)
            
            # Trend analysis
            if 'summary' in pattern_results and 'trend_analysis' in pattern_results['summary']:
                trend_analysis = pattern_results['summary']['trend_analysis']
                if trend_analysis:
                    st.write("**Trend Analysis:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        short_term = trend_analysis.get('short_term', {})
                        if short_term:
                            direction = short_term.get('direction', 'neutral')
                            strength = short_term.get('strength', 'weak')
                            change = short_term.get('change_pct', 0)
                            
                            if direction == 'bullish':
                                st.success(f"📈 **Short-term:** {strength} {direction} ({change:+.1f}%)")
                            else:
                                st.error(f"📉 **Short-term:** {strength} {direction} ({change:+.1f}%)")
                    
                    with col2:
                        medium_term = trend_analysis.get('medium_term', {})
                        if medium_term:
                            direction = medium_term.get('direction', 'neutral')
                            strength = medium_term.get('strength', 'weak')
                            change = medium_term.get('change_pct', 0)
                            
                            if direction == 'bullish':
                                st.success(f"📈 **Medium-term:** {strength} {direction} ({change:+.1f}%)")
                            else:
                                st.error(f"📉 **Medium-term:** {strength} {direction} ({change:+.1f}%)")
                    
                    with col3:
                        long_term = trend_analysis.get('long_term', {})
                        if long_term:
                            direction = long_term.get('direction', 'neutral')
                            strength = long_term.get('strength', 'weak')
                            change = long_term.get('change_pct', 0)
                            
                            if direction == 'bullish':
                                st.success(f"📈 **Long-term:** {strength} {direction} ({change:+.1f}%)")
                            else:
                                st.error(f"📉 **Long-term:** {strength} {direction} ({change:+.1f}%)")
                                
        except Exception as e:
            logger.error(f"Error displaying chart patterns: {str(e)}")
    
    def _display_volatility_forecast(self, forecast_results: Dict[str, Any]):
        """Display volatility forecasting results"""
        try:
            if 'error' in forecast_results:
                st.warning(f"Volatility forecast error: {forecast_results['error']}")
                return
            
            # Current volatility metrics
            if 'current_metrics' in forecast_results:
                metrics = forecast_results['current_metrics']
                st.write("**Current Volatility Metrics:**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_vol = metrics.get('current_volatility', 0) * 100
                    st.metric("Current Volatility", f"{current_vol:.1f}%")
                
                with col2:
                    vol_20d = metrics.get('vol_20d', 0) * 100
                    st.metric("20-Day Volatility", f"{vol_20d:.1f}%")
                
                with col3:
                    vol_percentile = metrics.get('volatility_percentile', 50)
                    st.metric("Volatility Percentile", f"{vol_percentile:.0f}%")
                
                with col4:
                    persistence = metrics.get('volatility_persistence', 0)
                    st.metric("Persistence", f"{persistence:.3f}")
            
            # Ensemble forecast
            if 'ensemble_forecast' in forecast_results:
                forecast = forecast_results['ensemble_forecast']
                st.write("**Volatility Forecast:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_forecast = np.mean(forecast.forecasted_volatility) * 100
                    st.metric("Avg Forecast (30d)", f"{avg_forecast:.1f}%")
                
                with col2:
                    trend = forecast.volatility_trend
                    trend_emoji = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                    st.metric("Forecast Trend", f"{trend_emoji} {trend.title()}")
                
                with col3:
                    horizon = forecast.forecast_horizon
                    st.metric("Forecast Horizon", f"{horizon} days")
            
            # Regime analysis
            if 'regime_analysis' in forecast_results:
                regime_analysis = forecast_results['regime_analysis']
                st.write("**Volatility Regime Analysis:**")
                
                current_regime = regime_analysis.get('current_regime', 'medium')
                regime_desc = regime_analysis.get('regime_description', '')
                
                if current_regime == 'extreme':
                    st.error(f"🚨 **Current Regime:** {current_regime.upper()}")
                elif current_regime == 'high':
                    st.warning(f"⚠️ **Current Regime:** {current_regime.upper()}")
                elif current_regime == 'low':
                    st.success(f"✅ **Current Regime:** {current_regime.upper()}")
                else:
                    st.info(f"📊 **Current Regime:** {current_regime.upper()}")
                
                st.write(f"*{regime_desc}*")
                
                # Regime frequencies
                if 'regime_frequencies' in regime_analysis:
                    freq = regime_analysis['regime_frequencies']
                    st.write("**Historical Regime Frequencies:**")
                    
                    freq_data = []
                    for regime, frequency in freq.items():
                        freq_data.append({
                            'Regime': regime.title(),
                            'Frequency': f"{frequency*100:.1f}%"
                        })
                    
                    if freq_data:
                        df = pd.DataFrame(freq_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Model availability
            if 'model_availability' in forecast_results:
                availability = forecast_results['model_availability']
                garch_available = availability.get('garch_models', False)
                
                if garch_available:
                    st.success("🎯 **Advanced GARCH models available** - High-quality volatility forecasts")
                else:
                    st.info("📈 **Using historical models** - Consider installing 'arch' package for advanced forecasts")
            
            # Forecast insights
            if 'forecast_insights' in forecast_results:
                insights = forecast_results['forecast_insights']
                if insights:
                    st.write("**Volatility Insights:**")
                    for insight in insights[:5]:
                        st.info(insight)
                        
        except Exception as e:
            logger.error(f"Error displaying volatility forecast: {str(e)}")
    
    def _analyze_hidden_gem_potential(self, symbol: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze hidden gem potential for the given symbol.
        
        Args:
            symbol: Asset symbol to analyze
            analysis_results: Existing analysis results from workflow
            
        Returns:
            Dictionary with hidden gem analysis results
        """
        try:
            st.info("🔍 Running comprehensive hidden gem analysis...")
            
            # Get comprehensive data using the gem data pipeline
            comprehensive_data = self.gem_data_pipeline.get_comprehensive_data(symbol, 'stock')
            
            if 'error' in comprehensive_data:
                return {'error': f"Data fetch failed: {comprehensive_data['error']}"}
            
            # Enhance data with existing workflow results
            enhanced_data = self._enhance_data_with_workflow_results(comprehensive_data, analysis_results)
            
            # Calculate gem score
            gem_score = self.gem_screener.calculate_composite_score(symbol, enhanced_data)
            
            # Get historical pattern analysis
            pattern_insights = self.historical_pattern_analyzer.get_pattern_insights(symbol, enhanced_data)
            
            # Combine results
            gems_analysis = {
                'symbol': symbol,
                'gem_score': gem_score,
                'pattern_insights': pattern_insights,
                'comprehensive_data': enhanced_data,
                'analysis_timestamp': pd.Timestamp.now(),
                'multi_bagger_potential': self._assess_multi_bagger_potential(gem_score, pattern_insights)
            }
            
            return gems_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing hidden gem potential for {symbol}: {e}")
            return {'error': str(e)}
    
    def _enhance_data_with_workflow_results(self, comprehensive_data: Dict[str, Any], 
                                          analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance comprehensive data with existing workflow analysis results"""
        try:
            # Add technical analysis results
            if 'technical_analysis' in analysis_results:
                tech_analysis = analysis_results['technical_analysis']
                comprehensive_data['enhanced_technical'] = {
                    'trend_strength': tech_analysis.get('trend_analysis', {}).get('trend_strength', 0),
                    'momentum_score': tech_analysis.get('momentum', {}).get('overall_momentum', 0),
                    'support_resistance': tech_analysis.get('support_resistance', {}),
                    'volatility_analysis': tech_analysis.get('volatility', {})
                }
            
            # Add risk analysis results
            if 'advanced_risk_analysis' in analysis_results:
                risk_analysis = analysis_results['advanced_risk_analysis']
                comprehensive_data['enhanced_risk'] = {
                    'var_analysis': risk_analysis.get('var_analysis', {}),
                    'drawdown_metrics': risk_analysis.get('drawdown_analysis', {}),
                    'regime_analysis': risk_analysis.get('regime_analysis', {})
                }
            
            # Add pattern analysis results
            if 'candlestick_patterns' in analysis_results:
                comprehensive_data['candlestick_patterns'] = analysis_results['candlestick_patterns']
            
            if 'chart_patterns' in analysis_results:
                comprehensive_data['chart_patterns'] = analysis_results['chart_patterns']
            
            # Add volatility forecasting results
            if 'volatility_forecast' in analysis_results:
                comprehensive_data['volatility_forecast'] = analysis_results['volatility_forecast']
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error enhancing data with workflow results: {e}")
            return comprehensive_data
    
    def _assess_multi_bagger_potential(self, gem_score, pattern_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess multi-bagger potential based on gem score and pattern analysis"""
        try:
            assessment = {
                'probability': 0.0,
                'confidence': 'Low',
                'time_horizon': '24+ months',
                'upside_potential': '100%+',
                'key_catalysts': [],
                'risk_factors': []
            }
            
            # Base assessment on composite score
            composite_score = gem_score.composite_score
            
            if composite_score >= 80:
                assessment['probability'] = 0.75
                assessment['confidence'] = 'High'
                assessment['upside_potential'] = '300%+'
            elif composite_score >= 65:
                assessment['probability'] = 0.55
                assessment['confidence'] = 'Medium-High'
                assessment['upside_potential'] = '200%+'
            elif composite_score >= 50:
                assessment['probability'] = 0.35
                assessment['confidence'] = 'Medium'
                assessment['upside_potential'] = '100%+'
            else:
                assessment['probability'] = 0.15
                assessment['confidence'] = 'Low'
                assessment['upside_potential'] = '50%+'
            
            # Enhance with pattern analysis
            if pattern_insights and 'replication_analysis' in pattern_insights:
                replication = pattern_insights['replication_analysis']
                pattern_probability = replication.get('replication_probability', 0)
                
                # Weight pattern probability
                assessment['probability'] = (assessment['probability'] * 0.7) + (pattern_probability * 0.3)
                
                # Update time horizon based on historical patterns
                avg_duration = replication.get('avg_pattern_duration', 24)
                if avg_duration < 12:
                    assessment['time_horizon'] = '6-12 months'
                elif avg_duration < 18:
                    assessment['time_horizon'] = '12-18 months'
                else:
                    assessment['time_horizon'] = '18-24 months'
                
                # Extract catalysts and risks
                assessment['key_catalysts'] = replication.get('success_factors', [])[:3]
                assessment['risk_factors'] = replication.get('risk_factors', [])[:3]
            
            # Add general catalysts based on scores  
            if gem_score.sector_score >= 70:
                assessment['key_catalysts'].append("Strong emerging sector tailwinds")
            
            if gem_score.catalyst_score >= 70:
                assessment['key_catalysts'].append("Multiple near-term catalysts identified")
            
            if gem_score.technical_score >= 70:
                assessment['key_catalysts'].append("Strong technical setup with accumulation patterns")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing multi-bagger potential: {e}")
            return {'probability': 0.0, 'confidence': 'Unknown', 'error': str(e)}
    
    def _display_hidden_gems_analysis(self, gems_analysis: Dict[str, Any], symbol: str):
        """Display comprehensive hidden gems analysis results"""
        try:
            gem_score = gems_analysis.get('gem_score')
            pattern_insights = gems_analysis.get('pattern_insights', {})
            multi_bagger_assessment = gems_analysis.get('multi_bagger_potential', {})
            
            if not gem_score:
                st.error("Gem score analysis failed")
                return
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score_color = "🟢" if gem_score.composite_score >= 70 else "🟡" if gem_score.composite_score >= 50 else "🔴"
                st.metric(
                    "💎 Gem Score",
                    f"{score_color} {gem_score.composite_score:.1f}/100",
                    help="Composite hidden gem potential score"
                )
            
            with col2:
                st.metric(
                    "🎯 Risk Rating",
                    gem_score.risk_rating,
                    help="Investment risk assessment"
                )
            
            with col3:
                probability = multi_bagger_assessment.get('probability', 0) * 100
                st.metric(
                    "📈 Multi-Bagger Probability",
                    f"{probability:.0f}%",
                    help="Probability of achieving 2x+ returns"
                )
            
            with col4:
                confidence = multi_bagger_assessment.get('confidence', 'Unknown')
                st.metric(
                    "✅ Confidence Level",
                    confidence,
                    help="Analysis confidence level"
                )
            
            # Detailed score breakdown
            st.subheader("📊 Detailed Score Breakdown")
            
            score_data = {
                'Category': [
                    'Sector Tailwinds', 'Fundamental Strength', 'Technical Setup',
                    'Hidden Status', 'Catalyst Potential', 'Smart Money'
                ],
                'Score': [
                    gem_score.sector_score,
                    gem_score.fundamental_score,
                    gem_score.technical_score,
                    100 - gem_score.visibility_score,  # Invert visibility (lower is better)
                    gem_score.catalyst_score,
                    gem_score.smart_money_score
                ],
                'Weight': ['25%', '20%', '20%', '15%', '15%', '5%']
            }
            
            # Create score breakdown chart
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=score_data['Category'],
                    y=score_data['Score'],
                    text=[f"{score:.1f}" for score in score_data['Score']],
                    textposition='auto',
                    marker_color=[
                        '#2E8B57' if score >= 70 else '#DAA520' if score >= 50 else '#CD5C5C'
                        for score in score_data['Score']
                    ]
                )
            ])
            
            fig.update_layout(
                title=f"{symbol} - Hidden Gem Score Breakdown",
                yaxis_title="Score (0-100)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Investment thesis and analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("💡 Investment Thesis")
                st.write(gem_score.investment_thesis)
                
                st.subheader("🚀 Primary Catalyst")
                st.write(gem_score.primary_catalyst)
                
                # Historical pattern analysis
                if pattern_insights and 'similar_patterns' in pattern_insights:
                    similar_patterns = pattern_insights['similar_patterns']
                    
                    if similar_patterns:
                        st.subheader("📈 Historical Pattern Analysis")
                        
                        top_pattern = similar_patterns[0]
                        
                        st.info(f"""
                        **Most Similar Pattern:** {top_pattern['historical_ticker']} ({top_pattern['similarity_score']:.1%} similarity)
                        
                        **Historical Performance:** {top_pattern['gain_achieved']:.0f}% gain over {top_pattern['pattern_duration_months']} months
                        
                        **Key Similarities:**
                        {chr(10).join(['• ' + sim for sim in top_pattern.get('key_similarities', [])[:3]])}
                        """)
                        
                        # Show pattern comparison table
                        if len(similar_patterns) > 1:
                            pattern_data = []
                            for pattern in similar_patterns[:5]:
                                pattern_data.append({
                                    'Historical Ticker': pattern['historical_ticker'],
                                    'Similarity': f"{pattern['similarity_score']:.1%}",
                                    'Gain Achieved': f"{pattern['gain_achieved']:.0f}%",
                                    'Duration (Months)': pattern['pattern_duration_months'],
                                    'Confidence': f"{pattern['pattern_confidence']:.1%}"
                                })
                            
                            st.write("**Similar Historical Patterns:**")
                            df = pd.DataFrame(pattern_data)
                            st.dataframe(df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("⚡ Multi-Bagger Assessment")
                
                # Probability gauge
                probability = multi_bagger_assessment.get('probability', 0)
                if probability >= 0.6:
                    prob_color = "🟢"
                    prob_text = "High"
                elif probability >= 0.4:
                    prob_color = "🟡"
                    prob_text = "Medium"
                else:
                    prob_color = "🔴"
                    prob_text = "Low"
                
                st.metric("Probability", f"{prob_color} {probability*100:.0f}% ({prob_text})")
                
                # Time horizon and upside
                time_horizon = multi_bagger_assessment.get('time_horizon', 'Unknown')
                upside_potential = multi_bagger_assessment.get('upside_potential', 'Unknown')
                
                st.metric("Time Horizon", time_horizon)
                st.metric("Upside Potential", upside_potential)
                
                # Action plan from gem score
                if hasattr(gem_score, 'action_plan') and gem_score.action_plan:
                    action_plan = gem_score.action_plan
                    
                    st.subheader("📋 Action Plan")
                    
                    if 'entry_range' in action_plan:
                        entry_range = action_plan['entry_range']
                        st.write(f"**Entry Range:** ${entry_range['low']:.2f} - ${entry_range['high']:.2f}")
                    
                    if 'stop_loss' in action_plan:
                        st.write(f"**Stop Loss:** ${action_plan['stop_loss']:.2f}")
                    
                    if 'targets' in action_plan:
                        targets = action_plan['targets']
                        st.write(f"**12M Target:** ${targets['12_month']:.2f}")
                        st.write(f"**24M Target:** ${targets['24_month']:.2f}")
                    
                    if 'position_sizing' in action_plan:
                        st.write(f"**Position Size:** {action_plan['position_sizing']}")
            
            # Key catalysts and risks
            catalysts = multi_bagger_assessment.get('key_catalysts', [])
            risks = multi_bagger_assessment.get('risk_factors', [])
            
            if catalysts or risks:
                col1, col2 = st.columns(2)
                
                with col1:
                    if catalysts:
                        st.subheader("🚀 Key Catalysts")
                        for catalyst in catalysts:
                            st.success(f"✅ {catalyst}")
                
                with col2:
                    if risks:
                        st.subheader("⚠️ Risk Factors")
                        for risk in risks:
                            st.warning(f"⚠️ {risk}")
            
            # Monitoring recommendations
            st.subheader("👀 Monitoring Recommendations")
            
            monitoring_items = [
                "📊 Volume expansion on price breakouts",
                "🏛️ Institutional ownership changes (13F filings)",
                "📈 Revenue growth acceleration quarter-over-quarter",
                "🔄 Sector rotation and capital flow trends",
                "📐 Technical pattern completion and confirmation",
                "📰 News catalysts and management commentary",
                "💰 Insider buying activity and option flows"
            ]
            
            # Display as expandable sections
            with st.expander("📋 Detailed Monitoring Checklist"):
                for item in monitoring_items:
                    st.write(f"• {item}")
            
            # Summary recommendation
            st.subheader("🎯 Summary Recommendation")
            
            if gem_score.composite_score >= 75:
                st.success(f"""
                🟢 **STRONG BUY** - High conviction hidden gem opportunity
                
                {symbol} scores {gem_score.composite_score:.1f}/100 with {probability*100:.0f}% multi-bagger probability.
                This represents a high-quality opportunity with strong fundamentals, technical setup,
                and historical pattern similarity. Consider {action_plan.get('position_sizing', '1-2% portfolio position')}.
                """)
            elif gem_score.composite_score >= 60:
                st.info(f"""
                🟡 **BUY** - Solid opportunity with moderate conviction
                
                {symbol} scores {gem_score.composite_score:.1f}/100 with {probability*100:.0f}% multi-bagger probability.
                This represents a reasonable opportunity that warrants inclusion in a diversified
                hidden gems portfolio. Consider smaller position sizing and careful monitoring.
                """)
            elif gem_score.composite_score >= 45:
                st.warning(f"""
                🟠 **SPECULATIVE** - High-risk, high-reward opportunity
                
                {symbol} scores {gem_score.composite_score:.1f}/100 with {probability*100:.0f}% multi-bagger probability.
                This is a speculative play that could pay off but carries significant risk.
                Only suitable for risk-tolerant investors with small position sizes.
                """)
            else:
                st.error(f"""
                🔴 **AVOID** - Does not meet hidden gem criteria
                
                {symbol} scores {gem_score.composite_score:.1f}/100 with {probability*100:.0f}% multi-bagger probability.
                Current analysis suggests limited multi-bagger potential. Consider other opportunities
                or wait for improved setup before considering investment.
                """)
                
        except Exception as e:
            logger.error(f"Error displaying hidden gems analysis: {e}")
            st.error(f"Error displaying analysis: {str(e)}")
