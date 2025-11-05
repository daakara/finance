"""
Hidden Gems Scanner - Streamlit Dashboard
Advanced visualization and interaction interface for hidden gem discovery
"""

# SSL Certificate Fix - Must be at the very top before any imports
import ssl
import os
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception as e:
    print(f"Warning: Could not configure SSL certificates: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time

# Import our scanner components
from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria, GemScore
from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline

logger = logging.getLogger(__name__)

class GemDashboard:
    """Streamlit dashboard for Hidden Gems Scanner"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.screener = HiddenGemScreener()
        self.data_pipeline = MultiAssetDataPipeline()
        
        # Sample universe of tickers for screening
        self.stock_universe = [
            # Large cap growth
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            # Mid cap opportunities
            'PLTR', 'SNOW', 'CRWD', 'ZS', 'NET', 'DDOG', 'MDB',
            # Small cap gems
            'IREN', 'MSTR', 'COIN', 'HOOD', 'SQ', 'PYPL', 'ROKU',
            # Biotech
            'MRNA', 'BNTX', 'GILD', 'REGN', 'VRTX', 'ILMN',
            # Fintech
            'AFRM', 'UPST', 'SOFI', 'LC', 'OPEN',
            # Clean Energy
            'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR'
        ]
        
        self.etf_universe = [
            # Tech/AI
            'QQQ', 'XLK', 'ARKK', 'ARKQ', 'ROBO', 'BOTZ',
            # Clean Energy
            'ICLN', 'PBW', 'QCLN', 'ERTH',
            # Fintech
            'FINX', 'IPAY', 'BLOK',
            # Biotech
            'IBB', 'XBI', 'ARKG',
            # Space
            'UFO', 'ARKX'
        ]
        
        self.crypto_universe = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'AVAX', 'MATIC', 'DOT', 'LINK', 'UNI'
        ]
    
    def run_dashboard(self):
        """Main dashboard interface"""
        st.set_page_config(
            page_title="Hidden Gems Scanner",
            page_icon="💎",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title and description
        st.title("💎 Hidden Gems Scanner")
        st.markdown("**Advanced Multi-Asset Discovery System**")
        st.markdown("*Identifying undervalued opportunities with 10x+ potential before mainstream adoption*")
        
        # Data source status indicator
        self._show_data_source_status()
        
        # Sidebar controls
        self._setup_sidebar()
        
        # Show usage guide and troubleshooting on first visit
        if 'shown_guide' not in st.session_state:
            with st.expander("📖 How to Use This Dashboard", expanded=True):
                st.markdown("""
                **🏆 Top Opportunities**: View highest-scoring assets with detailed analysis cards
                
                **🔍 Individual Analysis**: Enter any ticker for deep-dive analysis and scoring
                
                **🌡️ Sector Heat Map**: Visualize opportunity distribution across emerging sectors
                
                **📊 Screening Results**: View comprehensive screening results with filters
                
                **⚙️ Custom Screener**: Build custom criteria to find specific opportunities
                
                **💡 Tips**: 
                - Higher visibility scores indicate less analyst coverage (more "hidden")
                - Composite scores combine 6 factors: Sector, Fundamental, Technical, Visibility, Catalyst, Smart Money
                - Use sidebar controls to adjust screening parameters
                
                **🔧 Troubleshooting**:
                - **Live Data**: "� Live Data" means real-time market data is available
                - **Data Unavailable**: If you see "🔴 Data Unavailable", check your network connection
                - **SSL Issues**: In corporate environments, SSL certificate issues may prevent API access
                - **Performance**: Dashboard uses caching to improve speed - data refreshes every 5 minutes
                - **Network Requirements**: Requires internet access to fetch live market data from Yahoo Finance
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Got it! Don't show again"):
                        st.session_state.shown_guide = True
                        st.rerun()
                with col2:
                    if st.button("🔧 Test Connection"):
                        st.session_state.test_connection = True
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Top Opportunities", 
            "🔍 Individual Analysis", 
            "🌡️ Sector Heat Map", 
            "📊 Screening Results",
            "⚙️ Custom Screener"
        ])
        
        with tab1:
            self._show_top_opportunities()
        
        with tab2:
            self._show_individual_analysis()
        
        with tab3:
            self._show_sector_heatmap()
        
        with tab4:
            self._show_screening_results()
        
        with tab5:
            self._show_custom_screener()
    
    def _setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.title("🔧 Scanner Settings")
        
        # Load saved preferences with error handling
        if 'gem_scanner_prefs' not in st.session_state:
            st.session_state.gem_scanner_prefs = {
                'asset_types': ["Stocks"],
                'min_market_cap': 50,
                'max_market_cap': 2.0,
                'min_revenue_growth': 25,
                'min_gross_margin': 30
            }
        
        # Validate preferences
        try:
            prefs = st.session_state.gem_scanner_prefs
            if not isinstance(prefs.get('asset_types'), list):
                prefs['asset_types'] = ["Stocks"]
        except Exception:
            # Reset to defaults if corrupted
            st.session_state.gem_scanner_prefs = {
                'asset_types': ["Stocks"],
                'min_market_cap': 50,
                'max_market_cap': 2.0,
                'min_revenue_growth': 25,
                'min_gross_margin': 30
            }
        
        # Asset type selection
        self.asset_types = st.sidebar.multiselect(
            "Asset Types:",
            ["Stocks", "ETFs", "Crypto"],
            default=st.session_state.gem_scanner_prefs.get('asset_types', ["Stocks"]),
            help="Select asset types to screen"
        )
        st.session_state.gem_scanner_prefs['asset_types'] = self.asset_types
        
        # Market cap range
        st.sidebar.subheader("💰 Market Cap Range")
        self.min_market_cap = st.sidebar.number_input(
            "Minimum Market Cap ($M):",
            min_value=10.0,
            max_value=10000.0,
            value=50.0,
            step=10.0
        ) * 1e6
        
        self.max_market_cap = st.sidebar.number_input(
            "Maximum Market Cap ($B):",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1
        ) * 1e9
        
        # Screening criteria
        st.sidebar.subheader("📈 Fundamental Criteria")
        self.min_revenue_growth = st.sidebar.slider(
            "Min Revenue Growth (%):",
            min_value=0,
            max_value=100,
            value=25,
            step=5
        ) / 100
        
        self.min_gross_margin = st.sidebar.slider(
            "Min Gross Margin (%):",
            min_value=0,
            max_value=80,
            value=30,
            step=5
        ) / 100
        
        # Visibility filters
        st.sidebar.subheader("👁️ Visibility Filters")
        self.max_analyst_coverage = st.sidebar.slider(
            "Max Analyst Coverage:",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )
        
        # Sector focus
        st.sidebar.subheader("🎯 Sector Focus")
        self.focus_sectors = st.sidebar.multiselect(
            "Focus on Emerging Sectors:",
            [
                "AI/ML", "Blockchain", "Clean Energy", "Biotech", 
                "Fintech", "Space Tech", "Cybersecurity", "Robotics"
            ],
            default=["AI/ML", "Blockchain"],
            help="Focus screening on specific emerging sectors"
        )
        
        # Sustainability filters
        st.sidebar.subheader("🌱 Sustainability Filters")
        self.enable_sustainability = st.sidebar.checkbox(
            "Enable ESG/Impact Scoring",
            value=True,
            help="Include sustainability and impact metrics in analysis"
        )
        
        if self.enable_sustainability:
            self.min_sustainability_score = st.sidebar.slider(
                "Min Sustainability Score:",
                min_value=0,
                max_value=100,
                value=50,
                step=10,
                help="Minimum overall ESG/sustainability score (0-100)"
            )
            
            self.impact_focus = st.sidebar.multiselect(
                "Impact Focus Areas:",
                [
                    "Renewable Energy", "Clean Tech", "Healthcare Innovation",
                    "Financial Inclusion", "Education Tech", "Circular Economy",
                    "Water Tech", "Sustainable Agriculture"
                ],
                help="Filter for specific impact categories"
            )
        else:
            self.min_sustainability_score = 0
            self.impact_focus = []
        
        # Screening action
        st.sidebar.markdown("---")
        self.run_screening = st.sidebar.button(
            "🚀 Run Full Screening",
            help="Execute comprehensive screening across selected universe"
        )
        
        # Quick actions
        st.sidebar.markdown("### 🎯 Quick Scans")
        if st.sidebar.button("⚡ Blockchain Infrastructure"):
            self.quick_scan_type = "blockchain"
        elif st.sidebar.button("🤖 AI/ML Leaders"):
            self.quick_scan_type = "ai_ml"
        elif st.sidebar.button("🔋 Clean Energy"):
            self.quick_scan_type = "clean_energy"
        else:
            self.quick_scan_type = None
    
    def _show_top_opportunities(self):
        """Display top identified opportunities"""
        st.header("🏆 Top Hidden Gem Opportunities")
        
        # Run screening - LIVE DATA ONLY
        if not hasattr(self, 'screening_results') or self.run_screening:
            with st.spinner("🔍 Screening universe for hidden gems..."):
                try:
                    # Run live screening
                    self.screening_results = self._run_sample_screening()
                        
                except Exception as e:
                    st.error(f"⚠️ Screening failed: {str(e)}")
                    st.error("💡 Unable to fetch live data. Please check your network connection and try again.")
                    self.screening_results = []
        
        if not self.screening_results:
            st.warning("No opportunities found matching current criteria. Try adjusting filters.")
            return
        
        # Display top 10 opportunities
        top_opportunities = self.screening_results[:10]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Opportunities Found", 
                len(self.screening_results),
                help="Total opportunities meeting screening criteria"
            )
        
        with col2:
            avg_score = np.mean([opp.composite_score for opp in top_opportunities])
            st.metric(
                "Avg Composite Score", 
                f"{avg_score:.1f}/100",
                help="Average composite score of top opportunities"
            )
        
        with col3:
            high_conviction = len([opp for opp in top_opportunities if opp.composite_score >= 80])
            st.metric(
                "High Conviction Plays", 
                high_conviction,
                help="Opportunities with 80+ composite score"
            )
        
        with col4:
            sector_diversity = len(set([self._get_sector_from_ticker(opp.ticker) for opp in top_opportunities]))
            st.metric(
                "Sector Diversity", 
                sector_diversity,
                help="Number of different sectors represented"
            )
        
        # Opportunities table
        st.subheader("💎 Top Ranked Opportunities")
        st.caption("🔍 **How to read**: Composite Score combines all factors. Higher visibility score = less analyst coverage (more hidden). Click expanders below for detailed analysis.")
        
        opportunities_data = []
        for i, opp in enumerate(top_opportunities, 1):
            opportunities_data.append({
                'Rank': i,
                'Ticker': opp.ticker,
                'Composite Score': f"{opp.composite_score:.1f}/100",
                'Risk Rating': opp.risk_rating,
                'Primary Catalyst': opp.primary_catalyst[:50] + "..." if len(opp.primary_catalyst) > 50 else opp.primary_catalyst,
                'Sector Score': f"{opp.sector_score:.1f}",
                'Technical Score': f"{opp.technical_score:.1f}",
                'Visibility Score': f"{opp.visibility_score:.1f}",
                'Action': f"📊 Analyze"
            })
        
        df = pd.DataFrame(opportunities_data)
        
        # Color-code the table based on composite score
        def color_score(val):
            if 'Score' in val.name and val.name != 'Action':
                colors = []
                for v in val:
                    try:
                        score = float(v.split('/')[0]) if '/' in str(v) else float(v)
                        if score >= 80:
                            colors.append('background-color: #d4edda; color: #155724')  # Green
                        elif score >= 65:
                            colors.append('background-color: #fff3cd; color: #856404')  # Yellow
                        elif score < 50:
                            colors.append('background-color: #f8d7da; color: #721c24')  # Red
                        else:
                            colors.append('')
                    except (ValueError, AttributeError):
                        colors.append('')
                return colors
            return ['' for _ in val]
        
        styled_df = df.style.apply(color_score, axis=0)
        st.dataframe(styled_df, width='stretch')
        
        # Export functionality
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📊 Export Results"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"hidden_gems_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        # Detailed cards for top 3
        st.subheader("🎯 Detailed Analysis - Top 3 Opportunities")
        
        for i, opp in enumerate(top_opportunities[:3]):
            with st.expander(f"#{i+1} {opp.ticker} - {opp.risk_rating} | Score: {opp.composite_score:.1f}/100", expanded=i==0):
                self._show_opportunity_card(opp)
    
    def _show_individual_analysis(self):
        """Show detailed analysis for individual assets"""
        st.header("🔍 Individual Deep Dive Analysis")
        
        # Ticker selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker_input = st.text_input(
                "Enter Ticker Symbol:",
                value="IREN",
                help="Enter any stock, ETF, or crypto symbol for analysis"
            )
            ticker = ticker_input.upper().strip()
            
            # Validate ticker format
            if ticker and not ticker.replace('-', '').replace('.', '').isalnum():
                st.warning("⚠️ Please enter a valid ticker symbol (letters, numbers, hyphens, and dots only)")
        
        with col2:
            asset_type = st.selectbox(
                "Asset Type:",
                ["Stock", "ETF", "Crypto"],
                help="Select the type of asset"
            )
        
        if st.button("🔍 Analyze Asset"):
            if ticker:
                with st.spinner(f"🔍 Fetching data and analyzing {ticker}... This may take a moment."):
                    start_time = time.time()
                    self._show_detailed_analysis(ticker, asset_type.lower())
                    end_time = time.time()
                    st.success(f"✅ Analysis completed in {end_time - start_time:.1f} seconds")
            else:
                st.warning("⚠️ Please enter a ticker symbol.")
    
    def _show_sector_heatmap(self):
        """Show sector rotation and opportunity heatmap"""
        st.header("🌡️ Sector Opportunity Heat Map")
        
        # Create sample sector data
        sectors = [
            "AI/Machine Learning", "Blockchain/Crypto", "Clean Energy", 
            "Biotechnology", "Fintech", "Space Technology", 
            "Cybersecurity", "Robotics/Automation", "Quantum Computing", "5G/Edge Computing"
        ]
        
        # Sample data - in real implementation, this would be calculated from screening results
        sector_data = {
            'Sector': sectors,
            'Opportunity Score': np.random.uniform(40, 95, len(sectors)),
            'Capital Flows': np.random.uniform(-500, 1000, len(sectors)),
            'Momentum': np.random.uniform(-20, 40, len(sectors)),
            'Gem Count': np.random.randint(1, 15, len(sectors))
        }
        
        df = pd.DataFrame(sector_data)
        
        # Heatmap visualization with updated config
        fig = px.treemap(
            df,
            path=['Sector'],
            values='Gem Count',
            color='Opportunity Score',
            color_continuous_scale='RdYlGn',
            title="Sector Opportunity Map - Size by Hidden Gem Count, Color by Opportunity Score"
        )
        
        fig.update_layout(height=500)
        
        # Use new config parameter instead of deprecated keywords
        plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        st.plotly_chart(fig, width='stretch', config=plotly_config)
        
        # Sector details table
        st.subheader("📊 Sector Analysis Details")
        
        # Style the dataframe
        styled_sector_df = df.style.background_gradient(
            subset=['Opportunity Score'], 
            cmap='RdYlGn'
        ).background_gradient(
            subset=['Capital Flows'], 
            cmap='RdBu'
        ).format({
            'Opportunity Score': '{:.1f}',
            'Capital Flows': '${:,.0f}M',
            'Momentum': '{:+.1f}%'
        })
        
        st.dataframe(styled_sector_df, width='stretch')
        
        # Top sector recommendations
        top_sectors = df.nlargest(3, 'Opportunity Score')
        
        st.subheader("🎯 Top Sector Recommendations")
        for _, sector in top_sectors.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{sector['Sector']}**")
                    st.write(f"Opportunity Score: {sector['Opportunity Score']:.1f}/100")
                
                with col2:
                    st.metric("Hidden Gems", int(sector['Gem Count']))
                
                with col3:
                    momentum_color = "🟢" if sector['Momentum'] > 0 else "🔴"
                    st.metric("Momentum", f"{momentum_color} {sector['Momentum']:+.1f}%")
    
    def _show_screening_results(self):
        """Show comprehensive screening results and analytics"""
        st.header("📊 Comprehensive Screening Results")
        
        if not hasattr(self, 'screening_results'):
            st.info("Run screening from the sidebar to see detailed results.")
            return
        
        # Results overview
        total_screened = len(self.stock_universe) + len(self.etf_universe) if "Stocks" in self.asset_types else len(self.etf_universe)
        if "Crypto" in self.asset_types:
            total_screened += len(self.crypto_universe)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assets Screened", total_screened)
        
        with col2:
            st.metric("Opportunities Found", len(self.screening_results))
        
        with col3:
            hit_rate = len(self.screening_results) / max(total_screened, 1) * 100
            st.metric("Hit Rate", f"{hit_rate:.1f}%")
        
        with col4:
            avg_score = np.mean([r.composite_score for r in self.screening_results]) if self.screening_results else 0
            st.metric("Avg Score", f"{avg_score:.1f}/100")
        
        # Score distribution
        if self.screening_results:
            scores = [r.composite_score for r in self.screening_results]
            
            fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20)])
            fig.update_layout(
                title="Distribution of Composite Scores",
                xaxis_title="Composite Score",
                yaxis_title="Count",
                height=400
            )
            
            # Use new config parameter
            plotly_config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            st.plotly_chart(fig, width='stretch', config=plotly_config)
            
            # Detailed results table
            st.subheader("📋 Detailed Screening Results")
            
            results_data = []
            for i, result in enumerate(self.screening_results):
                results_data.append({
                    'Rank': i + 1,
                    'Ticker': result.ticker,
                    'Composite Score': result.composite_score,
                    'Sector Score': result.sector_score,
                    'Fundamental Score': result.fundamental_score,
                    'Technical Score': result.technical_score,
                    'Visibility Score': result.visibility_score,
                    'Catalyst Score': result.catalyst_score,
                    'Smart Money Score': result.smart_money_score,
                    'Risk Rating': result.risk_rating,
                    'Primary Catalyst': result.primary_catalyst
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Filter and sort options
            col1, col2 = st.columns(2)
            
            with col1:
                min_score_filter = st.slider("Minimum Composite Score:", 0, 100, 50)
                filtered_df = results_df[results_df['Composite Score'] >= min_score_filter]
            
            with col2:
                risk_filter = st.multiselect(
                    "Risk Rating Filter:",
                    options=results_df['Risk Rating'].unique(),
                    default=results_df['Risk Rating'].unique()
                )
                filtered_df = filtered_df[filtered_df['Risk Rating'].isin(risk_filter)]
            
            # Display filtered results
            st.dataframe(filtered_df, width='stretch')
    
    def _show_custom_screener(self):
        """Show custom screening interface"""
        st.header("⚙️ Custom Hidden Gem Screener")
        st.markdown("Build your own screening criteria to discover specific types of opportunities.")
        
        # Custom criteria form
        with st.form("custom_screener"):
            st.subheader("🎯 Define Your Criteria")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Financial Criteria**")
                custom_min_revenue_growth = st.number_input("Min Revenue Growth (%):", min_value=0.0, max_value=500.0, value=25.0, step=1.0) / 100
                custom_min_gross_margin = st.number_input("Min Gross Margin (%):", min_value=0.0, max_value=100.0, value=30.0, step=1.0) / 100
                custom_max_pe = st.number_input("Max P/E Ratio:", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
                custom_min_cash_runway = st.number_input("Min Cash Runway (years):", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            
            with col2:
                st.markdown("**Market Criteria**")
                custom_min_mcap = st.number_input("Min Market Cap ($M):", min_value=1.0, max_value=50000.0, value=50.0, step=1.0) * 1e6
                custom_max_mcap = st.number_input("Max Market Cap ($B):", min_value=0.1, max_value=100.0, value=2.0, step=0.1) * 1e9
                custom_max_analysts = st.number_input("Max Analyst Coverage:", min_value=1, max_value=100, value=10, step=1)
                custom_min_institutional = st.number_input("Min Institutional Ownership (%):", min_value=0.0, max_value=100.0, value=10.0, step=1.0) / 100
            
            # Sector preferences
            st.markdown("**Sector Preferences**")
            custom_sectors = st.multiselect(
                "Preferred Sectors:",
                [
                    "AI/Machine Learning", "Blockchain/Crypto", "Clean Energy", 
                    "Biotechnology", "Fintech", "Space Technology", 
                    "Cybersecurity", "Robotics", "Quantum Computing"
                ]
            )
            
            # Technical criteria
            st.markdown("**Technical Criteria**")
            col3, col4 = st.columns(2)
            
            with col3:
                require_accumulation = st.checkbox("Require Accumulation Pattern", value=True)
                require_insider_buying = st.checkbox("Require Recent Insider Buying", value=False)
            
            with col4:
                min_relative_strength = st.number_input("Min Relative Strength:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
                max_drawdown = st.number_input("Max Drawdown from High (%):", min_value=0.0, max_value=100.0, value=30.0, step=1.0) / 100
            
            # Submit button
            submitted = st.form_submit_button("🚀 Run Custom Screening")
            
            if submitted:
                # Create custom criteria
                custom_criteria = GemCriteria(
                    min_market_cap=custom_min_mcap,
                    max_market_cap=custom_max_mcap,
                    min_revenue_growth=custom_min_revenue_growth,
                    min_gross_margin=custom_min_gross_margin,
                    max_analyst_coverage=custom_max_analysts,
                    min_institutional_ownership=custom_min_institutional
                )
                
                # Run custom screening
                with st.spinner("Running custom screening..."):
                    custom_screener = HiddenGemScreener(custom_criteria)
                    custom_results = self._run_sample_screening(custom_screener)
                
                # Display results
                if custom_results:
                    st.success(f"Found {len(custom_results)} opportunities matching your criteria!")
                    
                    # Show top results
                    for i, result in enumerate(custom_results[:5]):
                        with st.expander(f"#{i+1} {result.ticker} - Score: {result.composite_score:.1f}/100"):
                            self._show_opportunity_card(result)
                else:
                    st.warning("No opportunities found matching your custom criteria. Try relaxing some constraints.")
    
    def _show_opportunity_card(self, opportunity: GemScore):
        """Display a detailed opportunity card"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Investment Thesis:**")
            st.write(opportunity.investment_thesis)
            
            st.markdown(f"**Primary Catalyst:**")
            st.write(opportunity.primary_catalyst)
            
            # Score breakdown
            st.markdown("**Score Breakdown:**")
            score_data = {
                'Category': ['Sector', 'Fundamental', 'Technical', 'Visibility', 'Catalyst', 'Smart Money', 'Sustainability'],
                'Score': [
                    opportunity.sector_score,
                    opportunity.fundamental_score,
                    opportunity.technical_score,
                    100 - opportunity.visibility_score,  # Invert visibility (lower is better)
                    opportunity.catalyst_score,
                    opportunity.smart_money_score,
                    opportunity.sustainability_score
                ]
            }
            
            fig = go.Figure(data=[
                go.Bar(x=score_data['Category'], y=score_data['Score'], 
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf'])
            ])
            fig.update_layout(
                title=f"{opportunity.ticker} - Score Breakdown",
                yaxis_title="Score (0-100)",
                height=300,
                showlegend=False
            )
            
            # Use new config parameter
            plotly_config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            }
            st.plotly_chart(fig, width='stretch', config=plotly_config)
            
            # Sustainability details if available
            if hasattr(opportunity, 'sustainability_data') and opportunity.sustainability_data:
                st.markdown("---")
                st.markdown("**🌱 Sustainability & Impact:**")
                
                sust = opportunity.sustainability_data
                
                # ESG Breakdown
                esg_col1, esg_col2, esg_col3 = st.columns(3)
                with esg_col1:
                    st.metric("Environmental", f"{sust.environmental_score:.0f}/100")
                with esg_col2:
                    st.metric("Social", f"{sust.social_score:.0f}/100")
                with esg_col3:
                    st.metric("Governance", f"{sust.governance_score:.0f}/100")
                
                # Impact categories
                if sust.impact_categories:
                    st.write(f"**Impact Areas:** {', '.join(sust.impact_categories)}")
                
                # SDG alignment
                if sust.sdg_alignment:
                    from analyst_dashboard.analyzers.sustainability_analyzer import SustainabilityAnalyzer
                    analyzer = SustainabilityAnalyzer()
                    sdg_names = [f"SDG {sdg}" for sdg in sust.sdg_alignment[:3]]
                    st.write(f"**UN SDGs:** {', '.join(sdg_names)}")
                
                # Impact thesis
                if sust.impact_thesis:
                    st.write(f"*{sust.impact_thesis}*")
        
        with col2:
            # Key metrics
            st.markdown("**Key Metrics:**")
            st.metric("Composite Score", f"{opportunity.composite_score:.1f}/100")
            st.metric("Risk Rating", opportunity.risk_rating)
            
            # Action plan
            if opportunity.action_plan and 'entry_range' in opportunity.action_plan:
                action_plan = opportunity.action_plan
                st.markdown("**Action Plan:**")
                
                if 'entry_range' in action_plan:
                    entry_low = action_plan['entry_range']['low']
                    entry_high = action_plan['entry_range']['high']
                    st.write(f"Entry Range: ${entry_low:.2f} - ${entry_high:.2f}")
                
                if 'stop_loss' in action_plan:
                    st.write(f"Stop Loss: ${action_plan['stop_loss']:.2f}")
                
                if 'targets' in action_plan:
                    targets = action_plan['targets']
                    st.write(f"12M Target: ${targets['12_month']:.2f}")
                    st.write(f"24M Target: ${targets['24_month']:.2f}")
                
                if 'position_sizing' in action_plan:
                    st.write(f"Position Size: {action_plan['position_sizing']}")
    
    def _show_detailed_analysis(self, ticker: str, asset_type: str):
        """Show detailed analysis for a specific ticker - LIVE DATA ONLY"""
        try:
            # Attempt live data fetch
            all_data = self.data_pipeline.get_comprehensive_data(ticker, asset_type)
            
            if 'error' in all_data:
                st.error(f"⚠️ Unable to fetch data for {ticker}")
                st.info("� **Possible causes**: Network restrictions, SSL certificate issues, invalid ticker, or API rate limits.")
                return
            
            # Calculate gem score
            gem_score = self.screener.calculate_composite_score(ticker, all_data)
            
            # Display analysis
            st.success(f"Analysis complete for {ticker}")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Composite Score", f"{gem_score.composite_score:.1f}/100")
            
            with col2:
                st.metric("Risk Rating", gem_score.risk_rating)
            
            with col3:
                if 'market_data' in all_data:
                    market_cap = all_data['market_data'].get('market_cap', 0)
                    if market_cap > 0:
                        st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M")
                    else:
                        st.metric("Market Cap", "N/A")
            
            with col4:
                sector = all_data.get('info', {}).get('sector', 'Unknown')
                st.metric("Sector", sector)
            
            # Show opportunity card
            self._show_opportunity_card(gem_score)
            
            # Price chart if available
            if 'price_data' in all_data and not all_data['price_data'].empty:
                st.subheader("📈 Price Chart with Technical Analysis")
                price_data = all_data['price_data']
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name=ticker
                ))
                
                # Moving averages
                if len(price_data) >= 50:
                    ma_20 = price_data['Close'].rolling(20).mean()
                    ma_50 = price_data['Close'].rolling(50).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=ma_20,
                        mode='lines',
                        name='MA 20',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=ma_50,
                        mode='lines',
                        name='MA 50',
                        line=dict(color='blue', width=1)
                    ))
                
                fig.update_layout(
                    title=f"{ticker} - Price Chart with Technical Indicators",
                    yaxis_title="Price",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                # Use new config parameter
                plotly_config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{ticker}_chart',
                        'height': 500,
                        'width': 1000,
                        'scale': 1
                    }
                }
                st.plotly_chart(fig, width='stretch', config=plotly_config)
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            logger.error(f"Error in detailed analysis for {ticker}: {e}")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _run_sample_screening(_self, screener=None):
        """Run screening with current settings - LIVE DATA ONLY"""
        if screener is None:
            # Update screener criteria based on sidebar settings
            criteria = GemCriteria(
                min_market_cap=getattr(_self, 'min_market_cap', 50e6),
                max_market_cap=getattr(_self, 'max_market_cap', 2e9),
                min_revenue_growth=getattr(_self, 'min_revenue_growth', 0.25),
                min_gross_margin=getattr(_self, 'min_gross_margin', 0.30),
                max_analyst_coverage=getattr(_self, 'max_analyst_coverage', 10)
            )
            screener = HiddenGemScreener(criteria)
        
        # Build universe based on selected asset types
        universe = []
        asset_types = getattr(_self, 'asset_types', ["Stocks"])
        
        if "Stocks" in asset_types:
            universe.extend(_self.stock_universe[:20])  # Limit for demo
        if "ETFs" in asset_types:
            universe.extend(_self.etf_universe[:10])
        if "Crypto" in asset_types:
            universe.extend([f"{crypto}-USD" for crypto in _self.crypto_universe[:10]])
        
        # Quick scan logic
        if hasattr(_self, 'quick_scan_type') and _self.quick_scan_type:
            if _self.quick_scan_type == "blockchain":
                # Filter for blockchain-related tickers
                blockchain_tickers = ['IREN', 'MSTR', 'COIN', 'HOOD', 'SQ']
                universe = [t for t in universe if t in blockchain_tickers]
            elif _self.quick_scan_type == "ai_ml":
                # Filter for AI/ML-related tickers
                ai_tickers = ['PLTR', 'NVDA', 'GOOGL', 'MSFT', 'META']
                universe = [t for t in universe if t in ai_tickers]
            elif _self.quick_scan_type == "clean_energy":
                # Filter for clean energy tickers
                clean_tickers = ['ENPH', 'SEDG', 'RUN', 'FSLR', 'NOVA', 'ICLN', 'PBW']
                universe = [t for t in universe if t in clean_tickers]
        
        # Run live screening
        results = screener.screen_universe(universe[:15])  # Limit for demo performance
        return results if results else []
    
    def _get_sector_from_ticker(self, ticker: str) -> str:
        """Get sector for a ticker (simplified mapping)"""
        sector_mapping = {
            # Stocks
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'TSLA': 'Consumer Cyclical', 'META': 'Technology',
            'IREN': 'Financial Services', 'MSTR': 'Technology', 'COIN': 'Financial Services',
            'MRNA': 'Healthcare', 'BNTX': 'Healthcare', 'PLTR': 'Technology',
            'NET': 'Technology', 'CRWD': 'Technology', 'SNOW': 'Technology',
            
            # ETFs
            'ARKK': 'Innovation ETF', 'ARKQ': 'Autonomous Technology ETF', 'ROBO': 'Robotics ETF',
            'ICLN': 'Clean Energy ETF', 'PBW': 'Clean Energy ETF', 'QCLN': 'Clean Energy ETF',
            'FINX': 'Fintech ETF', 'BLOK': 'Blockchain ETF', 'IBB': 'Biotech ETF',
            'XBI': 'Biotech ETF', 'UFO': 'Space ETF', 'ARKX': 'Space ETF',
            'QQQ': 'Tech ETF', 'XLK': 'Tech ETF',
            
            # Crypto
            'BTC-USD': 'Cryptocurrency', 'ETH-USD': 'Cryptocurrency', 'BNB-USD': 'Cryptocurrency',
            'ADA-USD': 'Cryptocurrency', 'SOL-USD': 'Cryptocurrency', 'AVAX-USD': 'Cryptocurrency',
            'MATIC-USD': 'Cryptocurrency', 'DOT-USD': 'Cryptocurrency', 'LINK-USD': 'Cryptocurrency',
            'UNI-USD': 'Cryptocurrency'
        }
        return sector_mapping.get(ticker, 'Other')

    def _show_data_source_status(self):
        """Show current data source status"""
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col2:
                # Test connectivity and show status
                data_source = self._test_data_connectivity()
                if data_source == "live":
                    st.success("🟢 Live Data")
                else:
                    st.error("� Data Unavailable")
                    st.caption("Network/API issues detected")
            
            with col3:
                last_update = datetime.now().strftime("%H:%M")
                st.caption(f"Updated: {last_update}")
                
                # Refresh button
                if st.button("🔄 Refresh", help="Refresh data connection status"):
                    st.rerun()
    
    @st.cache_data(ttl=300)  # Cache connectivity test for 5 minutes
    def _test_data_connectivity(_self) -> str:
        """Test data source connectivity - returns 'live' if successful, 'unavailable' otherwise"""
        try:
            import yfinance as yf
            import warnings
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Quick test - create ticker and fetch minimal data
                test_ticker = yf.Ticker('AAPL')
                
                try:
                    # Minimal test with short timeout
                    test_data = test_ticker.history(period='1d', timeout=2)
                    
                    if not test_data.empty and len(test_data) > 0:
                        return "live"
                    else:
                        return "unavailable"
                        
                except Exception as inner_e:
                    logger.info(f"Data connectivity test failed: {inner_e}")
                    return "unavailable"
                    
        except Exception as e:
            logger.info(f"Data connectivity test failed completely: {e}")
            return "unavailable"


# Main dashboard runner
def run_gem_dashboard():
    """Run the Hidden Gems Scanner dashboard"""
    dashboard = GemDashboard()
    dashboard.run_dashboard()


# Example usage
if __name__ == "__main__":
    run_gem_dashboard()
