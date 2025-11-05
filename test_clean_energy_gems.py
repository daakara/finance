"""
Test script to demonstrate finding clean energy hidden gems
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria
from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clean_energy_gems():
    """Test finding clean energy hidden gems"""
    print("\n" + "="*80)
    print("🔋 CLEAN ENERGY HIDDEN GEMS SCANNER - DEMO")
    print("="*80 + "\n")
    
    # Initialize scanner with relaxed criteria for demo
    criteria = GemCriteria(
        min_market_cap=50e6,      # $50M minimum
        max_market_cap=10e9,      # $10B maximum (increased for demo)
        min_revenue_growth=0.15,  # 15% (relaxed)
        min_gross_margin=0.20,    # 20% (relaxed)
        max_debt_equity=1.0,      # More lenient
        max_analyst_coverage=20   # Increased
    )
    
    screener = HiddenGemScreener(criteria)
    data_pipeline = MultiAssetDataPipeline()
    
    # Clean energy universe to scan
    clean_energy_tickers = [
        # Pure play solar
        'ENPH',  # Enphase Energy - microinverters
        'SEDG',  # SolarEdge Technologies
        'RUN',   # Sunrun - residential solar
        'FSLR',  # First Solar
        'NOVA',  # Sunnova Energy
        
        # Wind and diversified
        'NEE',   # NextEra Energy
        'AY',    # Atlantica Sustainable Infrastructure
        
        # Electric vehicles and batteries
        'CHPT',  # ChargePoint Holdings
        'BLNK',  # Blink Charging
        'QS',    # QuantumScape - solid-state batteries
        
        # Clean energy ETFs for comparison
        'ICLN',  # iShares Global Clean Energy ETF
        'PBW',   # Invesco WilderHill Clean Energy ETF
        'QCLN',  # First Trust NASDAQ Clean Edge Green Energy
        'TAN',   # Invesco Solar ETF
    ]
    
    print(f"📊 Scanning {len(clean_energy_tickers)} clean energy tickers...")
    print(f"🎯 Criteria: Market Cap ${criteria.min_market_cap/1e6:.0f}M - ${criteria.max_market_cap/1e9:.1f}B")
    print(f"💰 Min Revenue Growth: {criteria.min_revenue_growth*100:.0f}%")
    print(f"📈 Min Gross Margin: {criteria.min_gross_margin*100:.0f}%\n")
    
    # Scan each ticker
    results = []
    for ticker in clean_energy_tickers:
        try:
            print(f"\n🔍 Analyzing {ticker}...")
            
            # Determine asset type
            asset_type = 'etf' if ticker in ['ICLN', 'PBW', 'QCLN', 'TAN'] else 'stock'
            
            # Fetch data
            data = data_pipeline.get_comprehensive_data(ticker, asset_type)
            
            if 'error' in data:
                print(f"  ⚠️  Could not fetch data: {data.get('error')}")
                continue
            
            # Calculate gem score
            gem_score = screener.calculate_composite_score(ticker, data)
            
            if gem_score.composite_score > 0:
                results.append(gem_score)
                print(f"  ✅ Score: {gem_score.composite_score:.1f}/100")
                print(f"     🌱 Sustainability: {gem_score.sustainability_score:.1f}/100")
                print(f"     🎯 Sector: {gem_score.sector_score:.1f}/100")
                print(f"     💼 Fundamental: {gem_score.fundamental_score:.1f}/100")
                print(f"     📊 Technical: {gem_score.technical_score:.1f}/100")
                print(f"     👁️  Visibility: {100 - gem_score.visibility_score:.1f}/100 (lower is better)")
            else:
                print(f"  ❌ Did not meet screening criteria")
                
        except Exception as e:
            print(f"  ⚠️  Error analyzing {ticker}: {str(e)}")
            logger.error(f"Error details for {ticker}: {e}", exc_info=True)
    
    # Display results
    print("\n" + "="*80)
    print("🏆 TOP CLEAN ENERGY HIDDEN GEMS")
    print("="*80 + "\n")
    
    if not results:
        print("❌ No opportunities found matching criteria.")
        print("\n💡 Tips to find more results:")
        print("   • Increase max market cap threshold")
        print("   • Decrease min revenue growth requirement")
        print("   • Decrease min gross margin requirement")
        print("   • Increase max analyst coverage")
        return
    
    # Sort by composite score
    results.sort(key=lambda x: x.composite_score, reverse=True)
    
    print(f"✅ Found {len(results)} opportunities!\n")
    
    # Display top 5
    for i, gem in enumerate(results[:5], 1):
        print(f"\n{'='*80}")
        print(f"#{i} {gem.ticker} - Composite Score: {gem.composite_score:.1f}/100")
        print(f"{'='*80}")
        print(f"Risk Rating: {gem.risk_rating}")
        print(f"\n📊 Score Breakdown:")
        print(f"  • Sector (Emerging Markets): {gem.sector_score:.1f}/100")
        print(f"  • Fundamental Strength: {gem.fundamental_score:.1f}/100")
        print(f"  • Technical Setup: {gem.technical_score:.1f}/100")
        print(f"  • Hidden (Low Visibility): {100 - gem.visibility_score:.1f}/100")
        print(f"  • Catalyst Potential: {gem.catalyst_score:.1f}/100")
        print(f"  • Smart Money Flow: {gem.smart_money_score:.1f}/100")
        print(f"  • 🌱 Sustainability/Impact: {gem.sustainability_score:.1f}/100")
        
        print(f"\n💡 Investment Thesis:")
        print(f"  {gem.investment_thesis}")
        
        print(f"\n🎯 Primary Catalyst:")
        print(f"  {gem.primary_catalyst}")
        
        # Show sustainability details if available
        if hasattr(gem, 'sustainability_data') and gem.sustainability_data:
            sust = gem.sustainability_data
            print(f"\n🌱 ESG Breakdown:")
            print(f"  • Environmental: {sust.environmental_score:.1f}/100")
            print(f"  • Social: {sust.social_score:.1f}/100")
            print(f"  • Governance: {sust.governance_score:.1f}/100")
            
            if sust.impact_categories:
                print(f"\n💚 Impact Areas: {', '.join(sust.impact_categories)}")
            
            if sust.impact_thesis:
                print(f"\n🌍 Impact Thesis: {sust.impact_thesis}")
        
        # Show action plan if available
        if gem.action_plan and 'entry_range' in gem.action_plan:
            action_plan = gem.action_plan
            print(f"\n📋 Action Plan:")
            if 'entry_range' in action_plan:
                print(f"  • Entry Range: ${action_plan['entry_range']['low']:.2f} - ${action_plan['entry_range']['high']:.2f}")
            if 'stop_loss' in action_plan:
                print(f"  • Stop Loss: ${action_plan['stop_loss']:.2f}")
            if 'targets' in action_plan:
                targets = action_plan['targets']
                print(f"  • 12M Target: ${targets['12_month']:.2f}")
                print(f"  • 24M Target: ${targets['24_month']:.2f}")
            if 'position_sizing' in action_plan:
                print(f"  • Position Size: {action_plan['position_sizing']}")
    
    print("\n" + "="*80)
    print("✅ Clean Energy Hidden Gems Analysis Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_clean_energy_gems()
