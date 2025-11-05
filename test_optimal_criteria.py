"""
Test different screening criteria to find optimal defaults
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria
from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
import logging

logging.basicConfig(level=logging.WARNING)

def test_criteria(name, criteria, tickers):
    """Test a set of criteria and return results count"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    print(f"Max Market Cap: ${criteria.max_market_cap/1e9:.1f}B")
    print(f"Min Revenue Growth: {criteria.min_revenue_growth*100:.0f}%")
    print(f"Min Gross Margin: {criteria.min_gross_margin*100:.0f}%")
    print(f"Max Analyst Coverage: {criteria.max_analyst_coverage}")
    
    screener = HiddenGemScreener(criteria)
    results = screener.screen_universe(tickers)
    
    print(f"\n✅ Found {len(results)} opportunities")
    
    if results:
        # Show top 3
        for i, gem in enumerate(results[:3], 1):
            print(f"\n  #{i}. {gem.ticker} - Score: {gem.composite_score:.1f}/100")
            print(f"      Risk: {gem.risk_rating}")
    
    return len(results), results

def main():
    print("\n" + "="*70)
    print("🔬 TESTING OPTIMAL SCREENING CRITERIA")
    print("="*70)
    
    # Test universe - a good mix
    test_tickers = [
        # Clean energy (from our demo)
        'ENPH', 'SEDG', 'RUN', 'FSLR', 'CHPT', 'BLNK', 'PLUG',
        # Tech small/mid caps
        'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'MDB',
        # Biotech
        'MRNA', 'BNTX', 'BEAM',
        # Crypto-related
        'COIN', 'MSTR', 'IREN',
        # Others
        'ROKU', 'SOFI', 'HOOD', 'AFRM',
    ]
    
    print(f"\n📊 Testing with {len(test_tickers)} diverse tickers")
    
    # Test 1: Very Relaxed (should get lots of results)
    criteria1 = GemCriteria(
        min_market_cap=10e6,
        max_market_cap=100e9,  # $100B
        min_revenue_growth=0.0,  # 0%
        min_gross_margin=0.0,  # 0%
        max_analyst_coverage=50,
    )
    count1, results1 = test_criteria("Very Relaxed", criteria1, test_tickers)
    
    # Test 2: Moderately Relaxed
    criteria2 = GemCriteria(
        min_market_cap=50e6,
        max_market_cap=50e9,  # $50B
        min_revenue_growth=0.0,  # 0%
        min_gross_margin=0.10,  # 10%
        max_analyst_coverage=30,
    )
    count2, results2 = test_criteria("Moderately Relaxed", criteria2, test_tickers)
    
    # Test 3: Balanced (RECOMMENDED)
    criteria3 = GemCriteria(
        min_market_cap=50e6,
        max_market_cap=30e9,  # $30B
        min_revenue_growth=0.0,  # 0% (many companies have negative growth)
        min_gross_margin=0.15,  # 15%
        max_analyst_coverage=25,
    )
    count3, results3 = test_criteria("Balanced (RECOMMENDED)", criteria3, test_tickers)
    
    # Test 4: Current defaults
    criteria4 = GemCriteria(
        min_market_cap=50e6,
        max_market_cap=15e9,  # $15B
        min_revenue_growth=0.15,  # 15%
        min_gross_margin=0.20,  # 20%
        max_analyst_coverage=20,
    )
    count4, results4 = test_criteria("Current Defaults", criteria4, test_tickers)
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"\nVery Relaxed:          {count1} opportunities")
    print(f"Moderately Relaxed:    {count2} opportunities")
    print(f"Balanced (RECOMMENDED): {count3} opportunities ⭐")
    print(f"Current Defaults:      {count4} opportunities")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    print("\n🎯 Use BALANCED criteria for best results:")
    print("  • Max Market Cap: $30B (captures mid-caps)")
    print("  • Min Revenue Growth: 0% (many quality companies have temporary negative growth)")
    print("  • Min Gross Margin: 15% (realistic for various industries)")
    print("  • Max Analyst Coverage: 25 (still relatively under-followed)")
    print("\nThis gives a good balance between quality and quantity of opportunities.")
    
    if results3:
        print("\n🏆 Top Balanced Results:")
        for i, gem in enumerate(results3[:5], 1):
            print(f"\n  #{i}. {gem.ticker}")
            print(f"      Score: {gem.composite_score:.1f}/100")
            print(f"      Risk: {gem.risk_rating}")
            print(f"      Thesis: {gem.investment_thesis[:100]}...")

if __name__ == "__main__":
    main()
