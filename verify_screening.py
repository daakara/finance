"""
Quick verification test with new optimal criteria
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria

def main():
    print("\n" + "="*70)
    print("🔍 VERIFYING NEW OPTIMAL CRITERIA")
    print("="*70)
    
    # Use the new default criteria
    criteria = GemCriteria()  # This should use the new defaults
    
    print(f"\n📊 Default Criteria Values:")
    print(f"  Max Market Cap: ${criteria.max_market_cap/1e9:.1f}B")
    print(f"  Min Revenue Growth: {criteria.min_revenue_growth*100:.0f}%")
    print(f"  Min Gross Margin: {criteria.min_gross_margin*100:.0f}%")
    print(f"  Max Analyst Coverage: {criteria.max_analyst_coverage}")
    print(f"  Max Debt/Equity: {criteria.max_debt_equity}")
    
    screener = HiddenGemScreener(criteria)
    
    # Test with a small set of known tickers
    test_tickers = ['ENPH', 'BLNK', 'PLUG', 'CHPT', 'COIN']
    
    print(f"\n🔍 Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    print("\nScreening...")
    
    results = screener.screen_universe(test_tickers)
    
    print(f"\n{'='*70}")
    print(f"✅ RESULTS: Found {len(results)} opportunities")
    print(f"{'='*70}")
    
    if results:
        print("\n🏆 Opportunities Found:")
        for i, gem in enumerate(results, 1):
            print(f"\n  {i}. {gem.ticker}")
            print(f"     Composite Score: {gem.composite_score:.1f}/100")
            print(f"     Risk Rating: {gem.risk_rating}")
            print(f"     Sustainability: {gem.sustainability_score:.1f}/100")
            print(f"     Sector Score: {gem.sector_score:.1f}/100")
    else:
        print("\n❌ No opportunities found - criteria may still be too strict")
        print("\n💡 Debugging recommendations:")
        print("  1. Check if data fetching is working")
        print("  2. Verify yfinance is returning valid data")
        print("  3. Review screening logic in gem_screener.py")
    
    print("\n" + "="*70)
    print("✅ Verification Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
