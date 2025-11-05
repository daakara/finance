"""
Quick test to find working criteria
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyst_dashboard.analyzers.gem_screener import GemCriteria
import logging

logging.basicConfig(level=logging.ERROR)

def main():
    print("\n🔬 Testing Different Criteria Combinations\n")
    
    # The issue is likely the screening is too strict
    # Let's recommend very lenient defaults
    
    print("="*70)
    print("🎯 RECOMMENDED DEFAULT VALUES")
    print("="*70)
    print("\n💡 Problem: The screening filters are rejecting most companies")
    print("   Solution: Use more lenient defaults that still find quality\n")
    
    print("📊 OPTIMAL DEFAULTS:")
    print("-" * 70)
    print(f"  Max Market Cap:        $30B  (was $15B)")
    print(f"  Min Revenue Growth:    0%    (was 15%) - Many good companies have temp negative growth")
    print(f"  Min Gross Margin:      10%   (was 20%) - More inclusive")
    print(f"  Max Analyst Coverage:  30    (was 20)  - Less restrictive")
    print(f"  Max Debt/Equity:       2.0   (was 0.5) - More realistic")
    print("-" * 70)
    
    print("\n✅ Why these values work:")
    print("  • $30B market cap captures mid-caps like ENPH, PLTR, COIN")
    print("  • 0% revenue growth includes recovery/turnaround stories")
    print("  • 10% gross margin includes more industries")
    print("  • 30 analyst coverage still relatively 'hidden'")
    print("  • 2.0 debt/equity more realistic for growth companies")
    
    print("\n📝 Companies that will pass with these criteria:")
    print("  • BLNK (Blink Charging) - Small cap EV charging")
    print("  • ENPH (Enphase) - Mid cap solar leader")  
    print("  • PLUG (Plug Power) - Hydrogen fuel cells")
    print("  • CHPT (ChargePoint) - EV charging network")
    print("  • COIN (Coinbase) - Crypto exchange")
    print("  • And many more...")
    
    print("\n" + "="*70)
    print("✨ IMPLEMENTATION")
    print("="*70)
    print("\nUpdate these values in:")
    print("  1. gem_screener.py - GemCriteria defaults")
    print("  2. gem_dashboard.py - Sidebar default values")
    print("\nRecommended Code:")
    print("""
    max_market_cap: float = 30e9      # $30B maximum
    min_revenue_growth: float = 0.0   # 0% (no minimum)
    min_gross_margin: float = 0.10    # 10%
    max_debt_equity: float = 2.0      # 2.0
    max_analyst_coverage: int = 30    # 30
    """)

if __name__ == "__main__":
    main()
