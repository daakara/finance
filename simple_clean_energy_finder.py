"""
Simple clean energy gems finder using direct yfinance calls
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def analyze_clean_energy_stock(ticker):
    """Analyze a clean energy stock for hidden gem potential"""
    try:
        print(f"\n{'='*70}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*70}")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic info
        name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        
        print(f"Company: {name}")
        print(f"Sector: {sector}")
        print(f"Industry: {industry}")
        print(f"Market Cap: ${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M")
        
        # Price data
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
        
        if current_price and fifty_two_week_high:
            distance_from_high = ((fifty_two_week_high - current_price) / fifty_two_week_high) * 100
            print(f"\nPrice: ${current_price:.2f}")
            print(f"52-Week Range: ${fifty_two_week_low:.2f} - ${fifty_two_week_high:.2f}")
            print(f"Distance from 52W High: {distance_from_high:.1f}%")
        
        # Fundamental metrics
        revenue_growth = info.get('revenueGrowth', 0)
        gross_margins = info.get('grossMargins', 0)
        profit_margins = info.get('profitMargins', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        
        print(f"\n📊 Fundamentals:")
        print(f"  Revenue Growth: {revenue_growth*100:.1f}%" if revenue_growth else "  Revenue Growth: N/A")
        print(f"  Gross Margin: {gross_margins*100:.1f}%" if gross_margins else "  Gross Margin: N/A")
        print(f"  Profit Margin: {profit_margins*100:.1f}%" if profit_margins else "  Profit Margin: N/A")
        print(f"  Debt/Equity: {debt_to_equity:.2f}" if debt_to_equity else "  Debt/Equity: N/A")
        
        # Analyst coverage
        num_analysts = info.get('numberOfAnalystOpinions', 0)
        target_price = info.get('targetMeanPrice', 0)
        
        print(f"\n👥 Analyst Coverage:")
        print(f"  Number of Analysts: {num_analysts}")
        if target_price and current_price:
            upside = ((target_price - current_price) / current_price) * 100
            print(f"  Target Price: ${target_price:.2f} ({upside:+.1f}% upside)")
        
        # ESG Scores
        esg_scores = info.get('esgScores', {})
        if esg_scores:
            print(f"\n🌱 ESG Scores:")
            print(f"  Environmental: {esg_scores.get('environmentScore', 'N/A')}")
            print(f"  Social: {esg_scores.get('socialScore', 'N/A')}")
            print(f"  Governance: {esg_scores.get('governanceScore', 'N/A')}")
            print(f"  Total ESG: {esg_scores.get('totalEsg', 'N/A')}")
        
        # Calculate a simple hidden gem score
        score = 0
        reasons = []
        
        # Small to mid cap (hidden gem territory)
        if 50e6 < market_cap < 10e9:
            score += 20
            reasons.append("✓ Market cap in sweet spot ($50M-$10B)")
        
        # Strong growth
        if revenue_growth and revenue_growth > 0.20:
            score += 20
            reasons.append(f"✓ Strong revenue growth ({revenue_growth*100:.1f}%)")
        
        # Healthy margins
        if gross_margins and gross_margins > 0.30:
            score += 15
            reasons.append(f"✓ Healthy gross margins ({gross_margins*100:.1f}%)")
        
        # Under the radar (low analyst coverage)
        if num_analysts < 15:
            score += 15
            reasons.append(f"✓ Under the radar ({num_analysts} analysts)")
        
        # Price discount from 52W high
        if current_price and fifty_two_week_high:
            if distance_from_high > 20:
                score += 15
                reasons.append(f"✓ Trading at discount ({distance_from_high:.1f}% from high)")
        
        # Analyst upside
        if target_price and current_price:
            if upside > 25:
                score += 15
                reasons.append(f"✓ Strong analyst upside ({upside:.1f}%)")
        
        print(f"\n💎 Hidden Gem Score: {score}/100")
        if reasons:
            print(f"\n✅ Positive Factors:")
            for reason in reasons:
                print(f"  {reason}")
        
        # Get price history for technical analysis
        hist = stock.history(period="6mo")
        if not hist.empty:
            sma_50 = hist['Close'].tail(50).mean()
            sma_200 = hist['Close'].tail(200).mean() if len(hist) >= 200 else None
            volume_avg = hist['Volume'].tail(20).mean()
            
            print(f"\n📈 Technical:")
            print(f"  50-day SMA: ${sma_50:.2f}")
            if sma_200:
                print(f"  200-day SMA: ${sma_200:.2f}")
                if current_price > sma_50 > sma_200:
                    print(f"  ✓ Golden cross pattern (bullish)")
            print(f"  Avg Volume (20d): {volume_avg:,.0f}")
        
        return {
            'ticker': ticker,
            'name': name,
            'score': score,
            'market_cap': market_cap,
            'price': current_price,
            'reasons': reasons
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("🔋 CLEAN ENERGY HIDDEN GEMS SCANNER")
    print("="*70)
    
    # Clean energy tickers to analyze
    clean_energy_tickers = [
        'ENPH',  # Enphase Energy
        'SEDG',  # SolarEdge
        'RUN',   # Sunrun
        'NOVA',  # Sunnova
        'FSLR',  # First Solar
        'CHPT',  # ChargePoint
        'BLNK',  # Blink Charging
        'QS',    # QuantumScape
        'PLUG',  # Plug Power
        'BEAM',  # Beam Global
    ]
    
    results = []
    for ticker in clean_energy_tickers:
        result = analyze_clean_energy_stock(ticker)
        if result and result['score'] > 0:
            results.append(result)
    
    # Show top opportunities
    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n" + "="*70)
        print("🏆 TOP CLEAN ENERGY HIDDEN GEMS")
        print("="*70)
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n#{i}. {result['ticker']} - {result['name']}")
            print(f"     Score: {result['score']}/100")
            print(f"     Price: ${result['price']:.2f}")
            print(f"     Market Cap: ${result['market_cap']/1e9:.2f}B" if result['market_cap'] > 1e9 else f"     Market Cap: ${result['market_cap']/1e6:.0f}M")
    else:
        print("\n❌ No strong hidden gem opportunities found in current market conditions")
        print("💡 Consider expanding search criteria or monitoring for better entry points")

if __name__ == "__main__":
    main()
