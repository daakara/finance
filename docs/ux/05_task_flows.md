# ARX End-to-End Task Flows & Dead-End Recovery

## 1. Flow 1: Novice Goal-Driven Opportunity Discovery

```
[ HOME ]
  User lands with €500 looking for an investment
  ↓ Clicks "🔎 Find Opportunities" (Show me stocks worth researching)
[ SCREENER ]
  User sees 6 Goal Pills → Clicks "📈 Growing Companies"
  Screener updates shortlist: [ FIX, EME, PWR ]
  ↓ Clicks "FIX" card
[ TERMINAL (/stock/fix) ]
  Guided View loads:
  • Headline: "Interesting, but not ready yet"
  • Invalidation Risk: "Price is below 50-day average ($1,732.86). A drop below $1,445 weakens setup."
  • What Would Change This: "Reclaiming $1,732.86 on above-average volume."
  • Action: [ Set Alert for $1,733 ] [ Add to Watchlist ]
  ↓ User decides to keep exploring
  Clicks [ ← Back to "Growing Companies" Search (3 saved) ]
[ SCREENER ]
  User returns with exact filter state preserved.
```

---

## 2. Flow 2: Intermediate Stock Evaluation & Position Sizing

```
[ OMNISEARCH / DIRECT LINK ]
  User searches "NVDA"
  ↓
[ TERMINAL (/stock/nvda) ]
  Standard View loads:
  • Setup Score: 84 / 100 (High Confluence)
  • Confluence Breakdown: Chart 88%, Health 90%, Flow 75%, Macro 80%
  • Key Levels: Buy Zone $118–$122, Stop Loss $112 (-7%), Target 1 $144 (+20%), R:R 2.85:1
  ↓ User clicks [ Size Position ]
[ POSITION SIZER MODAL ]
  • Enter Account Equity: $5,000 | Risk Budget: 1.0% ($50 Max Risk)
  • Output: Buy 8.33 Shares ($999.60 Value) | Exact Stop: $112.00
  ↓ Clicks [ Add to Paper Portfolio ]
[ PORTFOLIO ]
  Position added to client storage; Cash reserves updated automatically.
```

---

## 3. Flow 3: Existing Position Health Review ("I Own It")

```
[ PORTFOLIO ]
  User clicks existing holding "AAPL" (Cost Basis $195.00, Current $217.85)
  ↓
[ TERMINAL (/stock/aapl) ]
  Terminal detects `isHeldInPortfolio === true`:
  • Badge: 💼 ACTIVE HOLDING (0.045 Shares, +11.7% Gain)
  • Posture: 🟢 HOLD / THESIS INTACT
  • Guidance: "Earnings growth and institutional flow remain supportive. Trend line intact."
  • Risk Alert Floor: "Review holding if price drops below $212.00 (50D SMA)."
```

---

## 4. Flow 4: Insufficient Evidence Safe Handling

```
[ SEARCH / UNKNOWN TICKER ]
  User enters low-liquidity or unlisted ticker "XYZ"
  ↓
[ TERMINAL ]
  Assessment Engine detects missing fundamental metrics or stale filings.
  • Assessment: ⚠️ INSUFFICIENT_EVIDENCE
  • Headline: "Assessment Unavailable for XYZ"
  • Transparency Explanation:
    - Missing 10-K SEC financial filings
    - Insufficient 50-day average trading volume
  • Safe Options:
    [ Set Volume Alert ]  [ Research Company Profile ]  [ Explore Liquid Peers ]
```
