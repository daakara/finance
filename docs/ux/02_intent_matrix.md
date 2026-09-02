# ARX User Job & Intent Matrix

## 1. Intent Model

ARX structures user interaction around **four primary user tasks**, separated cleanly from object state or cognitive presentation modes.

```typescript
export type UserIntent =
  | "DISCOVER"          // "Find stocks matching my investment or trading goals"
  | "ANALYZE"           // "Understand why a stock is moving and evaluate thesis health"
  | "COMPARE"           // "Decide between multiple competing opportunities"
  | "MANAGE_PORTFOLIO"; // "Inspect portfolio concentration, VaR risk, and position health"
```

---

## 2. Intent to Sub-Job Mapping

| Primary Intent | User Job (What the User is Trying to Accomplish) | Entry Point | Primary Destination |
| :--- | :--- | :--- | :--- |
| **`DISCOVER`** | • Find high-conviction breakout setups<br/>• Find profitable growing compounders (Lynch GARP)<br/>• Find deeply undervalued cash-flow bargains (Greenblatt)<br/>• Find asymmetric setups with tight risk floors (R:R $\ge 2:1$) | Home Intent Card<br/>Navbar Link | `/screener` with Goal Filters |
| **`ANALYZE`** | • Understand what a specific company does<br/>• Check if a ticker is currently actionable<br/>• Verify why ARX views a stock as Favorable or Mixed<br/>• Identify downside invalidation levels before entering | Home Quick Ticker<br/>OmniSearch (`Cmd+K`)<br/>Direct URL | `/stock/[ticker]` (Terminal) |
| **`COMPARE`** | • Compare up to 4 peer assets side-by-side<br/>• Identify which stock is best for Growth vs Value vs Momentum vs Quality<br/>• Evaluate factor correlation and relative strength | Home Intent Card<br/>Navbar Link | `/compare` & `/compare/[pair]` |
| **`MANAGE_PORTFOLIO`** | • Review position size allocation across holdings<br/>• Check portfolio-level Cornish-Fisher VaR (95%)<br/>• Detect macro sensitivity and sector concentration<br/>• Rebalance cash reserves vs active stock risk | Home Risk Gauge<br/>Navbar Link | `/portfolio` |

---

## 3. Intent Context Preservation (Dead-End Recovery)

When a user navigates from `DISCOVER` (Screener) into `ANALYZE` (Terminal), the user's intent, filter parameters, and candidate shortlist are preserved in `JourneyContextState`:

```typescript
export interface JourneyContextState {
  intent: UserIntent;
  screenerGoalId?: string; // e.g. "growing" | "undervalued" | "high_confluence"
  shortlist: string[];     // e.g. ["FIX", "EME", "PWR"]
  selectedSymbol: string;
  returnPath: string;      // e.g. "/screener?goal=growing"
}
```

The Terminal UI renders a persistent recovery breadcrumb:  
`[ ← Back to "Growing Companies" Search (3 candidates saved) ]`  
Clicking the link restores the exact filter and scroll position without data loss.
