# ARX Information Architecture & Navigation Topology

## 1. Global Sitemap & Route Hierarchy

```
ARX TERMINAL
│
├── / (Home / Intent Portal)
│   ├── Intent Action Router (Find · Analyze · Compare · Portfolio Risk)
│   ├── Market Regime Radar (SPX, VIX, 10Y Yield, Risk Regime)
│   ├── High-Confluence Alpha Spotlight (Top 3 setups)
│   └── Portfolio Health Snapshot
│
├── /screener (Opportunity Discovery)
│   ├── 6 Goal-Driven Opportunity Filters (Growing, Undervalued, Momentum, High R:R)
│   ├── Dual-Horizon Archetype Engine (Day Trader High Beta vs. Long-Term Compounders)
│   └── Candidate Shortlist & CSV Data Export
│
├── /stock/[ticker] (Adaptive Stock Terminal)
│   ├── Dynamic Adaptive View (Guided · Standard · Advanced)
│   ├── Factor Score Attribution Modal ("Why?")
│   ├── Multi-Horizon Toggle (Intraday, Swing, Position, Long-Term)
│   └── Position Sizer & Invalidation Alert Modals
│
├── /compare & /compare/[pair] (Multi-Asset Comparison)
│   ├── 4-Asset Side-by-Side Matrix
│   └── Objective Fit Scoring (Best for Growth vs Value vs Quality)
│
├── /portfolio (Zero-Login Local Portfolio)
│   ├── Account Net Worth & Cash Reserve Ribbon
│   ├── Multi-Asset Position Tracker with Stop/Target Triggers
│   └── Cornish-Fisher VaR & Stress Test Simulator
│
├── /smart-money (Institutional Flow Radar)
│   ├── Congressional STOCK Act Trades with plain-English context
│   └── SEC Form 4 Insider Net Transactions
│
└── /guide (Methodology Reference)
    └── Plain-English glossary of quantitative models and indicators
```

---

## 2. Navigation Taxonomy & Breadcrumbs

To eliminate navigation disorientation, every sub-view contains persistent breadcrumbs:

```
Home > Screener > Growing Companies > FIX (Terminal)
[ ← Back to "Growing Companies" Screener (3 candidates saved) ]
```

---

## 3. Responsive Content Reflow Rules

| Device Breakpoint | Width | Navigation Behavior | Terminal Reflow |
| :--- | :--- | :--- | :--- |
| **Workstation** | $\ge 1440\text{px}$ | Full topbar + horizontal intent cards + persistent sidebar | 3-column split (Chart + Factors + Execution Ladder) |
| **Laptop / Tablet Landscape** | $1024\text{px} - 1439\text{px}$ | Compact topbar + 2x2 intent grid | 2-column split (Chart + Tabbed Evidence Cards) |
| **Tablet Portrait** | $768\text{px} - 1023\text{px}$ | Sticky topbar + drawer menu | Stacked layout (Chart on top, Evidence below) |
| **Mobile** | $\le 767\text{px}$ | Compact header + bottom navigation bar | **Vertical Decision Narrative**: Verdict $\rightarrow$ Why $\rightarrow$ Risk $\rightarrow$ Action $\rightarrow$ Expandable Evidence |
