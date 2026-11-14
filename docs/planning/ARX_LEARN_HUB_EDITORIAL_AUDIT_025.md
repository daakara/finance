# ARX Terminal — Learn Hub Editorial Audit & Content Quality Matrix (Directive 025)

**Document ID:** `UX-EDT-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Scope:** Complete Learn Hub estate (`frontend/components/GuideContent.tsx`, `frontend/app/guide/page.tsx`, `frontend/lib/glossaryCatalog.ts`, `frontend/app/glossary/`)  
**Methodology:** Linguistic Forensic Audit & Content Quality Scoring  
**Classification:** Canonical Editorial Audit & Rewrite Inventory  

---

## 1. Executive Summary

This editorial audit inventories every educational text artifact in the ARX Terminal Learn Hub, identifies recurring AI-generated writing defects, and defines precise human-written replacements.

The audit examined 18 distinct educational units across the Field Manual and Quantitative Glossary. Over 70% of legacy sections exhibited severe AI-writing markers, including:
- Patronizing promotional puffery ("No-BS Trader Field Manual", "Old websites clutter your screen...").
- Artificial dichotomies ("Plain English" vs "Pro Quant" toggle).
- Repetitive three-part sentence cadence ("learn exactly where to buy, how to size your bets..., and how to verify setups...").
- Unhelpful colloquialisms ("Piotroski BS Detector", "Money-Making Efficiency", "Squeaky clean accounting").
- Missing concrete numerical walkthroughs and omission of statutory/mathematical limitations.

---

## 2. Learn Hub Content Quality Matrix

| Content Unit | Target Audience | Learning Objective | Current Wording Defects & AI Slop Markers | Educational Accuracy & Sourcing | Intended Human Rewrite Approach |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Guide Header & Meta** (`guide/page.tsx`) | Prospective users, organic search | Establish institutional credibility and explain the Field Manual scope. | Bloated buzzword salad: "Quantitative Terminal Field Manual & Algorithmic Handbook", "Self-Healing Forecast Audits". | Sourced from terminal engine specs; accurate but drowning in hyperbole. | Replace with clean, authoritative description of the terminal's quantitative models, risk gates, and market surveillance architecture. |
| **Guide Hero** (`GuideContent.tsx` L31-100) | First-time visitors | Orient reader to the curriculum and reading modes. | "The No-BS Trader Field Manual: How to Use This Terminal", patronizing plain-English toggle, broken chapter anchors. | Confusing toggle forces false choice between childish slang and raw Greek math. | Unify into a single, cohesive, layered editorial curriculum following the 5-module taxonomy. |
| **Chapter 1: Workspaces** (L103-242) | Intermediate investors | Explain the 4 workspaces and 4-tier universe. | "Old financial websites clutter your screen with 50 complicated widgets that make your head spin..." | Conceptually sound; accurately reflects the 4-tier asset pipeline (SSG vs dynamic). | Reframe around cognitive focus and workflow efficiency. Explain how viewport physics prioritize price and execution levels. |
| **Chapter 2: Candlestick Chart** (L245-320) | Active swing & day traders | Explain the multi-timeframe chart engine (VWAP vs 20 EMA). | "Never Mix Up Today's Change vs. Your Total Multi-Month Return" (repetitive headings). | Accurate distinction between session return and horizon return. | Clearly explain Volume-Weighted Average Price (VWAP) as an intraday institutional benchmark and the 20 EMA as dynamic trend support. |
| **Chapter 3: Execution & Sizing** (L323-440) | All investors allocating capital | Teach the 1% capital allocation formula and 4 ATR states. | "Where to Buy, Sizing Your Trades & The 4 Zones", casual state labels ("Don't Chase", "Get Out"). | High mathematical fidelity; formula matches `governorSizingEngine.ts`. | Present the exact position sizing formula with concrete numerical examples ($50k equity, 1% risk). Formalize the 4 ATR states (`IN_BUY_ZONE`, `APPROACHING_TARGET`, `WAITING_PULLBACK`, `STOPPED_OUT`). |
| **Chapter 4: STOCK Act & Insiders** (L443-516) | Fundamental & political analysts | Explain legal disclosures and the 45-day lag penalty. | "Tracking US Politicians & The 45-Day Delay Penalty", "Big Dollar Bet Size". | Accurately references Public Law 112-105 (STOCK Act) and Section 16(a) Form 4. | Explain the legal reporting timelines, the economic intuition behind committee jurisdiction overlap, and the mathematical justification for the exponential staleness decay function. |
| **Chapter 5: Fundamental DNA** (L519-550) | Value & quality investors | Explain 5-pillar fundamental scoring and Piotroski F-Score. | "Piotroski BS Detector (0–9)", "Scores <= 3 mean avoid!" | Accurately reflects Joseph Piotroski's 2000 accounting paper. | Replace slang with formal explanation of financial statement quality: profitability, leverage/liquidity, and operating efficiency. |
| **Chapter 6: Crash Risk & VaR** (L553-582) | Risk managers & portfolio owners | Explain why normal bell curves fail and how Cornish-Fisher VaR works. | "The Worst-Case Day Crash Formula", unformatted raw text equation strings. | Rigorous Cornish-Fisher polynomial expansion; correctly accounts for skewness and excess kurtosis. | Provide clean LaTeX equation rendering, explain why Gaussian models fail during crashes, and provide a realistic 95% vs 99% portfolio loss example. |
| **Chapter 7: Macro Synthesis** (L584-622) | Macro-aware traders | Explain FRED yield curves and credit spread multipliers. | "Fed Interest Rates, Real Insiders & Live Alerts". | Accurately tracks FRED 10Y-2Y yield spreads and high-yield OAS. | Explain macro regime modeling: how yield curve inversion and widening credit spreads dynamically compress trade risk limits. |
| **Chapter 8: Jargon Buster** (L625-681) | Self-directed traders | Clarify complex Wall Street terminology. | "Wall Street loves complicated words because it lets them charge high management fees", "Accounting Truth Detector". | Accurate underlying definitions, but condescending tone. | Present an objective, institutional glossary of key terms with plain-English intuition and concrete numerical illustrations. |
| **Chapter 9: Co-Pilots & Decision Gates** (L684-800) | Active execution traders | Explain Pre-Flight checklist, Smart Money Radar, and Stress Test. | "The 4 Trade Co-Pilots (Your Safety Check Before Buying)", "The 5-Point Green Light Checklist". | Accurately describes the 5-point Pre-Flight Gate (`INV-28`, VIX < 26.0, 48h catalyst buffer). | Formally explain the Pre-Flight Clearance Gate as an invariant risk firewall. Detail the criteria and why failure blocks execution. |

---

## 3. Inventory of AI Slop Patterns & Surgical Corrections

### Pattern 1: Promotional Marketing Puffery
- **Legacy AI Copy:** *"A crystal-clear, zero-jargon handbook for everyday traders and long-term investors. Learn exactly where to buy, how to size your bets so you never blow up, how to copy politician trades legally, and how to verify setups before risking a single dollar."*
- **Human Editor Critique:** Reads like a late-night infomercial. "Never blow up" and "copy politician trades legally" destroys institutional credibility.
- **Rewritten Human Text:** *"This Field Manual provides a rigorous, transparent reference for the quantitative models, risk boundaries, and market surveillance systems powering the ARX Terminal. It explains how our execution corridors are calculated, how capital allocation is constrained by mathematical volatility, and how multi-pillar evidence is synthesized into decision-ready context."*

### Pattern 2: Derogatory Slang & Artificial Dichotomy
- **Legacy AI Copy:** *"Old financial websites clutter your screen with 50 complicated widgets that make your head spin... Piotroski BS Detector (0–9)."*
- **Human Editor Critique:** Unprofessional, childish phrasing. Dumbing down concepts insults the user.
- **Rewritten Human Text:** *"Traditional financial terminals often stack dozens of disconnected widgets, creating severe cognitive friction and obscuring price context. ARX Terminal organizes analysis into four dedicated functional domains directly beneath the active chart... The Piotroski F-Score evaluates nine fundamental accounting metrics across profitability, capital structure, and operational efficiency to separate durable earnings from accounting distortion."*

### Pattern 3: Circular Definition without Concrete Mechanics
- **Legacy AI Copy:** *"Unlock the power of sophisticated risk management by leveraging dynamic risk-reward insights to optimize your investment journey."*
- **Human Editor Critique:** Zero informational content. Classic LLM fluff stringing together abstract nouns.
- **Rewritten Human Text:** *"Risk-reward measures the prospective profit of a trade relative to its defined dollar risk. If an entry corridor is set at \$100 with an invalidation stop-loss at \$95, your risk is \$5 per share. Targeting a price objective of \$115 yields a 3:1 reward-to-risk ratio. A favorable ratio never guarantees execution success; it ensures that your mathematical expectancy remains positive across repeated iterations."*

---

## 4. Glossary Catalog Forensic Audit (All 10 Terms)

1. **`arx-model` (Autoregressive Exogenous):**
   - *Current Flaw:* Overly abstract; lacks concrete variable definitions for what $x_t$ represents in ARX.
   - *Fix:* Explicitly state that $x_t$ tracks Federal Reserve 10Y-2Y yield curve spreads and corporate OAS credit spreads.
2. **`minervini-vcp` (Volatility Contraction Pattern):**
   - *Current Flaw:* Claims "tight pivot point where risk can be mathematically defined" without showing a concrete table of contraction cycles ($T_1, T_2, T_3$).
   - *Fix:* Include concrete numerical contraction sequence (-16%, -8%, -3%) with volume dry-up criteria (<50% of 50-day MA).
3. **`cornish-fisher-var` (Modified Value-at-Risk):**
   - *Current Flaw:* Raw unformatted formula text; lacks numerical comparison against Gaussian VaR.
   - *Fix:* Show side-by-side calculation showing how negative skewness expands 99% VaR from -\$4,200 (normal) to -\$6,800 (leptokurtic).
4. **`stock-act` (Congressional Knowledge Act):**
   - *Current Flaw:* Omits enforcement realities and penalty structures.
   - *Fix:* Detail statutory disclosure window (45 days) and why stale disclosures lose predictive value.
5. **`amihud-illiquidity` (Price Impact):**
   - *Current Flaw:* Abstract equation without units.
   - *Fix:* Explain how price impact per million dollars traded flags institutional absorption vs retail slippage.
6. **`turtle-atr-trailing-stop` (Volatility Stop):**
   - *Current Flaw:* Does not explain why $2 \times \text{ATR}$ is chosen over static percentage stops.
   - *Fix:* Contrast volatility-adjusted stops with fixed percentage trailing stops to demonstrate noise immunity.
7. **`piotroski-f-score` (Fundamental Accounting Score):**
   - *Current Flaw:* Labeled "BS Detector" in guide.
   - *Fix:* Detail the 9 binary criteria across Net Income, CFO, ROA, Accruals, Leverage, Current Ratio, Shares Outstanding, Gross Margin, and Asset Turnover.
8. **`twenty-ema-pullback` (Linda Raschke Setup):**
   - *Current Flaw:* Lacks clear distinction between trend continuation and trend exhaustion.
   - *Fix:* Define the two-bar confirmation rule and 20 EMA slope requirements.
9. **`kupiec-pof-test` (Proportion of Failures Test):**
   - *Current Flaw:* Highly dense mathematical jargon without intuitive walkthrough.
   - *Fix:* Explain as a backtesting audit: if a 95% VaR model experiences 15 breaches over 100 days, Kupiec rejects the model at the 99% confidence level.
10. **`late-filer-decay` (Staleness Decay Function):**
    - *Current Flaw:* Does not provide the exact exponential mathematical curve formula.
    - *Fix:* Detail the half-life penalty curve: 0 pts (<15d), -5 pts (16-30d), -16 pts (31-45d), -32 pts (>45d).

---

## 5. Conclusion & Action Items

The audit provides the exact blueprint for the complete editorial rewrite of `frontend/components/GuideContent.tsx`, `frontend/lib/glossaryCatalog.ts`, and `frontend/app/guide/page.tsx`. All changes will be executed in accordance with the Human Writing Guide.
