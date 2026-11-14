# ARX Terminal — Comprehensive UX Baseline Audit (Directive 025)

**Document ID:** `UX-AUD-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Methodology:** Project Brain Evidence-Based Audit & Multi-Viewport Inspection  
**Classification:** Canonical UX Baseline & Problem Inventory  
**Enforcement Authority:** UI Precedence `UI-0 > UI-1 > UI-2 > UI-3 > UI-4` & `rules/quant-invariants.md`

---

## 1. Executive Summary

This comprehensive baseline audit evaluates the user experience of the ARX Terminal across all desktop, tablet, and mobile presentations. It anchors the `ARX-UX-BRAIN-REDESIGN-025` initiative to transform ARX from a collection of powerful but disparate quantitative widgets into an institutional-grade financial decision platform.

The audit identifies three primary systemic friction areas:
1. **The Orphaned Learn Hub:** High-value educational materials (`/guide` and `/glossary`) are completely absent from global desktop navigation, the mobile navigation dock, and utility menus. Users cannot discover educational context without manual URL entry or Omnisearch.
2. **AI-Generated Copy & Jargon Slop:** The educational surfaces (`/guide`, `/glossary`) suffer from generic marketing puffery ("Unlock the power of...", "In today's fast-paced markets...", "Dive deep into..."), ungrounded acronym soup, and split vernacular toggles that patronize retail users while confusing professional analysts.
3. **Cognitive Overload & Discovery/Execution Ambiguity:** Multiple competing status badges across Radar, Analysis, and Setups create ambiguity between relative screening rank and trade execution eligibility.

---

## 2. Screen-by-Screen Diagnostic Evaluation

### Screen 1: Home / Terminal (`/`) & Single-Asset Analysis
- **1. What question brought the user here?** "Is this asset (e.g. NVDA, AAPL) worthy of capital right now, and what does the quantitative evidence say?"
- **2. What should the user understand in the first five seconds?**
  - Current spot price, market session status, and active ticker identity.
  - Overall Confluence conviction score (0–100) and whether the setup is actionable or in progress.
  - Immediate price relation to the 20 EMA and buy corridor.
- **3. Which information matters most?**
  - Decision state (e.g. `SETUP EVALUATION PENDING`, `NEAR SCREENING ZONE`).
  - Primary execution levels (Entry corridor, Stop-loss invalidation, Target 1).
  - Multi-pillar Confluence breakdown (Technical, Fundamental, Smart Money, Macro).
- **4. What action can the user legitimately take?**
  - Inspect multi-timeframe charts (1m to 5y).
  - Open the Position Sizer to calculate share quantity based on account risk.
  - Review Pre-Flight Clearance Gates (5-point checklist).
  - Transition to Trade Plan (`/setups`) or Portfolio (`/portfolio`).
- **5. Which elements are redundant, misleading or visually noisy?**
  - Excessive badges in the active ticker hero competing for visual attention.
  - Double reporting of percentage change (Daily vs Horizon) without immediate mathematical disambiguation in compact viewports.
  - Heavy borders and high-contrast widget dividers creating visual vibration.
- **6. Where does the interface assume expert knowledge?**
  - Abbreviations like "CF-VaR", "VCP", "ATS", "OAS Spreads", and "Piotroski F-Score" appear without immediate plain-language hover cards or context links.
- **7. What must remain visible for risk and decision integrity?**
  - Invalidation Stop-Loss price level.
  - Actionability status: Non-actionable assets must never display `READY TO BUY`.
  - Data freshness timestamp and quote authority source.

---

### Screen 2: Opportunity Radar (`/radar` & `/screener`)
- **1. What question brought the user here?** "What market opportunities deserve my analytical attention today?"
- **2. What should the user understand in the first five seconds?**
  - How many total assets were evaluated (e.g., 200 Stage 2 candidates out of 500 universe).
  - The top-ranked candidates sorted by Confluence score.
  - That high ranking does **not** equal an immediate buy order (Discovery $\ne$ Execution).
- **3. Which information matters most?**
  - Confluence Score (0–100) with color-coded risk bands (defensive rose for <48, cyan for 65–79, emerald for 80+).
  - Setup Category (Stage 2 VCP, GARP, Quality Compounder, Smart Money).
  - Screening Status (`NEAR SCREENING ZONE`, `MONITORING`, `OVEREXTENDED`).
- **4. What action can the user legitimately take?**
  - Filter by investment style or sector.
  - Click any ticker to transfer global context to Analysis (`/?symbol=TICKER`).
  - Switch between Card and Table layout modes.
- **5. Which elements are redundant, misleading or visually noisy?**
  - Table columns on smaller laptop screens cause horizontal scrolling without pinned ticker headers.
  - Redundant "Action" badges that mimic execution buttons but merely open search.
- **6. Where does the interface assume expert knowledge?**
  - Assumes users understand Stan Weinstein Stage 2 criteria and Mark Minervini Volatility Contraction Pattern geometry.
- **7. What must remain visible for risk and decision integrity?**
  - Clear banner stating that Radar is an observational discovery screener, not an automated buy recommendation engine.
  - Score provenance and snapshot publication timestamp (`build UB-ARX-EXPANSION-500`).

---

### Screen 3: Trade Plan & Setups (`/setups`)
- **1. What question brought the user here?** "What trades have satisfied all technical and risk criteria and are actionable right now?"
- **2. What should the user understand in the first five seconds?**
  - Which setups have cleared the Pre-Flight Gate.
  - Exact entry triggers and defined stop-loss invalidation prices.
- **3. Which information matters most?**
  - Asymmetric Reward-to-Risk ratio (minimum 2:1 target).
  - Exact capital allocation calculation based on risk budget.
  - Current price proximity to the entry trigger.
- **4. What action can the user legitimately take?**
  - Calculate exact share sizing using the Governor Sizing Engine.
  - Verify catalyst proximity (ensuring no binary earnings within 48h).
  - Mark setups for watchlist monitoring.
- **5. Which elements are redundant, misleading or visually noisy?**
  - Overly aggressive amber and green neon highlights that create artificial urgency.
- **6. Where does the interface assume expert knowledge?**
  - Assumes familiarity with ATR volatility banding and order execution types.
- **7. What must remain visible for risk and decision integrity?**
  - Fail-closed broker equity requirement: 0 shares calculated if equity is unverified (`INV-28`).
  - Warning when spot price has extended >2% past the optimal accumulation ceiling.

---

### Screen 4: Portfolio Risk Heatmap (`/portfolio`)
- **1. What question brought the user here?** "What total risk and systemic exposure am I carrying across my open positions?"
- **2. What should the user understand in the first five seconds?**
  - Net portfolio capital, open risk in dollar terms, and aggregate portfolio beta.
  - Which positions are currently breaching stop levels or over-concentrated.
- **3. Which information matters most?**
  - Cornish-Fisher 95% and 99% 1-day Modified Value-at-Risk ($).
  - Sector concentration and single-name exposure ceilings.
  - Macro stress scenario projections (e.g., -5% QQQ shock, +50bps yield shock).
- **4. What action can the user legitimately take?**
  - Rebalance simulated positions.
  - Adjust max capital risk per trade (1.0%–2.0%).
  - Trigger hypothetical macro stress simulations.
- **5. Which elements are redundant, misleading or visually noisy?**
  - Cluttered multi-color pie charts that lose legibility on mobile viewports.
- **6. Where does the interface assume expert knowledge?**
  - Assumes familiarity with covariance matrices, leptokurtic distribution, and Cornish-Fisher expansion.
- **7. What must remain visible for risk and decision integrity?**
  - 100% simulated execution warning (`HOLD_EXEC_001`).
  - Explicit warning that VaR represents statistical expectations under historical covariance, not absolute loss limits.

---

### Screen 5: Research & Opportunity Dossiers (`/research`)
- **1. What question brought the user here?** "What is the deep, fundamental, point-in-time thesis for this specific company?"
- **2. What should the user understand in the first five seconds?**
  - Whether a peer-reviewed research dossier exists for this company.
  - Intrinsic valuation range (Base, Bull, Bear) vs current market quote.
- **3. Which information matters most?**
  - ROIC quality moat, net debt bridge, and normalized free cash flow yield.
  - Disaggregation of statutory share buybacks from operational dilution.
- **4. What action can the user legitimately take?**
  - Read published research dossiers (e.g. ULTA, MEDP, ISRG, DECK).
  - Switch between scenario valuations (DCF, Multiples, Historical IRR).
- **5. Which elements are redundant, misleading or visually noisy?**
  - Excessive technical disclaimers printed before the executive summary.
- **6. Where does the interface assume expert knowledge?**
  - Assumes understanding of ASC 842 lease capitalization, WACC, and reinvestment rates.
- **7. What must remain visible for risk and decision integrity?**
  - Unrecorded review timestamps and explicit peer-review status badges.
  - Partition between zero-debt and funded-debt capital structures.

---

### Screen 6: Learn Hub Field Manual (`/guide`)
- **1. What question brought the user here?** "How do I master this terminal, understand its indicators, and make smarter trading decisions?"
- **2. What should the user understand in the first five seconds?**
  - A structured educational curriculum covering foundations, technical analysis, fundamentals, and using ARX.
  - That this is an authoritative, objective learning hub, not a promotional brochure.
- **3. Which information matters most?**
  - Clear conceptual explanations of key trading tools (Buy Zones, Position Sizer, VCP, Confluence, VaR).
  - Step-by-step interpretation guides and common pitfalls.
- **4. What action can the user legitimately take?**
  - Jump directly to specific chapters via quick-navigation links.
  - Toggle between Plain English and Pro Quant depth.
  - Click through to live terminal workflows.
- **5. Which elements are redundant, misleading or visually noisy?**
  - **Severe AI Slop & Hyperbole:** Sentences like "The No-BS Trader Field Manual: How to Use This Terminal", "Old financial websites clutter your screen with 50 complicated widgets that make your head spin...", and "Piotroski BS Detector".
  - Disorganized chapter numbering: Chapter 1 $\rightarrow$ Chapter 2 (Chart) $\rightarrow$ Chapter 3 (numbered `id="chapter-2"`) $\rightarrow$ Chapter 4 (numbered `id="chapter-3"`) $\rightarrow$ Chapter 7 $\rightarrow$ Chapter 8 $\rightarrow$ Chapter 9.
  - Split vernacular toggle creates artificial division rather than integrated, layered understanding.
- **6. Where does the interface assume expert knowledge?**
  - In Pro Quant mode, drops raw equations without step-by-step intuitive translation. In Plain English mode, dumbs down concepts excessively.
- **7. What must remain visible for risk and decision integrity?**
  - Explicit education-only disclaimer: Educational content is not individual investment advice.

---

### Screen 7: Learn Hub Quantitative Glossary (`/glossary` & `/glossary/[slug]`)
- **1. What question brought the user here?** "What does this specific financial or quantitative term mean, and how does ARX apply it?"
- **2. What should the user understand in the first five seconds?**
  - Plain-language definition of the term.
  - Mathematical formulation and key parameters.
  - Exactly where and how the metric is applied inside ARX Terminal.
- **3. Which information matters most?**
  - Short definition, concrete numerical example, and common misinterpretations.
- **4. What action can the user legitimately take?**
  - Search terms, filter by category, explore related terms.
  - Navigate to the terminal screen where the term is used.
- **5. Which elements are redundant, misleading or visually noisy?**
  - Overly academic preamble before the practical takeaway.
  - Missing cross-links from glossary terms directly into live terminal cards.
- **6. Where does the interface assume expert knowledge?**
  - Highly technical econometric formulas (e.g. Kupiec POF log-likelihood ratio, Cornish-Fisher polynomials) without concrete numerical walkthroughs.
- **7. What must remain visible for risk and decision integrity?**
  - Limitations of the model (e.g. why 95% VaR does not predict black swans).

---

### Screen 8: Global Navigation & Command Shell (`Navbar.tsx`, `UniversalOmniSearch.tsx`)
- **1. What question brought the user here?** "Where am I, how do I switch assets, and how do I navigate between analytical hubs?"
- **2. What should the user understand in the first five seconds?**
  - The canonical 4-hub journey: Radar (Find) $\rightarrow$ Analysis (Understand) $\rightarrow$ Trade Plan (Plan) $\rightarrow$ Portfolio (Manage), plus Research and Learn.
  - The active symbol context (e.g., NVDA).
- **3. Which information matters most?**
  - Clear active indicator for the current hub.
  - Fast search bar accessible via `/` or `Cmd+K`.
- **4. What action can the user legitimately take?**
  - Switch hubs with preserved asset context.
  - Search any ticker or topic.
  - Access system utilities (Shortcuts, Privacy, Cache Purge, Learn Hub).
- **5. Which elements are redundant, misleading or visually noisy?**
  - Sub-header Market Command Ribbon consumes vertical space without distinct utility on mobile.
  - Missing "Learn" link in top navigation forces users to rely on discovery by accident.
- **6. Where does the interface assume expert knowledge?**
  - Pro-trader keyboard shortcuts (`Alt+D`, `Alt+V`, `Alt+S`) are undiscoverable without hitting `?`.
- **7. What must remain visible for risk and decision integrity?**
  - Clear distinction between simulated mode and live market data.

---

## 3. Prioritized UX Defect Register

| Defect ID | Severity | Category | Screen / Surface | Description & User Impact | Remediation Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UX-DEF-001** | **P0** | **Information Architecture** | Global Shell (`Navbar.tsx`) | **Orphaned Learn Hub:** Zero links to `/guide` or `/glossary` in desktop navigation, mobile navigation dock, or utilities menu. | Add "Learn" to desktop navigation and mobile utilities drawer. |
| **UX-DEF-002** | **P0** | **Content Quality** | Learn Hub (`GuideContent.tsx`) | **AI Slop & Jargon Soup:** Pervasive marketing buzzwords, disjointed chapter IDs, childish vs hyper-dense dual toggles. | Rewrite entire guide using the 8-part human educational standard. |
| **UX-DEF-003** | **P1** | **Content Quality** | Glossary (`glossaryCatalog.ts`) | **Abstract Definitions:** Glossary entries lack realistic numerical examples, concrete limitations, and live terminal links. | Upgrade all 10 glossary terms with human-first explanations and numerical examples. |
| **UX-DEF-004** | **P1** | **Visual Hierarchy** | Analysis (`/`) | **Badge Overload:** Multiple status pills in the ticker hero compete for attention without clear decision hierarchy. | Group badges into Decision Status, Price/Context, and Evidence Badges. |
| **UX-DEF-005** | **P2** | **Accessibility** | Global Shell | **Touch Target Boundaries:** Some utility buttons in the header hover near the 40px boundary on compact screens. | Ensure all interactive targets strictly meet 44x44px (`UI-0`). |
| **UX-DEF-006** | **P2** | **Mobile Responsiveness**| Guide / Glossary | **Equation & Table Overflow:** LaTeX equations and multi-column tables risk horizontal overflow on 390px viewports. | Apply `overflow-x-auto` wrappers and responsive grid collapse. |
| **UX-DEF-007** | **P3** | **Discoverability** | Navigation | **Hidden Educational Context:** Key financial metrics on Analysis cards lack inline triggers to open relevant guide chapters. | Add contextual info icons linking directly to corresponding glossary terms. |

---

## 4. Conclusion & Next Steps

The baseline audit confirms that ARX Terminal possesses exceptional analytical depth and mathematical rigor, but suffers from fragmented navigation, editorial AI-slop in its educational surfaces, and cognitive noise in high-density analytical hubs.

Remediation will proceed through the Project Brain Design Critique and surgical implementation on `feat/ux-brain-redesign-025`.
