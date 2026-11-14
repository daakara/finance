# ARX Terminal — Comprehensive UX Redesign Specification (Directive 025)

**Document ID:** `UX-SPEC-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Architecture:** Next.js 14 App Router, Tailwind CSS, TypeScript  
**Design Authority:** `rules/ui-invariants.md`, `rules/quant-invariants.md`, `skills/ux-redesign-protocol`  
**Classification:** Canonical Engineering & Design System Specification  

---

## 1. Executive Summary & Design Philosophy

The ARX Terminal Redesign elevates the platform into an institutional, purpose-built quantitative decision engine. It eliminates generic "AI dashboard" tropes—such as random glowing gradients, decorative emoji-spam, bloated marketing hero banners, and patronizing binary language toggles—and establishes an unmistakable, disciplined, dark-first terminal aesthetic.

### Core Architectural Pillars
1. **Decision-First Hierarchy:** Every visual element is prioritized by its relevance to capital preservation and execution decision-making. Non-actionable assets are never styled with celebratory or urgent signals.
2. **First-Class Educational Integration:** The Learn Hub (`/guide` and `/glossary`) is elevated from an orphaned background route into a first-class member of the global shell, providing context-aware guidance throughout the terminal.
3. **WCAG 2.2 AA Conformance:** Strict 44x44px interactive target sizing, elevated slate-400 text contrast ratios (5.2:1 against dark surfaces), and full keyboard tab-order accessibility.

---

## 2. Institutional Design System Tokens

### 2.1 Color Palette & Semantic Tokens

```
Surface Base:      #090d14  (App canvas background)
Surface Card:      #0c1017  (Primary container surface)
Surface Elevated:  #111722  (Interactive cards, modal interiors)
Surface Highlight: #162030  (Hover states, active selections)
Border Subdued:    #1b2434  (Secondary element dividers)
Border Standard:   #243044  (Canonical container outlines)
Border Focused:    #38bdf8  (Active inputs, focus rings)

Text Primary:      #f8fafc  (Headlines, active ticker symbols)
Text Secondary:    #cbd5e1  (Body copy, quantitative labels)
Text Muted:        #94a3b8  (Metadata, timestamps - WCAG AA 5.2:1)

Signal Focus:      #38bdf8 / #22d3ee  (Cyan - Primary analytical focus)
Signal Valid:      #34d399 / #10b981  (Emerald - In buy zone, pristine accounting)
Signal Risk:       #f43f5e / #ef4444  (Rose - Stop loss, breached risk bounds)
Signal Caution:    #fbbf24 / #f59e0b  (Amber - Overextended, warning states)
```

### 2.2 Typography Scale & Tabular Precision
- **Tickers & Quotes:** `font-mono tracking-tight tabular-nums` (guarantees alignment across rapid price updates).
- **Body & Educational Text:** `font-sans leading-relaxed text-slate-300` (optimized for long-form reading comprehension).
- **Micro-Badges:** `font-mono uppercase text-[10px] tracking-wider px-2 py-0.5 rounded-md`.

### 2.3 Glassmorphism Restraint Policy (`UI-3`, `UI-4`)
- Restrained frosted glass is permitted **only** on sticky global overlays: `bg-[#0c1017]/95 backdrop-blur-md border-b border-[#243044]`.
- Content cards inside the viewport must use solid opaque surfaces (`#0c1017` and `#111722`) to avoid CPU-intensive composite re-rendering and maintain crisp text contrast on low-power mobile devices.

---

## 3. Global Navigation & Command Shell Blueprint

### 3.1 Desktop Navigation Header (`Navbar.tsx`)
The desktop navigation cleanly consolidates the platform's core surfaces:

```
[ARX TERMINAL]  [Radar] [Analysis] [Trade Plan] [Portfolio] [Research] [Learn]  [OmniSearch /]  [Theme] [⋯ Utilities]
```

- **Canonical Hubs (Find $\rightarrow$ Understand $\rightarrow$ Plan $\rightarrow$ Manage):**
  - **Radar (`/radar`):** Opportunity discovery and multi-factor screening.
  - **Analysis (`/`):** Deep single-asset quantitative valuation and DecisionTrace.
  - **Trade Plan (`/setups`):** Pre-flight gates and capital allocation execution.
  - **Portfolio (`/portfolio`):** Systemic risk heat and Cornish-Fisher VaR.
- **Deep Research:**
  - **Research (`/research`):** Peer-reviewed opportunity dossiers and fundamental models.
- **Institutional Education:**
  - **Learn (`/guide`):** Field Manual, algorithmic definitions, and quantitative curricula.
  - *Active state:* Highlighted with `bg-[#1b2434] text-cyan-400 font-semibold` when route matches `/guide` or `/glossary`.

### 3.2 Mobile Navigation Dock & Utilities Drawer
- **Mobile Bottom Dock (Fixed at viewport bottom):**
  - Buttons: **Radar**, **Analysis**, **Setups**, **Portfolio**, **Research**, and **Horizon Toggle (Day/Long)**.
- **Mobile Learn Hub Discovery:**
  - The Utilities Menu (`⋯`) prominently features **"Learn & Field Manual"** under the **HELP** section as the first item, complete with a dedicated Book/Compass icon linking to `/guide`.
  - The search palette (`UniversalOmniSearch`) indexes all Learn Hub chapters and glossary terms for instant mobile keyboard navigation.

---

## 4. Learn Hub Architecture & Editorial Specification

### 4.1 Curriculum Taxonomy (5 Canonical Modules)

```
MODULE 1: FOUNDATIONS
├── 1.1 Capital Preservation: The 1% Risk Allocation Rule
├── 1.2 Reading the Command Shell: Workspaces & Responsive Physics
└── 1.3 Active Asset Architecture: 4-Tier Universe (SSG vs Dynamic)

MODULE 2: TECHNICAL ANALYSIS & EXECUTION GEOMETRY
├── 2.1 Trend Templates: Stan Weinstein Stage 2 & Moving Average Alignment
├── 2.2 Volatility Contraction Pattern (VCP): Supply Exhaustion & Pivot Breakouts
├── 2.3 The 4 ATR Execution States: Buy Zone, Expanding, Extended, Stopped Out
└── 2.4 Multi-Timeframe Candlestick Engine: Intraday VWAP vs Macro 20 EMA

MODULE 3: FUNDAMENTAL PROFILES & ACCOUNTING FORENSICS
├── 3.1 Five-Pillar Fundamental DNA: Growth, Moat, and Valuation Ratios
├── 3.2 Piotroski 9-Point F-Score: Balance Sheet Truth vs Accounting Manipulation
└── 3.3 Return on Invested Capital (ROIC) vs Cash Conversion Quality

MODULE 4: SMART MONEY & REGULATORY FORENSICS
├── 4.1 SEC Form 4 Insider Tracking: Open-Market C-Suite Purchases
├── 4.2 Congressional STOCK Act Disclosures: Committee Oversight Correlation
└── 4.3 Disclosure Staleness Decay: Mitigating the 45-Day Statutory Reporting Lag

MODULE 5: USING ARX TERMINAL
├── 5.1 Interpreting the Confluence Score: Multi-Stream Composite Synthesis
├── 5.2 The 5-Point Pre-Flight Execution Gate: Verifying Clear Conditions
├── 5.3 Cornish-Fisher Modified VaR: Tail-Risk Modeling & Macro Stress Tests
└── 5.4 Simulated Execution Safeguards: Non-Discretionary Sizing vs Real Capital
```

### 4.2 Standard Lesson Blueprint (The 8-Part Human Template)
Each lesson adheres to a clear pedagogical flow:
1. **What This Means:** One or two sentences of plain-language conceptual definition.
2. **Why It Matters:** The direct economic consequence for real financial decisions.
3. **How It Works:** The mathematical formula or underlying market mechanism.
4. **Concrete Example:** Realistic numbers, ticker situations, or trade calculations.
5. **How to Interpret It:** What a high vs low reading indicates.
6. **Common Mistakes:** Misinterpretations, chasing late signals, or treating screener rankings as buy orders.
7. **Where It Appears in ARX:** Explicit cross-link to the relevant terminal hub or card.
8. **Key Takeaway:** One memorable summary sentence.

---

## 5. Quantitative Glossary Redesign Specification

### 5.1 Structure of `GlossaryTerm`
Every term in `frontend/lib/glossaryCatalog.ts` must provide:
- `slug`: Canonical URL slug (e.g., `cornish-fisher-var`).
- `name`: Full formal term name.
- `shortDefinition`: Natural-language executive summary.
- `category`: Econometric, Technical/Execution, or Regulatory/Smart Money.
- `latexFormula`: Formal LaTeX expression (KaTeX rendered).
- `detailedExplanation`: 3 structured paragraphs (Intuition, Mathematical Mechanics, Practical Financial Application).
- `concreteExample`: Realistic numerical calculation with step-by-step walkthrough.
- `limitations`: Explicit statement of when and why the model fails or has boundary conditions.
- `arxApplication`: How ARX Terminal utilizes the metric in live analysis.
- `keyTakeaway`: Concluding rule of thumb.
- `relatedTerms`: Array of 2–3 related glossary slugs.
- `relatedRoute`: Direct link to the corresponding terminal tool.

---

## 6. Implementation Guardrails & Non-Interference Invariants

1. **Zero Trading Engine Mutation:** The redesign touches strictly presentation components (`GuideContent.tsx`, `Navbar.tsx`, `glossaryCatalog.ts`, `globals.css`). No files under `backend/`, `engine/`, or risk calculators are modified.
2. **Preserve Operational Holds:** `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, and `PHASE_A3_UX_HOLD` remain strictly active.
3. **Parity Preservation:** All existing 48 frontend architectural tests in `npm run test:arch` must remain 100% passing.
