# 🍉 Finance Quantitative Terminal: Impeccable & Watermelon UX Blueprint

> **Design Engineering Standard & UX Architecture Specification**  
> Synthesizing **Anthropic Impeccable** (distinctive aesthetic direction, zero AI slop, OKLCH perceptual color, fluid typography) and **Watermelon Design Engineering** (concentric radii, tactile `0.96` press scales, optical alignments, tabular numerals, 40px touch targets, zero layout shifts).

---

## 1. 🎯 Product Purpose & Persona Context

### The Audience & Job-to-be-Done
`daakara/finance` is a real-time quantitative intelligence workspace serving two distinct financial personas:
1. **⚡ The High-Frequency / Intraday Scalper (`DAY_TRADER`)**: Requires rapid situational awareness, 5-minute VWAP / 20 EMA candlestick alignment, 14-day ATR dollar range, bid-ask spread liquidity, and tactile position-sizing calculations with zero cognitive lag.
2. **🏛️ The Fundamental & Quantitative Investor (`LONG_TERM`)**: Requires institutional risk modeling (Cornish-Fisher VaR, Sortino, Omega ratio), Piotroski F-score health, Joel Greenblatt Magic Formula & Peter Lynch GARP screening, clinical pipeline catalysts (e.g. Novo Nordisk Amycretin Phase 2/3), and multi-year DCF earnings growth trajectories.

---

## 2. 🎨 Aesthetic Direction: "Industrial Precision Mainframe"

We reject the generic "AI Slop" look (neon blue gradients, glowing glassmorphism, border-left colored accent stripes, floating modals). Instead, we commit to **Utilitarian High-Density Financial Engineering**:

```
========================================================================================
BRAND PERSONALITY: [ DENSE • OPINIONATED • SURGICALLY PRECISE ]
THEME: Dark Engineered Canvas tinted toward deep Slate-Navy
ACCENTS: Amber Gold (Intraday Momentum) & Cyan/Emerald (Fundamental Compounding)
========================================================================================
```

### Color Palette (Perceptually Uniform OKLCH tinting)
* **Canvas Background**: `oklch(0.12 0.015 250)` (`#070a11`) — Tinted deep slate-navy surface; avoids pure unnatural `#000000`.
* **Surface Paneling**: `oklch(0.16 0.02 250)` (`#0e131d`) — High-contrast container base.
* **Inner Wells / Matrices**: `oklch(0.10 0.015 250)` (`#080c14`) — Recessed telemetry wells for data grids.
* **Long-Term Accent**: `oklch(0.78 0.14 200)` (`#22d3ee` / `cyan-400`) — Surgical clarity for balance sheets & valuation models.
* **Day-Trader Accent**: `oklch(0.76 0.16 75)` (`#f59e0b` / `amber-500`) — High-urgency liquidity, volatility, and execution signals.
* **Positive Green**: `oklch(0.75 0.17 150)` (`#34d399` / `emerald-400`) — Non-fluorescent organic compounding.
* **Negative Red**: `oklch(0.68 0.20 25)` (`#f87171` / `rose-400`) — Measured downside risk warning.

---

## 3. 🍉 Watermelon Design Engineering Rules Applied

### 1. Concentric Border Radius
* Card Containers: `rounded-2xl` (`16px`)
* Inner Telemetry Wells (`p-3.5` / `14px` padding): `rounded-xl` (`12px`)
* Buttons & Mini Badges inside Wells: `rounded-lg` (`8px`)
* *Formula Enforced*: $R_{\text{outer}} = R_{\text{inner}} + \text{Padding}$. Never duplicate identical radius across nested layers.

### 2. Tactile Press States (`scale(0.96)`)
* Every interactive button, tab, and pill filter includes:
  ```css
  active:scale-[0.96] transition-transform duration-100 ease-out
  ```
* Prohibits values below `0.95` (which feel distorted/laggy) and values of `1.0` (which feel dead/unresponsive).

### 3. Tabular Numerals Everywhere
* All market numbers, share quantities, P/E multiples, ATR dollars, timestamps, and hit-rates apply:
  ```css
  tabular-nums font-mono
  ```
* Eliminates jitter and horizontal layout shift during live WebSocket ticker updates.

### 4. Minimum Touch Target Area (40×40px)
* Mobile touch targets (Bottom Navigation Dock, Timeframe Selectors, Watchlist rows) maintain a minimum hit area of 40×40px or use invisible hit-area padding (`p-2`) to avoid tap frustration.

---

## 4. 🧭 Dual-Horizon Platform Architecture Matrix

Every current and future page in the platform adheres to this blueprint:

| Route | Primary View Purpose | ⚡ Day Trader Lens (`DAY_TRADER`) | 🏛️ Long-Term Lens (`LONG_TERM`) | Universal Helper Text Required? |
|---|---|---|---|---|
| **`/` (Terminal)** | Core analytical workspace | `5m` VWAP charts, Position Sizer, ATR Execution, Walk-Forward Accuracy | `1D` EMA charts, Piotroski F-score, Cornish-Fisher VaR, Market Graph, Clinical Catalysts | No (Dual Active) |
| **`/screener`** | High-Alpha Discovery | High RVOL, Intraday Beta, 14-Day ATR Range, Opening Range Breakout Setups | Peter Lynch GARP, Joel Greenblatt Magic Formula, ROIC $\ge 25\%$, PEG $\le 1.0$ | No (Dual Active) |
| **`/compare`** | Competitor Comparison | Intraday Dollar Ranges, Optimal Session Hours, Bid-Ask Spread Liquidity | 5-Year EPS Forecasts, Clinical Trials (NVO vs LLY), Moats & Balance Sheets | No (Dual Active) |
| **`/settings`** | User Configuration | API keys, broker connections, data feeds | API keys, broker connections, data feeds | **Yes** *(Helper Text Badge)* |
| **`/docs`** | Methodologies | Quant formulas, VaR calculations, Data feeds | Quant formulas, VaR calculations, Data feeds | **Yes** *(Helper Text Badge)* |

---

## 5. 🏗️ Component Blueprint Specifications

### A. The Master Navigation Header (`Navbar.tsx`)
* **Concentric Radii**: `rounded-2xl` shell with `rounded-xl` role switcher toolbar.
* **Dual Roles**:
  - `⚡ Day Trade`: Visual feedback in tactile Amber gold (`bg-amber-500 text-slate-950`).
  - `🏛️ Long Term`: Visual feedback in tactical Cyan (`bg-cyan-500 text-slate-950`).
* **Mobile Adaptation**: Sticky floating bottom navigation dock (`fixed bottom-3`) with direct 1-tap navigation across `Terminal`, `Gems`, and `Compare`.

### B. Catalyst & 5-Year Earnings Module (`CatalystForecastCard.tsx`)
* **Clinical Trial Banner**: Displays Drug Name (e.g. *Amycretin*), Trial Phase (*Phase 2/3*), and Readout Horizon (*Q4 2026 - Q2 2027*).
* **Quant Milestone Schedule**: 4-step progressive timeline with color-coded impact chips (*Transformational* in Purple, *High Positive* in Emerald).
* **5-Year Growth Table**: Tabular-numeric revenue, net margins, projected EPS, and implied price targets through 2031.

### C. Head-to-Head Comparison Card (`frontend/app/compare/page.tsx`)
* **Layout**: 2-column side-by-side card grid with high-contrast metric comparisons.
* **Context-Preserved Terminal Jump**:
  - In Day Trade mode: `[ Trade NVO in Terminal (5m) → ]` opens `/?symbol=NVO` with 5m interval and VWAP ready.
  - In Long Term mode: `[ Analyze NVO in Terminal (1D) → ]` opens `/?symbol=NVO` with 1D interval and factor radar ready.
