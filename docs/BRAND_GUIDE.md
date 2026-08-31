# 🏛️ ARX TERMINAL — BRAND IDENTITY & DESIGN SYSTEM GUIDE

> **Version**: 2.0.0  
> **Last Updated**: August 2026  
> **Classification**: Single Source of Truth (SSOT) — Design & Brand Architecture

---

## 1. 🌐 Brand Essence & Mission

### 1.1 Brand Purpose
**ARX Terminal** is built to bridge the asymmetry between Wall Street institutional trading desks and disciplined retail market participants. We provide real-time quantitative analytics, multi-factor fundamental modeling, institutional smart money tracking, and statistical trade execution plans without bloated financial jargon, advertisements, or speculative noise.

### 1.2 Core Tagline
> **"No-BS Market Intelligence & Decision Engine"**

### 1.3 Brand Personality Pillars
- **🔬 Precise & Mathematical**: Grounded in Cornish-Fisher expansion, Markowitz covariance, and Minervini Stage 2 volatility contraction.
- **⚡ High-Speed & Low-Latency**: Sub-second client execution, instant keyboard shortcuts, zero-lag charting.
- **🛡️ Uncompromising & Honest**: Transparent data provenance (`LIVE`, `CACHED`, `RECONNECTING`), zero hallucinated numbers, and explicit downside risk boundaries.
- **🏛️ Institutional Quality, Modern UX**: Clean cyber-dark terminal aesthetics fused with editorial financial clarity.

---

## 2. 🔤 Logo System & Brand Marks

```
+-----------------------------------------------------------------------------+
|                               LOGO LOCKUPS                                  |
|                                                                             |
|  1. Primary Lockup (Desktop / Tablet)                                       |
|     +---------+                                                             |
|     |   ARX   |  TERMINAL                                                   |
|     +---------+  NO-BS MARKET INTEL                                         |
|                                                                             |
|  2. Compact Mobile Monogram (Mobile Navigation / App Icon / Favicon)        |
|     +---------+                                                             |
|     |   ARX   |                                                             |
|     +---------+                                                             |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### 2.1 The Monogram Badge
- **Container**: Rounded rectangle (`rounded-lg`, `border-radius: 0.5rem`).
- **Color**: Solid Cyan-600 (`#0891b2`) with subtle cyber glow (`shadow-[0_0_12px_rgba(6,182,212,0.25)]`).
- **Typography**: Bold uppercase monospace (`font-mono font-black text-white`).
- **Aspect Ratio**: 1:1 square badge (`w-7 h-7` on mobile, `w-8 h-8` to `w-9 h-9` on desktop).

### 2.2 Wordmark & Typography Lockup
- **Primary Title**: `TERMINAL` (Set in `font-mono font-bold tracking-tight text-white`).
- **Subtitle Badge**: `NO-BS MARKET INTEL` (Set in `font-mono text-[9px] text-cyan-400 tracking-wider uppercase`).
- **Invariant**: On mobile screens ($<640\text{px}$), the text `ARX` is never repeated next to the `[ARX]` badge. The monogram stands alone with clean distinction.

### 2.3 Clearspace & Usage Rules
- Minimum clearspace around the logo is equal to $0.5\times$ the badge width ($16\text{px}$).
- **Do Not**: Skew, rotate, outline, or apply generic drop-shadows to the letters.
- **Do Not**: Place the cyan logo over conflicting saturated backgrounds (e.g. bright blue or green).

---

## 3. 🌓 Dual-Theme Architecture

ARX Terminal supports two fully calibrated, high-contrast visual themes: **Cyber Dark** (Default) and **Paper Light** (Editorial).

```
+-----------------------------------------------------------------------------+
|                         CYBER DARK (Primary Terminal)                       |
|  Canvas: #070a10 | Surface: #111722 | Borders: #243044 | Accent: #06b6d4    |
+-----------------------------------------------------------------------------+
|                         PAPER LIGHT (Editorial Mode)                        |
|  Canvas: #f4f6f9 | Surface: #ffffff | Borders: #e2e8f0 | Accent: #0284c7    |
+-----------------------------------------------------------------------------+
```

### 3.1 Design Tokens (CSS Custom Properties)

| Token Name | Cyber Dark (Default) | Paper Light (`[data-theme="paper"]`) | Usage |
| :--- | :--- | :--- | :--- |
| `--bg-app` | `#070a10` (Deep Space) | `#f4f6f9` (Cool Paper) | Page background & viewport root |
| `--bg-surface` | `#111722` (Obsidian Slate) | `#ffffff` (Pure White) | Card containers, sidebar, navbar |
| `--bg-surface-hover` | `#1b2434` (Elevated Slate) | `#f1f5f9` (Soft Slate) | Table rows & card hover states |
| `--bg-element` | `#162030` (Inset Well) | `#e2e8f0` (Inset Light) | Input fields, search bars, chips |
| `--border-subtle` | `#243044` (Steel Boundary) | `#e2e8f0` (Light Border) | Card separators, table gridlines |
| `--border-strong` | `#364866` (Focus Border) | `#cbd5e1` (Medium Border) | Active focus rings, modal borders |
| `--text-main` | `#f0f4f8` (Crisp Light) | `#0f172a` (Ink Black) | Primary headings, prices, metrics |
| `--text-muted` | `#94a3b8` (Slate Muted) | `#475569` (Slate Dark) | Labels, descriptions, table headers |
| `--text-faint` | `#64748b` (Dim Slate) | `#94a3b8` (Muted Faint) | Timestamps, micro-disclaimers |

---

## 4. 🎨 Semantic Signal Color System

All colors carry strictly functional financial meaning. Decorative or distracting colors are prohibited.

| Signal Role | Color Name | Hex Code | Tailwind Class | Semantic Context |
| :--- | :--- | :--- | :--- | :--- |
| **Alpha / Profit** | Emerald | `#10b981` | `text-emerald-400`, `bg-emerald-500` | Bullish returns, positive PnL, Stage 2 breakouts, TP hit |
| **Risk / Loss** | Rose | `#f43f5e` | `text-rose-400`, `bg-rose-500` | Stop-loss floor, drawdown, bearish distribution trap, Stage 4 |
| **Momentum / Intel** | Cyan | `#06b6d4` | `text-cyan-400`, `bg-cyan-500` | Brand primary, selected tabs, optimal entry zones, active filters |
| **Warning / Caution** | Amber | `#f59e0b` | `text-amber-400`, `bg-amber-500` | Imminent earnings hazard, high VIX regime, delayed SEC filing |
| **Institutional / Whale** | Purple | `#a855f7` | `text-purple-400`, `bg-purple-500` | Dark pool sweeps, institutional accumulation, unusual options flow |
| **Government / Law** | Blue | `#3b82f6` | `text-blue-400`, `bg-blue-500` | Congressional STOCK Act disclosures, Senate/House oversight |

---

## 5. ⚡ Dual-Horizon Design Philosophy

ARX Terminal serves two distinct trader archetypes through an instant 1-click lens toggle:

```
+-----------------------------------------------------------------------------+
|                          DUAL-HORIZON LENSES                                |
|                                                                             |
|  ⚡ DAY TRADER MODE                                                         |
|  - Key Metrics: RVOL, ATR-14, VWAP Pullbacks, 5m/15m Intraday Candles       |
|  - Dominant Palette: Electric Amber & Flash Cyan                            |
|  - Primary Action: Hard Dollar Risk Stop, Fractional Kelly Sizing           |
|                                                                             |
|  🏛️ LONG-TERM COMPOUNDER MODE                                               |
|  - Key Metrics: ROIC, Piotroski F-Score, FCF Yield, Moats, Lynch GARP       |
|  - Dominant Palette: Institutional Emerald & Deep Slate                     |
|  - Primary Action: Fair Value Accumulation Zones, Secular Growth Thesis     |
+-----------------------------------------------------------------------------+
```

---

## 6. 💬 Dual-Vernacular Language System

To ensure mathematical precision for quants while keeping the platform accessible to everyday investors, the UI provides a real-time **Plain English vs. Pro Quant** vernacular switch (`finance:vernacular-change`):

| Interface Element | 💬 Plain English Mode | 🤓 Pro Quant Mode |
| :--- | :--- | :--- |
| **Entry Level** | Buy Zone ($XX.XX – $YY.YY) | Optimal Accumulation Range ($XX.XX – $YY.YY) |
| **Invalidation** | Stop Loss (Maximum Loss Floor) | Hard Invalidation Pivot (ATR Stop Floor) |
| **Target 1** | First Profit Goal (Take Half Off) | TP1 Tranche Scale Level (Scale 0.50x) |
| **Target 2** | Bonus Runner Target (Let It Ride) | TP2 Convexity Runner (Risk-Free Ratchet) |
| **Risk / Reward** | Reward-to-Risk Score (e.g. 2.5 to 1) | Convexity Ratio (e.g. 2.50 : 1.00 R:R) |
| **Smart Money** | Big Investors Are Quietly Buying | Institutional Stealth Accumulation / Asymmetric Flow |
| **Volatility Risk** | Calm Market / Stormy Market | Low Vol Regime (VIX <16) / High Vol Expansion (VIX >25) |

---

## 7. 🔤 Typography & Number Formatting Standards

### 7.1 Font Families
- **Monospace (Numbers & Financial Data)**: `JetBrains Mono`, `SF Mono`, `ui-monospace`, `monospace`.
  - Used for: Prices, ticker symbols, percentage changes, risk metrics, chart coordinates, financial statements.
  - Mandatory Class: `font-mono tabular-nums` (ensures uniform column alignment during live price ticks).
- **Sans-Serif (Body & Narrative)**: `Inter`, `ui-sans-serif`, `system-ui`, `sans-serif`.
  - Used for: Articles, research dossiers, tooltips, onboarding guides, committee notes.

### 7.2 Number Formatting Invariants
- **Prices $\ge \$1.00$**: Fixed 2 decimals (`$168.70`, `$319.64`).
- **Crypto / Micro-Prices ($< \$1.00$)**: 4 decimals (`$0.0425`).
- **Percentages**: Always signed with explicit sign (`+3.45%`, `-1.20%`).
- **Large Values**: Abbreviated with single capital letter (`$42.8M`, `$1.2B`, `$3.5T`).

---

## 8. 🧩 Component & UI Design Patterns

### 8.1 Data Provenance Badges
Every metric displays its exact upstream provenance:
- 🟢 **`REAL-TIME EXCHANGE`**: Pulsing emerald dot (`bg-emerald-400 animate-pulse`). Live WebSocket / REST stream.
- 🟡 **`VERIFIED CACHE (<15M)`**: Cyan dot (`bg-cyan-400`). Validated snapshot from local store within 15-minute TTL.
- 🔴 **`RECONNECTING`**: Amber dot (`bg-amber-400`). Offline resilience mode with zero fake price hallucination.

### 8.2 Modal Dialogs & Sheets
- **Backdrop**: Deep blur with 85% black wash (`bg-black/85 backdrop-blur-md`).
- **Container**: Rounded 2xl with subtle top highlight (`rounded-2xl border border-[#243044] bg-[#0c1017] shadow-2xl`).
- **Dismissibility**: Always dismissible via `Escape` key, top-right `✕` button, and backdrop tap.

### 8.3 Keyboard Ergonomics
| Keybinding | Global Action |
| :---: | :--- |
| <kbd>/</kbd> or <kbd>Cmd+K</kbd> | Open Universal OmniSearch Palette |
| <kbd>T</kbd> | Jump to Home Terminal |
| <kbd>S</kbd> | Jump to Screener |
| <kbd>P</kbd> | Jump to Private Portfolio |
| <kbd>R</kbd> | Toggle Day Trader $\leftrightarrow$ Long-Term Horizon Lens |
| <kbd>V</kbd> | Toggle Plain English $\leftrightarrow$ Pro Quant Vernacular |
| <kbd>Esc</kbd> | Dismiss active modal or palette |

---

## 9. 🖋️ Voice & Copywriting Guidelines

### 9.1 Tone Principles
1. **Fact-First, Zero Hype**: Never say "Skyrocket", "To the Moon", or "Guaranteed 10x". Say "High probability Stage 2 breakout based on 3-week volume contraction".
2. **Clear Invalidation**: Every trade idea must state where it is wrong (Stop Loss / Invalidation Floor).
3. **Institutional Accountability**: Disclose regulatory latency and statutory filing windows for political disclosures.

### 9.2 Example Copy Matrix
- ❌ **Poor Copy**: *"Look at this hot stock that Nancy Pelosi just bought, buy now!"*
- ✅ **ARX Copy**: *"Representative Nancy Pelosi (D-CA) disclosed a $1.0M purchase of NVDA on 2026-08-26 (transaction date: 2026-07-15). Statutory latency: 42 days. Key catalyst: Blackwell enterprise server ramp."*

---

## 10. 📁 Asset Manifest & Directory Structure

```
docs/
 ├── BRAND_GUIDE.md               <-- SSOT Brand Architecture in Codebase
 ├── ARCHITECTURE_DUAL_HORIZON_STANDARD.md
 └── CONTRIBUTING.md

frontend/
 ├── app/globals.css              <-- CSS Custom Property Token Implementations
 ├── components/
 │    ├── Navbar.tsx              <-- Brand Header & Logo Lockup
 │    ├── FeedFreshnessIndicator.tsx <-- Provenance Badges
 │    └── DataSourceBadge.tsx     <-- Live vs Cached Badges
 └── lib/
      └── constants.ts            <-- Design Tokens & Shared Palette Constants
```
