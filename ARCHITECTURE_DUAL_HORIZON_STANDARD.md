# Architectural UX Standard: Dual-Horizon Persona Policy (Day Trader vs. Long-Term Investor)

**Applies to**: All current and future frontend pages, views, modules, and components across `daakara/finance`.

---

## 1. 🎯 Mandatory Persona Policy

Every page and feature in the platform must support or explicitly declare its relationship to the two primary platform user journeys:

1. **⚡ Day Trader Persona (`DAY_TRADER`)**:
   - Focus: Liquidity, 14-day Average True Range ($\text{ATR}$ in \$), Relative Volume ($\text{RVOL}$), Intraday Beta, VWAP, Level-2 Order Flow, and Opening Range Breakout (ORB) execution.
   - Default Timeframe: `5m` (5-minute candlesticks with Unix epoch time scale).
   - Primary Color Theme Accents: **Amber (`amber-400` / `amber-500`)**.

2. **🏛️ Long-Term Investor Persona (`LONG_TERM`)**:
   - Focus: Return on Invested Capital ($\text{ROIC}$), PEG Ratio, Gross Margins, Piotroski F-Score (0-9), Cornish-Fisher Value-at-Risk (VaR), 5-Year Earnings Models, and Supply Chain Contagion.
   - Default Timeframe: `1D` (Daily candlesticks with ISO date format).
   - Primary Color Theme Accents: **Cyan / Emerald (`cyan-400` / `emerald-400`)**.

---

## 2. 🛠️ Implementation Rules for Future Pages

### Rule A: Stateful Dual-Horizon Lenses
If a page displays asset data, comparison metrics, or screening filters (e.g. `/`, `/screener`, `/compare`), it **MUST**:
- Synchronize with the global `FINANCE_USER_ROLE` (`localStorage.getItem("FINANCE_USER_ROLE")`).
- Render a top-level **Dual-Horizon Lens Toggle** (`⚡ Day Trader (ATR/Vol)` $\leftrightarrow$ `🏛️ Long-Term (ROIC/Trials)`).
- Dynamically swap metric matrices and strategic narratives depending on the active lens.
- Provide 1-tap **Context-Preserving CTA links** into the terminal:
  - Day Trader: `[ Trade {Symbol} in Terminal (5m) → ]`
  - Long-Term: `[ Analyze {Symbol} in Terminal (1D) → ]`

### Rule B: Single-Horizon or Informational Pages
If a page or sub-view is inherently single-horizon or purely informational (e.g., `/settings`, `/docs`, `/health`, or a specialized Backtester), it **MUST** include an explicit **Helper Text Badge** to manage expectations:

```tsx
{/* Example Standard Helper Text Badge */}
<div className="bg-[#0b1019] border border-[#243044] rounded-lg p-2.5 text-xs text-slate-400 flex items-center space-x-2">
  <span className="text-cyan-400">ℹ️</span>
  <span>
    <strong>Universal View:</strong> This view displays system-wide infrastructure and does not vary between Day Trader and Long-Term modes.
  </span>
</div>
```

---

## 3. 🌐 Global Storage Contract
- **Key**: `"FINANCE_USER_ROLE"`
- **Values**: `"DAY_TRADER"` | `"LONG_TERM"`
- **Persistence**: Persists across browser refreshes and cross-page navigation.

---

## 4. ⚡ Timeframe & Role State Decoupling Standard

### Rule C: Timeframe State Isolation
1. **Never trigger role re-synchronization on timeframe changes**:
   - The active timeframe state (`interval`) is subordinate to the active `userRole`.
   - Changing the timeframe from `1Y` to `1M` or `5m` to `1m` must NEVER cause the parent or child components (`Navbar`, `Watchlist`) to fire `onRoleChange`.
   - `handleRoleChange` in top-level pages must ALWAYS be wrapped in `useCallback(..., [])` to maintain stable reference identity.
2. **Dedicated Timeframe Taxonomy**:
   - **Intraday Scalps (`DAY_TRADER`)**: `1m`, `5m`, `15m`, `1h` (Formatted with Unix epoch seconds).
   - **Macro Horizons (`LONG_TERM`)**: `1m_hist` (1M), `6m_hist` (6M), `1y_hist` (1Y), `3y_hist` (3Y), `5y_hist` (5Y) (Formatted with ISO `YYYY-MM-DD` dates).
3. **Chart Engine Safety**:
   - TradingView Lightweight Charts must always resize and auto-fit via `chart.timeScale().fitContent()`.
   - Never call deprecated or non-existent methods like `resetTimeScale()`.

---

## 5. 🌓 Universal Dual-Theme Theming Standard

### Rule D: Theme-Aware Components & Canvas Subscriptions
1. **Dynamic HTML5 Canvas Adaptation**:
   - All financial chart components must bind to `"finance:theme-change"` and `MutationObserver` on `data-theme` to dynamically toggle between Dark (`#0b0f19` canvas, `#162032` grid) and Paper Light (`#ffffff` canvas, `#f1f5f9` grid).
2. **Opacity-Safe Global Classes**:
   - Never write single-theme hardcoded dark classes without ensuring `[data-theme="paper"]` wildcard substring rules map them to high-contrast paper equivalents.
3. **Badge Visibility Guarantees**:
   - Both Day Trader indicators (Amber ATR, VWAP) and Long Term metrics (Cyan ROIC, Emerald Piotroski) must retain high contrast in both themes.


