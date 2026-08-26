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
