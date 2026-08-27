# 🧠 Engineering Lessons Learned & Architectural Insights

> **Living Technical Knowledge Base**  
> Documenting systemic failure modes, financial engineering edge cases, rendering discrepancies, and automated QA gates discovered across development of `daakara/finance`.

---

## 1. 🔍 Client-Side Deep Linking & URL State Synchronization

### 🚨 What Went Wrong
When clicking `[ Analyze in Terminal → ]` on `/compare` or `/screener`, the browser correctly navigated to `/?symbol=NVO`. However, the Terminal rendered `AAPL` instead of `NVO`.
* **Root Cause**: `useState("AAPL")` only evaluated on the initial component mount. When client-side routing transitioned to `/`, React did not re-initialize the state, and `useSearchParams()` was not dynamically listened to.
* **Why Build QA Missed It**: `npm run build` and TypeScript only verify syntactic validity. `useState("AAPL")` is valid TypeScript. Python `pytest` verified the backend returned 200 for `NVO`, but did not test the browser client router.

### 🛡️ The Preventive Standard
1. **Always implement URL-to-State synchronization**:
   ```tsx
   const searchParams = useSearchParams();
   const urlSymbol = searchParams.get("symbol");

   useEffect(() => {
     if (urlSymbol && urlSymbol.toUpperCase() !== selectedSymbol) {
       setSelectedSymbol(urlSymbol.toUpperCase());
     }
   }, [urlSymbol]);
   ```
2. **Automated Static Route Contract Quality Gate**:
   Added `test_terminal_deep_link_query_param_binding` to `tests/test_nextjs_frontend_structure.py` to enforce that any page accepting deep links imports `useSearchParams`, wraps components in `<Suspense>`, and binds the query parameters.

---

## 2. 📈 Financial Chart Canvas Sizing & Multi-Horizon Timestamps

### 🚨 What Went Wrong
Candlestick charts intermittently appeared blank or dropped series data on initial page loads and when toggling between Day Trader and Long Term modes.
* **Root Cause 1 (Flex-1 Zero-Height Render)**: When TradingView Lightweight Charts instantiates inside a CSS `flex-1` container before the parent layout calculates its bounding box, `chartContainer.clientHeight` evaluates to `0px`, causing canvas rendering failure.
* **Root Cause 2 (Timestamp Mismatch)**:
  - **Intraday (`5m`)**: Requires numeric Unix epoch in **seconds** (`UTCTimestamp`). Passing ISO strings or millisecond timestamps causes the Lightweight Charts engine to crash or drop the series.
  - **Daily (`1D`)**: Requires **`YYYY-MM-DD` date strings**. Passing numeric timestamps or full ISO strings (`2026-08-25T00:00:00Z`) causes duplicate key rejections.

### 🛡️ The Preventive Standard
1. **Set explicit responsive heights on chart containers**:
   ```tsx
   <div className="w-full min-h-[320px] h-[340px] sm:h-[400px]">
   ```
2. **Strict Multi-Horizon Timestamp Sanitization**:
   ```ts
   if (isIntraday) {
     timeVal = typeof timeVal === "string" ? Math.floor(new Date(timeVal).getTime() / 1000) : Math.floor(timeVal > 20000000000 ? timeVal / 1000 : timeVal);
   } else {
     timeVal = typeof timeVal === "number" ? new Date(timeVal * 1000).toISOString().split("T")[0] : timeVal.split("T")[0];
   }
   ```
3. **Deduplicate timestamps** strictly with a `Set()` before calling `series.setData()`.

---

## 3. 🎯 Dual-Horizon Persona Policy (Day Trader vs. Long Term)

### 🚨 What Went Wrong
The Hidden Gems screener and Compare page originally displayed only multi-year fundamental metrics ($\text{ROIC}$, $\text{PEG}$, Gross Margins, Clinical Trials), alienating day traders seeking high-beta intraday momentum and ATR volatility.

### 🛡️ The Preventive Standard
* **Universal Dual Lenses**: Every analytical view (`/`, `/screener`, `/compare`) must synchronize with `localStorage.getItem("FINANCE_USER_ROLE")` and provide:
  - **⚡ Day Trader Lens**: 14-Day ATR Range (\$), Relative Volume (RVOL), Intraday Beta, Optimal Trading Window, and Scalp setups.
  - **🏛️ Long-Term Lens**: ROIC $\ge 25\%$, PEG $\le 1.0$, Gross Margins, Piotroski F-score, and 5-Year DCF Earnings Models.
* **Universal Single-Horizon Pages**: If a page is purely infrastructure (e.g. `/settings`, `/docs`), it must display the standard **Universal View Helper Badge** to manage user expectations.

---

## 4. ⚔️ SEO Presets vs. Dynamic Asset Selectors

### 🚨 What Went Wrong
Hardcoding static comparison cards (`NVO vs LLY` and `SPY vs QQQ`) satisfied Google SERP crawlers and AI Overviews, but prevented users from comparing arbitrary pairs (e.g. `NVDA vs TSLA` or `ELF vs DUOL`).

### 🛡️ The Preventive Standard
* **The Hybrid Matcher Architecture**:
  - Keep **SEO Curated Battleground Presets** as prominent quick-pick buttons for search engine indexing.
  - Provide **Dynamic Custom Matcher Dropdowns** allowing users to cross-compare any two assets from the platform database with deep-linkable URLs (`/compare?a=NVDA&b=TSLA`).

---

## 5. 🛡️ Security Hardening & Rate Limiting at the Edge

### 🚨 What Went Wrong
Wildcard CORS (`allow_origins=["*"]`) and unhedged public APIs risked scraping and denial-of-service.

### 🛡️ The Preventive Standard
1. **Strict CORS Whitelist**: Whitelist only authorized domains (`https://finance-xp8.pages.dev`, `http://localhost:3000`).
2. **Cloudflare Security Headers**: Enforce `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and `Strict-Transport-Security` in `frontend/public/_headers`.
3. **Distributed Sliding-Window Rate Limiter**: Implemented `RedisRateLimitMiddleware` with local in-memory fallback to guarantee zero-downtime protection.

---

## 6. ⏱️ Timeframe State Isolation, Re-Render Loops & Chart Engine API Contracts

### 🚨 What Went Wrong
Timeframe/interval selector pills (`1M`, `6M`, `1Y`, `3Y`, `5Y` and `1m`, `5m`, `15m`, `1h`) appeared completely unresponsive or intermittently locked up when tapped on mobile and desktop devices.
* **Root Cause 1 (Unmemoized Callback Reset Loop)**:
  - `Navbar.tsx` included `useEffect(() => { ... onRoleChange(saved); }, [onRoleChange])`.
  - In `page.tsx`, `handleRoleChange` was passed as an unmemoized inline function.
  - Whenever the user clicked ANY interval button, `page.tsx` re-rendered $\rightarrow$ created a new `handleRoleChange` function reference $\rightarrow$ triggered `Navbar.tsx`'s `useEffect` $\rightarrow$ called `onRoleChange(saved)` $\rightarrow$ immediately forced `interval` state back to `"1y_hist"` or `"5m"`, wiping out the user's click instantly.
* **Root Cause 2 (Lightweight Charts v4 API Breaking Call)**:
  - `PriceChart.tsx` called `chartRef.current.timeScale().resetTimeScale()`.
  - In Lightweight Charts v4, `resetTimeScale()` does not exist. Calling it threw an uncaught `TypeError`, was caught by a generic `catch (err)`, and silently skipped the vital `chart.timeScale().fitContent()` call, preventing viewport scaling.
* **Root Cause 3 (Substring Match Fallback Misclassification)**:
  - `api.ts` checked `const isIntraday = interval.includes("m") || interval.includes("h")`.
  - The 5-Year secular horizon uses monthly bars (`apiInterval = "1mo"`). Because `"1mo"` contains `"m"`, it was misclassified as an intraday scalp, truncating the dataset to 22 points instead of 60 monthly points.

### 🛡️ The Preventive Standard
1. **Strict State Isolation Between Role and Timeframe Selection**:
   - Role switching callbacks must never re-trigger on child mounts or prop identity changes.
   - Always memoize role switch handlers with `useCallback(..., [])`.
   - `Navbar.tsx` must only react to user toggle clicks and external broadcast events, never reading `localStorage` inside reactive dependency loops.
2. **Lightweight Charts v4 Safe Rescaling**:
   - Never call `resetTimeScale()`. Always use:
     ```ts
     if (chartRef.current) {
       chartRef.current.timeScale().fitContent();
     }
     ```
3. **Exact-Match Interval Categorization**:
   - Intraday intervals must be matched against an exact whitelist (`["1m", "5m", "15m", "30m", "1h"]`), never via loose `.includes("m")` substrings which collide with `"1mo"` (1 month).
4. **Mobile Touch & Accessibility Contract**:
   - All timeframe buttons must explicitly declare `type="button"` and `className="... touch-manipulation cursor-pointer min-h-[36px]"`.
5. **Automated Quality Gate**:
   - Enforced via `test_chart_timeframe_state_and_api_contract` in `tests/test_nextjs_frontend_structure.py`.

---

## 7. 🌓 Dual-Theme Token Synchronization, Canvas Pixel Adaptation & Arbitrary Utility Collisions

### 🚨 What Went Wrong
When switching from Cyber Dark to Paper Light theme (`data-theme="paper"`):
* **Root Cause 1 (Tailwind Opacity Class Selector Failure)**:
  - `Navbar.tsx` and container components used opacity-suffixed utilities: `bg-[#0c1017]/95`.
  - `globals.css` only targeted `.bg-[#0c1017]`. Because CSS class selectors must match the escaped class name exactly, `.bg-\[\#0c1017\]` did not match `.bg-\[\#0c1017\]\/95`. The header remained 95% dark while child text turned dark `#0f172a`, causing black-on-black invisible text.
* **Root Cause 2 (HTML5 Canvas Pixel Decoupling)**:
  - TradingView Lightweight Charts renders directly to an HTML5 `<canvas>` via JavaScript 2D rendering contexts.
  - Canvas pixels are completely decoupled from CSS stylesheets. Even though surrounding DOM cards turned white, chart canvas backgrounds remained hard-coded to `#0b0f19` with `#162032` dark gridlines.
* **Root Cause 3 (Arbitrary Utility Proliferation)**:
  - Dozens of disparate dark hexes (`bg-[#090d14]`, `bg-[#1e293b]`, `bg-[#080c14]`, `bg-[#0d131f]`) had no light theme definitions, leaving search inputs, badges (`LIVE`), and timeframe selector bars dark.

### 🛡️ The Preventive Standard
1. **Substring Attribute Selectors for Universal Theme Overrides**:
   - Use CSS attribute substring selectors `[class*="bg-[#..."]` in `globals.css` rather than literal escaped class names:
     ```css
     [data-theme="paper"] [class*="bg-[#0c1017]"],
     [data-theme="paper"] [class*="bg-[#111722]"] {
       background-color: #ffffff !important;
     }
     [data-theme="paper"] [class*="bg-[#090d14]"] {
       background-color: #f1f5f9 !important;
     }
     ```
2. **Dynamic Canvas Theme Subscriptions**:
   - All chart components (`PriceChart.tsx`, `TradingViewChart.tsx`) must:
     1. Read `document.documentElement.getAttribute("data-theme")` on initialization.
     2. Listen for custom window event `"finance:theme-change"`.
     3. Attach a `MutationObserver` on `data-theme` attribute mutations.
     4. Dynamically invoke `chart.applyOptions({ layout: { background: { color: isLight ? "#ffffff" : "#0b0f19" }, textColor: ... }, grid: { ... } })`.
3. **Contrast-Compliant Badge Matrix**:
   - Badge overlays (Amber catalyst, Cyan VWAP, Emerald/Rose metrics) must dynamically swap dark container tinting (`bg-*-950`) to soft pastel fills (`bg-*-50`/`bg-*-100`) with high-contrast foreground text (`text-*-800`/`text-*-900`).
4. **Automated Quality Gate**:
   - Enforced via `test_light_paper_theme_compliance` in `tests/test_nextjs_frontend_structure.py`.


