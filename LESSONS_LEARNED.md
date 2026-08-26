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
