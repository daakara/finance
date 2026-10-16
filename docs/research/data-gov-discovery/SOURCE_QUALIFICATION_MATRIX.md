# ARX TERMINAL — SOURCE QUALIFICATION MATRIX (CORRECTED)
**Backlog Reference:** `BACKLOG-DATA-GOV-DISCOVERY-001`  
**Workstream:** Worktree B (Independent Discovery & Qualification)  
**Target Repository:** `daakara/finance`  
**Branch:** `research/data-gov-discovery`  
**Base Commit:** `fcb43f7aae1797bde762b91b5c309817ab25f459`  
**Audit Date:** 2026-10-16

---

## 1. Decision Taxonomy & Evaluation Criteria

Every candidate use case is qualified across twelve standardized dimensions and assigned one of four normative decisions:

- **`ADOPT_FOR_FOLLOW_UP`:** A defined user need, demonstrated information gap, suitable source, and justified small next design/experiment. (Does *not* authorize connector implementation).
- **`DEFER`:** Potential value exists, but a specific prerequisite or unanswered architectural question prevents justified implementation. Requires an explicit reopening condition.
- **`REJECT_FOR_CURRENT_USE_CASE`:** Evidence establishes inadequate incremental value, temporal mismatch, or redundancy for the specific evaluated use case. Does *not* generalize rejection to all possible uses of the source.
- **`ACCESS_OR_QUALIFICATION_INCOMPLETE`:** Evidence is insufficient to decide due to catalog access limitations or unverified parameters.

---

## 2. Standardized Source Qualification Matrix

| Qualification Dimension | Source 1: CFTC Commitments of Traders (COT) | Source 2: U.S. Treasury Fiscal Data (Operating Cash Balance) | Source 3: Bureau of Labor Statistics (BLS API v2) | Source 4: Bureau of Economic Analysis (BEA API) | Source 5: OFAC Specially Designated Nationals (SDN) |
|---|---|---|---|---|---|
| **Primary Publishing Agency** | Commodity Futures Trading Commission (CFTC) | Bureau of the Fiscal Service, U.S. Dept. of the Treasury | Bureau of Labor Statistics, U.S. Dept. of Labor | Bureau of Economic Analysis, U.S. Dept. of Commerce | Office of Foreign Assets Control, U.S. Dept. of the Treasury |
| **Evaluated Dataset & Endpoint** | Legacy COT — Futures Only (`6dca-aqww`)<br>`https://publicreporting.cftc.gov/resource/6dca-aqww.json` | Daily Treasury Statement (DTS) Table 1 — Operating Cash Balance<br>`https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance` | Public Data API v2 — Consumer Price Index (`CUSR0000SA0`)<br>`https://api.bls.gov/publicAPI/v2/timeseries/data/` | Industry Economic Accounts — `GDPbyIndustry`<br>`https://apps.bea.gov/api/data?method=GetData&datasetname=GDPbyIndustry...` | Specially Designated Nationals List<br>`https://www.treasury.gov/ofac/downloads/sdn.csv` |
| **Data.gov Catalog Status** | Programmatic CKAN search returned HTTP 404; direct Socrata Open Data catalog verified. | Programmatic CKAN search returned HTTP 404; direct Treasury Fiscal Data catalog verified. | Programmatic CKAN search returned HTTP 404; direct BLS developer portal verified. | Programmatic CKAN search returned HTTP 404; direct BEA developer portal verified. | Programmatic CKAN search returned HTTP 404; direct Treasury OFAC distribution verified. |
| **Specific ARX User Question / Decision** | "Is aggregate speculative positioning in equity index futures at an extreme that warrants macro caution or sizing adjustment?" | "What is the daily closing cash balance of the U.S. Treasury, and has sovereign liquidity expanded or contracted recently?" | "What is the latest monthly headline Consumer Price Index level for macro context?" | "Which industry sectors show structural margin compression based on input-output accounts?" | "Is an instrument or issuer subject to U.S. sanctions embargoes?" |
| **Intended Use Archetype** | Descriptive macro sentiment context. | Descriptive sovereign liquidity context. | Descriptive macroeconomic context. | Structural economic analysis (incompatible with swing horizons). | Legal / compliance embargo screening. |
| **Existing ARX Capability** | Macro Ribbon (`api/routes/macro.py`): VIX tier, FRED yield spreads (`T10Y2Y`). Zero futures positioning. | Macro Ribbon: 10Y yield (`^TNX`) and yield curve spread. No daily sovereign cash balance tracking. | `analyst_dashboard/data/fred_fetcher.py`: Actively fetches and caches `CPIAUCSL` (Consumer Price Index). | Market price state: Sector SPDR ETF price action, moving averages, and relative strength. | `R07_NOT_DELISTED`: Exchange listing status and SEC regulatory filing verification. |
| **Incremental Information Gap** | Weekly reportable non-commercial and commercial positioning in equity index futures (`E-MINI S&P 500`). | Daily Treasury General Account (TGA) closing cash balance (`close_today_bal`). | None for headline monthly CPI. Granular sub-indices would be incremental. | Quarterly value-added and intermediate input accounts by 65+ NAICS industry groups. | Statutory sanctions designation status under IEEPA/TWEA. |
| **Temporal Contract & Latency** | **Cutoff:** Tuesday 23:59 ET.<br>**Published:** Friday 15:30 ET.<br>**Lag:** Strict 3 calendar days. Data is up to 10 days old by next Friday morning. | **Cutoff:** Daily $T$.<br>**Published:** Daily $T+1$ around 16:00 ET.<br>**Lag:** 1 business day. | **Cutoff:** Calendar month end.<br>**Published:** 8:30 AM ET ~12-16 days post month-end.<br>**Lag:** 2–3 weeks. | **Cutoff:** Calendar quarter end.<br>**Published:** 65 to 90+ days post quarter-end.<br>**Lag:** 2 to 3 months. | **Cutoff:** Real-time upon designation.<br>**Published:** Immediate static file push.<br>**Lag:** Near zero. |
| **Revision Policy & Vintage Risks** | Rarely revised historically; release delayed only if Friday is a Federal holiday. | Audited historical figures; occasional minor accounting reconciliations. | **Annual Revisions:** Seasonal adjustment factors revised every February for preceding 5 years. | **Extensive Revisions:** Multi-year benchmark revisions rewrite years of historical industry data. | Additive/subtractive deltas; archived in separate changes files. |
| **Authentication & Documented Limits** | **Unauthenticated:** Variable shared IP throttling.<br>**App Token:** Dedicated ~50k requests/day (Free registration). | **Zero Authentication:** No keys, no tokens.<br>**Rate Limit:** <= 100 req/min guideline; HTTP 429 if exceeded. | **v2 Unauth:** 25 req/day, 10-year limit.<br>**v2 with Key:** 500 req/day, 20-year span (Free key). | **Mandatory Key:** 36-char `UserID` required.<br>**Rate Limit:** 100 req/min, 30 errors/min. | **Zero Authentication:** Static file download over HTTPS. |
| **Identifiers & ARX Join Feasibility** | **Key:** Contract Market Name (`E-MINI S&P 500`) or code `13874A`. Requires manual mapping dictionary to `SPY`/`SPX`. | **Key:** `record_date` (`YYYY-MM-DD`). Direct 1:1 join with ARX calendar date. | **Key:** BLS Series ID (`CUSR0000SA0`). Joins via monthly date string. | **Key:** NAICS industry codes. Complex cross-walk needed from GICS/SIC. | **Key:** Entity Legal Name / Aliases. Poor joinability (no CIK, FIGI, or ticker). |
| **Normative Decision** | **`DEFER`** | **`DEFER`** | **`REJECT_FOR_CURRENT_USE_CASE`** | **`REJECT_FOR_CURRENT_USE_CASE`** | **`REJECT_FOR_CURRENT_USE_CASE`** |
| **Reopening Condition / Smallest Next Step** | Specification of futures-to-equity mapping dictionary, dual-timestamp schema, and Macro Ribbon UI display contract. | Product determination that displaying daily TGA cash balance addresses an explicit user need. | Reopen only if a future feature requires specialized BLS micro-series not mirrored by FRED. | Reopen only if ARX develops multi-year structural fundamental valuation models. | Reopen only if ARX introduces custodial, brokerage execution, or AML compliance workflows. |

---

## 3. Key Decision Justifications

### 3.1 Deferred Use Cases
1. **CFTC COT (`DEFER`):**
   - *User Benefit:* Provides macro awareness of extreme speculative crowding in index futures, complementing price-based trend indicators.
   - *Why Deferred:* Requires resolving: (a) dual-timestamp database schema (`observed_as_of_date` Tuesday vs `available_at_timestamp` Friday 15:30 ET) to prevent look-ahead bias in simulations, and (b) contract-to-index mapping dictionary.
2. **Treasury Operating Cash Balance (`DEFER`):**
   - *User Benefit:* Tracks net daily sovereign liquidity shifts (TGA cash balance withdrawals/deposits).
   - *Why Deferred:* While the API is open and zero-auth, adding TGA to the Macro Ribbon requires product prioritization confirming user demand for sovereign liquidity context over existing yield curve metrics.

### 3.2 Rejected Use Cases
1. **BLS CPI Connector (`REJECT_FOR_CURRENT_USE_CASE`):**
   - *Rationale:* For the specific evaluated use case (headline monthly CPI level), FRED (`fred_fetcher.py`) already provides `CPIAUCSL` through an established, tested fetcher. Maintaining a separate, rate-limited direct BLS connector provides zero incremental data.
2. **BEA Industry Economic Accounts (`REJECT_FOR_CURRENT_USE_CASE`):**
   - *Rationale:* Bounded strictly to ARX's active swing screening horizon (3 to 15 trading days). A 65–90 day publication latency and retrospective revisions provide inadequate incremental value over real-time Sector SPDR ETF technicals.
3. **OFAC SDN List (`REJECT_FOR_CURRENT_USE_CASE`):**
   - *Rationale:* ARX is an analytical terminal for US-listed securities. No current user story or functional specification calls for entity-level sanctions screening.
