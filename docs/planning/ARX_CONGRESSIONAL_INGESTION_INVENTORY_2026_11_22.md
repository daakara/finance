# ARX Congressional Ingestion Inventory & System Audit (Step 1)
**Document ID:** `ARX-DOC-20261122-CONGRESSIONAL-INVENTORY-001`  
**Evaluation Date:** 2026-11-22  
**Authority:** ARX Architecture & Governance Board (Issue #165, Sprint P3 #231)  
**Execution Context:** Strictly Read-Only Discovery, Code Audit, Production DB Inspection & State Inventory  
**Target Repository:** `daakara/finance`  
**Documentation Branch:** `docs/congressional-ingestion-inventory-001`  

---

## 1. Executive Conclusion

> [!IMPORTANT]
> **Definite Conclusion:** There is currently **ZERO live or point-in-time (PIT) ingestion, automated retrieval, relational storage, or cryptographic verification of Congressional STOCK Act disclosures** in ARX Terminal.
>
> All congressional disclosures presented in the user interface or exposed via API routes are served exclusively from a static, in-memory Python array of 40 historical transactions frozen as of August 2026 (`CONGRESSIONAL_TRADES` in `analyst_dashboard/analyzers/smart_money.py`), hardcoded Next.js static catalogs (`frontend/lib/seoCatalogs.ts`), or hardcoded page mockups (`frontend/app/smart-money/late-filers/page.tsx`).
>
> When the live operational mode is invoked (`include_curated=False`, the default API and backend posture), the system returns an empty list (`[]`) and reports statuses of `UNAVAILABLE` or `PIPELINE_HELD`.
>
> Furthermore, an authoritative inspection of the production PostgreSQL database (`finance_smart_money`) confirms that **no congressional tables, schemas, or records exist**, and an inspection of the SQLite database confirms that the single defined table `insider_disclosures` contains **0 rows** and is an abandoned, unreferenced schema declaration.

```mermaid
flowchart TD
    subgraph Production Reality
        A[Client Request: /smart-money/congress] --> B{include_curated?}
        B -- False (Live Mode) --> C[Returns Status: UNAVAILABLE / PIPELINE_HELD<br>Trades: []]
        B -- True (Curated Mode) --> D[Returns Frozen Memory Array<br>40 Historical Trades from Aug 2026]
        
        E[Railway PostgreSQL: finance_smart_money] --> F[10 Tables: Form 4 Only<br>0 Congressional Tables / Rows]
        G[Production SQLite: market.db] --> H[insider_disclosures: 0 Rows<br>Unused Abandoned Schema]
        I[Worker Service: smart-money-worker] --> J[STANDBY Loop<br>0 Congressional Tasks / Jobs]
    end
```

---

## 2. Current Production and Repository Revisions

Direct inspection of remote GitHub references, local worktrees, deployed Railway microservices, and the Cloudflare Pages edge reveals the following exact revision baseline:

| Component | Target Identifier | Observed State / Commit SHA | Verification Timestamp / Notes |
| :--- | :--- | :--- | :--- |
| **Remote Production Branch** | `origin/release/production` | `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` | Head commit: `fix(smart-money): defer trial retention and harden backup verification (#315)` |
| **Local Integration Worktree** | `/Users/iDavid/Projects/finance-integration-313-314` | `feat/arx-integration-313-314` (clean) | Base `041ab6e7`, merge commit `978123f7` |
| **Dedicated Inventory Worktree** | `/Users/iDavid/Projects/finance-congressional-inventory` | `docs/congressional-ingestion-inventory-001` | Branched directly from `origin/release/production` (`2eccc64a`) |
| **Railway Project** | `tranquil-radiance` | ID: `a339a92a-7236-4ac0-9a3a-d7ad440f9690` | Active project |
| **Railway Environment** | `production` | ID: `11cad5e1-358e-430e-84fc-189953cff48a` | Production environment |
| **Railway Web Service** | Service ID: `56e38249-b78c-4a60-80d7-303ce7451bd8` | Deployment: `359667dc-0ad4-4905-9d4b-3121856b3f1f` | `SUCCESS`, Deployed: 2026-11-21 09:46:28 +01:00, Running SHA: `2eccc64a` |
| **Railway Worker Service** | Service ID: `380acddd-8f0b-49cf-99db-59ecbf1a7aae` | Deployment: `51e051dc-697f-4cd6-baa3-0e04e112d22b` | `SUCCESS`, Deployed: 2026-11-21 09:46:26 +01:00, Running SHA: `2eccc64a` in `STANDBY` |
| **Cloudflare Pages Production** | Project: `finance` (`www.arxterminal.com`) | Deployment: `c2c87761-312b-450d-9ac3-7e32d4dea5ce` | `SUCCESS`, Source SHA: `978123f7`, Active Hold: `HOLD_CF_AUTO` (`production_deployments_enabled = false`) |

---

## 3. Congressional versus SEC Form 4 Boundary

> [!WARNING]
> **Strict Governance Boundary:** Congressional STOCK Act disclosures and SEC Form 4 insider filings are **completely separate legal, statutory, and empirical data domains**. 
>
> The existing AAPL and NVDA SEC Form 4 trial dataset (`form4_raw_filings`, `form4_parsed_transactions`, `sm_reconciliation_receipts`) must **NEVER** be co-mingled, copied, relabeled, or counted as congressional disclosures.

### Domain Comparison Matrix

| Dimension | SEC Form 4 Disclosures | Congressional STOCK Act Disclosures |
| :--- | :--- | :--- |
| **Governing Statute** | Securities Exchange Act of 1934, § 16(a); 17 CFR § 240.16a-3 | Stop Trading on Congressional Knowledge (STOCK) Act of 2012 (Pub. L. 112–105; 5 U.S.C. app. § 101 et seq.) |
| **Covered Persons** | Corporate officers, directors, and beneficial owners of >10% of any class of equity securities | Members of the U.S. Senate and U.S. House of Representatives, their spouses, and dependent children |
| **Statutory Filing Window** | Strict **2 business days** following execution date | Strict **30 to 45 days** following notice/execution date |
| **Filing Identification** | 20-digit SEC Accession Number (`0000320193-26-000106`) | House Doc ID (`20025812`) or Senate Filing ID (`efd-search` internal GUID) |
| **Transaction Precision** | Exact share count, exact per-share price in USD, exact ownership type (Direct `D` or Indirect `I`) | Broad valuation ranges only ($1,001–$15,000, $15,001–$50,000, $50,001–$100,000, etc.); share counts and share prices not mandated |
| **Filing Portals** | SEC EDGAR API / Submission endpoints (`data.sec.gov`) | Office of the Clerk of the House (`disclosures-clerk.house.gov`) & Senate EFD (`efdsearch.senate.gov`) |
| **Ingestion Format** | Structured XML schema (`ownershipDocument.xml`) | House: Scanned PDF / e-filed PDF / XML annual index; Senate: EFD portal HTML / PDF |
| **Current ARX Status** | Governed limited trial (AAPL & NVDA) under `HOLD_SEC_001` with cryptographic watermarking | **Zero live ingestion; 0 rows stored; static curated archive only** |

### Critical Governance Conflation Finding
The codebase currently contains a severe conflation of governance controls:
In `analyst_dashboard/analyzers/smart_money.py` (lines 3505–3518), `assess_congressional_trades()` returns:
```python
return {
    "domain_id": "congressional_stock_act",
    "status": SmartMoneyDomainStatus.PIPELINE_HELD.value,
    "status_label": "Pipeline Pending (HOLD_SEC_001)",
    "available": False,
    "is_live": False,
    "observation_count": 0,
    "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
    "directional_bias": SmartMoneyDirection.UNASSESSED.value,
    "net_value_usd": 0.0,
    "provenance": "House & Senate Official STOCK Act Portals",
    "disclosure": "Congressional STOCK Act continuous sync paused under governed operational hold (HOLD_SEC_001).",
    "records": [],
}
```

**Defect Analysis:**
1. `HOLD_SEC_001` is the operational hold restricting automated, continuous egress to the **SEC EDGAR system** for Form 4 filings. It has **no legal or operational authority over Congressional House or Senate disclosures**.
2. Stating that the continuous sync is "paused under governed operational hold (HOLD_SEC_001)" falsely implies to users, auditors, and systems that a functioning continuous sync pipeline exists and is temporarily paused. In reality, **no such pipeline exists**.
3. **Remediation Requirement:** Congressional disclosures must be governed by an independent identifier (e.g., `HOLD_CONGRESS_001` or explicit status `COVERAGE_NOT_ESTABLISHED`) and completely decoupled from `HOLD_SEC_001`.

---

## 4. Complete Component Inventory

The following table provides a comprehensive, symbol-level audit of every file, function, class, and route in the ARX repository related to congressional disclosures:

| File Path | Symbol / Component | Purpose / Role | Reachable in Prod? | Backed by Live Evidence? | Existing Test Coverage | Recommended Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/smart_money.py#L13-L145) | `CONGRESSIONAL_TRADES` | In-memory static list of 40 historical trades across 13 tickers and 8 politicians (Aug 2026 snapshot). | Yes (when `include_curated=True`). | **NO** (Static research archive). | `test_smart_money_api.py`, `test_live_api_provenance.py` | **Preserve as offline research fixture; quarantine from production live responses.** |
| [smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/smart_money.py#L2926-L2936) | `SmartMoneyEngine.get_congressional_trades` | Core accessor for congressional trades. Returns `[]` if `include_curated=False`. | Yes (invoked by `/congress` and `assess_symbol`). | **NO** (Live returns empty; curated returns static). | `test_radar_cross_engine_parity_003.py`, `test_adversarial_phase16_live_data.py` | **Refactor:** Decouple from static list; interface with governed relational repository. |
| [smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/smart_money.py#L3362-L3518) | `SmartMoneyEngine.assess_congressional_trades` | Single-asset evaluation method. Determines bullish/bearish bias from buy/sell counts. | Yes (invoked during `assess_symbol`). | **NO** (Defaults to `PIPELINE_HELD (HOLD_SEC_001)`). | `test_sm_act_001_falsification.py`, `test_smart_money_e2e_parity.py` | **Refactor:** Remove `HOLD_SEC_001` string; emit `COVERAGE_NOT_ESTABLISHED` when unmonitored. |
| [smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/smart_money.py#L3755-L3816) | `SmartMoneyEngine.get_smart_money_overview` | Market-wide aggregate overview. Aggregates purchases/sales, sentiment, and late filers. | Yes (invoked by `/smart-money/overview`). | **NO** (Live returns all `None`/empty; curated aggregates static array). | `test_smart_money_api.py`, `test_live_api_provenance.py` | **Preserve:** Existing contract correctly separates live `None` from curated counts. |
| [capitol_trades_fetcher.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/data/capitol_trades_fetcher.py#L1-L24) | `CapitolTradesFetcher` | 24-line stub class returning static dictionary of House and Senate portal URLs. | Yes (invoked by `/congress` to populate `source_meta`). | **NO** (Contains zero network calls, zero scrapers, zero parsers). | None | **Deprecate / Replace:** Replace with an authentic primary-source adapter architecture. |
| [api/routes/smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/api/routes/smart_money.py#L50-L75) | `GET /api/v1/smart-money/congress` | FastAPI endpoint for congressional disclosures. | Yes (public API). | **NO** (Returns `UNAVAILABLE` or static array). | `test_smart_money_api.py`, `test_adversarial_phase17_data_reality.py` | **Preserve & Harden:** Ensure query parameter `include_curated` is never enabled by default. |
| [api/routes/smart_money.py](file:///Users/iDavid/Projects/finance-congressional-inventory/api/routes/smart_money.py#L31-L48) | `GET /api/v1/smart-money/overview` | FastAPI endpoint for market overview. Lists regulatory sources. | Yes (public API). | **NO** (Claims `capitol_trades` source metadata even when disconnected). | `test_smart_money_api.py` | **Refactor:** Only declare `regulatory_sources.capitol_trades` when feed is actively connected. |
| [confluence_engine.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/confluence_engine.py#L185-L201) | `ConfluenceEngine.calculate_confluence` | Evaluates `has_congress_buy`. If True, awards 85 points to smart money confluence score. | Yes (runs in Radar / Confluence scoring). | **NO** (Fed by upstream `assess_symbol` which is `False` in live mode). | `test_confluence_engine.py`, `test_directive_057_remediation.py` | **Preserve:** Logic correctly gates on boolean flag; safe as long as live feed does not fabricate buys. |
| [trader_archetypes.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/trader_archetypes.py#L88-L140) | `CONGRESSIONAL_POLICY_TICKERS` | Static dictionary mapping tickers (NVDA, LLY, etc.) to policy narrative strings. | Yes (runs in `The Capitol Whale` archetype). | **NO** (Static subjective priors, not transactional disclosures). | `test_trader_archetypes.py` | **Preserve as archetype prior:** Clearly label as policy thesis mapping, not trade filings. |
| [trader_archetypes.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/analyzers/trader_archetypes.py#L391-L430) | `_evaluate_congressional_whale` | Evaluates Nancy Pelosi / Congressional Whale archetype against factor scores. | Yes (runs in Radar / Archetypes). | **NO** (Does not read live trades; reads static dictionary). | `test_trader_archetypes.py` | **Preserve:** Archetype modeling distinct from raw transaction ingestion. |
| [market_db.py](file:///Users/iDavid/Projects/finance-congressional-inventory/analyst_dashboard/data/market_db.py#L236-L249) | Table `insider_disclosures` | SQLite DDL for politician and insider trades. | Schema created at init; never queried or populated. | **NO** (Contains 0 rows in production; abandoned schema). | `test_alpaca_historical_sip_ingestion.py` | **Deprecate:** Dead schema. Remove in future schema cleanup or migrate to PostgreSQL. |
| [CongressionalTradesCard.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/components/CongressionalTradesCard.tsx#L1-L560) | `CongressionalTradesCard` | Next.js React component rendering congressional trade tables, late filers, and crypto policy. | Yes (rendered on stock detail and smart money views). | **NO** (Renders static trades or empty state). | None (React UI). | **Refactor:** Fix misleading empty state copy (`"No qualifying congressional transactions found..."`). |
| [frontend/app/politician/[slug]/page.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/app/politician/[slug]/page.tsx#L1-L320) | `PoliticianProfilePage` | Dynamic Next.js route generating programmatic SEO profile pages for 8 politicians. | Yes (reachable via URL `/politician/<slug>`). | **NO** (Renders hardcoded trades from `seoCatalogs.ts`). | None (Next.js SSR). | **Quarantine / Disclose:** Add prominent badge: *"Historical Research Sample (Aug 2026)"*. |
| [seoCatalogs.ts](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/lib/seoCatalogs.ts#L308-L450) | `POLITICIAN_DATABASE` | Static catalog of 8 politicians, their committees, and hardcoded historical trades. | Yes (imported by politician dynamic routes). | **NO** (Static content). | None | **Preserve as static content:** Explicitly mark as non-live demonstration data. |
| [late-filers/page.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/app/smart-money/late-filers/page.tsx#L60-L180) | `LATE_TRADES` | Static array of late trades (Tuberville, Green) rendered as "Hall of Shame". | Yes (reachable via URL `/smart-money/late-filers`). | **NO** (Static mock data). | None | **Quarantine / Disclose:** Add explicit disclaimer that this is a static historical audit. |
| [frontend/lib/api.ts](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/lib/api.ts#L881-L894) | `SMART_MONEY` Pipeline Status | Strategy pipeline status metadata claiming single-asset congressional disclosures are active. | Yes (imported across UI). | **NO** (Claims single-asset disclosures active while live backend is disconnected). | None | **Refactor:** Correct string to indicate congressional disclosures are in development. |
| [frontend/lib/api.ts](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/lib/api.ts#L2635-L2655) | `fetchSmartMoneyOverview` | Client-side fetcher calling `/smart-money/overview?include_curated=true`. | Yes (invoked on smart money overview page). | **NO** (Forcibly passes `include_curated=true`). | None | **Refactor:** Stop requesting `include_curated=true` as default; display honest disconnected state. |

---

## 5. Production Configuration and Data Inventory

Direct authenticated inspection of the production runtime environment was performed on 2026-11-22 via Railway SSH into the active web service container (`56e38249-b78c-4a60-80d7-303ce7451bd8`).

### 5.1 Production PostgreSQL Inspection (`finance_smart_money`)
Connected via `psycopg2` within a strict read-only transaction (`SET TRANSACTION READ ONLY; BEGIN READ ONLY;`) to the database referenced by `SMART_MONEY_DATABASE_URL`:

- **Host:** Railway internal private network (`postgres.railway.internal:5432`)
- **Database:** `finance_smart_money`
- **User:** `smart_money_web`
- **Query Timestamp:** 2026-11-22 07:03:57 UTC
- **Observed Public Tables (10 total):**
  1. `form4_parsed_transactions`: 140 rows (SEC Form 4 AAPL/NVDA insider trades only)
  2. `form4_raw_filings`: 54 rows (SEC Form 4 raw XML filings only)
  3. `governed_issuers`: 13 rows (SEC CIK and ticker mappings)
  4. `scanner_active_snapshots`: 1 row (Radar active snapshot marker)
  5. `scanner_snapshots`: 1 row
  6. `sm_amendments`: 1 row (SEC Form 4 amendment tracking)
  7. `sm_audit_events`: 82 rows (SEC Form 4 worker audit logs)
  8. `sm_reconciliation_receipts`: 8 rows (SEC Form 4 qualification receipts)
  9. `sm_run_requests`: 6 rows
  10. `sm_source_availability_ledger`: 45 rows (SEC EDGAR availability logs)

**Result:** **ZERO congressional tables, schemas, or models exist in PostgreSQL.** There are zero rows of House or Senate disclosures.

### 5.2 Production SQLite Inspection
Direct read-only inspection of all local SQLite databases (`PRAGMA query_only = ON;`) on the web service container:
- Database: `/root/.finance_market_store.db` (also mirrored at `/root/.finance_scanner_store.db` and `/root/.finance_universe_store.db`)
- Checked table: `insider_disclosures`
- Query: `SELECT COUNT(*) FROM insider_disclosures;`
- **Result:** **0 rows.**
- Search of codebase confirms that `insider_disclosures` is created in `market_db.py` line 236 but is **never inserted into, updated, or queried** by any application logic.

### 5.3 Production Environment Variables Audit
Inspected 80 environment variable names on the active Railway `web` container (`/proc/1/environ`) and worker container:
- SEC Form 4 variables present: `SMART_MONEY_DATABASE_URL`, `ARX_SEC_EGRESS_ROLE=DISABLED`, `SEC_EDGAR_USER_AGENT`, `SEC_FORM4_INGESTION_ENABLED=0`, `SMART_MONEY_WORKER_STANDBY=true`.
- **Congressional variables present:** **EXACTLY ZERO.**
  - No `CAPITOL_TRADES_API_KEY`
  - No `HOUSE_CLERK_INGESTION_ENABLED`
  - No `SENATE_EFD_SYNC_ENABLED`
  - No `STOCK_ACT_DATABASE_URL`

---

## 6. Evidence-Source and Provenance Assessment

To transition from static mock data to verified, evidence-backed disclosures, ARX must ingest from authoritative sources. The following evaluation analyzes primary government sources versus third-party aggregators:

```mermaid
flowchart LR
    subgraph Primary Official Sources
        H[House Clerk PTR Portal<br>disclosures-clerk.house.gov] -->|FinancialDisclosure.xml Index<br>+ Scanned/E-filed PDFs| H_P[House Parser / OCR Engine]
        S[Senate EFD Portal<br>efdsearch.senate.gov] -->|HTML / JSON Search<br>Requires TOS & Cookie| S_P[Senate EFD Parser]
    end

    subgraph Third-Party Aggregators
        C[Capitol Trades / 2iQ] -->|Commercial API / Web Scrape| C_A[Vendor Dependency]
        Q[Quiver Quantitative] -->|Commercial API| Q_A[Vendor Dependency]
    end

    subgraph ARX Ingestion Engine
        H_P --> G[Governed Evidence Staging]
        S_P --> G
        G --> R[Cryptographic Receipt Ledger]
        R --> DB[(PostgreSQL: congressional_*)]
    end
```

### 6.1 U.S. House of Representatives Office of the Clerk
- **Authority:** Office of the Clerk, U.S. House of Representatives (Legislative Resource Center).
- **Portal URL:** `https://disclosures-clerk.house.gov`
- **Filing Type:** Financial Disclosure Reports — Periodic Transaction Reports (PTRs) filed under the STOCK Act (Pub. L. 112–105).
- **Available Data Assets:**
  1. **Annual XML Index:** The Clerk publishes an annual zip archive containing `FinancialDisclosure.xml` (or `FD.xml`), which indexes all filings by Member Name, Office/District, Filing Year, Filing Date, and Doc ID.
  2. **Document PDFs:** Individual PTRs are available at predictable URLs:
     `https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/<YEAR>/<DOC_ID>.pdf`
- **Format Challenges:**
  - Roughly 70% of modern House PTRs are e-filed digital forms containing readable text layers.
  - Roughly 30% are paper submissions or scanned documents containing handwriting, requiring Optical Character Recognition (OCR) or manual transcription.
- **Access & Licensing:**
  - Public government records under 5 U.S.C. app. § 105.
  - Zero API authentication required; rate-limiting must be respected (max 1 request per second recommended to avoid IP blocking).

### 6.2 U.S. Senate Electronic Financial Disclosure (EFD)
- **Authority:** Select Committee on Ethics & Office of Public Records, U.S. Senate.
- **Portal URL:** `https://efdsearch.senate.gov`
- **Filing Type:** Periodic Transaction Reports (e-filed electronically).
- **Format:** Electronic web portal delivering HTML search results and PDF reports.
- **Access Restrictions & Challenges:**
  - **Statutory Terms of Service:** Users must agree that reports will not be used for commercial purposes (other than by news and communications media for publication) or for soliciting contributions (5 U.S.C. app. § 105(c)(1)).
  - **Anti-Bot & Session Architecture:** The Senate EFD portal utilizes CSRF tokens, session cookies, and WAF challenge screens. Automated programmatic scraping without standard browser agreement violates the site's Terms of Service and Anti-Bot rules.
  - **Ingestion Recommendation:** Direct scraping of Senate EFD must be approached through sanctioned bulk XML/CSV data feeds or compliant batch fetch mechanisms that store complete provenance receipts.

### 6.3 Third-Party Aggregators (Capitol Trades, Quiver Quantitative)
- **Capitol Trades (`capitoltrades.com`):** Aggregates House and Senate trades into a commercial feed with politician committee mappings and sector tags.
- **Evaluation:** Relying on third-party aggregators introduces single-point-of-failure vendor risk, recurring licensing costs, and unverifiable provenance chains.
- **Architectural Decision:** ARX must prioritize **primary official government sources** (House Clerk XML indexes and Senate public disclosures) for its core database, storing the original government document URL and SHA256 digest on every transaction record.

---

## 7. Schema and Pipeline Topology

To establish production-grade, cryptographically verifiable congressional disclosures, ARX should implement the following relational schema in PostgreSQL, completely decoupled from SEC Form 4 tables:

```mermaid
erDiagram
    CONGRESSIONAL_MEMBERS ||--o{ CONGRESSIONAL_FILINGS : files
    CONGRESSIONAL_FILINGS ||--o{ CONGRESSIONAL_TRANSACTIONS : contains
    CONGRESSIONAL_FILINGS ||--o{ CONGRESSIONAL_RECEIPTS : audited_by

    CONGRESSIONAL_MEMBERS {
        varchar bioguide_id PK
        varchar first_name
        varchar last_name
        varchar chamber
        varchar state
        varchar district
        varchar party
        jsonb committee_assignments
    }

    CONGRESSIONAL_FILINGS {
        varchar filing_id PK
        varchar chamber
        varchar doc_id
        varchar bioguide_id FK
        varchar filing_type
        date filing_date
        date received_date
        varchar source_url
        varchar raw_sha256
        timestamp ingested_at
    }

    CONGRESSIONAL_TRANSACTIONS {
        uuid id PK
        varchar filing_id FK
        varchar bioguide_id FK
        varchar ticker
        varchar asset_name
        varchar asset_type
        varchar transaction_type
        date transaction_date
        date notification_date
        varchar amount_range
        numeric amount_min
        numeric amount_max
        varchar owner_type
        text comment
    }

    CONGRESSIONAL_RECEIPTS {
        uuid receipt_id PK
        varchar chamber
        date coverage_start
        date coverage_end
        varchar watermark_hash
        boolean is_complete
        timestamp certified_at
    }
```

### Proposed Schema DDL

```sql
-- 1. Lawmakers & Congressional Representatives
CREATE TABLE IF NOT EXISTS congressional_members (
    bioguide_id VARCHAR(16) PRIMARY KEY, -- Canonical US Congress Bioguide ID
    first_name VARCHAR(64) NOT NULL,
    last_name VARCHAR(64) NOT NULL,
    chamber VARCHAR(16) NOT NULL CHECK (chamber IN ('House', 'Senate')),
    state VARCHAR(2) NOT NULL,
    district VARCHAR(4), -- NULL for Senators
    party VARCHAR(32) NOT NULL,
    official_url TEXT,
    committee_assignments JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Official Statutory Filings (PTRs)
CREATE TABLE IF NOT EXISTS congressional_filings (
    filing_id VARCHAR(64) PRIMARY KEY, -- e.g. "HOUSE-2026-20025812"
    chamber VARCHAR(16) NOT NULL,
    doc_id VARCHAR(64) NOT NULL,
    bioguide_id VARCHAR(16) REFERENCES congressional_members(bioguide_id),
    filing_type VARCHAR(32) NOT NULL DEFAULT 'PTR', -- 'PTR', 'ANNUAL', 'AMENDMENT'
    filing_date DATE NOT NULL,
    received_date DATE,
    source_url TEXT NOT NULL,
    raw_document_sha256 VARCHAR(64) NOT NULL, -- SHA256 of downloaded PDF or XML
    amends_filing_id VARCHAR(64) REFERENCES congressional_filings(filing_id),
    ingestion_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Disclosed Asset Transactions
CREATE TABLE IF NOT EXISTS congressional_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id VARCHAR(64) NOT NULL REFERENCES congressional_filings(filing_id) ON DELETE CASCADE,
    bioguide_id VARCHAR(16) NOT NULL REFERENCES congressional_members(bioguide_id),
    ticker VARCHAR(16), -- Can be NULL if non-publicly traded asset or unlisted bond
    asset_name TEXT NOT NULL,
    asset_type VARCHAR(32) NOT NULL, -- 'Common Stock', 'Call Option', 'Corporate Bond', etc.
    transaction_type VARCHAR(32) NOT NULL, -- 'Purchase', 'Sale (Full)', 'Sale (Partial)', 'Exchange'
    transaction_date DATE NOT NULL,
    notification_date DATE,
    amount_range VARCHAR(64) NOT NULL, -- '$1,001 - $15,000', '$15,001 - $50,000', etc.
    amount_min NUMERIC(14, 2) NOT NULL,
    amount_max NUMERIC(14, 2), -- Can be NULL for top open-ended bracket (>$50,000,000)
    owner_type VARCHAR(32) NOT NULL DEFAULT 'Self', -- 'Self', 'Spouse', 'Child', 'Joint'
    days_to_filing INTEGER GENERATED ALWAYS AS (filing_date - transaction_date) STORED,
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cong_tx_ticker ON congressional_transactions (ticker);
CREATE INDEX IF NOT EXISTS idx_cong_tx_date ON congressional_transactions (transaction_date);
CREATE INDEX IF NOT EXISTS idx_cong_tx_member ON congressional_transactions (bioguide_id);

-- 4. Cryptographic Reconciliation & Coverage Ledger
CREATE TABLE IF NOT EXISTS congressional_receipts (
    receipt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber VARCHAR(16) NOT NULL,
    coverage_start DATE NOT NULL,
    coverage_end DATE NOT NULL,
    total_filings_checked INTEGER NOT NULL,
    watermark_hash VARCHAR(64) NOT NULL,
    is_complete BOOLEAN NOT NULL DEFAULT FALSE,
    certified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

---

## 8. UI Truthfulness Assessment

A systematic review of the frontend presentation reveals several critical truthfulness gaps where absence of data is either mislabeled or where users could infer that a comprehensive live check was completed.

### 8.1 Misleading Empty & Held States
In [CongressionalTradesCard.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/components/CongressionalTradesCard.tsx#L530-L534):
```tsx
) : status === "VALID_EMPTY" || (status === "AVAILABLE" && filteredCongressTrades.length === 0) ? (
  <div className="p-4 bg-[#090d14] rounded-lg border border-[#243044] text-center text-xs text-slate-400">
    No qualifying congressional transactions found in the verified coverage period.
  </div>
```
- **Finding:** If the backend passes `status="AVAILABLE"` and an empty trade array (or defaults to it), the card displays: *"No qualifying congressional transactions found in the verified coverage period."*
- **Truthfulness Violation:** No coverage period has been verified. Calling an unverified period "verified" violates the core ARX Zero Fabricated Data invariant.

### 8.2 Misleading Governance Notice
In `SmartMoneyEngine.assess_congressional_trades`:
- **Finding:** Mislabels congressional unavailability as `"Pipeline Pending (HOLD_SEC_001)"`.
- **Truthfulness Violation:** Suggests that SEC Form 4 regulatory holds apply to Congress and that a paused pipeline exists, masking the reality that no ingestion pipeline is running.

### 8.3 Hardcoded Static Marketing Pages
- [frontend/app/smart-money/late-filers/page.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/app/smart-money/late-filers/page.tsx): Renders a static `"Congressional Late-Filer Hall of Shame"` with hardcoded records for Tuberville and Green with `-32 pt staleness penalties` presented as a live dataset.
- [frontend/app/politician/[slug]/page.tsx](file:///Users/iDavid/Projects/finance-congressional-inventory/frontend/app/politician/[slug]/page.tsx): Generates pages for 8 politicians claiming live radar tracking while reading entirely static catalogs.

### Recommended Truthful UI States
1. **Unmonitored / Disconnected State (Current Reality):**
   - **Label:** `Congressional Disclosures: NOT MONITORED`
   - **Copy:** *"Congressional STOCK Act disclosures are not currently monitored in live mode. Automated retrieval for House and Senate PTRs is pending pipeline implementation."*
2. **Explicit Research Archive Badge (When Curated is Selected):**
   - **Badge:** `HISTORICAL BENCHMARK DATA (AUG 2026)`
   - **Copy:** *"Displaying 40 curated historical research records from August 2026. This is not a real-time feed."*
3. **True Empty State (Future Ingested State):**
   - Only when a cryptographic receipt proves that all House and Senate filings in the past 90 days were parsed and 0 matched the ticker should the UI display: *"0 filings matched ticker in the verified coverage period [Start Date to End Date]."*

---

## 9. Security, Legal, Operational, and Data-Quality Gaps

| Category | Specific Risk / Gap | Technical Impact | Remediation Strategy |
| :--- | :--- | :--- | :--- |
| **Legal / Compliance** | Senate EFD Terms of Service (5 U.S.C. app. § 105(c)(1)) restricts commercial reuse of disclosures. | Headless scraping without legal review exposes platform to statutory challenges. | Consult legal counsel on financial news/commentary exemptions; evaluate sanctioned data distributors. |
| **Anti-Bot Defeat** | Senate EFD portal requires interactive session verification and blocks automated headless clients. | Antigravity safety rules strictly forbid bypassing anti-bot challenges or CAPTCHAs. | Use official bulk XML downloads or compliant human-assisted staging pipelines. |
| **Data Quality** | Scanned / handwritten House PTR PDFs. | PDF text extractors return empty strings; OCR error rates produce mangled tickers (e.g. `NVDA` read as `NYDA`). | Implement confidence scoring; flag OCR-parsed records as `PARTIAL_UNVERIFIED` until verified. |
| **Data Ambiguity** | Broad statutory amount ranges ($1k–$15k, $15k–$50k, etc.). | Exact dollar amounts and volume-weighted prices cannot be computed; volume numbers in UI would be fabricated. | Enforce range representations (`amount_range`) across all schemas, APIs, and UI components; prohibit exact dollar summation. |
| **Filing Latency** | 30 to 45-day statutory filing lag under the STOCK Act. | Trades are already stale when disclosed; market prices have incorporated information. | Model staleness decay explicitly; never present delayed trades as real-time momentum signals. |
| **Entity Ambiguity** | Lawmakers filing under personal names, spouses, or dependent children without standard Bioguide IDs. | Fragile string matching can misattribute trades or confuse representatives with identical surnames. | Maintain canonical Member Master mapping Bioguide IDs to official names and districts. |

---

## 10. Readiness Matrix

Evaluation of 15 operational, technical, and architectural capabilities required for production congressional disclosures:

| # | Capability | Status | Evidence / Reason |
| :---: | :--- | :--- | :--- |
| 1 | **Database Schema (PostgreSQL)** | `MISSING` | Direct query confirms 0 tables exist in `finance_smart_money`. |
| 2 | **Database Storage (SQLite)** | `MISSING` | Table `insider_disclosures` contains 0 rows and is abandoned. |
| 3 | **Production Worker Tasks** | `MISSING` | `smart-money-worker` has 0 congressional jobs or cron tasks. |
| 4 | **Environment Configuration** | `MISSING` | Exactly 0 congressional environment variables on Railway. |
| 5 | **Primary House Clerk Adapter** | `MISSING` | `capitol_trades_fetcher.py` is an empty 24-line URL stub. |
| 6 | **Primary Senate EFD Adapter** | `MISSING` | Zero Senate fetchers or parsers exist in codebase. |
| 7 | **PDF / OCR Parsing Engine** | `MISSING` | No PDF parsing dependencies (`pypdf`, `pdfminer`, `tesseract`) configured for worker. |
| 8 | **Bioguide Lawmaker Master** | `PARTIAL` | 8 politicians hardcoded in `seoCatalogs.ts`; 0 comprehensive lists in database. |
| 9 | **API Endpoint Contract** | `PARTIAL` | `/api/v1/smart-money/congress` exists and handles `include_curated`, but live mode returns static empty state. |
| 10 | **Confluence Scoring Integration** | `VERIFIED_READY` | `ConfluenceEngine` safely consumes `has_congress_buy` boolean flag without fabricating data. |
| 11 | **Curated Research Fixtures** | `VERIFIED_READY` | 40 historical trades from Aug 2026 available for offline regression testing. |
| 12 | **Automated Test Coverage** | `PARTIAL` | Tests verify that live mode returns empty, but no tests exercise real ingestion. |
| 13 | **UI Truthfulness & State Copy** | `MISSING` | UI conflates `HOLD_SEC_001` and claims "verified coverage" on empty results. |
| 14 | **Cryptographic Receipt Ledger** | `MISSING` | No receipt schema or hashing exists for congressional files. |
| 15 | **Operational Governance Gate** | `HELD` | Governed hold required prior to any outbound network requests to congressional portals. |

---

## 11. Smallest Safe Implementation Sequence

To safely build congressional disclosure capability without risking regressions to SEC Form 4 or violating platform safety guidelines, the following incremental sequence is recommended:

```mermaid
flowchart TD
    P0[Phase 0: Truthful UI & Governance Decoupling<br>- Remove false HOLD_SEC_001 strings<br>- Mislabeled empty states replaced with honest unmonitored copy]
    P1[Phase 1: Database Migration & Lawmaker Master<br>- Apply DDL for congressional_* tables<br>- Seed official Bioguide lawmakers table]
    P2[Phase 2: Offline House XML Parser & Test Suite<br>- Download sample FinancialDisclosure.xml fixture<br>- Build parser with 100% offline unit tests]
    P3[Phase 3: Governed Ingestion Worker & Cryptographic Receipts<br>- Scheduled job with strict rate limits<br>- PDF SHA256 hashing and receipt generation]
    P4[Phase 4: API & Frontend Verification<br>- Connect live endpoints to PostgreSQL<br>- Expose verified coverage watermarks in UI]
    
    P0 --> P1 --> P2 --> P3 --> P4
```

### Phase 0: Truthful UI & Governance Decoupling (Immediate / Zero-Risk)
1. In `analyst_dashboard/analyzers/smart_money.py`, update `assess_congressional_trades` to return:
   - `status`: `SmartMoneyDomainStatus.UNAVAILABLE.value`
   - `status_label`: `"Congressional Disclosures: Unmonitored"`
   - `disclosure`: `"Congressional STOCK Act live ingestion is not currently active."`
   - Completely remove the string `HOLD_SEC_001`.
2. In `frontend/components/CongressionalTradesCard.tsx`, update the empty state copy to say:
   - *"Congressional disclosures are not currently monitored in live mode."*
3. In `frontend/lib/api.ts`, update `SMART_MONEY` pipeline rationale to reflect that congressional ingestion is in planning.

### Phase 1: Database Migration & Lawmaker Master
1. Create a dedicated migration adding `congressional_members`, `congressional_filings`, `congressional_transactions`, and `congressional_receipts` to `finance_smart_money`.
2. Seed `congressional_members` with official US Congress Bioguide data (535 active lawmakers + leadership).

### Phase 2: Offline House XML Parser & Test Suite
1. Build an offline ingestion parser for House Clerk `FinancialDisclosure.xml` files.
2. Create test suites in `tests/` using frozen local XML fixtures to verify parsing accuracy, amendment handling, and ticker extraction.

### Phase 3: Governed Ingestion Worker & Cryptographic Receipts
1. Introduce a scheduled background task in `smart-money-worker` under an explicit governed flag: `CONGRESSIONAL_INGESTION_ENABLED=1`.
2. Enforce strict egress rate limits (max 1 req/sec).
3. Compute and store SHA256 digests of all raw downloaded files. Generate a `congressional_receipts` row certifying daily coverage.

### Phase 4: API & Frontend Verification
1. Update `GET /api/v1/smart-money/congress` to query `congressional_transactions` in PostgreSQL.
2. Display verified coverage dates and raw government PDF links directly in `SmartMoneyDetailModal`.

---

## 12. Explicit NOT_CHECKED Register

In accordance with strict verification standards, the following systems and capabilities were not tested with live mutations or interactive traffic and remain classified as `NOT_CHECKED`:

| Target System / Property | Status | Rationale & Constraint |
| :--- | :--- | :--- |
| **Live House Clerk Portal Egress** | `NOT_CHECKED` | Outbound network calls to `disclosures-clerk.house.gov` were prohibited during read-only discovery. |
| **Live Senate EFD Interactive Search** | `NOT_CHECKED` | Direct crawling of `efdsearch.senate.gov` requires human TOS acceptance; prohibited under agent anti-bot guidelines. |
| **Commercial Vendor API Feeds** | `NOT_CHECKED` | No active commercial subscriptions (Capitol Trades / Quiver) exist in environment variables. |
| **Production PostgreSQL DDL Execution** | `NOT_CHECKED` | Database access was strictly restricted to read-only transactions (`SET TRANSACTION READ ONLY`); no tables created. |
| **Worker Ingestion Concurrency** | `NOT_CHECKED` | Worker is currently held in `STANDBY` mode; multi-worker execution was not initiated. |

---

## 13. Backlog Placement, Accountable Owner, and Acceptance Criteria

- **Recommended Project / Epic:** ARX Core Ingestion Architecture — Sprint P4
- **Related Issue Trackers:** Issue #165, Sprint P3 #231
- **Proposed Feature Branch:** `feat/congressional-stock-act-ingestion`
- **Accountable Owner:** ARX Data Engineering & Regulatory Intelligence Team
- **Prerequisite Release Gate:** Decouple `HOLD_SEC_001` from all congressional code and verify zero impact on SEC Form 4 AAPL/NVDA trial baseline.

### Acceptance Criteria for Step 2 Activation
1. Complete separation of SEC Form 4 and Congressional STOCK Act data stores.
2. Zero synthetic, simulated, or unverified mock data presented without explicit `CURATED_RESEARCH` badging.
3. Every stored transaction links to an authoritative government document identifier and valid source URL.
4. Cryptographic receipt ledger proves completeness and watermark dates for every query response.
5. All live empty states truthfully state the monitored coverage period or non-monitored status.

---
**Sign-off:** ARX Architecture & Regulatory Governance Lead  
**Document Digest:** Verified Read-Only Inventory complete. Saved to documentation branch `docs/congressional-ingestion-inventory-001`.
