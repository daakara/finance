# ARX TERMINAL — EVIDENCE INDEX: DATA.GOV & PRIMARY SOURCES (CORRECTED)
**Backlog Reference:** `BACKLOG-DATA-GOV-DISCOVERY-001`  
**Workstream:** Worktree B (Independent Discovery & Qualification)  
**Target Repository:** `daakara/finance`  
**Branch:** `research/data-gov-discovery`  
**Base Commit:** `fcb43f7aae1797bde762b91b5c309817ab25f459`  
**Audit Date:** 2026-10-16

---

## 1. Evidence Protocol and Hash Integrity Statement

To maintain strict auditability while preventing repository bloat:
1. **Isolated Storage:** Raw sample response payloads were saved to local scratch storage:
   `C:\Users\akara\.gemini\antigravity\brain\1f9bcc84-ac95-4b1d-baf9-b4d4d76f13c7\scratch\data_gov_samples\`
2. **Hash Clarification:** All cryptographic digests listed below are **SHA-256 hashes of the exact raw HTTP response bytes as received over the wire**. They serve as tamper-evident fingerprints of the retrieved byte stream. They do *not* represent cryptographic proofs of institutional authenticity from the publishing agencies. Reserializing the JSON with different indentation or key ordering will produce a different hash.
3. **Reproducibility Limitations:** Live public APIs (especially daily Treasury cash balances and weekly CFTC reports) advance over time. Queries executed without explicit date filters will return newer observations than those recorded here.

---

## 2. Empirical Probe Log

### Probe 1: Data.gov CKAN API Programmatic Search (Catalog Discovery Probe)
- **Target URL:** `https://catalog.data.gov/api/3/action/package_search?q=cftc+commitments+of+traders&rows=3`
- **Method:** `GET`
- **Timestamp:** `2026-10-16T07:16:54Z`
- **Observed HTTP Status:** `HTTP 404: Not Found`
- **Finding:** Programmatic catalog discovery via the legacy CKAN `package_search` endpoint failed. In accordance with Phase 3 instructions, catalog discovery via this specific API endpoint is marked **`ACCESS_OR_QUALIFICATION_INCOMPLETE`**, and qualification proceeded directly using authoritative primary-agency documentation.

---

### Probe 2: CFTC Commitments of Traders (COT) — Legacy Futures Only
- **Target URL:** `https://publicreporting.cftc.gov/resource/6dca-aqww.json?%24limit=2&%24order=report_date_as_yyyy_mm_dd%20DESC`
- **Method:** `GET` | Headers: `User-Agent: ARX-Research-Discovery/1.0`
- **Timestamp:** `2026-10-16T07:18:43Z`
- **HTTP Status:** `200 OK` | Content-Type: `application/json; charset=utf-8`
- **Local Artifact:** `scratch/data_gov_samples/cftc_cot_sample.json`
- **Raw Bytes SHA-256:** `9488316df08075f8bfbc155faee4ce349faaa35ce1a1796c8eb7262dc14c6e91`
- **Observed Fields:** `id`, `market_and_exchange_names`, `report_date_as_yyyy_mm_dd`, `contract_market_name`, `open_interest_all`, `noncomm_positions_long_all`, `noncomm_positions_short_all`, `comm_positions_long_all`, `comm_positions_short_all`.
- **Finding:** Unauthenticated Socrata API query succeeded, returning flat JSON key-value records with numeric strings.

---

### Probe 3: CFTC COT — Specific `E-MINI S&P 500` Contract Market Probe
- **Target URL:** `https://publicreporting.cftc.gov/resource/6dca-aqww.json?contract_market_name=E-MINI%20S%26P%20500&%24order=report_date_as_yyyy_mm_dd%20DESC&%24limit=1`
- **Method:** `GET` | Headers: `User-Agent: ARX-Research-Discovery/1.0`
- **Timestamp:** `2026-10-16T07:19:44Z`
- **HTTP Status:** `200 OK` | Content-Type: `application/json; charset=utf-8`
- **Local Artifact:** `scratch/data_gov_samples/cftc_emini_sp500_sample.json`
- **Raw Bytes SHA-256:** `1dbb91b5042cf871d374534a66453ae661917f354f3be2ca0d2cb2f11400d331`
- **Observed Record Details:**
  - `contract_market_name`: `"E-MINI S&P 500"`
  - `market_and_exchange_names`: `"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"`
  - `cftc_contract_market_code`: `"13874A"`
  - `report_date_as_yyyy_mm_dd`: `"2026-10-06T00:00:00.000"` (Tuesday cutoff)
  - `open_interest_all`: `"1932227"`
  - `noncomm_positions_long_all`: `"212009"`
  - `noncomm_positions_short_all`: `"366079"`
  - `comm_positions_long_all`: `"1437320"`
  - `comm_positions_short_all`: `"1394255"`
  - `traders_tot_all`: `"441"`
- **Reconciliation Note:** At the retrieval timestamp (2026-10-16 07:19 UTC), the 2026-10-06 report was the latest available published report; the 2026-10-13 report was scheduled for release later that day at 15:30 ET.

---

### Probe 4: Treasury Fiscal Data — Operating Cash Balance (TGA)
- **Target URL:** `https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance?sort=-record_date&page[size]=5`
- **Method:** `GET` | Headers: `User-Agent: ARX-Research-Discovery/1.0`
- **Timestamp:** `2026-10-16T07:33:43Z`
- **HTTP Status:** `200 OK` | Content-Type: `application/json`
- **Local Artifact:** `scratch/data_gov_samples/treasury_operating_cash_balance_sample.json`
- **Raw Bytes SHA-256:** `12b0455e71ff28d567d9f96ee7859842361d5f2f3ecef28117607fbd56eb5fb9`
- **Observed Record Details:**
  - Endpoint Category: Daily Treasury Statement (DTS) Table 1.
  - Record Date: `"2026-10-14"`.
  - TGA Opening Balance (`open_today_bal`): `"920214"` ($920,214 Million = $920.21 Billion).
  - Total TGA Deposits: `"17536"` ($17.54 Billion).
  - Total TGA Withdrawals: `"51928"` ($51.93 Billion).
  - TGA Closing Balance (`open_today_bal`): `"885823"` ($885.82 Billion).
- **Finding:** Verifies the exact dataset and fields for Treasury General Account operating cash tracking. Zero authentication required.

---

### Probe 5: Treasury Fiscal Data — Debt to the Penny (Reference Debt Stock)
- **Target URL:** `https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?sort=-record_date&page[size]=2`
- **Method:** `GET` | Headers: `User-Agent: ARX-Research-Discovery/1.0`
- **Timestamp:** `2026-10-16T07:18:45Z`
- **HTTP Status:** `200 OK` | Content-Type: `application/json`
- **Local Artifact:** `scratch/data_gov_samples/treasury_debt_penny_sample.json`
- **Raw Bytes SHA-256:** `9bf7eb58db9cfdbb279a7863cb338b25c34796263595ebbbbb978939a8523c14`
- **Observed Record Details:** Record date `"2026-10-14"`, total public debt `"40329229827541.02"`.

---

### Probe 6: Bureau of Labor Statistics (BLS API v2) — CPI All Urban Consumers
- **Target URL:** `https://api.bls.gov/publicAPI/v2/timeseries/data/`
- **Method:** `POST` | Headers: `Content-Type: application/json`, `User-Agent: ARX-Research-Discovery/1.0`
- **Payload:** `{"seriesid": ["CUSR0000SA0"], "startyear": "2024", "endyear": "2024"}`
- **Timestamp:** `2026-10-16T07:18:52Z`
- **HTTP Status:** `200 OK` | Content-Type: `application/json`
- **Local Artifact:** `scratch/data_gov_samples/bls_cpi_sample.json`
- **Raw Bytes SHA-256:** `5b8219b16be9cebf03c155d9a263c9b78e121ee9046311e967a5b3a4f650deca`
- **Observed Record Details:** December 2024 index value `"317.604"`, response status `"REQUEST_SUCCEEDED"`.
- **Finding:** Demonstrates unauthenticated functionality, but confirms restrictive limits for multi-series querying.
