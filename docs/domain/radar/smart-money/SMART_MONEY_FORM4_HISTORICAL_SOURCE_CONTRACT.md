# ARX TERMINAL — RADAR SMART MONEY
## SEC FORM 4 HISTORICAL SOURCE CONTRACT

**Document ID**: `SMART-MONEY-FORM4-HIST-SOURCE-001`  
**Version**: `1.0.0`  
**Effective Date**: `2026-10-10`  
**Authority**: SEC EDGAR Regulatory Filing & Corporate Disclosure System  
**Statutory Basis**: Securities Exchange Act of 1934 — Section 16(a)  
**Governance Scope**: Historical Discovery Paths, Submission Indexes, Document Archives, Rate Control & Failure Classification

---

## 1. Upstream SEC Discovery Architecture

SEC EDGAR provides regulatory filing data through a deterministic hierarchy of public endpoints.

```
Upstream SEC EDGAR
  │
  ├── 1. Submissions Metadata Index
  │     └── GET /submissions/CIK{10-digit-cik}.json
  │           ├── filings.recent: Array-of-arrays (up to ~1,000 recent filings)
  │           │     ├── accessionNumber
  │           │     ├── filingDate
  │           │     ├── acceptanceDateTime  <-- CANONICAL PUBLIC AVAILABILITY
  │           │     ├── form ("4", "4/A")
  │           │     └── primaryDocument (e.g. "form4.xml")
  │           └── filings.files: Historical partition files (for >1,000 filings)
  │                 └── {"name": "CIK...-submissions-001.json", "filingTo": ..., "filingFrom": ...}
  │
  ├── 2. Historical Submissions Partitions
  │     └── GET /submissions/{file_name}
  │
  └── 3. SEC Archives Raw Document Delivery
        ├── Priority A: GET /Archives/edgar/data/{cik_int}/{clean_accession}/{primary_doc_name}
        └── Fallback B: GET /Archives/edgar/data/{cik_int}/{clean_accession}/{accession_number}.txt
```

---

## 2. Source Field Authority & Mapping

| Field | SEC Source Name | Format / Type | Purpose |
| :--- | :--- | :--- | :--- |
| **`accession_number`** | `accessionNumber` | String (e.g. `0000320193-26-000100`) | Immutable SEC submission identifier. |
| **`filing_date`** | `filingDate` | String `YYYY-MM-DD` | Calendar date of filing. |
| **`acceptance_datetime`** | `acceptanceDateTime` | ISO 8601 UTC string | **Canonical public availability timestamp**. |
| **`form_type`** | `form` | String (`"4"`, `"4/A"`) | Filing category (Original vs Amendment). |
| **`primary_document`** | `primaryDocument` | String (e.g. `form4.xml`) | Relative path to ownership XML document. |

---

## 3. Request Pattern & Rate Governance

1. **Fair Access Limit**: SEC enforces a hard ceiling of 10 requests per second.
2. **ARX Operating Limit**: Enforces **5.0 requests per second** maximum via `RateLimiter`.
3. **User-Agent Requirement**: Every HTTP header includes compliant institutional identification:  
   `User-Agent: FinanceTerminalResearchApp admin@financeterminal.internal`.
4. **Transport Resilience**:
   - Exponential backoff on HTTP 429 / 5xx (`1.0s -> 2.0s -> 4.0s`).
   - Maximum 3 retries per request.
   - Cache-first retrieval: If accession already exists with matching content hash, network call is bypassed.

---

## 4. Source Disposition Closed Set

No silent skips are permitted. Every filing encountered must resolve into one of seven governed dispositions:

1. `FETCHED_AND_PARSED`: Raw content fetched, parsed successfully, normalized transactions persisted.
2. `FETCHED_NO_RELEVANT_TRANSACTIONS`: Filing parsed successfully, but contained zero Section 16 acquisitions/dispositions.
3. `FETCH_FAILED_RETRYABLE`: Network timeout or HTTP 5xx during retrieval.
4. `FETCH_FAILED_TERMINAL`: Permanent HTTP 404 or corrupted transport error.
5. `PARSE_FAILED`: Corrupted XML syntax or unrecognized schema.
6. `UNSUPPORTED_DOCUMENT`: Non-XML legacy document format.
7. `AMENDMENT_UNRESOLVED`: Form 4/A amendment that could not be linked to prior filing lineage.

**Invariant**: After a declared COMPLETE bounded backfill, $\text{UNCLASSIFIED\_FILINGS} = 0$.
