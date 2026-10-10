# SMART MONEY SOURCE AVAILABILITY CONTRACT
**Wave 2B — Source-Availability Ledger & Knowledge-Time Semantics**
**Document ID**: `SM-DOC-WAVE2B-SOURCE-AVAILABILITY-CONTRACT`
**Authority**: Independent Smart Money Historical Data Authority

---

## 1. Executive Summary & Purpose

In financial econometric backtesting and production quantitative systems, look-ahead bias emerges whenever information is assumed known prior to its actual public dissemination. This contract specifies the authoritative knowledge-time model, the public-availability ledger schema, and the immutable rules governing SEC Form 4 availability timestamps.

---

## 2. Four Distinct Time Horizons

Every Smart Money record tracks four separate and non-fungible time horizons:

```
+-----------------------------------------------------------------------------------------+
|                               FOUR TEMPORAL HORIZONS                                    |
+-----------------------------------------------------------------------------------------+
| 1. Transaction Execution Time (`transaction_date`)                                      |
|    - The date on which the insider executed the trade (from XML <transactionDate>).     |
|    - CANNOT BE USED AS KNOWLEDGE TIME. Insiders typically file 1 to 2 business days    |
|      after trade execution. Using this date causes severe look-ahead leakage.           |
+-----------------------------------------------------------------------------------------+
| 2. SEC Public-Availability Time (`public_availability_at` / `acceptance_datetime`)      |
|    - The precise ISO timestamp when the SEC EDGAR system accepted the filing and made   |
|      it available to public feeds (derived from SEC submission `acceptanceDateTime`).   |
|    - THIS IS THE STRICT KNOWLEDGE-TIME BOUNDARY for all models and screeners.          |
+-----------------------------------------------------------------------------------------+
| 3. Local Ingestion Time (`first_fetched_at` / `source_fetch_time`)                      |
|    - The wall-clock timestamp when the ARX Terminal engine retrieved the filing XML.    |
|    - Used for ingestion lineage and latency tracking. Never substitutes for public time.|
+-----------------------------------------------------------------------------------------+
| 4. Normalization / Schema Time (`normalized_at`)                                        |
|    - The timestamp when the filing was parsed into tabular models and SQLite storage.   |
|    - Used for internal schema versioning and parser auditing.                           |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Public-Availability Timestamp Authority

### 3.1 Primary Authority: SEC `acceptanceDateTime`
The primary knowledge-time authority is the SEC EDGAR `acceptanceDateTime` field present in the company submissions JSON (`data.sec.gov/submissions/CIK{cik}.json`):
- Standard format: `YYYY-MM-DDTHH:MM:SS.000Z` (e.g., `2026-09-15T16:45:12.000Z`).
- Milliseconds and timezone offsets are normalized to UTC `YYYY-MM-DDTHH:MM:SSZ`.

### 3.2 Authoritative Fallback Rule
If a legacy SEC index or raw XML lacks an `acceptanceDateTime`, the system applies the conservative fallback:
```python
public_availability_at = f"{filing_date}T22:00:00Z"
```
*Rationale*: SEC EDGAR closes daily submission acceptance at 22:00:00 UTC (17:30 EST + 30 min buffer). Assuming availability at the end of the filing day prevents intraday look-ahead bias during historical replay.

---

## 4. Source-Availability Ledger Schema

All filing accessions are registered in the durable SQLite ledger table `sm_source_availability_ledger`:

```sql
CREATE TABLE IF NOT EXISTS sm_source_availability_ledger (
    accession_number TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    acceptance_datetime TEXT,
    public_availability_at TEXT NOT NULL,
    source_content_hash TEXT,
    source_url TEXT,
    first_fetched_at TEXT NOT NULL,
    disposition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_avail_cik_avail
    ON sm_source_availability_ledger(cik, public_availability_at);
CREATE INDEX IF NOT EXISTS idx_source_avail_avail
    ON sm_source_availability_ledger(public_availability_at);
```

---

## 5. Ledger Invariant: Non-Retroactive Availability Rule

### Rule `SOURCE_AVAILABILITY_CAN_BE_REWRITTEN_EARLIER = NO`
Once an accession has a recorded `public_availability_at` in the ledger, no update may shift that timestamp earlier in time:

```python
# Invariant enforcement in Form4IngestionManager:
existing = cur.execute(
    "SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number = ?",
    (accession,)
).fetchone()

if existing:
    existing_avail = existing[0]
    if new_avail < existing_avail:
        raise ValueError(
            f"Invariant Violation: Cannot rewrite public_availability_at to an earlier time "
            f"({new_avail} < {existing_avail}) for accession {accession}. "
            f"Rule: SOURCE_AVAILABILITY_CAN_BE_REWRITTEN_EARLIER = NO"
        )
```

This prevents accidental or fraudulent widening of information availability windows.

---

## 6. Content Integrity & Hash Verification

Every filing XML downloaded is hashed via SHA-256 (`source_content_hash`):
```python
source_content_hash = hashlib.sha256(xml_bytes).hexdigest()
```
The hash is stored in both `form4_raw_filings` and `sm_source_availability_ledger`. When replaying or verifying corpus integrity, any divergence in `source_content_hash` triggers an immediate data corruption alert.
