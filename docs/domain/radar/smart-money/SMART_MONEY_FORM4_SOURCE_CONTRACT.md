# SMART MONEY WAVE 1 — SEC FORM 4 SOURCE CONTRACT

## 1. Executive Summary

This document establishes the authoritative source contract for SEC Form 4 insider transaction data within ARX Terminal. It specifies the regulatory foundation, legal scope, publication latency, data model, identifiers, and deduplication identity required for deterministic, explainable, and truthful processing.

---

## 2. Source Authority & Scope

```text
SOURCE_AUTHORITY =
  SEC EDGAR (U.S. Securities and Exchange Commission Electronic Data Gathering, Analysis, and Retrieval system)

SOURCE_DOCUMENT =
  SEC Form 4 (Statement of Changes in Beneficial Ownership of Securities)

SOURCE_REGULATORY_BASIS =
  Section 16(a) of the Securities Exchange Act of 1934; Rule 16a-3

SOURCE_SCOPE =
  Section 16 corporate insiders: Officers, Directors, and Beneficial Owners of more than 10% of any class of equity securities registered under Section 12.

SOURCE_REFRESH_CHARACTERISTIC =
  Filing-driven, asynchronous event stream governed by statutory 2-business-day reporting deadline following execution date.

SOURCE_IS_REAL_TIME =
  NO

SOURCE_IS_INTRADAY_MARKET_TAPE =
  NO

SOURCE_IS_PROFIT_PREDICTOR =
  NO
```

---

## 3. Regulatory Timing & Latency Profile

1. **Statutory Deadline**: Under Section 16(a) and the Sarbanes-Oxley Act of 2002 (SOX), Form 4 must be filed before the end of the second business day following the day on which the transaction occurred.
2. **Acceptance Window**: SEC EDGAR accepts filings between 06:00 and 22:00 Eastern Time on business days. Filings submitted after 17:30 ET receive an acceptance timestamp on that date but are disseminated to public feeds until 22:00 ET.
3. **Dissemination Lag**: Public dissemination via `data.sec.gov` occurs within seconds to minutes of official SEC acceptance.
4. **Point-In-Time (PIT) Semantics**:
   - `transaction_date`: The economic execution date (`<transactionDate><value>YYYY-MM-DD</value></transactionDate>`).
   - `filing_date`: The official SEC filing date (`<periodOfReport>` / acceptance date).
   - `acceptance_datetime`: Exact SEC intake timestamp (ISO 8601).
   - `available_from`: Earliest timestamp when the filing was legally observable by ARX Terminal.
   - **Semantic Invariant**: ARX Terminal must never backtest or represent a trade as known before `available_from`.

---

## 4. SEC Form 4 Document Structure

Form 4 filings are submitted to EDGAR as SGML/XML submission packages containing the official `<ownershipDocument>` XML payload.

### Key XML Hierarchies

```xml
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>YYYY-MM-DD</periodOfReport>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001234567</rptOwnerCik>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <isOther>0</isOther>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>YYYY-MM-DD</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
```

---

## 5. Authoritative Identity & Deduplication

### Record Identity

A single Form 4 filing may contain multiple reporting owners (joint filers) and multiple transaction lines across non-derivative and derivative tables.

To prevent duplicate counts, double-crediting, or corruption across incremental ingestion runs, every parsed transaction has an immutable composite primary key:

```text
FORM4_RECORD_KEY =
  {accession_number} : {reporting_owner_cik} : {table_type} : {transaction_sequence}
```

Where:
- `accession_number`: 20-character EDGAR unique identifier (e.g., `0000320193-26-000102`).
- `reporting_owner_cik`: 10-digit zero-padded CIK of the insider.
- `table_type`: `NON_DERIVATIVE` or `DERIVATIVE`.
- `transaction_sequence`: 0-indexed integer of the transaction line within that table in the XML document.

### Amended Filings (Form 4/A)
- When `documentType == "4/A"`, the filing supersedes prior filings.
- The parser records `is_amendment = True` and links `original_accession_number` if referenced in XML header or footnotes.
- When an amendment replaces an existing accession, previous transactions from the superseded accession are marked `SUPERSEDED` and excluded from current active scanner snapshots.

---

## 6. Access and Ingestion Transport Rules

1. **Host Authority**:
   - `data.sec.gov/submissions/CIK{cik}.json`: Submissions list and metadata.
   - `www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}.txt`: Raw submission file.
   - `www.sec.gov/files/company_tickers.json`: Master ticker-to-CIK mapping index.
2. **Rate Limit Policy**:
   - SEC EDGAR imposes a strict maximum of 10 requests per second across all user IPs.
   - ARX Terminal enforces a client-side token bucket with a ceiling of **5 requests per second** (50% safety buffer).
3. **User-Agent Policy**:
   - SEC mandates User-Agent header format: `User-Agent: Sample Company Name AdminContact@<sample company domain>.com`.
   - ARX Terminal uses: `FinanceTerminalResearchApp admin@financeterminal.internal`.
4. **Caching & Local Storage**:
   - Submissions JSON cached locally with TTL of 1 hour during market hours, 24 hours on weekends.
   - Parsed Form 4 XML transactions persisted immutably in local cache database (`form4_transactions`).
