# SMART MONEY WAVE 1 — API V2 SPECIFICATION & DESIGN

## 1. Overview

This document specifies the target Smart Money API V2 contract. To preserve the Candidate 004 frozen release firewall and prevent mutations to shared API routing files, API V2 endpoints are designed modularly and isolated from existing frozen endpoints (`/api/screener/*`, `/api/smart-money/*`).

---

## 2. Target Endpoints

### 1. `GET /api/smart-money/v2/snapshot`
Returns the active market-wide / bounded scanner snapshot with complete provenance and readiness envelope.

**Response Schema:**
```json
{
  "scanner_id": "SMART_MONEY_FORM4",
  "version": "1.0.0",
  "status": "PIPELINE_PENDING",
  "validation_scope": "BOUNDED_ENGINEERING_SAMPLE",
  "generated_at": "2026-10-10T14:00:00Z",
  "universe": {
    "universe_id": "BOUNDED_SAMPLE",
    "universe_count": 10,
    "eligible_count": 10,
    "scannable_count": 10,
    "authority_status": "NOT_ESTABLISHED"
  },
  "coverage": {
    "coverage_status": "PARTIAL",
    "ingested_issuers": 10,
    "filings_parsed": 45,
    "transactions_parsed": 128
  },
  "freshness": {
    "policy_version": "1.0.0",
    "last_successful_ingestion": "2026-10-10T14:00:00Z",
    "max_filing_age_days": 30,
    "status": "FRESH"
  },
  "snapshot": {
    "snapshot_id": "snap-sm-form4-20261010-001",
    "matched_count": null,
    "candidates": []
  }
}
```

### 2. `GET /api/smart-money/v2/readiness`
Returns the diagnostic readiness checklist proving why the scanner is in its current state.

**Response Schema:**
```json
{
  "status": "PIPELINE_PENDING",
  "checks": {
    "source_ingestion_operational": true,
    "market_wide_universe_governed": false,
    "universe_denominator_verified": false,
    "freshness_policy_satisfied": true,
    "full_scan_completed": true,
    "snapshot_persisted": true
  },
  "blocker_reason": "Market-wide universe authority (ARX_US_EQUITIES 5,000+ symbols) unprovisioned in UniverseStore. Bounded engineering sample verified locally."
}
```

### 3. `GET /api/smart-money/v2/issuer/{symbol}`
Returns detailed Form 4 transaction breakdown and clustering history for a specific ticker symbol.

**Response Schema:**
```json
{
  "symbol": "AAPL",
  "issuer_name": "Apple Inc.",
  "issuer_cik": "0000320193",
  "filing_count_30d": 3,
  "recent_transactions": [
    {
      "accession_number": "0000320193-26-000045",
      "filing_date": "2026-08-20",
      "transaction_date": "2026-08-18",
      "reporting_owner": "Cook Timothy D",
      "role": "Chief Executive Officer",
      "transaction_code": "M",
      "signal_class": "OPTION_EXERCISE",
      "shares": 50000,
      "price": 0.0,
      "total_value_usd": 0.0,
      "conviction_class": "NON_DIRECTIONAL"
    }
  ],
  "cluster_summary_30d": {
    "open_market_purchases_count": 0,
    "open_market_sales_count": 1,
    "distinct_buyers": 0,
    "cluster_buy": false,
    "total_purchase_value_usd": 0.0
  }
}
```
