# ARX TERMINAL — SPRINT 2 / TRACK C
# TRACK B TO TRACK C DATA-QUALITY CERTIFICATION INTERFACE CONTRACT

**Document Identifier:** `C_PACKAGE_B_INTERFACE_CONTRACT.md`  
**Governing Authority:** ARX Closure Model v2 (Track C Ownership Boundary)  
**Contract Type:** Inbound Data-Quality Interface Specification  
**Producer (Upstream):** Track B (Market Data Ingestion & Quality Certification)  
**Consumer (Downstream):** Track C (Universe Publication & Attestation)  
**Status:** **PROPOSED & UNRECONCILED (AWAITING PACKAGE B CERTIFICATION)**  
**Target Milestone:** Sprint 2 Coverage Expansion Integration  

---

## 1. Context & Purpose

Track C governs universe publication, eligibility evaluation, and deterministic attestation. To preserve platform invariants and prevent unseasoned, corrupted, or stale market data from entering published universes, Track C **refuses to ingest raw provider market data directly**.

Instead, Track C establishes an explicit, cryptographic boundary:
- Track B owns provider acquisition, rate-limiting, session parsing, split adjustments, and data cleaning.
- Track C requires Track B to deliver an immutable, signed **Data-Quality Certification** for each candidate instrument.
- Without a valid, verified Data-Quality Certification, an instrument **cannot transition from State 2 (Provider-Resolvable) to State 3 (Data-Qualified)**.

> [!WARNING]
> **Current Operational Status: BLOCKED / DEFICIENT CREDENTIALS**  
> As formally attested in `package_b_market_data_ingestion.json` (`PKG-EXPANSION-150-B-MARKET-DATA-INGESTION`), Package B has **NOT** been certified. Live bounded probe requests revealed:
> 1. Polygon API key returned `HTTP 401 Unauthorized` (`Unknown API Key`).
> 2. Alpaca SIP feed returned `HTTP 403 Forbidden` (`Free tier lacks SIP consolidated entitlement`).
> 3. Alpaca IEX feed provides participant-only volume, violating institutional execution standards.
> 
> **Track C must treat this interface as PROPOSED. Zero completed Package B certificates exist in production.**

---

## 2. Inbound Certification Schema Specification

Each candidate security certification delivered by Track B must conform to the following schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "TrackBDataQualityCertification",
  "type": "object",
  "required": [
    "certification_id",
    "internal_security_id",
    "listing_association_id",
    "provider_instrument_mapping",
    "qualified_historical_coverage",
    "quality_status",
    "data_as_of",
    "corporate_action_consistency",
    "source_lineage",
    "certification_digest"
  ],
  "properties": {
    "certification_id": {
      "type": "string",
      "pattern": "^CERT_B_[A-F0-9]{12}$",
      "description": "Unique deterministic certification identifier"
    },
    "internal_security_id": {
      "type": "string",
      "pattern": "^ARX_SEC_[A-F0-9]{12}$",
      "description": "Durable internal security identifier matching identity.db"
    },
    "listing_association_id": {
      "type": "string",
      "pattern": "^ASSOC_[A-F0-9]{12}$",
      "description": "Durable listing association identifier matching identity.db"
    },
    "provider_instrument_mapping": {
      "type": "object",
      "required": ["provider_name", "provider_symbol", "provider_exchange", "provider_asset_id"],
      "properties": {
        "provider_name": { "type": "string", "enum": ["ALPACA_SIP", "POLYGON_CONSOLIDATED", "SEC_EDGAR_XBRL"] },
        "provider_symbol": { "type": "string" },
        "provider_exchange": { "type": "string" },
        "provider_asset_id": { "type": "string" }
      }
    },
    "qualified_historical_coverage": {
      "type": "object",
      "required": [
        "session_count",
        "earliest_session_date",
        "latest_session_date",
        "expected_calendar_sessions",
        "missing_sessions_count",
        "has_interior_prefix_gaps"
      ],
      "properties": {
        "session_count": { "type": "integer", "minimum": 50 },
        "earliest_session_date": { "type": "string", "format": "date" },
        "latest_session_date": { "type": "string", "format": "date" },
        "expected_calendar_sessions": { "type": "integer" },
        "missing_sessions_count": { "type": "integer", "maximum": 0 },
        "has_interior_prefix_gaps": { "type": "boolean", "const": false }
      }
    },
    "quality_status": {
      "type": "string",
      "enum": ["CERTIFIED_VALID", "DEFICIENT_STALE", "DEFICIENT_GAPS", "DEFICIENT_PARTICIPANT_ONLY"]
    },
    "data_as_of": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 UTC timestamp of latest bar included in certification"
    },
    "corporate_action_consistency": {
      "type": "object",
      "required": ["split_adjusted", "unhandled_corporate_actions", "distribution_reconciled"],
      "properties": {
        "split_adjusted": { "type": "boolean", "const": true },
        "unhandled_corporate_actions": { "type": "boolean", "const": false },
        "distribution_reconciled": { "type": "boolean", "const": true }
      }
    },
    "source_lineage": {
      "type": "object",
      "required": ["ingestion_batch_id", "feed_type", "ingested_at_utc", "storage_target"],
      "properties": {
        "ingestion_batch_id": { "type": "string" },
        "feed_type": { "type": "string", "enum": ["SIP_CONSOLIDATED", "HISTORICAL_REST_V2"] },
        "ingested_at_utc": { "type": "string", "format": "date-time" },
        "storage_target": { "type": "string" }
      }
    },
    "certification_digest": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$",
      "description": "Deterministic SHA-256 hex digest of the canonical JSON certification payload excluding digest field"
    }
  }
}
```

---

## 3. Python Dataclass Contract Definition

Track C will use the following dataclass contract (`analyst_dashboard/universe/contracts.py`) to parse and validate incoming certifications:

```python
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional
import hashlib
import json

class QualityCertificationStatus(str, Enum):
    CERTIFIED_VALID = "CERTIFIED_VALID"
    DEFICIENT_STALE = "DEFICIENT_STALE"
    DEFICIENT_GAPS = "DEFICIENT_GAPS"
    DEFICIENT_PARTICIPANT_ONLY = "DEFICIENT_PARTICIPANT_ONLY"

@dataclass(frozen=True)
class ProviderInstrumentMapping:
    provider_name: str
    provider_symbol: str
    provider_exchange: str
    provider_asset_id: str

@dataclass(frozen=True)
class QualifiedHistoricalCoverage:
    session_count: int
    earliest_session_date: str
    latest_session_date: str
    expected_calendar_sessions: int
    missing_sessions_count: int
    has_interior_prefix_gaps: bool

@dataclass(frozen=True)
class CorporateActionConsistency:
    split_adjusted: bool
    unhandled_corporate_actions: bool
    distribution_reconciled: bool

@dataclass(frozen=True)
class SourceLineage:
    ingestion_batch_id: str
    feed_type: str
    ingested_at_utc: str
    storage_target: str

@dataclass(frozen=True)
class TrackBDataQualityCertification:
    certification_id: str
    internal_security_id: str
    listing_association_id: str
    provider_instrument_mapping: ProviderInstrumentMapping
    qualified_historical_coverage: QualifiedHistoricalCoverage
    quality_status: QualityCertificationStatus
    data_as_of: str
    corporate_action_consistency: CorporateActionConsistency
    source_lineage: SourceLineage
    certification_digest: str

    def verify_digest(self) -> bool:
        """Verifies canonical content digest matches declared certification_digest."""
        payload = asdict(self)
        declared_digest = payload.pop("certification_digest")
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        computed_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return computed_digest == declared_digest

    def is_publication_ready(self) -> bool:
        """Evaluates whether certificate meets strict Track C publication acceptance standards."""
        return (
            self.verify_digest()
            and self.quality_status == QualityCertificationStatus.CERTIFIED_VALID
            and self.qualified_historical_coverage.session_count >= 50
            and self.qualified_historical_coverage.missing_sessions_count == 0
            and not self.qualified_historical_coverage.has_interior_prefix_gaps
            and self.corporate_action_consistency.split_adjusted
            and not self.corporate_action_consistency.unhandled_corporate_actions
        )
```

---

## 4. Track C Verification Protocol & Fail-Closed Rules

When Track C's `DeterministicUniverseBuilder` evaluates data readiness for an instrument during a universe build:

1. **Step 1: Identity Cross-Check**
   - Assert `internal_security_id` exists in `internal_securities` (`identity.db`).
   - Assert `listing_association_id` exists in `listing_associations` (`identity.db`).
   - If missing, fail immediately: `REJECTED_UNKNOWN_IDENTITY`.

2. **Step 2: Certificate Digest Verification**
   - Recompute canonical SHA-256 digest over the certificate payload.
   - If mismatch, fail immediately: `REJECTED_TAMPERED_CERTIFICATE`.

3. **Step 3: Quality Status Verification**
   - Verify `quality_status == "CERTIFIED_VALID"`.
   - If `DEFICIENT_*`, set `DataReadinessLedgerRow`:
     - `readiness_result = DATA_UNAVAILABLE`
     - `readiness_reason_code = INSUFFICIENT_HISTORY` or `DEFICIENT_PROVIDER_FEED`.

4. **Step 4: History Depth & Continuity Verification**
   - Assert `session_count >= 50`.
   - Assert `missing_sessions_count == 0`.
   - Assert `has_interior_prefix_gaps == False`.
   - If any condition fails, the candidate remains `DATA_UNAVAILABLE` and cannot be included in `scannable_membership`.

5. **Step 5: Corporate Action Attestation**
   - Assert `split_adjusted == True` and `unhandled_corporate_actions == False`.
   - Prevent price chart and VCP contraction distortion from unadjusted splits.

---

## 5. Summary of Package B Blockers

Until the following three operational dependencies are resolved by Track B, this interface remains proposed and unexercised in production:

| Blocker ID | Upstream Component | Defect Description | Required Resolution |
| :---: | :--- | :--- | :--- |
| **B-BLK-01** | Polygon.io API Gateway | HTTP 401 Unauthorized (`Unknown API Key`) | Provision active institutional Polygon key |
| **B-BLK-02** | Alpaca Markets Gateway | HTTP 403 Forbidden on `feed=sip` | Upgrade Alpaca subscription to SIP consolidated feed |
| **B-BLK-03** | Historical Ingestion Run | 61 of 133 launch candidates missing in production `market.db` | Execute bounded historical SIP ingestion with calendar audit |
