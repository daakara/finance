# SMART MONEY WAVE 2A — TICKER ↔ CIK MAPPING CONTRACT

## 1. Executive Summary

This contract defines the deterministic, fail-closed mapping between market ticker symbols and SEC EDGAR Central Index Keys (CIKs).

```text
CIK_MAPPING_POLICY_VERSION = "cik-mapping-v1.0"
AMBIGUOUS_MAPPING_ENTERS_SCANNABLE_POPULATION = NO
MISSING_CIK_ENTERS_SCANNABLE_POPULATION = NO
NAME_FUZZY_MATCHING_PERMITTED = NO
```

---

## 2. Mapping Entity Schema

Every mapped security record contains:

| Field | Type | Description |
|:---|:---|:---|
| `security_id` | `str` | Canonical security identifier (e.g. `US_EQ_NASDAQ_NVDA_0001045810`) |
| `ticker` | `str` | Uppercase, normalized ticker symbol |
| `exchange` | `str` | Primary listing exchange (e.g. `NASDAQ`, `NYSE`) |
| `issuer_name` | `str` | Official SEC registrant entity name |
| `issuer_cik` | `str` | 10-digit zero-padded SEC Central Index Key |
| `mapping_source` | `str` | Authoritative source (`SEC_COMPANY_TICKERS_EXCHANGE` or `ARX_VERIFIED_CIK_OVERRIDE`) |
| `mapping_as_of` | `str` | Point-in-time timestamp (ISO 8601 UTC) |
| `mapping_status` | `CikMappingStatus` | Governed mapping lifecycle state |
| `mapping_content_hash` | `str` | Deterministic SHA-256 hash of mapped attributes |

---

## 3. Governed Mapping States

| State | Description | Scannable Action |
|:---|:---|:---|
| `VERIFIED` | Ticker has exactly one unambiguous 10-digit CIK in authoritative SEC directory. | **Admitted to Scannable Population** |
| `MISSING_CIK` | Instrument exists in security master but CIK is unresolvable or null. | **Excluded / Fail-Closed** |
| `AMBIGUOUS` | Ticker matches multiple active CIKs or CIK matches multiple active tickers. | **Excluded / Quarantine** |
| `CONFLICT` | Provider CIK differs from SEC EDGAR official directory. | **Excluded / Quarantine** |
| `INACTIVE` | CIK corresponds to a corporate entity with deregistered reporting status. | **Excluded** |
| `STALE` | Mapping timestamp exceeds maximum allowable drift window (> 30 days). | **Excluded until refreshed** |

---

## 4. Conflict Resolution & Fail-Closed Rules

1. **One-to-Many Collision (Ticker -> Multiple CIKs)**:
   - If an input ticker resolves to multiple distinct CIKs in the upstream source, the mapping state transitions immediately to `AMBIGUOUS`.
   - **Zero Inference**: The system does not attempt fuzzy name similarity or heuristics to pick a "best" CIK. It fails closed and quarantines the symbol.
2. **Missing CIK**:
   - If an eligible common equity has no verified CIK in the SEC directory, it enters `ELIGIBLE_POPULATION` but CANNOT enter `SCANNABLE_POPULATION`.
   - It is counted in `eligible_count` and reported in `unresolved_missing_cik_count`.
3. **Leading Zero Formatting**:
   - SEC CIKs are strictly formatted as 10-digit zero-padded numeric strings (e.g. `320193` -> `"0000320193"`).
