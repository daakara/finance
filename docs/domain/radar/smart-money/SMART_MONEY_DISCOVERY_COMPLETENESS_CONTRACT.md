# ARX Terminal — Smart Money Discovery Completeness Authority Contract

**Contract ID:** `SM_DISCOVERY_COMPLETENESS_V1`  
**Governing Universe Build ID:** `UB_SM_20261010_R1_19ef857fec1f`  
**Required Partition Count:** `106` (Common Equity Subset)  
**Required Partition Set Hash:** `e8bfa161ef8ef747db32fb192ca45ee7eb373378fdb53a30a69bb8fdb118bc33`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`  
**Production Authorization:** `NOT_AUTHORIZED` (`RADAR_PRODUCTION_STATUS = PIPELINE_PENDING`)

---

## 1. Executive Summary & Epistemic Principle

This contract establishes the rigorous mathematical and procedural standards for establishing SEC Form 4 and Form 4/A discovery completeness across the governed 106-issuer common equity universe.

### The Non-Equivalence Axiom
1. **HTTP 200 Success $\neq$ Discovery Completeness**: A successful API poll merely proves transport reachability, not completeness.
2. **Zero Events Observed $\neq$ Source Closed**: An interval with zero filings does not prove an issuer submitted no filings; it only proves no filings were observed in that response.
3. **Processing All Observed Filings $\neq$ Completeness Certification**: Ingesting $N$ accessions observed from a feed proves processing correctness over $N$, but proves nothing about whether $N$ is the complete universe of filed events.

---

## 2. SEC EDGAR Submission Ingestion Model

Discovery targets the authoritative SEC EDGAR corporate submission feed:
$$\text{Endpoint}: \texttt{https://data.sec.gov/submissions/CIK\{issuer\_cik.zfill(10)\}.json}$$

### Primary and Continuation Structures
- **Primary Array**: `filings.recent` contains up to 1,000 most recent submissions across all form types.
- **Continuation Files**: `filings.files` contains an array of older historical chunk files (`CIK...-submissions-001.json`).

### Pagination Exhaustion Rule
A partition enumeration is certified as **PAGINATION_EXHAUSTED** if and only if:
1. All entries in `filings.recent` within the candidate window $[\tau_{\text{start}}, \tau_{\text{candidate}}]$ have been inspected.
2. If the oldest record in `filings.recent` is newer than $\tau_{\text{start}}$, all necessary `filings.files` chunks are fetched and consumed in reverse chronological order.
3. Zero continuation tokens remain unconsumed (`continuation_token_remaining = false`).
4. Loop detection confirms zero cyclic page references.
5. Incomplete, truncated, or malformed JSON payloads fail closed as `DISCOVERY_GAP` and block certification.

---

## 3. Governed Holdback Stability & Re-enumeration Anti-Entropy

SEC EDGAR accepts filings between 06:00 and 22:00 Eastern Time, but dissemination indexing and public indexing may exhibit minor public dissemination jitter.

### The Holdback Window Standard
- **Holdback Duration**: 24 hours ($\Delta \tau_{\text{holdback}} = 86,400\text{ s}$).
- **Re-enumeration Anti-Entropy**: Any candidate frontier $\tau_{\text{candidate}}$ must remain $\ge \Delta \tau_{\text{holdback}}$ behind wall-clock time $\tau_{\text{eval}}$.
- **Accession Set Stability Invariant**:
  $$\text{Hash}(\text{Accessions}_{\text{first\_enumeration}}) \equiv \text{Hash}(\text{Accessions}_{\text{post\_holdback}})$$
- If an accession appears after the initial poll but before holdback closure, it is recorded, and the accession set hash is updated prior to sealing.

---

## 4. Denominator Lock & Partition Set Invariant

The scannable population denominator is permanently locked:
- **Denominator**: Exactly `106` distinct common-equity issuers in `UB_SM_20261010_R1_19ef857fec1f`.
- **Prohibition on Missing Partition Erasure**: No missing or uncontactable partition may ever be excluded from the denominator to make coverage appear complete.
- **Fail-Closed Missing Partition Rule**: If even 1 partition out of 106 lacks a sealed `PartitionEnumerationReceipt` through $\tau_{\text{candidate}}$, certification fails closed with `MISSING_REQUIRED_PARTITION_ENUMERATION`.

---

## 5. Partition Enumeration Receipt Specification

Every partition poll emits an immutable `PartitionEnumerationReceipt`:

```json
{
  "partition_enumeration_receipt_id": "ENUM-0000320193-4a1b8c...",
  "partition_id": "PART-0000320193",
  "issuer_cik": "0000320193",
  "universe_build_id": "UB_SM_20261010_R1_19ef857fec1f",
  "required_partition_set_hash": "e8bfa161ef8ef747db32fb192ca45ee7eb373378fdb53a30a69bb8fdb118bc33",
  "window_start": "2026-10-09T01:02:33.000Z",
  "window_end": "2026-10-10T01:02:33.000Z",
  "source_contract_version": "1.0.0",
  "discovery_policy_version": "1.0.0",
  "pagination_policy_version": "1.0.0",
  "source_snapshot_manifest_hash": "...",
  "pages_expected_or_discovered": 1,
  "pages_consumed": 1,
  "pagination_exhausted": true,
  "continuation_token_remaining": false,
  "first_enumeration_accession_set_hash": "...",
  "post_holdback_accession_set_hash": "...",
  "accession_set_stable": true,
  "discovery_gap_count": 0,
  "enumeration_status": "CLOSED",
  "enumerated_at": "2026-10-10T01:05:00.000Z",
  "receipt_hash": "..."
}
```

---

## 6. Verification and Audit

Independent verification is executed by `tools/audit_smart_money_wave2c_certification.py`, verifying all 106 partition enumeration receipts against raw database rows, schema integrity, and cryptographic hash chains.
