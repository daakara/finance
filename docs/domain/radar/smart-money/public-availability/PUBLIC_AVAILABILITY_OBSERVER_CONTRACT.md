# ARX Terminal — Public Availability Observer Contract (Epoch 001)

**Document Identifier:** `PUBLIC_AVAILABILITY_OBSERVER_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent Public Availability Evidence Engineer & Point-in-Time Integrity Auditor  
**Operating Mode:** `SOURCE_FIRST` + `PROSPECTIVE_FIRST` + `FAIL_CLOSED` + `APPEND_ONLY` + `CLOCK_AWARE` + `TRANSPORT_AWARE` + `NO_PRODUCTION`

---

## 1. Purpose and Foundational Distinctions

This contract defines the prospective multi-channel observation architecture for SEC filings in ARX Terminal Smart Money.

It enforces strict separation between:
$$\text{SOURCE POPULATION AUTHORITY} \neq \text{PUBLIC AVAILABILITY TIME AUTHORITY} \neq \text{DECISION INPUT AVAILABILITY} \neq \text{EMPIRICAL SIGNAL QUALITY}$$

### Hard Invariants
1. **`SEC_ACCEPTANCE_AT_IS_FIRST_PUBLIC = NO`**: SEC EDGAR `<acceptanceDateTime>` represents internal SEC system receipt and validation, which precedes public web availability.
2. **`PDS_DELIVERED_AT_IS_FIRST_PUBLIC = NO`**: PDS dissemination occurs over dedicated contractor lines and does not establish prior or exact public website publication.
3. **`FIRST_PROVEN_PUBLIC_AT_IS_TRUE_FIRST_PUBLIC = NO`**: A local observation establishes an empirical upper bound on publication time ($\tau_{\text{public}} \le \tau_{\text{obs}}$), not the unobservable source-native instant of creation.
4. **`LATER_INFORMATION_BACKDATED = NO`**: Observations at $T_2$ cannot backdate content completeness to an earlier observation at $T_1$.
5. **`REQUEST_START_USED_AS_KNOWLEDGE_TIME = NO`**: Conservative observation time is strictly computed from response completion time plus maximum clock uncertainty.

---

## 2. Observer Identity & Run Governance

Every observation execution binds:
- `OBSERVER_RUN_ID`: Unique run identifier (`run_...`).
- `OBSERVER_POLICY_ID`: `ARX_SEC_OBSERVER_ACCESS_POLICY_V1`.
- `OBSERVER_POLICY_HASH`: Canonical hash of access policy rules.
- `OBSERVER_IMPLEMENTATION_GIT_SHA`: Current git commit SHA.
- `CLOCK_SOURCE` & `CLOCK_STATUS`: Certified clock status (`SYNCHRONIZED`, `DEGRADED`, `UNTRUSTED`).
- `STARTED_AT` & `COMPLETED_AT`: ISO8601 UTC execution timestamps.

---

## 3. Observation Lifecycle & Append-Only Ledger

Observations are recorded in five append-only relational models:
1. `sec_public_observer_source_policies`: Bound SEC Fair Access and ARX rate limit rules.
2. `sec_public_observer_runs`: Lifecycle and clock anchor for each observer execution.
3. `sec_public_observations`: Immutable raw response records, HTTP status, content hash, and transport metrics.
4. `sec_public_content_versions`: Version history tracking for identical endpoints returning different bytes.
5. `sec_public_knowledge_derivations`: Deterministic derivation of earliest proven metadata, content, and decision times.

Historical observation records are strictly immutable. No `UPDATE` of observation meaning is permitted.

---

## 4. Fair Access Policy Adherence

- **Official SEC Source**: `https://www.sec.gov/os/accessing-edgar-data`
- **Official Policy SHA-256**: `393dccc830a92f314e5d4aded255c43350dc4282fd73ad45ed5874c553e0ce05`
- **Official Ceiling**: 10 requests per second.
- **ARX Governed Limit**: 5 requests per second (token-bucket rate limited).
- **User-Agent**: `ArxTerminal/1.0 (akara@finance.app)`.
- **Indeterminate Backoff**: HTTP 403, 429, 5xx, and timeouts trigger backoff and are classified as `TRANSPORT_INDETERMINATE`. They are never interpreted as filing absence.
