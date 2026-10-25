# ARX TERMINAL — SPRINT 2 / TRACK C
# DETERMINISTIC PUBLICATION SAFETY CONTRACT & ROLLBACK SPECIFICATION

**Document Identifier:** `C_PUBLICATION_SAFETY_CONTRACT.md`  
**Governing Authority:** ARX Closure Model v2 (Track C Ownership Boundary)  
**Contract Type:** Publication Governance, Cryptographic Attestation & Rollback Architecture  
**Target Milestone:** Sprint 2 / Package C Architecture & Readiness Review  
**Status:** COMPLETE / CANONICAL SPECIFICATION  

---

## 1. Executive Summary & Safety Principles

The ARX Terminal universe publication system is designed around the principles of **mathematical determinism, cryptographic immutability, atomic execution, and instant reversibility**.

Under this contract:
1. **Zero Silent Mutation:** A published universe snapshot cannot be updated, edited, patched, or partially modified in place.
2. **Zero Partial Publication:** A universe build succeeds completely across all candidate securities or fails completely with zero database mutation.
3. **Cryptographic Identity Attestation:** Every build is sealed with a deterministic Merkle-tree attestation digest linking source records, normalization rules, eligibility evaluations, and data readiness states.
4. **Instant Fast-Pointer Rollback:** Because all historical builds are retained immutably in append-only storage, rolling back to any previous verified universe is an instantaneous $O(1)$ pointer update.

---

## 2. Versioned Publication Snapshots & Namespaces

### 2.1 Canonical Version Identifiers
All publication artifacts must be stamped with immutable semantic version tuples:

| Identifier Key | Canonical Value | Meaning / Authority |
| :--- | :--- | :--- |
| `UNIVERSE_VERSION` | `arx-universe-v1.0` | Structural schema and metadata version |
| `ELIGIBILITY_RULE_VERSION` | `eligibility-rule-v1.0` | Canonical Rule R01-R07 definitions |
| `NORMALIZATION_VERSION` | `norm-v1.0` | Casing, whitespace, and MIC normalization specification |
| `DATA_READINESS_POLICY_VERSION` | `readiness-v1.0` | Minimum 50-session OHLCV & live quote requirement |
| `NUMERIC_PRECISION` | `DECIMAL_FIXED_SCALE` | Elimination of floating-point representation drift |

### 2.2 Namespace Isolation Architecture
To prevent unverified securities from leaking into production, the platform defines three distinct universe namespaces:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          UNIVERSE NAMESPACE TAXONOMY                            │
├──────────────────────────────┬───────────────────┬──────────────────────────────┤
│ Namespace Identifier         │ Status            │ Population Scope             │
├──────────────────────────────┼───────────────────┼──────────────────────────────┤
│ ARX_US_EQUITIES              │ ACTIVE PRODUCTION │ 3 Baseline Pilots (AAPL/GOOG)│
│ ARX_US_EQUITIES_PILOT_3C     │ ARCHIVED PILOT    │ Historical 3-pilot sandbox   │
│ ARX_US_EQUITIES_EXPANSION_150│ PROPOSED ISOLATED │ 133 Radar Launch Candidates  │
└──────────────────────────────┴───────────────────┴──────────────────────────────┘
```

> [!IMPORTANT]
> The active scanner runner (`VCPScannerRunner`) queries `UNIVERSE_ID = "ARX_US_EQUITIES"`.  
> Publishing to `ARX_US_EQUITIES_EXPANSION_150` operates within a strictly isolated namespace. It **does not alter the active scanner universe** until an independent, user-approved default universe pointer cutover occurs.

---

## 3. Cryptographic Attestation & Content Hashing

Every universe build executes deterministic Merkle-tree hashing to guarantee bit-for-bit reproducibility regardless of execution platform, worker parallelism, or input sorting:

```
                            Universe Attestation Root
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
Source Population Hash       Eligibility Decision Hash      Readiness Decision Hash
(Sorted Source Securities)     (Sorted Decision Hashes)     (Sorted Readiness Hashes)
         │                             │                             │
         ▼                             ▼                             ▼
Normalization Spec Hash       Rule Evaluation Hash           Market Store Content Hash
  (Trim / Uppercase)         (R01-R07 Boolean Results)      (50 Bars & Live Price Check)
```

### 3.1 Hashing Specifications
All hashes use SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`, `default=str`):

1. **`source_population_hash`:**
   $$\text{Hash}\left(\left[\text{SourceSecurity}_1.\text{to\_dict}(), \dots, \text{SourceSecurity}_N.\text{to\_dict}()\right]\right)$$
   *Constraint:* Source securities are strictly sorted by `security_id` prior to serialization.
2. **`eligible_membership_hash`:**
   $$\text{Hash}\left(\text{sorted}\left(\left[\text{symbol} \mid \text{decision} == \text{ELIGIBLE}\right]\right)\right)$$
3. **`scannable_membership_hash`:**
   $$\text{Hash}\left(\text{sorted}\left(\left[\text{symbol} \mid \text{readiness} == \text{DATA\_READY}\right]\right)\right)$$
4. **`per_security_decision_hash`:**
   $$\text{Hash}\left(\left[\text{row}.\text{decision\_hash} \text{ for row in sorted by symbol}\right]\right)$$
5. **`readiness_decision_hash`:**
   $$\text{Hash}\left(\left[\text{row}.\text{readiness\_hash} \text{ for row in sorted by symbol}\right]\right)$$

---

## 4. Diff Previews & Membership Transition Accounting

Before any universe build can be committed to the persistent store, the builder must compute an explicit diff preview against the previous active build for that universe namespace:

$$\Delta = \text{Current Build} \setminus \text{Previous Build}$$

### 4.1 Transition Categories
Every candidate in the source population is classified into exactly one transition state:
- `ADDED`: Present and eligible/scannable in current build; absent in previous build.
- `REMOVED`: Present in previous build; absent or ineligible/unavailable in current build.
- `UNCHANGED`: Retained with identical eligibility and scannability status.

### 4.2 Automated Quarantine Triggers (Drift Circuit Breakers)
The build is immediately aborted and marked `publication_decision = "QUARANTINE"` if:
1. **Unexplained Removals:** Any baseline pilot security (`AAPL`, `GOOG`, `GOOGL`) is marked `REMOVED`.
2. **Excessive Churn:** Total turnover exceeds 15% of universe size without an operator-acknowledged corporate action or index rebalancing manifest.
3. **Accounting Mismatch:** The arithmetic closure formula fails:
   $$\text{source\_count} \ne \text{eligible\_count} + \text{ineligible\_count} + \text{unresolved\_count}$$
   $$\text{eligible\_count} \ne \text{data\_ready\_count} + \text{data\_unavailable\_count} + \text{data\_unresolved\_count}$$

---

## 5. Atomicity, Idempotency & Immutability Guarantees

### 5.1 Atomic Multi-Table Commit
A universe build involves writes across three database tables:
1. `universe_builds` (1 attestation row)
2. `universe_eligibility_ledger` ($N$ per-security eligibility rows)
3. `universe_readiness_ledger` ($E$ per-eligible-security readiness rows)

These operations are executed inside a **single native SQLite transaction** (`BEGIN IMMEDIATE`):
- If any insert fails, disk full occurs, or constraint is violated, SQLite executes a 100% rollback.
- **Zero Partial Publication:** No scenario exists where a build record is committed while ledger rows are missing.

### 5.2 Schema-Level Immutability Triggers
To protect historical records from tampering or accidental deletion, `store.py` installs database triggers on initialization:

```sql
CREATE TRIGGER trg_ubuild_no_update BEFORE UPDATE ON universe_builds
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be updated.');
END;

CREATE TRIGGER trg_ubuild_no_delete BEFORE DELETE ON universe_builds
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be deleted.');
END;

CREATE TRIGGER trg_eligibility_no_update BEFORE UPDATE ON universe_eligibility_ledger
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be updated.');
END;

CREATE TRIGGER trg_eligibility_no_delete BEFORE DELETE ON universe_eligibility_ledger
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be deleted.');
END;
```

### 5.3 Idempotency Guarantee
If the builder is executed multiple times with identical `SourcePopulationSnapshot` data and identical market data state:
- The generated `universe_build_id` or content hashes will match.
- Applying an identical build produces a verified `NO_OP_ALREADY_COMMITTED` without duplicating rows or raising primary key conflicts.

---

## 6. Rollback Architecture & Fast-Pointer Restoration

Because the universe store is **append-only**, previous builds are never overwritten or deleted. This enables instantaneous, zero-data-loss rollback:

### 6.1 Fast-Pointer Reversion
The active published universe is resolved by downstream consumers using:
```sql
SELECT * FROM universe_builds
WHERE universe_id = ? AND publication_decision = 'PUBLISH' AND construction_status = 'COMPLETE'
ORDER BY rowid DESC LIMIT 1;
```

To roll back from a defective build `B_new` to a known good build `B_prior`:
1. **Method A (Quarantine Invalidation):** Execute an attested administrative transaction writing a revocation record in an append-only `universe_publication_revocations` ledger.
2. **Method B (Re-Affirmation Commit):** Commit a new attestation row pointing back to the Merkle hashes and membership of `B_prior` with timestamp $t_{\text{rollback}}$, generating an auditable event without mutating historical rows.

### 6.2 Rollback Speed & Recovery Time Objective (RTO)
- **RTO:** $< 1.0\text{ second}$.
- **Recovery Point Objective (RPO):** Zero data loss; all historical ledgers remain 100% intact for post-mortem analysis.

---

## 7. Downstream Consumer Version Compatibility

To protect downstream consumers (Radar, Screener API, Schedulers) from silent schema drift:
1. `VCPScannerRunner` verifies that the `universe_version` stored in the active build matches its internal `VCP_UNIVERSE_VERSION` (`arx-universe-v1.0`).
2. If a version mismatch is detected (e.g., an experimental `arx-universe-v2.0` build was published), the scanner fails closed:
   - Emits log: `UNSUPPORTED_UNIVERSE_VERSION_DETECTED`.
   - Returns status: `PIPELINE_PENDING`.
   - Refuses to serve corrupted or misaligned candidate matches.
