# ARX Terminal — Smart Money
## Finding Admission Model & Universe Governance V1

### 1. Purpose & Separation of Concerns

The Finding Admission Model establishes strict separation between finding identity, assertion, admission, lifecycle, and classification across the ARX Terminal Smart Money governance control plane:

1. **Identity**: Preserved immutable mapping (`finding_id`, `original_subject`, `original_subject_hash`). Finding identifiers may never be repurposed or silently altered.
2. **Assertion Status**: Tracks whether a finding has been submitted/asserted (`ASSERTED`). Assertion does not grant automatic blocking authority.
3. **Admission Status**: Formal governance adjudication determines whether an asserted finding enters the authoritative decision universe:
   - `ADMITTED`: Accepted into the decision universe as an active blocker or prerequisite.
   - `DUPLICATE`: Identical in root defect to an existing finding; bound via `duplicate_of` to prevent double-counting.
   - `SUPERSEDED`: Replaced by a more comprehensive finding; bound via `superseded_by`.
   - `REJECTED`: Formal review determined the assertion is unfounded or non-applicable.
   - `ASSERTED_PENDING_REVIEW`: Awaiting formal council adjudication.
4. **Lifecycle Status**: Operational status (`OPEN`, `ACTIVE`, `IN_REMEDIATION`, `REMEDIATED`, `VERIFIED_RESOLVED`, `WAIVED`, `DISPROVED`, `CONTESTED`, `STALE`).
5. **Defect Classification**: 7-layer normative taxonomy (`ARCHITECTURE_DEFECT`, `POLICY_DEFECT`, `PROVIDER_SELECTION_PREREQUISITE`, `IMPLEMENTATION_PREREQUISITE`, `EXTERNAL_EVIDENCE_GAP`, `QUALIFICATION_PREREQUISITE`, `DOCUMENTATION_DEFECT`).

---

### 2. Normative Invariant: Admitted Blocker Rule

A finding may enter blocker computation and affect governance decisions if and only if:
$$\text{admission\_status} == \text{ADMITTED} \quad \land \quad \text{lifecycle\_status} \in \{\text{OPEN}, \text{ACTIVE}, \text{IN\_REMEDIATION}\} \quad \land \quad \text{severity} \in \{\text{CRITICAL}, \text{HIGH}, \text{BLOCKING}\}$$

Duplicate, superseded, rejected, or pending findings are strictly excluded from blocker computation:
- `DUPLICATE_COUNT_IN_BLOCKERS`: 0
- `REJECTED_COUNT_IN_BLOCKERS`: 0
- `SUPERSEDED_COUNT_IN_BLOCKERS`: 0

---

### 3. Universe Adjudication: FND-ARCH-001 through FND-ARCH-015

| Finding ID | Title / Subject | Admission Status | Duplicate Of | Lifecycle | Layer | Blocker Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `FND-ARCH-001` | Semantic hash identity collision | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorems A, B, Provisioning |
| `FND-ARCH-002` | Genesis release provenance | **ADMITTED** | — | `OPEN` | `EXTERNAL_EVIDENCE_GAP` | Blocks Theorem A, Provisioning |
| `FND-ARCH-003` | Timestamp authority provenance | **ADMITTED** | — | `OPEN` | `PROVIDER_SELECTION_PREREQUISITE` | Blocks Theorems B, D, Provisioning |
| `FND-ARCH-004` | RPO vs asynchronous replication | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorem C, Provisioning |
| `FND-ARCH-005` | Provider-specific custody semantics | **ADMITTED** | — | `OPEN` | `PROVIDER_SELECTION_PREREQUISITE` | Blocks Theorem D, Provisioning |
| `FND-ARCH-006` | OIDC claim completeness | **ADMITTED** | — | `OPEN` | `IMPLEMENTATION_PREREQUISITE` | Blocks Theorem B, Provisioning |
| `FND-ARCH-007` | Claimed-but-uncommitted Gate3 recovery | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorems B, C, Provisioning |
| `FND-ARCH-008` | Claim contention vs quarantine semantics | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorem C, Provisioning |
| `FND-ARCH-009` | Transient evidence fetch classification | **ADMITTED** | — | `OPEN` | `POLICY_DEFECT` | Blocks Theorem D, Provisioning |
| `FND-ARCH-010` | Stale fencing-token handling | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorem C, Provisioning |
| `FND-ARCH-011` | Denominator predicate alignment | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Pre-Arm, Provisioning |
| `FND-ARCH-012` | Git commit SHA truncation in Genesis Manifest | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorem A, Provisioning |
| `FND-ARCH-013` | Gate-3 lease-expiry recovery behavior | **DUPLICATE** | `FND-ARCH-007` | `OPEN` | `ARCHITECTURE_DEFECT` | Duplicate (non-blocking) |
| `FND-ARCH-014` | Transaction-log divergence from consensus | **ADMITTED** | — | `OPEN` | `ARCHITECTURE_DEFECT` | Blocks Theorem C, Provisioning |
| `FND-ARCH-015` | Candidate-freeze predicate coupling with denominator | **DUPLICATE** | `FND-ARCH-011` | `OPEN` | `ARCHITECTURE_DEFECT` | Duplicate (non-blocking) |

---

### 4. Population Totals & Parity Verification

- **Identity Registry Total**: 15 findings
- **Admission Registry Total**: 15 findings
- **Decision Registry Total**: 15 findings
- **Admitted Population**: 13 findings
- **Duplicate Population**: 2 findings (`FND-ARCH-013`, `FND-ARCH-015`)
- **Unadjudicated Findings**: 0
- **Open Architecture Defects**: 8 (`001`, `004`, `007`, `008`, `010`, `011`, `012`, `014`)
- **Universe Parity Status**: `PASS`
