# Three-Candidate Primary Listing Semantics Contract Decision Record

**Target Candidates:** AAPL, GOOGL, GOOG  
**Sprint / Scope:** Sprint 2 Bounded Three-Candidate Listing Evidence Adjudication  
**Date:** 2026-10-17  
**Decision Authority:** Product Owner Approval (Binding)  
**Status:** APPROVED & ADJUDICATED (Fail-Closed, Zero Production Adoption)  

---

### 1. Approved Semantic Direction

The Product Owner has approved the following authoritative semantic definition:

> **`primary_listing` identifies the designated primary exchange listing for the exact security class or instrument.**

#### 1.1 Excluded Derivations & Orthogonal Fields
`primary_listing` does **not** identify:
1. **The issuer's most important or voting share class:** Voting power is an internal corporate governance right, not an exchange listing designation.
2. **Country of incorporation or headquarters:** Legal jurisdiction and principal executive offices are tracked independently.
3. **ARX's selected US assessment line:** `selected_us_assessment_listing` remains an orthogonal caller selection parameter governing which line ARX evaluates for US market assessment.
4. **The underlying ordinary share's home market when evaluating an ADR:** For American Depositary Receipts, the primary exchange listing refers to the ADR instrument's designated US listing venue, not the underlying foreign ordinary share's home exchange.

#### 1.2 Derivation Standard
- Evidence may support a documented derivation without containing the literal words *"primary listing"*, provided the derivation remains distinguishable from an explicit publisher assertion.
- Conflicting or insufficient evidence remains strictly **UNKNOWN (`None`)**.
- This approval does **not**:
  - Automatically set AAPL, GOOG, or GOOGL to `True`.
  - Establish that exchange registration alone is sufficient.
  - Approve voting primacy.
  - Authorize identity promotion, population publication, or scanner activation.
  - Alter R02, R04, or other canonical eligibility predicates.

---

### 2. Alignment of Baseline & Removal of Rejected Paths

In accordance with PO direction, previous theoretical alternatives are aligned as follows:

1. **Removal of Voting-Primacy Path:**
   - Voting-primacy alternatives (e.g. designating Class A as primary solely due to voting rights while demoting Class C) are **permanently removed** from the acceptance path.
   - Distinct share classes (such as Alphabet Inc. Class A `GOOGL` and Class C `GOOG`) are adjudicated using the **identical semantic standard**.
2. **Removal of Sole Exchange Registration Equivalence:**
   - The assertion that sole Section 12(b) exchange registration legally establishes primary-listing status is **removed**.
   - The absence of another exchange in a regulatory filing cannot be treated as affirmative proof that no other listing exists.
3. **Removal of Mandatory Literalism Assertion:**
   - The claim that the baseline contract required the exact literal words *"primary listing"* is **removed**. Rule R06 requires the substantiated fact of primary listing, which may be established by direct identification or valid documented derivation.
4. **Preservation of `selected_us_assessment_listing`:**
   - Maintained as an independent, explicit caller configuration.

---

### 3. Canonical Rule Contract & Hash Verification

The executable predicate of Rule R06 in [`analyst_dashboard/universe/contracts.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/contracts.py) is strictly unchanged:

```python
"R06_PRIMARY_LISTING": {
    "operator": "==",
    "expected": True,
    "description": "Must be primary listing, not secondary or cross-listing",
}
```

- **Executable Predicate:** `operator == "=="`, `expected == True` (Unchanged).
- **Rule Description:** `"Must be primary listing, not secondary or cross-listing"` (Aligned with PO direction).
- **Rule Dictionary Digest (`ELIGIBILITY_RULES_HASH`):**  
  `1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1`  
  *(0-byte change to rule dictionary; exact hash preserved and verified).*

---

### 4. Approved Bounded Adjudication Rule

The narrow adjudication rule ([`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/primary_listing_evidence.py#L450-L550)) enforces the following evidence roles and derivation standards:

#### 4.1 Evidence Roles
1. **`DIRECT_PRIMARY_IDENTIFICATION`:** Direct identification of the exact instrument's primary/designated listing by an appropriate listing authority.
2. **`CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT`:** Class-specific registration facts (SEC Form 10-K Section 12(b)) and exchange-directory listings (Nasdaq Symbol Directory).
3. **`CONFLICTING_OR_DISCORDANT_OBSERVATION`:** Unreconciled venue conflicts (e.g. DDS TXSE vs NYSE, ACCS NYSE vs AMEX).
4. **`STALE_OR_OUT_OF_SCOPE_OBSERVATION`:** Temporally expired or mismatched evidence.
5. **`INSUFFICIENT_NARROW_OBSERVATION`:** Insufficient narrow observations or unevidenced candidates.

#### 4.2 Derivation Requirements
For any derived conclusion:
- **Exact Source Claims & Locators:** Must be recorded without omission.
- **Instrument/Class Matching:** Must verify exact security title and share class.
- **Source Hierarchy:** `PRIMARY_EXCHANGE_AUTHORITY > REGULATORY_FILINGS_SEC_12B > EXCHANGE_DIRECTORY`.
- **Negative Assumptions Prohibited:** The absence of another exchange in one filing does not establish exclusivity.
- **Equating Prohibited:** "Listed on Nasdaq" cannot be equated with "designated primary listing on Nasdaq".

---

### 5. Application & Adjudication Outcome for Sample Candidates

#### 5.1 Three Evaluated Candidates

| Candidate | Instrument / Class | Evidence Role | Direct vs Derived | Adjudication Status | Assigned Value | Required Evidence for True |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **AAPL** | Common Stock, $0.00001 par value | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | Corroborating narrower fact | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None`** (UNKNOWN) | Authoritative direct identification of designated primary listing venue. |
| **GOOGL** | Class A Common Stock, $0.001 par value | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | Corroborating narrower fact | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None`** (UNKNOWN) | Authoritative direct identification of designated primary listing venue for Class A. |
| **GOOG** | Class C Capital Stock, $0.001 par value | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | Corroborating narrower fact | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | **`None`** (UNKNOWN) | Authoritative direct identification of designated primary listing venue for Class C. |

#### 5.2 Four Unevaluated Sample Candidates (Explicitly Outside Task Scope)
- **TSM, ACCS, DDS, AACIU:** Labeled **`NOT_EVALUATED`** for this task. They remain outside this bounded evidence acquisition and retain their previously recorded archived states (`EVIDENCE_UNAVAILABLE` or `CONFLICTING_EVIDENCE`).

---

### 6. Replay & Boundary Integrity

Under both real-source replay and proposal contract simulation:
- **`production_adoption_allowed = False`**: Zero provisional identity promotion.
- **`is_batch_valid = False`**: Atomic whole-batch gate strictly fails closed.
- **`batch_conversion_count = 0`**: Zero `SourceSecurity` instances admitted.
- **`builder_status = "NOT_RUN"`**: Downstream builder rules strictly unexecuted.
