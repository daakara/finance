# Three-Candidate Primary Listing Semantics Contract Decision Record

**Target Candidates:** AAPL, GOOGL, GOOG  
**Sprint / Scope:** Sprint 2 Bounded Three-Candidate Listing Evidence Acquisition  
**Date:** 2026-10-17  
**Status:** PROPOSED & PRESERVED (Fail-Closed, Zero Policy Change)  

---

### 1. Exact Current Code & Governing Wording

#### 1.1 Canonical Rule R06 Contract
In [`analyst_dashboard/universe/contracts.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/contracts.py):
```python
"R06_PRIMARY_LISTING": {
    "name": "Primary Listing Verification",
    "description": "Must be primary listing (not secondary, dual, or OTC)",
    "field": "primary_listing",
    "required": True,
    "operator": "==",
    "value": True,
}
```

#### 1.2 Validation Requirements & Ingestion Guards
In [`analyst_dashboard/universe/validator.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/validator.py):
```python
if reqs.require_primary_listing:
    val = record.get("primary_listing")
    if val is None:
        issues.append(
            ValidationIssue(
                field="primary_listing",
                issue_type=ValidationIssueType.MISSING_REQUIRED_FIELD,
                message="Field 'primary_listing' is None (unknown/unevidenced) and fails closed under Rule R06.",
                code="UNKNOWN_PRIMARY_LISTING",
                severity=ValidationSeverity.BLOCKING,
            )
        )
```

#### 1.3 Atomic Conversion Boundary
In [`analyst_dashboard/universe/validator.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/validator.py):
```python
if validated.primary_listing is None:
    raise ValueError(
        "Cannot construct SourceSecurity without explicit, validated boolean value for 'primary_listing' (None/unknown is blocked)"
    )
```

---

### 2. Actual Source Claims

#### 2.1 Source A: Official Exchange Reference Source (The Nasdaq Stock Market LLC)
- **Source Artifact:** `data/research/exchange_reference/nasdaqlisted_20261017.txt`
- **Source URL:** `http://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt`
- **Retrieval Timestamp:** `2026-10-17T16:17:36Z`
- **Digest (SHA-256):** `9f5c42346e6ab594a9e16bed99757e38cda89645c776b987626c9f51813bc99b`
- **Extracted Records:**
  - `AAPL`: `AAPL|Apple Inc. - Common Stock|Q|N|N|40|N|N` (Line 24)
  - `GOOG`: `GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|40|N|N` (Line 2152)
  - `GOOGL`: `GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|40|N|N` (Line 2153)
- **Observed Claim:** All three securities are listed on The Nasdaq Stock Market in Market Category `Q` (Nasdaq Global Select Market).

#### 2.2 Source B: SEC EDGAR Submissions API & Form 10-K Cover Page §12(b) Registration Tables
- **Submissions API Metadata:**
  - `AAPL` (CIK `0000320193`): `tickers: ['AAPL']`, `exchanges: ['Nasdaq']`
  - `GOOG` / `GOOGL` (CIK `0001652044`): `tickers: ['GOOGL', 'GOOG']`, `exchanges: ['Nasdaq', 'Nasdaq']`
- **Form 10-K Cover Page §12(b) Table (Governing Statutory Registration):**
  - Section title: *"Securities registered pursuant to Section 12(b) of the Act"*
  - Table schema: `Title of each class` | `Trading Symbol(s)` | `Name of each exchange on which registered`
  - Rows for Apple Inc. (Accession `0000320193-24-000123` / `0000320193-25-000079`):
    - `Common Stock, $0.00001 par value per share` | `AAPL` | `The Nasdaq Stock Market LLC`
  - Rows for Alphabet Inc. (Accession `0001652044-24-000022` / `0001652044-25-000014`):
    - `Class A Common Stock, $0.001 par value` | `GOOGL` | `The Nasdaq Stock Market LLC (Nasdaq Global Select Market)`
    - `Class C Capital Stock, $0.001 par value` | `GOOG` | `The Nasdaq Stock Market LLC (Nasdaq Global Select Market)`

---

### 3. Competing Interpretations

| Interpretation | Conceptual Premise | Treatment of Regulatory Wording |
|:---|:---|:---|
| **1. Strict Semantic Literalism (Narrow Fact)** | `primary_listing` requires an affirmative regulatory statement stating the word *"primary"*. | Form 10-K §12(b) and Nasdaq directory prove exchange registration and listing, but neither document prints the literal string *"primary listing"*. Therefore, `primary_listing` must remain `None` (unknown). |
| **2. Sole-Venue Sovereign Registration Equivalence** | For a domestic US issuer, registration under §12(b) exclusively on a single national exchange constitutes the primary listing as a matter of law. | Because Apple and Alphabet register their equity classes solely on The Nasdaq Stock Market LLC (with zero secondary or dual national exchange registrations), this sole §12(b) registration IS their primary listing. |
| **3. Multi-Class Capital Structure Divergence** | In multi-class structures (Alphabet Class A vs Class C), shares differ in voting power. | **3a (Voting Primacy):** Only voting Class A is primary; non-voting Class C is a derivative/secondary class.<br/>**3b (Dual Primary Classes):** Both Class A and Class C are separate, independent primary listings of their respective classes on Nasdaq, as neither is a cross-listing of the other. |

---

### 4. Consequences for AAPL, GOOG and GOOGL

#### Under Interpretation 1 (Current Code Baseline):
- **AAPL:** `primary_listing = None` → Validation fails closed (`UNKNOWN_PRIMARY_LISTING`) → Conversion to `SourceSecurity` blocked.
- **GOOGL:** `primary_listing = None` → Conversion blocked.
- **GOOG:** `primary_listing = None` → Conversion blocked.
- **Replay Outcome:** 0 converted, 3 blocked. Whole batch fails closed. Zero admission authority conferred.

#### Under Interpretation 2 (Sole-Venue Equivalence):
- **AAPL:** Converts to `SourceSecurity` with `primary_listing = True`.
- **GOOGL / GOOG:** Requires resolution of Interpretation 3 (whether multi-class equity permits dual primary classes or requires voting primacy).

#### Under Interpretation 3 (Multi-Class Treatment):
- If 3a (Voting Primacy) is adopted: GOOGL converts to True; GOOG is marked `primary_listing = False` (or blocked). However, Rule R06 and Exchange Act Section 12(b) do not condition primary listing status on voting rights.
- If 3b (Dual Primary) is adopted: Both GOOGL and GOOG convert to `primary_listing = True` because each is a distinct, independently registered class listed on Nasdaq Global Select Market.

---

### 5. Smallest Compatible Resolution

1. **Preserve Current Invariants Strictly (No Policy Change)**:
   - For this sprint, maintain `assigned_primary_listing = None` across all three candidates (`AAPL`, `GOOGL`, `GOOG`).
   - Classify all three records as `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`.
   - Replay strictly fails closed with `0/3` admitted and `builder_status = 'NOT_RUN'`.
2. **Path for Future Architectural Ratification**:
   - If the operator/architectural council approves, a formal semantic mapping policy may be enacted that maps verified sole-venue Section 12(b) registration on an approved US exchange (XNAS/XNYS) to `primary_listing = True` for single-class issuers, and resolves multi-class capital structures under Interpretation 3b.
   - Until such ratification, zero synthetic values are substituted into the platform.
