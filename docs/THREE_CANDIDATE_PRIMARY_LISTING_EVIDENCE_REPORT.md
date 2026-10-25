# Three-Candidate Primary Listing Evidence Adjudication & Offline Replay Report

**Candidates:** AAPL, GOOGL, GOOG (Evaluated); TSM, ACCS, DDS, AACIU (Not Evaluated in this Bounded Task)  
**Sprint / Scope:** Sprint 2 Bounded Evidence Adjudication & R06 Semantic Recording  
**Date:** 2026-10-17  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-three-candidate-listing-evidence`  
**Branch:** `feat/sprint-2-three-candidate-listing-evidence`  
**Base Commit:** `6eb1d40356d74ebf19075ac493f12417bd175780`  
**Full Canonical-Rules Dictionary Digest:** `ELIGIBILITY_RULES_HASH = 1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1` (Full dictionary digest over R01–R07, unchanged)

---

### 1. Executive Summary

This report documents the recording of the Product Owner's approved **R06 semantic direction**, the implementation of the bounded evidence adjudicator (`PrimaryListingAdjudicator`), and the adjudication results for **AAPL**, **GOOGL**, and **GOOG**.

Following the approved semantic standard:
1. `primary_listing` identifies the designated primary exchange listing for the exact security class or instrument.
2. It does not identify the issuer's most important or voting share class, country of incorporation or headquarters, ARX's selected US assessment line (`selected_us_assessment_listing`), or the underlying ordinary share's home market when evaluating an ADR.
3. Evidence sufficiency permits either (Path A) attributable direct identification of designated primary exchange listing, or (Path B) a documented derivation supported by appropriate source authority and instrument-specific facts (exclusivity required only if the derivation relies on exclusivity). The adjudicator does not require both paths, nor does it universally demand exclusivity proof.
4. Conflicting or insufficient evidence remains strictly UNKNOWN (`None`).
5. Neither Section 12(b) exchange registration alone nor the absence of other exchanges in a Form 10-K constitutes proof of an exclusive primary listing.
6. Multi-class equity instruments (e.g. `GOOGL` Class A voting vs `GOOG` Class C non-voting) are evaluated under the identical standard: voting power does not confer primary status and absence of voting power does not subordinate a class.

Applying `R06_BOUNDED_EVIDENCE_ADJUDICATION_V1` (v1.0.0) to the authentic SEC Form 10-K filings and Nasdaq exchange directory rows, all three candidates are categorized as `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` with status `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY`. In strict adherence to R06 and the non-invention rule, all three candidates retain `assigned_primary_listing = None` (UNKNOWN).

Offline batch replay through the released platform capabilities confirms that whole-batch conversion fails closed with **0 converted, 0 admitted, builder status NOT_RUN**, and **zero production adoption authority conferred**.

---

### 2. Approved Semantic Decision & Rationale

#### 2.1 Approved PO Semantic Direction
The Product Owner approved the following semantic definition:
> `primary_listing` identifies the designated primary exchange listing for the exact security class or instrument.
>
> It does not identify:
> - The issuer's most important or voting share class.
> - Country of incorporation or headquarters.
> - ARX's selected US assessment line.
> - The underlying ordinary share's home market when evaluating an ADR.
>
> Evidence may support a documented derivation without containing the literal words "primary listing." A derivation must remain distinguishable from an explicit publisher assertion.
>
> Conflicting or insufficient evidence remains UNKNOWN.

#### 2.2 Canonical Rule Invariant & Exact Rule Hash
- **Rule Contract:** `CANONICAL_ELIGIBILITY_RULES["R06_PRIMARY_LISTING"]` in `analyst_dashboard/universe/contracts.py`:
  - `rule_id`: `"R06_PRIMARY_LISTING"`
  - `operator`: `"=="`
  - `expected`: `True`
  - `description`: `"Must be primary listing, not secondary or cross-listing"`
- **Digest Invariant:** `ELIGIBILITY_RULES_HASH = 1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1`
- **Integrity Verification:**
  1. **Full Canonical-Rules Dictionary Digest:** `1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1` is the digest of the entire `CANONICAL_ELIGIBILITY_RULES` dictionary across all rules R01 through R07, NOT an R06-only hash.
  2. **Unchanged Executable Predicate:** The R06 predicate (`operator == "=="`, `expected == True`) is 100% unchanged.
  3. **Unchanged Canonical Dictionary Bytes/Content:** The dictionary content and serialization bytes are strictly unchanged.
  4. **Newly Approved Semantic Interpretation:** Provides authoritative evidence sufficiency criteria (Path A direct identification or Path B documented derivation) without mutating rule dictionary definitions.

#### 2.3 Key Semantic Distinctions
1. **Exchange Registration Fact $\ne$ Designated Primary Listing Claim:** Section 12(b) of the Exchange Act requires national securities exchange trading registration. The Form 10-K Section 12(b) registration table certifies trading registration on The Nasdaq Stock Market LLC, but does not assert that Nasdaq is the designated primary exchange listing.
2. **Absence of Other Exchanges $\ne$ Proof of Exclusivity:** The absence of other registered venues in a Form 10-K table cannot be assumed as proof of exclusivity or primary designation. Exclusivity proof is required only if a specific derivation relies on exclusivity.
3. **Directory Listing Fact $\ne$ Designated Primary Listing Claim:** The Nasdaq Symbol Directory confirms active listing on Nasdaq under tier Q (Nasdaq Global Select Market), but "listed on Nasdaq" cannot be equated with "primary listing on Nasdaq".
4. **Voting Rights Neutrality:** Voting rights (1 vote for GOOGL Class A vs 0 votes for GOOG Class C) are corporate governance rights that do not confer or alter exchange listing primacy. Both classes are evaluated under the identical standard.
5. **Orthogonality of Assessment Selection:** `selected_us_assessment_listing` remains an orthogonal caller parameter designating the specific US assessment line and does not substitute for or alter `primary_listing`.

---

### 3. Adjudication Rule Specification (`R06_BOUNDED_EVIDENCE_ADJUDICATION_V1`)

- **Rule Identifier:** `R06_BOUNDED_EVIDENCE_ADJUDICATION_V1`
- **Rule Version:** `1.0.0`
- **Implementation:** [`PrimaryListingAdjudicator`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/primary_listing_evidence.py#L510-L660) in `analyst_dashboard/universe/primary_listing_evidence.py`.
- **Evidence Sufficiency Standards (Two Alternative Paths):**
  - **Path A (Attributable Direct Identification):** Direct identification of the exact instrument's designated primary exchange listing from an authoritative listing source. Path A does not require derivation logic or exclusivity proof.
  - **Path B (Documented Derivation):** A documented derivation supported by appropriate source authority and instrument-specific facts. Path B does not require literal publisher wording "primary listing". Proof of exclusivity is required *only* if the particular derivation relies on exclusivity.
  - **Non-Universal Exclusivity:** Exclusivity proof is NOT universally required. The adjudicator does NOT require both paths. Inference from the mere absence of other listings in one source remains rejected.
- **Qualified Source Hierarchy as Evidence Policy:**
  - $\text{PRIMARY\_EXCHANGE\_AUTHORITY} \succ \text{REGULATORY\_FILINGS\_SEC\_12B} \succ \text{EXCHANGE\_DIRECTORY}$
  - This hierarchy represents this adjudicator's claim-specific evidence policy for evaluating primary listing designations, not a universal factual ranking.
  - A higher-ranked source tier does not automatically erase conflicting claims; instrument scope, observation dates, and conflict dispositions are strictly preserved.
- **Evidence Roles:**
  - `DIRECT_PRIMARY_IDENTIFICATION`: Direct identification of the exact instrument's primary/designated listing by an authoritative source.
  - `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT`: Corroborating class-specific registration and exchange-directory facts.
  - `CONFLICTING_OR_DISCORDANT_OBSERVATION`: Conflicting, discordant, or disputed venue observations (e.g. DDS TXSE vs NYSE; ACCS OTC vs AMEX).
  - `STALE_OR_OUT_OF_SCOPE_OBSERVATION`: Stale or temporally mismatched observations.
  - `INSUFFICIENT_NARROW_OBSERVATION`: Insufficient observation or candidate outside evaluated scope.

---

### 4. Three-Candidate Adjudication Findings

| Field / Dimension | AAPL | GOOGL | GOOG |
|:---|:---|:---|:---|
| **Instrument Symbol** | `AAPL` | `GOOGL` | `GOOG` |
| **Share Class / Title** | Common Stock, $0.00001 par | Class A Common Stock, $0.001 par | Class C Capital Stock, $0.001 par |
| **Voting Power** | 1 vote per share | 1 vote per share | 0 votes per share (Non-voting) |
| **Issuer CIK** | `0000320193` (Apple Inc.) | `0001652044` (Alphabet Inc.) | `0001652044` (Alphabet Inc.) |
| **Filing Accession** | `0000320193-25-000079` (10-K) | `0001652044-26-000018` (10-K) | `0001652044-26-000018` (10-K) |
| **Registered Exchange** | The Nasdaq Stock Market LLC | Nasdaq Stock Market LLC (Tier Q) | Nasdaq Stock Market LLC (Tier Q) |
| **Adjudication Role** | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` |
| **Is Direct Assertion?** | `False` | `False` | `False` |
| **Is Derived Claim?** | `False` | `False` | `False` |
| **Adjudication Status** | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` |
| **Assigned Primary Listing** | **`None`** (UNKNOWN) | **`None`** (UNKNOWN) | **`None`** (UNKNOWN) |
| **Missing Evidence Definition** | Absence of other registered exchanges in Form 10-K cannot be assumed as proof of venue exclusivity; Direct identification of designated primary exchange listing from an authoritative listing authority or verified cross-exchange listing master is required; Section 12(b) exchange registration certifies trading registration on The Nasdaq Stock Market LLC, but does not assert an exclusive primary listing claim. | Same standard as AAPL; multi-class equity structure verified. Voting rights do not confer primary status and non-voting status does not subordinate class. | Same standard as AAPL; multi-class equity structure verified. Voting rights do not confer primary status and non-voting status does not subordinate class. |
| **Evaluation Rationale** | Evidence confirms active listing and proves registration under Section 12(b) of the Exchange Act, not designated primary listing. Exchange registration alone is insufficient, and absence of other venues cannot be assumed as proof of exclusivity. | Evidence confirms active listing and proves registration under Section 12(b) of the Exchange Act, not designated primary listing. Class A voting rights do not confer primary status. | Evidence confirms active listing and proves registration under Section 12(b) of the Exchange Act, not designated primary listing. Class C non-voting status does not subordinate. |

---

### 5. Seven-Candidate Adjudication Summary

The four other candidates from the sample population were **not evaluated** in this narrow 3-candidate task, strictly preserving existing archived observations and blocking population creep:

| Candidate | Task Scope | Adjudication Role | Evidence Classification | Primary Listing Value | Status & Blocking Rationale |
|:---|:---:|:---|:---|:---:|:---|
| **AAPL** | `EVALUATED` | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `None` | **BLOCKED:** Section 12(b) registration established; primary listing unevidenced. |
| **GOOGL** | `EVALUATED` | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `None` | **BLOCKED:** Class A 12(b) registration established; voting rights do not confer primary status. |
| **GOOG** | `EVALUATED` | `CORROBORATING_REGISTRATION_AND_DIRECTORY_FACT` | `EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY` | `None` | **BLOCKED:** Class C 12(b) registration established; non-voting does not subordinate class. |
| **TSM** | `NOT_EVALUATED` | `INSUFFICIENT_NARROW_OBSERVATION` | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | `None` | **BLOCKED:** Outside 3-candidate scope; Depositary Receipt ordinary share primacy unresolved. |
| **ACCS** | `NOT_EVALUATED` | `CONFLICTING_OR_DISCORDANT_OBSERVATION` | `CONFLICTING_EVIDENCE` | `None` | **BLOCKED:** Outside 3-candidate scope; Historical CIK ambiguity and OTC vs AMEX venue conflict. |
| **DDS** | `NOT_EVALUATED` | `CONFLICTING_OR_DISCORDANT_OBSERVATION` | `CONFLICTING_EVIDENCE` | `None` | **BLOCKED:** Outside 3-candidate scope; TXSE vs NYSE vs Other Listed 'F' venue conflict. |
| **AACIU** | `NOT_EVALUATED` | `INSUFFICIENT_NARROW_OBSERVATION` | `CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION` | `None` | **BLOCKED:** Outside 3-candidate scope; Multi-component SPAC unit contract semantics unresolved. |

---

### 6. Traffic-Light Adjudication & Admission Dashboard

```
=============================================================================================================
                                ARX TERMINAL — SPRINT 2 EVIDENCE ADJUDICATION
=============================================================================================================
Symbol   Scope         Exchange Venue   Role                  Assigned PL   Batch Gate   Production Promotion
-------------------------------------------------------------------------------------------------------------
AAPL     EVALUATED     NASDAQ (XNAS)    CORROBORATING_FACT    UNKNOWN(None) [BLOCKED]    [QUARANTINED]
GOOGL    EVALUATED     NASDAQ (XNAS)    CORROBORATING_FACT    UNKNOWN(None) [BLOCKED]    [QUARANTINED]
GOOG     EVALUATED     NASDAQ (XNAS)    CORROBORATING_FACT    UNKNOWN(None) [BLOCKED]    [QUARANTINED]
TSM      NOT_EVALUATED NYSE (XNYS)      INSUFFICIENT_OBS      UNKNOWN(None) [BLOCKED]    [QUARANTINED]
ACCS     NOT_EVALUATED DISCORDANT       CONFLICTING_OBS       UNKNOWN(None) [BLOCKED]    [QUARANTINED]
DDS      NOT_EVALUATED DISCORDANT       CONFLICTING_OBS       UNKNOWN(None) [BLOCKED]    [QUARANTINED]
AACIU    NOT_EVALUATED NASDAQ (XNAS)    INSUFFICIENT_OBS      UNKNOWN(None) [BLOCKED]    [QUARANTINED]
-------------------------------------------------------------------------------------------------------------
Whole-Batch Admission Status:   FAILED CLOSED (is_batch_valid = False, batch_conversion_count = 0)
Universe Builder Status:        NOT_RUN (builder_handoff_permitted = False)
Production Adoption Authority:  DISABLED (production_adoption_allowed = False, provisional_quarantine = True)
=============================================================================================================
```

---

### 7. Replay Verification: Real-Source vs Simulated Admission

Replay was executed across the full ledger using [`PrimaryListingReplayEngine.run_replay()`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/analyst_dashboard/universe/primary_listing_evidence.py#L740-L970):

| Replay Dimension | Simulated Mode (`is_simulation=True`) | Real Source Mode (`is_simulation=False`) | Invariant Enforced |
|:---|:---:|:---:|:---|
| **Production Adoption Authority** | `DISABLED` | `DISABLED` | `production_adoption_allowed = False` |
| **Provisional Quarantine** | `ENFORCED` (`True`) | `ENFORCED` (`True`) | All candidate IDs remain `PROV_SEC_*` |
| **Evaluated Candidates** | `3` (`AAPL`, `GOOGL`, `GOOG`) | `3` (`AAPL`, `GOOGL`, `GOOG`) | Bounded scope strictly respected |
| **Unevaluated Candidates** | `4` (`TSM`, `ACCS`, `DDS`, `AACIU`) | `4` (`TSM`, `ACCS`, `DDS`, `AACIU`) | Marked `NOT_EVALUATED` |
| **Isolated Probe Conversions** | **`0`** | **`0`** | Blocked by missing validated boolean `primary_listing` |
| **Conversion Blocked Count** | **`7`** | **`7`** | Fails closed on unevidenced or conflicting records |
| **Whole-Batch Gate (`is_batch_valid`)** | **`False`** | **`False`** | Atomic whole-batch fail-closed protection |
| **Batch Conversion Count** | **`0`** | **`0`** | Zero `SourceSecurity` instances created |
| **Batch Admitted Count** | **`0`** | **`0`** | Zero records admitted downstream |
| **Universe Builder Status** | **`NOT_RUN`** | **`NOT_RUN`** | Handoff prohibited (`builder_handoff_permitted = False`) |

---

### 8. Test Execution Summary

The focused test suite [`tests/test_universe_primary_listing_evidence.py`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-three-candidate-listing-evidence/tests/test_universe_primary_listing_evidence.py) contains 45 tests, all passing cleanly:

```bash
tests\test_universe_primary_listing_evidence.py ............................................. [100%]
============================= 45 passed in 4.19s ==============================
```

The newly added test class `TestPrimaryListingEvidenceAdjudication` verifies all required items:
1. **Class-level independence:** `GOOGL` and `GOOG` evaluated under identical standards; voting rights do not confer primary status; non-voting does not subordinate.
2. **Direct assertion vs derived evidence:** Direct publisher assertions pass as direct; derivations are distinguishable and cite supporting facts.
3. **Alternative evidence paths independence & non-universal exclusivity:** Verified that Path A (direct identification) and Path B (documented derivation) are independent alternatives, and exclusivity is not universally required (exclusivity required only if the derivation relies on exclusivity).
4. **Registration-only evidence:** SEC 12(b) and exchange directory listings remain narrower corroborating facts where primary designation is absent; absence of other exchanges does not prove exclusivity.
5. **Stale and conflicting evidence:** Handling of discordant venues (`DDS`/`ACCS` style) and multi-venue claims; verifies that source hierarchy is an evidence policy ranking, not a factual erasure tool (a higher-ranked source tier does not automatically erase conflicting claims).
6. **Unknown preservation:** Non-primary, conflicting or unadjudicated rows do not populate `primary_listing = True`; `to_source_security()` cannot be called when `primary_listing` is `None`.
7. **Rule-version and provenance retention:** Adjudication records retain rule ID (`R06_BOUNDED_EVIDENCE_ADJUDICATION_V1`), version (`1.0.0`), timestamp (`2026-10-17`), source hierarchy, exact claims, and record locators. Distinguishes full canonical-rules dictionary digest (`1ff50853...`) from unchanged R06 predicate and newly approved semantic interpretation.
8. **Whole-batch protection:** Adjudication failure or missing evidence in any candidate fails the batch conversion atomically.
9. **No implicit production promotion:** Verified that provisional candidate IDs (`PROV_SEC_*`) remain uncommitted and production adoption is blocked.

---

### 9. Remaining Exact Evidence Requirements

To establish `primary_listing = True` under R06 for any candidate, the following alternative evidence paths govern:
1. **Path A (Attributable Direct Primary Identification):** A direct assertion from an authoritative listing exchange authority (e.g. an exchange listing certificate, official listing authority master, or direct listing rule certification) explicitly designating the exact security class as the primary listing. Path A does not require derivation logic or proof of exclusivity.
2. **Path B (Documented Derivation):** A documented derivation citing specific, documented exchange authority operating rules and instrument-specific supporting facts, remaining fully distinguishable from an explicit publisher assertion. Proof of exclusivity is required *only* if the particular derivation relies on exclusivity.
3. **Rejected Inferences:** Inference from the mere absence of other listings in a regulatory filing remains strictly rejected under both paths.

Until attributable evidence satisfying Path A or Path B is acquired, all candidates strictly preserve `assigned_primary_listing = None` (UNKNOWN).

---

### 10. Strict Scope & Production Boundary Declarations

```ini
PRIMARY_LISTING_EVIDENCE_ADJUDICATION = THREE_CANDIDATE_OFFLINE_VERIFIED
R06_SEMANTIC_DIRECTION = RECORDED_AND_APPLIED
FULL_CANONICAL_RULES_DICTIONARY_DIGEST = 1ff5085341f449908d521c45261a35e409b143fdb01e73334950c891d09248e1 (UNMODIFIED)
UNCHANGED_R06_PREDICATE = (operator == "==", expected == True)
REAL_SOURCE_POPULATION_ADMISSION = NOT_ESTABLISHED
PRODUCTION_SECURITY_IDENTITY_ADOPTION = NO
PROVISIONAL_ID_PROMOTION = NO
PRODUCTION_DATABASE_WRITES = NONE
UNIVERSE_STORE_WRITES = NONE
ADDITIONAL_LIVE_SOURCE_ACQUISITION = NONE
MARKET_DATA_BACKFILL = NONE
STRATEGY_POLICY_CHANGES = NONE
UNIVERSE_PUBLICATION = NO
SCANNER_ACTIVATION = NO
SCHEDULER_ACTIVATION = NO
PROSPECTIVE_CAPTURE_ENABLEMENT = NO
EXPANDED_ASSET_IMPLEMENTATION = NO
MODEL_TUNING = FROZEN
EMPIRICAL_STRATEGY_VALIDATION = NOT_ESTABLISHED
GATE12_ADVANCEMENT = NO
BRANCH_OR_WORKTREE_DELETION = NO
```
