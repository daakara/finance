# ARX Terminal — Evidence-Led Discovery Pilot: Evaluation Report (Corrected)

**Document ID:** `ARX-DISCOVERY-PILOT-EVALUATION-REPORT-V2`  
**Governing Authority:** [daakara/finance#43](https://github.com/daakara/finance/issues/43) (`BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043`)  
**Methodological Context:** ARX Closure Model v2 (Decision-First, Minimum Sufficient Evidence)  
**Evaluation Date:** October 22, 2026  
**Information Cutoff Date:** October 22, 2026 (Audited SEC EDGAR Form 10-K FY2025 and Form 10-Q 1H 2026 filings)  
**Process Recommendation:** **`CONTINUE` (Qualified Recommendation for Further Bounded Experiment)**  

---

## 1. Executive Summary & Process Recommendation

### Process Recommendation: **`CONTINUE`**
> **Recommendation:** Proceed with another bounded discovery experiment, subject to separate product owner authorization.  
> The pilot demonstrated that structured, decision-focused research successfully separates operational enablers, traces primary financial filings, and uncovers critical differences in capital structure, cash generation, and customer concentration across data center infrastructure providers.

### Epistemic Qualifications & Honest Governance Boundaries
- **Candidate-Generation Status:** Automated candidate-generation effectiveness was **not established** in this pilot. The 14 evaluated companies represent an **exploratory manually selected comparison cohort**, not an automated full-market scan.
- **No Claims of Investment Alpha:** This evaluation assesses **research process usefulness**, not demonstrated investment skill, predictive alpha, or backtested stock performance.
- **Process Recommendation $\neq$ Automatic Authorization:** A `CONTINUE` recommendation indicates that the methodological approach warrants further bounded testing; it does **not** constitute automatic authorization to begin another cycle or deploy resources without prior approval.
- **Zero Public Surface Mutation:** All records remain internal pilot artifacts. The published Quality-Watch collection remains frozen at exactly six approved assets: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK`.

---

## 2. Universe Examined & Discovery Scope

### Population Lineage vs. Actual Examination
- **Reference Population Baseline:** Sprint 2 Package 1 Multi-Feed Baseline (`coverage_capacity_baseline.json`), identifying 16,448 distinct join keys and ~5,871 provisional operating common stocks.
- **Actual Examined Universe:** Bounded diagnostic sample of **14 operating companies**:
  - *10 comparison candidates from previous queue:* `KLAC`, `LULU`, `EME`, `VRT`, `ANET`, `CDNS`, `PWR`, `SNPS`, `NOW`, `ELF`.
  - *4 additional exploratory candidates:* `MPWR`, `ODFL`, `HUBB`, `FICO`.
- **Primary Source Grounding:** All material factual claims reconciled to audited SEC EDGAR Form 10-K (FY2025) and Form 10-Q (1H 2026 / Q2 2026 / Q3 2026) filings.

---

## 3. Findings Across Evaluated Candidates

### A. What Useful Research Findings Were Actually Demonstrated?
1. **`VRT` (Substantial Balance Sheet Deleveraging Uncovered):**
   - *Previous Error:* Prior drafts permanently disqualified Vertiv based on stale FY2023 figures ($2.8B net debt, interest consuming >20% of op income).
   - *Current Audited Reality:* Vertiv generated **$2,113.8M in CFO** and **$1,893.8M in FCF** in FY2025. By Q2 2026, cash reached **$2,810.6M** against long-term debt of **$2,939.8M**, reducing **Net Funded Debt to $129.2M** (near net cash). FY2025 interest expense fell to **$86.1M** (only 4.7% of GAAP operating income).
   - *Result:* Corrected triage from disqualified to **`ACTIVE_QUALITY_WATCH_CANDIDATE`** for subsequent formal Quality-Watch evaluation.
2. **`ANET` (Customer Concentration Clarified):**
   - Form 10-K FY2025 disclosures established that two customers accounted for 26% and 16% of sales, representing **42% consolidated revenue concentration** (up from 39% in FY23). While Arista’s balance sheet is pristine ($2.3B+ cash, zero debt, 42.8% op margin), this concentration represents a critical operational uncertainty that justifies internal monitoring before public publication.
3. **`MPWR` & `VRT` (Contextual Role Clarification):**
   - Replaced previous inaccurate claims that MPWR "solves the exact same bottleneck" or is "categorically superior" to VRT.
   - *Clarification:* MPWR addresses silicon-level DC-DC power delivery on accelerator boards; Vertiv addresses facility-level modular cooling and power infrastructure. These are complementary systems, not identical bottlenecks. Having zero debt does not inherently establish superior shareholder outcomes at respective market prices.
4. **`SNPS` (Factual Error Corrected):**
   - Corrected earlier inaccurate assertion that the Ansys acquisition was pending. Form 8-K establishes merger completion on **July 17, 2025**.
   - Deferred due to post-merger integration complexity, $9.0B in funded debt, and the need to observe multi-quarter post-merger cash flow and debt paydown.
5. **Removal of Unsupported Automatic Disqualifiers:**
   - Removed blanket disqualifications based on debt, customer concentration, or consumer focus (e.g., `LULU`). Candidates are evaluated contextually on financial resilience, competitive moat durability, and capital allocation.

---

## 4. Process Effort & Maintenance Analysis

| Metric Category | Observed / Estimated Value | Methodological Basis |
| :--- | :--- | :--- |
| **Initial Full-Queue Screening (14 Companies)** | ~2.5 hours | **ESTIMATED Active Work** (Manual SEC filing lookup, note extraction, triage authoring) |
| **Deep Investigation Preparation (3 Candidates)** | ~2.0 hours | **ESTIMATED Active Work** (Statements of cash flows, debt bridge, customer note reconciliation) |
| **Total Task Active Effort** | ~4.5 hours | **ESTIMATED Active Work** (No automated timecard maintained) |
| **Quarterly Monitoring Upkeep per Candidate** | ~30–45 minutes | **PROVISIONAL ESTIMATE** (Projected quarterly 10-Q review against review triggers) |
| **Historical Multi-Cycle Maintenance Burden** | N/A | **EXPLICITLY UNMEASURED** (Requires multi-quarter tracking to measure empirically) |

---

## 5. Verification & Safety Attestations

- **Automated Verification Script (`scripts/research/verify_discovery_pilot.py`):** Verified schema compliance, invariant preservation, and standard references.
- **Regression Unit Suite (`tests/test_evidence_led_discovery_pilot.py`):** Verified offline calculation reconciliation ($FCF = CFO - Capex$), baseline hash immutability, and standard references (**5/5 tests pass**).
- **Public Publication Invariant:** All internal pilot records enforce `public_publication_eligibility: INTERNAL_PILOT_RECORD_NOT_ELIGIBLE`.
- **Zero Public Mutation:** Published collection remains frozen at exactly 6 approved assets: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK`.
- **Baseline Immutability:** [`odp05_discovery_ledger.json`](file:///docs/research/opportunity_discovery/odp05_discovery_ledger.json) canonical SHA-256 (`a65be3b...`) verified intact.
- **Frontend Isolation:** Zero internal research records referenced in `frontend/`.
- **Operational Invariants:** Scanner execution, universe publication, and ingestion remain strictly inactive.

---

## 6. Next Bounded Action Supported by Evidence

Under the **`CONTINUE`** recommendation:
1. Maintain Draft PR [#45](https://github.com/daakara/finance/pull/45) as the delivery record for this corrected pilot and standardization task.
2. In a future separately authorized cycle, evaluate `VRT` formally against the six Quality-Watch pillars using its deleveraged FY2025/1H2026 financial baseline.
3. Design a prospective provenance tracking mechanism for automated candidate discovery before attempting broader universe scanning.
4. Enforce stopping rule: Cease active execution upon delivery of this report without premature operational activations or public manifest changes.
