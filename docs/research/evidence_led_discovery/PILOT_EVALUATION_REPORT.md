# ARX Terminal — Evidence-Led Discovery Pilot: Evaluation Report

**Document ID:** `ARX-DISCOVERY-PILOT-EVALUATION-REPORT-V1`  
**Governing Authority:** [daakara/finance#43](https://github.com/daakara/finance/issues/43) (`BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043`)  
**Methodological Context:** ARX Closure Model v2 (Decision-First, Minimum Sufficient Evidence)  
**Evaluation Date:** October 22, 2026  
**Evaluation Outcome / Binding Process Decision:** **`CONTINUE`**

---

## 1. Executive Summary & Binding Process Decision

### Process Decision: **`CONTINUE`**
> **Recommendation:** Authorize another bounded cycle of evidence-led discovery.  
> The pilot demonstrated clear research-process utility: it successfully expanded the candidate universe beyond the initial 10-name shortlist, discovered a structurally superior candidate (`MPWR`) solving the same physical constraint as a shortlist candidate (`VRT`) with vastly superior value capture, and prevented the false-positive qualification of heavily indebted thematic darlings.

### Epistemic Qualifications & Governance Boundaries
- **Research Utility Observed Now:** The `CONTINUE` decision evaluates the **usefulness of the research discovery process** in improving candidate prioritization and filtering out capital structure risks.
- **Hypothesis Outcomes Awaiting Future Evidence:** The operational developments tracked for `ANET`, `VRT`, and `MPWR` are ongoing; intermediate verification depends on subsequent quarterly SEC EDGAR filings (Forms 10-Q/8-K).
- **Investment Performance Unestablished:** This pilot does **not** claim demonstrated investment alpha, backtested predictive skill, or guaranteed shareholder returns.
- **Zero Public Mutation:** All records remain strictly internal. The published Quality Watch collection remains frozen at exactly six approved assets: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK`.

---

## 2. Universe Examined & Discovery Routes

### Population Lineage & Coverage
- **Baseline Authority:** Sprint 2 Package 1 Multi-Feed Population Baseline (`docs/research/radar-us-listed-coverage/coverage_capacity_baseline.json`).
- **Population Dimensions:** 16,448 distinct multi-feed keys; ~5,871 provisional non-derivative operating common stocks.
- **Examined Diagnostic Universe:** 14 operating companies:
  - 10 comparison names from the prior expansion queue: `KLAC`, `LULU`, `EME`, `VRT`, `ANET`, `CDNS`, `PWR`, `SNPS`, `NOW`, `ELF`.
  - 4 newly surfaced names from independent evidence routes: `MPWR`, `ODFL`, `HUBB`, `FICO`.

### Candidate-Generation Routes Tested
1. **Route 1: Company Financial & Operating Resilience (Routes A/B):** Screened for self-funding cash conversion (FCF/CFO > 90%), ROIC > 20%, and zero net debt.
2. **Route 2: Industry Physical / Power / Thermal Constraint (Route C):** Identified inelastic physical bottlenecks (substation transformer shortages, 100kW+ rack liquid cooling).
3. **Route 3: Customer / Supplier Value-Chain Tracing (Route E):** Traced hyperscaler capex line items to critical tier-1/tier-2 component enablers.

---

## 3. Informational Value & Cross-Candidate Findings

### A. Did any route surface a useful candidate absent from the queue?
**Yes, decisively.** Route 3 (Customer/Supplier Tracing) and Route 1 (Financial Resilience) surfaced **`MPWR` (Monolithic Power Systems)**:
- Both `VRT` and `MPWR` address the identical physical bottleneck: the severe power and thermal dissipation crisis in high-density AI data centers.
- `VRT` addresses it via bulky, capital-intensive facility infrastructure (CDUs, chillers, manifolds) encumbered by **$3.1B in funded debt**, **$184M in annual interest expense**, and modest ~11–13% ROIC.
- `MPWR` addresses it directly at the silicon packaging level (48V-to-core monolithic multiphase power management ICs) with **zero funded debt**, **$1.3B in liquid cash**, **27.0% GAAP operating margin**, **>25% ROIC**, and **proprietary BCD process IP**.
- *Takeaway:* Tracing supplier value chains beyond the obvious brand shortlist yielded a candidate with fundamentally superior value-capture dynamics.

### B. Did the process reveal material questions or change known priorities?
**Yes.**
1. **`VRT` (Priority Downgraded to `DEFERRED_WATCH`):** The pure industry-led theme made Vertiv appear to be an unassailable AI infrastructure winner. However, rigorous examination of Form 10-K Note 6 (Debt) revealed that interest obligations consume >20% of GAAP operating profit, while goodwill and intangibles comprise 49.2% of assets. Vertiv was definitively disqualified from Quality Watch consideration under Pillar B (Financial Resilience).
2. **`ANET` (Customer Concentration Surfaced):** Tracing hyperscaler relationships revealed that Meta Platforms (18%) and Microsoft (21%) represent **39% of total revenue**. While Arista's balance sheet is pristine ($6B+ cash, zero debt), this extreme customer concentration prevents public Quality Watch publication until customer diversification or margin preservation is proven.

### C. Did theme-led investigation add value beyond company-led evidence?
**Yes, but only as a dual-filter.** Theme-led investigation (Route 2) successfully directed attention to the high-density power/thermal constraint, which is where real capital expenditure is flowing. However, theme-led discovery alone is prone to narrative seduction; it surfaced over-leveraged equipment manufacturers. Only when coupled with company-led value-capture scrutiny (Route 1) did the discovery engine successfully separate durable compounders from leveraged balance sheets.

### D. Which signals produced weak or misleading candidates?
- **Misleading Signal:** "Broad AI Infrastructure Demand" / "Hyperscaler Capex Beneficiary." This narrative repeatedly surfaced companies with weak capital structures (`VRT`, `PWR`) or extreme dilution/capex burn (`IREN`, `CORZ`). Top-line demand is an insufficient condition for equity value creation.
- **Weak Signal:** "Consumer Brand Deceleration" (`LULU`, `ELF`). Retail turnarounds lacking a structural industrial or regulatory constraint proved uninformative for the Quality Watch compounder mandate.

---

## 4. Measured Effort & Process Efficiency

| Dimension | Measured Value | Tracking Status | Notes |
| :--- | :--- | :--- | :--- |
| **Initial Full-Queue Screening (14 Companies)** | ~2.5 hours | Measured (Current Task) | Cross-route screening, SEC EDGAR filing retrieval, and triage matrix authoring. |
| **Deep Investigation Preparation (3 Candidates)** | ~2.0 hours | Measured (Current Task) | Detailed 10-K/10-Q note extraction, debt bridge reconciliation, and opportunity ledger entry. |
| **Total Task Execution Effort** | ~4.5 hours | Measured (Current Task) | Focused analyst/engineering time for complete pilot delivery. |
| **Quarterly Monitoring Upkeep per Candidate** | ~30–45 minutes | Projected / Provisional | Estimated time to review 10-Q filing against observable assessment-change criteria. |
| **Historical Multi-Cycle Maintenance Effort** | N/A | **Explicitly Unmeasured** | No multi-quarter historical maintenance data exists yet; will be measured in subsequent cycles. |

---

## 5. Verification & Governance Audit Trail

### Verification Performed
1. **Automated Verification Script (`scripts/research/verify_discovery_pilot.py`):**
   - Verified schema completeness of `discovery_opportunity_ledger.json`.
   - Verified all five dimensional classifications across `ANET`, `VRT`, and `MPWR`.
   - Verified that all internal records enforce `public_publication_eligibility: INTERNAL_PILOT_RECORD_NOT_ELIGIBLE`.
   - Verified primary Form 10-K citations, accession numbers, and filing dates.
2. **Automated Unit Test Suite (`tests/test_evidence_led_discovery_pilot.py`):**
   - Executed via `python -m unittest` — 4/4 tests passed cleanly in 0.129s.
3. **Immutability & Safety Invariants Verified:**
   - `published_watch_manifest.json`: Exactly 6 authorized symbols (`FIX, DUOL, ULTA, MEDP, ISRG, DECK`). Zero mutations.
   - `odp05_discovery_ledger.json`: Canonical SHA-256 (`a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d`) verified intact.
   - Frontend Isolation: Verified that zero files in `frontend/` import or reference discovery pilot records.

---

## 6. Next Bounded Action Supported by Evidence

Under the **`CONTINUE`** decision, the supported next steps are:
1. **Deliver Pilot Deliverables:** Push feature branch `feat/research-evidence-led-discovery-pilot` and open a Draft PR on `daakara/finance`.
2. **Synchronize GitHub Issue #43:** Update issue #43 with delivery links, evaluation results, and the binding `CONTINUE` decision.
3. **Queue Candidate for Future Watch Assessment:** In a separate, subsequent cycle, subject `MPWR` and `ANET` to the canonical Quality Watch audit when valuation multiples compress or customer concentration resolves.
4. **Stopping Rule Enforcement:** Cease active execution upon Draft PR delivery and issue update. Do not prematurely promote candidates to the public surface.
