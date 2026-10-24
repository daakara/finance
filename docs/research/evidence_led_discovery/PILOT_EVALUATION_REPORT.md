# ARX Terminal — Evidence-Led Discovery Pilot: Evaluation Report (Reconciled)

**Document ID:** `ARX-DISCOVERY-PILOT-EVALUATION-REPORT-V3`  
**Governing Authority:** [daakara/finance#43](https://github.com/daakara/finance/issues/43) (`BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043`)  
**Methodological Context:** ARX Closure Model v2 (Decision-First, Minimum Sufficient Evidence)  
**Evaluation Date:** October 22, 2026  
**Information Cutoff Date:** October 22, 2026 (Audited SEC EDGAR Form 10-K FY2025 and Form 10-Q 1H 2026 filings)  
**Process Recommendation:** **`CONTINUE` (Qualified Recommendation for Further Bounded Experiment)**  

---

## 1. Executive Summary & Process Recommendation

### Process Recommendation: **`CONTINUE`**
> **Recommendation:** Proceed with another bounded discovery experiment, subject to separate product owner authorization.  
> The pilot demonstrated that structured, decision-focused research successfully uncovers critical differences in capital structure, cash generation, customer concentration, and architectural layers across infrastructure providers.

### Epistemic Qualifications & Governance Boundaries
- **Not Proof of Discovery Skill:** A `CONTINUE` recommendation reflects that the structured research format helped surface decision-relevant differences; it does not prove algorithmic or proprietary discovery skill.
- **Not Binding Authorization:** A `CONTINUE` recommendation is an evaluative finding, not an automatic authorization to begin another cycle or deploy resources.
- **Does Not Start Another Cycle:** No further discovery tasks or dossier drafting will commence without explicit, separate Product Owner instructions.
- **Zero Public Surface Mutation:** Published Quality-Watch collection remains frozen at exactly six approved assets: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK`.

---

## 2. Seven-Dimensional Pilot Evaluation

### 1. Research Structure and Isolation
- **Evaluation:** **SUCCESSFUL.**
- The pilot strictly isolates internal exploratory records (`docs/research/evidence_led_discovery/`) from the public research surface.
- All records enforce `public_publication_eligibility: INTERNAL_PILOT_RECORD_NOT_ELIGIBLE`.
- Codebase grep audits confirm zero import or bundle references in `frontend/`.
- The canonical published manifest (`published_watch_manifest.json`) and baseline ledger (`odp05_discovery_ledger.json`) remained completely unmutated.

### 2. Source Accuracy
- **Evaluation:** **RECONCILED TO AUDITED PRIMARY SOURCES.**
- Initial drafts relied on stale 2023/2024 citations and erroneous assumptions (e.g., Ansys merger pending, Vertiv permanently distressed).
- Reconciliation updated all 14 triage entries to October 2026 information cutoff, citing audited Form 10-K (FY2025) and interim Form 10-Q (1H 2026) filings with exact accession numbers and table locators.
- Factual errors were corrected: SNPS merger completion date confirmed as July 17, 2025 via Form 8-K; Vertiv deleveraging from $2.8B net debt to $129M verified via Form 10-K/10-Q; Arista customer concentration verified at 42% across two customers (Customer A 26%, Customer B 16%).

### 3. Useful Questions Surfaced
- **Evaluation:** **HIGH INFORMATIONAL VALUE.**
- The pilot successfully moved research inquiry away from generic operational summaries toward decisive uncertainties:
  - *ANET:* Can high GAAP operating margins (>40%) withstand 42% customer concentration if hyperscalers push for white-box or custom silicon solutions?
  - *VRT:* Can Vertiv maintain ~18% operating margins as global electrical conglomerates (Schneider, Eaton) expand modular liquid cooling capacity?
  - *MPWR:* Does proprietary BCD packaging protect silicon-level 48V power stages against multi-sourcing policies from major GPU and ASIC customers?

### 4. Defensible Priority Changes
- **Evaluation:** **SUPPORTED BY EVIDENCE.**
- **`VRT`:** Promoted from invalid automatic disqualification to `ACTIVE_QUALITY_WATCH_CANDIDATE` based on verified $1.89B FCF in FY25, net debt reduction to $129M, and interest expense declining to 4.7% of operating income.
- **`ANET`:** Maintained in internal monitoring queue (`PRIORITIZED_FOR_INTERNAL_MONITORING`) rather than promoted to public watch, appropriately recognizing that 42% customer concentration warrants observation.
- **`MPWR`:** Retained in internal monitoring queue (`PRIORITIZED_FOR_INTERNAL_MONITORING`), recognizing its silicon-level role as complementary to facility infrastructure without claiming inherent investment superiority.
- **`SNPS`:** Deferred based on post-merger integration complexity and $9.0B long-term debt burden, rather than a false "pending" status.
- **Contextual Deferrals:** Removed unsupported blanket disqualifiers for consumer exposure (`LULU`) or single-digit contractor margins (`PWR`, `EME`), deferring them contextually.

### 5. Candidate-Generation Effectiveness
- **Evaluation:** **INCONCLUSIVE / NOT ESTABLISHED.**
- The 14 evaluated companies represent an **exploratory manually selected comparison cohort**, not an automated full-market scan.
- The pilot did not test automated screening algorithms across the broader US-listed operating equity universe (~5,871 stocks).
- A future separately authorized discovery experiment is required to evaluate automated candidate generation, with prospective logging of search queries, timestamps, and filter cutoffs.

### 6. Effort Measurement
- **Evaluation:** **ESTIMATED / UNMEASURED.**
- Active research effort is estimated at approximately ~4.5 hours for initial cohort review and deep record preparation, with prospective maintenance estimated at ~30–45 minutes per candidate quarterly.
- No automated timecards or stopwatch logs were recorded; effort claims are explicitly disclosed as estimates rather than empirical measurements.

### 7. Investment Performance
- **Evaluation:** **OUT OF SCOPE / NOT CLAIMED.**
- This pilot evaluates research process and evidentiary discipline. It does not measure stock returns, claim predictive alpha, or promise outperformance.

---

## 3. Comparative Synthesis: Data Center Infrastructure Candidates

The table below contrasts the three deeper investigations using audited FY2025 and 1H 2026 SEC filings:

| Dimension | Arista Networks (`ANET`) | Vertiv Holdings (`VRT`) | Monolithic Power Systems (`MPWR`) |
| :--- | :--- | :--- | :--- |
| **Architectural Layer** | Network Fabric (800G/1.6T Ethernet) | Facility-Level Power & Modular Cooling | Board-Level Silicon Power Management (48V ICs) |
| **System Relationship** | Data Center Interconnect Fabric | Data Center Facility Thermal Management | Server Accelerator Board Power Conversion |
| **FY2025 GAAP Revenue** | $9,005.7M | $10,229.9M | $2,790.5M |
| **FY2025 GAAP OpMargin** | **42.8%** ($3,856.1M) | **17.9%** ($1,829.7M) | **26.1%** ($728.6M) |
| **1H 2026 GAAP OpMargin** | **44.1%** ($2,535.8M) | **18.2%** ($1,078.0M) | **30.5%** ($545.0M) |
| **FY2025 Free Cash Flow** | **$4,261.9M** (CFO $4,371.9M - Capex $110.0M) | **$1,893.8M** (CFO $2,113.8M - Capex $220.0M) | **$666.2M** (CFO $838.2M - Capex $172.0M) |
| **FCF Conversion (FCF/CFO)** | **97.5%** | **89.6%** | **79.5%** |
| **Funded Long-Term Debt** | **$0.0M** | $2,939.8M | **$0.0M** |
| **Cash & Liquid Investments** | $2,290.2M cash (plus ST investments) | **$2,810.6M cash** | $1,005.6M cash/ST investments |
| **Net Funded Debt** | **Net Cash** | **$129.2M (Near Net Cash)** | **Net Cash** |
| **Primary Operational Uncertainty** | **42% Customer Concentration (Two Customers: 26% and 16%)** | Competitive Supply Ramp & Project Execution | GPU Vendor Multi-Sourcing & Analog Pricing |
| **Triage Classification** | `PRIORITIZED_FOR_INTERNAL_MONITORING` | `ACTIVE_QUALITY_WATCH_CANDIDATE` | `PRIORITIZED_FOR_INTERNAL_MONITORING` |

---

## 4. Verification & Release Boundary Attestations

```ini
[ARX_RESEARCH_SAFETY_INVARIANTS]
PUBLISHED_COLLECTION = UNMUTED (FIX, DUOL, ULTA, MEDP, ISRG, DECK - exactly 6 symbols)
BASELINE_LEDGER_SHA256 = a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d (INTACT)
SCANNER_EXECUTION = NOT_ACTIVATED
UNIVERSE_PUBLICATION = NOT_PERFORMED
FRONTEND_BUNDLE_IMPORTS = ZERO_CROSS_BOUNDARY_LEAKS
```

---

## 5. Next Steps

1. Maintain Draft PR [#45](https://github.com/daakara/finance/pull/45) as the delivery record for this corrected pilot and standardization task.
2. In a future separately authorized cycle, evaluate `VRT` formally against the six Quality-Watch pillars using its deleveraged financial baseline.
3. If an automated candidate-generation experiment is authorized in the future, prospectively record search logs and timestamps.
4. Cease active execution under the stopping rule of ARX Closure Model v2.
