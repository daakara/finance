# ARX Terminal — Candidate Priority Reassessment After Discovery Cycle 2

**Document ID:** `ARX-CANDIDATE-PRIORITY-REASSESSMENT-V1`  
**Effective Date:** October 22, 2026 (Reconciled & Corrected)  
**Governing Authority:** Product Owner Authorization — Candidate Priority Reassessment after Discovery Cycle 2  
**Governing Standard:** `ARX-DECISION-RESEARCH-PROCESS-V1` & ARX Closure Model v2  
**Applicability:** Internal Research Experiment Only (Non-Production / Non-Public)  

---

## 1. Executive Summary & Epistemic Boundaries

Following the mainline adoption of Discovery Cycle 2 via PR #46 (Commit `404a566734306a9e51b442c10dfe5fe028e7d19f`), this document resolves outstanding material questions regarding internal candidate prioritization following Discovery Cycle 2.

```mermaid
flowchart TD
    subgraph EpistemicDistinctions["Mandatory Four-Dimensional Evaluation Matrix"]
        A["<b>A. Business Research Relevance</b><br/>Is there a consequential economic or business question here?"]
        B["<b>B. Research Urgency</b><br/>Why spend analyst hours investigating this company NOW vs others?"]
        C["<b>C. Market / Setup Readiness</b><br/>Do technical and valuation conditions create a timely decision?"]
        D["<b>D. Public Publication Eligibility</b><br/>Does the candidate satisfy Quality Watch Selection Standards?"]
    end
    A -.->|Does NOT Imply| B
    B -.->|Does NOT Imply| C
    C -.->|Does NOT Imply| D
```

### Mandatory Epistemic Boundaries
In accordance with `ARX-DECISION-RESEARCH-PROCESS-V1`, research prioritization requires strict separation across four distinct dimensions:
1. **Business Research Relevance:** Identifies whether a consequential economic mechanism or structural business transformation exists.
2. **Research Urgency:** Evaluates whether immediate analyst hours are justified relative to alternative internal tasks.
3. **Market / Setup Readiness:** Examines whether observable market pricing, technical structure, or valuation conditions make research actionable.
4. **Public Publication Eligibility:** Evaluates whether the candidate meets the strict, published Quality Watch Selection Standard.

A positive finding in one dimension does not establish the others. Business relevance does not create research urgency, nor does market volatility establish an attractive setup.

---

## 2. Reconciled TMDX Financial Analysis

### Primary-Source SEC Filing Authorities
- **Form 10-K FY2025:** Period ended December 31, 2025, filed February 24, 2026 (CIK: `0001756262`, Accession: `0001193125-26-067032`, [SEC EDGAR URL](https://www.sec.gov/Archives/edgar/data/1756262/000119312526067032/tmdx-20251231.htm)).
- **Form 10-Q Q2 2026:** Period ended June 30, 2026, filed August 4, 2026 (CIK: `0001756262`, Accession: `0001193125-26-332843`, [SEC EDGAR URL](https://www.sec.gov/Archives/edgar/data/1756262/000119312526332843/tmdx-20260630.htm)).
- **Form 8-K Earnings Release (Ex 99.1):** Filed August 4, 2026 (CIK: `0001756262`, Accession: `0001193125-26-332767`, [SEC EDGAR URL](https://www.sec.gov/Archives/edgar/data/1756262/000119312526332767/tmdx-ex99_1.htm)).

### Reconciled Statement of Operations & Cash Flow Comparison

The following table presents verified reported metrics from TransMedics Group, Inc.'s primary SEC filings across comparable annual and interim periods:

| Metric (in USD Millions) | FY2025 Full Year (Audited) | 1H 2025 (Six Months Ended Jun 30, 2025) | 1H 2026 (Six Months Ended Jun 30, 2026) | 1H YoY Change (%) | Q2 2025 (Three Months Ended Jun 30, 2025) | Q2 2026 (Three Months Ended Jun 30, 2026) | Q2 YoY Change (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Net Product Revenue** | $372.401M | $184.334M | $219.130M | +18.88% | $96.100M | $111.158M | +15.67% |
| **Service Revenue** | $233.093M | $116.573M | $144.751M | +24.17% | $61.270M | $78.790M | +28.59% |
| **Total Revenue** | **$605.494M** | **$300.907M** | **$363.881M** | **+20.93%** | **$157.370M** | **$189.948M** | **+20.70%** |
| **Cost of Revenue** | $242.688M | $116.090M | $149.522M | +28.80% | $60.781M | $76.750M | +26.27% |
| **Gross Profit** | **$362.806M** | **$184.817M** | **$214.359M** | **+15.98%** | **$96.589M** | **$113.198M** | **+17.20%** |
| **Gross Margin (%)** | 59.92% | 61.42% | 58.91% | -251 bps | 61.38% | 59.59% | -179 bps |
| **Research & Development (R&D)** | $69.1M | $33.094M | $56.511M | +70.76% | $15.934M | $31.632M | +98.52% |
| **Selling, General & Admin (SG&A)** | $185.1M | $87.713M | $120.815M | +37.74% | $44.088M | $57.830M | +31.17% |
| **Total Operating Expenses** | **$254.223M** | **$120.807M** | **$177.326M** | **+46.78%** | **$60.022M** | **$89.462M** | **+49.05%** |
| **Operating Income (Loss)** | **$108.583M** | **$64.010M** | **$37.033M** | **-42.14%** | **$36.567M** | **$23.736M** | **-35.09%** |
| **Operating Margin (%)** | 17.93% | 21.27% | 10.18% | -1,109 bps | 23.24% | 12.50% | -1,074 bps |
| **Total Other Expense, Net** | -$1.1M | -$1.152M | -$9.143M | N/A | -$0.385M | -$4.331M | N/A |
| **Provision for Income Taxes** | +$82.769M* | -$2.269M | -$5.893M | N/A | -$1.275M | -$4.723M | N/A |
| **GAAP Net Income** | **$190.291M\*** | **$60.589M** | **$21.997M** | **-63.70%** | **$34.907M** | **$14.682M** | **-57.94%** |
| **Cash from Operations (CFO)** | **$192.840M** | **$88.754M** | **$41.842M** | **-52.86%** | — | — | — |
| **Cash Purchases of PP&E** | **$59.251M** | **$36.113M** | **$43.749M** | **+21.14%** | — | — | — |
| **Simple Cash FCF (CFO - Cash PP&E)**| **+$133.589M** | **+$52.641M** | **-$1.907M** | **Deficit** | — | — | — |
| **Noncash Aircraft Lease Additions**| $0.0M | $0.0M | **$340.040M** | N/A | — | — | — |

*\*Includes a nonrecurring income tax benefit in Q4 2025 primarily resulting from a $103.3M valuation allowance release on deferred tax assets.*

### Reconciled Balance Sheet & Capital Structure Comparison

The table below reconciles cash and debt components, separating principal from carrying value and isolating finance lease liabilities:

| Balance Sheet Item (in USD Millions) | December 31, 2025 (Audited 10-K) | June 30, 2026 (Unaudited 10-Q) | Reported Change | Classification / Note Locator |
| :--- | :---: | :---: | :---: | :--- |
| **Cash and Cash Equivalents** | **$488.366M** | **$472.675M** | -$15.691M | Unrestricted cash reserves (Item 1, p. 1) |
| **Restricted Cash** | $0.500M | $18.438M | +$17.938M | Escrow / restricted collateral |
| **Total Cash, Equivalents & Restricted Cash**| **$488.866M** | **$491.113M** | +$2.247M | Total cash reported on cash-flow statement (Item 1, p. 5) |
| **1.50% Convertible Notes (Principal)** | $460.000M | $460.000M | $0.0M | Face value due 2028 (Note 7, p. 11) |
| **1.50% Convertible Notes (Carrying Value)**| $452.804M | $454.260M | +$1.456M | Net of unamortized issuance costs (Item 1, p. 1) |
| **Other Borrowings (Principal)** | $60.000M | $60.000M | $0.0M | $20.0M current / $40.0M noncurrent (Note 7, p. 11) |
| **Other Borrowings (Carrying Value)** | $59.587M | $59.743M | +$0.156M | $20.0M current / $39.743M noncurrent (Item 1, p. 1) |
| **Total Borrowings Principal (Excl. Leases)**| **$520.000M** | **$520.000M** | **$0.0M** | Total contractual borrowing principal |
| **Total Borrowings Carrying Value (Excl. Leases)**| **$512.391M** | **$514.003M** | **+$1.612M** | Total carrying debt on balance sheet |
| **Finance Lease Liabilities (Noncurrent)** | **$0.0M** | **$347.660M** | **+$347.660M** | Obligations for right-of-use aircraft (Note 7, p. 13) |
| **Total Obligations Principal (Incl. Leases)**| **$520.000M** | **$867.660M** | **+$347.660M** | Principal debt + finance lease liability |
| **Total Obligations Carrying Value (Incl. Leases)**| **$512.391M** | **$861.663M** | **+$349.272M** | Carrying debt + finance lease liability |
| **Net Debt: Principal Excl. Leases** | **+$31.634M** | **+$47.325M** | +$15.691M | Total Borrowings Principal less Cash ($472.675M) |
| **Net Debt: Carrying Value Excl. Leases**| **+$24.025M** | **+$41.328M** | +$17.303M | Total Borrowings Carrying Value less Cash |
| **Net Debt: Principal Incl. Leases** | **+$31.634M** | **+$394.985M** | **+$363.351M** | Total Obligations Principal less Cash |
| **Net Debt: Carrying Value Incl. Leases**| **+$24.025M** | **+$388.988M** | **+$364.963M** | Total Obligations Carrying Value less Cash |

*Note on Sign Convention:* Under the standard corporate finance convention (`Net Debt = Debt Obligations - Cash`), positive net debt indicates that obligations exceed cash reserves.

---

## 3. Causal Attribution & Balanced Assessment

To avoid replacing an uncritical bullish thesis with an equally unsupported bearish narrative, the observed financial trajectory is evaluated through four structured steps:

### A. Reported Financial Changes
1. **Revenue Growth Deceleration:** Consolidated revenue grew +20.93% YoY in 1H 2026 ($363.881M vs $300.907M in 1H 2025), down from +37.1% YoY in FY2025.
2. **Operating Expense Expansion:** Operating expenses rose +46.78% YoY in 1H 2026 ($177.326M vs $120.807M). In Q2 2026, operating expenses grew +49.05% YoY ($89.462M vs $60.022M), driven by R&D increasing +98.52% ($31.632M vs $15.934M) and SG&A increasing +31.17% ($57.830M vs $44.088M).
3. **Margin Compression:** Operating margin contracted from 21.27% in 1H 2025 to 10.18% in 1H 2026 (-1,109 bps). In Q2 2026, operating margin fell from 23.24% to 12.50% (-1,074 bps).
4. **Interim Cash Flow Trajectory:** Cash provided by operating activities dropped -52.86% YoY in 1H 2026 to $41.842M (vs $88.754M in 1H 2025). Simple cash FCF (CFO minus cash PP&E purchases of $43.749M) was negative -$1.907M, compared to positive +$52.641M in 1H 2025.
5. **Financing Commitments:** TMDX recognized $340.040M in noncash finance lease additions for right-of-use aircraft, establishing a $347.660M noncurrent lease liability as of June 30, 2026.

### B. Management's Disclosed Explanation (Commentary)
In the Q2 2026 earnings release (Form 8-K Ex 99.1) and Form 10-Q MD&A, management attributed the operating expense increase primarily to:
- Planned investment in **OCS Kidney** development and pre-commercial infrastructure.
- Investment in the next-generation **Gen 3.0** perfusion platform.
- Expansion of **clinical programs**, including the ENHANCE Part B and DENOVO clinical trials.
- Selected corporate and support infrastructure investments to support procedural scale.
- Management noted that stock-based compensation within operating expenses was $8.2M in Q2 2026 compared to $9.0M in Q2 2025.

### C. Plausible Alternative Explanations
While management identifies pipeline and trial investments as the primary expense driver, operating a dedicated nationwide aviation logistics network involves asset-heavy transportation economics:
- Aviation operating costs, including pilot compensation, fuel price volatility, routine maintenance, and FAA safety compliance, expand the company's fixed operating cost structure.
- Adding aircraft under finance leases creates multi-year debt service and maintenance obligations.
- If procedure growth decelerates, fixed flight overhead could create negative operating leverage, constraining margin expansion.

### D. Scope & Limitations of Forthcoming Review Trigger
The forthcoming publication of Form 10-Q for Q3 2026 serves as a bounded review trigger. Disclosures in the filing and earnings materials may help assess:
- **Operating-Expense Trajectory:** Whether R&D and SG&A growth rates moderate relative to top-line growth.
- **Cash Conversion:** Whether cash generated from operations rebounds relative to cash capital expenditures.
- **Lease Obligations:** Whether aircraft lease additions and commitments stabilize.
- **Productive Investment vs. Operational Drag:** Initial qualitative commentary or segment disclosures regarding the European PAD Aviation integration (completed July 1, 2026) and OCS clinical pipeline progress.

**Explicit Limitations:**
- Forthcoming quarterly filings may not disclose flight-hour utilization, route-level economics, or granular per-mission logistics costs.
- The filing is not guaranteed to conclusively resolve the tension between management's reinvestment explanation and the aviation friction hypothesis.
- If decisive operational evidence remains undisclosed following the event, recording that limitation and stopping research is a valid, expected outcome under the ARX Closure Model.

### Live Hypotheses Maintained
- **Hypothesis 1 (Productive Reinvestment):** The interim margin compression reflects a planned, front-loaded investment cycle in OCS Kidney, Gen 3.0, and clinical trials that will expand addressable organ indications and restore operating leverage.
- **Hypothesis 2 (Structural Capital Intensity):** Vertically integrating air logistics introduces transportation-sector cost dynamics and fixed lease debt commitments that structurally dampen long-term cash conversion per share.

---

## 4. Technical & Valuation Evidence Standards

### Technical State Qualification
- **Technical State:** In the absence of a verified, traceable market data provider receipt or auditable calculation archive for daily trading sessions in the current environment, the technical setup (including closing price levels, trailing moving average benchmarks, moving average slopes, relative strength, and volume accumulation/distribution) is formally classified as **`TECHNICAL_STATE_UNRESOLVED`**.
- Subjective interpretations regarding "why market participants sold" or assertions of "institutional distribution" or "failed bases" are withdrawn.
- The event-based research disposition for candidate companies does not depend on unverified technical numbers.

### Valuation State Qualification
- In the absence of live, primary-source broker consensus feeds and explicit dated discounted cash flow models with published terminal assumptions, forward P/E and multiple labels ("fair," "stretched," "compressed") are unverified.
- All candidate valuation assessments are classified as **`VALUATION_EXPECTATION_GAP_UNRESOLVED`** or **`VALUATION_NOT_ASSESSED`**.

---

## 5. Disposition & Concrete Action for Eaton (`ETN`)

Eaton Corporation plc (`ETN`, CIK: `0001551182`) was surfaced in Cycle 2 based on its market leadership in electrical distribution and high-voltage substation switchgear.

### Primary Disclosures Regarding Backlog (Form 10-K FY2025, Note 3 & Item 7)
- Eaton reports total backlog of **$19.8 billion** as of December 31, 2025 (with approximately 69% targeted for delivery within 12 months).
- Electrical Americas backlog was **$13,246 million** (representing +31% total growth and +19% organic growth YoY, with an organic book-to-bill ratio of 1.2).
- Note 3 defines backlog as unsatisfied or partially satisfied obligations where "only the amount of orders to which customers are firmly committed are included."

### Bounded Investigation Result: `NOT_DISCLOSED / UNRESOLVED`
A focused audit of Eaton's Form 10-K FY2025 reveals that **Eaton does not disclose the detailed percentage breakdown of Electrical Americas backlog between datacenter high-voltage switchgear versus commercial construction**, nor does it publish customer-specific contract cancellation penalty clauses.

### Concrete Disposition: `REVIEW_ON_EVENT`
Because the detailed sub-segment backlog composition cannot be extracted from accessible primary SEC filings prior to the next quarterly release, maintaining ETN as an urgent active investigation before earnings is unjustified.
- **Action:** Transition ETN to **`REVIEW_ON_EVENT`**.
- **Review Trigger:** Publication of SEC Form 10-Q for Q3 2026 (expected early November 2026).
- **Evidence to Inspect:** Electrical Americas quarterly book-to-bill ratio, organic order growth, and segment operating profit margin resilience.

---

## 6. Proportionate Internal Research Queue

The compact matrix below establishes the internal research disposition across all 14 examined and benchmarked candidates, avoiding arbitrary numerical quotas or artificial rank ordering:

| Candidate Symbol | Company Name | Decision-Relevant Supported Evidence | Important Unresolved Evidence | Internal Disposition | Specific Review Trigger / Action | Relative Allocation Rationale |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`ETN`** | Eaton Corporation plc | Leader in electrical distribution; Electrical Americas segment margin 29.9% ($3,972M); 79.5% FCF conversion; 'A-' credit rating. | Detailed backlog split (datacenter substations vs commercial construction) is `NOT_DISCLOSED`. | **`REVIEW_ON_EVENT`** | Form 10-Q Q3 2026 (Nov 2026) | Strong upstream positioning, but detailed backlog mix cannot be resolved prior to quarterly disclosure. |
| **`TMDX`** | TransMedics Group, Inc. | FY25 revenue $605.5M (Net product $372.4M / Service $233.1M); 1H26 revenue +20.9%; 1H26 operating income -42.1%; 1H26 simple cash FCF -$1.9M; $347.7M aircraft lease debt. | Productive clinical investment (OCS Kidney/Gen 3.0) vs fixed aviation logistics drag. | **`REVIEW_ON_EVENT`** | Form 10-Q Q3 2026 (Nov 2026) | Research effort deferred pending observable OpEx trajectory, lease commitments, and cash conversion. |
| **`PODD`** | Insulet Corporation | FY25 revenue $2.71B (+30.7%); operating income $473.8M (17.5% margin); >85% recurring pharmacy channel revenue. | International Omnipod 5 commercial adoption rate and gross margin sustainability. | **`REVIEW_ON_EVENT`** | Form 10-Q Q3 2026 (Nov 2026) | High recurring consumable revenue; review on Q3 international expansion metrics. |
| **`CEG`** | Constellation Energy Corp | Largest US merchant nuclear fleet; 20-year Microsoft PPA for Three Mile Island Unit 1 restart. | FERC regulatory tariff ruling on behind-the-meter datacenter colocation dockets. | **`REVIEW_ON_EVENT`** | FERC Colocation Docket Ruling | Strategic baseload power asset, but binary regulatory cost-allocation decision dominates near-term modeling. |
| **`TDG`** | TransDigm Group, Inc. | Proprietary commercial/military aerospace aftermarket parts (>45% EBITDA margin). | High financial leverage ($22B+ funded debt) sensitivity to debt refinancing rates. | **`INACTIVE`** | Form 10-K FY2026 (Nov 2026) | Pricing power benchmark; elevated leverage warrants monitoring rather than public research. |
| **`HUBB`** | Hubbell Incorporated | Consistent utility grid transmission and distribution components; ~20% operating margin. | Lower direct leverage to hyperscale high-voltage datacenter substations than Eaton. | **`INACTIVE`** | Relative multiple compression | Quality utility equipment play; secondary priority behind Eaton on datacenter power exposure. |
| **`NVR`** | NVR, Inc. | High ROIC (>35%) land-light homebuilding option model; zero funded debt. | Sensitivity of homebuilding absorption and lot option margins to elevated mortgage rates. | **`INACTIVE`** | Macro mortgage rate inflection | Disciplined capital allocator, but structural thesis is subordinate to macro interest rate cycle. |
| **`APP`** | AppLovin Corporation | Cash-generative mobile advertising recommendation engine (AXON 2.0). | Vulnerability to Apple / Google mobile app store privacy and attribution policy changes. | **`INACTIVE`** | Mobile platform policy review | Strong reported cash flows offset by extreme platform counterparty dependence and governance opacity. |
| **`LNTH`** | Lantheus Holdings, Inc. | High cash generation from Pylarify PSMA PET prostate cancer imaging diagnostic. | Multi-year terminal value durability given single-product concentration and generic entry. | **`INACTIVE`** | Patent litigation resolution | Diagnostic cash flow is vulnerable to medium-term generic competition; terminal certainty low. |
| **`POWI`** | Power Integrations, Inc. | High-voltage Gallium Nitride (GaN) ICs for datacenter power supplies and industrial power. | GaN datacenter growth overshadowed by consumer and industrial channel inventory overhang. | **`INACTIVE`** | Industrial channel inventory cycle | High technology quality, but consolidated results remain dominated by broader cyclical headwinds. |
| **`GEV`** | GE Vernova Inc. | Global power generation and grid electrification equipment post-spin from General Electric. | Magnitude and duration of legacy offshore wind contract loss provisions. | **`INACTIVE`** | Cessation of wind contract losses | Grid positioning offset by legacy contract liabilities and restructuring volatility. |
| **`MEDP`** | Medpace Holdings, Inc. | Published Quality Watch asset; full-service biotech CRO; zero funded debt; >95% FCF conversion. | Quarterly trial cancellation rates and biotech funding trends. | **`REFERENCE_BENCHMARK`** | Quarterly Watch Maintenance | Baseline quality benchmark to calibrate CRO and healthcare capital return standards. |
| **`PWR`** | Quanta Services, Inc. | Electric utility and renewable infrastructure contractor from initial expansion cohort. | Execution risks on fixed-price projects and lower operating margins (8-9%). | **`REFERENCE_BENCHMARK`** | Quarterly Peer Tracking | Infrastructure contractor benchmark to contrast against equipment manufacturers. |
| **`VRT`** | Vertiv Holdings Co | White-space datacenter liquid cooling and power distribution benchmark from Cycle 1. | Customer concentration among hyperscalers and higher financial leverage than Eaton. | **`REFERENCE_BENCHMARK`** | Quarterly Peer Tracking | Thermal cooling benchmark to monitor datacenter infrastructure buildout trends. |

---

## 7. Governance, Invariant Adherence & Verification

- **System Invariants Maintained:**
  - Published watch collection remains exactly 6 approved symbols (`FIX, DUOL, ULTA, MEDP, ISRG, DECK`).
  - Production snapshots remain unchanged and unreferenced by internal records.
  - Frozen ODP-05 discovery ledger SHA-256 (`a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d`) is untouched.
  - Zero scanner activation, universe expansion, live ingestion, or frontend code mutations.
- **Backlog Tracking Reconciled:**
  - `docs/planning/ENGINEERING_BACKLOG.md` maintains `BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046` as `COMPLETED_AND_ADOPTED` and records `BACKLOG-RESEARCH-PRIORITY-REASSESSMENT-047` as `COMPLETED` (Delivered via Pull Request #48, Branch `feat/research-candidate-priority-reassessment`).
- **Cross-Reference Synchronization:**
  - Priority notice callouts in `DISCOVERY_OPPORTUNITY_RECORDS.md`, `SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md`, and `CYCLE_2_EVALUATION_REPORT.md` are synchronized to reflect this reconciled reassessment.
