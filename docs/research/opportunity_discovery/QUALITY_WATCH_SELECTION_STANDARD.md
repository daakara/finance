# ARX Terminal — Curated Quality Watch Selection Standard & Cohort Triage

**Standard ID:** `ARX-QUALITY-WATCH-STANDARD-V1`  
**Date:** October 21, 2026  
**Governing Context:** ARX Closure Model v2 — Product Owner Directive: *"Focus on top quality watch; rejected has no value in this and is a waste."*

---

## 1. Quality Watch Selection Standard

ARX Research operates strictly as a curated collection of high-conviction **Quality Watch** dossiers. The surface exists to help investors identify worthwhile businesses, understand why they deserve attention now, and track the specific observable conditions required for capital allocation.

A candidate qualifies for published Quality Watch status if and only if it satisfies all six core pillars:

### A. Credible Business-Quality Thesis
* Durable structural competitive advantages, high customer switching costs, network effects, or defensible scale moats substantiated by observable business metrics (e.g., ROIC, organic customer retention, order backlog, loyalty membership scale).
* Clear economic differentiation within its industry.

### B. Understandable Financial Resilience
* Transparent capital structure: funded financial debt, cash/ST liquidity, and lease commitments (ASC 842) are clearly distinguished.
* Dependable operating cash flows that adequately support operating obligations without speculative reliance on external capital markets.
* Financing needs and equity dilution exposure (including stock-based compensation) are honestly represented.

### C. Useful Reason to Watch Now
* A distinct valuation tension, economic inflection, or business development that makes continuous monitoring worthwhile.
* The reference price embeds a demanding or unanchored growth hurdle, or the asset trades above conservative fair value, creating an actionable watch corridor for market pullbacks or execution inflections.

### D. Observable Watch Condition
* Specific, quantifiable conditions that define an attractive entry window (e.g., pullback into a mid-cycle valuation corridor, comp-store acceleration >6%).
* Explicit, observable invalidation criteria (e.g., operating margin compression below 10.0%, 2 consecutive quarters of negative comps, customer churn) that formally invalidate the thesis.

### E. Balanced, Traceable Research
* Material financial baseline inputs and core qualitative assertions are reconciled to authoritative primary disclosures (SEC EDGAR Forms 10-K and 10-Q).
* Explicit distinction between verified historical facts, derived calculations, model-generated expectations, and forward-looking analyst uncertainties.

### F. Appropriate Valuation Treatment
* Single-stage or multi-stage Reverse-DCF calculations reproduce deterministically through the canonical valuation engine when economically applicable.
* Where standard DCF modeling is structurally inapplicable or unreliable (e.g., early-stage binary capital structures, transformative unclosed acquisitions, unquantified multi-year capex burn), numerical targets are **suppressed (Fail-Closed)** with documented conditions for resumption.

---

## 1.1 Decision-Focused Research Process Standard Integration

In accordance with Product Owner Directive (October 22, 2026), all ARX candidate evaluations, queue prioritization, and dossier maintenance prospectively adhere to the canonical **Decision-Focused Research Process Standard** ([`docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md`](file:///docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md)):

1. **Step A: Define the Decision:** Explicitly state the operational decision (Investigate Further, Prioritize Queue, Prepare Quality-WATCH, Maintain/Revise, Deeper Analysis) with analytical horizon (3–5 years provisional) and information cutoff.
2. **Step B: Identify Decisive Uncertainties:** Isolate the few questions most capable of altering the outcome; no universal or arbitrary risk checklists.
3. **Step C: Connect Outcomes to Consequences:** Trace operational scenarios to cash generation, capex (growth vs maintenance), debt service, dilution, and downside resilience. Standard reverse-DCF is optional and suppressed for unsuitable business models.
4. **Step D: Investigate Proportionately:** Reconcile facts to primary SEC filings; classify facts, management commentary, derived calculations, and analyst assumptions; actively seek counter-evidence.
5. **Step E: Make the Assessment:** Document supported conclusions, unresolved items, and priority changes (`No differentiated expectation established` and `Insufficient evidence` are valid outcomes).
6. **Step F: Set Review Triggers and Stop:** Specify observable falsification events; cease research immediately when the decision is supported or evidence is unobtainable.

### Prospective Application & Governance Invariants:
- **Prospective Application:** New research strictly applies this 6-step process. Existing published dossiers adopt it prospectively during their next material trigger reviews.
- **Completed Releases Closed:** Completed production releases (PR #39, PR #41) remain immutable and are **not** reopened solely for template compliance.
- **Evidence Proportionality:** Different research decisions require different depths of evidence.
- **Separation of Invariants:** Technical verification does not establish investment performance; WATCH is not BUY; publication readiness and investment attractiveness remain separate questions.
- **No Publication Quotas:** No fixed asset-count release quotas apply to research publication.

---

## 2. Comprehensive Cohort Triage & Preparation Priority Matrix

All original baseline records and expansion candidates evaluated under the Quality Watch standard:

| Symbol | Company Name | Sector & Moat Rationale | Specific Watch Condition / Catalyst | Evidence Readiness | Accounting / Modeling Complication | Decision | Concise Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **`FIX`** | Comfort Systems USA, Inc. | Modular HVAC & electrical infrastructure moat; hyperscale data center backlog; >28% ROIC. | Pullback toward base corridor ($1,050–$1,100) or persistent >14% operating margin. | Primary 10-K/10-Q audited in ODP-05 baseline. | $1.80B net cash; customer advance billings. | **Selected** | Original Published WATCH; top-tier execution, self-funding cash flow, and verified DCF hurdle (30.07% CAGR). |
| **`DUOL`** | Duolingo, Inc. | Gamified language learning app with organic viral flywheel; 73% gross margin; subscription ARR growth. | Pullback toward $125–$130 or evidence of sustained AI tier monetization. | Primary 10-K/10-Q audited in ODP-05 baseline. | Documented zero debt; SBC dilution monitoring. | **Selected** | Original Published WATCH; pristine balance sheet, strong product moat, self-funding organic cash flow. |
| **`IREN`** | Iris Energy Limited | Scaled next-generation data center footprint (>5 GW secured power pipeline) for AI cloud compute. | Demonstration of steady-state compute operating cash flow covering ongoing capex without equity dilution. | Primary 20-F/6-K in ODP-05 baseline. | $11.60B liabilities; $25B–$30B guided FY27 capex; ATM equity dilution (47.1M shares issued in FY26); DCF suppressed. | **Held** | Bounded check: Fails Pillar B (Financial Resilience). Extreme capital intensity ($25B-$30B capex) and dilutive external financing dependence mean it does not currently fit the quality compounder collection. Direct research URL displays neutral unavailable state; stock page remains 100% active. |
| **`CORZ`** | Core Scientific, Inc. | High-density data center infrastructure hosting hyperscale AI under 12-year CoreWeave contract (395 MW active). | Completion of full 500 MW CoreWeave fleet and achievement of positive quarterly free cash flow post-capex. | Primary 10-K/10-Q in ODP-05 baseline. | $4.30B debt ($3.3B 7.75% senior notes); $797.5M Q2 2026 capex exceeds CFO; $(1.15B) net loss from warrant revaluations; DCF suppressed. | **Held** | Bounded check: Fails Pillar B (Financial Resilience). Massive debt service, capex burn exceeding operating cash flow, and warrant dilution risk mean it does not currently fit the quality compounder collection. Direct research URL displays neutral unavailable state; stock page remains 100% active. |
| **`ULTA`** | Ulta Beauty, Inc. | Elite specialty beauty retailer; >44M loyalty members (>95% sales); $1.57B TTM operating cash flow. | Pullback toward base corridor ($430–$450) or sustained comp-store sales acceleration >6%. | Primary 10-K/10-Q direct inspection completed; Note 7 debt audited. | ASC 842 store leases expensed in EBIT; $339.6M funded debt vs $213.5M cash ($126.1M net funded debt bridge). | **Selected** | First Published Expansion Addition; verified loyalty moat facing demanding 15.45% 5Y CAGR hurdle at $592.94. |
| **`MEDP`** | Medpace Holdings, Inc. | Full-service clinical contract research organization (CRO); ~20-22% operating margins; zero debt; >30% ROIC. | Small-to-midsize biotech funding stabilization (XBI index recovery) or pullback to base intrinsic value. | High; audited Form 10-K and 10-Q verified. | Pristine zero-debt balance sheet; clean GAAP cash flows. | **Selected (PR #41)** | Published Expansion Addition; exceptional capital efficiency, zero debt, and minimal accounting ambiguity. |
| **`ISRG`** | Intuitive Surgical, Inc. | Da Vinci robotic surgical monopoly; >80% recurring procedure/service revenue; massive net cash. | da Vinci 5 adoption ramp and procedure growth vs. demanding >50x P/E multiple. | High; audited Form 10-K and 10-Q verified. | High multiple requires careful reverse-DCF terminal growth boundary testing. | **Selected (PR #41)** | Published Expansion Addition; quintessential high-quality moat with clear valuation discipline. |
| **`DECK`** | Deckers Outdoor Corp. | Hoka and UGG brand momentum driving >20% operating margins and zero funded debt. | Wholesale inventory channel health and Hoka growth durability vs. running footwear competition. | High; 6:1 forward stock split post-audit reconciliation completed. | Retail ASC 842 lease treatment and split restatements verified. | **Selected (PR #41)** | Published Expansion Addition; strong consumer brand quality, verified split-adjusted share count. |
| **`KLAC`** | KLA Corp. | Semiconductor process diagnostic and yield control monopoly (>50% optical inspection market share). | Wafer fab equipment (WFE) cycle inflection and China export restriction exposure. | Moderate; June fiscal year end (10-K June 30). | Funded debt bridge and quarterly fiscal calendar offset. | **Held** | Top tier business, but capital structure and cycle normalization need deep inspection. |
| **`CDNS`** | Cadence Design Systems, Inc. | Electronic Design Automation (EDA) software duopoly; high recurring software maintenance; deep lock-in. | Valuation multiple (~45x P/E) vs. AI chip design complexity growth. | Moderate; requires SBC dilution schedule verification. | Acquired intangibles amortization and deferred revenue recognition. | **Held** | High quality, but SBC schedule and acquisition adjustments require dedicated reconciliation. |
| **`ANET`** | Arista Networks, Inc. | High-throughput cloud networking EOS software and 800G/1.6T AI switches; zero debt; >40% margins. | Cloud customer concentration (Meta & Microsoft >35% revenue) and White-box switch competition. | Moderate; customer concentration disclosures need verification. | Nil; pristine cash balance. | **Held** | Elite business; customer concentration risks require specific filing verification. |
| **`LULU`** | Lululemon Athletica Inc. | Premium athletic apparel brand; strong international expansion (China); zero funded debt. | US comp sales stagnation, product execution (Breezethrough pause), and margin defense. | Moderate; requires ASC 842 retail lease exclusion and international revenue breakdown. | Significant retail store lease commitments. | **Held** | Moat under test in North America; requires comp deceleration audit. |
| **`VRT`** | Vertiv Holdings Co | Critical data center power and thermal management (liquid cooling) provider; massive hyperscale orders. | Margin execution on fixed-price modular contracts and capacity ramp. | Moderate. | Post-SPAC warrant liabilities and significant funded debt structure ($3B+). | **Held** | Capital structure requires extensive debt-covenant and debt-bridge reconciliation. |
| **`PWR`** | Quanta Services, Inc. | Dominant utility grid interconnection and transmission infrastructure contractor. | Renewable grid interconnection delays and project delivery execution. | Moderate. | High acquisition debt ($4B+) and percentage-of-completion contract accounting. | **Held** | Project accounting and acquisition leverage require deep audit pass. |
| **`EME`** | EMCOR Group, Inc. | Specialty mechanical/electrical contractor benefiting from industrial reshoring and data centers. | Margin defense against project labor inflation and fixed-price cost escalations. | Moderate. | Working capital swings and multi-employer pension liabilities. | **Held** | Sector already represented by FIX; FIX exhibits superior capital efficiency. |
| **`SNPS`** | Synopsys, Inc. | EDA software monopoly alongside CDNS. | Pending $35B mega-acquisition of Ansys (regulatory approvals, massive debt issuance). | Moderate. | Ansys merger pending; capital structure in severe flux. | **Held** | Transformative pending transaction renders standalone valuation model obsolete until financing closes. |
| **`NOW`** | ServiceNow, Inc. | Enterprise workflow automation cloud monopoly (>98% renewal rate). | Growth durability at $12B+ scale and AI GenAI monetization (Pro Plus). | Moderate. | Massive cumulative stock-based compensation ($1.5B+ annually) requiring share dilution adjustment. | **Held** | Top tier business, but SBC adjustments must be audited before publication. |
| **`ELF`** | e.l.f. Beauty, Inc. | Fast-growing mass beauty disruptor; Gen-Z digital marketing moat; high ROIC. | Growth normalization from >40% to mid-teens and Tariffs/supply chain exposure to China manufacturing. | Moderate. | Working capital volatility and acquisition accounting (Naturium). | **Held** | High growth, but tariff sensitivity and Naturium integration require focused review. |
| **`MU`** | Micron Technology, Inc. | Commodity memory hardware (DRAM/HBM/NAND). | Severe cyclical memory troughs; lack of pricing power during downturns. | Audited in ODP-05 baseline (`NO_SUPPORTED_OPPORTUNITY`). | Cyclical earnings swings distort standard single-stage DCF models. | **Excluded (Rejected)** | Original research baseline rejected record. Removed from active research discovery and research-specific navigation per product owner direction. |
| **`PLTR`** | Palantir Technologies Inc. | Enterprise and defense AI data ontology software (AIP / Foundry). | Extreme valuation multiple (>100x P/E, >35x EV/Sales) and high historical SBC dilution. | Evaluated in scaling pilot drafts (`NO_SUPPORTED_OPPORTUNITY`). | Valuation suppressed/inapplicable under standard hurdle bounds. | **Excluded (Rejected)** | Exited active research preparation queue per product owner direction. No research effort spent producing rejected dossiers. |

---

## 3. Effective Published Quality Watch Collection

Under this selective publication model, the active published Research collection comprises exactly **6 Quality Watch dossiers** (authorized and released via PR #39 and PR #41):

1. **`FIX`** — Comfort Systems USA, Inc. (`WATCH_CONDITIONS` — Baseline)
2. **`DUOL`** — Duolingo, Inc. (`WATCH_CONDITIONS` — Baseline)
3. **`ULTA`** — Ulta Beauty, Inc. (`WATCH_CONDITIONS` — First Published Addition, PR #39)
4. **`MEDP`** — Medpace Holdings, Inc. (`WATCH_CONDITIONS` — Published Addition, PR #41)
5. **`ISRG`** — Intuitive Surgical, Inc. (`WATCH_CONDITIONS` — Published Addition, PR #41)
6. **`DECK`** — Deckers Outdoor Corp. (`WATCH_CONDITIONS` — Published Addition, PR #41)

All 6 published companies represent genuine business quality with explicit, observable watch conditions.
- **Held Baseline Assets (`IREN`, `CORZ`)**: Retain full historical source preservation in the baseline ledger; their research URLs render the neutral "No Published Watch Dossier" state, and their stock detail pages remain 100% operational on the Terminal without research badges.
- **Excluded Records (`MU`, `PLTR`)**: Completely removed from active research discovery, navigation rails, and stock-detail badges, but remain fully functional within the wider Terminal and stock fundamental pages.
