# ARX TERMINAL — OPPORTUNITY DOSSIER SCALING PILOT COHORT & ELIGIBILITY MATRIX

**Document Version:** 1.0.0-draft  
**Date:** 2026-10-21  
**Governing Contract:** CONTRACT-ARX-OPP-PILOT-SCALING-V1  
**Pricing Observation Session:** 2026-10-16 (Regular Close)  
**Target Universe:** 20 Companies Total (5 Baseline Cohort + 15 Pilot Expansion Additions)

---

## 1. Executive Summary

This document establishes the authoritative cohort selection, business model classification, filing accessibility, and valuation eligibility for the **Opportunity Dossier Scaling Pilot**.

The pilot scales the accepted five-company Proof of Concept (`FIX`, `DUOL`, `IREN`, `CORZ`, `MU`) into a repeatable, auditable research preparation pipeline targeting 20 operating companies.

### Core Selection Principles:
1. **Evidence Accessibility:** All companies must be SEC reporting issuers with accessible Forms 10-K and 10-Q filed via EDGAR.
2. **Pricing Provenance:** Market pricing is anchored strictly to verified regular close quotes from the frozen `2026-10-16` market session tape.
3. **Valuation Method Appropriateness:** Companies must operate within business structures where operating income (EBIT), net debt bridges, and free cash flows to the firm (FCFF) or equity (FCFE) can be mathematically modeled or where valuation suppression is formally justified.
4. **Structural Exclusions:** Financial institutions (banks), insurers, real estate investment trusts (REITs), and pre-revenue clinical biotechs are excluded from standard FCFF models due to structural balance sheet inapplicability.
5. **Clear Separation:** Pilot draft records are maintained in isolated staging ledgers and do not modify the frozen ODP-05 baseline (`a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d`) or production snapshot.

---

## 2. Comprehensive 20-Company Pilot Cohort

| # | Ticker | Company Name | CIK | Primary Exchange | Sector & Industry | Business Classification | Primary Filings | 2026-10-16 Ref Price | Valuation Eligibility | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline 1** | `FIX` | Comfort Systems USA, Inc. | `0001035983` | NYSE | Industrials / Specialty Contractors | Modular thermal & HVAC infrastructure for data centers | 10-K (2025-12-31), 10-Q (2026-06-30) | $1,438.57 | Computed & Verified | `WATCH_CONDITIONS` |
| **Baseline 2** | `DUOL` | Duolingo, Inc. | `0001562088` | NASDAQ | Technology / Application Software | Subscription language learning & mobile education | 10-K (2025-12-31), 10-Q (2026-06-30) | $141.76 | Computed & Verified | `WATCH_CONDITIONS` |
| **Baseline 3** | `IREN` | Iris Energy Ltd. | `0001878848` | NASDAQ | Technology / Data Center Infrastructure | Data center power infrastructure & AI cloud transition | 10-K (2026-06-30), 6-K (2026-08-28) | $46.54 | Valuation Suppressed | `WATCH_CONDITIONS` |
| **Baseline 4** | `CORZ` | Core Scientific, Inc. | `0001839341` | NASDAQ | Technology / Data Center Infrastructure | Post-reorganization AI hyperscale colocation | 10-K (2025-12-31), 10-Q (2026-06-30) | $17.65 | Valuation Suppressed | `WATCH_CONDITIONS` |
| **Baseline 5** | `MU` | Micron Technology, Inc. | `0000723125` | NASDAQ | Technology / Semiconductors | Cyclical memory (DRAM, HBM3E, NAND) hardware | 10-K (2026-09-03) | $878.08 | Computed with Qualification | `NO_SUPPORTED_OPPORTUNITY` |
| **Addition 1** | `ULTA` | Ulta Beauty, Inc. | `0001403568` | NASDAQ | Consumer Cyclical / Specialty Retail | Omnichannel cosmetics, fragrance & salon services | 10-K (2026-01-31), 10-Q (2026-08-01) | $592.94 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 2** | `MEDP` | Medpace Holdings, Inc. | `0001668397` | NASDAQ | Healthcare / Medical Research | Full-service Phase I-IV clinical research organization (CRO) | 10-K (2025-12-31), 10-Q (2026-06-30) | $618.47 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 3** | `DECK` | Deckers Outdoor Corp. | `0000910521` | NYSE | Consumer Cyclical / Footwear | Premium footwear & apparel (HOKA, UGG) | 10-K (2026-03-31), 10-Q (2026-06-30) | $85.72 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 4** | `KLAC` | KLA Corporation | `0000319201` | NASDAQ | Technology / Semiconductor Equipment | Process control, metrology & defect inspection systems | 10-K (2026-06-30) | $165.74 | Computed with Qualification | `WATCH_CONDITIONS` |
| **Addition 5** | `LULU` | Lululemon Athletica Inc. | `0001397187` | NASDAQ | Consumer Cyclical / Apparel Retail | Athletic apparel, footwear & lifestyle accessories | 10-K (2026-01-28), 10-Q (2026-08-02) | $97.10 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 6** | `EME` | EMCOR Group, Inc. | `0000105634` | NYSE | Industrials / Specialty Contractors | Electrical, mechanical construction & data center thermal | 10-K (2025-12-31), 10-Q (2026-06-30) | $721.66 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 7** | `VRT` | Vertiv Holdings Co | `0001674101` | NYSE | Industrials / Electrical Equipment | Data center power management & liquid cooling | 10-K (2025-12-31), 10-Q (2026-06-30) | $206.42 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 8** | `ANET` | Arista Networks, Inc. | `0001596532` | NYSE | Technology / Computer Hardware | Cloud and AI data center Ethernet switching & routing | 10-K (2025-12-31), 10-Q (2026-06-30) | $176.99 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 9** | `PLTR` | Palantir Technologies Inc. | `0001321655` | NASDAQ | Technology / Software Infrastructure | Enterprise AI operating system & defense analytics | 10-K (2025-12-31), 10-Q (2026-06-30) | $186.03 | Computed with Qualification | `NO_SUPPORTED_OPPORTUNITY` |
| **Addition 10** | `ISRG` | Intuitive Surgical, Inc. | `0001035267` | NASDAQ | Healthcare / Medical Instruments | Robotic-assisted surgical systems (da Vinci 5) | 10-K (2025-12-31), 10-Q (2026-06-30) | $426.70 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 11** | `CDNS` | Cadence Design Systems, Inc. | `0000813672` | NASDAQ | Technology / Software Application | Electronic design automation (EDA) software & IP | 10-K (2025-12-31), 10-Q (2026-06-30) | $319.51 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 12** | `PWR` | Quanta Services, Inc. | `0001050915` | NYSE | Industrials / Engineering & Construction | Power grid, substation, renewable & utility infrastructure | 10-K (2025-12-31), 10-Q (2026-06-30) | $631.98 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 13** | `SNPS` | Synopsys, Inc. | `0000883241` | NASDAQ | Technology / Software Application | Silicon-to-software EDA design automation & Ansys integration | 10-K (2025-10-31), 10-Q (2026-07-31) | $454.63 | Computed with Qualification | `WATCH_CONDITIONS` |
| **Addition 14** | `NOW` | ServiceNow, Inc. | `0001373715` | NYSE | Technology / Software Infrastructure | Enterprise cloud workflow digital automation platform | 10-K (2025-12-31), 10-Q (2026-06-30) | $141.95 | Computed & Verified | `WATCH_CONDITIONS` |
| **Addition 15** | `ELF` | e.l.f. Beauty, Inc. | `0001600033` | NYSE | Consumer Defensive / Household & Personal Products | High-growth prestige color cosmetics & skincare | 10-K (2026-03-31), 10-Q (2026-06-30) | $113.53 | Computed & Verified | `WATCH_CONDITIONS` |

---

## 3. Structural Business Sector Groupings

The 20 companies in the scaling pilot are organized into 5 coherent, economically rigorous sectors:

### Sector 1: Specialty Contractors & Critical Infrastructure
- **Companies:** `FIX`, `EME`, `VRT`, `PWR`
- **Economic Model:** Modular HVAC, thermal management, liquid cooling, electrical switchgear, and utility grid interconnection. Driven by hyperscale data center electrification and semiconductor fab buildouts.
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. Strong cash conversion; working capital swings moderated by milestone prepayments.
- **Lease / Debt Conventions:** Standard funded senior debt subtracted; unrestricted cash added to EV.

### Sector 2: Cloud, Networking & Workflow Software
- **Companies:** `DUOL`, `PLTR`, `ANET`, `NOW`
- **Economic Model:** Subscription recurring revenue, enterprise cloud platforms, AI workflow automation, and high-throughput data center switching.
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. High gross margins (65%-85%).
- **Qualification Factors:** Stock-based compensation (SBC) dilution must be explicitly monitored; `PLTR` priced-in growth evaluated against extreme multiples.

### Sector 3: Semiconductor Equipment & EDA Silicon Design
- **Companies:** `MU`, `KLAC`, `CDNS`, `SNPS`
- **Economic Model:** Semiconductor process yield control, silicon design verification, and memory hardware manufacturing.
- **Valuation Treatment:** Reverse-DCF FCFF.
- **Qualification Factors:** `MU` is marked `COMPUTED_QUALIFIED` and `NO_SUPPORTED_OPPORTUNITY` due to extreme memory cycle peaks/troughs; `SNPS` qualified by pending debt-funded acquisition structures (Ansys).

### Sector 4: High-Return Healthcare & Clinical Medical Technology
- **Companies:** `MEDP`, `ISRG`
- **Economic Model:** Capital-light clinical trial execution (`MEDP`) and recurring razor-and-blade robotic surgical procedures (`ISRG`).
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. Pristine net cash balance sheets, high returns on invested capital (>20% ROIC). Zero funded debt.

### Sector 5: Capital-Light Quality Consumer Brands
- **Companies:** `ULTA`, `DECK`, `LULU`, `ELF`
- **Economic Model:** Premium footwear, cosmetics retail, and athletic performance apparel.
- **Valuation Treatment:** Reverse-DCF FCFF under strict **Lease-as-Operating-Expense Convention** (ASC 842). Store operating leases are expensed in operating income and paid in CFO; lease liabilities are excluded from the net debt bridge to prevent double deduction.
- **Share Count Mechanics:** `DECK` audited post 6:1 forward stock split.

---

## 4. Documented Exclusions & Rejected Candidates

The following candidate business models were evaluated and rejected from this pilot cohort due to structural method incompatibility:

| Rejected Ticker | Organization | Category | Reason for Rejection | Justified Alternative Method Required |
|---|---|---|---|---|
| `BAC` / `JPM` | Bank of America / JPMorgan Chase | Commercial & Investment Banking | **Balance Sheet Debt is Operating Raw Material:** Banks fund earning assets (loans, securities) with liabilities (deposits, repo). Interest expense is an operating cost, not financing. Enterprise Value and FCFF models are mathematically undefined. | Excess Returns Model (ERM) or Dividend Discount Model (DDM) tied to Tier 1 Common Equity (CET1). |
| `PGR` / `CB` | Progressive / Chubb | Property & Casualty Insurance | **Float and Reserve Dynamics:** Operating cash flow is dominated by premium float timing and statutory loss reserves rather than operating margins. | Return on Equity (ROE) versus Price-to-Tangible Book Value (P/TBV) with combined ratio underwriting audits. |
| `PLD` / `EQIX` | Prologis / Equinix | Real Estate Investment Trusts (REITs) | **Depreciation Distortion:** Real estate GAAP net income and operating margins are severely distorted by multi-decade non-cash real property depreciation. | Funds From Operations (FFO) / Adjusted Funds From Operations (AFFO) or Net Asset Value (NAV) capitalization. |
| `ARWR` / `CRSP` | Arrowhead Pharma / CRISPR Therapeutics | Clinical-Stage Pre-Revenue Biotech | **Binary Clinical/Regulatory Pipeline Risk:** Operating cash flows are deeply negative due to R&D burn. Revenue CAGR has zero empirical anchor prior to FDA approval. | Probability-weighted Risk-Adjusted Net Present Value (rNPV) per clinical drug asset. |

---

## 5. Maintenance & Preparation Effort Analysis

| Category | Average Automated Preparation Time | Manual Review & Audit Time | Total Time per Dossier | Primary Bottleneck |
|---|---|---|---|---|
| **Pristine Balance Sheet (Net Cash, Zero Debt)** (e.g. `MEDP`, `ANET`, `ISRG`) | ~4.2 seconds | 15–20 minutes | ~20 minutes | Sourcing 10-Q note verification of zero debt |
| **Retail ASC 842 Lease-Heavy** (e.g. `ULTA`, `LULU`, `DECK`) | ~4.5 seconds | 20–25 minutes | ~25 minutes | Lease reconciliation vs net debt bridge |
| **Semiconductor Equipment / EDA** (e.g. `KLAC`, `CDNS`, `SNPS`) | ~5.1 seconds | 25–30 minutes | ~30 minutes | Checking multi-year cycle normalization |
| **Capital-Intensive Infrastructure / Suppressed** (e.g. `IREN`, `CORZ`) | ~3.8 seconds | 35–45 minutes | ~45 minutes | Establishing observable resume conditions |

**Measured Conclusion:** Automated extraction, validation, and solving reduces machine execution time to seconds, shifting human analyst effort entirely to qualitative thesis verification and accounting boundary review.
