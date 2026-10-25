# ARX TERMINAL — ALC V4 FINAL DATE AND OUTPUT PARITY CLOSURE
## Authoritative Resolution of Investment Timeline, Dated Return Parity, and Scenario Ledger Integrity
### Evidence-Grade Synthesis Under ARX Closure Model v2 — Read-Only Mode

---

## 1. Governance Authority & Decision Token

- **Governing Standard:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Target Security:** Alcon Inc. (NYSE: `ALC`, SIX: `ALC`)
- **Evaluation Baseline:** October 22, 2026 (Spot Price: **$67.43**)
- **Prior Independent Audit Disposition:** `ALC_CROSS_MODEL_OUTPUT_CERTIFICATION_PENDING_DATE_RECONCILIATION`
- **Current Final Decision Token:**

# `ALC_CROSS_MODEL_OUTPUT_CERTIFIED`

---

## 2. Resolution of the Investment Timeline

The V4 evaluation ledger identified an apparent tension between:
- Valuation entry date: **October 22, 2026**
- Forecast horizon: **FY2026E–FY2030E**
- Calendar days to December 31, 2030: **1,531 days** (4.19 years)
- Annual cash-flow periods: **5**

Under ARX Closure Model v2, the investment timeline is resolved as follows:

### Institutional Timeline Standards:
- **Convention A (Formally Adopted): Five Full Years (October 22, 2026 to October 22, 2031 / 1,826 Calendar Days)**
  - Exact calendar holding period: 1,826 days ($4 \times 365 + 366$ leap-year days in 2028).
  - Alcon's operational forecast (FY2026E–FY2030E) represents 5 full annual operating cycles establishing the terminal run-rate ($14,396.6M revenue, $3,815.1M EBITDA in Base Case) and cumulative 5-year balance sheet bridge (-$2,176.2M net debt reduction to net cash).
  - The equity value supported by these five years of operational execution and cash generation is realized upon exit at the 5.00-year investment horizon on **October 22, 2031**.
- **Convention B (Rejected): Truncated Period to December 31, 2030 (1,531 Calendar Days / 4.19 Years)**
  - Truncating the holding period to December 31, 2030 would create an irreconcilable financial contradiction: it would credit the investment with 5 full annual cash flow periods and 5 full annual dividend distributions while discounting over only 4.19 calendar years.
  - Furthermore, Alcon reports on a December 31 fiscal year; an exit on December 31, 2030 would occur before FY2030 audited results are finalized and published (typically in late February / early March 2031).
  - Therefore, **Convention A is the sole authoritative standard**.

---

## 3. Return Methodology: Annual IRR vs. Calendar Dated XIRR

To eliminate all ambiguity, ARX Terminal reports both return metrics side-by-side:

1. **`five_period_annual_irr` (Discrete 5-Period Annual IRR):**
   - Assumes discrete annual periods ($t = 0, 1, 2, 3, 4, 5$).
   - Equation:
     $$\text{NPV}(r) = -P_{\text{spot}} + \sum_{t=1}^{4} \frac{\text{DPS}_t}{(1 + r)^t} + \frac{\text{DPS}_5 + P_{\text{terminal}}}{(1 + r)^5} = 0$$

2. **`calendar_dated_xirr` (Actual/365 Calendar Dated XIRR):**
   - Sourced from Alcon's actual corporate distribution schedule: Alcon pays its annual dividend following the Annual General Meeting (AGM) in mid-May.
   - Day counts from October 22, 2026 ($t_0 = 0$):
     - Year 1 Dividend ($0.36): May 15, 2027 (Day 205, $t_1 = 205/365 = 0.5616$ yrs)
     - Year 2 Dividend ($0.39): May 15, 2028 (Day 571, $t_2 = 571/365 = 1.5644$ yrs; 2028 is a leap year)
     - Year 3 Dividend ($0.42): May 15, 2029 (Day 936, $t_3 = 936/365 = 2.5644$ yrs)
     - Year 4 Dividend ($0.45): May 15, 2030 (Day 1301, $t_4 = 1301/365 = 3.5644$ yrs)
     - Year 5 Dividend ($0.48): May 15, 2031 (Day 1666, $t_5 = 1666/365 = 4.5644$ yrs)
     - Terminal Exit ($P_{\text{terminal}}$): October 22, 2031 (Day 1826, $t_6 = 1826/365 = 5.0027$ yrs)
   - Equation:
     $$\text{NPV}(r) = \sum_{i=0}^{6} \frac{\text{CF}_i}{(1 + r)^{(d_i - d_0)/365.0}} = 0$$

### Side-by-Side Return Comparison:

| Scenario ID | Scenario Name | Terminal Price | Simple CAGR | 5-Period Annual IRR | Calendar Dated XIRR | Delta (XIRR - Annual) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `CONSERVATIVE` | Conservative (Published Reference) | $76.64 | 3.15% | **3.18%** | **3.19%** | +1 bp |
| `BASE_MULTIPLE` | Base Case (Exit Multiple) | $90.02 | 6.44% | **6.50%** | **6.51%** | +1 bp |
| `BASE_DCF` | Base Case (Gordon Growth DCF) | $91.82 | 6.85% | **6.92%** | **6.93%** | +1 bp |
| `OPTIMISTIC` | Optimistic (Published Reference) | $123.42 | 13.23% | **13.34%** | **13.36%** | +2 bps |
| `CONSERVATIVE_DYNAMIC` | Conservative (Cash-Constrained) | $76.70 | 3.17% | **3.20%** | **3.21%** | +1 bp |
| `OPTIMISTIC_DYNAMIC` | Optimistic (High-Cash Dynamic) | $129.06 | 14.23% | **14.34%** | **14.36%** | +2 bps |

*Note: Calendar XIRR is 1–2 bps higher than discrete annual IRR because Alcon's dividends are received in mid-May (approximately 5.2 months into each year), providing a modest reinvestment timing benefit.*

---

## 4. Authoritative 7-Event Dated Cash-Flow Schedule

The exact cash flows for **Base Case Gordon Growth DCF** ($91.82 terminal price) and **Base Case Multiple** ($90.02 terminal price) are serialized in `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`:

| Event # | Calendar Date | Event Classification | Days from Entry | Fraction of Years ($t_i$) | Cash Flow (Base DCF) | Cash Flow (Base Mult.) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | `2026-10-22` | `INITIAL_INVESTMENT` | 0 | 0.000000 | **-$67.43** | **-$67.43** |
| **1** | `2027-05-15` | `ANNUAL_DIVIDEND_YEAR_1` | 205 | 0.561644 | **+$0.36** | **+$0.36** |
| **2** | `2028-05-15` | `ANNUAL_DIVIDEND_YEAR_2` | 571 | 1.564384 | **+$0.39** | **+$0.39** |
| **3** | `2029-05-15` | `ANNUAL_DIVIDEND_YEAR_3` | 936 | 2.564384 | **+$0.42** | **+$0.42** |
| **4** | `2030-05-15` | `ANNUAL_DIVIDEND_YEAR_4` | 1301 | 3.564384 | **+$0.45** | **+$0.45** |
| **5** | `2031-05-15` | `ANNUAL_DIVIDEND_YEAR_5` | 1666 | 4.564384 | **+$0.48** | **+$0.48** |
| **6** | `2031-10-22` | `TERMINAL_SHARE_DISPOSITION`| 1826 | 5.002740 | **+$91.82** | **+$90.02** |
| **Total** | — | **Cumulative Net Cash Flow** | — | — | **+$26.49** | **+$24.69** |

---

## 5. Invertible Accumulation Ceilings & Invertibility Proofs

Accumulation ceilings represent the maximum allowable entry price today that guarantees an institutional hurdle rate (8.0%, 8.5%, or 10.0%).

### Ceiling Invariant Invertibility:
For entry price $P_{\text{entry}}$, the root-finder solution $\text{IRR}(P_{\text{entry}}) \equiv r_{\text{hurdle}}$ holds exactly to within $< 10^{-6}$.

| Scenario ID | Hurdle Rate | Annual IRR Ceiling ($) | Inverted Annual IRR | Calendar XIRR Ceiling ($) | Inverted Calendar XIRR |
|:---|:---:|:---:|:---:|:---:|:---:|
| `BASE_DCF` ($91.82) | **8.0%** | **$64.15** | 0.080000 | **$64.20** | 0.080000 |
| `BASE_DCF` ($91.82) | **8.5%** | **$62.70** | 0.085000 | **$62.75** | 0.085000 |
| `BASE_DCF` ($91.82) | **10.0%** | **$58.59** | 0.100000 | **$58.64** | 0.100000 |
| `BASE_MULTIPLE` ($90.02) | **8.0%** | **$62.93** | 0.080000 | **$62.97** | 0.080000 |
| `BASE_MULTIPLE` ($90.02) | **8.5%** | **$61.50** | 0.085000 | **$61.55** | 0.085000 |
| `BASE_MULTIPLE` ($90.02) | **10.0%** | **$57.47** | 0.100000 | **$57.52** | 0.100000 |
| `CONSERVATIVE` ($76.64) | **8.0%** | **$53.82** | 0.080000 | **$53.86** | 0.080000 |
| `CONSERVATIVE` ($76.64) | **8.5%** | **$52.60** | 0.085000 | **$52.65** | 0.085000 |
| `CONSERVATIVE` ($76.64) | **10.0%** | **$49.16** | 0.100000 | **$49.21** | 0.100000 |
| `OPTIMISTIC` ($123.42) | **8.0%** | **$85.65** | 0.080000 | **$85.69** | 0.080000 |
| `OPTIMISTIC` ($123.42) | **8.5%** | **$83.71** | 0.085000 | **$83.75** | 0.085000 |
| `OPTIMISTIC` ($123.42) | **10.0%** | **$78.20** | 0.100000 | **$78.25** | 0.100000 |
| `CONSERVATIVE_DYNAMIC` ($76.70) | **8.0%** | **$53.86** | 0.080000 | **$53.91** | 0.080000 |
| `CONSERVATIVE_DYNAMIC` ($76.70) | **8.5%** | **$52.65** | 0.085000 | **$52.69** | 0.085000 |
| `CONSERVATIVE_DYNAMIC` ($76.70) | **10.0%** | **$49.20** | 0.100000 | **$49.25** | 0.100000 |
| `OPTIMISTIC_DYNAMIC` ($129.06) | **8.0%** | **$89.49** | 0.080000 | **$89.53** | 0.080000 |
| `OPTIMISTIC_DYNAMIC` ($129.06) | **8.5%** | **$87.46** | 0.085000 | **$87.50** | 0.085000 |
| `OPTIMISTIC_DYNAMIC` ($129.06) | **10.0%** | **$81.71** | 0.100000 | **$81.75** | 0.100000 |

---

## 6. Mathematical Forensic Resolution of Historical Discrepancy

### Defect Remediation Record:
- **Prior Assertion:** Earlier reports claimed that the legacy published return of **7.37%** was produced by the formula:
  $$\text{CAGR} = \left(\frac{P_{\text{terminal}} + \$2.05}{\$67.43}\right)^{1 / 4.70} - 1$$
- **Forensic Disproof:**
  Substituting $P_{\text{terminal}} = \$91.83$ (Base DCF):
  $$\left(\frac{91.83 + 2.05}{67.43}\right)^{1 / 4.70} - 1 = (1.392258)^{0.212766} - 1 = 7.2948\% \approx 7.29\% \neq 7.37\%$$
  Substituting cumulative dividends of $\$2.10$:
  $$\left(\frac{91.83 + 2.10}{67.43}\right)^{1 / 4.70} - 1 = (1.392999)^{0.212766} - 1 = 7.3108\% \approx 7.31\% \neq 7.37\%$$
- **Binding Determination Under ARX Closure Model v2:**
  Agents must never manufacture unprovable rationalizations for historical anomalies. The earlier numbers (3.49%, 6.94%, 7.37%, 13.43%) are formally classified as:
  # `COMPUTATIONAL_MECHANISM_UNKNOWN_AND_NON_REPRODUCIBLE`
  They remain preserved in historical audit ledgers as superseded artifacts. The forward authoritative baseline is exclusively the verified and reproducible 6.92% (Annual IRR) / 6.93% (Calendar XIRR).

---

## 7. Master Canonical Scenario Matrix (Zero Divergence from Ledger)

All parameters below are drawn verbatim from `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`:

| Parameter | Conservative Reference | Base Case Multiple | Base Case DCF | Optimistic Reference | Conservative Dynamic | Optimistic Dynamic |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario Identifier** | `CONSERVATIVE` | `BASE_MULTIPLE` | `BASE_DCF` | `OPTIMISTIC` | `CONSERVATIVE_DYNAMIC` | `OPTIMISTIC_DYNAMIC` |
| **5-Yr Revenue CAGR** | 5.0% | 6.5% | 6.5% | 8.5% | 5.0% | 8.5% |
| **FY2030E Revenue ($M)** | $13,406.4 | $14,396.6 | $14,396.6 | $15,807.4 | $13,401.4 | $15,788.9 |
| **Core Operating Margin** | 16.5% | 18.5% | 18.5% | 20.0% | 16.5% | 20.0% |
| **FY2030E EBITDA ($M)** | $3,217.5 | $3,815.1 | $3,815.1 | $4,347.0 | $3,216.4 | $4,341.9 |
| **Terminal Method** | 10.65x EV/EBITDA | 10.65x EV/EBITDA | Gordon Growth DCF | 13.00x EV/EBITDA | 10.65x EV/EBITDA | 13.00x EV/EBITDA |
| **Enterprise Value ($M)** | $34,266.4 | $40,630.8 | $41,487.8 | $56,511.0 | $34,254.1 | $56,445.1 |
| **Ending Net Debt ($M)** | -$2,176.2 | -$2,176.2 | -$2,176.2 | -$2,176.2 | -$2,645.8 | -$5,290.8 |
| **Ending Diluted Shares (M)**| 475.524 | 475.524 | 475.524 | 475.524 | 481.062 | 478.348 |
| **Terminal Share Price ($)** | **$76.64** | **$90.02** | **$91.82** | **$123.42** | **$76.70** | **$129.06** |
| **Cumulative DPS ($)** | $2.10 | $2.10 | $2.10 | $2.10 | $2.10 | $2.10 |
| **Simple 5-Yr CAGR (%)** | 3.15% | 6.44% | 6.85% | 13.23% | 3.17% | 14.23% |
| **5-Period Annual IRR (%)** | **3.18%** | **6.50%** | **6.92%** | **13.34%** | **3.20%** | **14.34%** |
| **Calendar Dated XIRR (%)** | **3.19%** | **6.51%** | **6.93%** | **13.36%** | **3.21%** | **14.36%** |
| **8.0% Hurdle Ceiling ($)** | **$53.82** | **$62.93** | **$64.15** | **$85.65** | **$53.86** | **$89.49** |
| **8.5% Hurdle Ceiling ($)** | **$52.60** | **$61.50** | **$62.70** | **$83.71** | **$52.65** | **$87.46** |
| **10.0% Hurdle Ceiling ($)** | **$49.16** | **$57.47** | **$58.59** | **$78.20** | **$49.20** | **$81.71** |

---

## 8. Operational Prohibitions & Governance Enforcements

In accordance with ARX Closure Model v2:
- **`RESEARCH_PUBLICATION_AUTHORIZED: NO (Strictly Prohibited)`**
- **`PRODUCTION_DEPLOYMENT: NOT_AUTHORIZED`**
- **`ALC_SCANNER_ENROLLMENT: BLOCKED`**
- **`AUTOMATED_TRADING: BLOCKED`**
- **`PORTFOLIO_ALLOCATION: NONE`**
- **`HEAD_COMMIT_PRESERVED: 1d089301940b8b4125c0d060af6e89926b11a20f`**
- **`SPRINT_2_RELEASE_HOLD: ACTIVE (CONSOLIDATED_RELEASE_HOLD_RAILWAY_INCIDENT)`**

---

## 9. Automated Verification Suite Confirmation

All 44 tests across 4 suites pass in 1.07s:
- `tests/test_alc_cross_model_contract.py`: 12/12 PASS
- `tests/test_alc_cross_model_integrity.py`: 14/14 PASS
- `tests/test_alc_quantitative_integrity.py`: 11/11 PASS
- `tests/test_alc_valuation_reproduction.py`: 7/7 PASS

**Final Certification Verdict:**
# `ALC_CROSS_MODEL_OUTPUT_CERTIFIED`
