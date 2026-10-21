# ARX TERMINAL — SPRINT 2 TRACK C
# PUBLICATION ACCOUNTING FINAL RESULTS & ARITHMETIC CLOSURE

## 1. Accounting Closure Equations & Invariants

Under ARX Closure Model v2, universe publication accounting enforces exact mathematical closure across two required invariant equations:

### Equation 1: Source Closure
$$\text{source\_count} = \text{eligible\_count} + \text{ineligible\_count} + \text{unresolved\_count}$$

Every candidate introduced into the evaluation pipeline must resolve into exactly one of three mutually exclusive eligibility partitions:
- $\text{eligible\_count}$: Satisfies all R01–R07 rules.
- $\text{ineligible\_count}$: Fails at least one of R01–R07 rules.
- $\text{unresolved\_count}$: Ambiguous identity, unresolvable provider locator, or error state.

### Equation 2: Data Readiness Closure
$$\text{eligible\_count} = \text{data\_ready\_count} + \text{data\_unavailable\_count} + \text{data\_unresolved\_count}$$

For all candidates in $\text{eligible\_count}$, market data readiness must partition cleanly:
- $\text{data\_ready\_count}$: Certified market data meeting the operational builder readiness contract ($\ge 60$ daily bars).
- $\text{data\_unavailable\_count}$: Data uningested, missing provenance sidecar, or $< 60$ bars.
- $\text{data\_unresolved\_count}$: Conflicting feeds or unresolvable market data.

---

## 2. Evaluation Results Across Test Cohorts

### 2.1. Scenario A: Fully Qualified Post-Replacement Pilot Cohort (4 Pilot Candidates)

- **Source Population:** AAPL, GOOG, GOOGL, ABBV with certified 288-bar Alpaca SIP split-adjusted history.
- **Verification Evidence:** [`test_14_deterministic_digest_and_accounting_closure`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/tests/test_package_b_v3_canonical_consumer.py#L491-L533)

| Accounting Metric | Value | Invariant Check |
| :--- | :--- | :--- |
| `source_count` | 4 | Input candidate set |
| `eligible_count` | 4 | Passed R01–R07 rules |
| `ineligible_count` | 0 | 0 rule failures |
| `unresolved_count` | 0 | 0 identity ambiguities |
| **Source Closure** | **$4 = 4 + 0 + 0$** | **CLOSED (True)** |
| `data_ready_count` | 4 | 288 bars $\ge 60$ bars operational builder floor |
| `data_unavailable_count` | 0 | 0 unprovenanced or missing bars |
| `data_unresolved_count` | 0 | 0 data conflicts |
| **Data Readiness Closure** | **$4 = 4 + 0 + 0$** | **CLOSED (True)** |
| `scannable_count` | 4 | Denotes potential candidate eligibility ($\text{scannable\_count} == \text{data\_ready\_count}$) |
| `published_count` | 4 | Published to active State 5 pointer |
| `strategy_admitted_count` | **0** | State 6 boundary strictly closed (requires `G-VCP-012` for VCP or independent authority) |
| `scanner_enrolled_count` | **0** | State 7 boundary strictly closed (requires `G-S2-ACT` operational dispatch) |

---

### 2.2. Scenario B: Pre-Replacement Storage State (Uncertified Legacy Data)

- **Source Population:** AAPL (288 unprovenanced bars), GOOG (287 Polygon bars), GOOGL (288 unprovenanced bars), ABBV (0 bars).
- **Verification Evidence:** [`test_09_uncertified_production_like_data_fails_closed`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/tests/test_package_b_v3_canonical_consumer.py#L269-L330)

| Accounting Metric | Value | Invariant Check |
| :--- | :--- | :--- |
| `source_count` | 4 | Input candidate set |
| `eligible_count` | 0 | Cannot enter State 4 without passing State 3 (DATA_QUALIFIED) |
| `ineligible_count` | 0 | Data disqualified before rule evaluation |
| `unresolved_count` | 4 | Trapped at State 2 (`PROVIDER_RESOLVABLE`) |
| **Source Closure** | **$4 = 0 + 0 + 4$** | **CLOSED (True)** |
| `data_ready_count` | 0 | 0 candidates certified |
| `data_unavailable_count` | 0 | Breakdown applies only to eligible |
| `data_unresolved_count` | 0 | Breakdown applies only to eligible |
| `scannable_count` | **0** | Fails closed |
| `published_count` | **0** | Fails closed (`published_symbols = []`) |
| `strategy_admitted_count` | **0** | Strictly closed |
| `scanner_enrolled_count` | **0** | Strictly closed |

---

### 2.3. Scenario C: Candidate DATA_READY but Ineligible Under R01–R07

- **Source Population:** 1 candidate with 288 certified daily bars and live price (`DATA_READY`), but violating R05 (Currency: `EUR`) and R06 (Primary Listing: `False`).
- **Verification Evidence:** [`test_15_data_ready_but_r01_r07_ineligible_cannot_publish_or_scan`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/tests/test_package_b_v3_canonical_consumer.py#L536-L590)

| Accounting Metric | Value | Verification Finding |
| :--- | :--- | :--- |
| `source_count` | 1 | Single test candidate |
| `eligible_count` | 0 | Fails R05 (`EUR`) and R06 (`False`) |
| `ineligible_count` | 1 | Correctly categorized as `INELIGIBLE` |
| `unresolved_count` | 0 | Identity fully resolved |
| **Source Closure** | **$1 = 0 + 1 + 0$** | **CLOSED (True)** |
| `data_ready_count` | 0 | Not counted in eligible data-ready |
| `scannable_count` | **0** | **NOT scannable merely because data is present!** |
| `published_count` | **0** | **Excluded from State 5 published pointer!** |
| `published_symbols` | `[]` | Verified empty |
| `current_state` | `DATA_QUALIFIED` | Trapped at State 3; cannot transition to State 4 (`PUBLICATION_ELIGIBLE`) |

**Core Architectural Takeaway:** The presence of historical market data alone never confers publication eligibility or scannability. A candidate must affirmatively satisfy both data qualification (State 3) and statutory eligibility rules (State 4) before entering the scannable or published universe.

---

## 3. State 5, 6, and 7 Independent Governance Boundaries

Track C maintains absolute independence across the upper publication states:

- **State 5 (`PUBLISHED_UNIVERSE`):** Governed by [`PublicationAttestationEvidence`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/analyst_dashboard/universe/publication_state_machine.py#L305-L350). Confirms candidate data readiness, eligibility, and arithmetic accounting closure.
- **State 6 (`STRATEGY_ADMITTED`):** Governed by [`StrategyAdmissionEvidence`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/analyst_dashboard/universe/publication_state_machine.py#L353-L415). Requires `G-VCP-012` for VCP Epoch 002 (currently DEFERRED); independent strategies (e.g. Smart Money) admit under their own specific authority without requiring `G-VCP-012`.
- **State 7 (`ACTIVELY_SCANNED`):** Governed by [`ScannerDispatchEvidence`](file:///C:/Users/akara/Documents/Projects/finance-sprint-2-package-c/analyst_dashboard/universe/publication_state_machine.py#L418-L455). Strictly requires operational token `G-S2-ACT`. No background schedule or scanner execution is permitted without this affirmative token.
