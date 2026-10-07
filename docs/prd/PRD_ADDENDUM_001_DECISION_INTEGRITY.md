# PRD Addendum 001: Decision Integrity, Epistemic Consistency & Instrument-Aware Contracts

**Product**: ARX Terminal  
**Document ID**: `PRD-ADDENDUM-001-DECISION-INTEGRITY`  
**Parent Document**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Lifecycle Status**: `RATIFIED`  
**Milestone**: Synthesis E — Rescoped Wave 3 Specification  
**Classification**: Product & Domain Engineering Contract  
**Created**: 2026-10-07  
**Updated**: 2026-10-07 (Epistemic Correction & Canonical Instrument Formalization)  

---

## 1. Executive Intent & Invariant Foundation

This Addendum establishes binding product, data, and presentation contracts to resolve decision-quality and epistemic integrity defects identified during the Synthesis E Wave 3 authorization audit.

### 1.1 Core Epistemic Invariant
```text
CLAIM_SET ⊆ EVIDENCE_SET
```
$$\text{CLAIM\_SET} \subseteq \text{EVIDENCE\_SET}$$

The user interface, explanation layer, and pre-flight validation tooling must never present assertions, inferences, or narrative justifications that exceed the authentic evidence actively evaluated and fresh at runtime.

### 1.2 Non-Collapsing Semantic States Invariant
The following data states represent fundamentally different market and pipeline realities. They must never be casually collapsed into `0`, `false`, `FAIL`, or empty views:
```text
OBSERVED_EVENT_DATE_IS_IMMUTABLE
ZERO != MISSING
EMPTY_RESULT != SOURCE_UNAVAILABLE
PIPELINE_PENDING != ZERO
STALE != LIVE
NOT_APPLICABLE != FAIL
```
1. `ZERO`: A valid, verified measurement with mathematical value $0.0$.
2. `EMPTY_RESULT`: A valid query completed successfully against a live source and returned zero matching records.
3. `NOT_APPLICABLE`: A regulatory, economic, or technical concept does not apply to the specific security class (e.g., SEC Form 10-K for ETFs).
4. `SOURCE_UNAVAILABLE`: The authoritative upstream provider or feed is temporarily offline, degraded, or unreachable.
5. `PIPELINE_PENDING`: An architectural capability is designed and active for single-asset lookups, but market-wide background ingestion/screening is pending rollout.
6. `STALE`: Authentic evidence exists in store, but exceeds statutory or real-time freshness thresholds.
7. `UNKNOWN`: Security identity, listing state, or market classification cannot be verified safely.

---

## 2. Pre-Flight Product Contract

### 2.1 Role & Definition
```text
PRE_FLIGHT =
  EXECUTION_READINESS_VALIDATOR

PRE_FLIGHT_SECONDARY_DECISION_ENGINE =
  NO
```

The Pre-Flight Checklist is an informational trade-sanity and risk-hygiene validator positioned immediately before trade execution / trade plan export. It evaluates 5 structural conditions required before capital allocation.

### 2.2 Authority & Precedence
1. **Precedence Hierarchy**:
   ```text
   CANONICAL_DECISION_STATE
     >
   ACTIONABILITY
     >
   PRE_FLIGHT_CHECKS
   ```
   $$\text{CANONICAL\_DECISION\_STATE} \succ \text{ACTIONABILITY} \succ \text{PRE\_FLIGHT\_CHECKS}$$
2. **Clearance Invariant**:
   If `isDecisionActionable == false` (e.g., state is `VALID_SETUP`, `WAITING_PULLBACK`, `STAGE_4_CORRECTION`, `UNVERIFIED`), Pre-Flight **CANNOT** grant trade clearance (`isCleared = false`), regardless of whether all 5 checklist conditions evaluate to true.
3. **No Secondary Decision Engine**:
   Pre-Flight does not recalculate Bayesian confluence scores, swing corridors, or Minervini stages. It validates that execution prerequisites (e.g. R:R $\ge 2.0$, trend corridor proximity, capital health, catalyst buffer, macro calmness) are satisfied.
4. **Copy Demarcation (Already Deployed in Production Baseline `d194c15`)**:
   Pre-Flight copy strictly explains confirmation status without engine jargon:
   - **Plain English Subtitle**: *"5-point risk and confirmation checklist before entering this position."*
   - **Non-Actionable Banner (Plain English)**:  
     Title: `TRADE NOT CLEARED: AWAITING CONFIRMATION`  
     Body: *"A valid setup is forming for {symbol}, but the entry trigger has not confirmed yet. This checklist reviews setup and risk conditions; it does not bypass the required confirmation trigger."*
   - **Non-Actionable Banner (Pro Quant)**:  
     Title: `NON-ACTIONABLE STATE: AWAITING TRIGGER CONFIRMATION`  
     Body: *"Valid setup structure is present ({decisionState}), but execution criteria remain incomplete. Current state: WAIT FOR TRIGGER. Sizing and execution remain locked until confirmation conditions are satisfied."*
   - **Technical Provenance**: Collapsed progressive disclosure `<details>` element.

---

## 3. Specification of the Five Pre-Flight Checks

| Check ID | User Question | Domain Inputs | Authoritative Source | Freshness Threshold | Pass Rule | Fail Rule | Unavailable / Partial Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CHK-1: R:R** | *"Is the prospective reward at least double my risk?"* | `currentPrice`, `stopLoss`, `takeProfit1`, `riskRewardRatio` | `OptimalExecutionPlan` / TradingView Bar | Intraday (<15m) or Pinned EOD | `safeRR >= 2.0` AND `hasStopLoss` AND `hasTakeProfit` | `safeRR < 2.0` | Price, stop, or target is `null` or $\le 0$ |
| **CHK-2: Trend** | *"Is price in a healthy uptrend and inside the buy corridor?"* | `currentPrice`, `optimalEntryMin`, `optimalEntryMax`, `stagePhase`, `isStage4` | `OptimalExecutionPlan` / Minervini Analyzer | Intraday (<15m) or Pinned EOD | Not Stage 4 AND Price $\le$ EntryMax $\times 1.02$ AND Entry corridor defined | Stage 4 OR Price extended >2% above EntryMax | Entry levels `null` / uncalculated |
| **CHK-3: Capital Risk** | *"Does short-interest and company-quality evidence indicate elevated structural risk?"* | `shortFloat`, `qualityScore`, `verdict` | `MASTER_ASSET_CATALOG` / Fundamental Metrics | Catalog quarterly | `shortFloat <= 12%` AND `verdict` not "Turnaround" AND `qualityScore >= 60` | `shortFloat > 12%` OR `verdict` "Turnaround" OR `qualityScore < 60` | Evaluated via `CHECK_3_EVIDENCE_STATE` triad (Fail Closed) |
| **CHK-4: Catalyst** | *"Is there an imminent earnings report or binary event in the next 7 days?"* | `catalysts[]`, `expectedDate`, `category`, `hasImminentEarnings` | Corporate Calendar / SEC EDGAR | 24 Hours | Event date > 7 days away OR no binary events scheduled | Binary event (Earnings, FDA) within $\le 7$ days | Calendar feed unverified |
| **CHK-5: Macro** | *"Is broad market volatility calm enough to support new risk?"* | `vixLevel`, `regime` | `GET /api/v1/macro/ribbon` (`vix.value`) | Intraday (<15m) or Pinned EOD | `vixLevel < 26.0` | `vixLevel >= 26.0` | `vixLevel` is `null`, `undefined`, or upstream outage (`UNAVAILABLE`) |

### 3.1 Check 3 Final Epistemic Contract (`CHK-3-CAPITAL-RISK`)
- **CHECK_ID**: `CHK-3-CAPITAL-RISK`
- **USER_QUESTION**: *"Does short-interest and company-quality evidence indicate elevated structural risk?"*
- **Domain Inputs**: `shortFloat`, `qualityScore`, `verdict`
- **Title (Plain English)**: *"3. Capital Health & Squeeze Risk (Manageable short float & solvent balance sheet)"*
- **Title (Pro Quant)**: *"3. Structural Capital Risk (Short Interest Floor & Quality Rating)"*
- **Passed Copy (Plain English)**: *"Short interest is moderate (<12%) and financial quality metrics indicate stable balance sheet health."*
- **Passed Copy (Pro Quant)**: *"Capital Risk Guard: Short float (<12%) and fundamental solvency score confirm absence of acute balance sheet distress."*
- **Failed Copy (Plain English)**: *"⚠️ Warning: Elevated short interest (>12%) or turnaround rating indicates potential capital distress."*
- **Failed Copy (Pro Quant)**: *"⚠️ Capital Risk Warning: Elevated short interest (>12%), turnaround status, or sub-60 quality score detected."*
- **Unavailable Copy**: *"Capital risk evidence unavailable: Financial quality metrics unverified."*
- **Mandatory Invariant**:
  ```text
  CLAIM_SET ⊆ EVIDENCE_SET
  ```
  The following terms are **STRICTLY FORBIDDEN** from Check 3 user-facing copy:
  - `institutional accumulation`
  - `institutional unloading`
  - `institutional order flow`
  - `Congressional accumulation`
  - `options sweeps`
  - `smart money accumulation`
- **Provenance of `isDistributionTrap`**:
  `isDistributionTrap` in `OptimalEntryExitCard.tsx` historically attempted to reference options flow. Because single-asset options flow is unconnected, the check actually evaluated catalog fundamentals (`shortFloat > 12.0`, `turnaround` verdict, `qualityScore < 60`). Check 3 is therefore epistemically scoped strictly to **Capital Risk**.

### 3.2 Check 3 Unavailable & Partial Evidence Contract (Fail Closed)
```text
CHECK_3_EVIDENCE_STATE =
  COMPLETE
  | PARTIAL
  | UNAVAILABLE
```
Where:
- `COMPLETE`: All required evidence inputs (`shortFloat` AND `qualityScore`) are present.
- `PARTIAL`: Some but not all required evidence inputs are present (e.g., `shortFloat` is present but `qualityScore` is missing, or vice versa).
- `UNAVAILABLE`: None of the required evidence inputs are available.

**Mandatory Fail-Closed Rule**:
```text
PARTIAL_EVIDENCE =>
  NOT_CERTIFIED
```
If the UI or validator encounters `PARTIAL` evidence, it must **NEVER** silently treat the missing inputs as a complete PASS. Check 3 cannot be certified without all required inputs and must fail closed to `NOT_CERTIFIED` / `UNAVAILABLE`.

### 3.3 Check 5 VIX Binding & Forbidden Fallbacks
- If VIX observation is missing or upstream is degraded, Check 5 status is `UNAVAILABLE`.
- It is **STRICTLY FORBIDDEN** to inject synthetic numbers:
  ```text
  28.0
  99.0
  ```
  as user-visible fallbacks. Missing volatility must be surfaced honestly as `UNAVAILABLE`.

---

## 4. Macro / VIX Authority Contract

### 4.1 Single Canonical Authority
- **Canonical Endpoint**: `GET /api/v1/macro/ribbon` (`api/routes/macro.py`).
- **Telemetry Field**: `vix.value` (or `vix.level`), accompanied by `dataSource` and `observationTime`.
- **Consumer Parity**: Both `MarketCommandRibbon.tsx` and `PreFlightChecklistModal.tsx` (via `OptimalEntryExitCard.tsx`) must bind to this single observation:
  ```text
  PreFlight VIX observation == Market Command Ribbon VIX observation
  ```

### 4.2 Allowed States
```text
CANONICAL_VIX_STATES =
  LIVE
  | STALE
  | UNAVAILABLE
```
1. `LIVE`: Exchange open, observation age $< 15\text{ minutes}$.
2. `STALE`: Exchange closed / weekend, observation pinned to previous settlement.
3. `UNAVAILABLE`: Upstream feed degraded or offline. Pre-Flight renders:
   *"Market Volatility: Volatility evidence unavailable from live exchange feed. Macro guard cannot be certified."*

---

## 5. Instrument-Aware Evidence Applicability Contract

### 5.1 Canonical Security Master Taxonomy
ARX Terminal relies strictly on the canonical server security master (`analyst_dashboard/security_master`):
- `COMMON_STOCK`: Standard operating company equity (e.g. AAPL, NVDA, TSLA).
- `ETF_OR_REGISTERED_FUND`: Open-end investment company, ETF, or unit trust (e.g. SPY, QQQ, XLK, IWM).
- `ADR`: American Depositary Receipt for foreign issuers (e.g. TSM, BABA, ASML).
- `REIT`: Real Estate Investment Trust with specialized capital distribution metrics.
- `OTHER_SUPPORTED_SECURITY`: Preferred shares or closed-end funds with catalog coverage.
- `UNKNOWN`: Unclassified security $\implies$ Fail-closed.

### 5.2 Regulatory & Evidence Matrix by Instrument Class
```text
EVIDENCE_REQUIREMENT =
  f(canonical_instrument_class, regulatory_structure)
```

| Instrument Class | Regulatory Structure | Required Evidence | Optional Evidence | Not Applicable Evidence | Unknown / Incomplete Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **COMMON_STOCK** | Securities Act of 1933 / Exchange Act of 1934 | Corporate Financial Statements (SEC Form 10-K, 10-Q, 8-K; Operating Margins, ROIC, Revenue Growth, Solvency) | Form 4 Insider Filings, 13F Holdings | None | `EVIDENCE_INCOMPLETE` ("Audited SEC EDGAR 10-K/10-Q financial filings are unverified") |
| **ETF_OR_REGISTERED_FUND** | Investment Company Act of 1940 (Form N-CSR, N-PORT) | Fund Structure Evidence (AUM Liquidity, Net Expense Ratio, Benchmark Beta, Index Momentum) | Tracking Error, Sector Weights | **SEC Form 10-K, 10-Q, Form 4 C-Suite Insiders, Operating Margin, ROIC** | `EVIDENCE_INCOMPLETE` ("Fund structure profile unverified; AUM and benchmark tracking required") |
| **ADR** | Foreign Private Issuer (SEC Form 20-F, 6-K) | Foreign Issuer Audited Statements (Form 20-F, 6-K) or Catalog Fundamentals | FX Regime, Home-Market Liquidity | US Domestic Form 10-K, 10-Q | `EVIDENCE_INCOMPLETE` ("Foreign issuer Form 20-F/6-K disclosures unverified") |
| **REIT** | Internal Revenue Code Section 856 / Exchange Act | REIT Financial Statements (FFO / AFFO, Debt/EBITDA, Occupancy Rate) | Property Portfolio Geographic Spread | Traditional Operating Gross Margin | `EVIDENCE_INCOMPLETE` ("REIT statutory filings unverified") |
| **OTHER_SUPPORTED_SECURITY** | Specialized Entity | Par Value, Coupon, Liquidity Profile | Credit Rating | Minervini Momentum, High EPS Growth | `EVIDENCE_INCOMPLETE` |
| **UNKNOWN** | Unclassified Venue | All | None | None | **FAIL_CLOSED** (`isActionable = false`, `state = UNVERIFIED`) |

### 5.3 Mandatory Applicability Invariants
```text
NOT_APPLICABLE != MISSING
NOT_APPLICABLE != UNVERIFIED
NOT_APPLICABLE != FAILED

NO_INSTRUMENT_MAY_BE_FAILED_FOR_EVIDENCE_CLASSIFIED_NOT_APPLICABLE
```
1. **Canonical Routing**: Implementation must use canonical routing (`get_required_evidence_for_instrument(security_type)`) rather than an ad-hoc duplicated `if ETF` check.
2. **Prohibition of False Disqualification**: An ETF must **NEVER** be disqualified with:
   *"Execution withheld: Core SEC Form 10-Q/10-K financial filings are unverified."*
3. **Fund Profile Explanation**: For ETFs and Registered Funds, the explanation layer states:
   *"Fund / ETF Profile: Evaluated via fund liquidity, net expense ratio, and underlying index momentum. Corporate 10-K financial filings are not applicable."*

---

## 6. Smart Money Capability Model & Recency Contract

### 6.1 Subsystem Capability Matrix

| Capability | Status | Source Provider | Scope | Presentation Semantic |
| :--- | :--- | :--- | :--- | :--- |
| **CONGRESSIONAL_DISCLOSURES** | `ACTIVE_ARCHIVE` | US House & Senate STOCK Act | Historical Curated Dataset (August 2026) | Rendered with visible archive/staleness date badges. |
| **INSIDER_FILINGS** | `ACTIVE_API` | SEC EDGAR Form 4 Public API | Covered single-asset symbols | Rendered with filing timestamp and net buy/sell ratio. |
| **OPTIONS_FLOW** | `SOURCE_UNAVAILABLE` | Institutional Options Tape | Live exchange websocket required | Rendered as `FEED_UNAVAILABLE` (not empty). |
| **UNIVERSE_SCREENING** | `PIPELINE_PENDING` | Batch Screening Worker | Market-wide scanning | Rendered as `"Universe Scanner Pending"` on Radar. |
| **SINGLE_ASSET_ANALYSIS** | `ACTIVE` | Master Catalog & SEC Feeds | Individual covered symbols | Accessible via `/smart-money` and Asset Analysis hub. |

### 6.2 Immutability of Observed Event Dates
```text
OBSERVED_EVENT_DATE = IMMUTABLE
```
Historical trades and filings must retain their actual statutory occurrence dates.
- It is strictly **FORBIDDEN** to synthesize, roll forward, or shift historical dates to fit within an arbitrary sliding window (e.g. 30D).
- `30D_COUNT`:
  $$\text{30D\_COUNT} = \sum \mathbf{1}_{\{\text{today} - \text{actual\_filing\_date} \le 30\text{ days}\}}$$
- If all curated disclosures are older than 30 days:
  - $\text{30D\_COUNT} = 0$.
  - The UI must render an **Archived Disclosures** container clearly labeled:
    *"Curated Historical Archive (Filing Dates: August 2026). Zero filings in active 30-day window."*
  - Top "Actionable Radar Assets" cards must display the authentic filing date and an `ARCHIVED` badge, rather than silently masquerading as recent trades.

### 6.3 Single-Asset Empty vs. Unavailable Semantics
When `fetchAssetAnalytics(symbol)` is executed:
- If backend has not wired the specific ticker's smart money stream, the payload must return:
  `smartMoney: { status: "PIPELINE_PENDING", congressTrades: [], optionsFlow: [] }`
- The UI must distinguish:
  - `status === "PIPELINE_PENDING"`: *"Smart money tracking is pending ingestion for this symbol."*
  - `status === "SOURCE_UNAVAILABLE"`: *"Options flow provider feed currently offline."*
  - `status === "AVAILABLE" && count === 0`: *"Zero insider or congressional filings reported in monitored window."*

---

## 7. Reconciled Production Baseline (`d194c157c9c3577d3ad5329c4b228f25525292be`)

The following copy remediations have already been verified, merged to `main`, pushed to `origin`, and deployed to production (Railway backend `4c56936e` + Cloudflare Pages `finance-xp8`):

```text
ALREADY_COMPLETE =
  PRESERVE
```

| Component / Feature | Production Implementation | Classification | Action in Wave 3 |
| :--- | :--- | :---: | :--- |
| **Pre-Flight Plain Subtitle** | *"5-point risk and confirmation checklist before entering this position."* | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |
| **Non-Actionable Banner (Plain)** | `TRADE NOT CLEARED: AWAITING CONFIRMATION` + trigger confirmation explanation | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |
| **Non-Actionable Banner (Pro)** | `NON-ACTIONABLE STATE: AWAITING TRIGGER CONFIRMATION` + execution lock explanation | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |
| **Technical Provenance** | Collapsed `<details>` disclosure element | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |
| **Radar Smart Money Tab** | `🐋 Smart Money Universe Scanner Pending` badge + explanatory empty state | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |
| **Radar Minervini VCP Tab** | `⚡ Minervini VCP Universe Scanner Pending` badge + explanatory empty state | `ALREADY_COMPLETE` | **PRESERVE** — Do not recreate or modify. |

Wave 3 builds directly on top of this verified baseline and implements only the remaining epistemic, VIX, instrument-routing, and smart money recency contracts.

---

## 8. Rescoped Wave 3 Implementation Scope

### 8.1 Wave Title
**SYNTHESIS E WAVE 3 — DECISION INTEGRITY & EPISTEMIC CONSISTENCY**

### 8.2 Scope Inclusions (Authorized for Wave 3 Implementation)
- **A — Pre-Flight Epistemic Integrity**: Implement the final `CHK-3-CAPITAL-RISK` evidence/copy contract and `CHECK_3_EVIDENCE_STATE` fail-closed logic. Do not alter canonical actionability rules.
- **B — Canonical VIX Binding**: Remove derived/static fallback VIX values (`28.0`, `99.0`) from Pre-Flight. Bind Pre-Flight directly to canonical macro ribbon observation `GET /api/v1/macro/ribbon`.
- **C — Instrument-Aware Evidence Applicability**: Route evidence requirements through canonical security master (`analyst_dashboard/security_master`). Correct inappropriate stock-specific filing requirements for funds, ADRs, REITs, and other supported classes.
- **D — Smart Money Recency / Archive Semantics**: Preserve real event dates. Separate `RECENT`, `ARCHIVED`, `EMPTY_RESULT`, `SOURCE_UNAVAILABLE`, `PIPELINE_PENDING`. Fix contradictory count/card presentation.
- **E — Single-Asset Smart Money State Semantics**: Replace ambiguous bare empty arrays where necessary with explicit capability status. Do not fabricate activity.

### 8.3 Scope Exclusions (Strictly Frozen & Prohibited)
The following are strictly **OUT OF SCOPE** and frozen:
- `quantitative scoring weights`
- `confluence formulas`
- `position sizing`
- `Minervini stage definitions`
- `recommendation thresholds`
- `prospective validation evidence`
- `learning/tuning`
- `database schema`
- `new OPRA provider integration`
- `new universe scanner implementation`
- `original navbar/onboarding Wave 3 scope`

---

## 9. Acceptance Criteria & Test Verification Matrix

- **AC-W3-EP-001**: Every factual Pre-Flight claim is directly supported by evaluated evidence ($\text{CLAIM\_SET} \subseteq \text{EVIDENCE\_SET}$).
- **AC-W3-EP-002**: Pre-Flight Check 3 emits zero institutional/options/Congressional-flow claims unless those domains are genuinely evaluated.
- **AC-W3-EP-003**: Partial/missing required Check 3 evidence never silently produces fully-certified PASS semantics (`PARTIAL_EVIDENCE => NOT_CERTIFIED`).
- **AC-W3-MACRO-001**: All user-visible VIX observations resolve through one canonical authority (`GET /api/v1/macro/ribbon`).
- **AC-W3-MACRO-002**: Unavailable VIX produces `UNAVAILABLE`, never a fabricated numeric fallback (`28.0`, `99.0`).
- **AC-W3-INSTRUMENT-001**: No instrument fails because of evidence classified `NOT_APPLICABLE` by its canonical instrument evidence contract.
- **AC-W3-INSTRUMENT-002**: Instrument applicability is resolved through canonical security classification (`analyst_dashboard/security_master`), not duplicated ad-hoc assumptions.
- **AC-W3-SM-001**: Rolling-window counts use immutable authentic event dates ($\text{30D\_COUNT} = 0$ for August 2026 curated trades).
- **AC-W3-SM-002**: Out-of-window historical evidence is explicitly archived/stale with visible badges.
- **AC-W3-SM-003**: `ZERO`, `VALID_EMPTY`, `SOURCE_UNAVAILABLE`, `PIPELINE_PENDING`, and `STALE` remain distinct states.
- **AC-W3-ARCH-001**: No quantitative model, scoring, sizing or prospective-validation state changes.
- **AC-W3-PRESERVE-001**: Frozen Wave 1 invariants remain intact (`Wave1DeclutterPreservation.test.tsx`).
- **AC-W3-PRESERVE-002**: Frozen Wave 2 first-viewport and chart-refit behavior remains intact (`Wave2FirstViewportComposition.test.tsx`).
- **AC-W3-PRESERVE-003**: Already-deployed Pre-Flight/Radar copy remediation remains intact (`preflightRadarCopyRemediation.test.ts`).

---

*Ratified by Quantitative Engineering, UX Architecture & Product Governance.*
