# ARX Terminal — Radar & Investment Research Direction

**Recorded:** 2026-11-07
**Type:** Product and architecture direction / implementation target (NOT production acceptance)
**Source baseline:** `release/production` at `e766cf3c323017989aeda2cbe1a7880ddf4e1bfb`
**Governance:** Existing repository authorities remain in force. This document does not replace executable rules, authorize strategy changes or lift any operational hold.
**Status:** Owner-directed direction; implementation and acceptance pending.

## 1. Product mission and non-goals

ARX Radar is the **Stage 2 equity investment discovery and monitoring entry point** for investors of varying experience. It should identify technically qualified **research candidates** across the eligible market, explain why each candidate appeared, show trustworthy and timely evidence, link consistently to Analysis and Trade Plan, and track what subsequently happened. Stage 2 is a **strategy eligibility** criterion, not proof of financial attractiveness, favorable valuation, successful trading expectancy, portfolio fit, or a recommendation to buy.

A genuine Stage 2 candidate may be worth **watching** without being a **VCP-qualified setup** or a **trade-ready plan**. Primary long-oriented Radar discovery excludes Stage 1, 3 and 4 equities and unresolved stages; diagnostics and separately authorized strategies may retain such data outside the primary discovery projection. Neither user knowledge level nor frontend copy may alter canonical classification or trade actionability.

Non-goals for the next release: new scoring or ML engines, invented win probabilities, automated trading, full UI redesign, universal buy/sell advice, mandatory congressional data, or a new central governance service.

## 2. Confirmed operational evidence versus implementation state

### Accepted production operations — historical receipts (not a fresh production probe)

- **RADAR-014:** promoted 250-symbol reference cohort `UB-ARX-EXPANSION-250-20261107`, containing the original 133 and 117 additional hydrated symbols; provenance-backed historical OHLCV accepted. Not a permanent whitelist.
- **RADAR-015:** completed operator-authorized one-shot scan, run `run-00cc89d2a6b9`, snapshot `vcp-snap-1794040325-1a8a8f46`, 250/250 evaluated, **0 VCP-qualified**. The zero is a qualified-pattern observation, **not** a zero-Stage-2 assertion.
- The separate read-only Stage 2 forensic receipt reported **100 Stage 2 passes**, **150 non-Stage-2 or unresolved** (65 unresolved; 52 Stage 4; 33 Stage 1; 0 Stage 3), and **0 qualified VCP**. These are the receipt's reconstructed observations, not yet an accepted investor-facing Stage 2 publication or independent proof that every record had sufficient data.
- Existing product defects reported: primary `ALL` endpoint admits non-Stage-2 equities; the strict VCP snapshot publisher drops Stage 2 non-VCP stocks from its candidate list; primary endpoint uses legacy 35-symbol catalog; canonical stage provenance is not reliably transported. **Defect diagnosis = FIX_READY, not FIX_DEPLOYED.**
- Reconcile apparent `<200`-session unresolved cases against the previously accepted hydration window and the **effective data cutoff**. Do not assume that ingestion coverage automatically equals classifier-eligible coverage.
- Production source-session time and scan execution/evaluation time must remain distinguishable. A computed wall-clock `data_as_of` is not evidence of latest verified OHLCV session.

### Operating boundaries

`HOLD_SCAN_001` (natural recurring scan), `HOLD_SEC_001` (continuous SEC egress), `HOLD_EXEC_001` (live execution) remain active; Sprint 5 activation remains unauthorized. This document is not an activation directive. The one-shot scan already completed does not authorize a scheduler.

## 3. Canonical ownership / consumption matrix

| Decision | Authoritative producer or policy | Consuming role / rule |
|---|---|---|
| Eligible security and promoted universe | Security Master / Universe contracts and promotion | Radar reads the **active published build**, not a hardcoded 35 or 250 fixture |
| Price, volume, adjustments, source session | Market-data authority and data-quality policy | Scanner verifies sufficient history and quality; distinguish stale, invalid, partial, zero and unavailable |
| Stage 2 | Canonical long-history trend classifier | Primary Radar eligibility; preserve stage reasons, required sessions and rule version |
| VCP | Canonical VCP classifier and publication integrity | Separate strict pattern-qualified subset, not the Stage 2 admission rule |
| Entry, stop, target, R:R | Canonical Trade Plan / Risk calculations | Radar **consumes**, never independently recreates |
| Actionability | `DecisionHierarchyEngine` and applicable policy | Only canonical decision state may represent actionable setup; research status is distinct |
| Confluence, fundamentals, macro, Smart Money | Respective domain engines and evidence applicability | Supporting or limiting evidence; optional absent sources cannot falsely block core trend discovery |
| Portfolio constraints | Portfolio / risk-budget authority | Investor-specific suitability; cannot be inferred from general stock data |
| Order transmission | Financial execution control plane | No authority granted by Radar / Analysis / Trade Plan displays |

Existing source contracts include `analyst_dashboard/governance/decision_context.py`, `analyst_dashboard/analyzers/decision_hierarchy.py`, `analyst_dashboard/analyzers/scanner_contract.py`, `analyst_dashboard/analyzers/scanner_publication_integrity.py`, `docs/architecture/DATA_QUALITY_AND_EDGE_CASE_GOVERNANCE.md`, and `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md`.

**Open policy conflict:** `DecisionHierarchyEngine` enforces R:R >= **2.0** for `ACTIONABLE_SETUP`; the VCP publication specification contains a **1.85** floor. Determine if 1.85 is historical version metadata, screening-only metadata, or live divergent behavior. Do **not** change frozen scoring, historical evidence, or execute trades to reconcile the discrepancy. A 2:1 numeric pass is **necessary only where applicable**, not sufficient for actionability; target/stop assumptions must be defensible.

## 4. Minimal investor-facing research contract

Model this as a **projection of existing source contracts**, not automatically a new table, microservice, or risk engine:

- **Identity:** canonical security ID/ticker, security type, active universe build and eligible status.
- **Evidence:** authoritative source and market-data session, ingestion/scan timestamp, completeness, PIT eligibility, adjustment lineage, stale/unavailable/invalid conditions.
- **Discovery:** Stage 2 passed/failed/unresolved, predicate reasons and classifier version.
- **Setup:** VCP qualified/unqualified/uncomputed; maturity and explicit invalidation or required conditions.
- **Risk & trade:** referenced authoritative entry, stop, target, numeric R:R and canonical decision state; unsupported plan = not established, not pass.
- **Context:** fundamentals, market regime, Smart Money and other applicable sources with independent quality flags. Missing optional observations never become neutral or bullish facts.
- **Portfolio:** only evaluated when an actual portfolio and strategy/risk budget are supplied.
- **Reproducibility:** snapshot ID, source-as-of, first-published timestamp, ruleset/version and subsequent lifecycle transitions, without retrospective overwrite.

Required investor questions: **Why this stock? Why watch? What's missing for readiness? What would invalidate it? How fresh and trustworthy is the evidence?**

Use progressively richer explanation for beginners, intermediate users and experts **without** changing decision conclusions. No universal `BUY` badge based solely on Stage 2, VCP or a composite score.

## 5. Classification and publication truth table

| Condition | Primary Radar | VCP qualified list | Trade-ready claim |
|---|---|---|---|
| Stage 2 verified, VCP not qualified | Research/watch candidate | No | Only if separate canonical plan passes all relevant gates; never infer from stage |
| Stage 2 verified, VCP qualified | Research/setup candidate | Yes | Requires independent full canonical plan qualification |
| Stage 1/3/4 verified | Exclude from primary Stage 2 | No for Stage 2 VCP | No Stage 2 trade-ready claim |
| Stage unresolved or mandatory market data stale | Not a verified Stage 2 candidate; retain diagnostics | No | Fail closed |
| R:R below applicable 2:1 minimum | Can remain research candidate | VCP classification unaffected | No |
| Optional congressional/SEC feed unavailable | Core research retained with limitation | VCP classification unaffected | Must not invent supporting conviction |

Distinct authoritative counters: **eligible universe**, **data qualified**, **actually evaluated**, **Stage 2 pass**, **Stage 2 fail**, **unresolved**, **VCP pass**, **research displayed**, and **execution-ready per canonical plan**. Counts must carry denominator, as-of and filter semantics.

## 6. Evidence source separation and congressional disclosure policy

**SEC Form 4** corporate insider evidence and **Congressional STOCK Act** disclosures are distinct datasets with different provenance, timeliness, source access and analytic meaning.

The Congressional Trades live pipeline remains **unimplemented** according to the reported read-only investigation. The frontend `PIPELINE_PENDING` and explanation that no conclusion about recent transactions is possible are truthful; `0 trades` or verified-no-activity would be false. Historical curated records may not silently become live data or positive confluence. A hold on SEC continuous ingestion does not by itself establish congressional statutory-data coverage or implementation.

**Research-integrity review required before any live congressional launch:** individual source filing verification (not portal homepage alone), transaction/filing/publicly-available/first-observed dates, ownership and transaction-range semantics, reproducible returns and benchmark comparison, and evidence that sentiment, politician win-rate, legislative alignment or `HIGH` confidence labels are empirically justified. The current historical records contain performance and conviction fields whose underlying calculation and provenance are **not yet established**. This is a verification task, not a declaration that named historical claims are accurate.

Do not promote curated observations into live investment conviction, and do not equate filing latency with freshness at publication.

## 7. Release sequence, decision gates and stopping rule

**Wave 1 — focused Stage 2 remediation:** accept completed forensic diagnosis; fix primary list leakage, distinct Stage 2 projection, active universe resolution and stage transport using the canonical long-history classifier; reconcile 65 unresolved cases, session timestamp and 1.85/2.0 policy. Retain strict VCP publication. One coherent PR, targeted tests, no broad rewrite.

**Wave 2 — cross-module investor journey:** Radar -> Analysis -> Trade Plan -> Portfolio context. Reuse existing contract authority; test one Stage 2 non-VCP watch candidate, one R:R 1.5:1 not-trade-ready plan, stale mandatory data, optional source missing and non-Stage-2 exclusion. Clear beginner explanation plus expert evidence drill-down.

**Wave 3 — broader universe and outcome learning:** bounded data-capacity rehearsal, safe expansion beyond 250 toward the eligible market, EOD freshness only with separate scheduler authorization, and immutable prospective observation -> trigger -> invalidation -> outcome measures (MFE/MAE, stop/target, benchmark, costs, PIT). Coverage expansion and outcome capture can run in parallel; neither justifies weakening data quality.

**Change proportionality:** wording changes need focused UI tests; API/filter changes need contract tests; classifier/data provenance changes need version and integrity review; scoring/actionability changes require explicit policy authorization; financial execution requires separate authorization. Avoid duplicate audits and PR proliferation. Stop on failed mandatory gates. Keep actual deployment evidence distinct from proposed work.

## 8. Outstanding decision register

| ID | Decision / gap | State at documentation date | Next evidence |
|---|---|---|---|
| RADAR-D01 | Stage 2 non-VCP publication and non-Stage-2 leakage | Confirmed defect; fix pending | PR, tests and live acceptance |
| RADAR-D02 | Dynamic promoted-universe authority | Confirmed legacy coupling; fix pending | Resolver and denominator parity |
| RADAR-D03 | 65 unresolved vs hydration eligibility | Unreconciled | Source session and classifier input trace |
| GOV-D01 | 1.85 vs 2.0 reward-to-risk | Declared policy divergence | Authority/version trace |
| GOV-D02 | Data-as-of vs execution clock | Potentially misleading code path | Verified watermark source |
| UX-D01 | Integrated investor journey | Target, not accepted deployment | One end-to-end consistent asset example |
| EVID-D01 | Congressional curated claim provenance | Unverified | Filing-level source/calc samples |
| OPS-D01 | Full-universe scale and daily cadence | Deferred / separately authorized | Capacity and data-quality evidence |
| PERF-D01 | Prospective usefulness | Not established | Original publications and forward outcomes |

## 9. Historical precedence and documentation discipline

Historical snapshots and records (including the 2026-10-29 master status, architecture reconciliation, documentation inventory and older production release notes) are **not rewritten as though their date-specific claims were current**. Use the dated addenda and this direction document for later context. Existing executable rules and owner-approved governance policies retain precedence over this target-state document until qualified and released.

**This document records intent and evidence; it is not itself production qualification, investment advice, or permission to merge or deploy.**
