# ARX-CONFLUENCE-INTEGRITY-001: Audit Findings & Selection-Integrity Remediation

**Audit Identifier:** `ARX-CONFLUENCE-INTEGRITY-001`  
**Execution Date:** 2026-11-11  
**Worktree:** `/Users/iDavid/Projects/finance-confluence-audit`  
**Branch:** `audit/confluence-integrity-001`  
**Base SHA:** `a54a234422d8cdfcd6902f290a9ef7329ed0f131` (`origin/release/production`)  
**Verdict:** `ARX_CONFLUENCE_INTEGRITY_HOLD`

---

## 1. Executive Summary

A forensic audit of the ARX Weekly Top 3 Confluence Spotlight (ONON, MNST, ULTA) identified systemic selection-integrity defects that allowed candidates with incomplete evidence (2 of 4 pillars) and synthetic factor data to be presented as high-conviction setups.

The audit verified that:
1. Missing Smart Money (SEC 13F/insider) and Macro (FRED) data did not penalize candidate scores; scores were computed by renormalizing weights across only 2 pillars, inflating composite scores to 84.4.
2. A pre-invariant R:R label clamp (`max(1.85, rr_ratio)`) in `optimal_execution.py` allowed sub-floor candidates to display compliant reward:risk metrics prior to geometric level enforcement.
3. Factor data for all three candidates originated from identical template fixtures written on 2026-11-09 (`quality=88, piotroski=8, growth=85`).
4. Remediation implements strict coverage-aware score caps (2 pillars capped at 72.0; 3 pillars at 84.0), surfaces evidence provenance and suppression reasons to the UI, and eliminates pre-invariant R:R clamping.
5. Under remediated rules, all three candidates drop from 84.4 / 80.7 to 72.0, correctly failing the 75.0 actionability gate. The current snapshot yields zero actionable candidates (`NO_QUALIFYING_CANDIDATES`), triggering a mandatory governance hold.

---

## 2. Production Candidate Reconstruction (Pre-Fix vs. Post-Fix)

### Pre-Fix Baseline

| Candidate | Pre-Fix Score | Verified Pillars | Coverage Ratio | Incomplete Coverage | Execution Status | isActionable | Planned R:R |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ONON** | 84.4 | 2 / 4 (Tech, Fund) | 50.0% | True (SM, Macro missing) | `IN_BUY_ZONE` | **True** | 2.35 |
| **MNST** | 84.4 | 2 / 4 (Tech, Fund) | 50.0% | True (SM, Macro missing) | `IN_BUY_ZONE_AWAITING_TRIGGER` | False | 2.42 |
| **ULTA** | 80.7 | 2 / 4 (Tech, Fund) | 50.0% | True (SM, Macro missing) | `WAITING_PULLBACK` | False | 2.35 |

### Post-Fix Audit State

| Candidate | Post-Fix Score | Coverage Cap | Verified Pillars | Coverage Ratio | Execution Status | isActionable | Invalidation / Gate Result |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ONON** | **72.0** | 72.0 | 2 / 4 | 50.0% | `IN_BUY_ZONE` | **False** | Failed score gate (< 75.0) |
| **MNST** | **72.0** | 72.0 | 2 / 4 | 50.0% | `IN_BUY_ZONE_AWAITING_TRIGGER` | **False** | Failed score gate (< 75.0) + unconfirmed trigger |
| **ULTA** | **72.0** | 72.0 | 2 / 4 | 50.0% | `WAITING_PULLBACK` | **False** | Failed score gate (< 75.0) + outside buy zone |

---

## 3. Root Cause Analysis

### RC-1: Evidence Renormalization without Coverage Cap
When Smart Money or Macro pillars returned `None`, `confluence_engine.py` renormalized remaining weights to sum to 1.0. A candidate with strong technicals (85.0) and template fundamentals (83.8) achieved an 84.4 score despite complete absence of institutional regulatory validation.

**Remediation:** Enforced hierarchical coverage caps in `confluence_engine.py`:
- 2 pillars verified: Cap at **72.0** (blocks HIGH-CONVICTION badge and >=75 actionability gate).
- 3 pillars verified: Cap at **84.0**.
- 4 pillars verified: Full potential up to **96.0**.

### RC-2: Pre-Invariant R:R Clamping
`optimal_execution.py` applied `max(1.85, rr_ratio)` within `calculate_trade_levels` at `raw_plan` construction. This masked true structural risk/reward before downstream invariant enforcement adjusted take-profit geometry.

**Remediation:** Removed pre-invariant clamp; emitted `risk_reward_ratio` directly reflects underlying TP1/TP2 geometry relative to planned entry and structural stop loss.

### RC-3: Information Asymmetry in UI
`WeeklyConfluenceSpotlight.tsx` displayed `confluenceScore` and setup badges without exposing whether institutional evidence was present or whether the setup was suppressed by execution gates.

**Remediation:** 
- Surfaced `confluenceCoverageRatio`, `confluenceVerifiedPillarCount`, `confluenceIncompleteCoverage`, and `confluenceCoverageCap`.
- Added UI warning badge (`📋 N/4 pillars verified`) and explicit suppression indicator (`⏸ [Reason]`) on spotlight cards.

### RC-4: Synthetic Factor Snapshot Reliance
Factor inputs for ONON, MNST, and ULTA in production were derived from identical template values (`quality_score=88, piotroski=8, growth_score=85`). Fundamental differentiation was entirely absent.

**Remediation:** Identified and flagged for data pipeline remediation. Scoring weight changes intentionally excluded from this PR to preserve comparative baseline.

---

## 4. Code Changes Inventory

1. `analyst_dashboard/analyzers/confluence_engine.py`:
   - Added coverage cap logic: 2 pillars -> 72.0 max, 3 pillars -> 84.0 max, 4 pillars -> 96.0 max.
   - Exposed `incompleteCoverage`, `verifiedPillarCount`, `coverageCap`, and `coverageRatio` in score breakdown.

2. `analyst_dashboard/analyzers/optimal_execution.py`:
   - Removed pre-clamping `max(1.85, rr_ratio)` from raw plan construction.
   - Preserved geometric enforcement in `_enforce_execution_invariants` where TP1 is physically adjusted to meet the 1.85:1 minimum floor.

3. `api/routes/analytics.py`:
   - Enriched `_build_tactical_setup` payload with `confluenceCoverageRatio`, `confluenceVerifiedPillarCount`, `confluenceIncompleteCoverage`, `confluenceCoverageCap`, and `riskRewardActual`.

4. `frontend/components/WeeklyConfluenceSpotlight.tsx`:
   - Mapped new provenance fields through `topCandidates` memoization.
   - Rendered visual coverage indicator and suppression badge on spotlight cards.

5. `tests/test_confluence_integrity_001.py`:
   - 13 qualification tests validating score caps, R:R geometric derivation, fail-closed actionability, deterministic ranking, and operational holds.

---

## 5. Verification & Test Evidence

```
tests/test_confluence_integrity_001.py:
  test_rr_floor_enforced_by_geometry_not_label                   PASSED [  7%]
  test_ulta_style_low_rr_cannot_pass_eligibility_gate            PASSED [ 15%]
  test_high_score_cannot_override_stopped_out_status             PASSED [ 23%]
  test_high_score_cannot_override_severe_liquidity_hazard        PASSED [ 30%]
  test_2_pillar_coverage_capped_at_72                            PASSED [ 38%]
  test_3_pillar_coverage_capped_at_84                            PASSED [ 46%]
  test_full_4_pillar_coverage_retains_96_cap                     PASSED [ 53%]
  test_unavailable_smart_money_does_not_produce_positive_smart_badge PASSED [ 61%]
  test_ranking_deterministic_from_immutable_snapshot             PASSED [ 69%]
  test_two_qualifying_candidates_produce_two_results_not_three   PASSED [ 76%]
  test_non_actionable_candidate_cannot_trigger_live_execution     PASSED [ 84%]
  test_missing_stop_loss_blocks_actionability                    PASSED [ 92%]
  test_operational_holds_not_weakened                            PASSED [100%]

Related Regression Suites:
  tests/test_confluence_engine.py                                5/5   PASSED
  tests/test_optimal_execution.py                                7/7   PASSED
  tests/test_governance_decision_engine_policy.py                8/8   PASSED
  tests/test_canonical_decision_context.py                       7/7   PASSED

Total Suite Pass Rate: 40 / 40 (100%)
```

---

## 6. Release Governance Verdict & Remaining Gates

### Formal Verdict
`ARX_CONFLUENCE_INTEGRITY_HOLD`

### Rationale
While selection-integrity code defects (clamping, coverage caps, UI disclosure) are fully resolved and qualified, the current production universe yields zero qualifying actionable setups due to missing SEC Smart Money ingestion and missing FRED macro telemetry. Promoting these symbols to live execution or active recommendations remains blocked under fail-closed governance.

### Remaining Governance Gates Before Production Promotion
1. **SEC 13F & Insider Filing Ingestion:** Ingest authentic institutional ownership filings for target symbols.
2. **FRED Telemetry Feed:** Configure and verify macroeconomic indicator retrieval in the target environment.
3. **Shadow-Mode Scoring Redesign:** Evaluate proposed weighting adjustments against historical point-in-time walk-forward returns prior to weight modification.
4. **Draft PR Approval:** Formal code review and merge of `audit/confluence-integrity-001` into `release/production`.
