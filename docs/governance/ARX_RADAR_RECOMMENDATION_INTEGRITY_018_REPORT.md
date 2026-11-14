# ARX-RADAR-RECOMMENDATION-INTEGRITY-018
## Surgical Radar 45/100 Recommendation Integrity & Qualification Report

**Directive:** `ARX-RADAR-RECOMMENDATION-INTEGRITY-018`  
**System:** ARX Terminal (`daakara/finance`)  
**Production Release Baseline:** `32924dbe247b3aad582f23d68021468c95a56523`  
**Candidate PR:** [#279](https://github.com/daakara/finance/pull/279)  
**Evaluation Timestamp:** 2026-11-14T08:23:00Z  
**Classification:** P1 — Recommendation Integrity  

---

## 1. Executive Summary

This release surgically corrects misleading presentation in the Radar discovery scanner where relatively weak Confluence conviction scores (e.g. 45/100) were styled in bright emerald green (`text-emerald-400 bg-emerald-950/60 border border-emerald-800/60`) and displayed an actionable `READY TO BUY` screening badge. 

The institutional boundary is strictly preserved:
$$\text{Relative Discovery Ranking} \neq \text{Absolute Confluence Conviction} \neq \text{Trade Actionability}$$

---

## 2. Root Cause & Pre-Existing Defects

1. **Unconditional Emerald Score Badge:**
   In `frontend/app/radar/page.tsx` (lines 1964-1970 desktop table, lines 2151-2157 mobile cards, line 1046 hero card), every active candidate—regardless of whether its score was 45/100, 20/100, or 95/100—was styled in bright emerald green (`text-emerald-400 bg-emerald-950/60 border border-emerald-800/60`).
2. **Misleading Screening Copy:**
   `formatRadarExecutionStatus('READY_TO_BUY')` returned literal `'READY TO BUY'`, directly displaying an execution directive on screening candidates where `isActionable: false` is hardcoded.
3. **Ambiguity on Top-Ranked Candidate:**
   When sorting candidates by Confluence Score descending, the top candidate (e.g. 45/100 in an adverse macro tape) appeared as an attractive "buy recommendation" despite lacking canonical trigger clearance and sitting in a defensive score band.

---

## 3. Minimal Surgical Fix

1. **Alignment with Canonical ConfluenceEngine Tiers:**
   Implemented `resolveRadarScorePresentation(score, isDelisted, backendBadgeColor)` adhering to `analyst_dashboard/analyzers/confluence_engine.py`:
   - **80–100:** High Conviction $\to$ Emerald (`text-emerald-400 bg-emerald-950/60 border border-emerald-800/60`)
   - **65–79:** Constructive $\to$ Cyan (`text-cyan-400 bg-cyan-950/60 border border-cyan-800/60`)
   - **49–64:** Selective $\to$ Amber (`text-amber-400 bg-amber-950/60 border border-amber-800/60`)
   - **0–48:** Defensive / Capital Preservation $\to$ Rose (`text-rose-400 bg-rose-950/60 border border-rose-800/60`)
   - **Delisted:** Defensive $\to$ Rose (`text-rose-400 bg-rose-950/60 border border-rose-800/60`)
   - **Null / Negative / Out-of-bounds:** Muted $\to$ (`text-[var(--text-muted)]`)
   - *45/100 Presentation:* Explicitly styled in Rose (`text-rose-400 bg-rose-950/60 border border-rose-800/60`) with tooltip: `Defensive / Sub-Threshold Ranking (45/100) · Relative discovery ranking — not an actionable buy recommendation`.
2. **Elimination of Execution Directives from Screening Copy:**
   `formatRadarExecutionStatus('READY_TO_BUY')` demoted to `'NEAR SCREENING ZONE'` (matching `'IN_BUY_ZONE'`), preventing execution directives from appearing in discovery rows.
3. **Explicit Relative Discovery Qualification:**
   Updated Radar hero subtitle and sort descriptions:
   - Hero subtitle: `"Discovery Candidate · Relative discovery ranking — not an actionable buy recommendation"`
   - Sort explanation: `"Technical Confluence (Confluence Score descending), relative discovery ranking — not an actionable buy recommendation"`
4. **Classification of Earlier Architectural Test Failure:**
   Pre-existing literal string check in `tests/mobileNavigationDefects.test.ts` updated to accept the explicit symbol binding introduced in PR #263 (`<SynthesisMobileDock activeSymbol={hasExplicitSymbol && selectedSymbol ? selectedSymbol : null} />`).

---

## 4. Test & Verification Evidence

1. **Targeted Suite (`tests/radarRecommendationIntegrity.test.ts`):** **8/8 PASS**
   - 1.1 Candidate scoring 45/100 resolves to DEFENSIVE rose presentation with warning tooltip.
   - 2.1 Both desktop table and mobile cards dynamically consume `resolveRadarScorePresentation`.
   - 3.1 All boundaries (48/49, 64/65, 79/80) and edge cases strictly verified.
   - 4.1 `READY_TO_BUY` execution directive demoted to `NEAR SCREENING ZONE`.
   - 5.1 Hardcoded `isActionable: false` and `canSizeTrade: false` verified.
   - 6.1 Relative discovery ranking explicitly distinguished from actionable trade setups.
   - 7.1 Unverified, stale, and delisted statuses fail closed cleanly.
   - 8.1 Backend badgeColor overrides and parity preserved.
2. **Full Frontend Architectural Suite (`npm run test:arch`):** **46/46 PASS (100%)**
3. **TypeScript Typecheck (`npm run type-check`):** **0 errors**
4. **Parity Gate (`npm run test:parity`):** **100% PASS**
5. **Governor Sizing (`npm run test:sizing`):** **17/17 PASS**
6. **Backend Invariant Suite (`pytest`):** **53/53 PASS**

---

## 5. Phase A3 Governance Compliance

Under the Rule Precedence Hierarchy:
- **Level 1 (Security) & Level 5 / UI-0 (Non-bypassable product constraints):** Truthful financial disclosure and elimination of false buy recommendations are non-bypassable safety floors.
- **Phase A3 Scope:** Phase A3 explicitly governs comprehension of page guidance and holds roadmap redesign tracks (`UX_R3_PLUS = BLOCKED_PENDING_A3`). It does not prohibit P1 recommendation-safety corrections on shipped routes (Radar was shipped in Phase A0/A1 and Phase 5 post-launch refinement).
- **Zero Fabrication:** No human study data was fabricated; PR #274 remains in Draft status and held awaiting the 5-participant formative study.

---

## 6. Operational Holds Summary

All production operational holds remain strictly **ACTIVE**:
- `ARX_CONFLUENCE_INTEGRITY_HOLD`: **ACTIVE**
- `HOLD_SEC_001`: **ACTIVE**
- `HOLD_SCAN_001`: **ACTIVE**
- `HOLD_EXEC_001`: **ACTIVE**
- `PHASE_3_HOLD`: **ACTIVE**
- `PHASE_A3_UX_HOLD`: **ACTIVE**
