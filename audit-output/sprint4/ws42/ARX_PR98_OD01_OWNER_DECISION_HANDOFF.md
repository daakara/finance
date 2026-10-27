# ARX PR #98 OD-01 Catalyst Policy Owner Decision Handoff

**Directive ID:** `ARX_S4_PR98_UX_INTEGRATION_AND_ACCEPTANCE_020`  
**Repository:** `daakara/finance`  
**Primary PR:** [#98](https://github.com/daakara/finance/pull/98) (`feat/synthesis-e-v3-ux-parity-clean`)  
**Topic:** Policy Gate OD-01 — Corporate Catalyst Verification Policy  
**Authority:** Product/Research Owner Exclusively  
**Current Engineering Status:** `OWNER_PENDING` (Fail-Closed Preserved)  
**Date:** 2026-10-27  

---

## 1. Context & Purpose

During Sprint 4 architectural auditing, a key policy question was surfaced regarding Pre-Flight Check 4 (Corporate Event Runway & Earnings Volatility Risk): **What standard of corporate event evidence is required to clear an asset for tactical execution?**

PR #98 implements complete UX presentation support for catalyst verification, but governance mandates that the underlying policy decision between strict regulatory verification (Option A) versus tiered commercial aggregator ingestion (Option B) belongs solely to the Product/Research Owner.

This document prepares the formal decision package for Owner ratification.

---

## 2. Policy Options Analysis

### Option A: Strict Regulatory SEC Filings (EDGAR-Only)
- **Mechanism:** Corporate events are recognized solely upon official SEC EDGAR filings (Form 8-K, Form 10-Q/10-K filing date notifications) or direct primary exchange press releases.
- **Advantages:**
  - Zero vulnerability to third-party scraping errors, estimated date shifts, or unconfirmed calendar placeholders.
  - 100% legal and regulatory auditability.
- **Disadvantages / Trade-offs:**
  - Coverage constrained to ~55–65% of mid/small-cap universe until formal 8-K is lodged (often only 1–2 weeks prior to earnings).
  - Forces many fundamentally attractive setups into `CATALYST_EVENT_UNVERIFIED` fail-closed holds.

### Option B: Tiered Aggregator Hybrid (Tiered Confidence Ladder)
- **Mechanism:** Implements a three-tier catalyst confidence ladder:
  - **Tier 1 (CONFIRMED):** Official SEC filing or issuer press release $\rightarrow$ Full clearance granted.
  - **Tier 2 (EXCHANGE_ESTIMATED):** Primary exchange or Bloomberg/Refinitiv published date $\rightarrow$ Conditional clearance with "Estimated Date" badge.
  - **Tier 3 (ALGORITHMIC_PROJECTION):** Historical pattern projection without issuer confirmation $\rightarrow$ Prohibits clearance within 14 days of projected window.
- **Advantages:**
  - Expands universe coverage to >95% of liquid equities.
  - Transparently exposes data pedigree to quantitative users.
- **Disadvantages / Trade-offs:**
  - Requires maintaining confidence tier metadata in Security Master.
  - Requires ongoing monitoring of date revisions from aggregators.

---

## 3. Current Implementation Baseline in Reconciled PR #98

In PR #98 (`c25b03d3`), the frontend is constructed to be **100% forward-compatible with both options** while defaulting to strict fail-closed safety:

1. **Explicit Telemetry Typing:**
   In `frontend/lib/preflightEvaluator.ts` and `PreFlightChecklistModal.tsx`:
   - If `hasImminentEarnings === true` $\rightarrow$ Check 4 Fails (`BLOCKED: Imminent Earnings`).
   - If `hasImminentEarnings === null` (unverified telemetry) $\rightarrow$ Check 4 Fails (`BLOCKED: Corporate Event Schedule Unavailable`).
   - If `hasImminentEarnings === false` $\rightarrow$ Check 4 Passes (`CLEARED: No Imminent Earnings within 7 Days`).
2. **Pedigree Rendering:** The UI renders an explicit *"Source: Confirmed SEC Filing"* or *"Source: Unconfirmed Estimated Runway"* banner depending on the backend metadata envelope.
3. **Fail-Closed Default:** If the backend provides no confirmation status, the UI conservatively withholds execution clearance.

---

## 4. Engineering Recommendation & Owner Decision Prompt

### Recommendation:
1. **Merge PR #98 with current fail-closed baseline.** PR #98 requires zero code modifications regardless of whether Option A or Option B is selected by the Owner, because the UI components already faithfully render both strict and tiered states.
2. **Authorize Option B for Sprint 5 implementation.** Option B aligns with institutional terminal best practices by distinguishing confirmed vs. estimated dates without artificially blinding the trader to widely known projected earnings cycles.

### Owner Action Required:
Please indicate your binding decision:
- [ ] **Option A (Strict SEC Only):** Reject all non-EDGAR event dates. Assets without formal 8-K remain blocked from execution clearance.
- [ ] **Option B (Tiered Hybrid):** Authorize exchange and aggregator feeds under explicit confidence tiering (Tier 1 Confirmed vs Tier 2 Estimated).
- [ ] **Defer to Sprint 5 Planning:** Maintain existing fail-closed baseline; ratify catalyst policy during Sprint 5 governance session.
