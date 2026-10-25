# ARX TERMINAL — SPRINT 2 / TRACK C
# UNRESOLVED DEPENDENCIES & CROSS-TRACK BOUNDARIES
## Deliverable C11: External Blockers, Cross-Track Preconditions, Ownership Ledger & Activation Criteria

> [!IMPORTANT]
> **Authority:** ARX Closure Model v2 — Sprint 2 Track C Foundation Review.  
> **Status:** Authoritative Ledger of Unresolved External Dependencies. Zero unilateral assumptions permitted.

---

### 1. Executive Summary

Track C has completed local contract reconciliation, non-production implementation of the 7-state publication state machine, and verified the isolated publication foundation path.

However, moving from a verified local foundation to integrated production publication remains bounded by external dependencies owned by other tracks. This deliverable documents each unresolved dependency, its owning track, the exact failure mode if breached, and the objective activation criteria required for clearance.

---

### 2. Comprehensive Dependency Ledger

```text
========================================================================================================================
DEP ID  DEPENDENCY NAME             OWNING TRACK    CURRENT STATUS    IMPACT ON TRACK C
========================================================================================================================
UD-C01  Package B Ingestion &       Track B         B1_HISTORICAL_    Historical Alpaca SIP daily bars qualified under
        Entitlements Status                         SIP_READY_WITH_   delay policy (MAX_COMPLETED_SESSION=2026-10-19).
                                                    DELAY_CONTRACT    Real-time SIP restricted (403). B2/B3 pending.
                                                                      Production ingestion explicitly NOT authorized.
------------------------------------------------------------------------------------------------------------------------
UD-C02  B2 Identity-Projection      Track B         UNRECONCILED /    Blocks automated bulk candidate projection into
        Pipeline Execution                          PROVISIONAL       Track C publication format.
------------------------------------------------------------------------------------------------------------------------
UD-C03  Track G12 Gate 12           Track G12       ACTIVATION_       VCP Holdout Epoch 002 custody governance (G-VCP-012)
        Certification (VCP-Only)                    DEFERRED          blocks VCP strategy admission. Non-VCP strategies
                                                                      isolated under own qualification authorities.
------------------------------------------------------------------------------------------------------------------------
UD-C04  Rule R06 Primary-Listing    Track A / PO    POLICY_UNRESOLVED Gating rule R06 requires verified primary listing
        Evidence for Remaining 5                                      evidence before FIX, DUOL, ULTA, KLAC, DECK admit.
------------------------------------------------------------------------------------------------------------------------
UD-C05  Operational Scanner         Infrastructure  DORMANT           Governed by G-S2-ACT operational activation.
        Scheduler Dispatch                          / DISABLED        Neither VCP nor non-VCP strategy admission
                                                                      authorizes scanner enrollment without G-S2-ACT.
========================================================================================================================
```

---

### 3. Detailed Analysis of Dependencies

#### 3.1. UD-C01: Package B Ingestion Status & Entitlement Reconciliation
- **Owning Track:** Track B (Market Data Ingestion).
- **Current State:** `B1_HISTORICAL_SIP_READY_WITH_DELAY_CONTRACT` (prior status: `PACKAGE_B_READINESS_HOLD`).
- **Qualification Details:**
  - **B1 Historical Alpaca SIP Daily Bars:** Available and verified under the deterministic market-delay policy (`MAX_COMPLETED_SESSION = 2026-10-19`).
  - **Real-Time SIP Feed Restriction:** Explicitly restricted; HTTP 403 / unentitled on live WebSocket stream.
  - **B2 Identity Projection:** Remains pending authoritative execution/export matching `ARX_SEC_[0-9A-F]{12}`.
  - **B3 Pilot Safety Reconciliation:** Remains pending final certification report.
  - **Production Ingestion Boundary:** Explicitly **NOT** authorized by B1 acceptance. Ingestion into production stores remains dormant.
- **Impact on Track C:** Track C can qualify historical data schemas and delayed simulation paths for candidates up to session 2026-10-19, but live real-time feeds and production ingestion remain firewall-blocked.
- **Required Clearance Criterion:** Production certification of Package B with resolved credentials and completed backfill for target population.

#### 3.2. UD-C02: B2 Identity-Projection Pipeline Execution
- **Owning Track:** Track B.
- **Current State:** Provisional interface defined in `C_B_INTERFACE_RECONCILIATION.md`.
- **Impact on Track C:** Bulk ingestion outputs from B2 have not been generated in production. Track C tested against local fixtures and the 3 pilots.
- **Required Clearance Criterion:** Production execution of B2 identity projection matching `internal_security_id` format `ARX_SEC_[0-9A-F]{12}`.

#### 3.3. UD-C03: Track G12 Gate 12 Certification & Strategy Scope Reconciliation
- **Owning Track:** Track G12 (Model Governance & Quantitative Invariants).
- **Current State:** `G12_PREPARATION_READY_ACTIVATION_DEFERRED`.
- **Strategy Scope Reconciliation (Task 1):**
  - Gate 12 (`G-VCP-012`) is strictly the VCP Holdout Epoch 002 custody governance mechanism.
  - It blocks VCP strategy admission (`is_vcp_strategy() == True`) when uncertified.
  - Non-VCP strategies admit under their own strategy qualification authority without inheriting deferred VCP custody blockers.
- **Impact on Track C:** VCP strategies cannot admit published securities into State 6 (`STRATEGY_ADMITTED`) without Gate 12 sign-off. Non-VCP strategies may admit under their respective certified authorities.
- **Required Clearance Criterion:** Explicit PO and Quantitative Lead activation of Gate 12 for VCP strategy admission.

#### 3.4. UD-C04: Rule R06 Primary-Listing Evidence for Additional Candidates
- **Owning Track:** Track A / Product Owner.
- **Current State:** The 3 pilots (`AAPL`, `GOOG`, `GOOGL`) have verified primary listing associations. The additional 5 candidates (`FIX`, `DUOL`, `ULTA`, `KLAC`, `DECK`) remain unqualified pending authoritative primary venue evidence.
- **Impact on Track C:** The 5 additional candidates are quarantined in `UNRESOLVED` state during dry runs.
- **Required Clearance Criterion:** Approved registration plan evidence providing primary listing proof for the 5 candidates.

#### 3.5. UD-C05: Operational Scanner Scheduler Dispatch (G-S2-ACT)
- **Owning Track:** Operational Infrastructure.
- **Current State:** Dormant / Disabled (`G-S2-ACT` unactivated).
- **Governance Invariant:** Neither VCP strategy admission (under `G-VCP-012`) nor non-VCP strategy admission authorizes operational scanner enrollment under Sprint 2. Advancing from State 6 (`STRATEGY_ADMITTED`) to State 7 (`ACTIVELY_SCANNED`) strictly requires explicit operational activation authority `G-S2-ACT`.
- **Impact on Track C:** State 6 cannot transition to State 7.
- **Required Clearance Criterion:** Operational activation authorization (`G-S2-ACT`) following verified End-to-End staging rehearsal.

---

### 4. Risk Mitigation & Anti-Silo Governance

Track C has engineered complete local isolation to prevent these unresolved dependencies from blocking architectural progress:
1. **Provisional Interfaces:** Explicit provisional types decouple Track C from Track B execution delays.
2. **Fail-Closed Firewalls:** The state machine strictly aborts any forward transition where external evidence is absent.
3. **Population Isolation:** The 3 pilots provide a fully working reference population while all other securities remain safely quarantined.
