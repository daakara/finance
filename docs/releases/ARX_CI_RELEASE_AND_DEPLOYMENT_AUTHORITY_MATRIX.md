# ARX TERMINAL — CI, RELEASE & DEPLOYMENT AUTHORITY MATRIX
## Release Qualification Framework, GitHub Rulesets, Deployment Boundaries & Governance
**Review Date:** 29 October 2026  
**Mainline Anchor SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Governing Directives:** ARX-036-D3, ARX-036-D4, ARX-036-D4R, ARX-036-D5  
**Canonical Workflow:** `.github/workflows/release-qualification.yml`  
**Document Authority:** `CANONICAL_ACTIVE / RELEASE_GATE`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary & Separation of Responsibilities

ARX Terminal enforces an absolute cryptographic and operational boundary between **Mainline Code Integration** and **Production Release Deployment**:

```
[ Developer / Bot PRs ]
           │
           ▼
[ Four Required Mainline Checks ] ─────────► [ GitHub Ruleset 25187391 (main) ]
  1. Frontend Lint & Typecheck                     │
  2. Frontend Jest / Vitest Suite                  ▼
  3. Backend Pytest Suite              [ MERGE TO MAINLINE ] (e.g. PR #129 e10de3b)
  4. Security & Backlog Validation                 │
                                                   ▼
=========================== MANUAL OWNER PROMOTION BOUNDARY ===========================
                                                   │
                                                   ▼
[ Release Qualification Suite (RQ-01..12) ] ─► [ GitHub Ruleset 26024894 (release/production) ]
  • Frozen Manifest Integrity                      │
  • Cross-Engine Provenance (RQ-05)                ▼
  • Release Security & Isolation (RQ-12)  [ CONTROLLED PRODUCTION PROMOTION ]
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
         [ Cloudflare Pages Frontend ]                           [ Railway Production Backend ]
         • Branch: release/production                            • Commit: de5d2463cc4c6503bd700bcaf2b1bc36fd5e03d9
         • Deployments: MANUAL ONLY                              • Deployments: MANUAL ONLY
         • auto_deploy: FALSE                                    • autoDeployStatus: DISABLED
```

---

## 2. GitHub Rulesets & Branch Protection Policies

### 2.1 Ruleset `25187391` — Mainline (`main`)
- **Protected Branch Pattern:** `refs/heads/main`
- **Enforcement Level:** `ACTIVE`
- **Bypass Actors:** None (Solo Maintainer Fail-Closed Policy; even administrators are bound).
- **Required Pull Request Reviews:** 0 required approving reviews (optimized for authenticated bot / solo maintainer pipeline), but requires signed commit and status check pass.
- **Four Required Status Checks:**
  1. `ci/frontend-lint-typecheck`
  2. `ci/frontend-unit-tests`
  3. `ci/backend-tests`
  4. `governance/backlog-and-security-invariants`
- **Non-Required Checks:** The `Release Qualification Suite` (`release-qualification.yml`) is intentionally **non-required** on mainline PRs to prevent blocking routine atomic patches with release-stage deployment tests.

### 2.2 Ruleset `26024894` — Production Release (`release/production`)
- **Protected Branch Pattern:** `refs/heads/release/production`
- **Enforcement Level:** `ACTIVE`
- **Linear History Required:** True.
- **Force Pushes:** Blocked.
- **Deletions:** Blocked.
- **Required Gate:** Full pass of `.github/workflows/release-qualification.yml` and formal Product Owner signature.

---

## 3. GitHub App Least-Privilege Permission Matrix

In compliance with RQ-12, the GitHub App agent (`arx-terminal-agent[bot]`) operates under minimal necessary scopes:

| Scope | Permission Level | Operational Justification |
| :--- | :--- | :--- |
| `contents` | `write` | Creating and pushing engineering feature branches; committing patches. |
| `pull_requests` | `write` | Creating, updating, labeling, and merging compliant PRs. |
| `statuses` & `checks` | `write` | Reporting test and qualification results. |
| `workflows` | **`NONE / READ-ONLY`** | **CRITICAL SECURITY BOUNDARY:** GitHub App cannot modify `.github/workflows/*`. Any workflow update requires owner-mediated push. |
| `administration` | `none` | App cannot modify repository rulesets or bypass branch protections. |

---

## 4. Release Qualification Responsibilities (RQ-01 through RQ-12)

The Release Qualification framework formalizes the twelve pre-flight release gates:

- **RQ-01 (Clean Tree & Baseline Lineage):** Mainline commit is ancestor of release candidate.
- **RQ-02 (Frozen Manifest Integrity):** Hash verification of `FROZEN_ENGINE_MANIFEST_V2_5_1.json`.
- **RQ-03 (Security Scan & Secret Hygiene):** TruffleHog / GitGuardian zero secrets detected.
- **RQ-04 (Dual-Horizon Chart Geometry):** Verification of VWAP intraday and 20 EMA macro chart rendering.
- **RQ-05 (Cross-Engine & Account-Equity Provenance):** Strict verification of `PortfolioSummary` provenance status and governor sizing isolation (solved on mainline in PR #129).
- **RQ-06 (Radar Operational Truth):** Verification that scanner errors do not report 100% complete coverage (remediated in PR #110; PR #127 adds state hardening).
- **RQ-07 (Smart Money Containment):** Verification that `HOLD_SM1_001` remains active and background daemon is stopped.
- **RQ-08 (Corporate Events Blackout):** Verification that `HOLD_SEC_001` remains active and mock fixtures are isolated.
- **RQ-09 (Macro Regime Multipliers):** Verification that `MARKET_IN_CORRECTION` clamps risk exposure to $\le 0.5 \times$.
- **RQ-10 (Fail-Closed Broker Interlock):** Verification that `HOLD_EXEC_001` prevents live order egress.
- **RQ-11 (Platform Backup & Database Health):** Integrity of SQLite state snapshots and backup retention.
- **RQ-12 (Release Binding & Workflow Security):** Verification of workflow immutability and owner publication gating.

### Work Package 01 (WP-01) Status
- **Local Candidate Commit:** `df38cf52b41789c4644421da596e0f9d7f16ec7c`
- **Scope:** Formalization of `.github/workflows/release-qualification.yml` and hosted PR identity contract.
- **Authority Status:** `OWNER_PUBLICATION_PENDING`. Because GitHub App lacks `workflows: write`, this candidate cannot be published by an agent and awaits direct owner publication.

---

## 5. Hosting Platform Deployment Controls & Boundaries

### 5.1 Cloudflare Pages (Frontend)
- **Production URL:** `https://www.arxterminal.com/`
- **Production Project:** `finance`
- **Configuration:** `production_deployments_enabled = false`.
- **Deployment Trigger:** Deployments do **not** automatically trigger on Git push to `main` or `release/production`. Every production rollout requires explicit wrangler/dashboard promotion.
- **Active Production Deployment:** `f74a0db2`.

### 5.2 Railway (Backend Web Service)
- **Production URL:** `https://finance-production-XXXX.up.railway.app`
- **Active Production Commit:** `de5d2463cc4c6503bd700bcaf2b1bc36fd5e03d9`
- **Active Deployment ID:** `2fb9e82f-4b62-471e-abe2-63961a6c9c61`
- **Auto-Deploy Configuration:** `serviceInstanceAutoDeployStatus.enabled = false`.
- **Promotion Contract:** Merging a PR into `main` (such as PR #129) updates mainline Git state, but **does not deploy to Railway**. Controlled deployment requires explicit owner CLI trigger and preflight verification.
- **Rollback Readiness:** Last-known-good deployment `4d932efb-cf1f-4c89-9187-86e026a72e18` is preserved and available for instant one-click redeployment.
