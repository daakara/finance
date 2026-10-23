# ARX TERMINAL — RC5 CI RECOVERY VERIFICATION PLAN

**Date:** 2026-10-23T11:26:00Z (Local: 2026-10-23T13:26:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_READ_ONLY_PREFLIGHT_AUTHORIZED`  

---

## 1. Objective and Recovery Path

Following separate authorization and execution of the manifest succession procedure, continuous integration on PR #53 will transition from `FAILED` to `PASSED`. This document outlines the rigorous verification sequence ensuring that every required CI check completes successfully without bypassing any safety mechanisms.

---

## 2. Local Pre-Push Verification Suite

Before pushing the succession commit to `reconcile/sprint3-rc5-corporate-events`, the following five validation passes must be executed and confirmed in the local worktree:

| Suite Identifier | Exact Execution Command | Passing Criteria |
|---|---|---|
| **Tier 1 Core Invariants** | `python -m pytest -m tier1 -v` | 100% pass; 0 failures; manifest and engine compliance checks green. |
| **Corporate Events Unit & Edge** | `python -m pytest tests/test_corporate_events_*.py -v` | 100% pass; boundary and guidance edge-cases verified. |
| **Adversarial & Replay Gate** | `python -m pytest tests/test_measurement_adversarial_suite.py -v` | 100% pass; ledger contamination firewall and freeze compliance green. |
| **Static Linter & Security** | `ruff check . && flake8 && bandit -r analyst_dashboard api` | 0 errors, 0 warnings, 0 high/medium security findings. |
| **Frontend Static Export** | `cd frontend && npm run build` | Next.js SSG build succeeds; zero hydration or schema errors. |

---

## 3. Remote GitHub Actions CI Execution & Matrix Recovery

Once pushed, GitHub Actions will trigger workflow run on branch `reconcile/sprint3-rc5-corporate-events`:

```
                    ┌───────────────────────────────┐
                    │ Code Quality & Linter (Pass)  │
                    └───────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│ Backend Tests - Python 3.11   │           │ Backend Tests - Python 3.12   │
│ Tier 1 Gate: PASSED           │           │ Tier 1 Gate: PASSED           │
│ Full Pytest: PASSED           │           │ Full Pytest: PASSED           │
└───────────────────────────────┘           └───────────────────────────────┘
         └─────────────────────────┬─────────────────────────┘
                                   │ (All backend tests succeed)
                                   ▼
                    ┌───────────────────────────────┐
                    │ Static SSG Export Gate (Pass) │
                    └───────────────────────────────┘
```

### Expected Matrix Behavior:
- **Linter Job:** Completes in ~40s (PASSED).
- **Python 3.11 Job:** Completes in ~1m 45s (PASSED).
- **Python 3.12 Job:** Completes in ~1m 35s without matrix cancellation (PASSED).
- **Static SSG Export:** Successfully triggered after upstream completion (PASSED).

---

## 4. Post-CI Independent Readback

Using the verified GitHub App credentials (`arx-terminal-agent[bot]`), an independent readback will query the GitHub Check Runs API:

```python
# API Endpoint: GET /repos/daakara/finance/commits/{SUCCESSION_HEAD_SHA}/check-runs
```

### Acceptance Threshold:
1. `Code Quality & Security Linter` == `completed / success`
2. `Backend Test Suite - Python 3.11` == `completed / success`
3. `Backend Test Suite - Python 3.12` == `completed / success`
4. `Static SSG Export & Integrity Gate` == `completed / success`
5. PR #53 `mergeable_state` == `blocked` solely due to awaiting human review from repository owner (`daakara`).
