# ARX TERMINAL — DIRECTIVE 059
# Deliverable 4: Hosted CI/CD Enforcement & Pipeline Protection Plan

**Directive:** `ARX_SPRINT5_AUTHORIZATION_AND_CERTIFICATION_RECOVERY_059`  
**Target Repository:** `daakara/finance`  
**CI Engine:** GitHub Actions (`.github/workflows/ci.yml`)  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-d058-obligation-recovery`  
**Baseline:** `f1c7def39546076e98c506732b67e1ecdbfccfcf`  
**Timestamp:** `2026-10-27T21:53:00Z`  
**Status:** `ACTIVE SPECIFICATION`

---

## 1. Executive Summary

This plan outlines the hosted CI/CD pipeline integration and enforcement strategy for the ARX Terminal Backlog Integrity Validator (`validate_backlog_integrity.py`). 

The objective is to establish an automated, non-bypassable quality gate in GitHub Actions that prevents:
1. Untracked tasks, orphan PRs, or broken acceptance criteria from merging into mainline.
2. Premature activation of Sprint 5 features or relaxation of operational holds.
3. False assertions of production deployment or self-authored authorization records.

---

## 2. Multi-Stage Progressive Enforcement Architecture

The validator operates in three distinct progressive modes designed to balance developer velocity with cryptographic rigor:

```
+-----------------------------------------------------------------------------------------------+
|                             PROGRESSIVE CI ENFORCEMENT TIERS                                  |
+-------------------+----------------------------+-----------------------+----------------------+
| Stage             | CLI Invocation             | Execution Context     | Blocking Behavior    |
+-------------------+----------------------------+-----------------------+----------------------+
| 1. Scoped PR Gate | `--stage scoped`           | All Feature & Bug PRs | Fails on errors if   |
|                   | `--base-ref origin/main`   | (PR build job)        | governance files mod |
+-------------------+----------------------------+-----------------------+----------------------+
| 2. Full Regression| `pytest tests/`            | All PRs & Mainline    | Fails on any test    |
|                   | `test_backlog_...py -v`    | (Test suite job)      | regression (86 tests)|
+-------------------+----------------------------+-----------------------+----------------------+
| 3. Pre-Sprint 5   | `--pre-sprint5-closure`    | Sprint Transition PRs | Fails closed on any  |
|    Certification  |                            | & Pre-Release Tagging | missing PO authority |
+-------------------+----------------------------+-----------------------+----------------------+
```

---

## 3. GitHub Actions Pipeline Integration

### 3.1 Existing Pipeline Status (`.github/workflows/ci.yml`)
The current GitHub Actions CI workflow already incorporates GOV-001 at lines 104–115:

```yaml
    - name: Run Backlog Integrity Validator Suite (GOV-001)
      env:
        PYTHONPATH: "."
      run: |
        pytest tests/test_backlog_integrity_validator.py -v

    - name: Enforce Backlog Integrity & Closure Gate (GOV-001)
      env:
        PYTHONPATH: "."
      run: |
        python tools/governance/validate_backlog_integrity.py --stage scoped --base-ref origin/main
```

### 3.2 Evaluation of Current Implementation:
1. **Zero False Positives:** When non-governance code is edited, `detect_scoped_governance_changes()` returns `False`, and `--stage scoped` exits 0 without blocking unrelated engineering PRs.
2. **Immediate Enforcement:** When `docs/planning/arx_canonical_backlog.json` or related governance files are touched, the validator inspects the diff against `origin/main` and strictly enforces rules `BI-000` through `BI-012`.
3. **Adversarial Resilience:** The 86 automated tests in `test_backlog_integrity_validator.py` run on every commit and PR, guaranteeing that no change to the validator itself can introduce regressions.

---

## 4. Pre-Sprint 5 Kickoff Enforcement Protocol

When a pull request is submitted to transition the platform into Sprint 5, the CI pipeline must enforce `--pre-sprint5-closure`.

### Recommended Workflow Addition:
```yaml
    - name: Enforce Pre-Sprint 5 Closure Gate
      if: contains(github.head_ref, 'sprint5') || contains(github.head_ref, 'sprint-5') || github.event.pull_request.title == 'Sprint 5 Kickoff'
      env:
        PYTHONPATH: "."
      run: |
        python tools/governance/validate_backlog_integrity.py --pre-sprint5-closure
```

### What This Guarantees:
1. **Human Product Owner Gate:** If `sprint5_execution_authorized: false`, or if `authorizer != "PRODUCT_OWNER"`, CI will exit with code 1 and block merge.
2. **Production Deployment Truth:** If `DEF-056-001`, `002`, or `003` are not verified active on Railway and Cloudflare, CI will exit with code 1 citing `P5-003: held from production deployment`.
3. **Hold Integrity:** If any of the 11 immutable operational holds are relaxed, CI will exit with code 1 citing `P5-005`.

---

## 5. Security & Anti-Bypass Architecture

To prevent autonomous agents or unauthorized contributors from bypassing this gate:
1. **GitHub Protected Branches:** `main` requires passing status checks for:
   - `Run Backlog Integrity Validator Suite (GOV-001)`
   - `Enforce Backlog Integrity & Closure Gate (GOV-001)`
2. **Schema Hardening:** JSON Schema enforces typed structure on `sprint5_execution_authorization_record`.
3. **Non-Delegable Authorizer Check:** The string `"PRODUCT_OWNER"` is required. Strings such as `"AGENT"`, `"AI"`, `"AUTOMATED"` cause instant failure.
4. **Authoritative Git Baseline:** The validator queries `git show origin/main:docs/planning/arx_canonical_backlog.json` directly from Git object storage, preventing local tampering with the historical baseline.

---

## 6. Implementation Schedule

- **Phase 1 (Directive 058/059):** Validator hardened locally, 86 tests passing, worktree rebased on mainline `f1c7def`. (COMPLETED)
- **Phase 2 (Owner Decision):** Product Owner approves Directive 059 governance PR and publishes branch.
- **Phase 3 (Post-Merge):** CI automatically enforces progressive validation on all subsequent PRs.
- **Phase 4 (Sprint 5 Kickoff):** When production deployment Gate C is approved by Owner, `--pre-sprint5-closure` passes and unlocks Sprint 5 feature execution.
