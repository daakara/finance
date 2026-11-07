# ARX sprint portfolio — repository-local planning overlay

**Created:** 2026-11-07 · **Baseline:** `086719f87803cb98acbb3666780e20c9f1d88b54` · **Status:** PLANNED / NON-ACTIVATING.

This file mirrors the six sprint-planning GitHub issues without replacing the historical canonical task/obligation register. The machine-readable index is [arx_sprint_portfolio_2026_11_07.json](arx_sprint_portfolio_2026_11_07.json). The entry point is [GitHub issue #228](https://github.com/daakara/finance/issues/228).

## Sprint board

| Sprint | Tracker | Backlog scope | Existing open issues | Existing open PRs | Open canonical tasks | Guard |
|---|---|---|---:|---:|---:|---|
| P0 | [#228](https://github.com/daakara/finance/issues/228) | Historical reconciliation | 6 | 7 | 1 | NON_ACTIVATING |
| P1 | [#229](https://github.com/daakara/finance/issues/229) | Decision integrity and market-data freshness | 3 | 0 | 0 | BOUNDED_ENGINEERING |
| P2 | [#230](https://github.com/daakara/finance/issues/230) | Radar discovery reliability | 4 | 3 | 8 | NO_SCANNER_ACTIVATION |
| P3 | [#231](https://github.com/daakara/finance/issues/231) | Smart Money evidence quality | 1 | 0 | 0 | NO_CONTINUOUS_SEC_EGRESS |
| P4 | [#232](https://github.com/daakara/finance/issues/232) | Security and architecture hardening | 2 | 3 | 0 | RISK_BASED |
| HOLD | [#233](https://github.com/daakara/finance/issues/233) | Owner-gated holding queue | 4 | 4 | 11 | OWNER_GATE_REQUIRED |

**Coverage:** 20 pre-existing open issues, 17 open PRs, and all 20 non-complete canonical tasks. The register also contains 20 completed and 3 superseded canonical tasks retained as historical evidence, and 71 obligations; none were reactivated.

## Per-sprint ownership and dependencies

### P0 — Historical reconciliation

Triage stale PRs and completed Sprint 3 issues without reimplementing accepted work.

- **GitHub issues:** [#58](https://github.com/daakara/finance/issues/58), [#59](https://github.com/daakara/finance/issues/59), [#63](https://github.com/daakara/finance/issues/63), [#67](https://github.com/daakara/finance/issues/67), [#68](https://github.com/daakara/finance/issues/68), [#69](https://github.com/daakara/finance/issues/69)
- **Open PRs:** [#193](https://github.com/daakara/finance/pull/193), [#124](https://github.com/daakara/finance/pull/124), [#109](https://github.com/daakara/finance/pull/109), [#103](https://github.com/daakara/finance/pull/103), [#47](https://github.com/daakara/finance/pull/47), [#40](https://github.com/daakara/finance/pull/40), [#38](https://github.com/daakara/finance/pull/38)
- **Open canonical task IDs:** `ARX-S3-016`
- **Entry:** Read-only evidence reconciliation
- **Exit:** Each item disposition recorded with evidence and no duplicate active implementation
- **Restriction:** NON_ACTIVATING

### P1 — Decision integrity and market-data freshness

Falsify stale price/live macro, expired UI cache, OHLCV provenance, canonical execution gating and misleading readiness UX.

- **GitHub issues:** [#197](https://github.com/daakara/finance/issues/197), [#198](https://github.com/daakara/finance/issues/198), [#181](https://github.com/daakara/finance/issues/181)
- **Open PRs:** None
- **Open canonical task IDs:** None; tracked as new GitHub issue scope
- **Entry:** PREFLIGHT-001 existing scope, only confirmed defects
- **Exit:** Tests, scoped PR if necessary, qualified release and updated blockers
- **Restriction:** BOUNDED_ENGINEERING

### P2 — Radar discovery reliability

Quantify published universe discovery quality, freshness and truthful filters without natural scanner activation.

- **GitHub issues:** [#166](https://github.com/daakara/finance/issues/166), [#61](https://github.com/daakara/finance/issues/61), [#62](https://github.com/daakara/finance/issues/62), [#70](https://github.com/daakara/finance/issues/70)
- **Open PRs:** [#128](https://github.com/daakara/finance/pull/128), [#44](https://github.com/daakara/finance/pull/44), [#42](https://github.com/daakara/finance/pull/42)
- **Open canonical task IDs:** `ARX-S4-RADAR-001`, `ARX-S2-005`, `ARX-S2-007`, `ARX-S2-008`, `ARX-S2-009`, `ARX-S3-004`, `ARX-S3-005`, `ARX-S3-013`
- **Entry:** Evidence-only qualification and bounded non-activating fixes
- **Exit:** Source-backed Stage 2/membership evidence, measured category availability
- **Restriction:** NO_SCANNER_ACTIVATION

### P3 — Smart Money evidence quality

Point-in-time NVDA/Form 4 evidence, cohort quality and settlement without recurring egress.

- **GitHub issues:** [#165](https://github.com/daakara/finance/issues/165)
- **Open PRs:** None
- **Open canonical task IDs:** None; tracked as new GitHub issue scope
- **Entry:** Read-only audit; independent canary authorization required
- **Exit:** Source freshness report and prospective cohort quality, not unsupported calibration
- **Restriction:** NO_CONTINUOUS_SEC_EGRESS

### P4 — Security and architecture hardening

Close only confirmed remaining trust-boundary, runtime ownership and build provenance risks after SEC-002.

- **GitHub issues:** [#203](https://github.com/daakara/finance/issues/203), [#204](https://github.com/daakara/finance/issues/204)
- **Open PRs:** [#146](https://github.com/daakara/finance/pull/146), [#142](https://github.com/daakara/finance/pull/142), [#141](https://github.com/daakara/finance/pull/141)
- **Open canonical task IDs:** None; tracked as new GitHub issue scope
- **Entry:** Evidence from #203/#204, no repeated SEC-002
- **Exit:** Distinct verified gaps closed with regression tests or explicitly deferred
- **Restriction:** RISK_BASED

### HOLD — Owner-gated holding queue

Track ten blocked + one authorization-held canonical tasks and dormant activation/research PRs.

- **GitHub issues:** [#60](https://github.com/daakara/finance/issues/60), [#64](https://github.com/daakara/finance/issues/64), [#65](https://github.com/daakara/finance/issues/65), [#66](https://github.com/daakara/finance/issues/66)
- **Open PRs:** [#83](https://github.com/daakara/finance/pull/83), [#78](https://github.com/daakara/finance/pull/78), [#75](https://github.com/daakara/finance/pull/75), [#73](https://github.com/daakara/finance/pull/73)
- **Open canonical task IDs:** `ARX-S1-003`, `ARX-S2-002`, `ARX-S2-006`, `ARX-S2-010`, `ARX-S3-003`, `ARX-S3-006`, `ARX-S3-007`, `ARX-S3-008`, `ARX-S3-009`, `ARX-S3-014`, `ARX-S3-015`
- **Entry:** Owner approval separate from planning is mandatory
- **Exit:** Retain HOLD unless separately authorized; no automatic carryforward into active sprint
- **Restriction:** OWNER_GATE_REQUIRED


## How to use locally

1. Update the local repository from its actual current branch. Use `git status` first and preserve uncommitted changes; fetch and fast-forward `release/production` only if it is safe.
2. Read `docs/planning/ARX_SPRINT_PORTFOLIO_2026_11_07.md` and its JSON index. Run `python tools/governance/verify_sprint_portfolio.py` to validate issue/PR/task coverage against the canonical backlog.
3. Do not change the canonical `active_sprint=SPRINT_4`, `sprint5_execution_authorized=false`, historical task statuses or the 71 obligations as part of planning.
4. When an actual workstream completes, record its PR/release acceptance and reconcile the canonical register in a separately qualified change. Do not silently update this dated snapshot and claim current deployment.
5. Where P0 reveals superseded old tasks or PRs, close them only with evidence. P1 is the next bounded engineering work. P2/P3/P4 are sequenced work queues, not release authorizations.

### Hard holds

`HOLD_SCAN_001=ACTIVE`, `HOLD_SEC_001=ACTIVE`, `HOLD_EXEC_001=ACTIVE`, `VCP_NATURAL_SCHEDULER_ENABLED=0`; Sprint 5 is locked, no autonomous trading, recurrent SEC egress or model mutation. Mapping tasks into sprints **never** lifts a HOLD.
