# ARX Terminal — Active Work Register and Pragmatic Delivery Plan
**As of:** 2026-11-07; **branch basis:** release/production at `83bca0b69d2d56d30b4d3157ff492fae260b7f3a`.
**Authority:** operational inventory and prioritization, not approval to activate held services. This document supersedes the *current status* portion of older dated snapshots; immutable historical receipts and frozen manifests remain untouched.

## Verified production baseline
- GitHub production: `83bca0b69d2d56d30b4d3157ff492fae260b7f3a`, merged PR #222 (ARX-LEARNING-004), 2026-11-07. Railway web and Smart Money worker: SUCCESS on same SHA (independently checked).
- Cloudflare Pages canonical production: `6f6770e0e72f5ca129feafdbee2c2e3932047191` (Radar PR #220), SUCCESS. Its independent manual deployment policy remains enabled (`production_deployments_enabled=false`).
- PR #220 Radar single universe: accepted, 250 evaluated and 100 Stage 2 on Nov 6 market snapshot, primary filters restricted to Stage 2; *not* automatic scanning or trading. No further feature work for the closed correction.
- PR #221 learning shadow mode: merged, passive capture and non-production evaluator introduced. PR #222 corrects hardcoded census fallbacks, settlement idempotency/finality, authenticated census access, and safe replay defaults. No verified clean settled prospective cohort or autonomous retraining approval.
- Production safety posture: `HOLD_SCAN_001`, `HOLD_SEC_001`, `HOLD_EXEC_001` and Sprint 5 lock remain required. Operational flag values must be checked when a future activation is requested; code/receipt claims alone are not evidence of newly authorized work.

## Consolidated, nonduplicative action queue
| Order | Owner/reference | State | Pragmatic next action and exit |
|---|---|---|---|
| 0 | PR #223 | DUPLICATE of merged #222 | Same nine changed paths and exact Git blob SHAs as #222. Close without merge after noting duplicate; preserve historical PR link. Do not rerun qualification. |
| 1 | Learning: PR #222 / ARX-LEARNING-004 | DEPLOYED; empirical operation NOT PROVEN | One **read-only** authenticated census and capture-storage check, including actual clean prospective counts, source identity, and no model updates. Do not restart replay or build a scheduler. If zero clean matured records, mark `AWAITING_EVIDENCE`, not failure. |
| 2 | Radar: Issue #166 | SINGLE-UNIVERSE CORRECTION CLOSED; natural scheduling HELD | Update tracker to replace outdated 35/43 assumptions. Preserve 250 dynamic published universe and Stage 2 filter contract. Natural scans remain independently gated; only a concrete freshness/use-case need justifies bounded qualification. |
| 3 | Smart Money: Issue #165 | BOUNDED NVDA SEC CANARY ACCEPTED; continuous ingestion HELD | Verify most recent evidence freshness and current worker egress state before proposing a limited ingestion window. Avoid conflating SEC Form 4 insiders with separate Congress STOCK Act data. No recurring ingestion or score changes. |
| 4 | Security: Issue #203 | OPEN P1 | Triage evidenced exploit paths and authentication/data exposure first. Implement only confirmed P0/P1 boundaries, with targeted adversarial tests. |
| 5 | Preflight/market truth: #197 + #198 | DEFERRED P1; overlapping | One falsification pass for stale-price/live-macro classification, provenance/OHLCV atomicity and execution-gate protection. Promote to P0 only on demonstrated unsafe authority. Reuse #181 for copy improvement. |
| 6 | Architecture: #204 | OPEN P1 | Own cross-domain and storage authority mapping; do not overlap #198 or undertake a wholesale refactor. Fix only verified reliability coupling. |
| 7 | UX: #181 | APPROVED DEFERRED | Bundle small Reward-vs-Risk wording/formatting with next already-justified frontend release; preserve 2:1 decision threshold and raw numeric policy. |
| 8 | PR #193 | OPEN, stale production anchor | Compare changed paths against production; close if already incorporated, otherwise rebase and run only affected UI/semantics tests. Do not deploy unchanged stale branch. |
| 9 | PR #146 | DRAFT, merge conflict | Preserve as evidence; extract only unshipped Sentry/privacy fixes if active production defect warrants. No mass rebase without a verified need. |
| 10 | PRs #141, #142, #103, #109 | HISTORICAL GOVERNANCE / MIXED BASES | Reconcile against currently installed RQ controls and backlog validator. Close superseded PRs, do not integrate historical baselines wholesale. |
| 11 | PRs #128, #73, #44, #42, #47 | HISTORICAL MARKET/SPRINT POCs | Archive/close as superseded by dynamic universe and accepted single-universe Radar. Do not reactivate scans via old PRs. |
| 12 | PRs #124, #38, #40 | HISTORICAL UX / FIXES | Compare with released UI only if a current bug exists; no standalone release for superseded drafts. |
| 13 | PRs #75, #78, #83 | RESEARCH / HVA DRAFTS | Preserve research and provenance evidence, with no model or scanner activation; request evidence-specific work only when independently needed. |
| 14 | Sprint-3 issues #60-70, #58-59 | OPEN HISTORICAL REGISTER | Reconcile individually against merged changes and original acceptance; do not bulk-close without evidence. Test/permission issues may be closed administratively once confirmed non-production. |

## Product roadmap decisions / sequencing
1. **Do not create new implementations for completed Radar #220 or learning correction #222.** Accept deployment separately from ongoing data accumulation and model calibration.
2. **Next true product value:** credible fresh Stage 2 opportunity research with explicit as-of, published universe build, reasons and no fabricated trading authority. Measure quality before increasing universe size.
3. **Empirical learning:** passively collect real decision-time observations; subsequently resolve outcomes using trustworthy subsequent price bars in controlled offline evaluation. Keep systematic and user-selected cohorts separate. Never promote historical pending records to prospective evidence.
4. **Smart Money:** preserve SEC Form 4 point-in-time evidence and amendments; Congress disclosures are a different source, with distinct reporting delays and no demonstrated alpha. No Nancy Pelosi trading correlation claim without timestamp-aligned, survivorship-aware controls.
5. **Security and critical freshness:** fix verified trust-boundary/freshness defects *ahead of* cosmetic UX/architecture work.
6. **No parallel audit proliferation:** one owner issue per defect, one bounded PR per genuinely missing change, targeted regression plus mandatory RQ, then a single production acceptance receipt.
7. **No general Sprint 5 activation, live trading, autonomous tuning, SEC recurring ingestion or natural scanner scheduling** through documentation edits or prior conditional approvals.

## Release and evidence rules
- A merged PR, passed CI, deployed service and live behavior are separate states. Publish exact SHA and artifact IDs; mark untested items `NOT_VERIFIED`.
- The frontend may legitimately lag backend after backend-only PRs; only require Cloudflare redeploy when frontend bundle actually changes.
- Auto-deploy remains disabled for Cloudflare; avoid unnecessary configuration changes.
- Owner preapproval allows already scoped, verified, minimal changes, **not** lifting governance holds.
- Retain frozen historical source documents and release receipts as dated snapshots, with this register as the up-to-date navigation point.

## Sources
[PR #220](https://github.com/daakara/finance/pull/220), [#221](https://github.com/daakara/finance/pull/221), [#222](https://github.com/daakara/finance/pull/222), [#223](https://github.com/daakara/finance/pull/223); [Radar #166](https://github.com/daakara/finance/issues/166), [Smart Money #165](https://github.com/daakara/finance/issues/165), [security #203](https://github.com/daakara/finance/issues/203), [architecture #204](https://github.com/daakara/finance/issues/204), [technical debt #198](https://github.com/daakara/finance/issues/198), [preflight #197](https://github.com/daakara/finance/issues/197), [UX #181](https://github.com/daakara/finance/issues/181).
