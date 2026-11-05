# ARX Terminal — Current project and production status

**Verified against GitHub and Railway:** 2026-11-06. This is a **living status entry point**, not a substitute for immutable historical evidence. Reverify branch heads and deployment identities before any release decision.  
**Repository:** [daakara/finance](https://github.com/daakara/finance)  
**Current mainline at verification:** `5ebd758d1c6dfc9375140c731c1adf0791b29bc3`  
**Current production release branch at verification:** `ccd8fe1a3ebddf0defcc5a378c0d25c89cec6e3d`

## Deployment and release authority

| Component or workstream | State as verified/reported | Authority and limits |
| --- | --- | --- |
| Railway production web API | **SUCCESS** on `ccd8fe1a`; deployment `54aa4381-d893-4a0e-9119-ce1ce3cfdbd9` | GitHub/Railway verified; production web service online |
| Railway Smart Money worker | **SUCCESS** on `ccd8fe1a`; deployment `327ef32d-d8a5-4c2e-b962-597d087e1355` | Railway verified; operator reported STANDBY before release. **Latest post-#199 runtime role not independently rechecked here** |
| Production PostgreSQL | **ONLINE** | Railway health verified; application privilege assertions are from operator's Recovery 002R dossier |
| Smart Money Recovery 002R | **PRODUCTION_ACCEPTED**, PR [#195](https://github.com/daakara/finance/pull/195) merged | Operator verified SELECT-only evidence grants, denied writes and NVDA accession readback; independently verified merge and Railway deployments. See [release note](releases/2026-11-05_74123474_smart_money_recovery_002r.md) and [Issue #165 acceptance](https://github.com/daakara/finance/issues/165#issuecomment-6671363712) |
| Radar boot-scan containment | PR [#194](https://github.com/daakara/finance/pull/194) included in production lineage | Startup natural VCP boot-scan disabled; publication authority preserved |
| Radar historical VCP explainability | PR [#196](https://github.com/daakara/finance/pull/196) **merged to main**; selective production PR [#199](https://github.com/daakara/finance/pull/199) **merged and deployed** | Production SHA `ccd8fe1a`; Railway deployments SUCCESS. **Targeted live Radar functional acceptance not independently rechecked here**; see [release note](releases/2026-11-06_ccd8fe1a_radar_evidence_promotion.md) |
| Strategy Alignment | PR [#193](https://github.com/daakara/finance/pull/193) **open**, target `release/production` | Separate release workstream, not covered by Radar #199 |
| Radar natural discovery | Bounded launch-cohort data and classification receipts exist; recurring scheduler **not authorized** | The 133-candidate/122-data-ready cohort and valid zero-match research do **not** demonstrate continuous scheduled production operation. Follow [Issue #166](https://github.com/daakara/finance/issues/166) |
| Full-market Radar | **QUARANTINED / NOT FULLY OPERATIONAL** | 5,871-candidate expansion requires separate source coverage, identity and publication gates |
| Cloudflare Pages | Frontend is a separate deployment surface | Latest Cloudflare production SHA was **not reverified** in this reconciliation; do not infer parity from Railway deployment success |

## Required operational HOLDs

- `HOLD_SEC_001`: active; no continuous SEC egress or unapproved ingestion.
- `HOLD_SCAN_001`: active; `VCP_NATURAL_SCHEDULER_ENABLED=0` per latest operator receipt; no recurring natural scans.
- `HOLD_EXEC_001`: active; trading execution prohibited.
- Sprint 5 execution lock: canonical backlog `sprint5_execution_authorized=false`; no implied authorization from a planning document.
- Model tuning and other owner-gated work remain governed by their independent authority contracts.

## Radar source and evidence boundaries

- Distinguish `AVAILABLE`, `STALE`, `PIPELINE_PENDING`, `ERROR` and genuinely computed zero qualified matches. An HTTP 200 is not itself success.
- Historical current-store recalculations are **not** certified point-in-time snapshots. Missing `known_at` or post-cutoff evidence must fail closed.
- Existing 35-symbol historical VCP fixture and 43-item curated catalog do not establish the actively published scan universe.
- Accepted 122/133 bounded readiness must not be confused with full-US-market eligibility or broad 95% coverage certification.
- Discovery/screening eligibility does not equal Pre-Flight or live trading authorization.

## Sprint 5: planning state, not execution

The [canonical backlog](planning/arx_canonical_backlog.json) was last reconciled 2026-10-31, remains labelled Sprint 4 and lists eight Sprint 5 carryforwards. These may overlap with subsequent Radar PRs. **Proposed**, not yet committed, [Sprint 5 rescope](planning/ARX_SPRINT5_RESCOPE_PROPOSAL_2026_11_06.md): (A) Radar classification source integrity; (B) bounded operational readiness; (C) cross-surface decision truth. Reconcile task-by-task using actual current production evidence before changing lifecycle records. Do not mark tasks closed without verified acceptance or silently unlock Sprint 5.

Related deferred work: [Pre-Flight #197](https://github.com/daakara/finance/issues/197), [technical debt #198](https://github.com/daakara/finance/issues/198), [risk/reward presentation #181](https://github.com/daakara/finance/issues/181).

## Documentation authority and maintenance

1. **Running behavior:** source code at the **deployed SHA** plus production observation/receipts; mainline alone is not proof of deployment.
2. **Current integrated state:** this living status page, updated with the exact branch/deployment identities.
3. **Backlog authority:** `planning/arx_canonical_backlog.json` and its schema, subject to explicit authorization records.
4. **Release history:** immutable dated notes in [releases/](releases/README.md); never rewrite historical claims as if they were made today.
5. **Detailed architecture:** `architecture/` and component source; architectural proposals are not runtime evidence.
6. **Historical status snapshots:** `ARX_MASTER_PROJECT_STATUS_2026_10_29.md` and `ARX_MASTER_PROJECT_STATUS_2026_11_02.md` are date-bound, not current.

**Documentation update checklist:** For any production-facing release or material contract change, update the current status, link exact PR and commit, add a dated release receipt, reconcile impacted canonical tasks without inventing approvals, and verify links/referenced paths. Never relabel an unmerged PR as deployed. 
