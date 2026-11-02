# ARX Terminal — Current Project Status

**As of:** 2026-11-02  
**Scope:** Consolidated current-state addendum to the historical [29 October master baseline](ARX_MASTER_PROJECT_STATUS_2026_10_29.md). The 29 October document remains a point-in-time record; this file supersedes its *current* deployment and PR-status assertions, not its historical evidence.  
**GitHub main anchor at verification:** `cef6a2deb1e33f3f9ff453218d667f7d3d97872f`  
**Production branch anchor:** `7a513929bbcd74a3e568dcfe626bfac582ee68fb`

## Current dashboard

| Workstream | Verified status | Evidence and limitation |
| --- | --- | --- |
| Smart Money — PR #159 | MERGED to `main` on 2026-11-02 | Form 4 classification, provenance, bounded evidence and amendment handling |
| Radar — PR #160 | MERGED to `main` on 2026-11-02 | Selection-aware hero/navigation and catalog discovery |
| Release — PR #161 | MERGED to `release/production`; DEPLOYED | Cloudflare and Railway both successful on `7a513929` |
| PR #161 functional acceptance | PARTIAL | Live per-symbol evidence payload, browser navigation and boot-warmup containment remain open |
| Evidence governance — PR #162 | OPEN / 6 hosted checks SUCCESS / HOLD MERGE | Requires deletion authorization and observed-time replay corrections; no deployment |
| SEC Form 4 canary | BOUNDED CANARY REPORTED SUCCESSFUL | NVDA test does not authorize continuous polling; corroborating production activation has not been approved |
| SEC / scanner / execution | HOLDS RETAINED | `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`; runtime values require explicit verification at acceptance |
| Sprint 5 | NOT AUTHORIZED | Keep locked |
| Cloudflare production automation | DISABLED | Consider a separately approved CI/CD change after verifying branch protections |

## Current production anchors

- **Cloudflare Pages:** canonical deployment `afa7b72c-9fed-47fe-9baf-46ebc3e4eec3`, SUCCESS, commit `7a513929bbcd74a3e568dcfe626bfac582ee68fb`.
- **Railway web API:** deployment `bb891bb0-9c8b-4707-b18d-a23bf3c6cf39`, SUCCESS, same commit.
- **Observed API checks:** `GET /health` HTTP 200; `GET /api/v1/screener/run` HTTP 200; catalog-inclusive screener HTTP 200; PostgreSQL `smart_money_web` pool initialization.
- **Pending checks:** Smart Money evidence payloads by symbol; Radar browser selection/CTA; startup VCP `ERROR` snapshot, lease ownership and hold containment.

## Active decisions

1. **PR #162** — correct two confirmed governance weaknesses **on the existing PR**, retain the current policy/tests and hosted CI evidence, and stop short of merge/deploy.
2. **PR #161** — complete only outstanding live acceptance; do not repeat successful infrastructure qualification.
3. **SEC activation** — prepare a separately authorized bounded production canary / controlled polling decision; no hold lifting under this documentation effort.
4. **Deployment workflow** — GitHub approval gate can remain manual, followed by automatic deployments from a protected production branch only after a distinct configuration decision.

## Documentation authority

- [2026-11-02 release note](releases/2026-11-02_7a513929_smart-money-radar.md) — authoritative release-specific record.
- [Production release index](releases/README.md) — index of dated production release documents.
- [Root changelog](../CHANGELOG.md) — product change history.
- [29 October master status](ARX_MASTER_PROJECT_STATUS_2026_10_29.md) — historical baseline, not a current deployment claim.
- [PR #162](https://github.com/daakara/finance/pull/162) — authoritative **unmerged** governance candidate, not a production policy implementation.

**Status interpretation:** A successfully deployed commit is not proof of end-to-end functional acceptance. A green governance PR is not merged governance authority. Review claims against those boundaries before publishing a release-complete declaration.
