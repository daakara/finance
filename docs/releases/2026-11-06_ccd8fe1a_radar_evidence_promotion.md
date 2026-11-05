# ARX Terminal — Radar historical evidence production promotion

**Recorded:** 2026-11-06  
**Production PR:** [#199](https://github.com/daakara/finance/pull/199), squash-merged  
**Production release SHA:** `ccd8fe1a3ebddf0defcc5a378c0d25c89cec6e3d`  
**Upstream mainline implementation:** [PR #196](https://github.com/daakara/finance/pull/196), merge `5ebd758d1c6dfc9375140c731c1adf0791b29bc3`  
**Scope:** selectively promote historical VCP evidence integrity, temporal authority and explainability, preserving prior production scanner and Smart Money controls.

## Verified deployment

| Surface | Deployment ID | Verified status |
| --- | --- | --- |
| `release/production` | `ccd8fe1a3ebddf0defcc5a378c0d25c89cec6e3d` | GitHub merged PR #199 |
| Railway web | `54aa4381-d893-4a0e-9119-ce1ce3cfdbd9` | SUCCESS / ONLINE |
| Railway Smart Money worker | `327ef32d-d8a5-4c2e-b962-597d087e1355` | SUCCESS / ONLINE |
| Railway PostgreSQL | `110bafa2-3573-4e07-bf46-8ab5ef889993` | SUCCESS / ONLINE |
| Pending Railway work | None at verification | No manual redeploy necessary |

## Implementation intent (not a substitute for live response acceptance)

- Missing `known_at` or post-cutoff historical evidence must fail closed.
- Recalculation against current candle storage must not be represented as historically certified point-in-time evidence.
- Fast-path snapshots may report only predicate evidence genuinely persisted; missing individual vectors are unknown.
- Historical context requests without authoritative evaluation context must return `EVIDENCE_UNAVAILABLE`.
- Startup/boot-warmup must not publish authoritative natural VCP discoveries; existing PR #194 guards remain controlling.
- Neither a valid zero-match outcome nor an HTTP 200 proves recurring natural scanning is enabled.

## Final live production acceptance — Directive 011U

**Operator acceptance verdict:** `RADAR_VCP_HISTORICAL_EVIDENCE_PRODUCTION_ACCEPTED`.

Antigravity's 011U post-deploy receipt `ARX_RADAR_VCP_011U_PRODUCTION_ACCEPTANCE_REPORT.md` records direct production endpoint and browser checks against `https://web-production-470560.up.railway.app`, using production SHA `ccd8fe1a3ebddf0defcc5a378c0d25c89cec6e3d`:

| Test | Operator-reported live result |
| --- | --- |
| `GET /api/v1/screener/vcp/explain/LNTH` | HTTP 200; passed 9/10 normative predicates, `vcpQualified=false`; `PRED_VOLUME_DRY_UP` failed with observed 0.9884 versus max 0.70 |
| Provenance | `DETERMINISTIC_RECALCULATION`, `historicalPointInTimeCertified=false`, `bitemporalCutoffEnforced=true`; context `asOfDate=2026-11-05` |
| Unauthorized historical as-of `2024-01-01T00:00:00Z` | HTTP 422 `EVIDENCE_UNAVAILABLE`; no fabricated historical evaluation |
| Active snapshot | `vcp-snap-1793837026-768f94d4`; `AVAILABLE`, `matched_count=0`, no unexpected publication |
| `GET /api/v1/smart-money/sec-filings/NVDA` | HTTP 200, `AVAILABLE_WITH_RESULTS`, 7 filings, accession `0001696841-26-000012` |
| Frontend | Operator reports successful Cloudflare Pages production deployment and Radar page acceptance |
| Safety | Operator reports `HOLD_SCAN_001`, `HOLD_SEC_001`, `HOLD_EXEC_001` active, scheduler flag 0, SEC egress DISABLED, worker STANDBY |

**Evidence distinction:** GitHub PR merge and Railway deployment were independently reverified via connected services. The detailed live endpoint JSON, Cloudflare production readback, exact worker runtime HOLD values and frontend checks are sourced from the operator's 011U report; they were not independently rerun in this documentation session.

**Acceptance scope:** historical VCP evidence integrity, explainability and Smart Money regression isolation on the promoted production release. It does **not** authorize natural recurring scans, full-market publication, SEC ingestion, Sprint 5 execution or trading. Track broader activation in [Radar Issue #166](https://github.com/daakara/finance/issues/166).

**No authority changes:** This release is not authorization for new natural VCP scans, continuous scheduler activation, full-market universe publication, SEC ingestion, strategy tuning, Sprint 5 activation or trading execution.
