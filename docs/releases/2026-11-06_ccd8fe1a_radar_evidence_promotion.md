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

## Outstanding verification boundary

This documentation pass independently verified **GitHub merge and Railway deployment success**, but did **not** independently rerun production VCP temporal requests, snapshot provenance readback, test the UI or verify the full runtime HOLD configuration after the release.

Therefore this note records **`PRODUCTION_DEPLOYED — TARGETED_FUNCTIONAL_ACCEPTANCE_NOT_YET_VERIFIED_HERE`**, not blanket `RADAR_VCP_HISTORICAL_EVIDENCE_PRODUCTION_ACCEPTED`.

Reconcile the Radar operator's eventual post-deployment acceptance receipt against this exact SHA before changing the status. Track in [Issue #166](https://github.com/daakara/finance/issues/166).

**No authority changes:** This release is not authorization for new natural VCP scans, continuous scheduler activation, full-market universe publication, SEC ingestion, strategy tuning, Sprint 5 activation or trading execution.
