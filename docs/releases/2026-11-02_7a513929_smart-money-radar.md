# ARX Terminal — 2026-11-02 Smart Money and Radar Production Release

**Release branch:** `release/production`  
**Canonical production commit:** `7a513929bbcd74a3e568dcfe626bfac582ee68fb`  
**Release PR:** [#161](https://github.com/daakara/finance/pull/161) (merged 2026-11-02 09:53 UTC)  
**Included mainline PRs:** [#159](https://github.com/daakara/finance/pull/159) (Smart Money), [#160](https://github.com/daakara/finance/pull/160) (Radar)  
**Release status:** `DEPLOYED / INFRASTRUCTURE_VERIFIED / FUNCTIONAL_ACCEPTANCE_PARTIAL`

## Delivered

### Smart Money evidence
- Production integration of the canonical single-asset Smart Money evidence endpoint `/api/v1/smart-money/evidence/{symbol}` and the `/filings/{symbol}` alias.
- Form 4 transaction-classification controls, including open-market purchase versus non-conviction transaction handling, derivative exclusion and amendment deduplication.
- Fail-closed source availability and evidence classifications. Do not equate a held or incomplete source with a finding of zero qualifying insider transactions.
- PostgreSQL-backed Smart Money evidence integration, without authorizing continuous SEC polling.

### Radar
- Neutral initial hero action (`Explore Radar Assets`) and symbol-aware analysis action after an explicit selection.
- Expanded discovery across the seeded 35-symbol VCP reference universe and 43-symbol Master Catalog.
- Existing scanner capability status explicitly distinguishes an `ERROR` snapshot from successful screening.

## Production verification — 2026-11-02

| Component | Recorded production state |
| --- | --- |
| GitHub release branch | `7a513929bbcd74a3e568dcfe626bfac582ee68fb` |
| Cloudflare Pages canonical deployment | `afa7b72c-9fed-47fe-9baf-46ebc3e4eec3` — SUCCESS; same SHA |
| Railway web deployment | `bb891bb0-9c8b-4707-b18d-a23bf3c6cf39` — SUCCESS; same SHA |
| API startup / health | Startup successful; `GET /health` observed HTTP 200 |
| Screener | `/api/v1/screener/run` and `?include_catalog=true` observed HTTP 200 |
| Smart Money PostgreSQL | Pool initialized for `smart_money_web` |
| Cloudflare auto production deployment | Disabled at verification time; manual promotion was used |

**Verification limitation:** successful deployment, source-level route presence and observed screener responses are not equivalent to completed browser or endpoint-payload acceptance. In particular, live responses for Smart Money evidence on AAPL/NVDA/SPY/ONON/BTC-USD, Radar selection/navigation in production, and hold runtime values were not independently certified.

## Observed scanner containment question

Railway startup logs recorded a `BOOT_WARMUP` VCP invocation with `NON_EVIDENCE_BOOTSTRAP` context. A VCP snapshot was published with `status=ERROR`; a later startup call reused the same logical job; and a lease-heartbeat failure was reported. Source inspection confirms that startup calls `trigger_boot_warmup()`, which reaches `execute_market_wide_scan()`.

**Open acceptance condition:** verify whether this operation is authorized while `HOLD_SCAN_001` is active, that it cannot introduce unauthorized market-wide discovery or displace valid authoritative output, and that lease fencing prevents stale publication. This is not proof that the full production scanner was activated.

## Operational safety boundary

The release did **not** authorize continuous SEC egress (`HOLD_SEC_001`), full scanner scheduling (`HOLD_SCAN_001`), broker execution (`HOLD_EXEC_001`), trading model tuning or Sprint 5 activation. A separately reported bounded NVDA Form 4 canary is not authorization to lift the live-ingestion hold.

## Follow-on work (not part of this release)

- [PR #162](https://github.com/daakara/finance/pull/162) — SM-DATA-GOV-001 historical evidence and lifecycle governance; **OPEN, not merged or deployed**, despite green hosted CI. Review identified deletion authorization based on a token prefix rather than cryptographic proof, and SQLite `ARX_OBSERVED` replay that can accept SEC acceptance time as a substitute for ARX observation time. Both require targeted corrections.
- Complete the remaining PR #161 production browser/API acceptance and VCP boot-warmup containment check.
- Review release automation separately: protect `release/production` and automate Cloudflare/Railway deployment only after an approved promotion; do not silently enable automatic production deployment.

## Release decision

`PR161_INFRASTRUCTURE_ACCEPTED — FUNCTIONAL_ACCEPTANCE_PARTIAL`

This document reports verified infrastructure facts and explicitly preserves the remaining acceptance work. It must not be used as a declaration of unrestricted SEC or scanner operation.
