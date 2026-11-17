# SEC recurring ingestion remediation — 2026-11-17

Base: `933295cc07cd170dcb4a47e5b62a14384470a342` (`release/production`). Related tracker: #165.

## Findings and change

The worker injected `SecEdgarFetcher`, whose default `limit=1` bypassed the existing guarded multi-filing catalog path. Remove that injection and restrict the injected-fetcher path to explicit test mode. Retain existing actual-transmission limits (40/cycle, 4/issuer, 5 requests/second), authorization, host, backup and lock gates. Catalogs exceeding three unseen XML documents per issuer still fail closed; this change does not authorize an unbounded historical crawl or claim coverage beyond the recent catalog.

Raw storage previously committed before parsing and did not write the availability ledger. Step 6 now only fetches; Step 7 commits raw XML, SEC acceptance-time availability, parsed transactions, amendment links and reconciliation receipt in one shared transaction. Failed parsing or persistence rolls back all batch writes. Watermark reads use dictionary cursor keys and exclude missing availability or unresolved amendment evidence so incomplete evidence is reconsidered.

Extract `dateOfOriginalSubmission` and reporting-owner CIKs from XML. Resolve a base only when stored issuer, original filing date, reporting-owner CIK set, original Form 4 type, and availability ordering yield exactly one match. Do not infer owner identity from accession prefixes. An unresolved link can be repaired to resolved later; an existing resolved link is not overwritten. Unresolved amendments produce a partial reconciliation receipt and block snapshot publication. Persistence errors propagate rather than becoming warnings.

## Qualification evidence

Local targeted run: 92 passed, six PostgreSQL tests skipped because this execution environment denies local sockets. Python compilation and `git diff --check` pass. Matching production-branch blob identities were verified for the edited existing source and test files before editing the isolated checkout.

The existing Continuous Authorization Guard workflow now includes the new tests and requires the isolated PostgreSQL rehearsal (`ARX_REQUIRE_LOCAL_POSTGRES_TESTS=true`); socket denial there is a failure, not a skip. Rehearsal covers out-of-order amendment/base input, availability timestamps, idempotency, unresolved-link recovery, ambiguity, and atomic rollback on parse/ledger/amendment/receipt errors. Hosted results and release-manifest qualification remain pending until CI completes.

## Operational boundary

This candidate does not enable SEC ingestion, change Railway variables, dispatch a production request, query SEC, alter the NVDA frozen-baseline backfill, lift holds, or change scoring formulas. Production remains STANDBY. No claim of continuous operational readiness or current full-universe coverage is made. Subsequent activation still requires release qualification, deployment verification, current backup/restore evidence, lock/queue checks and a bounded authorized cycle. Snapshot evaluation of only the current cycle's transactions versus the historical PIT repository remains a separate activation-review concern.
