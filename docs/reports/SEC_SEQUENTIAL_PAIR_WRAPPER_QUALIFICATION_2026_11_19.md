# Sequential AAPL/NVDA wrapper qualification — PR #309

Directive: ARX-SEC-TWO-ISSUER-QUALIFICATION-001.
Base: 540075c85d34ec421c3da80e9cf7fdcf3e8d048f (PR #308).
Status: Review remediation candidate; exact-head hosted checks and PR results are authoritative.

## Source and historical evidence

The operator reported the 2026-11-19T22:17:32Z wrapper as 7,782 bytes with SHA-256
`2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3`.
Those original bytes are not independently attested by this remediation. This is
an operator-reported historical identity, not proof that the remediated code ran.
The historical two-request no-work receipt remains separate from current tests.
No historical before-state fingerprints are retroactively claimed.

The preceding v1.1.0 head and successful hosted run were actually
`5a760bd1048def4d8b2c0a8000162f6da61cd81e` and run `44632318474`.
The older report's `5a760bd138be620c54ea8f465a31a00908819077` full SHA was incorrect.
That successful run did not cover the review safeguards below. Current candidate
and manifest commits are recorded in PR #309 and its exact-head hosted workflow.

## Version 1.2.0 review safeguards

- CLI requires exactly one of --dry-run / --live. API also requires an explicit
  mode. Both require supplied timezone-aware issue/expiry timestamps; there is no
  automatic issue, renewal, or expiry generation. Invalid authority is rejected
  before database access or SEC egress.
- Aggregate timeout rejects booleans, non-finite, non-positive and >60 values.
  Exactly AAPL then NVDA retain 4 requests/30 seconds per issuer, 8 requests per
  pair, shared 5 RPS pacing and a >=1 second inter-issuer gap within the budget.
- A dedicated read-only, repeatable-read PostgreSQL inspection captures protected
  contents and queries the exact advisory lock. Missing protected tables or a
  failed preflight fingerprint block issuer execution.
- Postflight is attempted after success, exception and interruption. A failed
  inspection reports unknown lock state as NOT_CHECKED, never hardcoded false.
  Missing/mismatched fingerprints and held locks cannot return SUCCESS.
- Exceptions retain their original type and carry a pair_receipt; CLI emits it
  before a nonzero exit (130 for SIGINT). A prior issuer commit is reported and
  preserved, never described as pair-wide rollback.
- Measured aggregate duration includes postflight inspection. Forensic teardown
  may finish after the execution deadline; that overrun is explicitly FAILED,
  not a guarantee that every cleanup operation finishes within 60 seconds.
- DSNs are used as supplied; no hardcoded Railway proxy rewrite is performed.

The fixed plan remains AAPL's previously ingested pinned accession and NVDA's
historical Nov 17–18 interval, with Nov 19 cutoff. This is a preservation/no-work
rehearsal. Any protected-table content change fails acceptance, even if an issuer
committed successfully. A future unseen-filing trial requires a separate reviewed
accession-bound write-acceptance plan; do not repurpose this fixed rehearsal.

## Regression and release gates

Retain the 125 preceding tests with explicitly supplied test authority. Add
negative controls for mode/auth omission, conflicting flags, invalid aggregate
budgets, invalid/expired/naive timestamps, fingerprint failure before egress,
postflight failure, same-count content change, held-lock reporting, postflight
deadline overrun, SIGINT receipt/nonzero exit, and first-commit/second-failure
receipt preservation. Real PostgreSQL remains required for integration scenarios.
Local Python syntax validation is separate from hosted PostgreSQL execution.
Required release evidence: complete canary suite, 72-test SEC integrity suite,
backend regressions, frontend/typecheck, and RQ-01 through RQ-13 on exact head.
No passing claim is made for an unexecuted candidate.

## Recovery and operational scope

Prior restore evidence established structural/application reads. A separate test
exercises an immutability DELETE trigger in a disposable database; it does not
establish all restored trigger behavior, source data parity, or production role
and grant equivalence. These broader recovery dimensions remain NOT_CHECKED.

No live SEC requests or production database mutations are part of this
remediation. Recurring worker, scanner, trading and snapshot publication remain
held. The prior NO_CANDIDATE finding is timestamped operator evidence, not a
permanent assertion about future filings. Do not repeat live discovery merely to
qualify this wrapper. Radar's read-only readiness work is independent.
