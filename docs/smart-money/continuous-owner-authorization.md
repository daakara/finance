# SM-ACT-020 — Continuous OWNER Grant (Candidate Only)

This change is **not an SEC activation**. All production operational holds, Railway
roles, scheduled backups and scanner/trading controls remain unchanged.

## Contract

Live `ARX_SEC_EGRESS_ROLE=OWNER` is insufficient by itself. The worker and
continuous pipeline must additionally verify these operator-controlled
environment entries **before the first OWNER lock, at each worker cycle, and at
pipeline stage/issuer boundaries**:

| Variable | Required |
|---|---|
| `ARX_CONTINUOUS_INGESTION_AUTHORIZED` | exactly `true` |
| `ARX_CONTINUOUS_AUTH_ID` | unique, lower-case 32-digit hex identifier |
| `ARX_CONTINUOUS_AUTH_SCOPE` | `DAILY_10_ISSUER_FORM4` |
| `ARX_CONTINUOUS_AUTH_UNIVERSE_ID` | `UB-US-EQUITIES-20261025-001` |
| `ARX_CONTINUOUS_AUTH_RELEASE_SHA` | exact approved 40-character production SHA |
| `ARX_RELEASE_SHA` or `RAILWAY_GIT_COMMIT_SHA` | must match approved SHA |
| `ARX_CONTINUOUS_AUTH_ISSUED_AT` | UTC ISO 8601 `YYYY-MM-DDTHH:MM:SSZ` |
| `ARX_CONTINUOUS_AUTH_EXPIRES_AT` | UTC ISO 8601 expiry, strictly after issued; no more than 24h after issued |
| `ARX_SEC_EGRESS_ROLE` | `OWNER`, subject to all existing holds |

The only accepted issuer order is **NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA,
BRK-B, JPM, V**. Any mismatch, expired/future grant, missing release identity,
or malformed timestamps fails closed with credential-free error codes.

This is a **config-gated expiration mechanism**, not a cryptographically signed
grant, a remote revocation channel, or permission to promote OWNER. A Railway
environment change does not alter an existing container's in-memory environment.
A running process still terminates an OWNER cycle on expiry because it rechecks
the UTC clock; to stop a blocked HTTP request immediately, use the documented
container lifecycle controls. Before deployment validate the running image's
attested release SHA; the Railway OAuth connector cannot read env values.

## Remaining rollout prerequisites

- First natural PostgreSQL 16 scheduled backup verified (separate automation).
- Worker Sentry DSN, SDK initialization, `subsystem=smart_money` event tags and
  actual event-delivery evidence. Existing email alert rule alone is not proof.
- Non-ingesting tests for restart, expiration, lease loss and emergency stop.
- A complete egress interception budget: the existing pipeline checks each issuer,
  but this grant alone does **not** guarantee per-HTTP-request expiry enforcement
  if an underlying fetcher performs multiple HTTP requests or retries.
- Daily cadence enforcement, maximum 40 requests, <=5 RPS and emergency-stop
  operational signoff. These were enforced by the bounded pilot runner; do not
  assume its safeguards automatically apply to the continuous pipeline.
- Candidate-specific release admission and official CI gates.
- **Separate explicit production activation authorization**. Never repurpose
  the prior two-cycle canary approval for continuous OWNER role.

Do not set any of the OWNER/grant variables in production while implementing
or testing this draft change.
