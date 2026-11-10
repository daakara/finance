# SM-ACT-020 restore verification hardening

Base: `ecdb81776ef99690b862384deab5928ea4faf905` (`release/production`).

The existing restore verifier can return `VERIFIED_RECOVERABLE` after a nonzero
`pg_restore` exit or a failed application read. Its default table requirements
cover four of twelve canonical tables, and trigger counts are observations,
not acceptance conditions. This candidate closes those false-positive paths.

## Acceptance changes

- Reject every nonzero pg_restore result; request `--exit-on-error`.
- Execute globals with `psql -X -v ON_ERROR_STOP=1`; reject failed globals restore.
  Existing-role conflicts must be reconciled in an isolated globals rehearsal;
  they are no longer silently accepted. The positive database-only rehearsal
  explicitly omits globals, matching the routine backup default.
- Check manifest backup identity before downloading the database archive.
- Require twelve canonical tables; caller-provided table names add requirements
  and cannot weaken the canonical contract. Quote identifiers for row counts.
- Require the four canonical public triggers on the expected tables, enabled for
  ordinary writes, with BEFORE ROW UPDATE/DELETE events and the expected function.
- Require the application-read snapshot to match the active pointer when snapshot
  verification is requested.
- Reject primary destination aliases differing only in credentials, URL form,
  default port or loopback spelling. Without the existing explicit override,
  require an explicit local socket/loopback destination and reject service-file,
  implicit-host, remote-host and multi-host destinations. Locality is containment,
  not proof of an empty disposable instance; operators must still provision one.

## Evidence scope

The receipt records all twelve row counts but explicitly declares
`source_data_parity_verified=false`, `roles_and_grants_verified=false`, and
`trigger_behavior_verified=false`. Table/trigger structure and application reads
are not source-to-restored row-digest parity, role recovery, or behavioral proof
of the trigger function. No such broader certification is claimed.

This candidate does not rerun the production backup/restore, update its historical
receipt, change service configuration, lift holds, send synthetic alerts, or
activate SEC ingestion. The previous R2 checksum verification remains evidence
about archive integrity, independent of this verifier correction.
