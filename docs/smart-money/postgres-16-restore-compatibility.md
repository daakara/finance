# SM-ACT-020 / Issue #265 — PostgreSQL 16 Backup & Restore Tooling

## Production compatibility contract

ARX Smart Money production PostgreSQL is pinned to **major 16** (`SMART_MONEY_POSTGRES_MAJOR=16`).
The scheduled backup uses `pg_dump -Fc`. A PostgreSQL 17 `pg_dump` can generate a custom
archive format 1.16, and a PostgreSQL 17 `pg_restore` can emit
`SET transaction_timeout = 0`, which is unsupported by PostgreSQL 16.

**Required for production promotion:**
- Install and pin official, **unmodified PostgreSQL 16** client binaries in the backup-service image.
- Confirm `pg_dump --version` reports PostgreSQL 16 and the runtime can resolve that binary on PATH.
- A normal PG16 custom archive must have format **1.15**. The backup engine rejects other archive formats and records generating client/version metadata.
- The production server must be PostgreSQL 16. Do not allow `pg_dump` from major 17 simply because it connects successfully to a 16 server.
- Do not use or ship patched `pg_restore` binaries, including the disposable local workaround used in the 2026-11-11 qualification.

## Verify before deploying

1. Inspect the actual deployed backup-service container toolchain and install/packaging configuration. Verify `pg_dump --version` reports PostgreSQL 16, and no conflicting PATH entry selects a different version.
2. Run `python -m pytest tests/test_sm_act_020_pg_client_compat.py -q` along with the existing Smart Money backup/scheduled entry-point tests.
3. Validate an existing **representative** PG16-produced custom archive with an official unmodified `pg_restore 16` in an isolated disposable PG16 database (never production). Validate return code zero, all 12 canonical tables, the 4 immutability triggers, row parity, active snapshot identity, and SHA-256 source match.
4. Document the exact client image/tag/digest and backup deployment acceptance in issue #265, then complete owner-controlled promotion to `release/production`.

**Important deployment interlock:** The code in this branch deliberately fails closed if the currently deployed backup service still resolves `pg_dump 17`. Promoting the code **before** pinning PostgreSQL 16 binaries would cause the next scheduled backup to fail. Do not promote this candidate until the runtime prerequisite is verified.

## Operational boundaries

This effort does not authorize any new SEC traffic, scanner/trading execution, continuous SEC ingestion, PG restore to production, manual backup, or changes to existing holds. The isolated restore accepted in [issue #165](https://github.com/daakara/finance/issues/165#issuecomment-6803169666) remains documented; hardening its future reproducibility is a separate task.

A successful R2 upload is not, by itself, a verified restore. Preserve `restore_verified=false`/`PENDING_VERIFICATION` until an actual isolated restore receipt is archived.
