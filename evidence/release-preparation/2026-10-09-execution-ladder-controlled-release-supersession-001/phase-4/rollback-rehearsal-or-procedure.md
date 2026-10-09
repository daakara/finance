# Rollback Rehearsal & Procedure — P4-I10
If deployment rollback is triggered:
1. Roll back Railway service to deployment `0685134c-134c-4cce-8e0a-850299e18c34` (SHA `d97801e783620294454d1989164c907534ed4358`).
2. Roll back Cloudflare Pages project to deployment `ef378562-0fee-4529-a94f-e4d95a803cc5`.
3. Database compatibility: SQLite database requires ZERO DDL reversions and ZERO record deletions.
4. Historical and candidate rows remain safely persisted and queryable.
- Status: ROLLBACK_READINESS = PASS
