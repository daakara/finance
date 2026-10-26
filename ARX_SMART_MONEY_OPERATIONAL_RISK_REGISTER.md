# ARX TERMINAL — SPRINT 3 / TRACK A
# SMART MONEY OPERATIONAL RISK REGISTER

**Document ID:** `ARX_SMART_MONEY_OPERATIONAL_RISK_REGISTER`  
**Execution Timestamp:** `2026-10-26T08:58:00Z`  
**Framework:** ARX Closure Model v2 + Governance Efficiency v1.1  
**Target Mainline Baseline:** `86037f4ba54fb6b66f94056cb07299482960a636`  
**Classification:** Operational Risk Assessment & Pre-Activation Mitigation Ledger

---

## 1. Risk Register Summary Metrics

| Severity Rating | Total Count | Pre-Activation Blocking | Non-Blocking (Operational Guard) |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | 2 | 2 (`RSK-01`, `RSK-02`) | 0 |
| **HIGH** | 4 | 1 (`RSK-03`) | 3 (`RSK-04`, `RSK-05`, `RSK-06`) |
| **MEDIUM** | 4 | 0 | 4 (`RSK-07`, `RSK-08`, `RSK-09`, `RSK-10`) |
| **LOW / NEGLIGIBLE** | 2 | 0 | 2 (`RSK-11`, `RSK-12`) |
| **TOTAL** | **12** | **3 BLOCKING** | **9 MITIGATED / MANAGED** |

---

## 2. Comprehensive Operational Risk Register

| Risk ID | Title & Vulnerability | Severity | Prob. | Impact | Mitigation & Defense-in-Depth | Responsible Owner | Pre-Activation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | **Unverified Off-Host Backup Recoverability**<br/>Pre-ingestion backup manifests succeed in CI, but an actual `pg_restore` rehearsal from Cloudflare R2 into an isolated PostgreSQL 16 instance has not been executed against a live dump. | **CRITICAL** | Med | Critical | Strict 64-hex SHA-256 byte validation prevents ingestion if manifest corrupt. Must conduct isolated staging restore rehearsal (`BackupRestoreVerifier.verify_backup`) before live activation. | Infrastructure Owner | **BLOCKING** (`SM_GATE_03`) |
| **RSK-02** | **Worker Provisioning & Specification Vacuum**<br/>No dedicated `smart-money-worker` service exists on Railway; resource limits, restart policies, and container entrypoints are not yet deployed. | **CRITICAL** | High | Critical | Read-only inspection verified web/postgres online. Formal Railway deployment template with isolated credentials and restart backoff required. | Platform Ops Owner | **BLOCKING** (`SM_GATE_01`) |
| **RSK-03** | **Missing External Incident Alert Routing**<br/>Worker process failures log to stdout and mark DB status as `FAILED`, but no automated PagerDuty / Sentry alerting route notifies on-call operators of missed cycles. | **HIGH** | Med | High | Health envelope endpoint (`get_health_envelope`) ready; must configure Sentry/webhook notification routing on worker exceptions. | Observability Lead | **BLOCKING** (`SM_GATE_06`) |
| **RSK-04** | **Operator Misconception of Env Var Preemption**<br/>Operator assumes mutating `ARX_SEC_EGRESS_ROLE` in Railway UI immediately halts an actively running container process on blocking network I/O. | **HIGH** | Med | Med | Truthful operational runbook established: Setting env var handles future cycles (Operation A). Stopping active container requires platform teardown (`railway down` / restart), immediately dropping TCP connection and session advisory lock (Operation B). | Operations Lead | **MITIGATED** (Documented in Runbook) |
| **RSK-05** | **SEC EDGAR IP Throttling or 429 Lockout**<br/>SEC enforces strict 10 req/s ceiling. Excessive bursts or User-Agent format changes risk automated 429 lockout or IP bans. | **HIGH** | Low | High | Token bucket `RateLimiter` enforces $\le 8.0\text{ req/s}$ with jitter; genuine contact email enforced in User-Agent preflight; exponential backoff on 429/503. | SEC Compliance Lead | **MITIGATED** (Enforced in Code) |
| **RSK-06** | **Concurrent Worker Execution / Lock Contention**<br/>Accidental duplicate scheduling or manual triggers launch overlapping workers simultaneously. | **HIGH** | Med | High | Exclusive session advisory lock `LOCK_SEC_EGRESS_OWNER = 401007311509093371` claimed at Step 2. Competing worker fails closed immediately with `CONCURRENT_INGESTION_BLOCKED`. | Database Guardian | **MITIGATED** (Enforced in Code) |
| **RSK-07** | **Zombie / Stalled Worker Lease Expiration**<br/>Worker stalls during SEC fetch; operator queue leases expire and another worker picks up the request. | **MEDIUM** | Low | Med | Optimistic row-level lease fencing: `publish_scanner_snapshot_atomic` updates `sm_run_requests` where `request_id = %s AND lease_token = %s`. Stale worker loses lease and rolls back snapshot publication. | Core Pipeline Eng | **MITIGATED** (Enforced in Code) |
| **RSK-08** | **Partial Ingestion State Inconsistency**<br/>Worker process terminated halfway through ingesting Form 4 filings across the 10 issuers. | **MEDIUM** | Med | Low | Form 4 raw filings and parsed transactions are deduplicated via `content_hash` and `ON CONFLICT DO NOTHING`. Rerunning cycle safely resumes without duplicate database records. Active snapshot untouched. | Database Guardian | **MITIGATED** (Enforced in Code) |
| **RSK-09** | **Daylight Saving Time (EDT/EST) Scheduler Drift**<br/>Railway UTC cron fails to align with New York 22:30 local market close across seasonal time shifts. | **MEDIUM** | Low | Med | Dual-trigger UTC cron (`30 2,3 * * 2-6`) paired with `_step3_check_durable_date_eligibility_under_lock` evaluates `America/New_York` clock and skips redundant trigger. Tested across 11 DST boundary test cases. | Core Pipeline Eng | **MITIGATED** (Enforced in Code) |
| **RSK-10** | **Unauthorized Production Force Override**<br/>Operator executes runner CLI with `--force` on production without authorization, risking duplicate cycle. | **MEDIUM** | Low | High | Fail-closed production override gate: `--force` strictly rejected unless matching `ARX_PRODUCTION_FORCE_RECOVERY_TOKEN`, valid reason ($\ge 10$ chars), and approved scope are provided under advisory lock. | Security Guardian | **MITIGATED** (Enforced in Code) |
| **RSK-11** | **Database Disk Exhaustion from Snapshot Growth**<br/>Accumulation of daily snapshots and raw XML filings consumes database volume capacity. | **LOW** | Low | Low | Governed cohort is bounded (10 issuers, ~20 filings/day, ~200KB/day). R2 retention prunes old backup dumps; table growth is monitored via Railway volume telemetry. | Platform Ops Owner | **MANAGED** |
| **RSK-12** | **Radar Snapshot Contamination**<br/>Incomplete or malformed scanner run updates active serving snapshot in Radar. | **LOW** | Low | Critical | Active pointer in `scanner_active_snapshots` updated only upon 100% successful evaluation of all 10 issuers in single atomic database transaction. Any exception preserves existing snapshot. | Radar Core Eng | **MITIGATED** (Enforced in Code) |

---

## 3. Pre-Activation Action Items

To resolve the 3 **BLOCKING** risks before production activation:
1. **Resolve RSK-01:** Execute an isolated restore rehearsal (`python scripts/rehearse_r2_restore.py`) using disposable PostgreSQL test containers and verify byte parity and table counts.
2. **Resolve RSK-02:** Prepare Railway `smart-money-worker` service specification (Dockerfile, 512MB RAM, 0.5 vCPU, restart policy `on-failure:3`).
3. **Resolve RSK-03:** Configure Sentry DSN or operational webhook on the worker environment to trigger alerts on unhandled exceptions.
