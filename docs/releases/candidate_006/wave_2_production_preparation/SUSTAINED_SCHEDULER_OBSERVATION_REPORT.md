# SUSTAINED SCHEDULER OBSERVATION REPORT

**Release Lineage:** Candidate 006  
**Environment:** Railway Rehearsal (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)  
**Worker Deployment:** `89d26e9b-ac0d-4b59-9c6b-549a46522d62`  
**Date:** 2026-10-14  
**Longevity Verdict:** `NOT_YET_ESTABLISHED`  

---

## 1. Recorded Rehearsal Execution Passes

During rehearsal testing on 2026-10-14, two automated backup cycles executed successfully against Cloudflare R2:

1. **Cycle 1 (Initial Rehearsal Container Startup):**
   - Backup ID: `sm-backup-20261014T101110Z-723fe2e2`
   - Completion Time: 2026-10-14T10:11:15Z
   - Object: `rehearsal/sm-backup-20261014T101110Z-723fe2e2.tar.gz`
   - Trigger: Container boot pre-flight execution pass.
   - Status: Success, manifest verified.

2. **Cycle 2 (Post-Hotfix Container Startup):**
   - Backup ID: `sm-backup-20261014T101809Z-31d5e534`
   - Completion Time: 2026-10-14T10:18:14Z
   - Object: `rehearsal/sm-backup-20261014T101809Z-31d5e534.tar.gz`
   - Trigger: Redeployed container boot execution pass.
   - Status: Success, manifest verified.

## 2. Multi-Cycle Longevity Evaluation

The automated backup scheduler is configured with a default interval of **6 hours** (`21,600 seconds`).

- Rehearsal worker deployment `89d26e9b-ac0d-4b59-9c6b-549a46522d62` was deployed at approximately 10:18 UTC on 2026-10-14.
- As of 13:00 UTC, the continuous container runtime is approximately 2.7 hours.
- The next naturally scheduled periodic backup cycle is scheduled to execute at approximately **16:18 UTC**.
- Consequently, while startup pass automation and thread survival across sleep intervals are proven in unit and lifecycle tests, real-world multi-cycle periodic execution across 24-48 hours has **not yet elapsed**.

## 3. Production Readiness Criteria for Sustained Longevity

Before declaring sustained operational longevity established for production sign-off:
1. **Observation Window:** Continuous container uptime >= 24 hours without crashes or manual restarts.
2. **Cycle Count:** At least 4 consecutive automatic scheduled backup cycles recorded in Cloudflare R2 bucket (`arx-smart-money-backups`).
3. **Telemetry Parity:** Zero thread deaths (`is_alive == True`), zero unhandled exceptions (`consecutive_failures == 0`), and `age_since_last_success_sec <= 23,400s` (6.5 hours) throughout the run.
4. **Memory Profile:** Worker memory utilization remains flat without thread or connection leaks over the 24-hour period.
