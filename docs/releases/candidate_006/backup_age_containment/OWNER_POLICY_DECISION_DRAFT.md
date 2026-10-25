# ARX Terminal — Smart Money Form 4
# Owner Policy Decision Draft: Backup-Age Containment Ratification

**Target Release**: Candidate 006 (Durability Enhanced)  
**Governance Requirement**: Explicit Sole Owner Approval Mandatory  
**Current State**: PENDING OWNER RATIFICATION  

---

## 1. Executive Summary

As directed by the project owner under Option 2, engineering implementation and adversarial requalification of **fail-closed backup-age containment** have been completed in an isolated non-production branch (`arx/smart-money-candidate-006-backup-age-containment`).

All 30 adversarial test vectors and an 11-step non-production fault injection exercise have verified that:
1. Ingestion of SEC Form 4 filings suspends fail-closed when backup age exceeds 12 hours (43,200s).
2. Suspended ingestion does not block or deadlock the backup scheduler.
3. Successful completion of a verified off-volume Cloudflare R2 backup automatically restores ingestion eligibility.
4. Resumption of ingestion produces zero duplicate records and maintains 100% data integrity.
5. In production, containment cannot be accidentally activated without an explicit owner authorization flag (`ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=true`).

---

## 2. Policy Decisions Requiring Owner Action

The owner must review and explicitly decide upon the following four items prior to any future rehearsal deployment or production cutover:

### Decision Item 1: Containment Threshold Ratification
- **Proposed Value**: 12 Hours (43,200 Seconds).
- **Rationale**: Form 4 filing velocity concentrates during market close hours. A 12-hour window guarantees an off-volume disaster recovery point is captured twice daily (every 6 hours recommended), bounding maximum data loss to 12 hours in a catastrophic disaster.
- **Alternative Options**:
  - *Option 1A*: 12 Hours (Recommended).
  - *Option 1B*: 6 Hours (More stringent RPO, higher R2 Class A/B request volume).
  - *Option 1C*: 24 Hours (More relaxed, lower operational sensitivity).
- **Owner Action**: `[ ] RATIFY 12 HOURS | [ ] SELECT ALTERNATIVE`

### Decision Item 2: Production Activation Authorization Gate
- **Mechanism**: Enforce `ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=true` in production environment variables.
- **Fail-Closed Protection**: In production, if `ARX_BACKUP_CONTAINMENT_ENABLED=true` but `ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED` is missing or false, worker preflight immediately crashes the container with a critical configuration error.
- **Owner Action**: `[ ] APPROVE GATE MECHANISM | [ ] REQUIRE ADDITIONAL SAFEGUARDS`

### Decision Item 3: Rehearsal Environment Deployment
- **Context**: The rehearsal environment (`smart-money-rehearsal`, `7f7e4317-2b76-4a0e-bff1-33227da71e5d`) remains on the baseline commit and has NOT been modified.
- **Next Step**: When authorized, the remediation commit can be deployed to the existing rehearsal web and worker containers to verify Cloudflare R2 integration in live cloud runtime without provisioning new services.
- **Owner Action**: `[ ] AUTHORIZE REHEARSAL DEPLOYMENT | [ ] DEFER REHEARSAL DEPLOYMENT`

### Decision Item 4: Cloud Infrastructure & Storage Budget Ratification
- **Context**: Bounded by zero-cost tiers (Railway trial/starter + Cloudflare R2 10GB free tier).
- **Projected 12-Month Storage Spend**: \$0.00 / month (well below 10GB free tier and 1,000,000 Class A operations).
- **Owner Action**: `[ ] RATIFY ZERO-COST SPENDING CEILING`

---

## 3. Strict Pre-Flight Guarantees Maintained

Under this engineering phase:
- `PRODUCTION_MUTATIONS = PROHIBITED`
- `PRODUCTION_DEPLOYMENT = NOT_AUTHORIZED`
- `PRODUCTION_DATABASE_PROVISIONING = NOT_AUTHORIZED`
- `PRODUCTION_MIGRATION = NOT_AUTHORIZED`
- `PRODUCTION_CUTOVER = NOT_AUTHORIZED`
- `MERGE_TO_MAIN = NOT_AUTHORIZED`
- `OWNER_AUTHORIZATION = PENDING`
