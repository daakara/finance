# Smart Money Natural Rooting Workload Identity Policy V1

## Purpose & Scope
This policy governs the unattended short-lived OIDC workload identity profile for Smart Money external Sigstore natural rooting (`SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1`). It provides the production-grade identity contract for scheduled natural witness attestations without interactive human intervention, PATs, static signing keys, or bearer token persistence.

## Policy Status
- **Policy ID:** `SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1`
- **Status:** `RATIFIED`
- **Governing Epoch:** `EPOCH_001`
- **Version:** `1.0.0`

## Exact Workload Identity Constraints
1. **Authorized Issuer:** `https://token.actions.githubusercontent.com` (Zero wildcard).
2. **Authorized Repository:** `daakara/finance` (Zero wildcard).
3. **Authorized Repository ID:** `1087399232`.
4. **Authorized Repository Owner:** `daakara`.
5. **Authorized Repository Owner ID:** `53128824`.
6. **Authorized Workflow:** `Smart Money Rooting Scheduler`.
7. **Authorized Job Workflow Ref:** `daakara/finance/.github/workflows/smart_money_rooting.yml@refs/heads/arx/sigstore-workload-identity-qualification`.
8. **Authorized Ref:** `refs/heads/arx/sigstore-workload-identity-qualification`.
9. **Authorized Audience:** `sigstore`.
10. **Authorized Subject Rule:** `repo:daakara/finance:ref:refs/heads/arx/sigstore-workload-identity-qualification`.
11. **Authorized Environment:** `NONE`.
12. **Authorized Event Trigger:** `push` (or governed `schedule` / `workflow_dispatch`).
13. **Maximum Token Lifetime:** 600 seconds (`SHORT_LIVED_EPHEMERAL_WORKLOAD_TOKEN`).

## Governance Invariants
- **Static Signing Private Keys:** Prohibited.
- **Long-Lived Tokens:** Prohibited.
- **Personal Access Tokens (PATs):** Prohibited.
- **Token Disk Persistence:** Prohibited (In-memory acquisition only).
- **Token Logging / Leaking:** Prohibited.
- **Emergency Suspension Behavior:** `TERMINAL_HOLD_AND_IMMEDIATE_ADMISSION_HALT`.
- **Candidate 004 Interference:** Prohibited (Zero touches to `shadow_evidence.db`, `analyst_dashboard/vcp/**`, `natural_trigger.py`).
