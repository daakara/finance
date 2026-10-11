# ARX Terminal — Smart Money: Sigstore Unattended Scheduler Signing Closure Gate V1

## Executive Summary
This gate audits the operational and cryptographic capability for executing Smart Money's Sigstore R1/R2 external witness rooting in an unattended scheduler environment using short-lived workload identity.

### Outcome Classification: Outcome B (No Ambient Workload Identity Available)
1. **Discovery Result:**
   - **Runtime:** `LOCAL_WINDOWS_WORKSTATION` (Windows 11 AMD64 / PowerShell).
   - **Workload Identity Provider:** `NONE_AVAILABLE`.
   - **Ambient OIDC Endpoint:** Neither `ACTIONS_ID_TOKEN_REQUEST_URL` nor `SPIFFE_ENDPOINT_SOCKET` is provisioned in the local environment.
   - **Prohibition of Fallbacks:** Per Section 1, interactive device-flow authentication, personal GitHub identities, static private keys, and PATs are strictly prohibited for unattended signing.
   - **Fail-Closed Determination:** Under Outcome B (Section 31), the evaluator accurately reports `UNATTENDED_SCHEDULER_SIGNING = NOT_ESTABLISHED`. Ambient credentials were not simulated or fabricated.

2. **Policy Ratification:**
   - Ratified `SMART_MONEY_SIGSTORE_WORKLOAD_IDENTITY_POLICY_V1` defining exact constraints for issuer (`https://token.actions.githubusercontent.com`), repository (`owner/finance`), workflow (`.github/workflows/smart_money_rooting.yml`), ref (`refs/heads/main`), audience (`sigstore`), and max token lifetime (600s).

3. **Verifier-Enforced Negative Controls & Invariants:**
   - Real verifier logic tested and rejected wrong issuer, wrong subject, wrong repository, wrong workflow, wrong ref, wrong audience, and expired tokens.
   - Fail-closed behavior confirmed across all 9 outage classes with zero unsigned or local-only fallback permitted.
   - Idempotency confirmed for R1, Authority Event, and R2 retry paths.
   - Zero secrets logged, written to disk, or committed.

---

## Canonical State
- **Gate:** `SIGSTORE_UNATTENDED_SCHEDULER_SIGNING_CLOSURE_V1`
- **UNATTENDED_SCHEDULER_SIGNING:** `NOT_ESTABLISHED`
- **LIVE_R1_R2_PROVENANCE_CHAIN:** `SEALED`
- **SIGSTORE_INTERIM_PROFILE_SUITABLE_FOR_NATURAL_PROSPECTIVE_ROOTING:** `CONDITIONAL`
- **NATURAL_OBSERVER_ARMING:** `NOT_AUTHORIZED`
- **NEXT_ACTION:** `PROVISION_GOVERNED_WORKLOAD_IDENTITY_ENVIRONMENT`
- **Candidate 004 Firewall:** Preserved (0 diff vs `63131b9`)
