# Sigstore / Rekor Live Provenance & Operational Rooting Closure V2

**Gate ID:** `SIGSTORE_REKOR_OPERATIONAL_ROOTING_CLOSURE_V2`  
**Governing Epoch:** `EPOCH_001`  
**Branch:** `arx/public-availability-observer-epoch-001`  
**Baseline Functional Commit:** `0a09e612cb7fbc00ea54e6012c85b3b10b05b078`  
**Baseline Evidence Commit:** `43314a19d5e706ba8e1337d72e514541db8a2977`  
**Operating Mode:** `FAIL_CLOSED` + `LIVE_PUBLIC_SIGSTORE_REQUIRED` + `EXACT_IDENTITY_PINNING` + `NO_MOCK_PROVIDER_EVIDENCE` + `NON_NATURAL_QUALIFICATION_ONLY` + `NO_PRODUCTION_AUTHORITY` + `NO_NATURAL_CAPTURE` + `NO_PUSH` + `NO_MERGE` + `NO_DEPLOY`  

---

## 1. Executive Summary & Gating Verdict

This gate evaluates the 11 operational assumptions required before the interim `SIGSTORE_REKOR_INTERIM_V1` profile may be considered suitable for prospective natural anchor rooting.

Following rigorous investigation across public Rekor infrastructure, local Cosign runtime tooling, local Sigstore TUF trust-root artifacts, and operational verifiers:

1. **Assumptions 001, 002, and 003 (Live Canonical R1/R2 Provenance):** Fail closed into **Outcome B (`OPEN`)**. The reference bundle submitted during Gate V1 witnesses the connectivity test payload `arx-rekor-test.json` (SHA-256: `ab3a2ccd...`), not canonical R1 (`afdd45ea...`) or canonical R2 (`2e3d2aa2...`). Substituting test bundles for canonical payloads is strictly prohibited.
2. **Assumption 004 (Unattended Scheduler Signing):** Fails closed into **Outcome C (`OPEN`)**. Interactive signing via Cosign device flow is qualified, but unattended workload identity (via GitHub Actions OIDC or SPIFFE) is not provisioned in the local desktop runtime environment (`NOT_ESTABLISHED`).
3. **Assumptions 005 through 011 (Operational Integrity & Durability):** Fully verified and **`CLOSED`**.
   - Signer identity lifecycle governance contract ratified (`SMART_MONEY_SIGSTORE_SIGNER_IDENTITY_POLICY_V1`).
   - Rekor consistency monitoring, signed tree head parsing, and split-view/regression detection active.
   - Sigstore TUF trust-root cache inspected, verified at root version 15, with fail-closed missing/corrupt handling.
   - Local raw evidence dual-store durability, SHA-256 replication, and disaster recovery simulation verified with 100% hash and witnessed-root match.
   - Same-root second bundle substitution defense enforced via authoritative ledger event binding.
   - Timestamp semantics disaggregated: RFC 3161 external timestamps, Rekor integrated time, local NTP capture intervals, and SEC public availability observer times are strictly separated.
   - Interim natural rooting policy ratified (`SMART_MONEY_SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_V1`).

**Final Suitability Verdict:**
- `SIGSTORE_INTERIM_PROFILE_SUITABLE_FOR_NATURAL_PROSPECTIVE_ROOTING = CONDITIONAL`
- `NATURAL_OBSERVER_ARMING = NOT_AUTHORIZED`
- `NATURAL_DENOMINATOR = 0`

---

## 2. Assumptions Evaluation Matrix

| Assumption | Description | Status | Outcome / Block |
|---|---|---|---|
| **ASSUMP_001** | Live Canonical R1 Provenance on Public Rekor | **OPEN** | Outcome B: Canonical R1 not submitted to Rekor |
| **ASSUMP_002** | Live Canonical R2 Provenance on Public Rekor | **OPEN** | Outcome B: Canonical R2 not submitted to Rekor |
| **ASSUMP_003** | Exact Live R2 → Live R1 Bundle Binding | **OPEN** | Outcome B: Blocked by ASSUMP_001 / ASSUMP_002 |
| **ASSUMP_004** | Unattended Scheduler Signing Path | **OPEN** | Outcome C: Workload OIDC not provisioned in local runtime |
| **ASSUMP_005** | Signer Identity Lifecycle Governance | **CLOSED** | Policy V1 ratified; rotation & compromise procedures established |
| **ASSUMP_006** | Rekor Consistency & Split-View Monitoring | **CLOSED** | Merkle inclusion, checkpoint verification & split-view alarms verified |
| **ASSUMP_007** | Sigstore TUF Trust-Root Cache Governance | **CLOSED** | Cache audited (v15, SHA256 verified), fail-closed handling enforced |
| **ASSUMP_008** | Local Raw-Evidence Durability & DR | **CLOSED** | Synchronous backup replication and disaster recovery simulated |
| **ASSUMP_009** | Same-Root Multiple-Bundle Substitution Rejection | **CLOSED** | Authoritative event hash binding rejects unauthoritative valid bundles |
| **ASSUMP_010** | Timestamp Semantics Disaggregation | **CLOSED** | RFC 3161, Rekor, local NTP, and SEC times strictly disaggregated |
| **ASSUMP_011** | Interim Natural Rooting Policy Ratification | **CLOSED** | Policy V1 ratified with conditional suitability and arming block |

---

## 3. Epistemic Custody & Operational Architecture

### 3.1 Public Witness Model vs Raw Storage
Sigstore/Rekor operates strictly as an append-only transparency witness (`PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS`). It witnesses SHA-256 payload digests and leaf certificate assertions, but does not provide raw blob retention. Local raw evidence durability is guaranteed via synchronous dual-path storage (`LocalRawEvidenceDurabilityManager`), ensuring full disaster recovery capability without trusting third-party storage.

### 3.2 Unattended Workload Identity Contract
In production CI pipelines, Cosign authenticates against GitHub Actions OIDC (`https://token.actions.githubusercontent.com`) to request short-lived ephemeral Fulcio certificates without human intervention. In local desktop environments, this ambient token is absent; hence, the system reports `NOT_ESTABLISHED` without fallback to insecure mock credentials.

### 3.3 Zero-Trust Reviewer Reconstruction
The independent `SigstoreOperationalReviewer` reconstructs all verification decisions from raw inputs:
- Checkpoints parsed and checked for root equality and monotonic tree growth.
- TUF trust roots verified directly against disk metadata.
- Disaster recovery simulated by deleting primary records and verifying restored SHA-256 against witnessed root digests.
- Same-root bundle substitution rejected even when alternative bundles carry valid Fulcio certificates.
- Zero-tolerance privacy filtering rejects credentials, API keys, and sensitive tokens.

---

## 4. Preservation of Pre-Arm Safety Invariants

- **Candidate 004 Firewall:** Zero edits to `shadow_evidence.db`, `analyst_dashboard/vcp/**`, or `natural_trigger.py`.
- **Natural Denominator:** Preserved strictly at `0`.
- **Git Invariants:** No remote push, no merge to main, no production deploy.
