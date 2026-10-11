# Smart Money Sigstore Signer Identity Policy V1

**Policy ID:** `SMART_MONEY_SIGSTORE_SIGNER_IDENTITY_POLICY_V1`  
**Version:** `1.0.0`  
**Status:** `RATIFIED`  
**Governing Epoch:** `EPOCH_001`  
**Mode:** `FAIL_CLOSED`  

---

## 1. Scope & Objective

This policy contract governs the identity lifecycle, cryptographic verification pinning, planned rotation, compromise protocols, and emergency suspension for all Sigstore / Rekor external witness operations anchoring the Smart Money decision ledger.

---

## 2. Pinned Signer Identities

### 2.1 Interactive Qualification Signer
- **Certificate SAN:** `53128824+daakara@users.noreply.github.com`
- **OIDC Issuer:** `https://github.com/login/oauth`
- **Matching Rule:** `EXACT_PINNED` (wildcard matching strictly prohibited).
- **Usage:** Qualification ceremonies, manual audit attestations, and controlled test harness executions.

### 2.2 Unattended Workload Signer (Prospective Natural Pipeline)
- **OIDC Issuer:** `https://token.actions.githubusercontent.com`
- **Authorized Repository:** `owner/finance`
- **Authorized Workflow Scope:** `repo:owner/finance:ref:refs/heads/main`
- **Matching Rule:** `EXACT_PINNED` (wildcard matching strictly prohibited).
- **Token Class:** `SHORT_LIVED_EPHEMERAL_WORKLOAD_TOKEN` (audited per run, ephemeral credentials).

---

## 3. Signer Identity Lifecycle Protocols

### 3.1 Planned Rotation Procedure
1. Pinned SAN and OIDC issuer specifications are updated via versioned policy change.
2. A dual-witnessed transition record is recorded in the ledger.
3. Historical public verification certificates and TUF roots are archived locally.
4. Previously witnessed bundles retain full cryptographic validity.

### 3.2 Compromise Procedure
1. Upon suspected key or OIDC account compromise, the identity status transitions immediately to `COMPROMISED`.
2. Anchor finalization trips the fail-closed alarm: `NATURAL_OBSERVER_ARMING = NOT_AUTHORIZED`.
3. All prospective admissions are suspended.
4. Historical bundles witnessed prior to the verified compromise timestamp remain valid if verified by Rekor integrated time; any bundle timestamped during or after the compromise window is invalidated.

### 3.3 Emergency Suspension Procedure
Any automated anomaly detector, auditor subagent, or human operator can trigger immediate emergency suspension, freezing all external witness admissions without prior notice.

---

## 4. Invariants

- **INV-SIGN-1:** Wildcard SAN or wildcard issuer matching is prohibited.
- **INV-SIGN-2:** Mismatch between leaf certificate claims and pinned identity fails closed.
- **INV-SIGN-3:** Compromised identity status unconditionally halts natural capture.
