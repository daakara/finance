# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: WITNESS AND FORK RECOVERY CONTRACT V1

**Document Identity**: `urn:arx:contract:governance:witness_fork_recovery:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Independent Witness Requirements

To prevent split-brain equivocations and unilateral ledger alterations, the Cryptographic Witness must be completely decoupled from the primary ledger administrator:

1. **Separate Credential Boundary**: The witness keys and authentication credentials must not share IAM roles, KMS keys, or accounts with the ledger store.
2. **Separate Failure Domain**: The witness infrastructure must operate in an independent region, network boundary, or third-party transparency provider (e.g., Sigstore Rekor).
3. **Cross-Rewrite Impossibility**:
   - The ledger administrator cannot rewrite or forge witness timestamps or receipts;
   - The witness administrator cannot mutate or truncate ledger events.

---

### 2. Witness Outage Policy (Fail Closed)

$$\mathtt{WITNESS\_OUTAGE\_POLICY} = \mathtt{FAIL\_CLOSED}$$
$$\mathtt{UnwitnessedAuthorizationIssuance} = \mathtt{FALSE}$$

- If the external cryptographic witness service becomes unreachable or unresponsive, the control plane MUST transition immediately to fail-closed read-only status.
- **Strict Prohibition**: Under no circumstance may an authorization be issued or consumed without a verified, contemporaneous witness receipt. Alternative "fail-open" or "degraded" issuance states are strictly forbidden.

---

### 3. Fork Recovery Protocol

If conflicting events branching from the same parent or competing heads are detected across mirrors:

$$\mathtt{OnForkDetection} \implies \mathtt{FORK\_INCIDENT\_TRIGGERED}$$

The system executes the following 8-step deterministic recovery procedure:
1. **Block Authorization Issuance**: Immediately halt the issuance of all new provisioning and execution tokens.
2. **Preserve All Branches**: Archive both competing ledger branches completely. Silent branch deletion or pruning is strictly forbidden ($\mathtt{SilentBranchDeletion} = \mathtt{FALSE}$).
3. **Identify Last Common Head**: Calculate the greatest common ancestor sequence $k$ and hash $H_k$.
4. **Open Governance Incident**: Record an immutable `GOVERNANCE_INCIDENT` event detailing the fork coordinates, timestamps, and conflicting actor signatures.
5. **Select Canonical Branch by Recovery Authority**: The Root Council quorum convenes, audits the conflicting lineages, and signs an arbitration decision declaring the canonical branch.
6. **Permanently Record Rejected Branch**: Append the discarded branch hash and forensic audit payload to the ledger as an invalidation record.
7. **Publish New Witnessed Head**: Anchor the reconciled head to the independent witness.
8. **Trigger Requalification**: Invalidate all outstanding authorizations issued during or after sequence $k$, and trigger full system requalification.
