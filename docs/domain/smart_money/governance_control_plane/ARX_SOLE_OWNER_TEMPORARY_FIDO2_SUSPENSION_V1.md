# ARX TERMINAL — SMART MONEY GOVERNANCE CONTROL PLANE
## POLICY AMENDMENT: ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1

**Document Identity**: `urn:arx:contract:governance:policy_amendment:sole_owner_fido2_suspension:v1`  
**Prior Policy Version**: `urn:arx:contract:governance:root_identity_model:v1` (Root Quorum 3-of-4)  
**New Policy Version**: `ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1`  
**Operating Model**: `SOLE_OWNER`  
**Effective Timestamp**: `2026-10-14T01:30:00Z`  
**Review Checkpoint**: `2027-01-12T01:30:00Z` (90-day review interval)  
**Applicable Environments**: `production`, `smart-money-rehearsal`  
**Status**: DRAFT AMENDMENT (PENDING EXPLICIT OPERATOR RATIFICATION)  

---

### 1. Executive Rationale & Operating Context

1. **Sole Operator Context**: ARX Terminal is currently maintained and operated by a single human lead maintainer.
2. **Hardware Constraint**: FIDO2 hardware security keys are not yet enrolled or physically available.
3. **Circular Governance Blocker**: The earlier theoretical 3-of-4 multi-signature root quorum requirement cannot be satisfied without inventing fictitious identities or forging credentials.
4. **Resolution Strategy**: Rather than weakening technical controls or falsely asserting FIDO2 authentication occurred, this amendment formalizes the single-owner governance model while strictly requiring:
   - Explicit human owner release approval per release SHA.
   - Verified account-level Multi-Factor Authentication (MFA).
   - Exact commit SHA and environment binding.
   - Comprehensive technical, security, and recovery verification.
   - Immutable audit logging.

---

### 2. Contract Specification

```yaml
policy_id: ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1
operating_model: SOLE_OWNER
fido2:
  requirement_status: TEMPORARILY_SUSPENDED
  validation_performed: false
  quorum_satisfied: false
  reason: HARDWARE_DEVICE_NOT_YET_AVAILABLE
  reactivation_supported: true
  review_interval_days: 90
release_authorization:
  required_human_owners: 1
  explicit_owner_approval: REQUIRED
  existing_account_mfa: REQUIRED
  exact_release_sha_binding: REQUIRED
  technical_verification: REQUIRED
  security_verification: REQUIRED
  recovery_verification: REQUIRED
  rollback_plan: REQUIRED
  immutable_audit_record: REQUIRED
production:
  automatic_authorization: false
  deployment_requires_separate_approval: true
```

---

### 3. FIDO2 Reversible Reactivation Protocol

The transition back to hardware key enforcement is defined as a non-breaking, progressive 3-stage lifecycle:

$$\mathtt{TEMPORARILY\_SUSPENDED} \longrightarrow \mathtt{ENROLLMENT\_READY} \longrightarrow \mathtt{FIDO2\_REQUIRED}$$

1. **Stage 1: TEMPORARILY_SUSPENDED (Current)**  
   - Hardware key requirement suspended. Single-owner MFA + SHA binding active.
2. **Stage 2: ENROLLMENT_READY**  
   - Hardware key acquired. Key registered and public credential fingerprint recorded.
   - Self-test challenge executed and logged.
3. **Stage 3: FIDO2_REQUIRED**  
   - Formal policy amendment ratifying hardware enforcement.
   - Releases without valid hardware WebAuthn/FIDO2 assertion fail closed.

> [!IMPORTANT]
> The 90-day review date (`2027-01-12T01:30:00Z`) is an explicit governance review checkpoint to evaluate hardware availability. It does **not** cause an automatic release lockout.

---

### 4. Single-Owner Release Authorization Controls

Every production release under this policy requires an explicit `SoleOwnerReleaseApproval` artifact with:
- `release_sha`: Full 40-character commit hexadecimal string.
- `target_environment`: Must match `production`.
- `owner_identity`: Provenance-bound operator identity.
- `account_mfa_verified`: Boolean attestation of active account-level MFA.
- `valid_until`: RFC 3339 expiration timestamp (max validity: 24 hours).
- `technical_verification_passed`: Boolean confirmation of test suite execution.
- `security_verification_passed`: Boolean confirmation of security checklist execution.
- `recovery_verification_passed`: Boolean confirmation of restore verification.

Any mismatch or expired approval immediately rejects the deployment fail-closed.
