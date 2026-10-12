# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE
### ARTIFACT: BOOTSTRAP AUTHORITY CONTRACT V1

**Document Identity**: `urn:arx:contract:governance:bootstrap_authority:v1`  
**Governance Epoch**: `epoch-gov-001`  
**Schema Identity**: `ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1`  
**Authority Type**: `ONE_TIME_EXCEPTIONAL_AUTHORITY`  
**Purpose**: `ESTABLISH_OPERATIONAL_GOVERNANCE_CONTROL_PLANE`  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE CONTRACT  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Authority Boundary

The Bootstrap Authority is an exceptional, strictly bounded, one-time root mechanism designed to break circular authority dependencies and bootstrap the operational Governance Control Plane.

$$\mathtt{BootstrapAuthority} = \mathtt{ONE\_TIME\_EXCEPTIONAL\_AUTHORITY}$$

It exists exclusively to transition the ARX Terminal Smart Money system from an un-provisioned design state to an active, independently witnessed, normal governance operational regime.

#### Core Lifecycle Invariants
1. **Single-Use**: The bootstrap authority can be activated and exercised exactly once.
2. **Non-Reusable**: Once consumed and extinguished, it cannot be reactivated or re-exercised under any circumstance.
3. **Non-Inheritable**: Successor governance epochs or secondary operators cannot inherit bootstrap privileges.
4. **Non-Renewable**: Its validity period is strictly time-bound and cannot be extended.
5. **No Post-Genesis Operation**: Bootstrap authority ceases operation immediately upon genesis completion.
6. **No Action Issuance After Extinguishment**: Any attempt to issue actions after extinguishment is rejected fail closed.

---

### 2. Strict Scope Allowlist

The Bootstrap Authority is authorized to perform *only* the following 11 operations:

1. `RATIFY_ROOT_IDENTITY_MODEL`: Formalize the proposed root council roles and quorum structure.
2. `REGISTER_INITIAL_ROOT_IDENTITIES`: Bind nominated identities to root council slots.
3. `REGISTER_INITIAL_ROOT_KEYS`: Register initial public keys and credential fingerprints for root council members.
4. `CREATE_GOVERNANCE_LEDGER`: Initialize the external append-only governance authority ledger.
5. `CREATE_GOVERNANCE_WITNESS`: Configure the independent cryptographic witness service.
6. `RATIFY_GENESIS_POLICY`: Ratify the initial constitutional policy hash (`GOVERNANCE_POLICY_V1`).
7. `PUBLISH_GENESIS_MANIFEST`: Anchor the initial genesis manifest binding epoch, sequence 0, and root hashes.
8. `PUBLISH_GENESIS_LEDGER_EVENT`: Append the initial sequence 0 event to the authority ledger.
9. `VERIFY_GENESIS_WITNESS`: Secure independent external witness proof for sequence 0.
10. `ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY`: Formally activate normal operational governance.
11. `EXTINGUISH_BOOTSTRAP_AUTHORITY`: Permanently extinguish the bootstrap authority token.

---

### 3. Absolute Denylist (Prohibited Actions)

The Bootstrap Authority is strictly prohibited from executing any of the following 16 actions. Any request invoking these actions MUST be rejected fail closed:

1. `PROVISION_THEOREM_A`: Cannot provision release trust infrastructure.
2. `PROVISION_THEOREM_B`: Cannot provision OIDC Gate-3 authorization infrastructure.
3. `PROVISION_THEOREM_C`: Cannot provision transactional state authority.
4. `PROVISION_THEOREM_D`: Cannot provision evidence custody storage.
5. `QUALIFY_THEOREM_E`: Cannot execute live recovery qualification.
6. `ARM_NATURAL_OBSERVER`: Cannot arm the natural observation scheduler.
7. `ALTER_NATURAL_DENOMINATOR`: Cannot increment or alter the natural denominator ($\mathtt{NATURAL\_DENOMINATOR} = 0$).
8. `MODIFY_CANDIDATE_GENERATION_004`: Cannot mutate Candidate 004 bytes ($\mathtt{CANDIDATE004\_DIFF} = 0$).
9. `SUPPRESS_FINDING`: Cannot suppress any active architecture or governance finding.
10. `DELETE_FINDING`: Cannot delete or purge finding registry records.
11. `WAIVE_NON_WAIVABLE_FINDING`: Cannot grant waivers on non-waivable defect classes.
12. `ALTER_MODEL_OUTPUT`: Cannot alter scanner predictions or evaluation outputs.
13. `TUNE_MODEL`: Cannot modify model parameters or trigger retraining.
14. `AUTHORIZE_PRODUCTION`: Cannot issue production execution authorizations.
15. `DELETE_LEDGER_HISTORY`: Cannot delete, truncate, or rewrite authority ledger events.
16. `REWRITE_GENESIS`: Cannot alter sequence 0 once appended.

---

### 4. Bootstrap Authority Invariants

$$\forall a \in \mathtt{RequestedActions}: (a \in \mathtt{Allowlist}) \land (a \notin \mathtt{Denylist}) \implies \mathtt{PERMIT}$$
$$\forall a \notin \mathtt{Allowlist}: \mathtt{DENY\_FAIL\_CLOSED}$$

- **Absent Action Denial**: Any action not explicitly enumerated in the allowlist is automatically denied.
- **Out of Scope Acceptance**: $\mathtt{BOOTSTRAP\_OUT\_OF\_SCOPE\_ACTION\_ACCEPTED} = \mathtt{FALSE}$.
- **Candidate 004 Immutability**: The bootstrap authority has zero write access to Candidate Generation 004 artifacts.
- **Zero Natural Denominator**: The bootstrap authority cannot mutate the natural denominator.
- **Production Pre-Arm Immunity**: The bootstrap authority cannot authorize runtime observation or arming.
