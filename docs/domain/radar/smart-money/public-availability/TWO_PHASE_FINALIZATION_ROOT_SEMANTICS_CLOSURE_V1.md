# ARX Terminal — Smart Money: Two-Phase Finalization Root Semantics Closure Gate

**Gate Identifier:** `TWO_PHASE_FINALIZATION_ROOT_SEMANTICS_CLOSURE_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `7840eb6663541f44b3d440e992ad133780db334c`  
**Functional Commit:** `2592a0d245764ef3afe5991c657e1155a85beeee`  
**Authority:** Independent Finalization Root Semantics Auditor + Authority Transition Contract Auditor + Root Self-Reference Auditor + Post-Authority Closure Auditor + Candidate 004 Firewall Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `DETERMINISTIC` + `APPEND_ONLY` + `NON_CIRCULAR` + `RECEIPT_DRIVEN` + `NO_RETROACTIVE_AUTHORITY` + `NO_PRODUCTION`

---

## 1. Executive Summary & Problem Solved

In previous iterations, the finalization root contract exhibited a theoretical circularity dilemma:
> **The Circularity Dilemma:** If the root payload claimed `AUTHORITY_EFFECTIVE_STATUS = EFFECTIVE` and committed to its own timestamp before submission, it anticipated an authority grant prior to external confirmation. Conversely, if authority effectiveness depended on the external root receipt, the receipt could not be committed inside the root without causing a circular hash dependency ($\text{Hash}(P) \supset R \supset \text{Hash}(P)$).

This gate formally decomposes root finalization into a strict **Two-Phase Non-Circular Lifecycle**:

```
[Phase R1: Pre-Authority Root]
  └── Status: BLOCKED_PENDING_FINALIZATION_ROOT
  └── Effective At: null
  └── Submits to WORM Provider
        │
        ▼
[External Provider Receipt: R1]
  └── Verified by Strict 8-Point Check
        │
        ▼
[Authority Transition Event]
  └── Trigger: R1 Receipt Confirmation
  └── Emits: AUTHORITY_BECAME_EFFECTIVE (Immutable, Monotonic)
  └── Status: EFFECTIVE
        │
        ▼
[Phase R2: Post-Authority Closure Root]
  └── Binds R1 ID/Hash + R1 Receipt Hash + Transition Event ID/Hash/Seq
  └── Submits to WORM Provider (Within 300.0s Deadline)
  └── Audit Finality Root (Does NOT re-grant or re-decide eligibility)
```

---

## 2. Phase R1: Pre-Authority Admission Root Semantics

### Invariants (Sections 4, 6, 7)
- **Phase Name:** `PRE_AUTHORITY_ADMISSION_ROOT`
- **Payload Contents:**
  - Contains finalization events up to run completion.
  - `authority_eligibility`: `PASS` / `FAIL` / `NOT_EVALUATED`.
  - `authority_effective_status`: Strictly `BLOCKED_PENDING_FINALIZATION_ROOT`.
  - `authority_became_effective_at`: Strictly `None` (`null`).
  - `authority_transition_intent`: `"GRANT_IF_ROOT_CONFIRMED"`.
  - `authority_effectiveness_precondition`: `"R1_ROOT_RECEIPT_CONFIRMED"`.
- **Elimination of Self-Reference:**
  - Prohibits embedding future R1 receipt hash into R1 payload (`CIRCULAR_ROOT_PAYLOAD_PROHIBITED`).
  - Enforced by `build_r1_root_payload`: raises `ValueError` if any receipt is passed.

---

## 3. Receipt-Triggered Authority Transition Contract

### Invariants (Sections 8, 9, 10, 11)
- **Transition Trigger:** Strictly receipt verification of R1 (`R1ConfirmationStatus.CONFIRMED`).
- **Trigger Verification:**
  - 8-point check: receipt presence, HMAC signature, root hash match, parent run ID match, parent run final hash match, policy set match, ISO 8601 timestamp validity, and semantic ordering ($T_{\text{r1\_confirmed}} \ge T_{\text{r1\_created}}$).
- **Atomic Transition Event Emission:**
  - Emits immutable `AuthorityTransitionRecord` into `authority_transitions` table.
  - Event type: `AUTHORITY_BECAME_EFFECTIVE`.
  - Event sequence: Next monotonic integer in ledger sequence.
  - Event hash: SHA-256 over canonicalized event dictionary.
- **Clock Boundary Monotonicity:**
  $$T_{\text{observed}} \le T_{\text{eligible}} \le T_{R1\text{-confirmed}} \le T_{\text{effective}}$$
- **Idempotency & Replay:**
  - Duplicate identical trigger is an idempotent no-op (`DUPLICATE_IDENTICAL_AUTHORITY_TRIGGER = IDEMPOTENT_NO_OP`).
  - Duplicate trigger with conflicting parameters raises `CriticalIntegrityError` (`CRITICAL_INTEGRITY_CONFLICT`).

---

## 4. Phase R2: Post-Authority Closure Root Semantics

### Invariants (Sections 12, 13)
- **Phase Name:** `POST_AUTHORITY_CLOSURE_ROOT`
- **Commitments & Linkage:**
  - Commits directly to R1 root ID (`r1_root_id`), R1 root hash (`r1_root_hash`), and R1 receipt hash (`r1_receipt_hash`).
  - Commits directly to authority transition event ID (`authority_transition_event_id`), event hash (`authority_transition_event_hash`), and sequence (`authority_transition_sequence`).
  - Commits to `authority_effective_status = EFFECTIVE` and `authority_became_effective_at`.
- **Audit Finality Function:**
  - R2 provides external audit closure over the authority transition.
  - R2 does NOT re-decide authority eligibility (`R2_CAN_CHANGE_AUTHORITY_ELIGIBILITY = False`).
  - R2 does NOT retroactively grant authority or change the effective timestamp (`R2_CAN_BACKDATE_AUTHORITY = False`).
  - R2 cannot contain its own future receipt (`CIRCULAR_ROOT_PAYLOAD_PROHIBITED`).

---

## 5. Closure Deadline & Authority Suspension Contract

### Invariants (Sections 14, 15, 16)
- **Maximum Closure Delay:** `POST_AUTHORITY_CLOSURE_MAX_DELAY = 300.0s`.
- **Deadline Monitoring:**
  $$\text{closure\_deadline\_at} = T_{\text{effective}} + 300.0\,\text{seconds}$$
- **Expiry Behavior:**
  - If R2 is not confirmed before the deadline:
    - Status transitions to `PostAuthorityClosureStatus.EXPIRED`.
    - Authority transitions to `AuthorityEffectiveStatus.SUSPENDED_PENDING_CLOSURE`.
    - **No Retroactive Revocation:** Decisions made while `EFFECTIVE` (between $T_{\text{effective}}$ and deadline) remain valid (`R2_EXPIRY_RETROACTIVELY_REVOKES_PRIOR_VALID_DECISIONS = False`).
    - **Suspension of New Decisions:** Downstream firewall immediately blocks any new production decisions, empirical denominators, model tuning, or learning claims.
- **Conflict Behavior:**
  - Conflicting R2 receipt transitions closure to `CONFLICT` and authority to `SUSPENDED_INTEGRITY_HOLD`.

---

## 6. Threat Model Tamper Matrix (Sections 19–22)

| Threat Scenario | Injection Technique | Detection Mechanism | Status |
|---|---|---|---|
| **Authority Transition Rewrite** | Tampering with transition event hash or sequence | Forensic hash recomputation fails | `authority_transition_rewrite_detected = True` |
| **R1 Receipt Substitution** | Swapping valid R1 receipt for different run/root | R2 linkage parity check fails | `r1_receipt_substitution_detected = True` |
| **Duplicate R1 Notification** | Resending identical confirmed R1 receipt | Idempotency key lookup deduplicates | `IDEMPOTENT_NO_OP` |
| **R1 Conflict** | Corrupted receipt signature / mismatched root | Verification returns `CONFLICT` | `BLOCKED_INTEGRITY_HOLD` |
| **R2 Conflict** | Corrupted R2 signature / mismatched linkage | Verification returns `CONFLICT` | `SUSPENDED_INTEGRITY_HOLD` |
| **Timing Backdating** | Setting $T_{\text{effective}} < T_{R1\text{-confirmed}}$ | Strict monotonic clock check | `authority_time_tamper_detected = True` |

---

## 7. Crash Recovery Matrix (Section 23)

| Interruption Point | Process State at Recovery | Recovery Action | Restored Status |
|---|---|---|---|
| **Crash after R1 recording, before receipt** | R1 payload stored, no transition record | Query external provider; if confirmed, trigger transition | `BLOCKED_PENDING_FINALIZATION_ROOT` |
| **Crash after R1 confirmed, before transition** | R1 receipt received, no transition record | Re-evaluate authorization and record transition | `EFFECTIVE` upon record |
| **Crash after transition, before R2** | Transition recorded, closure `PENDING` | Check deadline (300s); if active, generate R2 | `EFFECTIVE` (within deadline) |
| **Crash after R2 submitted, before R2 receipt** | R2 payload stored, closure `PENDING` | Lookup R2 receipt; confirm closure or suspend if expired | `CONFIRMED` or `SUSPENDED` |

---

## 8. Downstream Firewall Enforcement (Sections 27, 28)

The `DownstreamUseFirewall` enforces fail-closed boundary protection across all lifecycle states:

| Authority Effective Status | Closure Status | Production Decision | Empirical Denominator | Model Tuning | Learning Claim |
|---|---|---|---|---|---|
| `BLOCKED_PENDING_FINALIZATION_ROOT` | Any | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| `BLOCKED_INTEGRITY_HOLD` | Any | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| `EFFECTIVE` | `PENDING` | **ALLOWED** | **ALLOWED** | **ALLOWED** | **ALLOWED** |
| `EFFECTIVE` | `CONFIRMED` | **ALLOWED** | **ALLOWED** | **ALLOWED** | **ALLOWED** |
| `SUSPENDED_PENDING_CLOSURE` | `EXPIRED` | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| `SUSPENDED_INTEGRITY_HOLD` | `CONFLICT` | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |

---

## 9. Verification Evidence

- **Synthetic Two-Phase Root Test Suite (`tests/test_smart_money_two_phase_root.py`):** 17 / 17 PASS
- **Finalization Root Regression Suite (`tests/test_smart_money_finalization_root.py`):** 23 / 23 PASS
- **Anchor Policy Resolution Suite (`tests/test_smart_money_anchor_policy_resolution.py`):** 23 / 23 PASS
- **Anchor Finalization Subsystem Suite (`tests/test_smart_money_anchor_finalization.py`):** 18 / 18 PASS
- **Public Availability Observer Suite (`tests/test_sec_public_availability_observer.py`):** 13 / 13 PASS
- **Total Suite Passing:** 94 / 94 tests PASS
- **Candidate 004 Protected Files Changed:** 0 (`shadow_evidence.db`, `analyst_dashboard/vcp`, `natural_trigger.py` unmodified)
- **PDS Denominator:** Strict `HOLD` preserved
- **Production Status:** Zero production release, zero remote push, zero merge to main
