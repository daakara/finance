# ARX Terminal — Smart Money
# Conditional Dependency Predicate Authority Model V1

**Identifier:** `urn:arx:governance:conditional_dependency_model:v1`  
**Status:** RATIFIED_NORMATIVE  
**Governance Epoch:** `epoch-gov-001`  
**Schema:** `ARX_SMART_MONEY_CONDITIONAL_DEPENDENCY_MODEL_V1`

---

## 1. Executive Summary & Purpose

The **Conditional Dependency Predicate Authority Model** establishes a deterministic, fail-closed mechanism for guarded obligations within the ARX Terminal Smart Money Governance Control Plane. Prior to this model, conditionality was evaluated imperatively and ad-hoc, creating risks of silent fail-open transitions, undefined truth states, self-disabling guards, and global blast radius misattribution.

This model establishes:
1. **Guarded Obligations:** Declarative, versioned dependencies between source claims and target outputs, governed by canonical predicates.
2. **Four-Valued Partial Logic:** Strict formal evaluation over `TRUE`, `FALSE`, `UNKNOWN`, and `CONFLICT`, banning `UNKNOWN -> OPTIONAL` and `UNKNOWN -> FALSE`.
3. **Provenance-Aware Fact Evaluation:** Proof-relevant input tracking where synthetic fixtures or test-doubles taint proofs as `NON_AUTHORITATIVE`, preventing capability expansion.
4. **Local Blast Radius Isolation:** Failures block only the direct target and its transitive descendants, preserving evaluability of independent graph branches.
5. **Anti-Bypass & Monotonicity:** Prohibits self-disabling guards, circular graph references, and unauthorized predicate weakening.

---

## 2. Guarded Obligation Structure

A conditional dependency is never modeled as an arbitrary boolean flag. It is formally defined as a **Guarded Obligation**:

```yaml
guarded_obligation:
  dependency_id: str                      # Unique URN identifier (urn:arx:dep:...)
  source_claim_id: str                    # Claim being required or conditioned
  target_output_id: str                   # Output or gate dependent on the claim
  declared_requirement: CONDITIONAL       # Baseline requirement class (CONDITIONAL)
  predicate_ref:
    predicate_id: str                     # URN of the governing predicate
    predicate_version: str                # Semantic version of predicate
    predicate_hash: str                   # Canonical SHA-256 hash of normalized AST
  when_true:
    effective_requirement: MANDATORY     # Requirement when predicate evaluates TRUE
  when_false:
    effective_requirement: IRRELEVANT    # Requirement when predicate evaluates FALSE
  when_unknown:
    effective_requirement: UNRESOLVED    # Requirement when predicate evaluates UNKNOWN
  when_conflict:
    effective_requirement: UNRESOLVED    # Requirement when predicate evaluates CONFLICT
  target_safety_class:                   # AUTHORIZATION_CRITICAL | SAFETY_CRITICAL | GOVERNANCE_CRITICAL | INFORMATIONAL
  unknown_behavior:                      # BLOCK_TARGET | MARK_UNRESOLVED | MARK_UNKNOWN
  conflict_behavior:                     # BLOCK_TARGET | MARK_UNRESOLVED | MARK_CONFLICT
  rationale: str                          # Human-readable governance rationale
  dependency_version: str                 # Semantic version of the edge definition
  dependency_hash: str                    # Canonical SHA-256 hash of the edge definition
```

---

## 3. Predicate Result Model & 4-Valued Logic

### 3.1 Canonical Result States
Predicates evaluate to one of four mutually exclusive canonical states:
* **`TRUE`**: The predicate condition is authoritatively proven satisfied.
* **`FALSE`**: The predicate condition is authoritatively proven not satisfied.
* **`UNKNOWN`**: Insufficient, missing, stale, revoked, or non-authoritative evidence exists to determine truth.
* **`CONFLICT`**: Authoritative evidence sources provide contradictory values with equal precedence.

### 3.2 Effective Requirement Mapping
The mapping from predicate evaluation result to effective requirement is frozen and immutable:
* `TRUE` $\longrightarrow$ `effective_requirement: MANDATORY`
* `FALSE` $\longrightarrow$ `effective_requirement: IRRELEVANT`
* `UNKNOWN` $\longrightarrow$ `effective_requirement: UNRESOLVED`
* `CONFLICT` $\longrightarrow$ `effective_requirement: UNRESOLVED`

> **Forbidden Transitions:**
> * `UNKNOWN -> OPTIONAL` (STRICTLY FORBIDDEN)
> * `UNKNOWN -> FALSE` (STRICTLY FORBIDDEN)
> * `UNKNOWN -> IRRELEVANT` (STRICTLY FORBIDDEN)
> * `CONFLICT -> OPTIONAL` (STRICTLY FORBIDDEN)
> * `CONFLICT -> IRRELEVANT` (STRICTLY FORBIDDEN)

---

## 4. Target Safety Classes & Default Failure Behaviors

Targets are categorized by risk profile, dictating how `UNRESOLVED` states affect system execution:

| Target Safety Class | Unknown Behavior | Conflict Behavior | Downstream Authorization | Capability Impact |
|:---|:---|:---|:---|:---|
| **`AUTHORIZATION_CRITICAL`** | `BLOCK_TARGET` | `BLOCK_TARGET` | BLOCKED | Fail-closed (no capability) |
| **`SAFETY_CRITICAL`** | `BLOCK_TARGET` | `BLOCK_TARGET` | BLOCKED | Fail-closed (no capability) |
| **`GOVERNANCE_CRITICAL`** | `MARK_UNRESOLVED` | `MARK_UNRESOLVED` | BLOCKED | Blocker logged, downstream halted |
| **`INFORMATIONAL`** | `MARK_UNKNOWN` | `MARK_CONFLICT` | UNAFFECTED | Annotated in audit logs |

**Core Invariant:** Unresolved conditionality may never increase capability or permit execution authorization.

---

## 5. Provenance & Synthetic Taint Model

### 5.1 Decisive Inputs
The evaluation engine deterministically tracks all inputs that participate in the proof of a predicate.
* If a predicate evaluates to `TRUE` or `FALSE`, every input necessary to establish that result is recorded as a **decisive input**.

### 5.2 Provenance Classes & Authority Taint
Facts are tagged with their provenance class:
* `CANONICAL_LEDGER`
* `EXTERNAL_OPERATIONAL`
* `REPOSITORY_BOUND`
* `OPERATOR_RATIFIED`
* `LOCAL_REFERENCE`
* `DESIGN_CONTRACT`
* `SYNTHETIC_FIXTURE`
* `TEST_DOUBLE`

**Taint Rules:**
1. If ANY decisive input has provenance `SYNTHETIC_FIXTURE` or `TEST_DOUBLE`, the resulting predicate evaluation is assigned:
   `predicate_authority: NON_AUTHORITATIVE`.
2. A non-authoritative predicate resolution may **never** satisfy an `AUTHORIZATION_CRITICAL` or `SAFETY_CRITICAL` guard.
3. **Non-Participating Fact Isolation:** A synthetic fact present in the context that is unread or not decisive does **not** taint independent authoritative proofs.

---

## 6. Anti-Bypass Invariants

1. **Self-Disabling Guard Prohibition (`GUARD_SELF_DISABLING_REFERENCES == 0`):**  
   A guard on source claim $C$ must not query $C$'s absence or establishment status to make $C$ irrelevant (e.g. `is_established(C) == False -> C is not required`). This represents an illegal fail-open bypass and is rejected at compile time.
2. **Graph Cycle Rejection (`PREDICATE_DEPENDENCY_CYCLES == 0`, `TARGET_REFERENCE_CYCLES == 0`):**  
   Guards cannot depend circularly on their target output, any descendant of their target output, or any derived readiness value that depends on the same guard.
3. **Capability Monotonicity:**  
   Ambiguity (`UNKNOWN` or `CONFLICT`) may preserve or reduce system capability; it can never increase capability.
