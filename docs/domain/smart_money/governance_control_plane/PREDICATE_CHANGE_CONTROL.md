# ARX Terminal — Smart Money
# Predicate Change Control & Mutation Governance V1

**Identifier:** `urn:arx:governance:predicate_change_control:v1`  
**Status:** RATIFIED_NORMATIVE  
**Governance Epoch:** `epoch-gov-001`  
**Schema:** `ARX_SMART_MONEY_PREDICATE_CHANGE_CONTROL_V1`

---

## 1. Overview & Principle of Monotonicity

Any modification to a dependency predicate, guarded obligation, or failure behavior carries the potential to subtly weaken security or authorization boundaries.
The **Predicate Change Control Protocol** guarantees that:
1. All changes are classified mechanically by comparing activation coverage across finite domain state spaces.
2. Silent weakening of security, custody, or authorization requirements is mathematically prevented.
3. Any weakening requires explicit, recorded operator ratification.
4. Any change to a predicate or dependency edge version immediately invalidates prior dependent decision snapshots and authorization tokens.

---

## 2. Predicate Relationship Classification

When comparing an existing predicate $P_{\text{old}}$ to a candidate predicate $P_{\text{new}}$, the relationship is evaluated using static state-space enumeration over declared input domains:

$$\text{PREDICATE\_CHANGE\_RELATIONSHIP} \in \{\text{EQUIVALENT}, \text{STRICTER}, \text{WEAKER}, \text{INCOMPARABLE}, \text{UNKNOWN}\}$$

### Definitions:
* **`EQUIVALENT`**: For every possible state combination $s \in \mathcal{S}$, $P_{\text{new}}(s) = P_{\text{old}}(s)$.
* **`STRICTER`**: The obligation applies in the same or more contexts. Formally, $P_{\text{new}}(s) = \text{TRUE} \implies P_{\text{old}}(s) = \text{TRUE}$, and $\exists s: P_{\text{old}}(s) \neq \text{TRUE} \land P_{\text{new}}(s) = \text{TRUE}$.
* **`WEAKER`**: The obligation applies in strictly fewer contexts. Formally, $P_{\text{old}}(s) = \text{TRUE} \implies P_{\text{new}}(s) = \text{TRUE}$, and $\exists s: P_{\text{new}}(s) \neq \text{TRUE} \land P_{\text{old}}(s) = \text{TRUE}$.
* **`INCOMPARABLE`**: Some contexts gained activation while other contexts lost activation.
* **`UNKNOWN`**: The domain is infinite or relationship cannot be mechanically proven.

---

## 3. Governance Approval Gate

The following mutations **strictly require** explicit operator governance approval (`governance_approval: true`):
1. **`MANDATORY_TO_CONDITIONAL`**: Converting a mandatory requirement to a guarded condition.
2. **`CONDITIONAL_TO_OPTIONAL`**: Converting a guarded condition to an optional dependency.
3. **`PREDICATE_WEAKER`**: Deploying a weaker predicate version.
4. **`PREDICATE_INCOMPARABLE`**: Deploying an incomparable predicate version.
5. **`UNKNOWN_BEHAVIOR_WEAKENED`**: Downgrading `unknown_behavior` (e.g. from `BLOCK_TARGET` to `MARK_UNKNOWN`).
6. **`CONFLICT_BEHAVIOR_WEAKENED`**: Downgrading `conflict_behavior` (e.g. from `BLOCK_TARGET` to `MARK_CONFLICT`).

**Enforcement:** Attempting to compile or apply any of the above mutations without explicit ratification raises `UnauthorizedPredicateWeakeningException` and fails closed.

---

## 4. Decision & Authorization Invalidation (TOCTOU Defense)

1. A dependency policy is hashed into `dependency_policy_hash`.
2. Evaluated guarded obligations produce a canonical `dependency_resolution_hash`.
3. Authorizations bind directly to both `decision_hash` and `dependency_resolution_hash`.
4. If a predicate or edge changes version or hash:
   $$\text{dependency\_resolution\_hash}_{\text{current}} \neq \text{dependency\_resolution\_hash}_{\text{auth}}$$
   The authorization is immediately marked `STALE` and rejected upon exercise.
