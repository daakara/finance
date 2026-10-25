# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE SPECIFICATION
### CLAIM SEMANTICS AND ORTHOGONAL AUTHORITY MODEL V1

**Document Identity**: `urn:arx:spec:claim-semantics-model:v1`  
**Security Classification**: RESTRICTED — GOVERNANCE CONTROL PLANE CONTRACT  
**Status**: CANONICAL AND RATIFIED  

---

### 1. Executive Summary & Problem Statement

Prior iterations of the canonical governance decision pipeline conflated epistemic observation state with authority, provenance, attestation, and freshness within a linear scalar lattice:

$$\mathtt{NOT\_ESTABLISHED} \to \mathtt{LOCAL\_REFERENCE} \to \mathtt{REPOSITORY\_BOUND} \to \mathtt{OPERATOR\_RATIFIED} \to \mathtt{EXTERNAL\_OPERATIONAL} \to \mathtt{EXTERNALLY\_QUALIFIED}$$

This scalar formulation violated fundamental epistemic invariants:
1. It conflated *what is known* (epistemic state) with *who asserts it* (provenance/attestation) and *when it was confirmed* (freshness).
2. It prevented representing valid state combinations such as $\mathtt{NOT\_ESTABLISHED}$ backed by $\mathtt{REPOSITORY\_GROUNDED}$ audit proof, or $\mathtt{CONFLICT}$ existing across $\mathtt{EXTERNAL\_OPERATIONAL}$ witnesses.
3. It collapsed disagreements and unverified absence directly into $\mathtt{NOT\_ESTABLISHED}$ without an authoritative basis of completeness, effectively laundering $\mathtt{UNKNOWN}$ states.

This specification replaces the scalar lattice with a multi-dimensional orthogonal claim semantics model.

---

### 2. Multi-Dimensional Orthogonal Claim Architecture

Every governed claim $C$ is uniquely characterized by five independent orthogonal dimensions and an explicit State Basis:

$$C = \langle \mathtt{claim\_state}, \mathtt{provenance\_class}, \mathtt{attestation\_status}, \mathtt{freshness}, \mathtt{authority\_scope}, \mathtt{state\_basis} \rangle$$

#### 2.1 Dimension 1: Claim State (`claim_state`)
The epistemic verification state of the claim proposition:
- `ESTABLISHED`: Formally proven by verified evidence matching all semantic acceptance criteria.
- `NOT_ESTABLISHED`: Formally proven absent by an authoritative search over a bounded, complete evidence universe.
- `UNKNOWN`: Insufficient evidence or unexecuted observation window. The epistemic value is indeterminate.
- `CONFLICT`: Irreconcilable disagreement between contradictory evidence records of equal or incomparable authority.
- `STALE`: Previously established observation whose maximum allowable validity TTL has expired.
- `REVOKED`: Explicitly countermanded by upstream revocation authority or compromise disclosure.

#### 2.2 Dimension 2: Provenance Class (`provenance_class`)
The physical origin and custody chain of the supporting evidence:
- `REPOSITORY_GROUNDED`: Recomputed directly from immutable Git tree blobs at the bound `source_freeze_sha`.
- `OPERATOR_ATTESTED`: Supported by signed operator credentials or cryptographic attestation bundles.
- `EXTERNAL_OPERATIONAL`: Derived from live external infrastructure (Sigstore Rekor, RFC 3161 TSA, WORM storage).
- `GENERATED_DERIVATION`: Deterministically produced by local policy engines or compilers.
- `NONE`: No physical evidence record attached (valid only for unevidenced or baseline absence assertions).

#### 2.3 Dimension 3: Attestation Status (`attestation_status`)
The formal governance ratification status of the claim:
- `UNRATIFIED`: Raw observation without formal governance ratification.
- `LOCAL_REFERENCE`: Evaluated and confirmed against local test suites and repository invariant rules.
- `OPERATOR_RATIFIED`: Bound to an explicit, cryptographically verifiable operator ratification artifact.
- `EXTERNALLY_ATTESTED`: Certified by a qualified external verification authority or independent witness.

#### 2.4 Dimension 4: Freshness (`freshness`)
Temporal validity status relative to the evaluation clock:
- `CURRENT`: Observed within the bounded clock skew window ($\pm 300\text{s}$) and within active validity TTL.
- `STALE`: Time-to-live expired; requires re-observation or audit re-derivation.
- `REVOKED`: Timestamp superseded or invalidated by explicit counter-attestation.

#### 2.5 Dimension 5: Authority Scope (`authority_scope`)
The precise domain boundary for which the claim is authoritative:
- Examples: `CANDIDATE_FREEZE_IDENTITY`, `GOVERNANCE_ADMISSION_DECISION`, `THEOREM_READINESS`, `EVIDENCE_CUSTODY_SPECIFICATION`.
- Global unqualified `AUTHORITATIVE` claims are strictly prohibited.

---

### 3. State Basis Specification for `NOT_ESTABLISHED`

The claim state `NOT_ESTABLISHED` must never mean merely *"we did not look"* or *"no record was provided"*. Asserting `NOT_ESTABLISHED` requires an explicit, structurally verified `StateBasis`:

```yaml
state_basis:
  basis_type: "AUTHORITATIVE_CONFIRMED_ABSENCE"
  authority_source: "urn:arx:authority:canonical-manifest"
  search_scope: "REPOSITORY_TREE"
  source_set: ["docs/domain/smart_money/", "analyst_dashboard/"]
  completeness_assertion: "EXHAUSTIVE_TREE_BLOB_SCAN"
  observation_window: "2026-10-12T00:00:00Z/2026-10-12T12:00:00Z"
  observed_at: "2026-10-12T12:00:00Z"
```

If any completeness parameter is omitted or cannot be authoritatively established, the claim state MUST default fail-closed to:
$$\mathtt{claim\_state} = \mathtt{UNKNOWN}$$

Invariants:
- `NOT_ESTABLISHED_WITHOUT_COMPLETE_STATE_BASIS == 0`
- `UNKNOWN_LAUNDERED_AS_NOT_ESTABLISHED == 0`

---

### 4. Conflict Preservation and Propagation Rules

#### 4.1 Supersession Rule
Record $R_2$ supersedes record $R_1$ if and only if:
1. $R_1.\mathtt{claim\_id} == R_2.\mathtt{claim\_id}$
2. $R_1.\mathtt{authority\_scope} == R_2.\mathtt{authority\_scope}$
3. $R_2.\mathtt{observed\_at} > R_1.\mathtt{observed\_at}$ (temporal ordering)
4. An explicit governed precedence rule ordains that $R_2.\mathtt{provenance\_class} > R_1.\mathtt{provenance\_class}$.

If precedence cannot be established, the system must NEVER collapse to $\mathtt{NOT\_ESTABLISHED}$ or $\mathtt{UNKNOWN}$. It MUST resolve to:
$$\mathtt{claim\_state} = \mathtt{CONFLICT}$$

Invariants:
- `CONFLICT_COLLAPSED_TO_NOT_ESTABLISHED == 0`
- `CONFLICT_COLLAPSED_TO_UNKNOWN == 0`

#### 4.2 Causal Conflict Propagation
Conflict propagates strictly through the derivation chain:
$$\text{Evidence } \mathtt{CONFLICT} \implies \text{Claim } \mathtt{CONFLICT} \implies \text{Fact } \mathtt{CONFLICT} \implies \text{Predicate } \mathtt{CONFLICT}$$

#### 4.3 Decisive Partial Logic Short-Circuits
In accordance with 4-valued Kleene/Belnap partial logic, conflict is preserved unless decisive operands determine the output:
- **Disjunction ($\lor$)**: $\mathtt{TRUE} \lor \mathtt{CONFLICT} = \mathtt{TRUE}$
- **Conjunction ($\land$)**: $\mathtt{FALSE} \land \mathtt{CONFLICT} = \mathtt{FALSE}$
- **Undecided**: $\mathtt{UNKNOWN} \land \mathtt{CONFLICT} = \mathtt{CONFLICT}$, $\mathtt{NOT}(\mathtt{CONFLICT}) = \mathtt{CONFLICT}$
