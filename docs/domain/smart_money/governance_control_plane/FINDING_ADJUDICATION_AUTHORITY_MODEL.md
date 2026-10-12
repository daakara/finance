# ARX TERMINAL — SMART MONEY
## GOVERNANCE CONTROL PLANE SPECIFICATION
### FINDING ADJUDICATION AUTHORITY, DUPLICATE STANDARDS & ANTI-EQUIVOCATION MODEL V1

**Document Identity**: `urn:arx:spec:finding-adjudication-authority-model:v1`  
**Security Classification**: RESTRICTED — GOVERNANCE CONTROL PLANE CONTRACT  
**Status**: CANONICAL AND RATIFIED  

---

### 1. Purpose & Authority Boundary

This specification establishes the normative rules governing finding admission, reclassification, duplicate resolution, and ratification authority within the ARX Terminal Governance Control Plane.

It prevents:
1. False claims of operator ratification (`OPERATOR_RATIFIED (local reference governance)`) when no cryptographically signed operator artifact exists;
2. Arbitrary or superficial duplicate classifications that conceal distinct architectural defects;
3. Circular, self-referential, or orphan duplicate chains;
4. Finding adjudication equivocation where the same finding sequence resolves inconsistently.

---

### 2. Operator Ratification Artifact Requirement

The label `OPERATOR_RATIFIED` is strictly reserved for findings backed by an authentic, cryptographically bound operator ratification artifact:

```yaml
operator_ratification:
  ratification_id: "urn:arx:ratification:operator-genesis-lead:2026-10-12"
  operator_identity: "urn:arx:identity:operator-lead-alpha:v1"
  finding_id: "FND-ARCH-001"
  adjudication_hash: "<sha256-hash-of-adjudication-record>"
  action: "RATIFY_ADMISSION"
  issued_at: "2026-10-12T12:00:00Z"
  scope: "GOVERNANCE_ADMISSION_DECISION"
```

In the absence of an authentic operator ratification artifact:
- `attestation_status` MUST be set to `LOCAL_REFERENCE`;
- `operational_authority` MUST be set to `false`;
- Claiming `OPERATOR_RATIFIED` without an authentic ratification artifact is strictly prohibited:
  $$\mathtt{OPERATOR\_RATIFIED\_WITHOUT\_RATIFICATION\_ARTIFACT} == 0$$

---

### 3. Duplicate Adjudication Standard

A finding may only be adjudicated as `DUPLICATE` if it is formally proven to share the exact same underlying root cause and causal dependency path as an existing canonical target. Similar wording or common subsystem domains are strictly insufficient.

Every duplicate adjudication requires a structured `DuplicateAnalysis`:
```yaml
duplicate_analysis:
  candidate_finding_id: "FND-ARCH-013"
  canonical_target_id: "FND-ARCH-007"
  root_cause_equivalence: "Both findings identify identical failure modes in Gate-3 uncommitted state recovery on runner crash or lease expiry."
  causal_path_equivalence: "Both findings map to Theorem B and Theorem C blocking dependencies."
  evidence_overlap: ["urn:arx:evidence:audit:gate3-recovery:001", "urn:arx:evidence:audit:lease-expiry-recovery:001"]
  verdict: "CONFIRMED_DUPLICATE"
```

Invariants:
- `DUPLICATE_WITHOUT_ROOT_CAUSE_EQUIVALENCE == 0`
- `DUPLICATE_WITHOUT_CANONICAL_TARGET == 0`
- `DUPLICATE_CYCLE_COUNT == 0` (cycles $A \to B \to A$ strictly fail-closed)
- `DUPLICATE_SELF_REFERENCE == 0` ($A \to A$ strictly rejected)

---

### 4. Finding Adjudication Anti-Equivocation

Every finding adjudication is sequentially tracked in an append-only hash chain:

$$H_i = \mathtt{sha256}(\mathtt{JCS}(\langle \mathtt{finding\_id}, i, H_{i-1}, \mathtt{adjudication\_body} \rangle))$$

Any attempt to commit two differing adjudications for the same finding at the same sequence index constitutes equivocation and triggers immediate fail-closed rejection:
$$\mathtt{FINDING\_ADJUDICATION\_EQUIVOCATION} == 0$$
