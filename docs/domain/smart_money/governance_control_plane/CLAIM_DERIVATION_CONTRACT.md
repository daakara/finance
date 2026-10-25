# Claim Derivation & State Basis Contract

**Gate:** `CANONICAL INPUT AUTHORITY, REGISTRY COMPLETENESS & DECISION-CHAIN CLOSURE V1`  
**Standard:** `ARX-CLAIM-DERIVATION-CONTRACT-V1`

---

## 1. Principles of Governed Claim Derivation

1. **Non-Self-Assertion:**
   No canonical claim may be asserted directly by an operator, caller, or runtime parameter. Every claim must trace through a registered derivation rule in `CANONICAL_CLAIM_DERIVATION_POLICY.json`.
   
2. **Repository-Bound Grounding:**
   Claims that assert repository state (such as Candidate 004 freeze identity or source release provenance) must derive directly from bytes recomputed from the Git object database at the verified `SOURCE_FREEZE_SHA`.

3. **Candidate 004 Dual Identity:**
   - `CLAIM-CANDIDATE004-FREEZE-FILE-IDENTITY`: Established solely by the SHA-256 digest of `docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json` matching `297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90`.
   - `CLAIM-CANDIDATE004-SEMANTIC-CLOSURE-IDENTITY`: Semantic closure hash `53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6`. Because full automated dynamic semantic closure recomputation from arbitrary raw runtime inputs is not yet operationalized, if independent recomputation is absent, the semantic closure recomputation status is formally marked `NOT_ESTABLISHED` (or `ESTABLISHED_BY_RECORDED_ATTESTATION`). Retrieval of a static constant is never conflated with live dynamic mathematical derivation.

---

## 2. Six-Valued Claim State Model

Canonical claims must occupy exactly one of six states:

| Claim State | Definition | Admissible Transitions |
|-------------|------------|------------------------|
| `ESTABLISHED` | Authoritative evidence confirms the claimed condition is met. | -> `STALE`, `REVOKED`, `CONFLICT` |
| `NOT_ESTABLISHED` | Authoritative evidence confirms the condition is explicitly absent or failed. Requires a valid `state_basis`. | -> `ESTABLISHED`, `CONFLICT` |
| `UNKNOWN` | No authoritative evidence exists. Absence of evidence is UNKNOWN, NOT NOT_ESTABLISHED. | -> `ESTABLISHED`, `NOT_ESTABLISHED`, `CONFLICT` |
| `CONFLICT` | Mutually contradictory authoritative evidence records were observed. | -> `ESTABLISHED`, `NOT_ESTABLISHED` (upon governed resolution) |
| `STALE` | Previously established evidence has exceeded its maximum validity window. | -> `ESTABLISHED` (via fresh evidence), `REVOKED` |
| `REVOKED` | Explicit revocation event issued by an authorized governance authority. | Terminal unless new ceremony executed |

---

## 3. Strict State-Basis Rule for NOT_ESTABLISHED

A claim may only enter state `NOT_ESTABLISHED` if supported by an authoritative `state_basis`:

```json
{
  "basis_type": "AUTHORITATIVE_CONFIRMED_ABSENCE",
  "authority_ref": "urn:arx:authority:governance-control-plane",
  "state_source_ref": "urn:arx:source:local-git-tree",
  "observed_at": "2026-10-12T10:00:00Z",
  "validity": "CURRENT"
}
```

Any claim lacking evidence where no authoritative absence probe was executed must be classified as `UNKNOWN`. Laundered transitions (`UNKNOWN -> NOT_ESTABLISHED` without authoritative basis) are strictly prohibited (`UNKNOWN_STATE_LAUNDERED_AS_NOT_ESTABLISHED == 0`).
