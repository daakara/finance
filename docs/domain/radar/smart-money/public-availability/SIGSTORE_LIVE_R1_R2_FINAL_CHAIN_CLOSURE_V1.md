# ARX Terminal — Smart Money: Live R1/R2 Final Chain Closure Gate V1

## Executive Summary
This gate establishes the formal cryptographic closure and sealing of the **Live R1/R2 Provenance Chain** under the Sigstore/Rekor public transparency witness architecture. All preceding operational and temporal defects from V1 have been resolved in the successor evidence namespace `evidence/sigstore-live-r1-r2-v2/`:
1. **Temporal Ordering:** Successor Authority Event (`2026-10-11T17:54:25Z`) and Successor R2 (`2026-10-11T17:54:27Z`) are strictly causally and chronologically posterior to the verified live external R1 Rekor witness time (`2026-10-11T17:31:03Z`).
2. **Second Live Same-Root Bundle Defense:** The exact canonical R1 payload was submitted a second time to public Sigstore producing `r1_bundle_b.sigstore.json` (`28024a42a7279...`). While cryptographically valid, verifier logic explicitly rejected substituting Bundle B into the chain when Bundle A is authoritative (`f35c0b91c1f5...`).
3. **Verifier-Enforced Negative Controls:** Cross-run transplants, phase inversions, payload bit flips, and authority event byte mutations are all rejected by actual verifier logic.
4. **Policy Set Authenticity:** Frozen policy source artifact `policy/policy_set.json` is preserved and validated.
5. **TUF Provenance:** Complete TUF metadata set (`root.json`, `snapshot.json`, `targets.json`, `timestamp.json`, `trusted_root.json`) is preserved and verified.
6. **Reviewer Identity Bound:** Hashes for `sigstore_operational_reviewer.py`, `canonicalization.py`, `sigstore_interim_verifier.py`, and git commit SHA are permanently bound into the manifest.
7. **Package-Only Reconstruction V2:** 0 mismatches, status `PASS`.

---

## Canonical Closure State
- **Gate:** `LIVE_R1_R2_FINAL_CHAIN_CLOSURE_V1`
- **Prior Chain Status:** `CRYPTOGRAPHICALLY_VALID_BUT_TEMPORALLY_DEFECTIVE`
- **Successor Run ID:** `run_live_qual_002`
- **Live R1/R2 Provenance Chain:** `SEALED`
- **Unattended Scheduler Signing:** `OPEN` (separate pending gate)
- **Natural Observer Arming:** `NOT_AUTHORIZED`
- **Natural Denominator Target:** `0`
- **Candidate 004 Interference:** `NONE` (0 diff vs `63131b9`)
