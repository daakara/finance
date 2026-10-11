# ARX Terminal — SEC Evidence Compatibility & Replay Contract (Epoch 001)

**Document Identifier:** `SEC_EVIDENCE_COMPATIBILITY_AND_REPLAY_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Six-Level Historical Replay Hierarchy

Evidence bundles must support completely offline, deterministic verification without network access to SEC:

1. **`BYTE_INTEGRITY_REPLAY`**: Verifies SHA-256 hashes of all retained raw payloads.
2. **`SOURCE_POPULATION_REPLAY`**: Re-evaluates the EOD exact-set reconciliation theorem across all listed filenames.
3. **`SCOPED_EXPECTED_SET_REPLAY`**: Replays the Form 4 and 106-CIK scope filters against the closed population.
4. **`PUBLIC_OBSERVATION_REPLAY`**: Verifies public observation records, HTTP statuses, and conservative knowledge times.
5. **`DECISION_INPUT_REPLAY`**: Re-evaluates the `SMART_MONEY_OPEN_MARKET_PURCHASE_V1` dependency contract.
6. **`CLOSURE_PROOF_REPLAY`**: Recomputes the entire bundle hash chain from genesis to head.

---

## 2. Invariable Historical Replay Principle

Historical bundles must always replay under their **original** source contract, parser version, canonicalization rules, and compatibility epoch. An automated function such as `upgrade_old_bundle_to_current_semantics()` is strictly prohibited in authoritative replay paths.

---

## 3. Contract Change Requalification Rule

If the SEC updates its PDS technical specification or dissemination protocols:
- `SOURCE_CONTRACT_CHANGE_DETECTED = YES`
- `SOURCE_AUTHORITY_STATUS = REQUALIFICATION_REQUIRED`
- No new bundles may be sealed with `AUTHORITATIVE_CLOSED_INTERVAL` until the new contract passes formal qualification.
