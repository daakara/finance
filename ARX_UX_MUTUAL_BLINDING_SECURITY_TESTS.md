# ARX Terminal — Mutual Blinding Enforcement & Qualification Security Tests (Gate G)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate G — Mutual Blinding and Qualification Release  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `VERIFIED_MUTUAL_BLINDING_CONTROLS`  

---

## 1. Blinding & Scoring Architectural Controls

The scoring bridge is implemented in `analyst_dashboard/research/custody/scoring_bridge.py` via `MutualBlindingScoringBridge`. It enforces strict cryptographic and procedural blinding barriers between research participants, gold standard answers, and qualification adjudication:

```
[Observer 1 Submission] ---> Sealed in Vault (.enc)
                                      |
                     [Mutual Blinding Isolation Barrier]
                     Neither observer sees peer responses
                     Neither observer receives Gold Standard
                                      |
[Observer 2 Submission] ---> Sealed in Vault (.enc)
                                      |
                                      v
          Are BOTH Submissions Independently Sealed in Ledger?
                         /                     \
                       YES                      NO -> BlindingError (Halt Scoring)
                       /
[Ephemeral In-Memory Decryption via Custody Key] (Zero Plaintext Disk Writes)
                       |
       +---------------+---------------+
       |                               |
       v                               v
[Measurement A: Observer vs Gold]    [Measurement B: Pairwise Observer 1 vs 2]
- Micro-Pooled Kappa >= 0.85         - Micro-Pooled Kappa >= 0.85
- Macro-Active Kappa >= 0.85         - Dispute Logging for Arbitration
- Critical Safety FN == 0
       |                               |
       +---------------+---------------+
                       |
                       v
[Authoritative Gate Verdict Resolution]
Synthetic Mode: SYNTHETIC_REHEARSAL_VERIFIED
(Never emitted as HUMAN_OBSERVER_QUALIFIED)
```

---

## 2. Technical Blinding Invariants

1. **Observer-to-Observer Seclusion:** Observer A cannot access Observer B's ciphertext; each submission is encrypted with a unique 96-bit nonce and distinct associated authenticated data (`aad = f"SUBM-{batch_id}-{observer_id}|{sha}"`).
2. **Gold Standard Quarantine:** Gold keys reside in `tests/fixtures/qualification/custodian_package/gold_key_consolidated_30.json` and are never packaged into observer distribution packages (`ARX_OBSERVER_QUALIFICATION_BUNDLE.zip`). The `IntakePipeline` has zero references to gold keys.
3. **Scoring Lockstep Precondition:** Scoring requires both raters to have active, non-withdrawn records in the custody ledger with status `PAIR_SEALED_ELIGIBLE_FOR_SCORING`. Scoring with a single submission raises `BlindingError`.
4. **In-Memory Decryption:** During scoring, ciphertexts are decrypted directly into memory buffers. Plaintext ratings are never written to permanent disk files.
5. **Synthetic Boundary Protection:** In synthetic reference mode (`is_synthetic_mode = True`), passing scores are emitted strictly as `SYNTHETIC_REHEARSAL_VERIFIED`. The system refuses to mark synthetic runs as `HUMAN_OBSERVER_QUALIFIED`.

---

## 3. Adversarial Test Evidence

The mutual blinding engine was verified across 5 adversarial test scenarios in `tests/test_evidence_custody_security.py`:

| Test ID | Scenario Description | Tested Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_14` | Gold-Key Access Prevention | `IntakePipeline` has zero gold key attributes | **PASSED** |
| `test_15` | Cross-Access Prevention | Rater 1 cannot read Rater 2 sealed evidence | **PASSED** |
| `test_16` | Premature Scoring Defense | Paired scoring with 1 submission raises `BlindingError` | **PASSED** |
| `test_17` | Dual Sealed Scoring Execution | Both sealed; scores evaluate to `SYNTHETIC_REHEARSAL_VERIFIED` | **PASSED** |
| `test_18` | Withdrawal Invalidation | Withdrawal purges data; subsequent scoring fails closed | **PASSED** |

### Execution Command:
```powershell
python -m pytest tests/test_evidence_custody_security.py -k "gold_key or cross_access or premature or dual or withdrawal" -v
```
**Exit Code:** `0` (5 passed)

---

## 4. Operational Boundaries

- **Human Observer Calibration:** `NOT_COMPLETED` (Held pending separate authorization)
- **Real Observer Qualification:** `0 Qualified`
- **Pilot P00 Activation:** `STRICTLY NOT AUTHORIZED`
