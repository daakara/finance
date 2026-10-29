# ARX TERMINAL — MARKET-WIDE RESEARCH DECISION & EVIDENCE REGISTER
## Comprehensive Reconciliation of Directives ARX-MW-004A through ARX-MW-004H V2
**Review Date:** 29 October 2026  
**Parent Investigation:** Market-Wide Top 35 Opportunity Discovery (MW-004 Series)  
**Ratification Status:** `ARCHITECTURE_DIRECTION_RATIFIED / PROPOSED_FOR_OWNER_RATIFICATION`  
**Historical Research Baseline Anchor:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`  
**Current Mainline Reference:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` (PR #129 sizing isolation merged)  
**Draft PR #128 Remote Reference:** `3fff68108593f304275cdf18b696c8697f5b8257` (`poc/arx-market-wide-top35` — **OPEN, UNMERGED REFERENCE**)  
**Candidate Implementation Commit:** `f154032152710a51b209d80a4a21dfb655d312e4` (Tree `30345ac093a913e6a50c8d0a018d77d52083c010`)  
**Ratification Addendum Commit:** `ac074ea487a55c276ee64db59f3aa8f278d2b992`  
**Operational Posture:** `LOCAL_RESEARCH_ONLY / EXACT_SOURCE / FAIL_CLOSED`  

---

## 1. Architectural Ratification & Decision Ledger

Under Directive ARX-MW-004H, the Product Owner reviewed comparative empirical evidence across architectural options and issued a formal ratification verdict:

### 1.1 Selected Canonical Architecture
- **Architecture Selected:** **Model E+ (Enhanced Security Registry Core) with Derived F1-Style Opportunity Projections**.
- **Formal Verdict:** `ARCHITECTURE_DIRECTION_RATIFIED`.
- **Equivalence Finding:** Within the tested market-wide discovery scope, Model E+ and Model F1 are verified as:
  $$\text{FUNCTIONALLY\_EQUIVALENT\_WITHIN\_TESTED\_SCOPE}$$
- **Comparative Options Evaluated & Rejected:**
  - *Option C (Multi-Lane Partitioned Registries):* Rejected due to excessive state duplication, cross-lane synchronization overhead, and divergence risks in security identifiers.
  - *Model F2 (Stateless Ephemeral Opportunity Streams):* Rejected due to loss of temporal provenance, inability to track cohort state transitions across evaluation cycles, and high re-computation cost.
  - *Full Persistent Opportunity Object Architecture (Pure F1):* Determined to be premature and unjustified. Introducing a mandatory secondary persistent opportunity entity incurs broad lifecycle costs (schema migrations, dual-write consistency, GC policies) that are unnecessary when Model E+ achieves identical ranking and projection fidelity.

### 1.2 The "15-Line Adapter" Reality
Early investigation notes colloquially characterized the Model E+ to F1 projection adapter as a simple "15-line mapping function." The formal review clarifies that while the core Python mapping is concise, the true lifecycle scope includes:
- Serialization firewalls (`to_dict()` sanitization).
- Dataclass trust boundary validators (`verify_research_trust_boundary()`).
- Invariant filters enforcing 0-share execution clamps.
- Strategy-specific qualification predicates.
Treating this contract as merely "15 lines" trivializes the defensive boundary engineering required to isolate research signals from execution systems.

---

## 2. Measurement, Scoring Remediation & Falsification of Contaminated Evidence

### 2.1 The Scoring Fallback Defect
Early iterations of the MW-004 research suffered from a silent naming discrepancy:
- Python model produced: `confluence_score`
- Deserializer expected: `confluenceScore`
- Fallback behavior: Silently injected a static score of `65.0`.

This naming defect contaminated initial discovery runs, causing unranked assets with missing data to cluster at exactly 65.0 and distorting cohort qualification.

### 2.2 Falsification of Historical Findings
All earlier research conclusions that relied on scores generated under the unpatched fallback are formally classified as:
$$\text{FALSIFIED\_AND\_SUPERSEDED}$$
These historical artifacts are preserved strictly for audit integrity and must never be cited as empirical performance evidence.

### 2.3 Score Contract Hardening
The frozen candidate (`f154032`) resolves the defect through structural typing:
- Introduction of explicit `ScoreProvenance` enum:
  - `GENUINE_COMPUTED`: All four analytical pillars evaluated with live data.
  - `PENALIZED_MISSING`: Verified absence of data resulting in mathematical score degradation.
  - `IMPUTED_NEUTRAL`: Transparent, logged model default with zero execution weight.
- Deserialization validation rejects unrecognized attributes and guarantees that missing tokens cannot masquerade as valid scores.

---

## 3. Universe, Population Arithmetic & Candidate Identification

The universe filtering funnel has been mathematically reconciled across all artifacts:

```
[ Full SEC Master Directory ] ────────────────────> 13,308 Records
                      │
                      ▼ (Common Stock Filter: Share Class 'A', Ordinary)
[ Eligible US Common Equities ] ─────────────────> 5,792 Equities
                      │
                      ▼ (Bounded Clean Store Ingestion: Minimum Daily Liquidity)
[ Stored Scannable Equities ] ───────────────────> 127 Equities (2.19% of Eligible Universe)
                      │
                      ├──────────────────────────> 41 Equities: Stage 1 Rejections (Volume, Price < $5, History < 50d)
                      ▼
[ Stage 2 Evaluated Equities ] ──────────────────> 86 Equities
                      │
                      ├──────────────────────────> 46 Equities: Cohort A Qualified (R:R >= 2.0, Trend Stage 2)
                      ├──────────────────────────> 40 Equities: Cohort B Developing (Setup Forming, Consolidating)
                      ▼
[ Top 35 Research Display Pool ] ───────────────> 35 Equities Displayed in Terminal Cockpit
                      │
                      └──────────────────────────> 11 Equities: Qualified Overflow Pool (Retained in Memory)
```

### 3.1 Bounded Store Database Reconciliation
- **Baseline SQLite Hash (MW-004G):** `ab31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615` (4,714,496 bytes).
- **Candidate SQLite Hash (MW-004H):** `00c0690efebfa5e419beb3890c3bd251b2182b0994f50a4c32359bb18834d273` (5,705,728 bytes).
- **Row-Level Bitwise Parity:** `asset_ohlcv_daily` contains exactly 40,310 daily bars across both databases. Content hash is bitwise identical (`27934736360afd0fbd266c20c3663191a0a46426bd2ecc4494f96ae43805f41e`). The 991 KB size difference is proven to originate solely from B-tree page pre-allocation for five newly defined, completely empty research schema tables (`asset_catalyst_registry`, `asset_factor_snapshots`, `asset_ohlcv_provenance`, `insider_disclosures`, `sqlite_sequence`).

---

## 4. Historical Replay & Point-In-Time Limitations

A critical finding of Directive MW-004F is that the ARX research store does not capture microsecond SEC EDGAR filing acceptance timestamps for older quarters. 

Consequently, historical simulations can guarantee **Historical Reconstruction** (evaluating states using known historical data without forward lookahead bias), but cannot guarantee **Exact Original Replay** (reproducing the exact microsecond order of historical information arrival). 

The research candidate strictly enforces this boundary:
- Replay mode token is locked to: `HISTORICAL_RECONSTRUCTION`.
- Setting replay mode to `EXACT_ORIGINAL_REPLAY` without PIT acceptance timestamps raises a fatal validation exception (`FN-005`).

---

## 5. Seven Corrective Verifications on Frozen Candidate (`f154032`)

The candidate implementation commit `f154032152710a51b209d80a4a21dfb655d312e4` (Tree `30345ac093a913e6a50c8d0a018d77d52083c010`) successfully passes all seven corrective gates:

1. **`DEFECT-001` (In-Place Ranking Mutation):** Deep-copy isolation in `top35_selector.py` prevents downstream presentation consumers from mutating underlying ranking scores.
2. **`DEFECT-002` (Research Authority Bleed):** Strict dataclass barriers prevent research opportunity objects from being passed to execution endpoints. Confluence engine clamps trade sizing to 0 shares.
3. **`FN-001` (Missing Evidence Deserialization):** Deserializer verifies that missing evidence tokens are preserved and cannot be overwritten by default values.
4. **`FN-002` (Cohort Qualification Revalidation):** Selector enforces strict qualification predicates ($R:R \ge 2.0$, candle count $\ge 50$, Stage $\ne 4$) prior to cohort placement.
5. **`FN-003` (Score Provenance Transparency):** Distinguishes calculated 0.0 scores from missing-data null states.
6. **`FN-004` (Trust Boundary Firewall):** Blocks unauthorized execution permission escalation within research payloads.
7. **`FN-005` (Replay Mode Enforcement):** Rejects `EXACT_ORIGINAL_REPLAY` assertions without verified microsecond timestamp lineage.

### Verification Test Commands & Evidence
```bash
python -m pytest tests/test_arx_mw002_adversarial_falsification.py  # 18 passed in 4.92s
python tests/test_arx_mw002_adversarial_falsification.py            # 18 passed in 2.74s
python scratch/test_independent_candidate_verification.py           # 7/7 checks passed
python scratch/verify_mw004h_manifest_integrity.py                  # 100% bitwise row match
```

---

## 6. Remote Reference PR #128 Preservation

- **PR Number:** #128 (`poc/arx-market-wide-top35`)
- **Remote Head Commit:** `3fff68108593f304275cdf18b696c8697f5b8257`
- **Policy:** PR #128 is strictly retained as an open, unmerged research reference. It is not merged, closed, or overwritten with candidate `f154032`.
- **Candidate Status:** Candidate `f154032` is frozen locally in the dedicated research worktree and proposed for formal owner ratification.
