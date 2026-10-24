# ARX TERMINAL — EPOCH 4 OBSERVATION MANIFEST V6 SUCCESSION RECORD

**Date:** 2026-10-23T11:48:00Z (Local: 2026-10-23T13:48:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_MUTATION_AUTHORIZED`  
**Target Manifest File:** `EPOCH_4_MANIFEST_V6.json`  
**Superseded Manifest:** `EPOCH_4_MANIFEST_V5.json`  

---

## 1. Lineage & Cryptographic Parentage

- **Manifest Version:** `6.0.0`
- **Parent Manifest:** `EPOCH_4_MANIFEST_V5.json`
- **Parent Runtime SHA:** `99bea6ebc9b4f31634ef994bd2630710d89555940a5e491e98e1285668ce04c5`
- **Certified Observation Governance Hash:** `dba05d9f625ef06d2497c1b3c4e412f64b5bfb210c3569aaad33fbb9d1bf9a01`
- **Decision Engine Pinned Version:** `2.5.2`
- **Certified Source Snapshot Commit:** `7980d3a6e5dde0fe172d215c729ec213b9a164b1`

---

## 2. Inventory of Governed Files (10/10 Audited)

| File Path | Previous SHA-256 (V5) | Certified SHA-256 (V6) | Delta Justification |
|---|---|---|---|
| `analyst_dashboard/governance/passive_capture.py` | `fd3cae6f...` | `fd3cae6f682640b7182468dba053c0e97bcf9f1a73a2ca01ef5d6b81199dea93` | Unmodified |
| `analyst_dashboard/governance/experiment_ledger.py` | `233f996c...` | `897920d2ed5379f0b021f8452e82760599402600e069df96317c313e14196cc1` | Lineage & V2.5.2 bindings |
| `analyst_dashboard/governance/governance_db.py` | `aa11e792...` | `aa11e7920c290d5acd65b3c93bc232e42d481f59b2a49800b9407ed4e9658c8f` | Unmodified |
| `analyst_dashboard/governance/storage.py` | `ebb0ceec...` | `ebb0ceecac1acef022d23637f97efb96fd751763f4078412e1f95d052354c889` | Unmodified |
| `analyst_dashboard/data/fred_fetcher.py` | `d88dd714...` | `d88dd714d76af3eb112a966db02b71f10308c8e36b3f5f089734a5c3c4307c08` | Unmodified |
| `api/routes/analytics.py` | `e16428bd...` | `361ae1daa6ca5379222b65f00cc31c4bad0ec5589d53dac3d7c2ad9e39aea278` | Corporate Events RC5 API route |
| `analyst_dashboard/data/alpaca_fetcher.py` | `f409b756...` | `0764e55916c8a426a5397926c46ac891fcbeb3f12a1b11bb0b80ec3ee64714c8` | Preserved mainline baseline |
| `analyst_dashboard/data/market_price_state.py` | `83846723...` | `83846723d76a9b3e0a91cda0fc9373179075fe5773ebceb89a1e30e3e505a797` | Unmodified |
| `api/routes/screener.py` | `da146804...` | `054206f9f7b12bcf2a771b6f0f16618964430eb8dc4d0d5ae1c5e0a07d6c88af` | Preserved mainline baseline |
| `analyst_dashboard/analyzers/gem_screener.py` | `7ffbe27a...` | `fa1c425f1f09bc3b45c6b6406a5521e1884a96cb7a01c12ff94d316a1639f780` | Preserved mainline baseline |
