# Smart Money Sigstore Interim Natural Rooting Policy V1

**Policy ID:** `SMART_MONEY_SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_V1`  
**Version:** `1.0.0`  
**Status:** `RATIFIED`  
**External Root Profile:** `SIGSTORE_REKOR_INTERIM_V1`  
**External Root Function:** `PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS`  
**Governing Epoch:** `EPOCH_001`  
**Mode:** `FAIL_CLOSED`  

---

## 1. Scope & Objective

This policy ratifies the operational requirements under which the interim Sigstore / Rekor transparency profile may be adopted for prospective natural anchor rooting. It defines the strict boundary conditions that must be fulfilled before natural observer arming is authorized.

---

## 2. Prerequisites for Prospective Natural Rooting

Prior to transition of natural prospective rooting to `AUTHORIZED`:
1. **Live Canonical R1 Provenance:** The canonical R1 pre-authority payload must be witnessed on public Rekor (`ASSUMP_001`).
2. **Live Canonical R2 Provenance:** The canonical R2 post-authority closure payload must be witnessed on public Rekor (`ASSUMP_002`).
3. **Exact Live R2 → R1 Binding:** Cryptographic hash binding linking live R2 to live R1 bundle must be proven on public Rekor (`ASSUMP_003`).
4. **Unattended Signing Identity:** Continuous unattended workload signing via short-lived OIDC tokens in CI/CD runner must be provisioned and verified (`ASSUMP_004`).
5. **Operational Verification Stack:** Continuous Rekor consistency monitoring, local dual-store raw evidence durability, cached TUF trust-root governance, and same-root substitution protection must be active (`ASSUMP_005`–`ASSUMP_011`).

---

## 3. Current Suitability Verdict

- **Interim Profile Suitability:** `CONDITIONAL`
- **Natural Observer Arming:** `NOT_AUTHORIZED`
- **Natural Denominator:** `0`
- **State Justification:** Assumptions 001, 002, 003, and 004 remain open pending live public submission and CI workload token configuration.

---

## 4. Operational Invariants

- **INV-ROOT-1:** Reusing connectivity test bundles (`arx-rekor-test.sigstore.json`) for canonical R1/R2 is strictly prohibited.
- **INV-ROOT-2:** Inability to produce unattended workload credentials halts natural admission fail-closed.
- **INV-ROOT-3:** Rekor split-view or tree regression halts admission unconditionally.
- **INV-ROOT-4:** All raw evidence blobs must have synchronous backup verified by SHA-256 replication.
