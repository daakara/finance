# ARX Terminal — Independent Observer Restricted Submission Custody Policy (v2.1.0)

> [!IMPORTANT]
> **GOVERNANCE STATUS: RESTRICTED SUBMISSION CUSTODY MODEL RATIFIED**  
> Under ARX Closure Model v2, human observer research submissions are treated as sensitive research data. Raw submissions containing human evaluations and potentially identifying researcher notes must NOT be committed to ordinary, public, or broadly distributed Git repository history. True custody integrity is established through cryptographic intake hashing, separate identity mapping, and restricted off-repository vault controls.

**Authority:** Lead Investment Decision Systems Auditor & Custody Officer  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Scope:** Intake Provenance, Cryptographic Hashing, Access Control, De-Identification, and Mutual Blinding  
**Date:** 2026-10-23  

---

## 1. Principles of Restricted Evidence Custody

1. **No Identifiable Submissions in Public Git:** Raw human rating files and identity linking tables are excluded from ordinary version control tracking (`.gitignore`) to satisfy research ethics, IRB guidelines, and data protection mandates (GDPR/privacy laws).
2. **Cryptographic Intake Hashing:** The SHA-256 digest of each raw submission file is computed immediately at the intake boundary before the file is opened or moved.
3. **Independent Custody Ledger:** Intake events, timestamps, file sizes, and SHA-256 digests are recorded in an append-only, access-restricted custody ledger.
4. **Separate Identity Mapping (Pseudonymization):**
   - Candidate legal identities, contact details, and institutional credentials are encrypted and stored in a private identity vault accessible solely to the Custody Officer.
   - Ratings files reference only anonymous rater keys (`OBS-HUMAN-01`, `OBS-HUMAN-02`).
5. **Mutual Blinding & Lockstep Scoring:**
   - Neither observer may view peer ratings or aggregate scoring before submitting.
   - Evaluation scripts are triggered only after both independent submissions are verified and sealed.
6. **Data Retention & Disposal:** Raw rating files are retained in the secure restricted vault for the duration of the audit cycle and deleted or permanently archived in accordance with research protocol disposal policies.

---

## 2. Ingestion & Scoring Workflow

```
[Observer Returns Rating JSON Files via Secure Channel]
                            |
           [1. Intake Provenance Verification]
           Verify secure channel, timestamp, and pseudonym token
                            |
           [2. Immediate Cryptographic Hashing]
           Compute SHA-256 byte digest and record in Custody Ledger
                            |
           [3. Schema & Trial Completeness Audit]
           Validate Draft-07 schema (ALPHA-01..10, BETA-01..10, GAMMA-01..10)
                            |
           [4. Secure Vault Placement (Off-Git)]
           Store in access-controlled lockbox directory (git-ignored)
                            |
           Are BOTH Observers Submitted and Locked?
                        /               \
                      YES                NO -> Standstill (Hold scoring)
                      /
           [5. Lockstep Automated Scoring via scoring_utility.py]
```

---

## 3. Current Lockbox Status
- **Observer 1 (`OBS-HUMAN-01`):** `AWAITING_HUMAN_SUBMISSION` (Status: `NOT_EXECUTED`)
- **Observer 2 (`OBS-HUMAN-02`):** `AWAITING_HUMAN_SUBMISSION` (Status: `NOT_EXECUTED`)
- **Active Submissions in Vault:** 0
- **Adjudication Lockbox State:** `LOCKED_PENDING_SUBMISSIONS`
