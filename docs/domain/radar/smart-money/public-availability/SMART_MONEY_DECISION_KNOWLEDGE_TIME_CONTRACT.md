# ARX Terminal — Smart Money Decision Knowledge-Time Contract (Epoch 001)

**Document Identifier:** `SMART_MONEY_DECISION_KNOWLEDGE_TIME_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent Decision-Input Availability Auditor  
**Operating Mode:** `DECISION_DEPENDENCY_AWARE` + `APPEND_ONLY` + `FAIL_CLOSED`

---

## 1. Frozen Decision-Input Contract Reference

This contract binds the exact dependencies required before a Form 4 filing can trigger Smart Money decision rules.

- **Contract ID:** `SMART_MONEY_OPEN_MARKET_PURCHASE_V1`
- **Canonical Content Hash:** `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0`
- **Hash Algorithm:** `SM_CERT_CANON_V1`

---

## 2. Decision Completeness Rules

A filing achieves `DECISION_INPUT_COMPLETE` if and only if it satisfies all normative dependency rules:

### A. Transaction Coding Semantics
1. **Required Transaction Code:** `transactionCode == 'P'` (Open market or private purchase).
2. **Required Direction:** `transactionAcquiredDisposedCode == 'A'` (Acquired).
3. **Strict Rejection:** `transactionCode == 'A'` (Grant, award, or other acquisition pursuant to Rule 16b-3) is **strictly rejected** as open market purchase evidence (`REQUIRED_TRANSACTION_CODE_MISSING`).

### B. Required Transaction Amount Fields
1. `transactionShares`: Must be present, positive numerical value ($> 0$).
2. `transactionPricePerShare`: Must be present, non-negative numerical value ($\ge 0$).
3. Missing or null price/shares results in `REQUIRED_PRICE_MISSING` or `REQUIRED_SHARES_MISSING`.

### C. Reporting Owner Identification
1. `rptOwnerCik`: Must be present and non-empty.
2. `reportingOwnerRelationship`: At least one active role flag (`isDirector`, `isOfficer`, `isTenPercentOwner`, `isOther`) must be true.

### D. Governed Footnote Dependency Rule
- Only footnotes that explicitly condition or modify the required transaction fields (`transactionCode`, `transactionShares`, `transactionPricePerShare`, `transactionAcquiredDisposedCode`) are decision-relevant.
- If a referenced footnote id on a required field is unresolved $\implies \text{REQUIRED\_FOOTNOTE\_UNRESOLVED}$.
- Footnotes attached to non-consumed fields (e.g., holdings following transaction, nature of indirect ownership, general remarks) do **not** block decision completeness.

---

## 3. Knowledge-Time Derivation

For any Form 4 decision consuming insider transaction parameters:
$$\tau_{\text{knowledge}} = \tau_{\text{decision\_complete}} = \min_{obs \in \text{DECISION\_INPUT\_COMPLETE}} (\tau_{\text{conservative}}(obs))$$

### Content Regression Invariant
If filing content was complete at $T_1$, but a later re-fetch at $T_2$ returns partial or corrupted bytes:
- `CONTENT_AVAILABILITY_REGRESSION` is flagged.
- Earlier proven knowledge time $\tau_{\text{knowledge}} = T_1$ remains permanently valid.
- Valid evidence cannot be retroactively destroyed by later transmission failures.
