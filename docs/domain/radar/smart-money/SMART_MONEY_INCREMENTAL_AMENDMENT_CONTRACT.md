# ARX TERMINAL — SMART MONEY WAVE 2C
## FORM 4/A AMENDMENT RECONCILIATION & POINT-IN-TIME ISOLATION CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-AMEND-001
VERSION: 2.1.0
STATUS: RATIFIED
GOVERNANCE: ZERO_RETROACTIVE_LEAKAGE
```

---

### 1. Independent Public-Availability Principle

1. An amended Form 4 (Form 4/A) is an independent public event with its own SEC acceptance datetime:
   $$T_{\text{avail}}(4/A) > T_{\text{avail}}(4)$$
2. Form 4/A NEVER alters or overwrites the historical public availability or transaction records of the original Form 4.
3. Both the original filing and the amendment remain durably preserved in `form4_raw_filings` and `form4_parsed_transactions`.

---

### 2. Lineage Tracking (`sm_amendments`)

When a Form 4/A is ingested, the engine links the filings bidirectionally:
- Forward link: `accession(4/A) -> accession(4)` (`relationship = 'AMENDMENT'`)
- Reverse link: `accession(4) -> accession(4/A)` (`relationship = 'SUPERSEDED_OR_CORRECTED'`)

---

### 3. Point-in-Time Query Semantics (`as_of`)

In point-in-time evaluation at knowledge time $T_{\text{query}}$:

$$\text{Visible}(\text{query}) = \{ \text{tx} \mid \text{public\_availability\_at}(\text{tx}) \le T_{\text{query}} \}$$

1. **Before Amendment Arrival ($T_{\text{query}} < T_{\text{avail}}(4/A)$)**:
   - Query sees ONLY original Form 4 transactions.
   - Zero retroactive leakage or amendment values appear.
2. **After Amendment Arrival ($T_{\text{query}} \ge T_{\text{avail}}(4/A)$)**:
   - Query sees both original and amendment transactions, with amendment relationship reflected.
   - Dedup and reconciliation respect the historical sequence without rewriting past states.
