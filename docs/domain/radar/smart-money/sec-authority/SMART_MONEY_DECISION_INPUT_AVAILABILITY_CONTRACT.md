# ARX Terminal — Smart Money Decision-Input Availability Contract (Epoch 001)

**Document Identifier:** `SMART_MONEY_DECISION_INPUT_AVAILABILITY_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Contract ID:** `SMART_MONEY_OPEN_MARKET_PURCHASE_V1`  
**Contract SHA-256:** `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Purpose & Audit Grounding

Replaces subjective terms such as "semantically usable" with a deterministic, mathematically verifiable dependency contract derived directly from the active Smart Money Form 4 parser (`form4_parser.py`).

---

## 2. Dependency Specification

### Required Metadata Fields:
1. `issuerCik` (10-digit zero-padded CIK)
2. `documentType` (Must be `"4"` or `"4/A"`)
3. `periodOfReport` (Filing date / transaction date anchor)

### Required Reporting Owner Fields:
1. `rptOwnerCik` (10-digit insider CIK)
2. `rptOwnerName` (Insider entity/individual name)
3. `relationship` (At least one of `isDirector`, `isOfficer`, `isTenPercentOwner` must be boolean parseable)

### Conditional Fields:
1. `officerTitle`: Mandatory if `isOfficer == True`.

### Required Non-Derivative Transaction Fields:
1. `transactionDate` (ISO Date `YYYY-MM-DD`)
2. `transactionCode` (Must equal `"P"` for Open Market Purchase)
3. `transactionAcquiredDisposedCode` (Must equal `"A"` for Acquired)
4. `transactionShares` (Must parse as float $> 0.0$)
5. `transactionPricePerShare` (Must parse as float $> 0.0$)
6. `directOrIndirectOwnership` (Must equal `"D"` or `"I"`)

### Footnote Dependencies:
- Any footnote ID referenced via `<footnoteId id="...">` must resolve to an entry in `<footnotes><footnote id="...">...</footnote></footnotes>`.

---

## 3. Decision-Input Knowledge Time

`FIRST_PROVEN_DECISION_INPUT_COMPLETE_AT` is established at the **`response_completed_at`** timestamp of the earliest valid public observation that satisfies all dependency rules above.
