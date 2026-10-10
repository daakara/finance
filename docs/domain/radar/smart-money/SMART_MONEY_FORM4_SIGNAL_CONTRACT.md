# SMART MONEY WAVE 1 — SEC FORM 4 SIGNAL CONTRACT

## 1. Executive Summary

This document establishes the deterministic, explainable, and rule-based signal semantics for SEC Form 4 insider transactions in ARX Terminal.

```text
SIGNAL_IS_RULE_BASED = YES
SIGNAL_IS_EXPLAINABLE = YES
OPAQUE_MODEL_SCORE = NO
MODEL_TUNING = FROZEN
```

ARX Terminal does not use opaque machine learning or returns-fitted scoring to rank insider transactions. All signals are arithmetic derivations grounded exclusively in statutory Form 4 reporting codes and disclosures.

---

## 2. Authoritative Form 4 Transaction Code Classification

The SEC defines standard transaction codes (General Instruction 8 to Form 4). In ARX Terminal, these codes are classified into four mutually exclusive behavioral categories:

| SEC Code | SEC Description | ARX Signal Class | Conviction Impact |
|:---|:---|:---|:---|
| **P** | Open market or private purchase of securities | `OPEN_MARKET_PURCHASE` | **Positive Buying Conviction** (if price > 0 and acquired) |
| **S** | Open market or private sale of securities | `OPEN_MARKET_SALE` | **Disposal / Trimming** |
| **M** | Exercise or conversion of derivative security (option) | `OPTION_EXERCISE` | **Non-Directional / Excluded from Purchase Conviction** |
| **A** | Grant, award, or other acquisition of stock | `EQUITY_GRANT` | **Non-Directional / Excluded from Purchase Conviction** |
| **G** | Bona fide gift of securities | `GIFT` | **Non-Directional / Excluded from Purchase Conviction** |
| **F** | Payment of exercise price or tax liability by withholding securities | `TAX_WITHHOLDING` | **Non-Directional / Excluded from Sale Conviction** |
| **D** | Sale or disposition to the issuer | `ISSUER_DISPOSITION` | **Non-Directional / Excluded from Sale Conviction** |
| **C** | Conversion of derivative security | `CONVERSION` | **Non-Directional / Excluded** |
| **J** | Other acquisition or disposition (must specify in footnote) | `OTHER_TRANSACTION` | **Non-Directional / Fail Closed** |

---

## 3. Directional Classification Semantics

### Positive Insider Buying Conviction Criteria
An insider transaction is classified as `POSITIVE_INSIDER_BUYING` **if and only if** ALL of the following criteria are satisfied:
1. `transaction_code == "P"` (Open market purchase).
2. `acquired_disposed_code == "A"` (Acquired).
3. `price_per_share > 0.0` (Real capital outlay).
4. `shares > 0`.
5. `is_derivative == False` (Common equity / non-derivative table).
6. Security title represents equity voting/economic shares (e.g. Common Stock, Ordinary Shares, Class A Common Stock).

Transactions with code `A` (equity awards/grants), `M` (option exercises), `G` (gifts), or `P` with `$0.00` price **must fail closed** and be excluded from positive buying conviction.

### Disposal Conviction Criteria
An insider transaction is classified as `DISPOSAL` if:
1. `transaction_code == "S"` (Open market sale).
2. `acquired_disposed_code == "D"` (Disposed).
3. `price_per_share > 0.0`.
4. `shares > 0`.

Transactions with code `F` (shares withheld by issuer to cover tax withholding upon vesting) are NOT voluntary open-market selling conviction and are classified as `TAX_WITHHOLDING`.

---

## 4. Insider Role & Hierarchy Weighting

When reporting owners submit Form 4, their relationship to the issuer is categorized by statutory check-boxes:
1. `is_director`: True/False.
2. `is_officer`: True/False (with `officer_title`).
3. `is_ten_percent_owner`: True/False.
4. `is_other`: True/False.

For explainable evidence presentation, roles are categorized in priority order:
- **C-Suite / Key Executive**: CEO, CFO, COO, President, Chairman. (Highest operational visibility into current corporate performance).
- **Other Officer**: General Counsel, CTO, EVP, SVP, VP.
- **Independent Director**: Board member without executive office.
- **Ten Percent Owner**: Institutional or activist blockholder holding >10%.

---

## 5. Clustering & Issuer-Level Aggregation

Individual Form 4 transactions are aggregated at the issuer level across defined recency windows (default: **30 calendar days**, secondary: **90 calendar days**).

### Issuer Conviction Metrics (Arithmetic Only)
- `net_purchase_value_usd`: Sum of `(shares * price)` for all `OPEN_MARKET_PURCHASE` transactions minus sum for `OPEN_MARKET_SALE` in the window.
- `gross_purchase_value_usd`: Sum of `(shares * price)` for all `OPEN_MARKET_PURCHASE`.
- `distinct_buyer_count`: Number of unique `reporting_owner_cik` with at least one `OPEN_MARKET_PURCHASE`.
- `cluster_buy`: True if `distinct_buyer_count >= 2` within a 30-day window. (A cluster of multiple distinct insiders committing personal capital is a recognized corporate governance signal).
- `c_suite_buying`: True if at least one buyer has `officer_title` matching CEO, CFO, or President.

### Transparent Conviction Ranking Formula
A transparent, linear conviction rank (0 to 100) is calculated without hidden coefficients:
```text
BASE_SCORE = 0

IF gross_purchase_value_usd >= 1,000,000:
    BASE_SCORE += 40
ELIF gross_purchase_value_usd >= 250,000:
    BASE_SCORE += 30
ELIF gross_purchase_value_usd >= 50,000:
    BASE_SCORE += 20
ELSE:
    BASE_SCORE += 10

IF cluster_buy (distinct_buyer_count >= 2):
    BASE_SCORE += 30
ELIF distinct_buyer_count == 1:
    BASE_SCORE += 15

IF c_suite_buying:
    BASE_SCORE += 20
ELIF any_officer_or_director:
    BASE_SCORE += 10

IF most_recent_purchase_days_ago <= 7:
    BASE_SCORE += 10
ELIF most_recent_purchase_days_ago <= 14:
    BASE_SCORE += 5

CONVICTION_RANK = MIN(100, BASE_SCORE)
```

Every component of `CONVICTION_RANK` is directly attributable in the returned evidence envelope (e.g. `reasons: ["CEO Open-Market Purchase ($425k)", "Cluster Buy: 2 Insiders in 14 Days"]`).

---

## 6. Prohibited Representations

The UI and API must NEVER claim:
1. "Smart money is buying" (unless citing specific verified transactions).
2. "Profitable signal" or "Guaranteed alpha" (insiders frequently buy into declining secular trends).
3. "Real-time insider trades" (statutory delay is up to 2 business days).
