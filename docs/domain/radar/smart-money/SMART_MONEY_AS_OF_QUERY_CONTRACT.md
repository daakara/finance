# SMART MONEY AS-OF QUERY CONTRACT
**Wave 2B — Point-in-Time Query Semantics & Zero Look-Ahead Invariants**
**Document ID**: `SM-DOC-WAVE2B-AS-OF-QUERY-CONTRACT`
**Authority**: Independent Smart Money Historical Data Authority

---

## 1. Executive Summary & Purpose

This contract defines the mathematical rules, SQL implementation, and lineage guarantees for point-in-time ("as-of") historical queries in the Smart Money engine. It provides formal guarantees against look-ahead bias and details how filing amendments (Form 4/A) are reconciled historically.

---

## 2. The Core Knowledge-Time Invariant

### Mathematical Invariant:
$$\text{DECISION\_CAN\_USE\_RECORD\_AT\_T} \iff \text{public\_availability\_at} \le T$$

A quantitative model or decision system evaluating market state at time $T$ may **only** observe records whose SEC public availability timestamp satisfies $\text{public\_availability\_at} \le T$.

### Look-Ahead Leakage Case Study:
Consider an insider open-market purchase:
- Transaction Date: `2026-09-10`
- SEC EDGAR Acceptance: `2026-09-12T18:30:00Z`
- Query Time $T$: `2026-09-11T23:59:59Z`

If an algorithm filters by `transaction_date <= 2026-09-11`, the trade would appear in the backtest on September 11. However, the trade was not publicly known until September 12 at 18:30 UTC. Evaluating this trade on September 11 represents a **fatal look-ahead bias leak**.

Under this contract:
$$\text{public\_availability\_at} (2026\text{-}09\text{-}12\text{T}18:30:00\text{Z}) > T (2026\text{-}09\text{-}11\text{T}23:59:59\text{Z}) \implies \text{EXCLUDED}$$

The trade is strictly invisible at $T$.

---

## 3. Query Implementation (`get_transactions_as_of`)

The point-in-time query is implemented in `Form4IngestionManager` and `SmartMoneyBackfillEngine`:

```python
def get_transactions_as_of(
    self,
    as_of_time: str,
    ticker: Optional[str] = None,
    cik: Optional[str] = None,
    limit: int = 100
) -> List[SmartMoneyTransaction]:
    """Retrieve transactions publicly known as of a specific point in time.
    
    Guarantees:
    - Filters strictly on `public_availability_at <= as_of_time`.
    - Excludes filings made public after as_of_time regardless of transaction_date.
    - Resolves amendments: if an amendment was accepted AFTER as_of_time, the original
      uncorrected filing is returned as it was known at time T.
    """
```

### SQL Boundary Definition:
```sql
SELECT t.tx_id, t.accession_number, t.cik, t.ticker, t.issuer_name,
       t.reporting_owner_name, t.reporting_owner_cik, t.reporting_owner_relationship,
       t.transaction_date, t.security_title, t.transaction_type, t.shares,
       t.price_per_share, t.total_value, t.shares_owned_following,
       t.is_open_market_purchase_evidence, t.is_rule_10b5_1,
       l.acceptance_datetime, l.public_availability_at, l.first_fetched_at,
       t.normalized_at
FROM form4_transactions t
JOIN sm_source_availability_ledger l ON t.accession_number = l.accession_number
WHERE l.public_availability_at <= :as_of_time
  AND (:cik IS NULL OR t.cik = :cik)
  AND (:ticker IS NULL OR t.ticker = :ticker)
ORDER BY l.public_availability_at DESC, t.transaction_date DESC
LIMIT :limit;
```

---

## 4. Amendment Lineage & Historical Reconstruction

When an insider files an amended return (Form 4/A) to correct an earlier filing:
1. **Never Overwrite Historical Reality**: The original Form 4 filing is **never** deleted or overwritten in `form4_raw_filings` or `form4_transactions`.
2. **Amendment Registry (`sm_amendments`)**:
   ```sql
   CREATE TABLE IF NOT EXISTS sm_amendments (
       amendment_accession TEXT PRIMARY KEY,
       original_accession TEXT NOT NULL,
       cik TEXT NOT NULL,
       relationship_type TEXT NOT NULL, -- 'AMENDMENT', 'SUPERSEDES'
       amendment_date TEXT NOT NULL,
       reason_description TEXT,
       created_at TEXT NOT NULL
   );
   ```
3. **Temporal Behavior**:
   - For queries with $T < \text{amendment.public\_availability\_at}$: The system returns the original uncorrected filing, exactly as the market saw it at that time.
   - For queries with $T \ge \text{amendment.public\_availability\_at}$: The system reflects the amended transaction state.

---

## 5. Verification Test Invariants

The as-of query implementation is verified by the following test assertions:
- `test_as_of_query_excludes_future_filings`: A filing with `transaction_date = '2026-09-01'` and `acceptance_datetime = '2026-09-03T16:00:00Z'` queried as of `'2026-09-02T23:59:59Z'` returns **0 rows**.
- `test_as_of_query_includes_past_filings`: The same filing queried as of `'2026-09-03T17:00:00Z'` returns **1 row**.
- `test_deterministic_replay_produces_identical_manifest`: Querying as-of $T$ across multiple engine sessions produces bitwise identical transaction lists.
