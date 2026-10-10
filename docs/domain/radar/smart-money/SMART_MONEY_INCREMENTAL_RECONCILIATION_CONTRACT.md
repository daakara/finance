# ARX TERMINAL — SMART MONEY WAVE 2C
## INCREMENTAL VS FULL REBUILD ORACLE RECONCILIATION CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-RECON-001
VERSION: 2.1.0
STATUS: RATIFIED
PARITY_TOLERANCE: 0_CONTENT_DELTA
ORACLE_STATUS: EXACT_MATCH
```

---

### 1. Oracle Parity Invariant

$$\Delta(\text{DB}_{\text{incremental}}, \text{DB}_{\text{full\_rebuild}}) \equiv 0$$

An incremental ingestion database updated across $N$ sequential polling cycles must produce an identical point-in-time output state as a full clean batch rebuild from raw authoritative SEC sources as of the same knowledge time $T_{\text{as\_of}}$.

---

### 2. Reconciliation Audit Procedure

The method `SmartMoneyIncrementalEngine.reconcile_incremental_vs_full_rebuild(as_of_time)` performs an automated mathematical proof of parity:

1. Clones an isolated temporary SQLite database in-memory/scratch.
2. Initializes Schema 2.1.0 tables.
3. Ingests all raw filings with `public_availability_at <= as_of_time` into the clean temporary database using standard parsing logic.
4. Queries both databases for transactions as of $T_{\text{as\_of}}$.
5. Computes set symmetric difference:
   - Records in incremental but missing in full rebuild: MUST BE 0.
   - Records in full rebuild but missing in incremental: MUST BE 0.
6. Compares content hashes of all matching record keys:
   - Hash mismatches: MUST BE 0.
7. Emits verification result: `{"status": "EXACT_PARITY_VERIFIED", "total_transactions_compared": N, "discrepancies": 0}`.
