# ARX TERMINAL — SPRINT 2 PACKAGE C
# PUBLICATION FOUNDATION: IMPLEMENTATION VS LONG-TERM DESIGN

**Document Identifier:** `C_PUBLICATION_IMPLEMENTATION_VS_DESIGN.md`  
**Assessment Timestamp:** `2026-10-21T09:25:00Z` (`2026-10-21T11:25:00 Europe/Berlin, UTC+2`)  
**Governing Authority:** ARX Closure Model v2 (*Decision-First, Minimum Sufficient Evidence*)  
**Target Codebase:** `analyst_dashboard/universe/isolated_publication.py`  
**Formal Architectural Disposition:** `IMPLEMENTATION_BOUNDARIES_CLARIFIED_AND_CERTIFIED`

---

## 1. Executive Summary & Purpose

To prevent semantic drift, over-engineering, and false production assumptions, this document establishes clear, unambiguous boundaries between the **current Track C foundation implementation** and the **long-term production architecture design**.

Every implemented feature is scoped to the minimum sufficient contract necessary to guarantee determinism, auditability, and fail-closed safety without introducing speculative runtime overhead.

---

## 2. Core Architectural Distinctions

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CURRENT IMPLEMENTATION VS LONG-TERM DESIGN                                │
├──────────────────────────┬─────────────────────────────────────┬───────────────────────────────────────┤
│ Dimension                │ Current Track C Implementation      │ Long-Term Production Design           │
├──────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
│ 1. Publication Content   │ Pure Deterministic SHA-256 Digest   │ Hierarchical Cryptographic Structures │
│    Digest Architecture   │ Canonical sort -> Compact JSON      │ (e.g. Merkle Trees, Patricia Tries).  │
│                          │ -> SHA-256 (64 hex characters).     │ NOT IMPLEMENTED. Pure SHA-256 used.   │
├──────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
│ 2. Publication Pointer & │ In-Memory Pointer Swap & Rollback   │ Durable Production Database Rollback  │
│    Rollback Mechanism    │ History stack in engine memory.     │ (SQLite Online WAL Backup API         │
│                          │ Verified via `rollback_publication`.│ to `.db.pre_b_pilot.bak`).            │
├──────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
│ 3. Universe Population   │ Local Rehearsal Test Population     │ Active Railway Production Universe    │
│    Boundaries            │ 4 pilot candidates (AAPL, GOOG,     │ 3 established pilots (AAPL, GOOG,     │
│                          │ GOOGL, ABBV) + adversarial fixtures.│ GOOGL). ABBV remains unpublished.    │
├──────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
│ 4. Market Data Storage   │ In-Memory Disposable Projections    │ Persistent SQLite Storage             │
│    State                 │ Zero disk writes, zero DB locks.    │ `/root/.finance_market_store.db`      │
│                          │ Replaces provisional mock fixtures. │ (5.34 MB, unmutated on HOLD).         │
├──────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
│ 5. Operational Scanner   │ Dormant / Gated by G-S2-ACT         │ Live Celery / Cron Background Loops   │
│    Execution             │ Zero background worker tasks.       │ `VCP_NATURAL_SCHEDULER_ENABLED='1'`   │
│                          │ State 7 transition strictly gated.  │ (Currently locked at '0').            │
└──────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 3. Deep-Dive: Implementation Boundaries

### 3.1 Content Digest: Pure SHA-256 vs Merkle Trees
- **What is Implemented:** `canonical_json_digest()` sorts dictionary keys and candidate records deterministically, serializes to UTF-8 without whitespace, and applies standard SHA-256:
  ```python
  def canonical_json_digest(obj: Any) -> str:
      serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
      return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
  ```
- **What is NOT Implemented:** No binary Merkle trees, proof trees, or leaf-level inclusion cryptographic branches.
- **Why Pure SHA-256 is Sufficient:** For universe sizes of 4 to 150 securities, a single canonical SHA-256 digest provides 100% order-invariant determinism, collision resistance ($2^{-256}$), and zero dependency overhead.

### 3.2 Rollback Architecture: In-Memory Pointer vs Durable WAL Recovery
- **What is Implemented:** `IsolatedPublicationEngine` maintains `_publication_pointer` and `_historical_pointers: List[UniversePublicationPointer]`. Calling `rollback_publication(reason)` pops the immediate prior pointer, increments the build sequence, tags `is_rollback=True`, and records the audit reason.
- **What is NOT Implemented:** Track C does not manipulate production database backup files or issue SQLite WAL checkpoint commands.
- **Why Separation is Mandatory:** Durable disk backup/recovery is owned by Track B3 (`scripts/ingest_alpaca_historical.py` SQLite Online Backup API) and the Railway production deployment operator. Conflating application-layer pointer rollback with filesystem-level database restoration would create dangerous operational ambiguity.

### 3.3 Population Isolation: 4 Local Pilot Candidates vs 3 Production Pilots
- **What is Implemented:** The local rehearsal environment tests 4 pilot candidates (`AAPL`, `GOOG`, `GOOGL`, `ABBV`) derived from the Package B V3 manifest, alongside adversarial test cases (unprovenanced, mismatched MIC, contradictory strategy types).
- **Production Truth:** Active Railway production dashboard currently serves exactly **3 pilot securities** (`AAPL`, `GOOG`, `GOOGL`). Newly adopted `ABBV` (`ARX_SEC_BFA51AC4C0B5`) has 0 market bars in production storage and is NOT published to the active production universe.
- **Firewall Verification:** Local rehearsal evidence is strictly compartmentalized under `LOCAL_INTEGRATION` and `DISPOSABLE_END_TO_END`. No test artifact can be interpreted as production publication authority.

---

## 4. Production Release Gate Invariant Affirmations

In strict accordance with ARX Closure Model v2:
1. **Zero Production Mutation:** Zero writes were executed against production databases (`/root/.finance_market_store.db` and `/root/analyst_dashboard/data/identity.db`).
2. **Zero Ingestion Execution:** No historical SIP data was pulled into production; Package B ingestion remains in status `PROPOSED_NOT_AUTHORIZED`.
3. **Zero Scanner Activation:** Background schedulers remain dormant (`VCP_NATURAL_SCHEDULER_ENABLED='0'`).
4. **VCP Gate 12 Unbroken:** Deferred `G-VCP-012` status remains locked.
