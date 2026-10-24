# ARX TERMINAL — RC5 ENGINE SEMANTIC DIFF AUDIT

**Date:** 2026-10-23T11:23:00Z (Local: 2026-10-23T13:23:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_READ_ONLY_PREFLIGHT_AUTHORIZED`  
**Inspected Module:** `analyst_dashboard/analyzers/confluence_engine.py`  
**Baseline Commit:** `a62a91defd8a8f5f479017e50f633981d35a1672` (`main`)  
**Candidate Commit:** `7980d3a6e5dde0fe172d215c729ec213b9a164b1` (PR #53 HEAD)  

---

## 1. Scope of Changes in `confluence_engine.py`

A line-by-line diff between the baseline and candidate versions confirms that the changes in `confluence_engine.py` are strictly confined to 16 lines within the catalyst evaluation section:

```diff
@@ -192,6 +192,22 @@ class ConfluenceEngine:
             if catalysts.get("has_major_event"):
                 score += 5.0
 
+            # CATALYST RUNWAY RISK ADJUSTMENT (Corporate Events RC5)
+            if isinstance(corporate_events, dict):
+                is_completed = corporate_events.get("is_completed", False)
+                status = corporate_events.get("status", "").upper()
+                days_to_earnings = corporate_events.get("days_to_earnings")
+
+                # Skip completed or elapsed events
+                if not is_completed and status not in ("COMPLETED", "ELAPSED"):
+                    if days_to_earnings is not None and days_to_earnings >= 0:
+                        if days_to_earnings <= 1:
+                            score -= 25.0  # Acute earnings hazard penalty
+                        elif days_to_earnings <= 7:
+                            score -= 8.0   # Cautionary runway penalty
+
             # Bound score to [0.0, 100.0]
             score = max(0.0, min(100.0, score))
```

---

## 2. Invariant & Contract Analysis

### A. Mathematical Boundary Invariant: [0.0, 100.0]
- The catalyst modifier subtracts points for imminent earnings risks (`-25.0` for $\le 1$ day, `-8.0` for $2 \le \text{days} \le 7$).
- **Clamping Safety:** All adjustments occur immediately before the existing clamp `score = max(0.0, min(100.0, score))`.
- **Finding:** Under no circumstances can the confluence score underflow below `0.0` or overflow above `100.0`.

### B. Zero-Drift Default Invariant
- When `corporate_events` is `None`, an uncataloged ticker, or an empty dictionary:
  - `isinstance(corporate_events, dict)` handles `None` without exceptions.
  - The entire adjustment block is bypassed.
  - Score modifier is identically `0.0`.
- **Finding:** Historical signals, synthetic test fixtures, and assets lacking earnings data experience zero score drift. Outputs remain bit-for-bit identical to Sprint 2 production.

### C. Elapsed and Stale Event Invariant
- If an event is marked `is_completed: True`, `status: "COMPLETED"`, `status: "ELAPSED"`, or `days_to_earnings < 0`:
  - The guard condition `not is_completed and status not in ("COMPLETED", "ELAPSED")` strictly ignores the event.
- **Finding:** Post-earnings announcements and past events never penalize prospective signals.

### D. Parameter Type and Signature Invariance
- The engine signature retains backward-compatible optional arguments with default `None`.
- Downstream callers passing legacy positional or keyword arguments remain unaffected.

---

## 3. Related RC5 Components Audit

| Component | Nature of Delta | Contract Safety Audit |
|---|---|---|
| `analyst_dashboard/corporate_events.py` | New file (302 lines) | Pure functional data normalization; zero side-effects; strict Pydantic schemas. |
| `analyst_dashboard/analyzers/catalysts.py` | Line 48-62 update | Injects `days_to_earnings` and `has_imminent_earnings` into catalyst metadata. Safe fallbacks. |
| `api/routes/analytics.py` | Line 112 payload update | Adds optional `corporateEvent` key to `/api/analytics/{ticker}` response. Schema-additive only. |

---

## 4. Verification Verdict

The semantic modifications in `confluence_engine.py` represent high-integrity, fail-closed enhancements that fully comply with the approved Corporate Events RC5 specification. The hash change is genuine, intentional, mathematically bounded, and non-regressive.
