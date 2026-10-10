# ARX Terminal — Smart Money State Composition Contract

**Contract ID:** `SM_STATE_COMPOSITION_V1`  
**Composer Implementation:** `analyst_dashboard/smart_money/state_composition.py`  
**Data Model:** `SmartMoneyCompositeState`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`

---

## 1. Executive Summary & Epistemic Scope Rule

Downstream UI consumers (such as Radar and Terminal) require an unambiguous, scope-bound representation of Smart Money state. Ambiguous global booleans (e.g. `"is_available": true` or `"fresh": true`) are strictly prohibited because they mislead users into believing that a governed prototype over 106 stocks represents authoritative market-wide coverage of 4,000+ US equities.

---

## 2. The Formal State 7-Tuple

Every product API payload emitting Smart Money state must serialize the formal 7-tuple:

$$(\mathcal{S}_{\text{scope}}, \mathcal{P}_{\text{pipeline}}, \mathcal{C}_{\text{completeness}}, \mathcal{F}_{\text{freshness}}, \mathcal{G}_{\text{gap}}, \mathcal{R}_{\text{scan\_result}}, \mathcal{D}_{\text{decision\_eligibility}})$$

### Member Vector Definitions

1. **$\mathcal{S}_{\text{scope}}$ (Scope Kind)**:
   - `GOVERNED_SUBSET` (106 common equities in `UB_SM_20261010_R1_19ef857fec1f`).
   - `MARKET_WIDE_AUTHORITATIVE` (Full US equity market, not yet established).

2. **$\mathcal{P}_{\text{pipeline}}$ (Pipeline State)**:
   - `NOT_INITIALIZED`
   - `BOOTSTRAPPING`
   - `OPERATIONAL`
   - `DEGRADED`
   - `FAILED`

3. **$\mathcal{C}_{\text{completeness}}$ (Completeness State)**:
   - `UNKNOWN`
   - `CERTIFIED`
   - `BLOCKED_BY_GAPS`
   - `INTEGRITY_BREACH`

4. **$\mathcal{F}_{\text{freshness}}$ (Freshness State)**:
   - `FRESH` ($\Delta \tau \le 24\text{h}$)
   - `STALE` ($24\text{h} < \Delta \tau \le 72\text{h}$)
   - `EXPIRED` ($\Delta \tau > 72\text{h}$)
   - `UNKNOWN` (Breach or gap blocks assessment)

5. **$\mathcal{G}_{\text{gap}}$ (Scoped Gap State)**:
   - `NONE` (Zero blocking gaps)
   - `OPEN_RETRYABLE`
   - `OPEN_DISCOVERY`
   - `OPEN_AMENDMENT`
   - `OPEN_SOURCE_INTEGRITY`

6. **$\mathcal{R}_{\text{scan\_result}}$ (Scan Result State)**:
   - `NOT_EVALUATED`
   - `MATCHES`
   - `NO_MATCHES`

7. **$\mathcal{D}_{\text{decision\_eligibility}}$ (Decision Eligibility State)**:
   - `ELIGIBLE_WITHIN_SCOPE` (Only when pipeline operational, certified complete, fresh, gap-free, within governed 106).
   - `NOT_ELIGIBLE` (All other conditions).

---

## 3. The Market-Wide Readiness Theorem

$$\text{Theorem: If } \mathcal{S}_{\text{scope}} \equiv \texttt{GOVERNED\_SUBSET}, \text{ then } \text{MarketWideCapabilityState} \equiv \texttt{NOT\_AUTHORIZED}.$$

*Proof:*  
The governed scannable set contains exactly 106 common-equity issuers out of several thousand active SEC reporting companies. Proving 100% mathematical discovery completeness over 106 issuers does not establish completeness over non-governed issuers. Therefore, claiming market-wide availability while operating under `GOVERNED_SUBSET` constitutes a severe epistemic integrity violation.

---

## 4. Prohibition on Global Ambiguous Booleans

Any API endpoint, serializer, or frontend adapter that emits unadorned global booleans:
- `is_available`
- `available`
- `is_fresh`
- `fresh`
- `ready`
- `no_matches`

strictly fails runtime validation via `SmartMoneyStateComposer.reject_ambiguous_global_booleans()`. All capability claims must be explicitly scope-qualified:
- `scoped_capability`: `AVAILABLE_WITHIN_SCOPE`
- `market_wide_capability`: `NOT_AUTHORIZED`
