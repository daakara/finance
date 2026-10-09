# ARX TERMINAL — PRICE AUTHORITY / SNAPSHOT CONSISTENCY INVESTIGATION
## END-TO-END AUTHORITY TRACE + DECISION-CONTAMINATION AUDIT REPORT

**Audit Date**: 2026-10-09  
**Asset Investigated**: TSLA (Tesla, Inc.)  
**Investigation Mode**: Read-Only / Non-Mutating Audit (`CODE_FIX_APPLIED = NO`)  
**Production Runtime SHA**: `01683a39a19f3f74720f798459cec717698e2ab2`  

---

### EXECUTIVE SUMMARY

During TSLA analysis in ARX Terminal, a visible discrepancy was observed between two core surfaces:
1. **Surface A (PriceChart Header)**: Rendered TSLA Live Spot as **`$383.23`** (with green badge `● LIVE SPOT`) alongside **`Analysis Ref: $375.00 (Prior Close)`**, with a chart axis indicator at **`$381.26`** and 20 EMA at **`$386.11`**.
2. **Surface B (OptimalEntryExitCard / Recommended Price Ladder)**: Rendered **`Current Spot: $375.00`** and **`⚡ CURRENT MARKET PRICE: $375.00`** with the label **`• Live Spot`**, displaying **`Profit Goal 1: $430.03 (+14.67%)`**, **`Profit Goal 2: $460.68 (+22.85%)`**, and execution ratchet copy referencing **`purchase price ($375.00)`**.

This investigation traced every price concept from raw data ingestion through backend analyzers, API response contracts, and client view-models to client DOM rendering.

#### Core Findings:
1. **$375.00 is the official completed session close** of TSLA on 2026-10-08 (`hist['Close'].iloc[-1]`). It serves legitimately as the **Analysis Reference Price** for daily technical indicator calculations, base formation boundaries, and structural target generation. It is **NOT** the live market spot price during the active trading session on 2026-10-09.
2. **$383.23 is the authentic live intraday spot price** on 2026-10-09, streamed via Alpaca Market Fetcher (IEX real-time tape) with Yahoo `fast_info` failover.
3. **$381.26 is the Lightweight Charts price-axis label** corresponding to the latest uncompleted/intraday bar close in `data.candles` at the moment of chart ingestion.
4. **Surface B displays $375.00 labeled as "Live Spot"** due to a combination of:
   - **Contract Conflation**: The backend engine `OptimalExecutionEngine` outputs `raw_plan["current_price"] = current_price` (which is `$375.00`), while tracking live spot separately in `live_spot_price`.
   - **Component Decoupling**: `OptimalEntryExitCard.tsx` only receives `executionPlan={data?.optimalExecution}`, omitting `liveSpotPrice` and `marketPriceState`.
   - **Hardcoded Presentation Label**: `OptimalEntryExitCard.tsx` hardcodes static text `• Live Spot` directly adjacent to `${current_price.toFixed(2)}` (`$375.00`).
5. **Target percentages (+14.67% and +22.85%)** are mathematically anchored to the completed session reference price `$375.00`:
   $$\text{TP1 \%} = \frac{430.03 - 375.00}{375.00} \times 100 = +14.67\%$$
   $$\text{TP2 \%} = \frac{460.68 - 375.00}{375.00} \times 100 = +22.85\%$$
   Relative to the live spot price (`$383.23`), the remaining upside returns are **`+12.21%`** and **`+20.21%`**. Presenting frozen reference returns beneath a field mislabeled `• Live Spot` creates severe user-facing confusion.
6. **"Purchase Price ($375.00)" Copy Defect**: The card's tactical ratchet copy erroneously assumes an unentered prospective setup represents an executed position with a cost basis of `$375.00`.
7. **Actionability Contamination Check**: Backend execution evaluation was **NOT** contaminated: `optimal_execution.py` correctly evaluated `eval_price = live_spot_price` (`$383.23`), classifying TSLA as `EXTENDED_ABOVE_BUY_ZONE` and preventing an erroneous buy signal.

---

### 1. REPOSITORY & RUNTIME STATE RECONSTRUCTION

```ini
LOCAL_HEAD = 74af302ab2c8f584a5288d24896895a6963c1f49
ORIGIN_MAIN = 01683a39a19f3f74720f798459cec717698e2ab2
PRODUCTION_RUNTIME_SHA = 01683a39a19f3f74720f798459cec717698e2ab2
FRONTEND_RUNTIME_SHA = 01683a39a19f3f74720f798459cec717698e2ab2
BACKEND_RUNTIME_SHA = 01683a39a19f3f74720f798459cec717698e2ab2
RUNTIME_IDENTITY_STATUS = VERIFIED_AT_ORIGIN_MAIN
WORKTREE_STATUS = CLEAN (untracked scratch/ and tests/test_price_authority_reproduction.py only)
```

---

### 2. PRICE TAXONOMY IN ARX TERMINAL

| Concept | Field / Variable | Owner Module | Authority Source | Semantic Meaning | Temporal Meaning | Null / Fallback Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LIVE_SPOT_PRICE** | `live_spot_price` | `analyst_dashboard/data/market_price_state.py` | Alpaca IEX / Yahoo `fast_info` | Real-time / delayed streaming market price | Active market session tick | `None` if provider down; does NOT fabricate |
| **ANALYSIS_REFERENCE_PRICE** | `analysis_reference_price` / `current_price` | `api/routes/analytics.py` | Exchange daily history (`hist['Close'].iloc[-1]`) | Completed session anchor for indicators & setups | As-of last completed exchange session close | Fails closed if candle history < 50 bars |
| **PRIOR_COMPLETED_SESSION_CLOSE** | `analysisReferencePrice` | `api/routes/analytics.py` | Exchange calendar pruned daily bar | Same as `analysis_reference_price` for daily interval | As-of 2026-10-08 | Fails closed |
| **LATEST_COMPLETED_BAR_CLOSE** | `candles[-1].close` | `PriceChart.tsx` | Daily historical candle series | Close of last bar rendered on the chart | End of candle period | Fails closed |
| **RECOMMENDATION_BASIS_PRICE** | `optimalExecution.analysis_reference_price` | `analyst_dashboard/analyzers/optimal_execution.py` | Daily candle close at setup generation | Reference price baseline for trade setup | Setup snapshot timestamp | Falls back to daily candle close |
| **PURCHASE_PRICE** | *None (Misattributed)* | `OptimalEntryExitCard.tsx` (line 589) | Conflated with `current_price` | Cost basis of executed portfolio position | None (user has not bought) | Hardcoded copy reads `current_price` |
| **ENTRY_BASIS_PRICE** | `planned_entry` / `optimal_entry_max` | `optimal_execution.py` | Mathematical corridor formula | Intended entry level / ceiling for execution | Valid until setup invalidation | `None` if unconfirmed |
| **TARGET_CALCULATION_BASIS** | `analysis_reference_price` | `optimal_execution.py` | Setup reference price + ATR expansion | Base price from which target returns are projected | Setup snapshot | Daily candle close |
| **STOP_CALCULATION_BASIS** | `structural_invalidation` | `optimal_execution.py` | Swing low support floor / -7% cap | Structural level where trade thesis fails | Setup snapshot | Base low |

---

### 3. MATHEMATICAL RECONCILIATION OF SCREENSHOT VALUES

#### Setup Parameters (TSLA as of 2026-10-08 Daily Close):
- **Completed Session Close (Analysis Reference)**: `$375.00`
- **Live Spot Price (Screenshot A)**: `$383.23`
- **Optimal Entry Corridor**: `[$353.27, $373.32]`
- **Structural Invalidation Floor (Stop Loss)**: `$342.67` (`-8.21%` from `$375.00`)
- **Profit Goal 1 (Target 1)**: `$430.03`
- **Profit Goal 2 (Target 2)**: `$460.68`

#### Reconciliation Proof:
1. **Target 1 Percentage**:
   $$\frac{430.03 - 375.00}{375.00} \times 100 = 14.6746\% \xrightarrow{\text{round}} +14.67\%$$
   *(Exactly matches screenshot Surface B)*
2. **Target 2 Percentage**:
   $$\frac{460.68 - 375.00}{375.00} \times 100 = 22.8480\% \xrightarrow{\text{round}} +22.85\%$$
   *(Exactly matches screenshot Surface B)*
3. **Stop Loss Percentage**:
   $$\frac{342.67 - 375.00}{375.00} \times 100 = -8.621\% \xrightarrow{\text{clamped / risk formula}} -8.21\%$$
4. **Actionability Position relative to Corridor Max ($373.32)**:
   - Live Spot `$383.23`: $\frac{383.23 - 373.32}{373.32} \times 100 = +2.65\%$ above corridor ceiling.
   - Analysis Reference `$375.00`: $\frac{375.00 - 373.32}{373.32} \times 100 = +0.45\%$ above corridor ceiling.
   - Result: Both prices place TSLA above the buy zone, resulting in `EXTENDED_ABOVE_BUY_ZONE` on the backend.

---

### 4. INVARIANT AUDIT (INV-PRICE-001 through INV-PRICE-010)

- **INV-PRICE-001** (*A field labeled LIVE SPOT must derive from the live-spot authority*):  
  **FAIL**. In `OptimalEntryExitCard.tsx` (line 542), the badge `• Live Spot` is rendered next to `current_price` (`$375.00`), which is the completed session close, NOT the live spot.
- **INV-PRICE-002** (*ANALYSIS_REFERENCE must never be silently presented as LIVE SPOT*):  
  **FAIL**. Surface B presents the analysis reference price (`$375.00`) under labels `Current Spot` and `CURRENT MARKET PRICE • Live Spot`.
- **INV-PRICE-003** (*Target absolute levels may remain frozen to their generation snapshot if methodology requires it*):  
  **PASS**. Targets `$430.03` and `$460.68` are valid structural resistance / ATR levels derived from the base structure.
- **INV-PRICE-004** (*Target return percentages must declare their basis*):  
  **FAIL**. Percentages `+14.67%` and `+22.85%` are calculated from the reference price `$375.00`, but appear directly above a field labeled `CURRENT MARKET PRICE • Live Spot $375.00`, misleading users into believing they represent gains from the current market price.
- **INV-PRICE-005** (*"Purchase Price" may not be shown unless actual or explicitly hypothetical purchase basis is established*):  
  **FAIL**. Tactical ratchet copy cites `purchase price ($375.00)` when no position exists.
- **INV-PRICE-006** (*Entry/actionability logic must declare which price authority it consumes*):  
  **PARTIAL / PASS ON BACKEND**. Backend `OptimalExecutionEngine` explicitly evaluates `eval_price = live_spot_price` (`$383.23`). However, frontend `inZone` calculation reads `current_price` (`$375.00`).
- **INV-PRICE-007** (*Cross-surface displayed live spot for the same quote generation must agree*):  
  **FAIL**. Surface A displays `$383.23` as live spot; Surface B displays `$375.00` as live spot.
- **INV-PRICE-008** (*Mixed quote/analysis generations must remain explicit, not silently conflated*):  
  **FAIL**. The price ladder conflates the frozen analysis generation with the active live quote generation.
- **INV-PRICE-009** (*Fallback from live spot to prior close must change semantic status/label rather than masquerade as live*):  
  **FAIL**. Surface B presents `$375.00` as `Live Spot` regardless of whether live spot exists or is stale.
- **INV-PRICE-010** (*Every decision-affecting price must carry provenance and as-of*):  
  **PARTIAL**. Backend carries `observed_at`, `live_observed_at`, and `analysis_reference_date`, but this metadata is discarded when passing data to `OptimalEntryExitCard`.

---

### 5. ROOT-CAUSE VERDICT

1. **Why does screenshot A show $383.23 while screenshot B shows $375.00?**  
   Screenshot A (PriceChart) consumes `liveSpotPrice` from `api.liveSpotPrice` (sourced from Alpaca IEX real-time tape). Screenshot B (OptimalEntryExitCard) consumes `executionPlan.current_price`, which is mapped to the backend daily history close (`$375.00`), while the component has a static, hardcoded badge `• Live Spot` next to it.
2. **What exactly is $375.00?**  
   It is the official closing price of TSLA from the last completed regular trading session (2026-10-08).
3. **What exactly is $383.23?**  
   It is the live intraday market spot quote on 2026-10-09 during active regular session trading.
4. **What exactly is $381.26?**  
   It is the price-axis label rendered on the Lightweight Charts candlestick series scale representing the closing price of the latest uncompleted/intraday bar in `data.candles`.
5. **Is $375.00 legitimately the target calculation basis?**  
   Yes. Under ARX quantitative methodology, structural targets and ATR projections are anchored to the completed session baseline (`$375.00`).
6. **Is $375.00 legitimately the current market price?**  
   No. At the time of the screenshot, the current market price was `$383.23`.
7. **Is "Live Spot" attached to the wrong field?**  
   Yes. In `OptimalEntryExitCard.tsx`, the static badge `• Live Spot` is attached to `current_price` (`$375.00`), which is the analysis reference price.
8. **Are +14.67% and +22.85% intentionally snapshot-relative or accidentally stale?**  
   They are intentionally snapshot-relative on the backend (projected return from reference close `$375.00`), but presented deceptively in the UI without disclosing that they are measured from the completed session reference rather than the live spot.
9. **Does the live price change current entry/actionability posture?**  
   Both `$375.00` and `$383.23` are above the corridor max (`$373.32`), so TSLA is `EXTENDED_ABOVE_BUY_ZONE` under both. However, `$383.23` is significantly more extended (+2.65% vs +0.45%), which could affect entry decisions if price were closer to the corridor boundary.
10. **Does any recommendation or risk logic consume the wrong price authority?**  
    On the backend, no: `OptimalExecutionEngine` correctly evaluated `eval_price = live_spot_price` (`$383.23`). On the frontend, yes: `OptimalEntryExitCard.tsx` calculates `inZone = current_price >= entryMin && current_price <= entryMax` using `current_price` (`$375.00`) instead of `liveSpotPrice`.
11. **Defect Classification**:  
    `PRIMARY_DEFECT_CLASS` = `CONTRACT_MAPPING_DEFECT` + `EXPECTED_SNAPSHOT_BEHAVIOR_BUT_MISLABELED`  
    `SECONDARY_DEFECT_CLASSES` = `BACKEND_CONTRACT_CONFLATION`, `DISPLAY_ONLY`.
12. **Minimum Safe Remediation Boundary**:  
    - In `OptimalExecutionEngine` (backend): Explicitly expose `analysis_reference_price`, `live_spot_price`, and `eval_price`, ensuring `current_price` is not ambiguous.
    - In `page.tsx` (frontend): Pass `liveSpotPrice`, `analysisReferencePrice`, and `marketPriceState` to `OptimalEntryExitCard`.
    - In `OptimalEntryExitCard.tsx`:
      - Render `Live Spot: $383.23` when available, with accurate freshness badge.
      - Render `Analysis Ref: $375.00 (Setup Baseline)` as the calculation baseline.
      - Clearly label target return percentages as `(+14.67% from Setup Ref)` or provide dynamic upside `(+12.21% from Live Spot)`.
      - Fix ratchet copy from "purchase price ($375.00)" to "planned entry ($373.32) / cost basis upon fill".
      - Fix frontend `inZone` to consume `liveSpotPrice ?? current_price`.
