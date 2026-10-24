# ARX Terminal — Bounded Wave-Fragmentation Research Specification

**Document Identifier**: `ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION`  
**Workstream Identifier**: `ARX_VCP_RESEARCH_WAVE_FRAGMENTATION_01` (Nonproduction Research Proposal)  
**Revision**: `2.3.0` (Contract Freeze with Binding Corrections)  
**Execution Timestamp**: `2026-10-24T01:25:00+02:00`  
**Governing Framework**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Final Approval Authority**: David Akara  
**Final Status**: `ARX_VCP_CANDIDATE_006_CONTRACT_FROZEN_FOR_IMPLEMENTATION`

---

## Document Revision & Chronological Change History

| Revision | Date / Timestamp | Author / Context | Summary of Changes |
| :---: | :---: | :---: | :--- |
| `1.0.0` | 2026-10-24T00:41:00+02:00 | ARX Research Agent | Initial comparative analysis of fixed percentage, volatility-scaled, and structural merging approaches. Defined preliminary 14 synthetic scenarios. |
| `2.0.0` | 2026-10-24T00:50:00+02:00 | ARX Research Agent | Qualified method family comparisons; resolved causal state machine with irreversible confirmation; made 5%/3-session rule executable; formally deferred post-hoc containment folding; fixed strict experiment budget. |
| `2.1.0` | 2026-10-24T00:55:00+02:00 | ARX Research Agent | Removed wave-count-only success claims; adopted Option A (simpler elapsed-session confirmation without 2% floor); unified 5% numerical comparison; specified next-wave peak tracking; added owner ratification table. |
| `2.3.0` | 2026-10-24T01:25:00+02:00 | ARX Research Agent / Owner Authorized | **Contract Freeze with Binding Corrections**: (1) Preserved strict tightening $D_{\text{next}} \le 0.95 \times D_{\text{prev}}$ (removed $D_{\text{next}} \le D_{\text{prev}} + 0.01$); (2) Preserved initial depth floor $D_1 \ge 0.08$; (3) Faithful expansion extraction (segmentation records genuine expansion, predicate layer evaluates and rejects under tightening rule); (4) Accurate elapsed time description ($j - t_{\text{trough}} \ge 3$ elapsed eligible sessions post-trough; no 2% support floor); (5) Numerical guard defined as an explicit narrow acceptance band; (6) Authoritative ratio comparisons with nonfinite/nonpositive rejection; (7) Same-bar non-ordering policy strictly applied to outside bars (peak update precedes rollover; rollover requires subsequent bar). |
| `2.2.0` | 2026-10-24T01:05:00+02:00 | ARX Research Agent | **Contract Freeze & Causal Interval Finalization**: (1) Corrected next-wave candidate peak eligibility to strictly post-trough bars $[t_{\text{trough}} + 1, t_{\text{conf}}]$; (2) Specified complete causally eligible trough interval $[t_{\text{peak}} + 1, j_{\text{roll}}]$ with same-bar non-ordering policy; (3) Added 7 synthetic timelines with exact bar indices, prices, and state transitions; (4) Formalized fixed-start prefix consistency for confirmed records; (5) Qualified historical expansion expectations, removing circular ground-truth assumptions; (6) Disaggregated control accounting (38 negatives, 7 abstentions); (7) Accurately defined numerical guards; (8) Complete frozen parameter inventory. |

---

## 1. Executive Summary, Recommendation & Owner Decisions

### 1.1 Objective & Context
Candidate 005 demonstrated deterministic base-origin anchoring, pre-base truncation, and qualified `HVA00-POS-11 / LLY` (2 waves: $12.0\% \to 4.8\%$, volume ratio $0.4511$). However, on volatile leaders such as `NVDA` (`POS-01`, `POS-02`), `FTNT` (`POS-03`, `POS-04`), and `VRT` (`POS-07`, `POS-08`), Candidate 005 exhibited severe **wave fragmentation**—segmenting macro consolidations into 10 to 22 micro-waves. This caused automated rejection under wave count limits ($> 6$ waves) and falsely indicated progressive expansion where human chartists observe a unified macro contraction.

This specification finalizes the research proposal for resolving wave fragmentation in `Candidate 006`. It defines an exact, deterministic, point-in-time causal state machine that unifies intra-wave downswings while preserving all existing screening predicates and release gates.

### 1.2 Core Recommendation
**Recommend Approach C (Causal State Machine with Undercut Continuation & Structural Rebound Prominence Gate).**

- **Mechanism**: Shift wave segmentation from arbitrary local noise thresholds to topological swing structure. A declining contraction leg remains active as long as intermediate rebounds fail to confirm structural prominence and subsequent prices make lower lows (Undercut Continuation). Once a rebound confirms structural prominence ($\Delta_{\text{rec}} \ge 0.0500 - 10^{-7}$ closing recovery persisting across $\ge 3$ elapsed trading sessions without an undercut), the wave boundary is **irreversibly confirmed**. Any subsequent breakdown below that trough is recognized as a **separate contraction wave** and evaluated for progressive volatility expansion.
- **Confirmation Contract (Option A)**: Adopt simpler elapsed-session confirmation without an artificial 2% intermediate floor.
- **Architectural Deferrals**: Containment folding ($D_{\text{sub}} < 0.50 \cdot D_{\text{parent}}$), ATR volatility scaling, and fixed-threshold sweeps remain **formally deferred**.
- **Principal Risk**: Over-consolidation. If structural rebound confirmation is set too strict, a true distributed expansion with slow rolling tops could be delayed in recognition. (Safeguards are detailed in Section 5).
- **Proposed Experiment Budget**: Exactly **one candidate variant** (`ARX_VCP_CLASSIFIER_V1_CANDIDATE_006`), tested against the 60 exposed development cases across a maximum of **two evaluation cycles** (with the second run strictly reserved for documented harness corrections). Zero access to protected holdout Epoch 002.

### 1.3 Exact Decisions Required from Product Owner (David Akara)
1. **Ratification of Approach C (Causal State Machine)**: Ratify the Causal State Machine with Undercut Continuation and Structural Rebound Prominence as the sole authorized research direction, confirming the deferral of fixed-threshold sweeps (Approach A), ATR scaling (Approach B), and post-hoc containment folding.
2. **Approval of Rebound Confirmation Contract (Option A)**: Ratify Option A (elapsed trading duration $\ge 3$ sessions, no undercut, close recovery $\ge 5.0\%$, with zero intermediate-close support floor).
3. **Approval of Finite Experiment Budget**: Authorize testing of exactly one candidate implementation (`Candidate 006`) against the 60 exposed development cases, with a strict ceiling of two evaluation cycles.
4. **Ratification of Eight-Dimensional Progression Criteria**: Formally ratify that evaluation success requires satisfying the eight-dimensional progression criteria in Section 7. Wave-count reduction on NVDA and FTNT (e.g. into the range 2–5) is recognized solely as an exposed development diagnostic, not an authoritative geometry oracle or sufficient condition for acceptance.
5. **Holdout Custody Deferral**: Confirm that Protected Holdout Epoch 002 remains sealed and unaccessed throughout this research workstream.

---

## 2. Baseline to Preserve & Accepted Development Findings

### 2.1 Preserved Repository Baseline
- **Worktree**: `C:\Users\akara\Documents\Projects\finance-vcp-candidate-005`
- **Branch**: `feat/vcp-candidate-005-implementation`
- **Baseline Commit**: `13d4e0862561ac0ebb486d0ca59bbd5defb14075`
- **Tree SHA**: `40cb84d2952eedef7715defa62559bb20a5d94de`
- **Default Classifier**: Candidate 004 (`ARX_VCP_CLASSIFIER_V1_CANDIDATE_004`) remains the active system default.
- **PR #73 Status**: Production activation remains strictly on **HOLD**.
- **Candidate Identifier Status**: `ARX_VCP_CLASSIFIER_V1_CANDIDATE_006` is verified unassigned across all VCP classifiers and reserved for this proposal.

### 2.2 Accepted Development Findings & Corrected Provenance
1. **Sensitivity**: Candidate 005 qualified **1 of 15** historical positive-label development cases.
2. **Authoritative Qualifying Case**: Sole qualified case is **`HVA00-POS-11 / LLY`** (Eli Lilly & Co) at evaluation cutoff **`2026-01-07T21:00:00Z`**. (This supersedes earlier conversational notes attributing the cutoff to August 2023). Telemetry: 2 waves ($12.0\% \to 4.8\%$), volume dry-up ratio $0.4511 \le 0.70$, Stage 2 confirmed.
3. **`HVA00-POS-03` Identity**: Authoritative manifest proves `POS-03` is **`FTNT`** (Fortinet Inc., cutoff `2026-10-02`), which failed due to 11 fragmented waves. Ticker `META` is not part of the development manifest.
4. **NVDA Dual Rejection**: Under Candidate 004, NVDA compiled 4 waves and failed on volume ($0.8667 > 0.70$). Under Candidate 005, local peak/trough segmentation fractured the consolidation into 10 waves ($> 6$ limit), failing contraction count and monotonic tightening while independently failing volume ($0.9141 > 0.70$). Both causes co-exist.
5. **Control Specificity**: Across 45 control inputs, Candidate 005 produced **38 negative classifications**, **7 insufficient-data abstentions**, and **0 false-positive qualifications** ($0.0\%$ false positive rate).
6. **No Implementation Defect**: Candidate 005 conformed strictly to specification HVA-02 (Rev 2.0.0); wave fracture is an algorithmic limitation of fixed noise thresholds, not a software bug.

---

## 3. Task 1: Qualified Failure Definition & Target Recognition Behavior

### 3.1 Qualified Comparison of the Three Rule Families
The comparison between segmentation families reflects empirical development observations rather than unsupported categorical claims:

```
+------------------------------------------------------------------------------------------------------------------------------------+
|                                                QUALIFIED METHOD FAMILY COMPARISON                                                  |
+------------------------------------+------------------------------------+----------------------------------------------------------+
| Approach A: Fixed Percentage Rules | Approach B: Volatility-Scaled Rules| Approach C: Recommended Structural Hypothesis            |
| (Unchanged Candidate 005 Baseline) | (Deferred Alternative Hypothesis)  | (Causal State Machine with Undercut Continuation)        |
+------------------------------------+------------------------------------+----------------------------------------------------------+
| - Static Delta & Duration Threshold| - ATR-normalized Dynamic Amplitude | - Topological Swing State Machine                        |
| - Observed failure on NVDA/FTNT    | - Causal Self-Inflation Hazard     | - Scale-Invariant & Beta-Robust                          |
| - Conjunction flaw & undercut trap | - 4-5 New Tuned Parameters         | - Irreversible Boundary Confirmation                     |
| - Simple, deterministic baseline   | - High Overfitting Risk            | - Minimal Parameter Burden (Prominence Threshold)        |
| --> STATUS: UNCHANGED BASELINE     | --> STATUS: DEFERRED ALTERNATIVE   | --> STATUS: RECOMMENDED FOR BOUNDED EXPERIMENT           |
+------------------------------------+------------------------------------+----------------------------------------------------------+
```

1. **Approach A (Unchanged Fixed-Threshold Baseline — Candidate 005)**:
   - *Observed Limitations*: In Candidate 005, noise merging is governed by the conjunction $\Delta < 4.0\%$ AND $\tau \le 3$ sessions. On development cases with elevated daily volatility (`NVDA`, `FTNT`, `VRT`), any 4-day lateral pause triggered premature leg closure. When prices subsequently broke down to a lower low, the algorithm treated the drop as a new contraction wave, multiplying wave counts.
   - *Qualification*: This demonstrates a concrete limitation of Candidate 005's specific conjunctive design on the observed high-beta development corpus. It does not prove that all fixed-threshold designs are universally invalid. Approach A remains preserved as an unmutated baseline.
2. **Approach B (Deferred Volatility-Scaled Alternative — ATR Normalization)**:
   - *Mechanism*: $\theta_{\text{amp}}(t) = \text{clamp}(\lambda \cdot \frac{\text{ATR}_{14}(t)}{C_t}, \, \theta_{\min}, \, \theta_{\max})$.
   - *Reason for Deferral*: While volatility normalization can scale thresholds across assets, it introduces 4–5 new hyperparameters ($\lambda, N_{\text{atr}}, \theta_{\min}, \theta_{\max}, \theta_{\text{dur}}$) and creates a causal self-inflation hazard (a violent swing inflates ATR, which inflates the noise threshold, potentially absorbing real distribution as noise). Crucially, scaling amplitude does not solve the topological problem of undercut continuation. It is deferred as an unnecessarily complex alternative.
3. **Approach C (Recommended Structural Hypothesis — Causal State Machine)**:
   - *Mechanism*: Enforces topological swing hierarchy. Intermediate rallies that fail to achieve structural prominence and are subsequently undercut remain unified in the parent downswing. Once a rebound achieves structural prominence, the wave boundary is confirmed irreversibly.
   - *Reason for Recommendation*: Solves the observed failure mode directly with minimal parameter overhead and scale invariance.

### 3.2 Target Recognition State & Invariant Decoupling
The target recognition state is:
$$\mathcal{S}_{\text{target}} = \{\text{Macro Geometry Valid}\} \cap \{\text{Screening Invariants Uncompromised}\}$$

The research objective is **not** to force NVDA or FTNT to qualify as final VCP setups. Under frozen development evidence, NVDA independently failed Volume Dry-Up ($0.9141 > 0.70$).
No independent historical geometry reference exists for the development cases; authoritative wave labels cannot be manufactured from desired outputs. Observations on NVDA and FTNT provide diagnostic evidence of wave-count behavior, but cannot serve as the sole success oracle. Full evaluation requires checking all 15 positive cases and 45 controls against the eight progression dimensions (Section 7).

---

## 4. Architectural Resolution of Containment Folding

### 4.1 Forensic Analysis of Post-Segmentation Containment Folding
Revision 1.0.0 proposed an auxiliary rule: folding adjacent waves post-segmentation if $D_{\text{sub}} < 0.50 \cdot D_{\text{parent}}$ and the subordinate wave was contained within the parent wave's price span.

Rigorous manual reasoning reveals that post-hoc containment folding suffers from fatal structural flaws:
1. **Destruction of Valid Tightening**: In a textbook Minervini VCP pattern, Wave 1 drops 16%, and Wave 2 drops 6%. Because $6\% < 0.50 \times 16\% = 8\%$, a naive containment folding rule would fold Wave 2 into Wave 1! The algorithm would destroy the very tightening contraction it was designed to detect.
2. **Circularity Hazard**: If folding is conditioned on whether depths tighten ($D_k \le D_{k-1} \times 0.95$), segmentation becomes circular with predicate evaluation. Wave boundaries would be determined by whether they make the classifier pass, violating the separation between observation extraction and predicate scoring.
3. **Expansion-Masking Hazard**: In a distributing stock, Wave 1 drops from \$100 to \$80 (20%). Wave 2 rallies to \$95 and drops to \$85 (10.5%). Wave 3 rallies to \$92 and drops to \$81 (12.0% — genuine volatility expansion). Because both Wave 2 and Wave 3 occur within the \$100–\$80 vertical span of Wave 1, vertical containment folding would collapse them, masking the genuine expansion and falsely reporting a clean setup.
4. **Violation of Irreversible Confirmation**: Folding confirmed waves post-hoc means that a wave boundary declared confirmed at bar $t$ could be erased at bar $t+10$, destroying causal prefix consistency.

### 4.2 Formal Deferral of Containment Folding
**Decision**: Post-segmentation containment folding is **formally deferred and excluded** from Candidate 006.

Candidate 006 relies exclusively on the **Causal State Machine with Undercut Continuation and Rebound Prominence** (Section 5). By ensuring that downswings do not close prematurely during intra-wave chop, the causal state machine eliminates wave fragmentation at the source, rendering post-hoc folding unnecessary.

---

## 5. Causal State Machine Specification & Next-Wave Peak Tracking

### 5.1 Formal States & Timestamps
At any evaluation session $j \in [t_{\text{anchor}}, T_{\text{cutoff}}]$, the segmentation engine operates in one of five deterministic states:

```
[STATE_DOWNSWING] 
       |
       | (Close >= Trough * (1.0 + theta_prom - eps) and no undercut for 1-2 sessions)
       v
[STATE_PROVISIONAL_REBOUND]
       |
       |---+ (Low drops < Trough: UNDERCUT BEFORE CONFIRMATION)
       |   | --> Resets to STATE_DOWNSWING; Trough = new low; Provisional Peak cancelled.
       |
       | (Rebound persists >= 3 elapsed trading sessions without undercut and Close meets threshold)
       v
[STATE_CONFIRMED_REBOUND]  <=== BOUNDARY IRREVERSIBLY CONFIRMED AT t_confirmed
       |
       | (Price rolls over: Close <= Candidate Peak * (1.0 - theta_rollover + eps) on session j > t_peak)
       v
[STATE_NEW_CONTRACTION]
       |
       |---+ (Price makes new lower low: UNDERCUT AFTER CONFIRMATION)
           | --> Recognized as Wave k+1 with trough L_new. Evaluated for expansion!
```

1. **`STATE_DOWNSWING`**: Active downward contraction leg. Trough is continuously updated: $L_{\text{trough}} = \min(L_{\text{trough}}, L_j)$.
2. **`STATE_PROVISIONAL_REBOUND`**: Price close satisfies recovery threshold, but has not yet satisfied the 3-session elapsed duration requirement. Candidate trough remains provisional.
3. **`STATE_CONFIRMED_REBOUND`**: The rebound has satisfied both the recovery threshold and 3-session duration without an undercut. The wave boundary $(P_k, T_k)$ is **permanently confirmed**.
4. **`STATE_NEW_CONTRACTION`**: Price rolls over from confirmed candidate peak $P_{k+1}$ by $\ge 4.0\%$. Wave $k+1$ is actively seeking its own trough.
5. **`STATE_FINAL_CONFIRMATION`**: Cutoff boundary verification. Final wave must be confirmed prior to or on $T_{\text{cutoff}}$.

### 5.2 Timestamp Separation
For every structural turning point, three distinct timestamps are tracked:
- $t_{\text{extremum}}$: The exact bar index where the price low or high physically occurred ($L_{t_{\text{extremum}}} = L_{\text{trough}}$).
- $t_{\text{knowable}}$: The bar index where the recovery first touched $\ge 5.0\%$ above $L_{\text{trough}}$ ($j \ge t_{\text{extremum}} + 1$).
- $t_{\text{confirmed}}$: The bar index where all confirmation conditions are fulfilled ($j \ge t_{\text{extremum}} + 3$).

### 5.3 Next-Wave Peak Eligibility & Complete Trough Interval Rules

#### A. Peak Eligibility Strictly Follows Previous Trough (Correction 1)
- **Problem Resolved**: In daily OHLC, a bar's high and low sequencing is unknown. Reusing the previous trough bar's high ($H_{t_{\text{trough}}}$) risks using a pre-trough price.
- **Rule**: When Wave $k$ is confirmed at session $t_{\text{conf}}$, the candidate peak for Wave $k+1$ ($P_{k+1}$) is initialized strictly using bars **following the trough occurrence bar**:
  $$P_{k+1} = \max_{m \in [t_{\text{trough}, k} + 1, \, t_{\text{conf}}]} H_m, \quad t_{\text{peak}, k+1} = \min \left\{ m \in [t_{\text{trough}, k} + 1, \, t_{\text{conf}}] : H_m = P_{k+1} \right\}$$
- **Chronological Tie-Breaking**: If multiple bars share the maximum high, the **earliest** bar index is retained.
- **Reset Invariant**: If a lower low occurs before confirmation ($L_j < L_{\text{trough}, k}$), the trough resets to $j$, and the previous candidate recovery interval is discarded. The reset trough bar's high is **never** reused as evidence of a subsequent peak.

#### B. Same-Bar Non-Ordering Policy & Rollover Gate (Correction 2)
- **Same-Bar Non-Ordering Policy**:
  1. A daily bar $j$ that observes a higher high ($H_j > P_{k+1}$) updates the provisional peak to $P_{k+1} = H_j$ at $t_{\text{peak}, k+1} = j$.
  2. Because intraday sequencing is unavailable from daily OHLC, bar $j$ **cannot simultaneously establish an ordered decline from its own high**.
  3. A structural rollover must be observed on a **strictly subsequent eligible bar** ($j > t_{\text{peak}, k+1}$).
- **Rollover Condition on Session $j > t_{\text{peak}, k+1}$**:
  $$\Delta_{\text{roll}}(j) = \frac{P_{k+1} - C_j}{P_{k+1}} \ge \theta_{\text{rollover}} - \epsilon_{\text{float}}$$
  where $\theta_{\text{rollover}} = 0.0400$ (4.0000%) and $\epsilon_{\text{float}} = 10^{-7}$.
- **Complete Causally Eligible Trough Interval**:
  When rollover is confirmed at session $j_{\text{roll}} > t_{\text{peak}, k+1}$, the downswing has officially begun. The running trough for Wave $k+1$ is initialized across **every eligible subsequent session** between the peak and rollover:
  $$L_{\text{trough}, k+1} = \min_{m \in [t_{\text{peak}, k+1} + 1, \, j_{\text{roll}}]} L_m, \quad t_{\text{trough}, k+1} = \min \left\{ m \in [t_{\text{peak}, k+1} + 1, \, j_{\text{roll}}] : L_m = L_{\text{trough}, k+1} \right\}$$
  (Peak bar low $L_{t_{\text{peak}}}$ is excluded for identical causal reasons; all evaluated bars are strictly post-peak).
- **Subsequent Downswing Extension**:
  For subsequent sessions $j > j_{\text{roll}}$ in `STATE_NEW_CONTRACTION`:
  $$\text{If } L_j < L_{\text{trough}, k+1} \implies L_{\text{trough}, k+1} = L_j, \quad t_{\text{trough}, k+1} = j$$

#### C. Outside Bar Precedence & Same-Bar Non-Ordering Policy
If an outside bar occurs on session $j$:
- **In `STATE_CONFIRMED_REBOUND` (seeking peak expansion/rollover)**: If $H_j > P_{k+1}$, high expansion takes precedence ($P_{k+1} = H_j, t_{\text{peak}, k+1} = j$). Under the **same-bar non-ordering policy**, a bar establishing or updating a candidate peak cannot also confirm an ordered decline from that peak. The bar's low is disregarded for trough initialization on session $j$, and downswing rollover evaluation requires a subsequent eligible session ($j > t_{\text{peak}}$).
- **In `STATE_PROVISIONAL_REBOUND` or `STATE_DOWNSWING`**: The **undercut takes precedence**, resetting $L_{\text{trough}} = L_j, t_{\text{trough}} = j$ and cancelling provisional recovery.

#### D. Cutoff Semantics
- **Dataset ends during confirmed rebound**: Wave $k$ is confirmed. Wave $k+1$ is uninitiated. Final wave is Wave $k$. If Wave $k$ satisfies final-wave criteria (recovery $\ge 1.5\%$ above trough, pivot proximity), it evaluates downstream.
- **Dataset ends during unconfirmed downswing**: If $T_{\text{cutoff}}$ occurs while in `STATE_DOWNSWING` or `STATE_NEW_CONTRACTION`: the final wave lacks confirmed rebound $\to$ evaluates as `FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF` and fails `PRED_CONTRACTION_SEQUENCE_VALID`.

---

### 5.4 State Transition Matrix

| Current State | Trigger Event at Session $j$ | Guard Condition | Next State | Action / Telemetry Side Effect |
| :--- | :--- | :--- | :--- | :--- |
| **`DOWNSWING`** | Bar Low $L_j < L_{\text{trough}}$ | Always | **`DOWNSWING`** | Update trough: $L_{\text{trough}} = L_j, t_{\text{trough}} = j$. Running downswing extends. |
| **`DOWNSWING`** | Bar Close $C_j \ge L_{\text{trough}} \times (1.0 + \theta_{\text{prom}} - \epsilon)$ | $L_j \ge L_{\text{trough}}$ | **`PROVISIONAL_REBOUND`** | Initialize provisional rebound: $t_{\text{knowable}} = j, \tau_{\text{reb}} = 1, P_{\text{reb}} = H_j, t_{\text{reb}} = j$. |
| **`PROVISIONAL_REBOUND`** | Bar Low $L_j < L_{\text{trough}}$ | **Undercut Before Confirmation** | **`DOWNSWING`** | **Provisional rebound cancelled.** Reset: $L_{\text{trough}} = L_j, t_{\text{trough}} = j$. Counter-trend rally unified into downswing. |
| **`PROVISIONAL_REBOUND`** | $L_j \ge L_{\text{trough}}$ | $j - t_{\text{trough}} < 3$ | **`PROVISIONAL_REBOUND`** | Increment duration: $\tau_{\text{reb}} += 1$. Update rebound high: $P_{\text{reb}} = \max(P_{\text{reb}}, H_j)$. |
| **`PROVISIONAL_REBOUND`** | $C_j \ge L_{\text{trough}} \times (1.0 + \theta_{\text{prom}} - \epsilon)$ | $j - t_{\text{trough}} \ge 3$ and $L_m \ge L_{\text{trough}}$ | **`CONFIRMED_REBOUND`** | **Wave boundary confirmed irreversibly.** Record Wave $k$: peak $P_k$, trough $L_{\text{trough}}$ at $t_{\text{trough}}$, confirmation date at $j$. Initialize $P_{k+1} = \max_{m \in [t_{\text{trough}}+1, j]} H_m$. |
| **`CONFIRMED_REBOUND`** | High $H_j > P_{k+1}$ | $j > t_{\text{peak}, k+1}$ | **`CONFIRMED_REBOUND`** | Update candidate peak: $P_{k+1} = H_j, t_{\text{peak}, k+1} = j$. (Cannot rollover on same bar). |
| **`CONFIRMED_REBOUND`** | Close $C_j \le P_{k+1} \times (1.0 - \theta_{\text{roll}} + \epsilon)$ | $j > t_{\text{peak}, k+1}$ (Subsequent bar) | **`NEW_CONTRACTION`** | Lock peak $(P_{k+1}, t_{\text{peak}, k+1})$. Initiate Wave $k+1$ downswing: $L_{\text{trough}, k+1} = \min_{m \in [t_{\text{peak}}+1, j]} L_m$. |
| **`NEW_CONTRACTION`** | Bar Low $L_j < L_{\text{trough}, k}$ | **Undercut After Confirmation** | **`NEW_CONTRACTION`** | **Prior wave boundary is NOT erased.** Wave $k+1$ trough breaks below Wave $k$. Evaluated as potential expansion! |

---

## 6. Exact 5% / Three-Session Prominence Rule & Numerical Semantics

### 6.1 Authoritative Numerical Comparison
To eliminate binary representation discrepancies, one single mathematical definition governs all recovery evaluations across formulas, pseudocode, transition tables, and tests:

$$\Delta_{\text{rec}}(j) = \frac{C_j - L_{\text{trough}}}{L_{\text{trough}}} \ge \theta_{\text{prom}} - \epsilon_{\text{float}}$$

where:
- Price field: Daily close $C_j$ (positive real number).
- Trough price: $L_{\text{trough}} = \min_{i \in [t_{\text{peak}}, t_{\text{trough}}]} L_i > 0$.
- Threshold constant: $\theta_{\text{prom}} = 0.0500$ (5.0000%, dimensionless ratio).
- Guard: $\epsilon_{\text{float}} = 10^{-7}$ (strictly an IEEE-754 representation guard; defines an effective accepted boundary of $\Delta_{\text{rec}} \ge 0.04999990$).
- *Algebraic Price-Space Equivalent*: $C_j \ge L_{\text{trough}} \times (1.0 + \theta_{\text{prom}} - \epsilon_{\text{float}})$ is explanatory; the dimensionless ratio form is authoritative.
- Invalid/nonpositive price: If $L_{\text{trough}} \le 0$ or $C_j \le 0$, the algorithm fails closed with `INSUFFICIENT_DATA`.

### 6.2 Rollover Comparison
$$\Delta_{\text{roll}}(j) = \frac{P_{k+1} - C_j}{P_{k+1}} \ge \theta_{\text{rollover}} - \epsilon_{\text{float}}$$
where $\theta_{\text{rollover}} = 0.0400$ (4.0000%) and $\epsilon_{\text{float}} = 10^{-7}$ (effective boundary: $\Delta_{\text{roll}} \ge 0.03999990$).

---

### 6.3 Complete Parameter Inventory

| Parameter Name | Symbol | Proposed Value | Dimension / Units | Substantive Role | Ratification Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Rebound Prominence** | $\theta_{\text{prom}}$ | `0.0500` | Dimensionless ratio (5.0%) | Minimum closing recovery above trough to qualify structural prominence. | **PENDING OWNER RATIFICATION** |
| **Elapsed Rebound Duration** | $\theta_{\text{dur}}$ | `3` | Elapsed trading sessions | Minimum elapsed trading days post-trough to confirm boundary. | **PENDING OWNER RATIFICATION** |
| **Rollover Pullback** | $\theta_{\text{rollover}}$ | `0.0400` | Dimensionless ratio (4.0%) | Minimum closing pullback from candidate peak to lock peak and initiate Wave $k+1$. | **PENDING OWNER RATIFICATION** |
| **Floating Guard** | $\epsilon_{\text{float}}$ | `1e-7` | Dimensionless epsilon | IEEE-754 representation guard on ratio boundaries. | **PENDING OWNER RATIFICATION** |
| **Primary Depth Floor** | $\theta_{d1}$ | `0.0800` | Dimensionless ratio (8.0%) | Minimum initial contraction depth required for valid base anchor (Inherited C005). | FROZEN BASELINE |
| **Intermediate Depth Floor**| $\theta_{dk}$ | `0.0400` | Dimensionless ratio (4.0%) | Minimum depth for intermediate contraction waves (Inherited C005). | FROZEN BASELINE |
| **Final Handle Depth Floor**| $\theta_{df}$ | `0.0200` | Dimensionless ratio (2.0%) | Minimum depth for final handle wave near cutoff (Inherited C005). | FROZEN BASELINE |
| **Tightening Reduction** | $\rho_{\text{tight}}$ | `0.0500` | Relative ratio (5.0%) | Relative reduction required: $D_k \le D_{k-1} \times (1.0 - \rho_{\text{tight}})$ (Inherited C005). | FROZEN BASELINE |
| **Ascending Ceiling** | $\theta_{\text{ceiling}}$| `1.0300` | Price ratio (3.0%) | Ceiling threshold before base anchor restarts on new high (Inherited C005). | FROZEN BASELINE |
| **Symmetric Anchor Radius**| $k_{\text{anchor}}$ | `3` | Trading sessions | Symmetric peak confirmation radius for base anchor (Inherited C005). | FROZEN BASELINE |
| **Lookback Max** | $N_{\text{max}}$ | `90` | Trading sessions | Maximum base search lookback window (Inherited C005). | FROZEN BASELINE |
| **Lookback Min** | $N_{\text{min}}$ | `15` | Trading sessions | Minimum base search lookback window (Inherited C005). | FROZEN BASELINE |

---

### 6.4 Synthetic Timelines for Rebound Confirmation & Next-Wave Tracking (Timelines 1–7)

#### Timeline 1: Intervening Low Lower Than Both Endpoints (Correction 2 Verification)
- $t_3$: Wave 1 confirmed with trough \$100.00 at $t_0$.
- $t_4$: High \$110.00, Low \$105.00, Close \$109.00. Candidate peak $P_2 = 110.00, t_{\text{peak}, 2} = t_4$.
- $t_5$: High \$108.00, Low \$104.00, Close \$107.00. $\Delta_{\text{roll}} = (110 - 107)/110 = 2.7\% < 4\%$. Peak remains 110.00.
- $t_6$: High \$107.00, Low **\$102.00**, Close \$106.00. $\Delta_{\text{roll}} = (110 - 106)/110 = 3.6\% < 4\%$. Peak remains 110.00.
- $t_7$: High \$106.00, Low \$103.50, Close \$105.00. $\Delta_{\text{roll}} = (110 - 105)/110 = 4.54\% \ge 4.0\%$.
- **Result**: **Rollover confirmed at $t_7$.** State transitions to `STATE_NEW_CONTRACTION`. Peak locked at \$110.00 ($t_4$).
- **Trough Evaluation**: Running trough evaluates over eligible subsequent bars $m \in [t_5, t_7]$:
  $$L_{\text{trough}, 2} = \min(L_{t_5}=104.0, \, L_{t_6}=102.0, \, L_{t_7}=103.5) = \$102.00 \text{ at index } t_6$$
  *Notice*: The intervening low \$102.00 at $t_6$ is strictly lower than both $L_{t_{\text{peak}}}$ and $L_{t_7}$. It is correctly captured!

#### Timeline 2: New Peak During Provisional Rebound
- $t_0$: Low \$100.00, Close \$100.00 (`STATE_DOWNSWING`).
- $t_1$: High \$106.00, Low \$100.50, Close \$105.50 (+5.5%, `PROVISIONAL`, $P_{\text{reb}} = 106.00$).
- $t_2$: High \$108.00, Low \$102.00, Close \$107.50 (+7.5%, `PROVISIONAL`, $P_{\text{reb}} = 108.00$).
- $t_3$: High \$109.00, Low \$104.00, Close \$108.50 (+8.5%, $j - t_0 = 3$).
- **Result**: **`STATE_CONFIRMED_REBOUND`** at $t_3$. Wave 1 trough permanently locked at $t_0$ (\$100.00). Candidate peak $P_2 = 109.00$ at $t_3$.

#### Timeline 3: Equal-High Ties
- $t_3$: Wave 1 confirmed. Rebound reaches high \$110.00 at $t_4$.
- $t_5$: High \$108.00, Close \$107.00.
- $t_6$: High \$110.00 exactly ($H_{t_6} = H_{t_4} = 110.00$), Close \$108.00.
- **Result**: **Chronological priority applies.** Candidate peak index remains $t_4$. $t_6$ does not overwrite $t_{\text{peak}}$.

#### Timeline 4: Outside Bar Containing New High and New Low (Same-Bar Non-Ordering Applied)
- $t_3$: Wave 1 confirmed with trough \$100.00. Rebound candidate peak is \$106.00.
- $t_4$: Outside bar while in `STATE_CONFIRMED_REBOUND`: High \$108.00 ($> 106.00$), Low \$98.00, Close \$98.50.
- **Result**: Under the **same-bar non-ordering policy**, bar $t_4$ updates the candidate peak to \$108.00 ($t_{\text{peak}, 2} = t_4$). A single bar cannot simultaneously establish a new high and confirm an ordered downswing from that new peak. Low \$98.00 is disregarded for trough initialization on session $t_4$. Rollover evaluation is deferred to subsequent sessions ($j > t_4$).
- $t_5$: High \$107.00, Low \$97.50, Close \$99.00. $\Delta_{\text{roll}}(t_5) = (108.00 - 99.00)/108.00 = 8.33\% \ge 4.0\%$. Rollover confirmed at $t_5$. Peak locked at \$108.00 ($t_4$). Running trough initialized across $m \in [t_4 + 1, t_5]$: $L_{\text{trough}, 2} = \min(\text{lows}[t_5]) = 97.50$ at $t_5$. Wave 1 remains immutable.

#### Timeline 5: Rollover Immediately on the Session Following Peak Bar
- $t_4$: Candidate peak $P_2 = 110.00$ at $t_4$.
- $t_5$: High \$109.00, Low \$104.00, Close \$105.00. $\Delta_{\text{roll}} = (110 - 105)/110 = 4.54\% \ge 4.0\%$.
- **Result**: **Rollover confirmed at $t_5$.** Wave 2 peak locked at \$110.00 ($t_4$). Running trough initialized to $L_{t_5} = 104.00$ at $t_5$.

#### Timeline 6: Cutoff Before Next-Wave Rollover
- $t_3$: Wave 1 confirmed. Candidate peak $P_2 = 110.00$ established at $t_4$.
- $t_5$ ($T_{\text{cutoff}}$): High \$109.00, Close \$108.00. Dataset ends.
- **Result**: Wave 2 downswing never initiated. Final wave is Wave 1. If Wave 1 satisfies final-wave criteria, it evaluates downstream.

#### Timeline 7: Undercut Before vs After Prior-Wave Confirmation
- *Scenario 7A (Before Confirmation)*: $t_0$: Low \$100.00. $t_1$: Close \$105.20 (`PROVISIONAL`). $t_2$: Low \$98.50 ($< 100.00$).
  $\implies$ **Provisional rebound cancelled.** Trough resets to \$98.50 at $t_2$. Wave 1 continues downward.
- *Scenario 7B (After Confirmation)*: $t_0 \to t_3$: Wave 1 confirmed ($P_1 = 120.00, T_1 = 100.00$). At $t_5$, Peak 2 locked at \$110.00. At $t_7$: Low \$96.00 ($< 100.00$).
  $\implies$ **Wave 1 remains immutable.** Recognized as Wave 2 with trough \$96.00 ($D_2 = (110 - 96)/110 = 12.7\%$). Because $D_2 > D_1$ ($12.7\% > 16.7\%$ is false, but if $D_2$ exceeds allowed reduction), progressive tightening evaluates expansion and fails.

---

## 7. Acceptance Criteria & Falsification Framework

### 7.1 Eight-Dimensional Progression Criteria
Acceptance for development progression requires satisfying all eight independent dimensions:

| Dimension | Minimum Sufficient Evidence Required for Acceptance |
| :--- | :--- |
| **A. Synthetic Rule Conformance** | **14 / 14** synthetic test scenarios (SYN-01 to SYN-14) PASS bit-exact with zero regressions. |
| **B. Causal Prefix Consistency** | Fixed-start prefix invariant: Confirmed wave boundaries $(P_k, T_k, t_{\text{conf}, k}, D_k)$ must remain **bit-exact identical** across prefixes $B_{0:N}$ and $B_{0:N+M}$. |
| **C. Expansion Rejection Preservation** | Expanding synthetic test (SYN-02) **MUST FAIL** `PRED_PROGRESSIVE_TIGHTENING`. Historical expansion cases (`DHLGY`, `MEDP`, `ROST`) must be evaluated without assuming Candidate 005 failure codes are ground truth. |
| **D. Full 15-Case Geometry Audit** | Segmentations and reason codes recorded across all 15 positive cases and 9 distinct issuers. Wave-count reduction is an exposed diagnostic, not an oracle. |
| **E. Screening Invariant Integrity** | Trend Template (200 SMA), Volume Dry-Up ($\le 0.70$), and Pivot Zone rules remain 100% frozen. |
| **F. Control Specificity** | Exactly **0 / 45** false positives across Strata 2–6 ($0.0\%$ false positive rate). |
| **G. Evaluation Coverage** | Control abstentions must strictly not exceed baseline 7 cases ($\ge 38 / 45$ coverage). Negative classifications and abstentions must be reported separately. |
| **H. Candidate 004/005 Preservation** | Candidate 004 remains default; Candidate 004 & 005 bit-exact reproducibility 100% verified. |

### 7.2 Dependent Security Accounting
The 15 positive-label cases represent **9 distinct corporate securities**:
- Paired cutoffs: `NVDA` (2), `FTNT` (2), `PLTR` (2), `VRT` (2), `DHLGY` (2), `LLY` (2) = 12 cases.
- Singletons: `RTX` (1), `MEDP` (1), `ROST` (1) = 3 cases.
Total: 15 cases across 9 distinct corporate issuers. All reporting must disclose both metrics. Ticker grouping is reported as descriptive accounting of paired observations, not as statistical evidence.

### 7.3 Control Breakdown & Disaggregated Baselines
Across the 45 control inputs:
- Total baseline: **38 negative classifications (`VCP_NON_QUALIFIED`)**, **7 insufficient-data abstentions (`VCP_INSUFFICIENT_DATA`)**, and **0 false positives**.
- Stage 3/4 Distribution Gate (10 inputs): 9 negative classifications + 1 abstention.
- Non-VCP / Failed-Base Gate (20 inputs): 19 negative classifications + 1 abstention.
- Developing VCP Gate (10 inputs): 10 negative classifications.
- Insufficient Evidence Gate (5 inputs): 5 abstentions.

---

## 8. Finite Development Experiment Budget

1. **Candidate Ceiling**: Exactly **one candidate implementation** (`ARX_VCP_CLASSIFIER_V1_CANDIDATE_006`). No parameter grid sweeps.
2. **Evaluation Ceiling**: A maximum of **two full evaluation runs** against the preserved 60-case development corpus:
   - *Run 1*: Initial frozen benchmark run.
   - *Run 2 (Conditional)*: Permitted **only** if a defect in the evaluation harness is discovered. Algorithm code, parameters, manifests, labels, and cutoffs must remain 100% unmutated.
   - Any parameter or algorithmic modification after Run 1 requires a new version identifier and separate owner authorization.
3. **Holdout Sealing**: Protected Holdout Epoch 002 remains unopened and unavailable.

---

## 9. Minimal Future Implementation Scope

If authorized by David Akara, the future implementation scope is tightly restricted to:

1. **Classifier Engine (`analyst_dashboard/vcp/classifier.py`)**:
   - Add method `_detect_structural_contractions()` (~140 lines).
   - Implement the causal state machine (Section 5) and prominence rule Option A (Section 6).
   - Wire version selection under `VERSION_CANDIDATE_006 = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_006"`.
2. **Contracts & Provenance (`analyst_dashboard/analyzers/`)**:
   - `scanner_contract.py`: Add `VCP_CANDIDATE_006_RULESET_VERSION = "1.2.0"`.
   - `scanner_publication_integrity.py`: Register `CANONICAL_VCP_RULESET_SPEC_V1_2_0` and hash.
3. **Permanent Test Suite (`tests/`)**:
   - Implement `tests/test_vcp_candidate_006_conformance.py` (SYN-01 through SYN-14, ~380 lines).
   - Update `tests/test_vcp_candidate_reproducibility.py` to verify side-by-side execution of C004, C005, and C006 without state leakage.

---

## 10. Compact Owner Decision & Ratification Table

| Decision Item | Exact Proposed Rule | Technical Rationale | Ratification Status |
| :--- | :--- | :--- | :---: |
| **1. Primary Architecture** | **Approach C (Causal State Machine with Undercut Continuation)** | Directly solves the conjunctive noise trap and undercut blindness without arbitrary parameters. | **PENDING OWNER RATIFICATION** |
| **2. Architectural Deferrals** | **Defer Containment Folding, ATR Scaling, and Fixed Sweeps** | Analysis proved post-hoc folding destroys valid tightening contractions, creates circularity, and masks true expansion. | **PENDING OWNER RATIFICATION** |
| **3. Rebound Confirmation** | **Option A (Simpler Elapsed-Session Confirmation)**: $j - t_{\text{trough}} \ge 3$, no undercut, Close meets 5% threshold. Zero intermediate floor. | Eliminates the unratified 2% support floor; avoids rejecting gradual quiet bases without domain justification. | **PENDING OWNER RATIFICATION** |
| **4. Authoritative Numerical Formula** | $\Delta_{\text{rec}}(j) = \frac{C_j - L_{\text{trough}}}{L_{\text{trough}}} \ge 0.0500 - 10^{-7}$ on daily close. | Single unambiguous mathematical definition with floating-point guard; eliminates IEEE-754 discrepancy. | **PENDING OWNER RATIFICATION** |
| **5. Peak Eligibility Interval** | Candidate peak initialized strictly on bars following previous trough: $m \in [t_{\text{trough}, k} + 1, \, t_{\text{conf}}]$. | Eliminates intraday high/low ambiguity on the trough bar; avoids using pre-trough highs. | **PENDING OWNER RATIFICATION** |
| **6. Complete Trough Interval & Rollover** | Running trough evaluated across all subsequent bars $m \in [t_{\text{peak}} + 1, \, j_{\text{roll}}]$; rollover requires subsequent bar ($j > t_{\text{peak}}$). | Enforces same-bar non-ordering policy; captures intervening lows between peak and rollover. | **PENDING OWNER RATIFICATION** |
| **7. Progression Criteria** | **Eight-Dimensional Criteria** (Section 7). Wave count is a diagnostic observation, not an acceptance oracle. | Prevents wave-count gaming; guarantees control specificity (0 false positives) and expansion rejection preservation. | **PENDING OWNER RATIFICATION** |
| **8. Experiment Budget** | Exactly 1 candidate (`Candidate 006`), maximum 2 runs on 60 exposed development cases. Zero holdout access. | Enforces finite empirical evaluation budget; prevents iterative parameter fitting. | **PENDING OWNER RATIFICATION** |
| **9. Protected Holdout Boundary** | **Epoch 002 remains sealed and unaccessed.** | Preserves independent verification integrity under ARX Closure Model v2. | **PENDING OWNER RATIFICATION** |

---

## 11. Operational Boundary Confirmation

- **No Implementation Performed**: Specification preparation only. Zero production code, test code, or prototypes created.
- **No Repository Mutations**: Git tree `40cb84d2952eedef7715defa62559bb20a5d94de` remains unchanged. Working tree is clean.
- **Production Status**: Candidate 004 remains the system default. PR #73 remains on production activation HOLD.
- **Holdout Status**: Protected Holdout Epoch 002 remains sealed.
