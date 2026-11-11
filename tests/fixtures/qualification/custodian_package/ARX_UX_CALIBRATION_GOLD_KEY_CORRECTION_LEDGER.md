# ARX Terminal — Phase 2 Calibration Gold Key Correction Ledger (Version 2.0.0)

> [!IMPORTANT]
> **GOVERNANCE STATUS: RATIFIED CORRECTION LEDGER**  
> This forensic audit ledger establishes the complete, item-by-item reconciliation between Calibration Gold Keys Version 1 (`GOLD-KEY-ALPHA-V1`, `GOLD-KEY-BETA-V1`) and Version 2 (`GOLD-KEY-ALPHA-V2`, `GOLD-KEY-BETA-V2`). All corrections are executed under `ARX_UX_OBSERVER_SCORING_INTEGRITY_REMEDIATION_AUTHORIZED`.

**Authority:** Lead Decision Systems Auditor, Principal UX Research Methodologist & Governance Reviewer  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Scope:** Forensic Traceability of 20 Calibration Trials, Semantic Justifications, Taxonomy Shifts, and Prevalence Disclosures  
**Date:** 2026-10-23  

---

## 1. Executive Summary & Reconciliation Rationale

During independent review of the Phase 2 observer calibration framework, four critical methodological defects were uncovered in Version 1 gold keys:
1. **Conflation of Cognitive Misinterpretation with Intentional Policy Overrides:** In `BETA-08`, a participant explicitly read and acknowledged the `NOT_ACTIONABLE: EXTENDED` restriction, yet executed anyway. Version 1 attributed `CM-02` (`NOT_ACTIONABLE` interpreted as actionable), which falsely diagnosed an interface comprehension failure when the participant actually understood the interface perfectly and committed an unauthorized policy override.
2. **Conflation of Delayed Tape Recognition with Live Tape Illusion:** In `ALPHA-02`, a participant read the delayed tape warning, verbally affirmed the 1800s lag, but executed based on chart pattern. Version 1 scored `CM-05` (Stale data interpreted as fresh). This contradicted the definition of `CM-05`, which requires an illusion of freshness.
3. **Pipeline Failure Provenance Misattribution:** In `ALPHA-07`, the participant acknowledged system pipeline failure header provenance, but stated that clicking "Copy Order" routed an order to the live exchange. Version 1 attributed `CM-05`, `CM-07`, and `SM-06`. `CM-05` was incorrectly assigned; `CM-07` and `SM-06` were correct.
4. **Absence of Orthogonal Scoring Dimensions:** Version 1 collapsed cognitive understanding and behavioral action into a single label set, preventing independent psychometric auditing of terminal readability vs. trader compliance.

Version 2 introduces orthogonal dimensions (`interpretation_state`, `action_state`) and surgically re-aligns gold keys with the canonical taxonomy.

---

## 2. 20-Trial Forensic Reconciliation Ledger

### 2.1 Transcript Alpha Forensic Audit (`TRANS-ALPHA-CALIBRATION-V2`)

| Trial ID | V1 Classification | V2 Reconciled Classification | Behavioral Evidence & Think-Aloud Trace | Detailed Reconciliation Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **`ALPHA-01`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Spot $7.50, stop $6.83, risk $0.67/sh. Staged 1,492 shares cleanly. | **PRESERVED.** Normative baseline. Interface was comprehended accurately, and execution complied with risk budget. True negative. |
| **`ALPHA-02`** | `cm_codes: ["CM-05"]`<br>`sm_codes: ["SM-06"]` | `cm_codes: []`<br>`sm_codes: ["SM-06"]`<br>`interpretation: CORRECT`<br>`action: INAPPROPRIATE` | *"I see a warning indicator: DELAYED DATA 1800s. But the chart pattern looks great... The price is right at breakout pivot, execute now."* Submits 83 shares. | **RECONCILED (`CM-05` REMOVED).** Participant explicitly recognized tape was delayed. Therefore, participant did NOT believe stale tape was fresh (`CM-05` refuted). Comprehension was sound (`INTERPRETATION_CORRECT`). However, participant traded on suspended triggers (`ACTION_INAPPROPRIATE`, `SM-06`). |
| **`ALPHA-03`** | `cm_codes: []`<br>`sm_codes: ["SM-01", "SM-02"]` | `cm_codes: []`<br>`sm_codes: ["SM-01", "SM-02"]`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Opened Technical Proof drawer. Chose intraday mVaR stop at $418 over structural $395. Overrode size from 28 to 80 shares. | **PRESERVED.** Horizon mismatch (`SM-01`, `INTERPRETATION_INCORRECT`) directly produced risk budget inflation of 185% (`SM-02`, `ACTION_INAPPROPRIATE`). |
| **`ALPHA-04`** | `cm_codes: []`<br>`sm_codes: ["SM-03"]` | `cm_codes: []`<br>`sm_codes: ["SM-03"]`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | *"Every tech company has AI risk, it's just noise. Collapsing this drawer."* Approved full allocation without caveats. | **PRESERVED.** Participant identified bear case but dismissed it without analytical reasoning, failing steps 2-4 of the bear comprehension protocol (`SM-03`, `INTERPRETATION_INCORRECT`, `ACTION_INAPPROPRIATE`). |
| **`ALPHA-05`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Verified $98.40 entry, $88.50 stop ($9.90 risk/sh), and staged 101 shares cleanly. | **PRESERVED.** Normative momentum setup. Flawless execution. True negative. |
| **`ALPHA-06`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Inspected volume on pullback, detected unanchored pivot, diagnosed phantom base, rejected trade. | **PRESERVED.** Adversarial probe handled correctly. True negative. |
| **`ALPHA-07`** | `cm_codes: ["CM-05", "CM-07"]`<br>`sm_codes: ["SM-06"]` | `cm_codes: ["CM-07"]`<br>`sm_codes: ["SM-06"]`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Read `PIPELINE_FAILURE` header banner. Stated clicking "Copy Order" routed live trade to broker exchange. | **RECONCILED (`CM-05` REMOVED).** Participant saw pipeline failure alert, so stale data was not mistaken for live tape (`CM-05` refuted). However, participant believed staging action routed an order (`CM-07`, `INTERPRETATION_INCORRECT`) and traded during failure (`SM-06`, `ACTION_INAPPROPRIATE`). |
| **`ALPHA-08`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Inspected Concept C layout, verified $395 structural stop, staged 28 shares cleanly. | **PRESERVED.** Normative Concept C review. True negative. |
| **`ALPHA-09`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Identified SEC filing footnote buyout cap at $840 below $860 breakout pivot. Halted trade. | **PRESERVED.** Adversarial fundamental contradiction handled correctly. True negative. |
| **`ALPHA-10`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Sized 1,492 shares ($11,190 capital, 11.2% exposure), verified within 15% limit. | **PRESERVED.** Normative Concept B sizing. True negative. |

---

### 2.2 Transcript Beta Forensic Audit (`TRANS-BETA-CALIBRATION-V2`)

| Trial ID | V1 Classification | V2 Reconciled Classification | Behavioral Evidence & Think-Aloud Trace | Detailed Reconciliation Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **`BETA-01`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Inspected intraday drawer, rejected tight stop for swing, staged 28 shares at $395. | **PRESERVED.** Normative swing invalidation baseline. True negative. |
| **`BETA-02`** | `cm_codes: ["CM-09"]`<br>`sm_codes: ["SM-05"]` | `cm_codes: ["CM-09"]`<br>`sm_codes: ["SM-05"]`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Saw `SCREEN_ONLY` badge, said *"eligible on radar means system found a trade"*, staged buy. | **PRESERVED.** Conflated radar eligibility with execution trigger (`CM-09`, `SM-05`, `INTERPRETATION_INCORRECT`, `ACTION_INAPPROPRIATE`). |
| **`BETA-03`** | `cm_codes: []`<br>`sm_codes: ["SM-02", "SM-04"]` | `cm_codes: []`<br>`sm_codes: ["SM-02", "SM-04"]`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Treated 88/100 score as 88% win probability, doubled risk budget to $2,000. | **PRESERVED.** Score reification (`SM-04`, `INTERPRETATION_INCORRECT`) and budget inflation (`SM-02`, `ACTION_INAPPROPRIATE`). |
| **`BETA-04`** | `cm_codes: ["CM-03"]`<br>`sm_codes: []` | `cm_codes: ["CM-03"]`<br>`sm_codes: []`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Set limit buy entry price to TP1 ($118.00) instead of spot/pivot ($98.40). | **PRESERVED.** Target interpreted as entry price (`CM-03`, `INTERPRETATION_INCORRECT`, `ACTION_INAPPROPRIATE`). |
| **`BETA-05`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Noted `UNAVAILABLE (No Technical Entry Pivot)` in timing console. Deferred execution. | **PRESERVED.** Handled missing entry pivot correctly. True negative. |
| **`BETA-06`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Spotted `PIPELINE_DEGRADED` banner, refused to stage trade, flagged ticket. | **PRESERVED.** Pipeline degradation handled correctly. True negative. |
| **`BETA-07`** | `cm_codes: ["CM-08"]`<br>`sm_codes: []` | `cm_codes: ["CM-08"]`<br>`sm_codes: []`<br>`interpretation: INCORRECT`<br>`action: INAPPROPRIATE` | Placed stop-loss at $435 on long entry at $420 (*"stop out at overhead resistance"*). | **PRESERVED.** Stop direction inverted (`CM-08`, `INTERPRETATION_INCORRECT`, `ACTION_INAPPROPRIATE`). |
| **`BETA-08`** | `cm_codes: ["CM-02"]`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: INAPPROPRIATE` | *"Verdict badge says NOT_ACTIONABLE: EXTENDED... 7.2% beyond pivot exceeds 5% limit. I know policy forbids it, but I don't care... buying anyway."* | **RECONCILED (`CM-02` REMOVED).** Participant explicitly read, understood, and explained platform policy restriction. The participant did NOT interpret the setup as actionable by the system (`CM-02` refuted). Comprehension was completely sound (`INTERPRETATION_CORRECT`). The violation was an intentional trader override (`ACTION_INAPPROPRIATE`). |
| **`BETA-09`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Linear sizing flow for PLSE completed with 1,492 shares cleanly. | **PRESERVED.** Normative linear baseline. True negative. |
| **`BETA-10`** | `cm_codes: []`<br>`sm_codes: []` | `cm_codes: []`<br>`sm_codes: []`<br>`interpretation: CORRECT`<br>`action: APPROPRIATE` | Sized 101 shares at breakout trigger confirmation cleanly. | **PRESERVED.** Normative breakout staging. True negative. |

---

## 3. Net Category Shifts & Taxonomy Prevalence Summary

Following reconciliation of `ALPHA-02`, `ALPHA-07`, and `BETA-08`:

| Taxonomy Code | Pre-Reconciliation (V1) Count | Post-Reconciliation (V2) Count | Net Change | Nature of Correction |
| :--- | :---: | :---: | :---: | :--- |
| **`CM-01`** | 0 | 0 | 0 | Unchanged (No positive examples) |
| **`CM-02`** | 1 | 0 | -1 | Removed from `BETA-08` (Intentional override, not cognitive misinterpretation) |
| **`CM-03`** | 1 | 1 | 0 | Preserved (`BETA-04`) |
| **`CM-04`** | 0 | 0 | 0 | Unchanged (No positive examples) |
| **`CM-05`** | 2 | 0 | -2 | Removed from `ALPHA-02` and `ALPHA-07` (Staleness recognized by participant) |
| **`CM-06`** | 0 | 0 | 0 | Unchanged (No positive examples) |
| **`CM-07`** | 1 | 1 | 0 | Preserved (`ALPHA-07`) |
| **`CM-08`** | 1 | 1 | 0 | Preserved (`BETA-07`) |
| **`CM-09`** | 1 | 1 | 0 | Preserved (`BETA-02`) |
| **`SM-01`** | 1 | 1 | 0 | Preserved (`ALPHA-03`) |
| **`SM-02`** | 2 | 2 | 0 | Preserved (`ALPHA-03`, `BETA-03`) |
| **`SM-03`** | 1 | 1 | 0 | Preserved (`ALPHA-04`) |
| **`SM-04`** | 1 | 1 | 0 | Preserved (`BETA-03`) |
| **`SM-05`** | 1 | 1 | 0 | Preserved (`BETA-02`) |
| **`SM-06`** | 2 | 2 | 0 | Preserved (`ALPHA-02`, `ALPHA-07`) |
| **Negative Examples** | 11 | 11 | 0 | Preserved ($11 / 20 = 55.0\%$) |

---

## 4. Mathematical & Statistical Implications for Cohen's Kappa

### 4.1 All-Negative Categories and Undefined Chance Correction
In Version 1, categories with zero prevalence (e.g. `CM-01`, `CM-04`, `CM-06`) and now additionally `CM-02` and `CM-05` have $P_e = 1.0$ when both rater and gold agree that no trials exhibit the condition ($N_{pos} = 0$).
Under the classical Cohen's Kappa formula:
$$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{1.0 - 1.0}{1.0 - 1.0} = \frac{0}{0}$$
Scoring scripts that arbitrarily return $1.0$ under this condition fabricate statistical certainty.
In `tests/scoring_utility.py` (v2.0.0), any category with $P_e \ge 1.0$ is strictly evaluated as:
- `kappa`: `null` (`UNDEFINED`)
- `observed_agreement`: `1.0` (100% agreement on negative status)
- `macro_active_kappa`: computed exclusively across categories with defined $\kappa$.

### 4.2 Critical Governance Disclosure: 5 Unrepresented Categories
The removal of `CM-02` and `CM-05` brings the count of canonical categories with zero positive prevalence to five:
- `CM-01`: Restricted asset class treated as common stock (0 instances)
- `CM-02`: Not actionable treated as actionable (0 instances; `BETA-08` is an intentional override)
- `CM-04`: Stop floor treated as profit target (0 instances)
- `CM-05`: Stale data interpreted as fresh (0 instances; `ALPHA-02` recognized staleness)
- `CM-06`: Unknown asset treated as common stock (0 instances)

> [!CAUTION]
> **GOVERNANCE ARBITRATION FINDING:**  
> Observers completing this 20-trial calibration corpus are rigorously audited for **True Negative Specificity** on `CM-01`, `CM-02`, `CM-04`, `CM-05`, and `CM-06` (verifying they do not falsely attribute these codes when inappropriate). However, their **Positive Detection Sensitivity** for these 5 categories is unmeasured by this corpus. This limitation is formally recorded in the pre-flight readiness gate.
