# ARX Terminal vNext — Phase 28: Decision Improvement Rating (DIR) & Behavioral Cohort Migration Framework

**Document Version**: 1.0.0  
**Status**: APPROVED & INSTITUTIONAL PRODUCTION READY  
**Classification**: Proprietary Institutional Architecture Document  
**Date**: September 2026  
**Primary Author**: ARX Institutional Architecture Team  
**Reviewers**: Head of Product, Institutional UX Architect, Chief Risk Officer  

---

## 1. Executive Mandate & Strategic Vision

Traditional financial intelligence and institutional order routing systems focus exclusively on point-in-time decision evaluation (*"What was the Sharpe ratio? Did this trade beat the benchmark?"*). They remain blind to the central determinant of long-term risk-adjusted alpha: **the decision-maker's behavioral trajectory**.

Phase 28 establishes the **Decision Improvement Rating (DIR)** as the primary North Star metric of the ARX Terminal. DIR shifts the platform paradigm:
$$\text{Decision Evaluation Engine} \longrightarrow \text{Decision Improvement Operating System}$$

DIR quantitatively answers the foundational executive question:
> **"Is this portfolio manager, trader, or executive becoming a measurably better decision-maker over time?"**

Alongside DIR, the **Behavioral Cohort Migration Framework** systematically tracks, measures, and accelerates the transition of capital allocators from passive signal consumers into disciplined, high-velocity institutional operators.

---

## 2. Mathematical Formulation of DIR

DIR is a bounded institutional index ($[0, 100]$) computed as the weighted linear combination of five orthogonal behavioral dimensions:

$$\mathbf{DIR} = 0.30(\text{DQG}) + 0.20(\text{BAS}) + 0.20(\text{RAS}) + 0.15(\text{DRS}) + 0.15(\text{LVI})$$

### 2.1 Component 1: Decision Quality Growth (DQG) — 30% Weight
Normalizes the decision-maker's empirical Decision Quality Score (DQS) progression from their baseline towards the institutional theoretical maximum ($100$):

$$\text{DQG} = \left( \frac{\text{Current DQS} - \text{Baseline DQS}}{100 - \text{Baseline DQS}} \right) \times 100$$

For the canonical executive profile (Baseline DQS = 62, Current DQS = 74):
$$\text{DQG} = \left( \frac{74 - 62}{100 - 62} \right) \times 100 = \frac{12}{38} \times 100 = 31.5789\% \approx 31.6$$
$$\text{Weighted Contribution} = 31.5789 \times 0.30 = 9.4737 \text{ pts}$$

### 2.2 Component 2: Behavioral Adoption Score (BAS) — 20% Weight
Measures the conversion fidelity of quantitative model signals and AI Mentor recommendations into live executions:

$$\text{BAS} = \left( \frac{\text{Recommendations Followed}}{\text{Recommendations Issued}} \right) \times 100$$

For the canonical portfolio (79 followed out of 112 issued):
$$\text{BAS} = \left( \frac{79}{112} \right) \times 100 = 70.5357\% \approx 70.5\%$$
$$\text{Weighted Contribution} = 70.5357 \times 0.20 = 14.1071 \text{ pts}$$

### 2.3 Component 3: Rule Adherence Score (RAS) — 20% Weight
Calculates policy discipline across four non-negotiable risk boundaries:

$$\text{RAS} = 0.30(\text{Stop Loss}) + 0.20(\text{Macro Invalidation}) + 0.20(\text{Position Sizing}) + 0.30(\text{Risk Controls})$$

Given:
- Stop Loss Adherence = $91.0\%$
- Macro Invalidation Adherence = $85.0\%$
- Position Sizing Adherence = $88.0\%$
- Risk Controls Adherence = $84.0\%$

$$\text{RAS} = 0.30(91) + 0.20(85) + 0.20(88) + 0.30(84) = 27.3 + 17.0 + 17.6 + 25.2 = 87.10\%$$
$$\text{Weighted Contribution} = 87.10 \times 0.20 = 17.4200 \text{ pts}$$

### 2.4 Component 4: Drift Resistance Score (DRS) — 15% Weight
Inverts behavioral thesis drift (emotional impulse trades, revenge sizing, mandate violations):

$$\text{DRS} = 100 - \text{Behavioral Drift}\%$$

For current drift = $21.0\%$:
$$\text{DRS} = 100 - 21.0 = 79.00\%$$
$$\text{Weighted Contribution} = 79.00 \times 0.15 = 11.8500 \text{ pts}$$

### 2.5 Component 5: Learning Velocity Index (LVI) — 15% Weight
Measures the rate of behavioral adaptation derived from post-decision reviews, journal depth, and AI coaching sessions:

$$\text{LVI} = 68.00$$
$$\text{Weighted Contribution} = 68.00 \times 0.15 = 10.2000 \text{ pts}$$

### 2.6 Total Raw & Final Canonical DIR
$$\text{Raw DIR} = 9.4737 + 14.1071 + 17.4200 + 11.8500 + 10.2000 = 63.0508 \approx 63.05$$
$$\mathbf{Final\ DIR} = \mathbf{63} \quad (\text{Tier: } \mathbf{IMPROVING})$$
$$\text{Peer Percentile} = \text{Top 28\% (Faster than 72\% of institutional cohort)}$$

---

## 3. Six Strict Institutional Validation Gates

To ensure auditability, institutional integrity, and risk guardrails, DIR calculations enforce six fail-closed validation rules:

| Gate | Rule ID | Threshold | Canonical Status | Failure Behavior |
|---|---|---|---|---|
| 1 | `RULE_1_MIN_OBSERVATIONS` | $\ge 30$ recorded decisions | **42 (PASS)** | Sets `isDataSufficient = false`; renders warning badge |
| 2 | `RULE_2_MIN_RECOMMENDATIONS` | $\ge 20$ issued recommendations | **112 (PASS)** | Flags low-sample variance warning on BAS |
| 3 | `RULE_3_MIN_REVIEWS` | $\ge 10$ retrospective outcome reviews | **28 (PASS)** | Applies dampening penalty to LVI |
| 4 | `RULE_4_MAX_DRIFT_CAP` | $\le 60.0\%$ behavioral drift | **21.0% (PASS)** | If drift $> 60\%$, DIR is hard-capped at $\le 70.0$ |
| 5 | `RULE_5_BEHAVIOR_PENALTY` | $\ge 50.0\%$ Rule Adherence Score | **87.1% (PASS)** | If RAS $< 50\%$, $-15.0\text{ pts}$ penalty deducted |
| 6 | `RULE_6_CONFIDENCE_THRESHOLD` | $\ge 70.0\%$ statistical CI certainty | **88.0% (PASS)** | Marks dataset as `isLowConfidenceDataset = true` |

---

## 4. 90-Day Trajectory Projection & Driver Decomposition

- **Current DIR**: 63 / 100
- **90-Day Projected DIR**: 69 / 100 (+6.0 pts)
- **Projected Confidence**: 88.0% (Bayesian model projection)
- **Target Tier Benchmark**: 75.0 ("High Performer" tier)
- **Strongest Positive Driver**: **+6.2 pts** — Stop-Loss Discipline & Post-Loss Risk Compression
- **Largest Negative Obstacle**: **-4.1 pts** — Late Profit-Taking Drift on Macro Regime Shifts

---

## 5. 6-Stage Behavioral Cohort Migration Framework

ARX models institutional user maturity across six distinct behavioral cohorts:

```
[1. Consumer]  (18%, -4% YoY) -> Browse & Watch
      │
      ▼ (CAR 38% transition)
[2. Investigator] (21%, -3% YoY) -> Chart & Drill-Down
      │
      ▼ (CAR 34% transition)
[3. Practitioner] (29%, +2% YoY) -> Execute Rules & Sizing
      │
      ▼ (CAR 29% transition)
[4. Learner] (20%, +3% YoY) -> Outcome Journaling & Attribution
      │
      ▼ (CAR 22% transition)
[5. Optimizer] (9%, +1.5% YoY) -> Stress-Testing & Edge Tuning
      │
      ▼ (CAR 14% transition)
[6. Operator] (3%, +0.5% YoY) -> Autonomous Institutional Governance
```

### Institutional Migration Health KPIs
1. **Cohort Advancement Rate (CAR)**: **31.0%** (Target: $> 25.0\%$ | **PASS**)
2. **Cohort Regression Rate (CRR)**: **4.0%** (Target: $< 10.0\%$ | **PASS**)
3. **Time to Maturity (TTM)**: **142 days** (Target: $< 180\text{ days}$ | **PASS**)
4. **Cohort Velocity Score (CVS)**: **0.033 / day** (Target: $> 0.025 / \text{day}$ | **PASS**)
5. **Net Flow Direction**: **+6.0% net shift** from passive tiers into active disciplined execution tiers.

---

## 6. Enhanced Statistical Confidence Bands (Wilson Score & Mean SE)

Every behavioral KPI receives uncertainty bounds calculated via Wilson score intervals (binary outcomes) and standard error of the mean (continuous metrics):

### Three-Tier Color-Coding Hierarchy
1. **Green (`#22c55e`)**: Entire confidence interval is strictly above target benchmark ($L_{\text{bound}} > \text{Target}$).
2. **Yellow (`#f59e0b`)**: Target benchmark is contained within the confidence interval ($L_{\text{bound}} \le \text{Target} \le U_{\text{bound}}$).
3. **Red (`#ef4444`)**: Entire confidence interval is strictly below target benchmark ($U_{\text{bound}} < \text{Target}$).

### Strict Sample Size Guard
- If $n < 30$, rendering confidence intervals is statistically irresponsible.
- The UI intercepts $n < 30$ and outputs an explicit fallback badge: `Data Insufficient (n < 30)`.

---

## 7. Verification & Production Certification

The framework is verified by automated test suite `frontend/scripts/verify-dir-framework.mjs`:
- Contracts & Data Models: 11/11 tests passed
- DIR Math & Weights: 21/21 tests passed
- 6 Strict Validation Gates: 10/10 tests passed
- Behavioral Cohort Migration: 14/14 tests passed
- UI Components & Master Dashboard: 25/25 tests passed
- **Total Suite Pass Rate**: **81 / 81 tests passed (100%)**

### Architectural Invariant Checklist
- [x] Phase 26 Quantitative Freeze observed (Zero Python backend changes)
- [x] Anti-Cyan Palette strictly enforced (Emerald for growth, Amber for warnings, Slate for neutral)
- [x] Responsive layout tested across 390px mobile and 1440px desktop
- [x] Zero hydration errors or unescaped JSX characters
- [x] Next.js production build: Exit Code 0, Shared JS Bundle $\le 100.0\text{ KB}$

---

**Sign-off**:  
*ARX Institutional Architecture Committee & Head of Product*  
*Certification Date: September 2026*
