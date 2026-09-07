# ARX Terminal vNext: Outcome Intelligence & Institutional Learning Loop

## Document Context
- **Version**: 1.0.0
- **Status**: FROZEN / RATIFIED
- **Scope**: Sprint 7 Engineering Implementation & Institutional Learning Gates
- **Author**: ARX Architecture & Quantitative Governance Group

---

## 1. Executive Summary & The Seventh Dimension

ARX Terminal vNext completes the seven-stage institutional decision intelligence continuum:
$$\begin{aligned}
\text{Sprint 1} &\longrightarrow \mathbf{Decision\ Workspace} && \text{(Orientation, Hierarchy, 65/35 Geometry)} \\
\text{Sprint 2} &\longrightarrow \mathbf{Explainability\ Workspace} && \text{(Progressive Disclosure, Causal Confluence)} \\
\text{Sprint 3} &\longrightarrow \mathbf{Change\ Intelligence} && \text{(Materiality Filtering, Delta Trust Index)} \\
\text{Sprint 4} &\longrightarrow \mathbf{Portfolio\ Attention\ Intelligence} && \text{(Morning Briefing, Cross-Ticker Aggregation)} \\
\text{Sprint 5} &\longrightarrow \mathbf{Committee\ Governance\ Platform} && \text{(Shared Baselines, Consensus, SHA-256 Audit Trail)} \\
\text{Sprint 6} &\longrightarrow \mathbf{Predictive\ Attention\ Intelligence} && \text{(Calibration Engine, ECE $\le 5\%$, Rollback Safety)} \\
\mathbf{Sprint\ 7} &\longrightarrow \mathbf{Outcome\ Intelligence} && \text{(Were we right? If so, why? If not, what did we learn?)}
\end{aligned}$$

### The Core Architectural Mandate
> **Every prediction must have an observable outcome. Every outcome must have an attribution record. Every attribution must be auditable.**

$$\mathbf{Core\ Pipeline}:\ \text{Prediction} \longrightarrow \text{Baseline Snapshot} \longrightarrow \text{Execution Window} \longrightarrow \text{Observed Outcome} \longrightarrow \mathbf{Outcome\ Attribution} \longrightarrow \mathbf{Learning\ Metrics} \longrightarrow \mathbf{Decision\ Journal}$$

---

## 2. Prediction-to-Outcome Lifecycle

Predictions progress through three strictly governed lifecycle stages:

### Stage 1: Prediction Registration
- Occurs when a predictive hypothesis is validated and emitted.
- Binds setup score, conviction score, execution state, macro regime, snapshot hash, and mandatory `expirationAt`.
- Enforces immutability: **INV-O3 (Immutable Prediction History)**. Once created, prediction parameters are frozen.

### Stage 2: Observation Window
- Tracks target attainment, stop breach, time expiration, and regime rotation across defined observation horizons:
  - **Tactical Trader**: 5–20 market sessions
  - **Core Investor**: 30–90 market sessions
  - **Macro Allocation / Advisor**: 90–180 market sessions

### Stage 3: Outcome Classification
A prediction definitively resolves into one of five discrete outcome classes:
1. `SUCCESS`: Price reached target objective prior to stop floor or expiration.
2. `PARTIAL_SUCCESS`: Meaningful favorable excursion achieved ($>50\%$ of corridor) before pullback.
3. `FAILURE`: Stop loss breached or risk invalidation floor hit.
4. `EXPIRED`: Observation window elapsed without either target or stop being triggered.
5. `INVALIDATED`: External macro shock or regime shift (e.g. `RISK_ON` $\to$ `DEFENSIVE`) invalidated the core thesis premises.

---

## 3. Causal Attribution Framework

An outcome record without attribution is an uninterpretable metric. Sprint 7 mandates eight formal attribution categories:

| Category | Typical Outcome | Trigger / Mechanism |
| :--- | :--- | :--- |
| `EXECUTION_SUCCESS` | `SUCCESS` | Favorable execution corridor entry followed by target capture. |
| `TARGET_REACHED` | `SUCCESS` | Setup objective fulfilled with persistent institutional accumulation. |
| `EXECUTION_FAILURE` | `FAILURE` | Execution corridor breakdown or slippage past entry buffer. |
| `STOP_TRIGGERED` | `FAILURE` | Price breached stop floor during adverse price movement. |
| `REGIME_CHANGE` | `INVALIDATED` | Macro regime flipped from thesis premise (e.g. `RISK_ON` to `DEFENSIVE`). |
| `FLOW_DECAY` | `FAILURE` / `EXPIRED` | Institutional smart money flow evaporated ($Z$-score decayed below 0). |
| `VALIDATION_BREAKDOWN` | `FAILURE` | Core quantitative criteria (VCP, Magic Formula) failed during holding period. |
| `THESIS_EXPIRED` | `EXPIRED` | Maximum observation time limit reached without thesis completion. |

---

## 4. Institutional Learning Loop & KPIs

The learning loop does **not** perform automated black-box model retraining. Instead, it computes auditable performance metrics that feed committee reviews and calibration monitors:

### 4.1 Prediction Actionability Rate (PAR)
$$\text{PAR} = \frac{\text{Reviewed Predictions}}{\text{Displayed Predictions}} \ge \mathbf{50\%}$$

### 4.2 Calibration Stability Gate
$$\text{ECE} \le \mathbf{0.05\ (5\%)} \quad \text{across 30 consecutive observation windows.}$$

### 4.3 Signal Effectiveness & Attribution Metrics
- **Driver Win Rates**: Tracking win rate per primary driver (e.g., Institutional Accumulation: $72\%$, Regime Alignment: $69\%$).
- **Failure Driver Distribution**: Tracking root causes of failures (e.g., Regime Deterioration: $42\%$, Flow Reversal: $24\%$, Stop Triggered: $21\%$).

---

## 5. Formal Invariants & Rules

- **INV-O1 (Mandatory Observable Outcome)**: Every completed prediction must map to an `OutcomeRecord`.
- **INV-O2 (Mandatory Causal Attribution)**: Every outcome record must have an `attributionCategory` and non-empty `explanation`.
- **INV-O3 (Immutable Prediction Record)**: Outcome evaluation cannot modify initial prediction parameters.
- **INV-O4 (Deterministic Attribution)**: Same inputs produce identical primary/secondary attribution drivers.
- **INV-O5 (Chained Audit Provenance)**: Prediction, outcome, and attribution IDs are cryptographically chained and verifiable.

---

## 6. Release Gates (Sprint 7)

- **G7.1**: 100% of completed predictions produce valid `OutcomeRecord` entries.
- **G7.2**: 100% of outcomes contain populated attribution category and explanation.
- **G7.3**: Historical prediction records remain 100% immutable upon outcome resolution.
- **G7.4**: Attribution engine executes with 100% deterministic reproducibility.
- **G7.5**: Outcome audit chains linked through immutable UUIDv4 references.
- **G7.6**: Learning metrics (win rates, driver distributions, PAR) generated accurately.
- **G7.7**: Decision journal and outcome dashboard operational with full visual compliance.
