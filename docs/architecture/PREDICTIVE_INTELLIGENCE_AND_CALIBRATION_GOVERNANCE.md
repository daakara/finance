# ARX Terminal vNext: Predictive Intelligence & Calibration Governance Specification

## Document Context
- **Version**: 1.0.0
- **Status**: FROZEN / RATIFIED
- **Scope**: Sprint 6 Engineering Implementation & Institutional Safety Gates
- **Author**: ARX Architecture & Quantitative Governance Group

---

## 1. Executive Summary & Paradigm Shift

ARX Terminal vNext evolves through six progressive capability layers:
$$\begin{aligned}
\text{Sprint 1} &\longrightarrow \mathbf{Decision\ Workspace} && \text{(Orientation, Hierarchy, 65/35 Geometry)} \\
\text{Sprint 2} &\longrightarrow \mathbf{Explainability\ Workspace} && \text{(Progressive Disclosure, Causal Confluence)} \\
\text{Sprint 3} &\longrightarrow \mathbf{Change\ Intelligence} && \text{(Materiality Filtering, Delta Trust Index)} \\
\text{Sprint 4} &\longrightarrow \mathbf{Portfolio\ Attention\ Intelligence} && \text{(Morning Briefing, Cross-Ticker Aggregation)} \\
\text{Sprint 5} &\longrightarrow \mathbf{Committee\ Governance\ Platform} && \text{(Shared Baselines, Consensus, SHA-256 Audit Trail)} \\
\mathbf{Sprint\ 6} &\longrightarrow \mathbf{Predictive\ Intelligence} && \text{(What will likely require attention next?)}
\end{aligned}$$

### The Core Architectural Invariant
> **$\mathbf{Prediction \ne Fact}$. Predictions remain hypotheses with calibrated probabilities until confirmed by future observations. No prediction may influence users without calibration, monitoring, validation, and rollback controls.**

$$\mathbf{Core\ Flow}:\ \text{Raw Signals} \longrightarrow \text{Prediction Engine} \longrightarrow \text{Calibration Layer} \longrightarrow \text{Validation Layer} \longrightarrow \text{Monitoring Layer} \longrightarrow \text{Prediction Surface}$$

---

## 2. Sprint 5 Threat Model & Enterprise Defense

To protect committee decisions, shared baselines, audit trails, and decision provenance from malicious or accidental degradation, Sprint 5 establishes seven core threat boundaries:

### Threat Area 1: Permission Escalation
- **Threat**: An Analyst role gains CIO-equivalent authority (e.g. approving baselines).
- **Mitigation**: Triple-enforcement architecture:
  1. UI Action Guard (`canApproveBaseline`)
  2. API Endpoint Authorization Middleware (returns HTTP 403)
  3. Repository Layer Enforcement (`BaselineRepository.approveBaseline` throws `UNAUTHORIZED_APPROVAL`)
- **Rule**: Never trust client-side UI tokens alone.

### Threat Area 2: Baseline Hijacking
- **Threat**: Competing active baselines created for the same ticker/committee causing consensus fragmentation.
- **Mitigation**: Database & Repository Invariant B1:
  $$\text{UNIQUE}(\text{committee\_id}, \text{ticker}, \text{status} = \text{'ACTIVE'})$$
  Creating a new approved baseline automatically supersedes the incumbent; superseded baselines are permanently immutable.

### Threat Area 3: Audit Log Tampering
- **Threat**: Historical events modified, deleted, or timestamps rewritten.
- **Mitigation**: Append-only cryptographic storage.
  $$\text{Forbidden}:\ \text{UPDATE audit\_events},\ \text{DELETE audit\_events}$$
  $$\text{Chaining}:\ H_n = \text{SHA256}(\text{eventId} + \text{timestamp} + \text{actorId} + \text{action} + H_{n-1})$$
  Any mutation immediately breaks `verifyChain()` with $< 1\text{ms}$ detection latency.

### Threat Area 4: Audit Replay Attack
- **Threat**: Legitimate audit events maliciously re-submitted to distort history.
- **Mitigation**: Globally unique, idempotent UUIDv4 `eventId` enforced by `UNIQUE(event_id)`.

### Threat Area 5: Committee Collusion
- **Threat**: A single Portfolio Manager proposes and self-approves baseline alterations.
- **Mitigation**: **Approval Separation Rule**:
  $$\text{Creator} \ne \text{Approver}$$
  PM proposes; only designated CIO can formally activate.

### Threat Area 6: Audit Scalability
- **Threat**: Event explosion across hundreds of committees and thousands of tickers over multiple years.
- **Mitigation**: Tiered storage architecture:
  - Hot Storage (0–12 Months): Active relational/IndexedDB query layer.
  - Warm Archive (1–7 Years): Compressed, partitioned by `(committee_id, year, month)`.
  - Cold Archive (7+ Years): Immutable WORM cloud compliance buckets.

### Threat Area 7: Audit Verification Cost
- **Threat**: Full re-verification of millions of events becomes computationally prohibitive ($O(N)$).
- **Mitigation**: Checkpoint hashes stored every 1,000 events:
  $$H_{\text{checkpoint}} = \text{Hash}(E_{1000k})$$
  Verification is incremental ($O(k)$ from latest checkpoint).

---

## 3. Sprint 6 Prediction Invariants & Formal Contracts

Every prediction record emitted within ARX Terminal must satisfy six mandatory invariants:

| Invariant | Name | Rule | Failure Action |
| :--- | :--- | :--- | :--- |
| **INV-P1** | **Mandatory Expiration** | Every prediction must contain an ISO `expirationAt` timestamp. | Throws `EXPIRATION_REQUIRED` |
| **INV-P2** | **Probability Bounds** | $0.0 \le \text{probability} \le 1.0$. | Throws `INVALID_PROBABILITY` |
| **INV-P3** | **Prediction $\ne$ Fact** | Status can never be instantiated as `CONFIRMED`. | Throws `PREDICTION_CANNOT_BE_CONFIRMED_AT_CREATION` |
| **INV-P4** | **Immutable Outcomes** | Once evaluated (`CORRECT`, `INCORRECT`, `PARTIAL`), outcome records cannot be modified or deleted. | Throws `OUTCOME_IMMUTABLE` |
| **INV-P5** | **Reproducibility Hash** | Must reference non-empty `currentStateHash` linking to exact input snapshot. | Throws `SNAPSHOT_HASH_REQUIRED` |
| **INV-P6** | **Explainability First** | `rationale.length > 0` required. Zero black-box predictions. | Throws `RATIONALE_REQUIRED` |

---

## 4. Prediction Calibration Governance Framework

A prediction cannot be surfaced to users without empirical calibration demonstrating that predicted probabilities equal observed frequencies:
$$\mathbb{P}(\text{Outcome} = 1 \mid \hat{p} = p) \approx p$$

### 4.1 Calibration Buckets
Predictions are partitioned into 10 disjoint confidence bins:
$$B_m = \{i : \hat{p}_i \in [b_{m-1}, b_m)\},\quad m \in \{1, \dots, 10\}$$

For each bucket $B_m$, the empirical accuracy and average confidence are computed:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i, \qquad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

### 4.2 Mandatory Calibration Targets
1. **Expected Calibration Error (ECE)**:
   $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| \le \mathbf{0.05\ (5\%)}$$
2. **Brier Score**:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2 \le \mathbf{0.15}$$

---

## 5. Drift Monitoring & Alert Thresholds

The Drift Monitor continuously evaluates feature distributions, market regimes, and forecast outcomes over trailing 30-day windows.

| Metric | Warning Threshold ($>10\%$) | Critical Threshold ($>20\%$) | Action on Critical |
| :--- | :--- | :--- | :--- |
| **Feature Drift** | Distribution shift $>10\%$ | Distribution shift $>20\%$ | Deployment freeze on new models |
| **Regime Drift** | Macro regime divergence $>10\%$ | Divergence $>20\%$ | Re-calibrate regime probability weighting |
| **Outcome Drift** | Hit rate degradation $>10\%$ | Degradation $>20\%$ | Trigger soft rollback (hide UI predictions) |

---

## 6. Multi-Tier Rollback Safety Architecture

ARX provides instantaneous rollback mechanisms to isolate degraded or corrupted forecasting models:

```
                          [Model Health Check]
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     [Normal Operations]    [Soft Rollback]       [Hard Rollback]
      - ECE ≤ 0.05           - DTI < 90%           - ECE > 15%
      - Brier ≤ 0.15         - DPR < 75%           - Precision < 50%
      - Precision ≥ 70%      - Minor Drift         - Critical FP > 20%
                             Action:               - Audit / Hash Failure
                             - Predictions logged  Action:
                             - UI safely hidden    - Immediate deactivation
                                                   - Restore previous model
                                                   - Latency < 1.0s
```

---

## 7. Model Lifecycle API Contracts & Invariants

```
POST /api/models                     --> Register model (Status: REGISTERED)
POST /api/models/{modelId}/promote   --> Promote model (Fails with 412 if unvalidated)
POST /api/models/{modelId}/rollback  --> Emergency rollback to previous version
GET  /api/models/{modelId}/calibration --> Returns { ece: 0.034, brierScore: 0.12, samples: 1482 }
GET  /api/predictions                --> Returns active predictions only
POST /api/predictions/{id}/acknowledge --> Marks prediction reviewed
```

- **Single Active Model Invariant**: Exactly one registered model may have status `ACTIVE` at any given time.
- **Promotion Safety Gate**: `promoteModel()` enforces that model calibration metrics have been evaluated and satisfy $\text{ECE} \le 0.05$ and $\text{Brier} \le 0.15$. Unvalidated models fail promotion with HTTP 412 (`VALIDATION_REQUIRED`).

---

## 8. Release Gates (Sprint 6 Quality Standard)

- **G6.1**: Model registry operational with Single Active Model invariant enforced.
- **G6.2**: Predictions 100% traceable to source snapshot hashes and causal factor drivers.
- **G6.3**: Calibration ECE $\le 5\%$.
- **G6.4**: Brier Score $\le 0.15$.
- **G6.5**: Drift monitoring active with automated warning/critical triggers.
- **G6.6**: Prediction outcome repository 100% append-only and immutable.
- **G6.7**: Soft and Hard rollback verified with restoration time $< 1\text{s}$.
- **G6.8**: No unvalidated model can reach `ACTIVE` state.
- **G6.9**: Prediction UI adheres strictly to Anti-Cyan design system and human-first explanations.
- **G6.10**: Full regression suite (Sprints 1 through 6) 100% passing with zero failures.
