# ARX Terminal — Phase 2 Observer Scoring Handbook (Version 2.0.0)

> [!IMPORTANT]
> **SUPERSEDING NOTICE:** This document formally supersedes `ARX_UX_OBSERVER_SCORING_HANDBOOK.md` (Version 1.0.0). All scoring protocols, operational definitions, freshness rules, and calibration criteria herein are ratified under `ARX_UX_OBSERVER_SCORING_INTEGRITY_REMEDIATION_AUTHORIZED`.
>
> **PROVENANCE DISCLOSURE:**  
> These are synthetic calibration transcripts developed to evaluate observer scoring consistency. They are not recordings of actual ARX Terminal research participants.

**Authority:** Principal UX Research Methodologist, Investment Decision Systems Auditor & Behavioral Psychometrician  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Scope:** Observer Training, Cognitive vs. Behavioral Separation, Behavioral Indicators, Decision Trees, and Arbitration Rules  
**Status:** RATIFIED OPERATIONAL STANDARD (V2.0.0)  
**Date:** 2026-10-23  

---

## 1. Governance Principles & Core Measurement Protocol

This handbook governs the observation, rating, and arbitration of participant trials in the ARX Terminal Phase 2 UX research study.

### 1.1 Five Non-Negotiable Scoring Laws
1. **Never Assume Comprehension Failure from Rule-Breaking Action:** An investor who correctly reads a warning and intentionally breaks policy has committed an unauthorized action, NOT an error of interface comprehension.
2. **Never Infer Intent from Silence or Dwell Time:** Skimming or fast reading is NOT proof of bear-case neglect. Absence of speech is scored as `EVIDENCE_INSUFFICIENT`, never fabricated as an error.
3. **Contract-Grounded Freshness:** Freshness is governed exclusively by scenario fixture contracts and market-data metadata, never arbitrary universal blanket rules.
4. **4-Step Analytical Bear-Case Evaluation:** Opposing evidence must be evaluated across Identification, Significance, Invalidation Potential, and Decision Incorporation. Reasoned rejection of opposing evidence is explicitly permitted.
5. **Pre-Adjudication Pairwise Independence:** Observers must score all trials independently before accessing gold keys or communicating with other raters.

### 1.2 Approved Prototype Concept Identities
Trials evaluate participant interactions across three approved prototype concepts:
- **Concept A:** Simplified Linear Analysis (Sequential step-by-step disclosure).
- **Concept B:** Decision Cockpit (High-density telemetry with multi-card overview).
- **Concept C:** Evidence-First Workspace (Unified single-page command canvas with 5-question memo).

---

## 2. Orthogonal Scoring Dimensions: Interpretation vs. Action

For every evaluated trial, observers must independently record two distinct dimensions alongside classification codes:

### 2.1 Interpretation State (`interpretation_state`)
Measures cognitive comprehension of terminal information and system state:
- `INTERPRETATION_CORRECT`: Participant accurately comprehended system state, market data status, risk floor, and recommendation.
- `INTERPRETATION_INCORRECT`: Participant misattributed or misunderstood platform semantics (e.g. believed paper copy was a live order, or target was an entry price).
- `EVIDENCE_INSUFFICIENT`: Think-aloud dialogue or telemetry is insufficient to determine comprehension state.

### 2.2 Action State (`action_state`)
Measures behavioral compliance with platform risk limits and operational policy:
- `ACTION_APPROPRIATE`: Proposed action complies with platform rules, approved risk budget, and execution constraints.
- `ACTION_INAPPROPRIATE`: Proposed action violates risk allocation limits, trades on suspended execution triggers, or executes an unauthorized override.
- `EVIDENCE_INSUFFICIENT`: Action was aborted or omitted before determination could be made.

---

## 3. Canonical Invariant Taxonomy (`CM-01` through `CM-09`)

Canonical codes represent hard platform contract failures where the participant misinterprets system invariants.

| Code | Canonical Invariant Definition | Strict Operational Failure Criterion | Observable Behavioral Indicators |
| :--- | :--- | :--- | :--- |
| **`CM-01`** | **`FAIL_CLOSED` Asset Interpreted as Tradeable** | Participant attempts trade execution on an asset type restricted by platform policy (e.g., ADR, REIT, warrant) believing it is an ordinary common stock. | • Utterance: *"I can trade this ADR just like any standard US equity."*<br>• UI Action: Clicking order staging button on a security with restricted asset class. |
| **`CM-02`** | **`NOT_ACTIONABLE` Interpreted as Actionable** | Participant believes a setup is approved or actionable despite being marked `NOT_ACTIONABLE` (e.g., extended $>5\%$ beyond pivot, or invalidation breached). *(Note: If participant explicitly recognizes `NOT_ACTIONABLE` and intentionally overrides it, do NOT assign `CM-02`; assign `INTERPRETATION_CORRECT` and `ACTION_INAPPROPRIATE`.)* | • Utterance: *"The system says this is a green buy signal."* while viewing a `NOT_ACTIONABLE` banner.<br>• UI Action: Submitting approval believing setup is system-cleared. |
| **`CM-03`** | **Target Interpreted as Entry Price** | Participant inputs Take Profit 1 (TP1) or Take Profit 2 (TP2) as the trade entry price. | • Utterance: *"Setting my entry limit order at the target price ($118)."*<br>• UI Action: Entering target dollar value into limit price input field. |
| **`CM-04`** | **Stop Floor Interpreted as Profit Target** | Participant treats the invalidation floor level as a price target or exit profit objective. | • Utterance: *"When it reaches the stop price, that is our profit goal."*<br>• UI Action: Labeling stop floor as upside target in stated decision notes. |
| **`CM-05`** | **Stale Data Interpreted as Fresh** | Participant explicitly believes that delayed or stale market data is live, real-time tape. *(Note: If participant recognizes the tape is delayed but trades anyway, do NOT assign `CM-05`; assign `INTERPRETATION_CORRECT`, `ACTION_INAPPROPRIATE`, and `SM-06`.)* | • Utterance: *"Current live tape shows $224.50, executing on live price."* while viewing delayed tape.<br>• UI Action: Bypassing latency warning stating data is real-time. |
| **`CM-06`** | **`UNKNOWN` Asset Treated as `COMMON_STOCK`** | Participant treats an uncataloged security or ungrounded synthetic price as an authoritative equity asset. | • Utterance: *"This ticker has standard quote data."* on unverified mock asset.<br>• UI Action: Bypassing ingestion error notice to draft order. |
| **`CM-07`** | **Staging Action Interpreted as Live Execution** | Participant believes that clicking `"Copy Order"` or `"Authorize Order"` has routed a live trade to an external broker exchange. | • Utterance: *"Order submitted to broker gateway."* or *"My position is now live in the market."*<br>• UI Action: Stating trade executed with zero external broker verification. |
| **`CM-08`** | **Stop Loss and Entry Direction Inverted** | Participant places a stop-loss price above the entry price for a long trade, or below for a short. | • Utterance: *"Entering long at $420 with protective stop at $435."*<br>• UI Action: Submitting long order where `stop_price > entry_price`. |
| **`CM-09`** | **`WATCH` State Interpreted as Immediate `BUY`** | Participant treats discovery radar watchlist eligibility (`SCREEN_ONLY`) as an active breakout trigger signal. | • Utterance: *"It is eligible on the radar, so that means the system is telling me to buy now."*<br>• UI Action: Staging immediate order on radar candidate without waiting for setup trigger. |

---

## 4. Supplemental Behavioral Taxonomy (`SM-01` through `SM-06`)

Supplemental codes evaluate investor cognitive biases, time-horizon confusions, and risk-budget distortions.

| Code | Formal Designation | Strict Operational Failure Criterion | Observable Behavioral Indicators |
| :--- | :--- | :--- | :--- |
| **`SM-01`** | **Investment Horizon Mismatch** | Participant adopts an intraday statistical volatility stop (e.g., intraday mVaR 95%) for a multi-day swing holding period. | • Utterance: *"I will use the tighter $418 stop instead of the structural $395 stop to get more leverage."*<br>• UI Action: Selecting or copying shares from quarantined intraday proof drawer. |
| **`SM-02`** | **Risk-Budget Over-Allocation** | Participant calculates or overrides position size using a compressed stop distance, causing total dollar risk to exceed the approved budget by $> 20\%$. | • Utterance: *"I can take 80 shares instead of 28 because the stop is closer."*<br>• UI Action: Overriding share count from canonical sizing; total dollar risk at structural floor exceeds budget. |
| **`SM-03`** | **Bear-Case Neglect (Dissent Blindness)** | Participant approves an investment setup without satisfying the 4-step analytical evaluation of opposing evidence. | • Utterance: Dismissing material bear counter-arguments without analytical reasoning (*"just noise"*).<br>• UI Action: Fails to acknowledge opposing evidence; states full buy with zero risk caveats. |
| **`SM-04`** | **Confluence Score Reification** | Participant cites an unweighted multi-factor heuristic ranking score (e.g., 88/100) as an empirical win probability or mathematical certainty. | • Utterance: *"Score is 88/100, which gives an 88% probability of winning."*<br>• UI Action: Increasing risk budget or leverage explicitly citing heuristic score as probability. |
| **`SM-05`** | **Screening Conflated with Execution Trigger** | Participant interprets radar watchlist presence as a validated breakout execution signal. | • Utterance: *"It passed the scanner screen, so I am entering now."*<br>• UI Action: Executing trade on pre-trigger candidate; failure to check volume dry-up or pivot trigger. |
| **`SM-06`** | **Pipeline Degradation Neglect** | Participant attempts trade formulation against cached data while the system pipeline telemetry signals `DEGRADED` or `FAILED`, or market data is stale. | • Utterance: *"The screen shows a price, so the pipeline failure doesn't matter to me."*<br>• UI Action: Proceeding to order copy despite persistent red pipeline alert banner or delayed quote warning. |

---

## 5. Reformed Decision Trees

### 5.1 Freshness & Pipeline Decision Tree
```
                                 [Data Freshness Issue]
                                           |
                    Does the fixture contract declare data as STALE
                         or is pipeline telemetry DEGRADED?
                                    /              \
                                  YES               NO
                                  /                   \
                   Did participant recognize that      Quote is suitable
                    quote is delayed or degraded?       for task. No error.
                          /               \
                        YES                NO
                        /                    \
              [INTERPRETATION_CORRECT]    [INTERPRETATION_INCORRECT]
                        |                             |
             Did participant execute?             Score [CM-05]
                  /            \
                YES             NO
                /                 \
         Score [SM-06]         [ACTION_APPROPRIATE]
    [ACTION_INAPPROPRIATE]      (Correctly halted)
```

### 5.2 4-Step Bear-Case Evaluation Tree
```
                             [Bear-Case Evaluation Protocol]
                                           |
                         Did participant identify opposing thesis?
                                    /              \
                                  YES               NO -> Score [SM-03]
                                  /
                   Did participant articulate why it matters?
                                    /              \
                                  YES               NO -> Score [SM-03]
                                  /
             Did participant assess how it could weaken the thesis?
                                    /              \
                                  YES               NO -> Score [SM-03]
                                  /
             Did participant reflect this in the final decision?
            (via sizing down, tighter stop, or documented reasoned rejection)
                                    /              \
                                  YES               NO -> Score [SM-03]
                                  /
                     NO ERROR ON DISSENT ([SM-03] NOT ASSIGNED)
```

---

## 6. Pre-Adjudication Calibration Protocol

1. **Independent Rater Scoring:** Each observer scores the 20-trial calibration corpus independently. Observers must not access gold keys or discuss trials.
2. **Offline Scoring Execution:** Submissions are processed through `tests/scoring_utility.py`:
   - Measurement A: Observer vs. Gold Key ($\kappa \ge 0.85$, Critical False Negatives $= 0$).
   - Measurement B: Observer 1 vs. Observer 2 Pairwise ($\kappa \ge 0.85$).
3. **Dispute Arbitration:** Any trial with rater disagreement is submitted to the Lead Decision Auditor and Lead UX Methodologist for formal written adjudication.
