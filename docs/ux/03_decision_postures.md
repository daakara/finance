# ARX Decision Postures & Human Translation Schema

## 1. Internal Engine Postures vs. User-Facing Vocabulary

ARX separates internal state engine enums from human-centered UI vocabulary. ARX supports decision-making; it does not dictate trades.

```typescript
export type DecisionPosture =
  | "RESEARCH"      // Early stage: Evaluating company fundamentals & business model
  | "WATCH"         // Mid stage: Setup interesting, but waiting for trigger / pullback
  | "ACQUIRE"       // Entry stage: Price in actionable zone with favorable confluence
  | "HOLD"          // Management stage: Thesis intact, price above key trend support
  | "TRIM"          // Risk stage: Momentum fading or position over-extended
  | "EXIT_REVIEW"   // Invalidation stage: Key trend/fundamental floor breached
  | "AVOID";        // Negative stage: High balance sheet leverage or hostile macro
```

---

## 2. Human Translation Table

| Decision Posture | Human UI Label | Primary Tone | Rationale & Guidance | Available Actions |
| :--- | :--- | :--- | :--- | :--- |
| **`RESEARCH`** | **Evaluating Company** | Neutral / Informative | *"Company is under initial fundamental evaluation. No technical entry trigger is active."* | • Add to Watchlist<br/>• Read Fundamentals<br/>• Compare Peers |
| **`WATCH`** | **Wait for Trigger** | Caution / Prudent | *"Interesting setup, but price remains below key trend average. Do not rush—wait for confirmation."* | • Set Trigger Alert<br/>• Watch Support Level<br/>• Size Hypothetical Trade |
| **`ACQUIRE`** | **Actionable Setup** | Constructive / Precise | *"Multiple technical and fundamental factors align favorably within optimal risk/reward bounds."* | • Calculate Position Size<br/>• Set Hard Stop Loss<br/>• View Execution Ladder |
| **`HOLD`** | **Thesis Intact** | Reassuring / Disciplined | *"Core business fundamentals and price structure remain healthy. Continue holding."* | • Monitor Key Support<br/>• Check Portfolio VaR<br/>• Stress Test Macro |
| **`TRIM`** | **Consider Trimming** | Warning / Prudent | *"Price is over-extended or momentum is slowing. Consider securing partial gains to de-risk."* | • Trim 25%–50% Position<br/>• Move Stop to Breakeven<br/>• Rebalance Cash |
| **`EXIT_REVIEW`** | **Thesis Needs Review** | High Caution / Urgent | *"Key support or fundamental threshold has broken. The original setup thesis is invalidated."* | • Review Exit Strategy<br/>• Execute Stop Loss<br/>• Preserve Capital |
| **`AVOID`** | **Unfavorable Setup** | Defensive / Firm | *"Severe balance sheet strain, negative institutional flow, or adverse macro conditions. Stay away."* | • Find Alternatives<br/>• Review Risk Drivers |

---

## 3. The `UNKNOWN` Ownership Protocol

When user ownership is not established via client portfolio storage, ARX does not guess. The terminal presents an interactive 1-click posture anchor:

```
┌──────────────────────────────────────────────────────────────┐
│ What is your current relationship with this stock?            │
│ [ 🔍 I'm considering buying ]  [ 💼 I already own it ]  [ 📊 Just researching ] │
└──────────────────────────────────────────────────────────────┘
```

Selecting an option updates `OwnershipState` (`OWNED` | `NOT_OWNED` | `UNKNOWN`) in session state and immediately recalculates the active `DecisionPosture`.
