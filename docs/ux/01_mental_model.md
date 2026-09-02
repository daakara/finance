# ARX Mental Model: Cognitive Decision Support vs. Prediction

## 1. Core Philosophy

ARX is a **deterministic, evidence-based financial decision support engine**, not a black-box stock predictor or robo-advisor.

### 1.1 What ARX Does
- **Measures Empirical Confluence**: Aggregates multi-factor evidence across company financial health, price structure, institutional money flows, and macro regime tailwinds.
- **Contextualizes Findings**: Evaluates whether current data favors entering, holding, trimming, or avoiding risk based on the user's selected time horizon and ownership state.
- **Enforces Traceability**: Every conclusion links directly to an underlying metric, reference benchmark, timestamp, and explicit conditions that would alter the assessment.
- **Highlights Downside Invalidation**: Establishes unambiguous risk floors and invalidation thresholds before discussing profit goals.

### 1.2 What ARX Does NOT Do
- **Does NOT Predict the Future**: ARX never claims to know where a stock will trade tomorrow. High confluence means historical and quantitative odds are aligned, not that a positive outcome is guaranteed.
- **Does NOT Provide Prescriptive Financial Dictation**: ARX never commands a user to `BUY` or `SELL`. It states the thesis health, risk posture, and relevant trade-offs.
- **Does NOT Fabricate Precision**: If evidence is incomplete or stale, ARX explicitly emits `INSUFFICIENT_EVIDENCE` rather than guessing.

---

## 2. The Seven-Layer Semantic Pipeline

Financial conclusions must originate from the state engine, not from isolated React components. Every user interaction traverses the following dependency graph:

```
[ User Context ] (Intent, Ownership, Horizon, Experience Mode)
       ↓
[ Quant Evidence ] (Fundamentals, Price Action, Smart Money, Macro)
       ↓
[ Empirical Assessment ] (Favorable, Mixed, Unfavorable, Insufficient)
       ↓
[ Decision Posture ] (Research, Watch, Acquire, Hold, Trim, Exit Review, Avoid)
       ↓
[ Explanation Engine ] (What, Why, Evidence Proof, Risk, What Would Change Assessment)
       ↓
[ Adaptive Presentation ] (Guided / Standard / Advanced Lens)
       ↓
[ Available User Actions ] (Research, Size Position, Set Alert, Track Thesis)
```

---

## 3. Experience Mode as a Cognitive Lens

`ExperienceMode` (`GUIDED` · `STANDARD` · `ADVANCED`) is a **presentation density and explanation filter**, NOT a separate product journey.

| Mode | Cognitive Role | User Mindset | What the UI Emphasizes |
| :--- | :--- | :--- | :--- |
| **🟢 Guided** | **Interpretation & Education** | *"Tell me what this means."* | Plain-English summaries, intuitive health ratings, explicit risk boundaries, zero mathematical clutter. |
| **🔵 Standard** | **Decision Support** | *"Help me decide & execute."* | Balanced confluence scores, key support/resistance levels, risk-reward ratios, position sizing. |
| **🟣 Advanced** | **Investigation & Control** | *"Let me interrogate the models."* | Raw quantitative factors (ROIC, VaR 95%, RVOL, ATR, Beta), multi-panel charts, execution ladders, data provenance. |

---

## 4. Multi-Modal Accessibility & Honest Calibration

1. **Color-Blind Invariant**: Financial meaning is never communicated by color alone. Every indicator pairs color with an explicit semantic icon, label, and high-contrast border:
   - 🟢 `[ ▲ POSITIVE / FAVORABLE ]` (Emerald + Up Triangle)
   - 🔴 `[ ▼ NEGATIVE / INVALIDATED ]` (Rose + Down Triangle)
   - 🟡 `[ ◼ CAUTION / WATCH ]` (Amber + Square)
   - 🔵 `[ ◆ INTERACTIVE / FOCUS ]` (Cyan + Diamond)
2. **Epistemic Integrity**: The system refuses to display ungrounded statistical certainty (e.g. *"87% probability of rally"* or *"Strong Buy"*). It reports evidence quality (`HIGH`, `MEDIUM`, `LOW`) and observable metric divergence.
