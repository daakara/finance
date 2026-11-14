# ARX Terminal — Learn Hub Human Writing Guide (Directive 025)

**Document ID:** `UX-GDL-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Scope:** Educational and Interface Copy across the ARX Platform  
**Authority:** Project Brain Editorial Panel & `skills/plain-writing`  
**Classification:** Canonical Editorial Style & Voice Guide  

---

## 1. Editorial Voice & Mission Statement

The ARX Learn Hub exists to provide transparent, mathematically sound financial education. It must read like an experienced institutional practitioner explaining concepts to an intelligent, critical colleague.

We reject both extremes of bad financial writing:
1. **The Promotional AI Hype:** Superficial buzzwords ("game-changing", "cutting-edge", "unlock the power"), exclamation points, and breathless promises of wealth.
2. **The Inaccessible Academic Wall:** Denser-than-necessary mathematical prose designed to intimidate rather than enlighten.

Our standard is **rigorous clarity**: direct sentences, concrete numbers, transparent limitations, and respectful pedagogy.

---

## 2. Core Human Writing Principles

### Rule 1: Concrete Over Abstract
Never describe a metric using abstract nouns alone. Ground every concept in dollars, percentage points, share counts, or time units.
- *Bad:* "Optimize portfolio exposure through dynamic position-sizing algorithms."
- *Good:* "If your account is \$50,000 and you cap risk at 1% (\$500) on a stock with a \$5 stop-loss, buy exactly 100 shares."

### Rule 2: Active Voice Over Passive Evasion
State clearly who or what is performing the action.
- *Bad:* "Profits may be realized as targets are approached by the asset."
- *Good:* "Traders scale out of a portion of the position as price reaches Target 1."

### Rule 3: Direct Sentence Structure & Rhythm
Avoid predictable three-part constructions ("It analyzes X, optimizes Y, and empowers Z"). Vary sentence length. Keep paragraphs to 2–4 sentences focusing on a single core point.

### Rule 4: Honest Discussion of Limitations
Every quantitative indicator has conditions under which it fails. State them explicitly.
- *Mandatory:* Mention the 45-day lag on Congressional filings.
- *Mandatory:* Clarify that Minervini VCP setups fail in choppy or bear markets.
- *Mandatory:* Explain that Cornish-Fisher VaR models historical distribution and cannot predict unprecedented systemic freezes.

### Rule 5: Teach Interpretation, Never Urge Action
ARX Terminal is an institutional intelligence tool, not an automated financial advisor.
- *Forbidden:* "Buy this stock now before it breaks out."
- *Required:* "Price has contracted into an optimal accumulation corridor with support at \$142. A close below \$138 invalidates the pattern."

---

## 3. The 8-Part Human Lesson Blueprint

When explaining any trading concept, setup, or mathematical model, follow this sequential structure:

```
1. What this means      → Plain-language definition (1-2 sentences).
2. Why it matters       → Real-world financial consequence.
3. How it works         → Underlying market mechanism or formula.
4. Concrete example     → Realistic scenario with exact numbers.
5. How to interpret it  → What high, low, or boundary readings mean.
6. Common mistakes      → Misinterpretations and false assumptions.
7. Where it appears     → Direct link to relevant ARX terminal workflow.
8. Key takeaway         → One memorable concluding sentence.
```

---

## 4. Banned Phrases & Human Replacements

| Banned AI Slop Phrase | Problem | Preferred Human Replacement |
| :--- | :--- | :--- |
| "Unlock the power of..." | Cliché marketing puffery | "Understand how [X] calculates..." |
| "Navigate the dynamic landscape..." | Vague corporate metaphor | "Trade through volatile market conditions..." |
| "Harness actionable insights..." | Empty buzzword salad | "Identify trade setups that meet defined risk rules..." |
| "In today's fast-paced markets..." | Generic filler intro | Delete entirely; start with the core thesis. |
| "Dive deep into..." | Overused AI transitional cliché | "Examine the three core components of..." |
| "A comprehensive, holistic approach..." | Redundant, uninformative | "Synthesizing fundamental, technical, and regulatory data..." |
| "Piotroski BS Detector" | Unprofessional colloquialism | "Piotroski Balance Sheet Accounting Quality Score" |
| "Where to Buy" | Misleading directive implication | "Optimal Accumulation Corridor / Buy Zone" |
| "Never blow up your account" | Sensationalist guarantee | "Limit single-trade capital loss to 1% of equity" |
| "Seamlessly integrate..." | Marketing jargon | "Connects directly to..." |

---

## 5. Domain-Specific Transformation Examples

### Example A: Risk Management & Position Sizing
- **Before (AI Slop):**  
  *"Unlock the power of sophisticated risk management by leveraging dynamic risk-reward insights to optimize your investment journey and ensure you never experience catastrophic drawdowns in today's unpredictable market landscape."*
- **After (Human Institutional Standard):**  
  *"The foundation of professional capital preservation is the 1% risk rule. On any single trade, the maximum allowable dollar loss should never exceed 1% of total portfolio equity. If you manage a \$50,000 account, your maximum risk budget per trade is \$500. Sizing is determined strictly by the distance to your technical stop-loss, not emotional conviction. If buying at \$100 with a stop at \$95 (\$5 risk per share), you purchase exactly 100 shares (\$10,000 position, risking \$500). If the trade fails, 99% of your capital remains intact to trade tomorrow."*

### Example B: Technical Analysis (Minervini VCP)
- **Before (AI Slop):**  
  *"Harness the revolutionary Volatility Contraction Pattern to seamlessly capture massive multi-bagger breakout setups before Wall Street even notices."*
- **After (Human Institutional Standard):**  
  *"Mark Minervini's Volatility Contraction Pattern (VCP) identifies the transition of shares from motivated sellers into institutional accumulators. During a multi-week base, price undergoes successive contraction waves, each narrower than the last (for example: an initial 18% pullback, followed by a 9% pullback, followed by a 4% consolidation). When volume dries up dramatically on the final contraction, overhead supply is exhausted. A breakout above the pivot on heavy volume signals that demand has overcome remaining resistance."*

### Example C: Smart Money & Congressional Forensics
- **Before (AI Slop):**  
  *"Copy high-conviction trades from Capitol Hill insiders and profit alongside Washington's elite lawmakers with our proprietary alignment algorithm."*
- **After (Human Institutional Standard):**  
  *"Under the STOCK Act of 2012, members of Congress must disclose securities transactions within 45 days. Because 45 days is a long latency in liquid equity markets, disclosures must be evaluated critically. ARX quantifies legislative alignment by evaluating whether a lawmaker sits on a committee with direct regulatory or budgetary jurisdiction over the company (such as Armed Services members purchasing defense contractors). Trades filed past statutory deadlines receive an exponential staleness penalty, as historical price action has frequently already absorbed the informational edge."*

---

## 6. Mathematical & Formula Formatting Guidelines

- Write LaTeX formulas cleanly using KaTeX standard notation.
- Always define every parameter immediately beneath the equation.
- Provide a plain-language summary of what the equation computes before presenting the notation.
- Ensure all inline variables are enclosed in backticks or inline LaTeX (`\sigma`, `\alpha`).
- Never print unparsed raw LaTeX code blocks or unreadable single-line strings.
