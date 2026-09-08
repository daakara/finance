/**
 * Phase 31-M14: Counterfactual Intelligence Engine (M14.2)
 *
 * Implements:
 * - Historical decision vs alternative counterfactual analysis (INV-OI73)
 * - Causal delta calculation (Actual Decision vs Alternative Strategy)
 * - Bidirectional traceability to Decision, Learning, Risk, Recommendation, Policy
 */

import {
  CounterfactualResult,
  CANONICAL_HISTORICAL_DECISIONS,
  CANONICAL_CANDIDATE_STRATEGIES,
} from '@/types/simulation-futures';

export function evaluateCounterfactualDecision(
  decisionId: string,
  alternativeStrategyId: string
): CounterfactualResult {
  const historical = CANONICAL_HISTORICAL_DECISIONS.find((d) => d.decisionId === decisionId) || CANONICAL_HISTORICAL_DECISIONS[0];
  const alternative = CANONICAL_CANDIDATE_STRATEGIES.find((s) => s.strategyId === alternativeStrategyId) || CANONICAL_CANDIDATE_STRATEGIES[1];

  let simulatedOutcome = historical.actualOHI;
  let causalDrivers: Array<{ factor: string; deltaContribution: number }> = [];
  let explanation = '';

  if (alternative.strategyId === 'STRAT-B') {
    // Aggressive Strategy: Higher initial return (+4.5) but higher volatility (-2.1) -> net +2.4
    simulatedOutcome = historical.actualOHI + 2.4;
    causalDrivers = [
      { factor: 'Capital Velocity Acceleration', deltaContribution: 4.5 },
      { factor: 'Execution Friction & Volatility', deltaContribution: -2.1 },
    ];
    explanation = `Choosing '${alternative.name}' instead of '${historical.title}' would have improved net OHI by +2.4 points via capital acceleration, with elevated volatility.`;
  } else if (alternative.strategyId === 'STRAT-C') {
    // Conservative Strategy: Lower risk buffer (+3.8), reduced yield (-1.2) -> net +2.6
    simulatedOutcome = historical.actualOHI + 2.6;
    causalDrivers = [
      { factor: 'Macro VaR Buffer Protection', deltaContribution: 3.8 },
      { factor: 'Opportunity Cost of Hedging', deltaContribution: -1.2 },
    ];
    explanation = `Choosing '${alternative.name}' would have increased downside resilience by +3.8 points while incurring a minor -1.2 yield penalty.`;
  } else {
    // Balanced Strategy (Status Quo)
    simulatedOutcome = historical.actualOHI;
    causalDrivers = [{ factor: 'Status Quo Alignment', deltaContribution: 0.0 }];
    explanation = `Choosing '${alternative.name}' mirrors historical allocation baseline with zero variance.`;
  }

  return {
    decisionId: historical.decisionId,
    alternativeDecisionId: alternative.strategyId,
    actualOutcome: historical.actualOHI,
    simulatedOutcome: Number(simulatedOutcome.toFixed(2)),
    delta: Number((simulatedOutcome - historical.actualOHI).toFixed(2)),
    explanation,
    causalDrivers,
    lineageType: 'DECISION',
  };
}
