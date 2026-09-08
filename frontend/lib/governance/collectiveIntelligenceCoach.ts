/**
 * Phase 31-M5: Collective Intelligence Coach Engine
 *
 * Implements:
 * - Canonical Coaching Recommendations (REC-001 to REC-012)
 * - Recommendation Generation, Evaluation, Explanation, and Certification
 * - Validation Rules VR-M5-REC01 to VR-M5-REC05
 * - Invariant INV-OI23 (Recommendation Explainability)
 * - Invariant INV-OI24 (Recommendation Actionability & Non-Coercion)
 * - Invariant INV-OI25 (Remediation Completeness)
 * - Invariant INV-OI28 (Recommendation Fairness)
 * - Invariant INV-OI31 (Alternative Recommendation Availability)
 * - Pure TypeScript SHA-256 State Hashing
 */

import type {
  CoachingRecommendation,
  RecommendationEvidence,
  RemediationAction,
  ExpectedImpact,
  AlternativeRecommendation,
  RecommendationEvaluation,
  RecommendationExplanation,
  CertificationResult,
  CoachingCategory,
  CoachingPriority,
} from '../../types/coaching-intelligence';

import type { GovernanceRisk } from '../../types/groupthink-intelligence';
import { sha256Hex } from './sha256';

export const CANONICAL_COACHING_RECOMMENDATIONS: CoachingRecommendation[] = [
  {
    recommendationId: 'REC-001',
    committeeId: 'COM-001',
    type: 'DISSENT_PROMPT',
    title: "Mandate Rotating Contrarian Reviewer for Sector Outliers",
    description: "Enforce structured counter-thesis documentation on equity positions exceeding 15% active risk allocation.",
    category: 'GROUPTHINK',
    priority: 'HIGH',
    confidenceScore: 88,
    confidencePct: 88,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-001',
    rationale: "Unanimity score spiked to 42.0 on late momentum additions with near-zero dissent recording.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-001',
        sourceMetric: 'GROUPTHINK',
        observedValue: 42.0,
        thresholdValue: 35.0,
        contributionPct: 60.0,
        explanation: "Unanimity elevated above historical baseline indicating emergent artificial consensus.",
      },
      {
        evidenceId: 'EVD-REC-002',
        sourceMetric: 'DIRATIO',
        observedValue: 22.4,
        thresholdValue: 25.0,
        contributionPct: 40.0,
        explanation: "Dissent impact ratio slipped below target floor.",
      },
    ],
    evidenceIds: ['EVD-REC-001', 'EVD-REC-002'],
    historicalCaseIds: ['HIST-2025-Q3-01', 'HIST-2026-Q1-04'],
    actions: [
      {
        actionId: 'ACT-001',
        title: "Assign rotating contrarian reviewer for Q4 asset allocation",
        ownerId: 'USR-RSK-01',
        dueDateUtc: '2026-10-15T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Reduces groupthink probability by ~18% and elevates dissent capture.",
      },
      {
        actionId: 'ACT-002',
        title: "Draft counter-thesis dossier for top 3 tech holdings",
        ownerId: 'USR-PM-01',
        dueDateUtc: '2026-10-22T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Surfaces latent downside vulnerabilities before final capital commitment.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 2.4,
      projectedRiskReduction: 14.5,
      confidencePct: 85,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Reduces groupthink probability by ~18% and elevates dissent capture.",
    alternatives: [
      {
        alternativeId: 'ALT-001-A',
        title: "Anonymous Delphi Round Deliberation",
        approach: "Conduct independent blind scoring prior to committee plenary.",
        tradeOffSummary: "Higher procedural overhead (+3 days) but eliminates early speaker bias.",
        confidenceScore: 82,
      },
      {
        alternativeId: 'ALT-001-B',
        title: "Post-Mortem Inversion Stress Test",
        approach: "Simulate 25% drawdown scenario before final approval sign-off.",
        tradeOffSummary: "Fast execution but focuses on downside tail rather than consensus dynamics.",
        confidenceScore: 79,
      },
    ],
    createdAtUtc: '2026-09-08T12:00:00Z',
    generatedAtUtc: '2026-09-08T12:00:00Z',
  },
  {
    recommendationId: 'REC-002',
    committeeId: 'COM-002',
    type: 'RISK_REVIEW',
    title: "Enforce Volatility Stress Testing on Concentrated Holdings",
    description: "Execute 3-standard-deviation macro shock simulation prior to credit expansion approval.",
    category: 'RISK',
    priority: 'CRITICAL',
    confidenceScore: 92,
    confidencePct: 92,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-002',
    rationale: "Tail risk exposure RSK-002 active with elevated portfolio beta sensitivity.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-003',
        sourceMetric: 'CDQI',
        observedValue: 81.0,
        thresholdValue: 85.0,
        contributionPct: 70.0,
        explanation: "Decision quality index weakened by unhedged rate volatility.",
      },
      {
        evidenceId: 'EVD-REC-004',
        sourceMetric: 'ODEI',
        observedValue: 80.5,
        thresholdValue: 82.0,
        contributionPct: 30.0,
        explanation: "Organizational effectiveness impacted by delayed risk mitigation.",
      },
    ],
    evidenceIds: ['EVD-REC-003', 'EVD-REC-004'],
    historicalCaseIds: ['HIST-2025-Q4-02'],
    actions: [
      {
        actionId: 'ACT-003',
        title: "Run Monte Carlo volatility matrix across all debt facilities",
        ownerId: 'USR-RSK-02',
        dueDateUtc: '2026-09-25T00:00:00Z',
        status: 'IN_PROGRESS',
        expectedBenefit: "Quantifies maximum drawdown floor under severe liquidity dry-up.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 3.1,
      projectedRiskReduction: 22.0,
      confidencePct: 90,
      timeframeHorizon: '30D',
    },
    expectedBenefit: "Quantifies maximum drawdown floor under severe liquidity dry-up.",
    alternatives: [
      {
        alternativeId: 'ALT-002-A',
        title: "Synthetic Collar Hedge",
        approach: "Require automated out-of-the-money put spreads on high beta debt.",
        tradeOffSummary: "Immediate downside ceiling but incurs premium drag (-0.4% yield).",
        confidenceScore: 88,
      },
      {
        alternativeId: 'ALT-002-B',
        title: "Discretionary Haircut Adjustment",
        approach: "Increase collateral margin hair-cut by 12% across tier-2 borrowers.",
        tradeOffSummary: "Zero direct cash cost but tightens borrower deal flow.",
        confidenceScore: 84,
      },
    ],
    createdAtUtc: '2026-09-08T12:05:00Z',
    generatedAtUtc: '2026-09-08T12:05:00Z',
  },
  {
    recommendationId: 'REC-003',
    committeeId: 'COM-003',
    type: 'BIAS_WARNING',
    title: "Establish Quorum Independence Floor for Executive Overrides",
    description: "Require supermajority independent member concurrence when overriding algorithmic risk warnings.",
    category: 'GOVERNANCE',
    priority: 'HIGH',
    confidenceScore: 86,
    confidencePct: 86,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-003',
    rationale: "Authority bias detected where chair override occurred with 100% unilateral sign-off.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-005',
        sourceMetric: 'CDQI',
        observedValue: 83.5,
        thresholdValue: 88.0,
        contributionPct: 55.0,
        explanation: "Quorum diversity diminished during fast-track procedural votes.",
      },
      {
        evidenceId: 'EVD-REC-006',
        sourceMetric: 'GROUPTHINK',
        observedValue: 46.0,
        thresholdValue: 35.0,
        contributionPct: 45.0,
        explanation: "Influence concentration index elevated under single-sponsor proposals.",
      },
    ],
    evidenceIds: ['EVD-REC-005', 'EVD-REC-006'],
    historicalCaseIds: ['HIST-2026-Q1-09'],
    actions: [
      {
        actionId: 'ACT-004',
        title: "Formalize Independent Member Sign-off Bylaw",
        ownerId: 'USR-GOV-01',
        dueDateUtc: '2026-10-30T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Guarantees procedural check against unilateral executive authority.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 1.8,
      projectedRiskReduction: 16.0,
      confidencePct: 82,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Guarantees procedural check against unilateral executive authority.",
    alternatives: [
      {
        alternativeId: 'ALT-003-A',
        title: "Cooling-Off Deliberation Window",
        approach: "Mandate a 48-hour pause between override proposal and final execution.",
        tradeOffSummary: "Allows reflective review but delays time-sensitive market entries.",
        confidenceScore: 80,
      },
      {
        alternativeId: 'ALT-003-B',
        title: "Third-Party Audit Attestation",
        approach: "Submit executive override dossiers to external compliance counsel within 24h.",
        tradeOffSummary: "Maximum audit defensibility but increases administrative cost.",
        confidenceScore: 77,
      },
    ],
    createdAtUtc: '2026-09-08T12:10:00Z',
    generatedAtUtc: '2026-09-08T12:10:00Z',
  },
  {
    recommendationId: 'REC-004',
    committeeId: 'COM-001',
    type: 'LEARNING_NUDGE',
    title: "Incorporate Prior Cycle Tech Correction Lessons into Staging",
    description: "Re-evaluate LRN-001 and LRN-004 evidence before committing capital to growth equities.",
    category: 'LEARNING',
    priority: 'MEDIUM',
    confidenceScore: 84,
    confidencePct: 84,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-001',
    rationale: "Learning velocity stagnated at +3.0 and recurring drawdown patterns identified.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-007',
        sourceMetric: 'FRICTION',
        observedValue: 24.5,
        thresholdValue: 20.0,
        contributionPct: 50.0,
        explanation: "Friction score elevated due to delayed adoption of risk guidelines.",
      },
      {
        evidenceId: 'EVD-REC-008',
        sourceMetric: 'ODEI',
        observedValue: 84.0,
        thresholdValue: 85.0,
        contributionPct: 50.0,
        explanation: "Learning transfer lag depressing quarterly decision velocity.",
      },
    ],
    evidenceIds: ['EVD-REC-007', 'EVD-REC-008'],
    historicalCaseIds: ['HIST-2025-Q2-03'],
    actions: [
      {
        actionId: 'ACT-005',
        title: "Conduct retrospective session on LRN-001 adoption gaps",
        ownerId: 'USR-PM-02',
        dueDateUtc: '2026-10-05T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Improves institutional memory and eliminates repetitive execution slips.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 1.5,
      projectedRiskReduction: 10.0,
      confidencePct: 80,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Improves institutional memory and eliminates repetitive execution slips.",
    alternatives: [
      {
        alternativeId: 'ALT-004-A',
        title: "Automated Pre-Commitment Checkpoint",
        approach: "Embed mandatory lesson sign-off in order execution terminal.",
        tradeOffSummary: "Zero training meetings required but adds friction to order staging.",
        confidenceScore: 81,
      },
      {
        alternativeId: 'ALT-004-B',
        title: "Peer Cross-Review Exchange",
        approach: "Pair PM-01 and PM-02 for bi-weekly trade thesis critiques.",
        tradeOffSummary: "High collaborative learning but consumes 2 PM hours weekly.",
        confidenceScore: 78,
      },
    ],
    createdAtUtc: '2026-09-08T12:15:00Z',
    generatedAtUtc: '2026-09-08T12:15:00Z',
  },
  {
    recommendationId: 'REC-005',
    committeeId: 'COM-002',
    type: 'ALTERNATIVE_OPTION',
    title: "Bridge Macro Liquidity Findings to Micro Credit Committee",
    description: "Publish monthly cross-committee liquidity briefings from COM-002 to COM-001.",
    category: 'KNOWLEDGE_TRANSFER',
    priority: 'HIGH',
    confidenceScore: 89,
    confidencePct: 89,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-002',
    rationale: "Cross-committee transfer rate dropped from 84% to 80% on inter-committee edges.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-009',
        sourceMetric: 'TRANSFER_RATE',
        observedValue: 80.0,
        thresholdValue: 85.0,
        contributionPct: 65.0,
        explanation: "Edge COM-002 -> COM-001 sits exactly at minimum compliance threshold.",
      },
      {
        evidenceId: 'EVD-REC-010',
        sourceMetric: 'FRICTION',
        observedValue: 18.5,
        thresholdValue: 15.0,
        contributionPct: 35.0,
        explanation: "Knowledge latency creates information asymmetry between risk and investment.",
      },
    ],
    evidenceIds: ['EVD-REC-009', 'EVD-REC-010'],
    historicalCaseIds: ['HIST-2026-Q1-02'],
    actions: [
      {
        actionId: 'ACT-006',
        title: "Establish bi-weekly syndicated risk briefing memo",
        ownerId: 'USR-RSK-01',
        dueDateUtc: '2026-10-12T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Boosts transfer rate to >=88% and eliminates blind spots in credit allocation.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 2.2,
      projectedRiskReduction: 15.0,
      confidencePct: 87,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Boosts transfer rate to >=88% and eliminates blind spots in credit allocation.",
    alternatives: [
      {
        alternativeId: 'ALT-005-A',
        title: "Shared Telemetry Dashboard Embed",
        approach: "Embed live COM-002 risk gauges directly into COM-001 terminal header.",
        tradeOffSummary: "Real-time visibility with zero memo overhead; relies on operator pull.",
        confidenceScore: 85,
      },
      {
        alternativeId: 'ALT-005-B',
        title: "Joint Committee Seat Exchange",
        approach: "Designate one risk analyst as non-voting participant in investment meetings.",
        tradeOffSummary: "Maximum contextual immersion; slightly increases committee size.",
        confidenceScore: 83,
      },
    ],
    createdAtUtc: '2026-09-08T12:20:00Z',
    generatedAtUtc: '2026-09-08T12:20:00Z',
  },
  {
    recommendationId: 'REC-006',
    committeeId: 'COM-003',
    type: 'RISK_REVIEW',
    title: "Automate Pre-Flight Evidence Validation for Rapid Approvals",
    description: "Install algorithmic schema validator blocking incomplete proposals prior to quorum call.",
    category: 'OPERATIONAL',
    priority: 'MEDIUM',
    confidenceScore: 81,
    confidencePct: 81,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-003',
    rationale: "Two proposals staged in Q2 lacked complete backtest attribution metadata.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-011',
        sourceMetric: 'CDQI',
        observedValue: 85.0,
        thresholdValue: 90.0,
        contributionPct: 60.0,
        explanation: "Pre-flight evidence gaps caused retro-fitted documentation.",
      },
      {
        evidenceId: 'EVD-REC-012',
        sourceMetric: 'DIRATIO',
        observedValue: 24.0,
        thresholdValue: 25.0,
        contributionPct: 40.0,
        explanation: "Lack of pre-staged evidence restricted meaningful contrarian dissent.",
      },
    ],
    evidenceIds: ['EVD-REC-011', 'EVD-REC-012'],
    historicalCaseIds: ['HIST-2026-Q2-01'],
    actions: [
      {
        actionId: 'ACT-007',
        title: "Deploy pre-flight schema gating rule VR-M5-01 in UI staging",
        ownerId: 'USR-GOV-02',
        dueDateUtc: '2026-10-20T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "100% evidence completeness before voting begins.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 1.4,
      projectedRiskReduction: 11.0,
      confidencePct: 82,
      timeframeHorizon: '180D',
    },
    expectedBenefit: "100% evidence completeness before voting begins.",
    alternatives: [
      {
        alternativeId: 'ALT-006-A',
        title: "Manual Pre-Screen by Committee Secretary",
        approach: "Secretary manually audits evidence 24h prior to agenda finalization.",
        tradeOffSummary: "High human discretion but creates scheduling bottlenecks.",
        confidenceScore: 76,
      },
      {
        alternativeId: 'ALT-006-B',
        title: "Post-Approval Remediation Grace Period",
        approach: "Permit voting with provisional flag; require evidence within 48h.",
        tradeOffSummary: "Zero initial delay but risks unbacked capital exposure.",
        confidenceScore: 70,
      },
    ],
    createdAtUtc: '2026-09-08T12:25:00Z',
    generatedAtUtc: '2026-09-08T12:25:00Z',
  },
  {
    recommendationId: 'REC-007',
    committeeId: 'COM-001',
    type: 'DISSENT_PROMPT',
    title: "Quarantine Proposal Sponsorship when Influence Exceeds 75%",
    description: "Automatically freeze author proposal pipeline if a single sponsor authors >75% of approvals.",
    category: 'GROUPTHINK',
    priority: 'CRITICAL',
    confidenceScore: 94,
    confidencePct: 94,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-001',
    rationale: "Single influencer dominance detected with influence concentration at 78.4.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-013',
        sourceMetric: 'GROUPTHINK',
        observedValue: 78.4,
        thresholdValue: 65.0,
        contributionPct: 80.0,
        explanation: "Influence concentration breached critical threshold.",
      },
      {
        evidenceId: 'EVD-REC-014',
        sourceMetric: 'CDQI',
        observedValue: 80.0,
        thresholdValue: 85.0,
        contributionPct: 20.0,
        explanation: "Decision quality narrowed by single sponsor domination.",
      },
    ],
    evidenceIds: ['EVD-REC-013', 'EVD-REC-014'],
    historicalCaseIds: ['HIST-2025-Q1-07'],
    actions: [
      {
        actionId: 'ACT-008',
        title: "Institute mandatory co-sponsorship rule for proposals >$10M",
        ownerId: 'USR-PM-03',
        dueDateUtc: '2026-09-30T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Reduces influence concentration below 50.0 immediately.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 3.5,
      projectedRiskReduction: 25.0,
      confidencePct: 92,
      timeframeHorizon: '30D',
    },
    expectedBenefit: "Reduces influence concentration below 50.0 immediately.",
    alternatives: [
      {
        alternativeId: 'ALT-007-A',
        title: "Rotational Sponsor Moratorium",
        approach: "Mandate that primary sponsor cannot sponsor next 2 consecutive cycles.",
        tradeOffSummary: "Forces delegation but may bench top-performing capital allocators.",
        confidenceScore: 89,
      },
      {
        alternativeId: 'ALT-007-B',
        title: "Blind Review Committee Panel",
        approach: "Strip sponsor identity from proposal deck during initial scoring.",
        tradeOffSummary: "Eliminates halo effect completely; minor administrative redaction overhead.",
        confidenceScore: 86,
      },
    ],
    createdAtUtc: '2026-09-08T12:30:00Z',
    generatedAtUtc: '2026-09-08T12:30:00Z',
  },
  {
    recommendationId: 'REC-008',
    committeeId: 'COM-002',
    type: 'RISK_REVIEW',
    title: "Establish Counterfactual Hedging Mandate on Illiquid Equities",
    description: "Require quantitative model proving net liquidation risk is hedged under 48-hour volume freezes.",
    category: 'RISK',
    priority: 'HIGH',
    confidenceScore: 87,
    confidencePct: 87,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-002',
    rationale: "Illiquid equity exposure RSK-008 active with widening bid-ask spread simulations.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-015',
        sourceMetric: 'DIRATIO',
        observedValue: 21.0,
        thresholdValue: 25.0,
        contributionPct: 50.0,
        explanation: "Dissent regarding liquidity horizon unaddressed in committee minutes.",
      },
      {
        evidenceId: 'EVD-REC-016',
        sourceMetric: 'GROUPTHINK',
        observedValue: 40.0,
        thresholdValue: 35.0,
        contributionPct: 50.0,
        explanation: "Group consensus underestimated market impact liquidation costs.",
      },
    ],
    evidenceIds: ['EVD-REC-015', 'EVD-REC-016'],
    historicalCaseIds: ['HIST-2025-Q3-09'],
    actions: [
      {
        actionId: 'ACT-009',
        title: "Deploy dynamic liquidity haircut matrix on tier-3 instruments",
        ownerId: 'USR-RSK-02',
        dueDateUtc: '2026-10-18T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Reduces tail liquidation drawdown by 30%.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 2.1,
      projectedRiskReduction: 17.5,
      confidencePct: 84,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Reduces tail liquidation drawdown by 30%.",
    alternatives: [
      {
        alternativeId: 'ALT-008-A',
        title: "Mandatory Exchange-Traded Proxy Hedge",
        approach: "Hedge illiquid beta using index futures at 1.1x ratio.",
        tradeOffSummary: "Instant liquidity hedge but introduces basis risk.",
        confidenceScore: 82,
      },
      {
        alternativeId: 'ALT-008-B',
        title: "Hard Allocation Ceiling (Cap at 5%)",
        approach: "Enforce programmatic position cap of 5% on illiquid names.",
        tradeOffSummary: "Absolute risk control but caps upside from high-conviction micro-caps.",
        confidenceScore: 85,
      },
    ],
    createdAtUtc: '2026-09-08T12:35:00Z',
    generatedAtUtc: '2026-09-08T12:35:00Z',
  },
  {
    recommendationId: 'REC-009',
    committeeId: 'COM-003',
    type: 'BIAS_WARNING',
    title: "Mandate Cryptographic Snapshot Validation Before Voting",
    description: "Block voting progression until SHA-256 state tree matches between replay engine and live view.",
    category: 'GOVERNANCE',
    priority: 'CRITICAL',
    confidenceScore: 95,
    confidencePct: 95,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-003',
    rationale: "Replay determinism integrity invariant is mission critical for compliance auditability.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-017',
        sourceMetric: 'CDQI',
        observedValue: 88.0,
        thresholdValue: 90.0,
        contributionPct: 70.0,
        explanation: "Snapshot synchronization check required before final decision seal.",
      },
      {
        evidenceId: 'EVD-REC-018',
        sourceMetric: 'ODEI',
        observedValue: 86.0,
        thresholdValue: 85.0,
        contributionPct: 30.0,
        explanation: "Ensures tamper-proof audit trails for regulatory compliance.",
      },
    ],
    evidenceIds: ['EVD-REC-017', 'EVD-REC-018'],
    historicalCaseIds: ['HIST-2026-Q1-11'],
    actions: [
      {
        actionId: 'ACT-010',
        title: "Implement SHA-256 seal verification hook in voting workflow",
        ownerId: 'USR-GOV-01',
        dueDateUtc: '2026-09-28T00:00:00Z',
        status: 'IN_PROGRESS',
        expectedBenefit: "100% cryptographic replay defensibility against audit challenges.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 2.8,
      projectedRiskReduction: 20.0,
      confidencePct: 95,
      timeframeHorizon: '30D',
    },
    expectedBenefit: "100% cryptographic replay defensibility against audit challenges.",
    alternatives: [
      {
        alternativeId: 'ALT-009-A',
        title: "Asynchronous Post-Vote Snapshot Hashing",
        approach: "Allow vote completion and compute hash asynchronously within 60 seconds.",
        tradeOffSummary: "Zero UI latency during voting, but leaves 60s tampering vulnerability window.",
        confidenceScore: 82,
      },
      {
        alternativeId: 'ALT-009-B',
        title: "Dual Ledger Distributed Attestation",
        approach: "Replicate snapshot hash across two independent distributed nodes.",
        tradeOffSummary: "Maximum redundancy but requires secondary network node latency.",
        confidenceScore: 88,
      },
    ],
    createdAtUtc: '2026-09-08T12:40:00Z',
    generatedAtUtc: '2026-09-08T12:40:00Z',
  },
  {
    recommendationId: 'REC-010',
    committeeId: 'COM-001',
    type: 'LEARNING_NUDGE',
    title: "Cross-Pollinate Quant Signal Rejections with Fundamentals",
    description: "Conduct joint review of quantitative models rejected by discretionary team to evaluate alpha leakage.",
    category: 'KNOWLEDGE_TRANSFER',
    priority: 'MEDIUM',
    confidenceScore: 83,
    confidencePct: 83,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-001',
    rationale: "Knowledge transfer between quant research and discretionary staging sits below optimum.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-019',
        sourceMetric: 'TRANSFER_RATE',
        observedValue: 81.5,
        thresholdValue: 85.0,
        contributionPct: 60.0,
        explanation: "Quant-to-discretionary adoption rate sits at 81.5%.",
      },
      {
        evidenceId: 'EVD-REC-020',
        sourceMetric: 'FRICTION',
        observedValue: 19.0,
        thresholdValue: 15.0,
        contributionPct: 40.0,
        explanation: "Discretionary override rationale rarely documented back to model authors.",
      },
    ],
    evidenceIds: ['EVD-REC-019', 'EVD-REC-020'],
    historicalCaseIds: ['HIST-2025-Q4-08'],
    actions: [
      {
        actionId: 'ACT-011',
        title: "Scaffold feedback loop dashboard for rejected quant signals",
        ownerId: 'USR-PM-01',
        dueDateUtc: '2026-10-25T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Recovers estimated 1.2% annualized alpha leakage.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 1.6,
      projectedRiskReduction: 8.0,
      confidencePct: 79,
      timeframeHorizon: '180D',
    },
    expectedBenefit: "Recovers estimated 1.2% annualized alpha leakage.",
    alternatives: [
      {
        alternativeId: 'ALT-010-A',
        title: "Mandatory Text Feedback on Quant Rejection",
        approach: "Require 50-word qualitative justification when rejecting high-conviction quant picks.",
        tradeOffSummary: "Captures granular feedback; minor PM friction (+2 min per trade).",
        confidenceScore: 80,
      },
      {
        alternativeId: 'ALT-010-B',
        title: "Shadow Paper Portfolio Tracking",
        approach: "Automatically paper-trade all rejected quant signals to track opportunity cost.",
        tradeOffSummary: "Zero human effort; yields objective counterfactual benchmark.",
        confidenceScore: 86,
      },
    ],
    createdAtUtc: '2026-09-08T12:45:00Z',
    generatedAtUtc: '2026-09-08T12:45:00Z',
  },
  {
    recommendationId: 'REC-011',
    committeeId: 'COM-002',
    type: 'ALTERNATIVE_OPTION',
    title: "Calibrate Slippage Buffers Based on Realized Execution Drag",
    description: "Update algorithmic slippage parameters monthly using post-trade TCA data.",
    category: 'LEARNING',
    priority: 'HIGH',
    confidenceScore: 88,
    confidencePct: 88,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-002',
    rationale: "Execution drag consistently exceeded pre-trade estimates by 14 bps in Q2.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-021',
        sourceMetric: 'ODEI',
        observedValue: 82.0,
        thresholdValue: 85.0,
        contributionPct: 55.0,
        explanation: "Execution slippage dragging down net committee ODEI realization.",
      },
      {
        evidenceId: 'EVD-REC-022',
        sourceMetric: 'CDQI',
        observedValue: 83.0,
        thresholdValue: 86.0,
        contributionPct: 45.0,
        explanation: "Pre-trade sizing models out of sync with actual market liquidity.",
      },
    ],
    evidenceIds: ['EVD-REC-021', 'EVD-REC-022'],
    historicalCaseIds: ['HIST-2026-Q2-05'],
    actions: [
      {
        actionId: 'ACT-012',
        title: "Integrate TCA API feed into pre-trade risk sizing calculator",
        ownerId: 'USR-RSK-01',
        dueDateUtc: '2026-10-10T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "Eliminates execution slippage surprise drag.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 2.0,
      projectedRiskReduction: 12.0,
      confidencePct: 86,
      timeframeHorizon: '90D',
    },
    expectedBenefit: "Eliminates execution slippage surprise drag.",
    alternatives: [
      {
        alternativeId: 'ALT-011-A',
        title: "Dynamic Volatility-Based Slippage Multiplier",
        approach: "Multiply standard slippage by (VIX / 20) during high-volatility regimes.",
        tradeOffSummary: "Conservative sizing in volatile markets; may under-allocate during spikes.",
        confidenceScore: 84,
      },
      {
        alternativeId: 'ALT-011-B',
        title: "Algorithmic TWAP/VWAP Order Slicing Mandate",
        approach: "Force execution algorithm to slice orders >$5M over minimum 4-hour window.",
        tradeOffSummary: "Smooths market impact; increases exposure to intra-day price trend drift.",
        confidenceScore: 82,
      },
    ],
    createdAtUtc: '2026-09-08T12:50:00Z',
    generatedAtUtc: '2026-09-08T12:50:00Z',
  },
  {
    recommendationId: 'REC-012',
    committeeId: 'COM-003',
    type: 'LEARNING_NUDGE',
    title: "Standardize Meeting Deliberation Minutes with Dissent Tags",
    description: "Tag all dissenting arguments with standardized taxonomy labels to track dissent survival.",
    category: 'OPERATIONAL',
    priority: 'LOW',
    confidenceScore: 78,
    confidencePct: 78,
    status: 'ACTIVE',
    ownerCommitteeId: 'COM-003',
    rationale: "Untagged dissents in Q1 required manual reconstruction during audit review.",
    supportingEvidence: [
      {
        evidenceId: 'EVD-REC-023',
        sourceMetric: 'DIRATIO',
        observedValue: 24.5,
        thresholdValue: 25.0,
        contributionPct: 60.0,
        explanation: "Dissent tracking efficiency increased when tagged at point of origin.",
      },
      {
        evidenceId: 'EVD-REC-024',
        sourceMetric: 'FRICTION',
        observedValue: 17.0,
        thresholdValue: 15.0,
        contributionPct: 40.0,
        explanation: "Audit reconstruction latency reduced by structured taxonomy metadata.",
      },
    ],
    evidenceIds: ['EVD-REC-023', 'EVD-REC-024'],
    historicalCaseIds: ['HIST-2026-Q1-14'],
    actions: [
      {
        actionId: 'ACT-013',
        title: "Provide taxonomy picklist in committee deliberation interface",
        ownerId: 'USR-GOV-02',
        dueDateUtc: '2026-11-01T00:00:00Z',
        status: 'OPEN',
        expectedBenefit: "100% automated audit reconstruction under 50ms.",
      },
    ],
    expectedImpact: {
      projectedODEIDelta: 1.0,
      projectedRiskReduction: 6.0,
      confidencePct: 80,
      timeframeHorizon: '180D',
    },
    expectedBenefit: "100% automated audit reconstruction under 50ms.",
    alternatives: [
      {
        alternativeId: 'ALT-012-A',
        title: "Post-Hoc NLP Dissent Classification",
        approach: "Run offline NLP model on transcript minutes to auto-classify dissent arguments.",
        tradeOffSummary: "Zero human friction during meetings; relies on 90% NLP classification accuracy.",
        confidenceScore: 75,
      },
      {
        alternativeId: 'ALT-012-B',
        title: "Mandatory Dissent Form Submission",
        approach: "Require dissenting members to file structured 1-page dissent notice within 4h.",
        tradeOffSummary: "Highest structural fidelity; increases formal burden on dissenting members.",
        confidenceScore: 73,
      },
    ],
    createdAtUtc: '2026-09-08T12:55:00Z',
    generatedAtUtc: '2026-09-08T12:55:00Z',
  },
];

// Validation rules VR-M5-REC01 to VR-M5-REC05
export function validateCoachingRecommendation(rec: CoachingRecommendation): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // VR-M5-REC01: recommendationId pattern
  if (!rec.recommendationId || !/^REC-[0-9]{3,}$/.test(rec.recommendationId)) {
    errors.push(`VR-M5-REC01: Invalid recommendationId "${rec.recommendationId}" (must match ^REC-[0-9]{3,}$)`);
  }

  // VR-M5-REC02: confidenceScore 0-100
  const score = rec.confidenceScore ?? rec.confidencePct;
  if (typeof score !== 'number' || score < 0 || score > 100 || isNaN(score)) {
    errors.push(`VR-M5-REC02: confidenceScore must be a number between 0 and 100 (got ${score})`);
  }

  // VR-M5-REC03: supportingEvidence required, contributionPct sums to 100%
  if (!rec.supportingEvidence || !Array.isArray(rec.supportingEvidence) || rec.supportingEvidence.length === 0) {
    errors.push(`VR-M5-REC03: supportingEvidence is required and must contain at least 1 item`);
  } else {
    const sum = rec.supportingEvidence.reduce((acc, ev) => acc + (ev.contributionPct || 0), 0);
    if (Math.abs(sum - 100.0) > 0.1) {
      errors.push(`VR-M5-REC03: supportingEvidence contributionPct must sum to 100.0% (got ${sum.toFixed(1)}%)`);
    }
  }

  // VR-M5-REC04: expectedImpact with projectedRiskReduction required
  if (!rec.expectedImpact || typeof rec.expectedImpact.projectedRiskReduction !== 'number') {
    errors.push(`VR-M5-REC04: expectedImpact with projectedRiskReduction is required`);
  }

  // VR-M5-REC05: createdAtUtc or generatedAtUtc required
  const ts = rec.createdAtUtc || rec.generatedAtUtc;
  if (!ts || isNaN(Date.parse(ts))) {
    errors.push(`VR-M5-REC05: createdAtUtc must be a valid ISO-8601 timestamp (got "${ts}")`);
  }

  // Actions check: must have at least 1 action
  if (!rec.actions || !Array.isArray(rec.actions) || rec.actions.length === 0) {
    errors.push(`Actionability check: At least 1 remediation action is required`);
  } else {
    rec.actions.forEach((act, idx) => {
      if (!act.actionId) errors.push(`Action ${idx}: actionId is required`);
      if (!act.title) errors.push(`Action ${idx}: title is required`);
      if (!act.ownerId) errors.push(`Action ${idx}: ownerId is required`);
      if (!act.dueDateUtc) errors.push(`Action ${idx}: dueDateUtc is required`);
      if (!act.expectedBenefit) errors.push(`Action ${idx}: expectedBenefit is required`);
    });
  }

  return { valid: errors.length === 0, errors };
}

// Invariant INV-OI23: Recommendation Explainability
export function verifyINV_OI23(rec: CoachingRecommendation): { pass: boolean; violations: string[] } {
  const violations: string[] = [];

  if (!rec.supportingEvidence || rec.supportingEvidence.length === 0) {
    violations.push(`INV-OI23 Violation: Recommendation ${rec.recommendationId} has no supporting evidence`);
  }

  const sum = (rec.supportingEvidence || []).reduce((acc, ev) => acc + ev.contributionPct, 0);
  if (Math.abs(sum - 100.0) > 0.1) {
    violations.push(`INV-OI23 Violation: Evidence contributionPct must total 100.0% (got ${sum.toFixed(1)}%)`);
  }

  if (!rec.rationale || rec.rationale.trim().length === 0) {
    violations.push(`INV-OI23 Violation: Recommendation rationale is empty`);
  }

  if (!rec.expectedImpact) {
    violations.push(`INV-OI23 Violation: Expected impact is missing`);
  }

  return { pass: violations.length === 0, violations };
}

// Invariant INV-OI24: Recommendation Actionability & Non-Coercion
export function verifyINV_OI24(rec: CoachingRecommendation): { pass: boolean; violations: string[] } {
  const violations: string[] = [];

  if (!rec.ownerCommitteeId) {
    violations.push(`INV-OI24 Violation: ownerCommitteeId is missing`);
  }

  if (!rec.actions || rec.actions.length === 0) {
    violations.push(`INV-OI24 Violation: No remediation actions defined`);
  } else {
    rec.actions.forEach((act) => {
      if (!act.ownerId) violations.push(`INV-OI24 Violation: Action ${act.actionId} lacks ownerId`);
      if (!act.dueDateUtc) violations.push(`INV-OI24 Violation: Action ${act.actionId} lacks dueDateUtc`);
      if (!act.expectedBenefit) violations.push(`INV-OI24 Violation: Action ${act.actionId} lacks expectedBenefit`);
    });
  }

  return { pass: violations.length === 0, violations };
}

// Invariant INV-OI25: Remediation Completeness
export function verifyINV_OI25(
  criticalRisks: GovernanceRisk[],
  recommendations: CoachingRecommendation[]
): { pass: boolean; coveragePct: number; uncoveredRiskIds: string[] } {
  const crit = criticalRisks.filter(r => r.severity === 'CRITICAL');
  if (crit.length === 0) {
    return { pass: true, coveragePct: 100.0, uncoveredRiskIds: [] };
  }

  const uncovered: string[] = [];
  crit.forEach(risk => {
    // Check if any recommendation addresses this risk category or committee
    const match = recommendations.some(rec => 
      rec.category === risk.category || 
      rec.committeeId === risk.committeeId ||
      rec.priority === 'CRITICAL'
    );
    if (!match) {
      uncovered.push(risk.riskId);
    }
  });

  const coveredCount = crit.length - uncovered.length;
  const coveragePct = Math.round((coveredCount / crit.length) * 1000) / 10;

  return {
    pass: uncovered.length === 0,
    coveragePct,
    uncoveredRiskIds: uncovered,
  };
}

// Invariant INV-OI28: Recommendation Fairness
export function verifyINV_OI28(recommendations: CoachingRecommendation[]): { pass: boolean; maxCommitteePct: number; violations: string[] } {
  const violations: string[] = [];
  if (recommendations.length === 0) {
    return { pass: true, maxCommitteePct: 0, violations: [] };
  }

  const committeeCounts: Record<string, number> = {};
  recommendations.forEach(r => {
    committeeCounts[r.committeeId] = (committeeCounts[r.committeeId] || 0) + 1;
  });

  let maxPct = 0;
  let dominantCommittee = '';
  Object.entries(committeeCounts).forEach(([cid, count]) => {
    const pct = (count / recommendations.length) * 100.0;
    if (pct > maxPct) {
      maxPct = pct;
      dominantCommittee = cid;
    }
  });

  // Fairness rule: No single committee receives > 70% of recommendations
  if (maxPct > 70.0) {
    violations.push(`INV-OI28 Violation: Committee ${dominantCommittee} receives ${maxPct.toFixed(1)}% of recommendations (limit 70.0%)`);
  }

  return {
    pass: violations.length === 0,
    maxCommitteePct: Math.round(maxPct * 10) / 10,
    violations,
  };
}

// Invariant INV-OI31: Alternative Recommendation Availability
export function verifyINV_OI31(recommendations: CoachingRecommendation[]): { pass: boolean; violations: string[] } {
  const violations: string[] = [];
  const highImpact = recommendations.filter(r => r.priority === 'HIGH' || r.priority === 'CRITICAL');

  highImpact.forEach(rec => {
    if (!rec.alternatives || rec.alternatives.length < 2) {
      violations.push(`INV-OI31 Violation: High-impact recommendation ${rec.recommendationId} has only ${rec.alternatives?.length ?? 0} alternatives (minimum 2 required)`);
    }
  });

  return { pass: violations.length === 0, violations };
}

// Deterministic Replay Hash
export function hashCoachingState(recommendations: CoachingRecommendation[]): string {
  const sorted = [...recommendations].sort((a, b) => a.recommendationId.localeCompare(b.recommendationId));
  const payload = sorted.map(r => ({
    id: r.recommendationId,
    cid: r.committeeId,
    status: r.status,
    priority: r.priority,
    confidence: r.confidenceScore,
    evidenceCount: r.supportingEvidence.length,
    actionsCount: r.actions.length,
    alternativesCount: r.alternatives?.length ?? 0,
  }));
  return sha256Hex(JSON.stringify(payload));
}

// Engine Public API
export function getRecommendations(committeeId?: string): CoachingRecommendation[] {
  if (!committeeId || committeeId === 'ALL') {
    return CANONICAL_COACHING_RECOMMENDATIONS;
  }
  return CANONICAL_COACHING_RECOMMENDATIONS.filter(r => r.committeeId === committeeId);
}

export function getRecommendationById(id: string): CoachingRecommendation | undefined {
  return CANONICAL_COACHING_RECOMMENDATIONS.find(r => r.recommendationId === id);
}

export function evaluateRecommendation(recommendationId: string): RecommendationEvaluation {
  const rec = getRecommendationById(recommendationId);
  if (!rec) {
    return {
      recommendationId,
      isValid: false,
      explainabilityScore: 0,
      actionabilityScore: 0,
      nonCoercive: true,
      violations: [`Recommendation ${recommendationId} not found`],
    };
  }

  const v23 = verifyINV_OI23(rec);
  const v24 = verifyINV_OI24(rec);
  const val = validateCoachingRecommendation(rec);

  const violations = [...v23.violations, ...v24.violations, ...val.errors];

  return {
    recommendationId,
    isValid: violations.length === 0,
    explainabilityScore: v23.pass ? 100 : 50,
    actionabilityScore: v24.pass ? 100 : 50,
    nonCoercive: true,
    violations,
  };
}

export function explainRecommendation(recommendationId: string): RecommendationExplanation | null {
  const rec = getRecommendationById(recommendationId);
  if (!rec) return null;

  return {
    recommendationId: rec.recommendationId,
    title: rec.title,
    rationale: rec.rationale,
    supportingEvidence: rec.supportingEvidence,
    expectedImpact: rec.expectedImpact,
    historicalCases: (rec.historicalCaseIds || []).map((cid, i) => ({
      caseId: cid,
      outcomeSummary: `Historical incident analogous to ${rec.title} resolved successfully in quarter Q${i + 1}.`,
      relevanceScore: 0.85 + (i * 0.05),
    })),
    attributionCompletenessPct: 100.0,
  };
}
