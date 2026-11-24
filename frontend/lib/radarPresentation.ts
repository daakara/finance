import { ExecutionStatus } from '../types/decisionContract';

export function formatRadarExecutionStatus(status: ExecutionStatus): string {
  switch (status) {
    case 'DELISTED_SECURITY':
      return 'DELISTED / INACTIVE';
    case 'IN_BUY_ZONE':
      return 'NEAR SCREENING ZONE';
    case 'READY_TO_BUY':
      return 'NEAR SCREENING ZONE';
    case 'IN_BUY_ZONE_AWAITING_TRIGGER':
      return 'AWAITING TRIGGER';
    case 'APPROACHING_TARGET':
      return 'APPROACHING TARGET';
    case 'WAITING_PULLBACK':
      return 'PULLBACK PENDING';
    case 'STOPPED_OUT':
      return 'STOPPED OUT';
    case 'INSUFFICIENT_HISTORY':
      return 'INSUFFICIENT HISTORY';
    case 'UNVERIFIED_ASSET':
      return 'UNVERIFIED ASSET';
    case 'STALE_MARKET_DATA':
      return 'STALE MARKET DATA';
    default:
      return 'UNKNOWN';
  }
}

export type RadarScoreTier = 'HIGH_CONVICTION' | 'CONSTRUCTIVE' | 'SELECTIVE' | 'DEFENSIVE' | 'UNAVAILABLE';

export interface RadarScorePresentation {
  tier: RadarScoreTier;
  colorClass: string;
  badgeClass: string;
  label: string;
  tooltip: string;
}

export function resolveRadarScorePresentation(
  score: number | null,
  isDelisted: boolean,
  backendBadgeColor?: string | null
): RadarScorePresentation {
  if (isDelisted) {
    return {
      tier: 'DEFENSIVE',
      colorClass: 'text-rose-400',
      badgeClass: 'text-rose-400 bg-rose-950/60 border border-rose-800/60',
      label: score !== null ? String(score) : '—',
      tooltip: 'Delisted Security · Trading Suspended',
    };
  }
  if (score === null || !Number.isFinite(score) || score < 0) {
    return {
      tier: 'UNAVAILABLE',
      colorClass: 'text-[var(--text-muted)]',
      badgeClass: 'text-[var(--text-muted)] font-normal font-mono border border-[var(--border-subtle)] bg-[var(--surface-2)]',
      label: '—',
      tooltip: 'Confluence Score Unavailable · Insufficient Data',
    };
  }

  const displayScore = Math.round(score);

  // Exact canonical boundary alignment with ConfluenceEngine:
  // 80–100: High conviction (Emerald)
  // 65–79:  Constructive (Cyan)
  // 49–64:  Selective (Amber)
  // 0–48:   Defensive / capital preservation (Rose)
  // Ensures scores like 45/100 cannot be mistaken for an actionable buy recommendation.
  if (backendBadgeColor === 'emerald' || (score >= 80 && backendBadgeColor !== 'amber')) {
    return {
      tier: 'HIGH_CONVICTION',
      colorClass: 'text-emerald-400',
      badgeClass: 'text-emerald-400 bg-emerald-950/60 border border-emerald-800/60',
      label: String(displayScore),
      tooltip: `High Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'cyan' || (score >= 65 && score < 80)) {
    return {
      tier: 'CONSTRUCTIVE',
      colorClass: 'text-cyan-400',
      badgeClass: 'text-cyan-400 bg-cyan-950/60 border border-cyan-800/60',
      label: String(displayScore),
      tooltip: `Constructive Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'amber' || (score >= 49 && score < 65)) {
    return {
      tier: 'SELECTIVE',
      colorClass: 'text-amber-400',
      badgeClass: 'text-amber-400 bg-amber-950/60 border border-amber-800/60',
      label: String(displayScore),
      tooltip: `Selective Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  return {
    tier: 'DEFENSIVE',
    colorClass: 'text-rose-400',
    badgeClass: 'text-rose-400 bg-rose-950/60 border border-rose-800/60',
    label: String(displayScore),
    tooltip: `Defensive / Sub-Threshold Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
  };
}

export interface RadarCandidateInput {
  ticker: string;
  name?: string | null;
  sector?: string | null;
  vcpStage?: string | null;
  volumeDryUpPct?: number | null;
  rvol?: string | null;
  confluenceScore?: number | null;
  categories?: string[];
  executionStatus?: ExecutionStatus | string;
  opportunityExplanation?: string | null;
  catalyst?: string | null;
  riskRewardRatio?: number | null;
  verifiedPillarCount?: number | null;
  smartMoneyEvidenceState?: string | null;
}

export interface RadarLearnLink {
  label: string;
  href: string;
  tooltip: string;
  category: 'TECHNICAL' | 'FUNDAMENTAL' | 'SMART_MONEY' | 'RISK_GOVERNANCE';
}

export interface RadarCandidateRationale {
  headline: string;
  summary: string;
  technicalContext: string;
  fullNarrative: string;
  learnLinks: RadarLearnLink[];
  coverageText: string;
}

export function resolveRadarCandidateCoverage(pillarCount: unknown): string {
  if (typeof pillarCount === 'number' && Number.isInteger(pillarCount) && pillarCount >= 0 && pillarCount <= 4) {
    return `${pillarCount}/4 Pillars Verified`;
  }
  return 'Coverage: Not Checked';
}

export function resolveRadarCandidateRationale(candidate: RadarCandidateInput): RadarCandidateRationale {
  const ticker = candidate.ticker || 'UNKNOWN';
  const execStatus = String(candidate.executionStatus || 'UNKNOWN');
  const sector = candidate.sector?.trim() || null;
  const sectorText = sector ? ` within the ${sector} sector` : '';
  const categories = Array.isArray(candidate.categories) ? candidate.categories : [];
  const vcpStage = candidate.vcpStage?.trim() || null;
  const stageDisplay = vcpStage || 'Pattern Unassigned';

  const coverageText = resolveRadarCandidateCoverage(candidate.verifiedPillarCount);

  // 1. Delisted / Ceased Trading
  if (execStatus === 'DELISTED_SECURITY') {
    const summary = 'Trading suspended for delisted security.';
    const technicalContext = 'Historical price and volume metrics preserved for point-in-time audit integrity. Active execution prohibited.';
    return {
      headline: 'Trading Suspended',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Audit Provenance',
          href: '/guide#module-foundations',
          tooltip: 'Learn how ARX preserves immutable historical baselines and corporate action lineage',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 2. Unverified Market Data / Feed Missing
  if (execStatus === 'UNVERIFIED_ASSET') {
    const summary = `Primary market data feed and trade tape unverified for ${ticker}.`;
    const technicalContext = 'Execution levels and position sizing suppressed until authoritative exchange feed clearance.';
    return {
      headline: 'Market Feed Unverified',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Data Governance',
          href: '/guide#module-foundations',
          tooltip: 'Learn about primary market data verification requirements',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 3. Insufficient History Depth (matches applicable backend contract without universal 250-session claim)
  if (execStatus === 'INSUFFICIENT_HISTORY') {
    const summary = `Historical bar depth insufficient for quantitative modeling on ${ticker}.`;
    const technicalContext = 'Candidate bar count does not meet minimum lookback requirements. Execution parameters and discovery signals suppressed pending sufficient history.';
    return {
      headline: 'Insufficient History Depth',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Data Governance',
          href: '/guide#module-foundations',
          tooltip: 'Learn about minimum historical session requirements for quantitative models',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 4. Stale Market Data / Session Drift
  if (execStatus === 'STALE_MARKET_DATA') {
    const summary = `Market data timestamps for ${ticker} precede the authoritative completed session cutoff.`;
    const technicalContext = 'Signals suppressed pending exchange refresh to prevent stale-session decision bias.';
    return {
      headline: 'Stale Market Data',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Market Freshness',
          href: '/guide#module-foundations',
          tooltip: 'Learn about market data recency and session cutoff verification',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  const hasVcp = categories.includes('VCP');
  const hasGarp = categories.includes('VALUE_GARP');
  const hasSm = categories.includes('SMART_MONEY');

  const dryUpText =
    typeof candidate.volumeDryUpPct === 'number' && candidate.volumeDryUpPct > 0
      ? ` with ${candidate.volumeDryUpPct}% volume dry-up`
      : '';
  const rvolText =
    candidate.rvol && candidate.rvol !== 'N/A' && candidate.rvol !== '—'
      ? ` on ${candidate.rvol} relative volume`
      : '';

  let headline: string;
  let summary: string;
  let technicalContext: string;

  const explicitExplanation =
    candidate.opportunityExplanation &&
    candidate.opportunityExplanation.trim().length > 0 &&
    candidate.opportunityExplanation !== 'Catalyst disclosures unavailable.'
      ? candidate.opportunityExplanation.trim()
      : null;

  if (explicitExplanation) {
    headline = `${ticker} · Discovery Candidate`;
    summary = explicitExplanation;
    technicalContext = 'Sourced directly from quantitative screening evaluation. Single-asset trigger clearance required.';
  } else if (hasVcp && hasGarp) {
    headline = `${ticker} · VCP & Value GARP Screener Qualification`;
    summary = `${ticker} screened under Volatility Contraction (${stageDisplay}${dryUpText}) and Value GARP criteria${sectorText}.`;
    technicalContext = 'Screener qualification reflects rule-based filter screening; single-asset DecisionTrace clearance required for execution parameters; discovery ranking is not an automatic buy signal.';
  } else if (hasVcp && hasSm) {
    headline = `${ticker} · VCP & Smart Money Screener Qualification`;
    summary = `${ticker} qualifies under VCP screening criteria (${stageDisplay}${dryUpText}) and Smart Money screener snapshot; Form 4 filing transactions unverified${rvolText}${sectorText}.`;
    technicalContext = 'Screener qualification does not establish institutional accumulation or verified order flow. Independent single-asset trigger clearance required.';
  } else if (hasVcp) {
    headline = `${ticker} · Volatility Contraction Pattern Observation`;
    summary = `${ticker} qualifies under VCP screening criteria (${stageDisplay}${dryUpText})${sectorText}.`;
    technicalContext = 'Pattern qualification reflects technical screening filters; does not establish institutional accumulation or supply absorption. Execution requires single-asset DecisionTrace clearance.';
  } else if (hasGarp) {
    headline = `${ticker} · Value GARP Screener Qualification`;
    summary = `${ticker} qualifies under Value GARP screening criteria${sectorText ? ` within ${sector}` : ''}.`;
    technicalContext = 'Category membership reflects fundamental metric filtering; single-asset verification required to evaluate balance sheet resilience.';
  } else if (hasSm) {
    headline = `${ticker} · Smart Money Observation`;
    summary = `${ticker} classified under Smart Money screener snapshot; Form 4 filing transactions unverified${rvolText}${sectorText}.`;
    technicalContext = 'Prospective cohort classification based on screener criteria. Categorization alone does not establish institutional accumulation or insider ownership.';
  } else {
    headline = `${ticker} · Multi-Factor Discovery Candidate`;
    summary = `${ticker} identified through multi-pillar screening${sectorText ? ` in ${sector}` : ''} (${stageDisplay}).`;
    technicalContext = 'Identified in discovery universe screening; single-asset validation required for execution parameters.';
  }

  // Compile Contextual Learn Hub & Glossary Deep-Links
  const learnLinks: RadarLearnLink[] = [];

  // Always include Confluence ranking reference
  learnLinks.push({
    label: 'Confluence Ranking',
    href: '/glossary#arx-model',
    tooltip: 'ARX Econometric Model and Confluence Score calibration (relative ranking != buy recommendation)',
    category: 'RISK_GOVERNANCE',
  });

  const stageLower = vcpStage ? vcpStage.toLowerCase() : '';
  if (hasVcp || stageLower.includes('vcp') || stageLower.includes('stage 2') || stageLower.includes('breakout')) {
    learnLinks.push({
      label: 'VCP Mechanics',
      href: '/guide#module-technical',
      tooltip: 'Field Manual: Minervini Stage 2 Trend Template and contraction wave geometry',
      category: 'TECHNICAL',
    });
    learnLinks.push({
      label: 'Glossary: VCP',
      href: '/glossary#minervini-vcp',
      tooltip: 'Quantitative Dictionary: Mark Minervini Volatility Contraction Pattern formula',
      category: 'TECHNICAL',
    });
  }

  if (hasGarp) {
    learnLinks.push({
      label: 'Fundamental Module',
      href: '/guide#module-fundamental',
      tooltip: 'Field Manual: Piotroski F-Score and fundamental quality screening',
      category: 'FUNDAMENTAL',
    });
    learnLinks.push({
      label: 'Piotroski F-Score',
      href: '/glossary#piotroski-f-score',
      tooltip: 'Quantitative Dictionary: 9-point fundamental health framework',
      category: 'FUNDAMENTAL',
    });
  }

  if (hasSm) {
    learnLinks.push({
      label: 'Smart Money Forensics',
      href: '/guide#module-smart-money',
      tooltip: 'Field Manual: SEC Form 4 forensics and insider order flow tracking',
      category: 'SMART_MONEY',
    });
    learnLinks.push({
      label: 'STOCK Act Decays',
      href: '/glossary#stock-act',
      tooltip: 'Quantitative Dictionary: Statutory 45-day reporting decay curves',
      category: 'SMART_MONEY',
    });
  }

  if (candidate.riskRewardRatio && candidate.riskRewardRatio >= 2) {
    learnLinks.push({
      label: 'M-VaR Risk Fencing',
      href: '/glossary#cornish-fisher-var',
      tooltip: 'Cornish-Fisher Modified VaR and risk floor calibration',
      category: 'RISK_GOVERNANCE',
    });
  }

  return {
    headline,
    summary,
    technicalContext,
    fullNarrative: `${summary} ${technicalContext}`,
    learnLinks,
    coverageText,
  };
}

export interface RadarEmptyStateCopy {
  title: string;
  description: string;
  reasonCode: string | null;
  recoveryCondition: string | null;
}

export function resolveRadarEmptyStateCopy(params: {
  activeFilter: 'ALL' | 'VALUE_GARP' | 'VCP' | 'SMART_MONEY';
  searchQuery: string;
  isTickerQuery: boolean;
  cleanQ: string;
  ownershipFilter: string;
  isPortfolioVerified: boolean;
  allAssetsCount: number;
  universeScopeLabel?: string;
  referenceUniverseCount?: number | null;
  capabilities: {
    VCP?: {
      status?: string;
      matchedCount?: number | null;
      populationDenominator?: number | null;
      universeBuildId?: string | null;
      publicationDigest?: string | null;
      lifecycleState?: string;
      freshnessStatus?: string;
      isRollback?: boolean;
      reason?: string | null;
      recoveryCondition?: string | null;
      dataAsOf?: string;
    };
    SMART_MONEY?: {
      status?: string;
      matchedCount?: number | null;
      populationDenominator?: number | null;
      universeBuildId?: string | null;
      publicationDigest?: string | null;
      lifecycleState?: string;
      freshnessStatus?: string;
      reason?: string | null;
      recoveryCondition?: string | null;
      dataAsOf?: string;
    };
  };
  vcpStatus: string;
  vcpCount: number | null;
  smStatus: string;
  smCount: number | null;
}): RadarEmptyStateCopy {
  const {
    activeFilter,
    searchQuery,
    isTickerQuery,
    cleanQ,
    ownershipFilter,
    isPortfolioVerified,
    allAssetsCount,
    universeScopeLabel,
    referenceUniverseCount,
    capabilities,
    vcpStatus,
    vcpCount,
    smStatus,
    smCount,
  } = params;

  if (searchQuery) {
    return {
      title: `No assets match "${searchQuery}" in current confluence scan`,
      description: isTickerQuery
        ? `Asset "${cleanQ}" is not in the active published universe (${allAssetsCount > 0 ? `${allAssetsCount} assets evaluated` : (universeScopeLabel || 'ARX_US_EQUITIES')}). You can scan it live across the exchange tape or review its tactical setup.`
        : 'No confluence candidates match this search. Try searching by symbol (e.g. NVDA, CPRX, ASML, TSLA) or reset the filter.',
      reasonCode: null,
      recoveryCondition: null,
    };
  }

  if (ownershipFilter === 'MY_HOLDINGS') {
    return {
      title: !isPortfolioVerified
        ? 'Portfolio sync unavailable (Ownership UNKNOWN)'
        : 'No active Radar candidates currently held in portfolio',
      description: !isPortfolioVerified
        ? 'The authoritative portfolio connection could not be verified. Radar renders canonically with ownership state UNKNOWN.'
        : "None of your current portfolio positions meet the active Radar screening confluence criteria. Switch to 'All Candidates' or 'Unheld Candidates' to view market setups.",
      reasonCode: !isPortfolioVerified ? 'PORTFOLIO_OWNERSHIP_UNKNOWN' : null,
      recoveryCondition: !isPortfolioVerified
        ? 'Reconnect portfolio state or clear the portfolio filter to inspect market-wide candidates.'
        : null,
    };
  }

  if (ownershipFilter === 'NEW_OPPORTUNITIES' && !isPortfolioVerified) {
    return {
      title: 'Portfolio sync unavailable (Ownership UNKNOWN)',
      description: 'Filtering for unheld candidates requires verified portfolio state. Currently running in degraded UNKNOWN mode.',
      reasonCode: 'PORTFOLIO_OWNERSHIP_UNKNOWN',
      recoveryCondition: 'Reconnect portfolio state or switch to All Candidates.',
    };
  }

  if (activeFilter === 'VCP') {
    const vcpCap = capabilities.VCP;
    if (vcpStatus === 'ERROR') {
      const reason = vcpCap?.reason || 'SNAPSHOT_ERROR';
      const isEmptyUniv = reason === 'BUILD_MEMBERSHIP_EMPTY';
      return {
        title: isEmptyUniv
          ? 'Minervini VCP — Universe Membership Empty'
          : 'Minervini VCP — Scanner Snapshot Error',
        description: isEmptyUniv
          ? 'The scannable universe membership is currently empty (BUILD_MEMBERSHIP_EMPTY). Automated screening is withheld pending authorized universe publication. Single-asset VCP geometry is available in Analysis (/) and Setups (/setups).'
          : `Scanner snapshot returned an error (${reason}). Missing or failed data is never represented as a zero-result scan. Single-asset VCP geometry remains available in Analysis (/) and Setups (/setups).`,
        reasonCode: reason,
        recoveryCondition:
          vcpCap?.recoveryCondition ||
          'Restore scanner snapshot store connectivity and verify active snapshot lookup succeeds.',
      };
    }

    if (vcpStatus === 'PIPELINE_PENDING') {
      const reason =
        vcpCap?.reason ||
        (vcpCap?.isRollback
          ? 'ROLLED_BACK_PUBLICATION_POINTER'
          : vcpCap?.freshnessStatus === 'STALE'
          ? 'STALE_SNAPSHOT'
          : 'NO_ACTIVE_SNAPSHOT');
      const title = vcpCap?.isRollback
        ? 'Minervini VCP — Publication Rolled Back (Scanner Suspended)'
        : vcpCap?.freshnessStatus === 'STALE'
        ? 'Minervini VCP — Historical Snapshot Stale (Pending Refresh)'
        : 'Minervini VCP — Universe Scanner Pending';
      const description = vcpCap?.isRollback
        ? `Universe publication ${vcpCap.universeBuildId || ''} was rolled back. Automated scanner execution is suspended.`
        : vcpCap?.freshnessStatus === 'STALE'
        ? `Historical VCP snapshot${vcpCap.dataAsOf ? ` (${vcpCap.dataAsOf})` : ''} is stale and cannot certify current scanner availability.`
        : vcpCap?.populationDenominator && vcpCap?.universeBuildId && vcpCap?.publicationDigest
        ? `${vcpCap.populationDenominator} securities registered in universe build ${vcpCap.universeBuildId}. Automated market-wide volatility contraction screening is pending operational activation.`
        : vcpCap?.lifecycleState === 'COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED'
        ? 'Automated market-wide volatility contraction screening is not scheduled. Single-asset VCP geometry is available in Analysis (/) and Setups (/setups).'
        : 'Automated market-wide volatility contraction screening is not active yet. Single-asset VCP geometry is still available in the Analysis (/) and Setups (/setups) hubs.';
      return {
        title,
        description,
        reasonCode: reason,
        recoveryCondition:
          vcpCap?.recoveryCondition ||
          'Commit an authoritative, verified VCP publication snapshot under the active release SHA via an authorized batch scan.',
      };
    }

    if (vcpStatus === 'AVAILABLE' && vcpCount === 0) {
      const evalCount =
        typeof vcpCap?.populationDenominator === 'number' && vcpCap.populationDenominator > 0
          ? vcpCap.populationDenominator
          : typeof referenceUniverseCount === 'number' && referenceUniverseCount > 0
          ? referenceUniverseCount
          : null;
      const evalLabel = evalCount !== null ? `${evalCount} Evaluated` : 'Published Universe Evaluated';
      const scopePhrase = evalCount !== null ? `all ${evalCount} published equities` : 'all published equities';
      return {
        title: `Minervini VCP — Zero Matching Contraction Setups (${evalLabel})`,
        description: `Across ${scopePhrase} evaluated for the current market session, zero candidates satisfied all 10 normative Minervini VCP criteria. Stage 2 trend qualification does not imply confirmed volatility contraction geometry. Single-asset VCP geometry remains available in Analysis (/) and Setups (/setups).`,
        reasonCode: null,
        recoveryCondition: null,
      };
    }
  }

  if (activeFilter === 'SMART_MONEY') {
    const smCap = capabilities.SMART_MONEY;
    if (smStatus === 'ERROR') {
      const reason = smCap?.reason || 'STORAGE_ERROR';
      return {
        title: 'Smart Money — Scanner Snapshot Error',
        description: `Smart Money snapshot retrieval failed (${reason}). Failed snapshot reads are never displayed as zero matches. Single-asset insider and congressional filing analysis remains available on /smart-money.`,
        reasonCode: reason,
        recoveryCondition:
          smCap?.recoveryCondition ||
          'Restore Smart Money snapshot store connectivity and verify active snapshot lookup succeeds.',
      };
    }

    if (smStatus === 'PIPELINE_PENDING') {
      const reason =
        smCap?.reason ||
        (smCap?.freshnessStatus === 'STALE' ? 'STALE_SNAPSHOT' : 'NO_ACTIVE_SMART_MONEY_SNAPSHOT');
      const title =
        smCap?.freshnessStatus === 'STALE'
          ? 'Smart Money — Historical Snapshot Stale (Pending Refresh)'
          : 'Smart Money — Universe Scanner Pending';
      const description =
        smCap?.freshnessStatus === 'STALE'
          ? `Historical Smart Money Form 4 snapshot${smCap.dataAsOf ? ` (${smCap.dataAsOf})` : ''} is stale and cannot certify current availability.`
          : reason === 'POSTGRES_READ_DISABLED'
          ? 'Smart Money Postgres snapshot reads are disabled by operational read control. Single-asset Form 4 analysis remains available on /smart-money.'
          : smCap?.populationDenominator && smCap?.universeBuildId && smCap?.publicationDigest
          ? `${smCap.populationDenominator} securities registered in universe build ${smCap.universeBuildId}. Automated market-wide institutional-flow screening is pending operational activation.`
          : 'Automated market-wide institutional-flow screening is not active yet. Single-asset insider and congressional filing analysis is still available for individual symbols on /smart-money.';
      return {
        title,
        description,
        reasonCode: reason,
        recoveryCondition:
          smCap?.recoveryCondition ||
          'Publish a verified Smart Money Form 4 snapshot once SEC ingestion holds and authorization gates are satisfied.',
      };
    }

    if (smStatus === 'AVAILABLE' && smCount === 0) {
      return {
        title: 'Smart Money — Zero Stage 2 Matches Evaluated',
        description:
          'Active Smart Money Form 4 snapshot evaluated cleanly with zero qualifying Stage 2 candidates for this filter view.',
        reasonCode: null,
        recoveryCondition: null,
      };
    }
  }

  return {
    title: `No qualifying candidates currently in "${activeFilter === 'VALUE_GARP' ? 'Value / GARP' : activeFilter === 'ALL' ? 'All Confluences' : activeFilter}"`,
    description:
      activeFilter === 'ALL'
        ? 'Try clearing your search query or refreshing the tape to discover active confluence setups.'
        : 'Try switching to "All Confluences" to view all active setups.',
    reasonCode: null,
    recoveryCondition: null,
  };
}

