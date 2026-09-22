"""ARX Canonical Decision Context and Evidence Contract Foundation.

Phase 0/1 Architecture Remediation:
Defines the canonical, fail-closed contracts for cross-engine evidence encapsulation,
data completeness tracking, and authoritative decision evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceQualityState(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    PROVISIONAL = "PROVISIONAL"
    FALLBACK = "FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


class EvidenceDomain(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    FUNDAMENTALS = "FUNDAMENTALS"
    MACRO = "MACRO"
    LIQUIDITY = "LIQUIDITY"
    ORDER_FLOW = "ORDER_FLOW"
    DERIVATIVES = "DERIVATIVES"


class TimeHorizon(str, Enum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITION = "POSITION"
    LONG_TERM = "LONG_TERM"


class UserRole(str, Enum):
    DAY_TRADER = "DAY_TRADER"
    SWING_TRADER = "SWING_TRADER"
    LONG_TERM = "LONG_TERM"


class DataCompleteness(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    DEGRADED = "DEGRADED"


class DecisionState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STALE_DATA = "STALE_DATA"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    VALID_SETUP = "VALID_SETUP"
    ACTIONABLE_SETUP = "ACTIONABLE_SETUP"


class ARXDecisionAuthority(str, Enum):
    BACKEND_CANONICAL = "BACKEND_CANONICAL"
    DISPLAY_ONLY_MARKET_DATA = "DISPLAY_ONLY_MARKET_DATA"


ACTIONABLE_EXECUTION_STATUSES = frozenset({"IN_BUY_ZONE", "READY_TO_BUY"})


def is_status_actionable(status: Optional[str]) -> bool:
    if not status:
        return False
    return status in ACTIONABLE_EXECUTION_STATUSES


def evaluate_actionability(
    decision_state: Optional[DecisionState | str],
    execution_status: Optional[str],
) -> bool:
    """Strict fail-closed evaluation of joint decision actionability.
    
    Both the decision state must be strictly ACTIONABLE_SETUP and the execution status
    must be in ACTIONABLE_EXECUTION_STATUSES. Missing or undefined state returns False.
    """
    if not is_status_actionable(execution_status):
        return False
    if isinstance(decision_state, DecisionState):
        return decision_state == DecisionState.ACTIONABLE_SETUP
    return decision_state == "ACTIONABLE_SETUP"


@dataclass
class MarketEvidenceContract:
    candles_count: int
    last_close: float
    vwap: Optional[float] = None
    atr_14: Optional[float] = None
    provider: str = "unknown"
    as_of: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FundamentalEvidenceContract:
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    source: str = "none"
    filing_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MacroEvidenceContract:
    regime_label: Optional[str] = None
    yield_spread_10y_2y: Optional[float] = None
    inflation_rate: Optional[float] = None
    fred_observation_date: Optional[str] = None
    source: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityEvidenceContract:
    spread_bps: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    liquidity_gate_passed: bool = False
    source: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ARXEvidenceItem:
    domain: EvidenceDomain
    quality: EvidenceQualityState
    source: str
    observed_at: str
    payload: Dict[str, Any]
    is_stale: bool = False
    staleness_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value if isinstance(self.domain, EvidenceDomain) else str(self.domain),
            "quality": self.quality.value if isinstance(self.quality, EvidenceQualityState) else str(self.quality),
            "source": self.source,
            "observed_at": self.observed_at,
            "payload": self.payload,
            "is_stale": self.is_stale,
            "staleness_reason": self.staleness_reason,
        }


@dataclass
class ExecutionLevels:
    entry_min: Optional[float] = None
    entry_max: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_loss_pct: Optional[float] = 0.0
    target_1: Optional[float] = None
    target_1_pct: Optional[float] = 0.0
    target_2: Optional[float] = None
    target_2_pct: Optional[float] = 0.0
    risk_reward_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionVerdict:
    symbol: str
    horizon: TimeHorizon
    user_role: UserRole
    is_actionable: bool
    can_size_trade: bool
    decision_state: DecisionState | str
    execution_status: str
    verdict_label: str
    disqualification_reason: Optional[str]
    confluence_score: float
    observation_date: str
    levels: ExecutionLevels
    data_completeness: DataCompleteness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon": self.horizon.value if isinstance(self.horizon, TimeHorizon) else str(self.horizon),
            "user_role": self.user_role.value if isinstance(self.user_role, UserRole) else str(self.user_role),
            "is_actionable": self.is_actionable,
            "can_size_trade": self.can_size_trade,
            "decision_state": self.decision_state.value if isinstance(self.decision_state, DecisionState) else str(self.decision_state),
            "execution_status": self.execution_status,
            "verdict_label": self.verdict_label,
            "disqualification_reason": self.disqualification_reason,
            "confluence_score": self.confluence_score,
            "observation_date": self.observation_date,
            "levels": self.levels.to_dict(),
            "data_completeness": self.data_completeness.value if isinstance(self.data_completeness, DataCompleteness) else str(self.data_completeness),
        }


@dataclass
class ARXDecisionContext:
    decision_id: str
    symbol: str
    horizon: TimeHorizon
    user_role: UserRole
    timestamp: str
    market_evidence: ARXEvidenceItem
    fundamental_evidence: ARXEvidenceItem
    macro_evidence: ARXEvidenceItem
    liquidity_evidence: ARXEvidenceItem
    evidence_completeness: DataCompleteness
    is_degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "horizon": self.horizon.value if isinstance(self.horizon, TimeHorizon) else str(self.horizon),
            "user_role": self.user_role.value if isinstance(self.user_role, UserRole) else str(self.user_role),
            "timestamp": self.timestamp,
            "market_evidence": self.market_evidence.to_dict(),
            "fundamental_evidence": self.fundamental_evidence.to_dict(),
            "macro_evidence": self.macro_evidence.to_dict(),
            "liquidity_evidence": self.liquidity_evidence.to_dict(),
            "evidence_completeness": self.evidence_completeness.value if isinstance(self.evidence_completeness, DataCompleteness) else str(self.evidence_completeness),
            "is_degraded": self.is_degraded,
        }


@dataclass
class ARXDecision:
    context: ARXDecisionContext
    verdict: DecisionVerdict
    confluence_score: float
    model_trace: Dict[str, Any]
    authority: ARXDecisionAuthority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "verdict": self.verdict.to_dict(),
            "confluence_score": self.confluence_score,
            "model_trace": self.model_trace,
            "authority": self.authority.value if isinstance(self.authority, ARXDecisionAuthority) else str(self.authority),
        }


def create_degraded_decision_context(
    symbol: str,
    horizon: TimeHorizon = TimeHorizon.SWING,
    user_role: UserRole = UserRole.LONG_TERM,
    market_data: Optional[Dict[str, Any]] = None,
) -> ARXDecisionContext:
    """Create a fail-closed degraded decision context when analytical engines are unreachable."""
    now = datetime.now(timezone.utc).isoformat()
    market_data = market_data or {}
    has_market = bool(market_data.get("last_close"))

    return ARXDecisionContext(
        decision_id=f"degraded-{symbol.upper()}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        symbol=symbol.upper(),
        horizon=horizon,
        user_role=user_role,
        timestamp=now,
        market_evidence=ARXEvidenceItem(
            domain=EvidenceDomain.MARKET_DATA,
            quality=EvidenceQualityState.FALLBACK if has_market else EvidenceQualityState.UNAVAILABLE,
            source=market_data.get("provider", "yahoo_finance_direct"),
            observed_at=market_data.get("as_of", now),
            payload={
                "candles_count": market_data.get("candles_count", 0),
                "last_close": market_data.get("last_close", 0.0),
                "vwap": market_data.get("vwap"),
                "atr_14": market_data.get("atr_14"),
                "provider": market_data.get("provider", "yahoo_finance_direct"),
                "as_of": market_data.get("as_of", now),
            },
            is_stale=False,
        ),
        fundamental_evidence=ARXEvidenceItem(
            domain=EvidenceDomain.FUNDAMENTALS,
            quality=EvidenceQualityState.UNAVAILABLE,
            source="none",
            observed_at=now,
            payload={
                "pe_ratio": None,
                "market_cap": None,
                "sector": None,
                "source": "none",
                "filing_date": None,
            },
            is_stale=True,
            staleness_reason="Backend analytics engine unreachable",
        ),
        macro_evidence=ARXEvidenceItem(
            domain=EvidenceDomain.MACRO,
            quality=EvidenceQualityState.UNAVAILABLE,
            source="none",
            observed_at=now,
            payload={
                "regime_label": None,
                "yield_spread_10y_2y": None,
                "inflation_rate": None,
                "fred_observation_date": None,
                "source": "none",
            },
            is_stale=True,
            staleness_reason="Backend analytics engine unreachable",
        ),
        liquidity_evidence=ARXEvidenceItem(
            domain=EvidenceDomain.LIQUIDITY,
            quality=EvidenceQualityState.UNAVAILABLE,
            source="none",
            observed_at=now,
            payload={
                "spread_bps": None,
                "avg_volume_30d": None,
                "liquidity_gate_passed": False,
                "source": "none",
            },
            is_stale=True,
            staleness_reason="Backend analytics engine unreachable",
        ),
        evidence_completeness=DataCompleteness.DEGRADED,
        is_degraded=True,
    )
