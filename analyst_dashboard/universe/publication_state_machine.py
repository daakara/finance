"""
analyst_dashboard/universe/publication_state_machine.py

Sprint 2 Track C: Seven-State Universe Publication & Scanner Gating State Machine.
Enforces the fail-closed lifecycle from registered identity to active scanning:

1. REGISTERED_IDENTITY
2. PROVIDER_RESOLVABLE
3. DATA_QUALIFIED
4. PUBLICATION_ELIGIBLE
5. PUBLISHED_UNIVERSE
6. STRATEGY_ADMITTED
7. ACTIVELY_SCANNED

Core Architectural Rules:
- Every forward transition requires explicit qualifying evidence.
- Zero state-skipping: forward progress is strictly incremental (1 step at a time).
- Fail-closed: missing, invalid, or ambiguous evidence immediately aborts transition.
- Strict anti-leakage firewall: State 5 (PUBLISHED_UNIVERSE) grants NO strategy admission
  or scanner execution authority.
- Authority governance separation (Step 4):
  * G-VCP-012 governs the independently deferred VCP Holdout Epoch 002 strategy admission process.
  * G-S2-ACT governs Sprint 2 operational scanner activation and dispatch.
  * Deferred VCP Gate 12 (G-VCP-012) does NOT authorize scanner enrollment under Sprint 2.
- Canonical association ID validation (Step 3):
  * Durable format: ASSOC_<internal_security_id>_<operating_mic>_<symbol>_<effective_from>
  * Standard hex format: ASSOC_[0-9A-F]{12}
  * Provisional format: PROV_ASSOC_<provisional_sec_id>_<operating_mic>_<symbol>_<date>
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type


# ─── 0. Canonical Association Identifier Validation ───────────────────────────

CANONICAL_DURABLE_ASSOC_PATTERN = re.compile(
    r"^ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z0-9]{4}_[A-Z0-9\._]+_\d{4}-\d{2}-\d{2}$"
)
CANONICAL_HEX_ASSOC_PATTERN = re.compile(r"^ASSOC_[0-9A-F]{12}$")
PROVISIONAL_ASSOC_PATTERN = re.compile(r"^PROV_ASSOC_.+$")


def validate_canonical_association_id(association_id: str) -> bool:
    """
    Validates canonical association_id according to Package A accepted formats:
    1. Durable format: ASSOC_<internal_security_id>_<operating_mic>_<symbol>_<effective_from>
       e.g. ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17
    2. Hex format: ASSOC_[0-9A-F]{12}
    3. Provisional format: PROV_ASSOC_<provisional_sec_id>_<operating_mic>_<symbol>_<date>
       e.g. PROV_ASSOC_PROV_SEC_A562B55CDF_XNAS_AAPL_20261017
    """
    if not association_id or not isinstance(association_id, str):
        return False
    clean_id = association_id.strip()
    if CANONICAL_DURABLE_ASSOC_PATTERN.match(clean_id):
        return True
    if CANONICAL_HEX_ASSOC_PATTERN.match(clean_id):
        return True
    if PROVISIONAL_ASSOC_PATTERN.match(clean_id):
        return True
    return False


# ─── 1. Canonical State Enumeration ──────────────────────────────────────────

class PublicationState(IntEnum):
    """
    Authoritative 7-state lifecycle for securities within the ARX Terminal platform.
    Progresses monotonically from registration to operational scanning.
    """
    REGISTERED_IDENTITY = 1
    PROVIDER_RESOLVABLE = 2
    DATA_QUALIFIED = 3
    PUBLICATION_ELIGIBLE = 4
    PUBLISHED_UNIVERSE = 5
    STRATEGY_ADMITTED = 6
    ACTIVELY_SCANNED = 7

    def label(self) -> str:
        return self.name

    @classmethod
    def from_label(cls, label: str) -> PublicationState:
        try:
            return cls[label.strip().upper()]
        except KeyError:
            raise ValueError(f"Unknown PublicationState label: {label}")


# ─── 2. Transition Exceptions ────────────────────────────────────────────────

class PublicationStateTransitionError(ValueError):
    """Base error for all illegal or unverified publication state transitions."""
    pass


class EvidenceMissingError(PublicationStateTransitionError):
    """Raised when mandatory evidence is null or absent."""
    pass


class EvidenceAmbiguityError(PublicationStateTransitionError):
    """Raised when evidence contains conflicting or unverified assertions."""
    pass


class StateOrderViolationError(PublicationStateTransitionError):
    """Raised when attempting non-monotonic or state-skipping transitions."""
    pass


class PolicyViolationError(PublicationStateTransitionError):
    """Raised when evidence violates canonical rules or release gates."""
    pass


# ─── 3. Evidence Containers ──────────────────────────────────────────────────

@dataclass(frozen=True)
class IdentityRegistrationEvidence:
    """
    Evidence qualifying State 1 (REGISTERED_IDENTITY).
    Proves identity registration in internal master (Package A).
    """
    internal_security_id: str
    primary_symbol: str
    operating_mic: str
    association_id: Optional[str] = None
    issuer_cik: Optional[str] = None
    registration_timestamp: str = ""
    registration_plan_id: str = ""
    status: str = "ACTIVE"
    continuity_basis: str = "INTERNAL_LEDGER_REGISTERED"
    certifies_continuity: bool = False

    def __post_init__(self):
        if not re.match(r"^ARX_SEC_[0-9A-F]{12}$", self.internal_security_id):
            raise EvidenceAmbiguityError(
                f"Invalid internal_security_id format: '{self.internal_security_id}'. "
                "Must match 'ARX_SEC_<12_HEX>'."
            )
        if not self.primary_symbol.strip():
            raise EvidenceMissingError("primary_symbol must not be empty.")
        if not self.operating_mic.strip():
            raise EvidenceMissingError("operating_mic must not be empty.")
        if self.association_id and not validate_canonical_association_id(self.association_id):
            raise EvidenceAmbiguityError(
                f"Invalid association_id format: '{self.association_id}'. "
                "Must match accepted canonical Package A format."
            )


@dataclass(frozen=True)
class ProviderResolutionEvidence:
    """
    Evidence qualifying State 1 -> State 2 (PROVIDER_RESOLVABLE).
    Confirms attributable mapping between internal ID and external provider symbols.
    """
    internal_security_id: str
    symbol: str
    operating_mic: str
    association_id: Optional[str] = None
    alpaca_symbol: Optional[str] = None
    polygon_symbol: Optional[str] = None
    figi_composite: Optional[str] = None
    is_ambiguous: bool = False
    dispute_reason: Optional[str] = None
    resolution_authority: str = "PACKAGE_A_ADMISSION_MAPPING"
    resolution_timestamp: str = ""

    def validate(self) -> None:
        if self.is_ambiguous:
            raise EvidenceAmbiguityError(
                f"Provider resolution for {self.symbol} is ambiguous: {self.dispute_reason or 'Unspecified ambiguity'}"
            )
        if not self.symbol.strip():
            raise EvidenceMissingError("symbol must not be empty.")
        if not self.operating_mic.strip():
            raise EvidenceMissingError("operating_mic must not be empty.")
        if not self.alpaca_symbol and not self.polygon_symbol:
            raise EvidenceMissingError(
                f"No provider symbols resolved for internal ID {self.internal_security_id}."
            )
        if self.association_id and not validate_canonical_association_id(self.association_id):
            raise EvidenceAmbiguityError(
                f"Invalid association_id format for {self.symbol}: '{self.association_id}'."
            )


@dataclass(frozen=True)
class MarketDataQualityEvidence:
    """
    Evidence qualifying State 2 -> State 3 (DATA_QUALIFIED).
    Confirms market data meets Tier 1 Operative Builder Contract (>= 50 daily bars floor).
    """
    symbol: str
    candle_count: int
    has_live_price: bool
    adjustment_state: str  # Must be 'SPLIT_ADJUSTED_ONLY'
    structural_quality: str  # Must be 'CLEAN'
    provenance_feed: str  # Must be 'SIP_CONSOLIDATED'
    data_as_of: str
    evidence_timestamp: str = ""
    builder_floor_bars: int = 50

    def validate(self) -> None:
        clean_sym = self.symbol.strip().upper()
        if not clean_sym:
            raise EvidenceMissingError("symbol must not be empty.")
        if self.candle_count < self.builder_floor_bars:
            raise PolicyViolationError(
                f"Insufficient historical coverage for {clean_sym}: {self.candle_count} bars "
                f"is less than required builder floor ({self.builder_floor_bars} bars)."
            )
        if not self.has_live_price:
            raise PolicyViolationError(
                f"Missing live price observation for {clean_sym}."
            )
        if self.adjustment_state != "SPLIT_ADJUSTED_ONLY":
            raise PolicyViolationError(
                f"Invalid adjustment state '{self.adjustment_state}' for {clean_sym}. "
                "Must be 'SPLIT_ADJUSTED_ONLY' (cash dividends unadjusted)."
            )
        if self.structural_quality != "CLEAN":
            raise PolicyViolationError(
                f"Structural data quality degraded or corrupt for {clean_sym}: '{self.structural_quality}'."
            )
        if self.provenance_feed != "SIP_CONSOLIDATED":
            raise PolicyViolationError(
                f"Invalid market-data provenance feed '{self.provenance_feed}' for {clean_sym}. "
                "Must be 'SIP_CONSOLIDATED' (IEX or participant-only feeds rejected)."
            )


@dataclass(frozen=True)
class EligibilityEvidence:
    """
    Evidence qualifying State 3 -> State 4 (PUBLICATION_ELIGIBLE).
    Confirms satisfaction of canonical rules R01-R07.
    """
    security_id: str
    symbol: str
    rule_results: Dict[str, bool]
    is_eligible: bool
    reason_code: str
    universe_rule_version: str = "eligibility-rule-v1.0"
    evaluation_timestamp: str = ""

    def validate(self) -> None:
        required_rules = {
            "R01_ASSET_CLASS",
            "R02_SECURITY_TYPE",
            "R03_LISTING_STATUS",
            "R04_PRIMARY_EXCHANGE",
            "R05_CURRENCY",
            "R06_PRIMARY_LISTING",
            "R07_NOT_DELISTED",
        }
        missing_rules = required_rules - set(self.rule_results.keys())
        if missing_rules:
            raise EvidenceMissingError(
                f"Eligibility evaluation incomplete for {self.symbol}. Missing rules: {sorted(missing_rules)}"
            )
        failing_rules = [r for r, passed in self.rule_results.items() if not passed]
        if failing_rules:
            raise PolicyViolationError(
                f"Security {self.symbol} failed eligibility rules: {failing_rules}. "
                f"Reason code: {self.reason_code}"
            )
        if not self.is_eligible:
            raise PolicyViolationError(
                f"Security {self.symbol} marked not eligible: reason={self.reason_code}"
            )


@dataclass(frozen=True)
class PublicationAttestationEvidence:
    """
    Evidence qualifying State 4 -> State 5 (PUBLISHED_UNIVERSE).
    Confirms accounting closure and cryptographic publication digest for isolated namespace.
    """
    universe_build_id: str
    universe_id: str
    snapshot_id: str
    publication_digest: str
    accounting_closure_verified: bool
    publication_decision: str  # 'PUBLISH' vs 'QUARANTINE'
    authorized_by: str
    publication_timestamp: str = ""

    def validate(self) -> None:
        if not self.accounting_closure_verified:
            raise PolicyViolationError(
                f"Accounting closure not verified for build {self.universe_build_id}. Publication quarantined."
            )
        if self.publication_decision != "PUBLISH":
            raise PolicyViolationError(
                f"Publication decision is '{self.publication_decision}' for build {self.universe_build_id}. "
                "Must be 'PUBLISH'."
            )
        if not self.publication_digest or len(self.publication_digest) != 64:
            raise EvidenceMissingError(
                f"Invalid or missing SHA-256 publication_digest: '{self.publication_digest}'."
            )
        if not self.authorized_by.strip():
            raise EvidenceMissingError("authorized_by must be explicitly specified.")


CANONICAL_STRATEGY_TYPES: Set[str] = {"VCP", "SMART_MONEY", "VALUE_GARP", "MOMENTUM", "BREAKOUT"}


@dataclass(frozen=True)
class StrategyAdmissionEvidence:
    """
    Evidence qualifying State 5 -> State 6 (STRATEGY_ADMITTED).
    Strategy-Specific Governance Isolation (Task 1):
    - For VCP strategies (strategy_id starting with 'VCP'): Requires G-VCP-012 certification
      under the independently deferred VCP Holdout Epoch 002 process.
    - For Non-VCP strategies: Governed by canonical strategy qualification authority;
      deferred VCP Gate 12 (G-VCP-012) custody requirements are NOT applied unless explicitly contracted.
    
    CRITICAL GOVERNANCE INVARIANT:
    Strategy admission (under either G-VCP-012 or non-VCP authority) does NOT authorize
    scanner enrollment or operational activation under Sprint 2. Operational dispatch
    strictly requires G-S2-ACT authority.
    """
    symbol: str
    strategy_id: str
    min_price_floor: float
    min_history_bars: int
    admission_approved: bool
    approved_by: str
    strategy_type: Optional[str] = None         # Authoritative strategy type (e.g. 'VCP', 'SMART_MONEY', 'VALUE_GARP')
    g_vcp_012_certified: Optional[bool] = None  # Mandatory for VCP strategies; not required for non-VCP
    strategy_certified: bool = True             # Strategy-specific qualification certification
    governance_authority: Optional[str] = None  # Defaults to G-VCP-012 for VCP; G-STRAT-<ID> for non-VCP
    admission_timestamp: str = ""

    def __post_init__(self) -> None:
        if self.governance_authority is None:
            default_auth = "G-VCP-012" if self.is_vcp_strategy() else f"G-STRAT-{self.strategy_id.strip().upper()}"
            object.__setattr__(self, "governance_authority", default_auth)

    @staticmethod
    def _is_vcp_id(strategy_id: str) -> bool:
        clean = strategy_id.strip().upper()
        return clean == "MINERVINI_VCP" or "VCP" in clean.split("_") or clean.startswith("VCP")

    def is_vcp_strategy(self) -> bool:
        """
        Determines whether strategy belongs to the VCP domain governed by G-VCP-012.
        Prefers authoritative strategy_type if supplied; otherwise checks canonical scanner IDs
        and prefixes (e.g. 'MINERVINI_VCP', 'VCP', 'VCP_STANDARD').
        """
        if self.strategy_type:
            return self.strategy_type.strip().upper() == "VCP"
        return self._is_vcp_id(self.strategy_id)

    @property
    def effective_governance_authority(self) -> str:
        return self.governance_authority or ("G-VCP-012" if self.is_vcp_strategy() else f"G-STRAT-{self.strategy_id.strip().upper()}")

    def validate(self) -> None:
        clean_sym = self.symbol.strip().upper()
        clean_strat = self.strategy_id.strip().upper()
        if not clean_strat:
            raise EvidenceMissingError("strategy_id must not be empty.")

        # Adversarial & Contradictory Strategy Type Validation
        if self.strategy_type is not None:
            norm_type = self.strategy_type.strip().upper()
            if norm_type not in CANONICAL_STRATEGY_TYPES:
                raise PolicyViolationError(
                    f"Invalid strategy_type '{self.strategy_type}' for {clean_sym}. "
                    f"Must be one of {sorted(CANONICAL_STRATEGY_TYPES)}."
                )
            id_is_vcp = self._is_vcp_id(clean_strat)
            type_is_vcp = (norm_type == "VCP")
            if id_is_vcp != type_is_vcp:
                raise PolicyViolationError(
                    f"Contradictory strategy specification for {clean_sym}: strategy_id '{clean_strat}' "
                    f"and strategy_type '{self.strategy_type}' conflict on VCP classification. "
                    "Adversarial or mismatched strategy parameters rejected."
                )

        # VCP Strategy Isolation: strictly enforce G-VCP-012
        if self.is_vcp_strategy():
            if not self.g_vcp_012_certified:
                raise PolicyViolationError(
                    f"Governance authority G-VCP-012 certification absent for {clean_sym} under VCP strategy {clean_strat}. Strategy admission blocked."
                )
        else:
            # Non-VCP Strategy Isolation: verify general strategy certification without requiring G-VCP-012
            if not self.strategy_certified:
                raise PolicyViolationError(
                    f"Strategy certification absent for non-VCP strategy {clean_strat} on {clean_sym}."
                )

        if not self.admission_approved:
            raise PolicyViolationError(
                f"Formal strategy admission unapproved for {clean_sym} under strategy {clean_strat}."
            )
        if self.min_price_floor < 5.0:
            raise PolicyViolationError(
                f"Rule L03 violation for {clean_sym}: price floor {self.min_price_floor} < $5.00."
            )
        if self.min_history_bars < 260:
            raise PolicyViolationError(
                f"Strategy benchmark violation for {clean_sym}: history {self.min_history_bars} bars < 260 bars."
            )
        if not self.approved_by.strip():
            raise EvidenceMissingError("approved_by must be explicitly specified.")


@dataclass(frozen=True)
class ScannerDispatchEvidence:
    """
    Evidence qualifying State 6 -> State 7 (ACTIVELY_SCANNED).
    Confirms Sprint 2 operational activation under governance authority G-S2-ACT.

    CRITICAL GOVERNANCE INVARIANT (Step 4):
    Deferred VCP Gate 12 (G-VCP-012) does NOT authorize scanner enrollment under Sprint 2.
    Operational scanner dispatch requires explicit G-S2-ACT operational activation.
    """
    symbol: str
    scanner_id: str
    g_s2_act_authorized: bool  # Sprint 2 operational activation authority (G-S2-ACT)
    schedule_active: bool
    operational_dispatch_authorized: bool
    governance_authority: str = "G-S2-ACT"
    authorized_strategy_id: Optional[str] = None  # Strategy authorizing this specific scanner
    dispatch_timestamp: str = ""

    def validate(self) -> None:
        clean_sym = self.symbol.strip().upper()
        clean_scanner = self.scanner_id.strip().upper()
        if not clean_scanner:
            raise EvidenceMissingError("scanner_id must not be empty.")
        if not self.g_s2_act_authorized:
            raise PolicyViolationError(
                f"Sprint 2 operational activation authority G-S2-ACT absent for {clean_sym}. "
                "Deferred VCP Gate 12 (G-VCP-012) does NOT authorize scanner enrollment."
            )
        if not self.schedule_active:
            raise PolicyViolationError(
                f"Scanner schedule dormant or inactive for {clean_sym}."
            )
        if not self.operational_dispatch_authorized:
            raise PolicyViolationError(
                f"Operational dispatch unauthorized for {clean_sym} under G-S2-ACT."
            )
        if self.authorized_strategy_id:
            clean_strat = self.authorized_strategy_id.strip().upper()
            if clean_strat not in clean_scanner and clean_scanner not in clean_strat:
                raise PolicyViolationError(
                    f"Scanner enrollment unauthorized: scanner '{self.scanner_id}' does not match "
                    f"authorized strategy '{self.authorized_strategy_id}'."
                )


# ─── 4. State Machine Engine ─────────────────────────────────────────────────

class PublicationStateMachine:
    """
    Pure, fail-closed state machine governing the 7 publication states.
    Validates evidence and enforces the monotonic execution boundary.
    """

    TRANSITION_EVIDENCE_MAP: Dict[Tuple[PublicationState, PublicationState], Type] = {
        (PublicationState.REGISTERED_IDENTITY, PublicationState.PROVIDER_RESOLVABLE): ProviderResolutionEvidence,
        (PublicationState.PROVIDER_RESOLVABLE, PublicationState.DATA_QUALIFIED): MarketDataQualityEvidence,
        (PublicationState.DATA_QUALIFIED, PublicationState.PUBLICATION_ELIGIBLE): EligibilityEvidence,
        (PublicationState.PUBLICATION_ELIGIBLE, PublicationState.PUBLISHED_UNIVERSE): PublicationAttestationEvidence,
        (PublicationState.PUBLISHED_UNIVERSE, PublicationState.STRATEGY_ADMITTED): StrategyAdmissionEvidence,
        (PublicationState.STRATEGY_ADMITTED, PublicationState.ACTIVELY_SCANNED): ScannerDispatchEvidence,
    }

    @classmethod
    def validate_transition(
        cls,
        current_state: PublicationState,
        next_state: PublicationState,
        evidence: Any,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether current_state can transition to next_state with given evidence.
        Returns (True, None) on success, or (False, error_message).
        """
        try:
            cls._check_transition_preconditions(current_state, next_state, evidence)
            return True, None
        except PublicationStateTransitionError as err:
            return False, str(err)

    @classmethod
    def transition(
        cls,
        current_state: PublicationState,
        next_state: PublicationState,
        evidence: Any,
    ) -> PublicationState:
        """
        Executes verified transition from current_state to next_state.
        Fails closed by raising PublicationStateTransitionError if invalid.
        """
        cls._check_transition_preconditions(current_state, next_state, evidence)
        return next_state

    @classmethod
    def _check_transition_preconditions(
        cls,
        current_state: PublicationState,
        next_state: PublicationState,
        evidence: Any,
    ) -> None:
        # 1. Monotonicity & order check: must progress exactly +1 step forward
        if next_state != current_state + 1:
            raise StateOrderViolationError(
                f"Illegal state transition from {current_state.name} ({int(current_state)}) "
                f"to {next_state.name} ({int(next_state)}). Forward transitions must be strictly "
                "incremental (exactly 1 step at a time). Zero state-skipping permitted."
            )

        # 2. Evidence presence check
        pair = (current_state, next_state)
        expected_type = cls.TRANSITION_EVIDENCE_MAP.get(pair)
        if expected_type is None:
            raise StateOrderViolationError(f"No valid transition mapped for {pair}.")

        if evidence is None:
            raise EvidenceMissingError(
                f"Missing mandatory evidence for transition {current_state.name} -> {next_state.name}. "
                f"Expected instance of {expected_type.__name__}."
            )

        if not isinstance(evidence, expected_type):
            raise EvidenceMissingError(
                f"Invalid evidence type for transition {current_state.name} -> {next_state.name}. "
                f"Expected {expected_type.__name__}, got {type(evidence).__name__}."
            )

        # 3. Evidence-specific semantic validation
        if hasattr(evidence, "validate") and callable(getattr(evidence, "validate")):
            evidence.validate()

    @classmethod
    def get_required_evidence_type(
        cls,
        current_state: PublicationState,
        next_state: PublicationState,
    ) -> Optional[Type]:
        """Returns the required evidence class for a given state transition."""
        return cls.TRANSITION_EVIDENCE_MAP.get((current_state, next_state))
