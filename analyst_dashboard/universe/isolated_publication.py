"""
analyst_dashboard/universe/isolated_publication.py

Sprint 2 Track C: Minimum Deterministic Universe-Publication Foundation.
Implements the isolated publication path for ARX Terminal universe expansion:

1. Deterministic Candidate Normalization & Deduplication.
2. Canonical Association ID Validation & Disambiguation (Step 3).
3. Canonical Eligibility (R01-R07) and Data Readiness Evaluation.
4. Invariant Accounting Closure:
   source_count == eligible + ineligible + unresolved
   eligible == data_ready + data_unavailable + data_unresolved
5. Order-Invariant SHA-256 Publication Content Digest.
6. Immutable Build Records & Atomic Publication Pointer with Rollback Invariants.
7. Reproducible In-Memory Dry-Run Output.
8. Population Isolation (3 pilots validated, 5 additional candidates unqualified pending B2).
9. Strict Anti-Inference & Fixture Firewall (Historical fixture cannot be substituted for live universe).
10. Zero Implicit Scanner Enrollment & Governance Authority Separation (Step 4):
    G-VCP-012 governs deferred VCP Epoch 002 strategy admission; G-S2-ACT governs operational dispatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .publication_state_machine import (
    PublicationState,
    PublicationStateMachine,
    ProviderResolutionEvidence,
    MarketDataQualityEvidence,
    EligibilityEvidence,
    PublicationAttestationEvidence,
    StrategyAdmissionEvidence,
    ScannerDispatchEvidence,
    PublicationStateTransitionError,
    PolicyViolationError,
    EvidenceMissingError,
    EvidenceAmbiguityError,
    validate_canonical_association_id,
)
from .fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE

logger = logging.getLogger("arx.universe.isolated_publication")

SCHEMA_VERSION: str = "1.0.0"
DEFAULT_EXPANSION_NAMESPACE: str = "ARX_US_EQUITIES_EXPANSION_150"
CANONICAL_MIC_LIST = {"XNAS", "XNYS", "ARCX", "BATS", "NASDAQ", "NYSE", "ARCA"}


def canonical_json_digest(obj: Any) -> str:
    """Computes deterministic SHA-256 digest of arbitrary structured payload."""
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ─── 1. Candidate Model ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class IsolatedUniverseCandidate:
    """
    Representation of an incoming candidate security for isolated universe evaluation.
    Supports canonical association_id format from Package A (Step 3).
    """
    internal_security_id: str
    symbol: str
    operating_mic: str = "XNAS"
    association_id: Optional[str] = None
    security_type: str = "COMMON_STOCK"
    asset_class: str = "EQUITY"
    listing_status: str = "ACTIVE"
    currency: str = "USD"
    primary_listing: Optional[bool] = True
    delisting_date: Optional[str] = None
    trading_suspension_date: Optional[str] = None
    alpaca_symbol: Optional[str] = None
    polygon_symbol: Optional[str] = None
    figi_composite: Optional[str] = None
    candle_count: int = 0
    has_live_price: bool = False
    adjustment_state: str = "SPLIT_ADJUSTED_ONLY"
    structural_quality: str = "CLEAN"
    provenance_feed: str = "SIP_CONSOLIDATED"
    data_as_of: Optional[str] = None
    is_ambiguous_identity: bool = False
    is_qualified_candidate: bool = True

    def normalize(self) -> IsolatedUniverseCandidate:
        """Deterministically trims and uppercases fields."""
        return IsolatedUniverseCandidate(
            internal_security_id=self.internal_security_id.strip(),
            symbol=self.symbol.strip().upper(),
            operating_mic=self.operating_mic.strip().upper(),
            association_id=self.association_id.strip() if self.association_id else None,
            security_type=self.security_type.strip().upper(),
            asset_class=self.asset_class.strip().upper(),
            listing_status=self.listing_status.strip().upper(),
            currency=self.currency.strip().upper(),
            primary_listing=bool(self.primary_listing) if self.primary_listing is not None else None,
            delisting_date=self.delisting_date.strip() if self.delisting_date else None,
            trading_suspension_date=self.trading_suspension_date.strip() if self.trading_suspension_date else None,
            alpaca_symbol=self.alpaca_symbol.strip().upper() if self.alpaca_symbol else None,
            polygon_symbol=self.polygon_symbol.strip().upper() if self.polygon_symbol else None,
            figi_composite=self.figi_composite.strip().upper() if self.figi_composite else None,
            candle_count=int(self.candle_count),
            has_live_price=bool(self.has_live_price),
            adjustment_state=self.adjustment_state.strip().upper(),
            structural_quality=self.structural_quality.strip().upper(),
            provenance_feed=self.provenance_feed.strip().upper(),
            data_as_of=self.data_as_of.strip() if self.data_as_of else None,
            is_ambiguous_identity=bool(self.is_ambiguous_identity),
            is_qualified_candidate=bool(self.is_qualified_candidate),
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── 2. Accounting & Result Models ───────────────────────────────────────────

@dataclass(frozen=True)
class AccountingClosureSummary:
    """
    Exact arithmetic accounting closure record.
    Unambiguous Semantics (Phase 6):
    - scannable_count: Denotes POTENTIAL candidate eligibility (DATA_READY eligible securities).
    - published_count: Securities published to the State 5 universe pointer.
    - strategy_admitted_count: Securities admitted to State 6 (strictly 0 at State 5).
    - scanner_enrolled_count: Securities enrolled in active State 7 dispatch (strictly 0 without G-S2-ACT).
    """
    source_count: int
    eligible_count: int
    ineligible_count: int
    unresolved_count: int
    data_ready_count: int
    data_unavailable_count: int
    data_unresolved_count: int
    scannable_count: int
    published_count: int = 0
    projected_published_count: int = 0
    strategy_admitted_count: int = 0
    scanner_enrolled_count: int = 0
    is_closed: bool = True
    closure_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """Per-candidate decision record."""
    candidate: IsolatedUniverseCandidate
    eligibility_decision: str  # 'ELIGIBLE', 'INELIGIBLE', 'UNRESOLVED'
    eligibility_reason: str
    rule_results: Dict[str, bool]
    data_readiness_decision: str  # 'DATA_READY', 'DATA_UNAVAILABLE', 'DATA_UNRESOLVED'
    data_readiness_reason: str
    current_publication_state: PublicationState

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_canonical_dict(),
            "eligibility_decision": self.eligibility_decision,
            "eligibility_reason": self.eligibility_reason,
            "rule_results": self.rule_results,
            "data_readiness_decision": self.data_readiness_decision,
            "data_readiness_reason": self.data_readiness_reason,
            "current_publication_state": self.current_publication_state.name,
        }


@dataclass(frozen=True)
class UniversePublicationPointer:
    """Atomic, immutable publication pointer for an active universe."""
    universe_id: str
    build_id: str
    publication_digest: str
    published_symbols: List[str]
    published_at_utc: str
    schema_version: str = SCHEMA_VERSION
    pointer_manifest_hash: str = ""
    previous_build_id: Optional[str] = None
    is_rollback: bool = False
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── 3. Isolated Publication Engine ──────────────────────────────────────────

class IsolatedPublicationEngine:
    """
    Minimum deterministic universe-publication engine for isolated expansion testing.
    Guarantees:
    - Zero production database mutation.
    - Zero state-leakage into live trading or active scanners.
    - Full accounting closure and order-invariant hashing.
    - Canonical association ID validation (Step 3).
    - Separation of G-VCP-012 from G-S2-ACT operational activation (Step 4).
    """

    def __init__(self, universe_id: str = DEFAULT_EXPANSION_NAMESPACE):
        self.universe_id = universe_id
        self._publication_pointer: Optional[UniversePublicationPointer] = None
        self._historical_pointers: List[UniversePublicationPointer] = []

    @property
    def current_pointer(self) -> Optional[UniversePublicationPointer]:
        return self._publication_pointer

    @classmethod
    def normalize_and_deduplicate(
        cls,
        candidates: List[IsolatedUniverseCandidate],
    ) -> Tuple[List[IsolatedUniverseCandidate], List[Dict[str, Any]]]:
        """
        Normalizes candidate records and flags duplicate securities, listing locators,
        or invalid/duplicate association IDs.
        Returns (deduplicated_candidates, exclusion_ledger).
        """
        seen_sec_ids: Set[str] = set()
        seen_locators: Set[Tuple[str, str]] = set()  # (symbol, operating_mic)
        seen_assoc_ids: Set[str] = set()
        deduped: List[IsolatedUniverseCandidate] = []
        exclusions: List[Dict[str, Any]] = []

        for cand in candidates:
            norm = cand.normalize()
            # 1. Check association_id validity if present (Step 3)
            if norm.association_id:
                if not validate_canonical_association_id(norm.association_id):
                    exclusions.append({
                        "symbol": norm.symbol,
                        "internal_security_id": norm.internal_security_id,
                        "association_id": norm.association_id,
                        "reason": "INVALID_ASSOCIATION_ID_FORMAT",
                        "details": f"Association ID '{norm.association_id}' does not match canonical Package A formats.",
                    })
                    continue

                if norm.association_id in seen_assoc_ids:
                    exclusions.append({
                        "symbol": norm.symbol,
                        "internal_security_id": norm.internal_security_id,
                        "association_id": norm.association_id,
                        "reason": "DUPLICATE_ASSOCIATION_ID",
                        "details": f"Association ID '{norm.association_id}' already registered in input population.",
                    })
                    continue

            # 2. Check duplicate internal security ID
            if norm.internal_security_id in seen_sec_ids:
                exclusions.append({
                    "symbol": norm.symbol,
                    "internal_security_id": norm.internal_security_id,
                    "reason": "DUPLICATE_INTERNAL_SECURITY_ID",
                    "details": f"Internal ID {norm.internal_security_id} already registered in input population.",
                })
                continue

            # 3. Check duplicate active listing locator
            locator = (norm.symbol, norm.operating_mic)
            if locator in seen_locators:
                exclusions.append({
                    "symbol": norm.symbol,
                    "internal_security_id": norm.internal_security_id,
                    "reason": "DUPLICATE_LISTING_LOCATOR",
                    "details": f"Locator ({norm.symbol}, {norm.operating_mic}) already registered in input population.",
                })
                continue

            seen_sec_ids.add(norm.internal_security_id)
            seen_locators.add(locator)
            if norm.association_id:
                seen_assoc_ids.add(norm.association_id)
            deduped.append(norm)

        # Deterministic order: sort by internal_security_id, then symbol
        deduped.sort(key=lambda c: (c.internal_security_id, c.symbol))
        return deduped, exclusions

    @classmethod
    def compute_publication_digest(cls, candidates: List[IsolatedUniverseCandidate]) -> str:
        """
        Computes SHA-256 publication content digest over sorted, normalized candidate representations.
        Order-invariant: candidates are deterministically sorted before serialization.
        """
        sorted_cands = sorted(
            [c.normalize() for c in candidates],
            key=lambda c: (c.internal_security_id, c.symbol)
        )
        raw_payload = [c.to_canonical_dict() for c in sorted_cands]
        return canonical_json_digest(raw_payload)

    @classmethod
    def evaluate_candidate(
        cls,
        cand: IsolatedUniverseCandidate,
        as_of_date: str,
    ) -> CandidateEvaluationResult:
        """
        Evaluates a candidate through the initial publication states:
        1. REGISTERED_IDENTITY
        2. PROVIDER_RESOLVABLE
        3. DATA_QUALIFIED
        4. PUBLICATION_ELIGIBLE
        """
        norm = cand.normalize()
        current_state = PublicationState.REGISTERED_IDENTITY

        # 1. State 1 -> State 2 (Provider Resolution)
        # Check provider resolution evidence
        try:
            prov_evidence = ProviderResolutionEvidence(
                internal_security_id=norm.internal_security_id,
                symbol=norm.symbol,
                operating_mic=norm.operating_mic,
                association_id=norm.association_id,
                alpaca_symbol=norm.alpaca_symbol,
                polygon_symbol=norm.polygon_symbol,
                figi_composite=norm.figi_composite,
                is_ambiguous=norm.is_ambiguous_identity,
                resolution_timestamp=as_of_date,
            )
            current_state = PublicationStateMachine.transition(
                current_state,
                PublicationState.PROVIDER_RESOLVABLE,
                prov_evidence,
            )
        except PublicationStateTransitionError as err:
            return CandidateEvaluationResult(
                candidate=norm,
                eligibility_decision="UNRESOLVED",
                eligibility_reason=f"PROVIDER_RESOLUTION_FAILED: {err}",
                rule_results={},
                data_readiness_decision="DATA_UNRESOLVED",
                data_readiness_reason=str(err),
                current_publication_state=current_state,
            )

        # 2. State 2 -> State 3 (Market Data Quality)
        # Unqualified candidates (e.g. 5 non-pilots pending Package B data) stop here
        if not norm.is_qualified_candidate or norm.candle_count == 0 or not norm.has_live_price:
            data_decision = "DATA_UNAVAILABLE"
            data_reason = "UNQUALIFIED_PENDING_PACKAGE_B_DATA" if not norm.is_qualified_candidate else "INSUFFICIENT_HISTORY"
            return CandidateEvaluationResult(
                candidate=norm,
                eligibility_decision="UNRESOLVED",
                eligibility_reason="MARKET_DATA_UNQUALIFIED",
                rule_results={},
                data_readiness_decision=data_decision,
                data_readiness_reason=data_reason,
                current_publication_state=current_state,
            )

        try:
            md_evidence = MarketDataQualityEvidence(
                symbol=norm.symbol,
                candle_count=norm.candle_count,
                has_live_price=norm.has_live_price,
                adjustment_state=norm.adjustment_state,
                structural_quality=norm.structural_quality,
                provenance_feed=norm.provenance_feed,
                data_as_of=norm.data_as_of or as_of_date,
                evidence_timestamp=as_of_date,
            )
            current_state = PublicationStateMachine.transition(
                current_state,
                PublicationState.DATA_QUALIFIED,
                md_evidence,
            )
            # Enforce Operational Builder Readiness Contract (60 daily bars)
            if norm.candle_count < 60:
                data_decision = "DATA_UNAVAILABLE"
                data_reason = f"INSUFFICIENT_BUILDER_READINESS_HISTORY: {norm.candle_count} bars < 60 bars operational contract"
            else:
                data_decision = "DATA_READY"
                data_reason = "PASSED_MARKET_DATA_QUALITY"
        except PublicationStateTransitionError as err:
            return CandidateEvaluationResult(
                candidate=norm,
                eligibility_decision="UNRESOLVED",
                eligibility_reason=f"DATA_QUALIFICATION_FAILED: {err}",
                rule_results={},
                data_readiness_decision="DATA_UNAVAILABLE",
                data_readiness_reason=str(err),
                current_publication_state=current_state,
            )

        # 3. State 3 -> State 4 (Eligibility Rules R01-R07)
        obs_date = as_of_date[:10] if as_of_date else ""

        # R01: Asset Class
        r01 = norm.asset_class == "EQUITY"

        # R02: Security Type
        r02 = norm.security_type == "COMMON_STOCK"

        # R03: Listing Status (Venue trading status ACTIVE at obs_date)
        if norm.trading_suspension_date and obs_date and obs_date >= norm.trading_suspension_date:
            r03 = False
        else:
            r03 = norm.listing_status == "ACTIVE"

        # R04: Primary Exchange (Major US Exchange)
        r04 = norm.operating_mic in CANONICAL_MIC_LIST

        # R05: Currency
        r05 = norm.currency == "USD"

        # R06: Primary Listing
        r06 = norm.primary_listing is True

        # R07: Not Delisted
        if norm.delisting_date:
            r07 = bool(obs_date and obs_date < norm.delisting_date)
        else:
            r07 = norm.listing_status != "DELISTED"

        rule_results = {
            "R01_ASSET_CLASS": r01,
            "R02_SECURITY_TYPE": r02,
            "R03_LISTING_STATUS": r03,
            "R04_PRIMARY_EXCHANGE": r04,
            "R05_CURRENCY": r05,
            "R06_PRIMARY_LISTING": r06,
            "R07_NOT_DELISTED": r07,
        }

        all_rules_passed = all(rule_results.values())
        if all_rules_passed:
            elig_decision = "ELIGIBLE"
            elig_reason = "PASSED_ALL_RULES"
            try:
                el_evidence = EligibilityEvidence(
                    security_id=norm.internal_security_id,
                    symbol=norm.symbol,
                    rule_results=rule_results,
                    is_eligible=True,
                    reason_code=elig_reason,
                    evaluation_timestamp=as_of_date,
                )
                current_state = PublicationStateMachine.transition(
                    current_state,
                    PublicationState.PUBLICATION_ELIGIBLE,
                    el_evidence,
                )
            except PublicationStateTransitionError as err:
                elig_decision = "UNRESOLVED"
                elig_reason = f"ELIGIBILITY_TRANSITION_FAILED: {err}"
        else:
            elig_decision = "INELIGIBLE"
            failing = [k for k, v in rule_results.items() if not v]
            elig_reason = f"FAILED_RULES: {', '.join(failing)}"

        return CandidateEvaluationResult(
            candidate=norm,
            eligibility_decision=elig_decision,
            eligibility_reason=elig_reason,
            rule_results=rule_results,
            data_readiness_decision=data_decision,
            data_readiness_reason=data_reason,
            current_publication_state=current_state,
        )

    @classmethod
    def evaluate_accounting_closure(
        cls,
        evaluations: List[CandidateEvaluationResult],
        source_count: int,
        is_active_publication: bool = False,
        active_pointer_count: int = 0,
    ) -> AccountingClosureSummary:
        """
        Verifies arithmetic accounting invariants:
        1. source_count == eligible + ineligible + unresolved
        2. eligible == data_ready + data_unavailable + data_unresolved

        Contextual Publication Semantics:
        - When evaluated during candidate inspection or dry-run, published_count reflects
          the currently committed publication pointer count (or 0 if uncommitted),
          while projected_published_count reports candidates qualifying for State 5.
        - When evaluated during active publication commit, published_count reports the
          newly published candidate count.
        """
        eligible = [e for e in evaluations if e.eligibility_decision == "ELIGIBLE"]
        ineligible = [e for e in evaluations if e.eligibility_decision == "INELIGIBLE"]
        unresolved = [e for e in evaluations if e.eligibility_decision == "UNRESOLVED"]

        eligible_count = len(eligible)
        ineligible_count = len(ineligible)
        unresolved_count = len(unresolved)

        # For eligible candidates, evaluate data readiness breakdown
        data_ready = [e for e in eligible if e.data_readiness_decision == "DATA_READY"]
        data_unavail = [e for e in eligible if e.data_readiness_decision == "DATA_UNAVAILABLE"]
        data_unres = [e for e in eligible if e.data_readiness_decision == "DATA_UNRESOLVED"]

        data_ready_count = len(data_ready)
        data_unavailable_count = len(data_unavail)
        data_unresolved_count = len(data_unres)
        # scannable_count denotes POTENTIAL eligibility (DATA_READY eligible securities)
        scannable_count = data_ready_count
        projected_published_count = data_ready_count

        if is_active_publication:
            published_count = data_ready_count
        else:
            published_count = active_pointer_count

        strategy_admitted_count = 0
        scanner_enrolled_count = 0

        closure_errors: List[str] = []

        # Check invariant 1: Source Closure
        if source_count != (eligible_count + ineligible_count + unresolved_count):
            closure_errors.append(
                f"Source closure violation: source_count ({source_count}) != "
                f"eligible ({eligible_count}) + ineligible ({ineligible_count}) + unresolved ({unresolved_count})"
            )

        # Check invariant 2: Readiness Closure
        if eligible_count != (data_ready_count + data_unavailable_count + data_unresolved_count):
            closure_errors.append(
                f"Readiness closure violation: eligible_count ({eligible_count}) != "
                f"data_ready ({data_ready_count}) + data_unavailable ({data_unavailable_count}) + data_unresolved ({data_unresolved_count})"
            )

        is_closed = len(closure_errors) == 0
        error_msg = "; ".join(closure_errors) if closure_errors else None

        return AccountingClosureSummary(
            source_count=source_count,
            eligible_count=eligible_count,
            ineligible_count=ineligible_count,
            unresolved_count=unresolved_count,
            data_ready_count=data_ready_count,
            data_unavailable_count=data_unavailable_count,
            data_unresolved_count=data_unresolved_count,
            scannable_count=scannable_count,
            published_count=published_count,
            projected_published_count=projected_published_count,
            strategy_admitted_count=strategy_admitted_count,
            scanner_enrolled_count=scanner_enrolled_count,
            is_closed=is_closed,
            closure_error=error_msg,
        )

    def execute_dry_run(
        self,
        candidates: List[IsolatedUniverseCandidate],
        as_of_date: str = "2026-10-19",
    ) -> Dict[str, Any]:
        """
        Executes a pure in-memory, reproducible dry-run of universe publication.
        Makes zero database writes and triggers no live processes.
        """
        deduped, exclusions = self.normalize_and_deduplicate(candidates)
        evaluations = [self.evaluate_candidate(c, as_of_date) for c in deduped]
        active_count = len(self._publication_pointer.published_symbols) if self._publication_pointer else 0
        accounting = self.evaluate_accounting_closure(
            evaluations, len(deduped), is_active_publication=False, active_pointer_count=active_count
        )
        digest = self.compute_publication_digest(deduped)

        published_candidates = [
            e.candidate.symbol
            for e in evaluations
            if e.current_publication_state >= PublicationState.PUBLICATION_ELIGIBLE
            and e.data_readiness_decision == "DATA_READY"
        ]

        build_id = f"dryrun-{int(datetime.now(timezone.utc).timestamp())}"

        return {
            "is_dry_run": True,
            "build_id": build_id,
            "universe_id": self.universe_id,
            "schema_version": SCHEMA_VERSION,
            "as_of_date": as_of_date,
            "publication_digest": digest,
            "accounting": accounting.to_dict(),
            "published_symbols": sorted(published_candidates),
            "evaluations": [e.to_dict() for e in evaluations],
            "exclusions": exclusions,
            "zero_production_mutation_verified": True,
            "zero_scanner_enrollment_verified": True,
        }

    def publish_universe(
        self,
        candidates: List[IsolatedUniverseCandidate],
        as_of_date: str,
        authorized_by: str,
        build_id: Optional[str] = None,
    ) -> UniversePublicationPointer:
        """
        Publishes the isolated universe to an in-memory pointer.
        Fails closed on accounting closure errors or missing authorization.
        """
        deduped, exclusions = self.normalize_and_deduplicate(candidates)
        evaluations = [self.evaluate_candidate(c, as_of_date) for c in deduped]
        accounting = self.evaluate_accounting_closure(
            evaluations, len(deduped), is_active_publication=True
        )

        if not accounting.is_closed:
            raise PolicyViolationError(
                f"Cannot publish universe {self.universe_id}: Accounting closure failed: {accounting.closure_error}"
            )

        actual_build_id = build_id or f"ubuild-iso-{int(datetime.now(timezone.utc).timestamp())}"
        digest = self.compute_publication_digest(deduped)

        # Qualifying State 4 -> State 5
        published_symbols = sorted([
            e.candidate.symbol
            for e in evaluations
            if e.current_publication_state >= PublicationState.PUBLICATION_ELIGIBLE
            and e.data_readiness_decision == "DATA_READY"
        ])

        attestation = PublicationAttestationEvidence(
            universe_build_id=actual_build_id,
            universe_id=self.universe_id,
            snapshot_id=f"snap-{digest[:12]}",
            publication_digest=digest,
            accounting_closure_verified=accounting.is_closed,
            publication_decision="PUBLISH",
            authorized_by=authorized_by,
            publication_timestamp=as_of_date,
        )
        attestation.validate()

        now_iso = datetime.now(timezone.utc).isoformat()
        manifest_hash = canonical_json_digest({
            "universe_id": self.universe_id,
            "build_id": actual_build_id,
            "digest": digest,
            "symbols": published_symbols,
            "authorized_by": authorized_by,
        })

        new_pointer = UniversePublicationPointer(
            universe_id=self.universe_id,
            build_id=actual_build_id,
            publication_digest=digest,
            published_symbols=published_symbols,
            published_at_utc=now_iso,
            schema_version=SCHEMA_VERSION,
            pointer_manifest_hash=manifest_hash,
            previous_build_id=self._publication_pointer.build_id if self._publication_pointer else None,
        )

        if self._publication_pointer:
            self._historical_pointers.append(self._publication_pointer)

        self._publication_pointer = new_pointer
        return new_pointer

    def rollback_publication(
        self,
        reason: str,
    ) -> UniversePublicationPointer:
        """
        Rolls back active publication pointer to the immediate previous pointer.
        Fails closed if no prior pointer exists.
        """
        if not self._historical_pointers:
            raise PolicyViolationError(
                f"Cannot rollback universe {self.universe_id}: No historical publication pointer exists."
            )

        prev_pointer = self._historical_pointers.pop()
        now_iso = datetime.now(timezone.utc).isoformat()

        rollback_pointer = UniversePublicationPointer(
            universe_id=prev_pointer.universe_id,
            build_id=prev_pointer.build_id,
            publication_digest=prev_pointer.publication_digest,
            published_symbols=list(prev_pointer.published_symbols),
            published_at_utc=now_iso,
            schema_version=SCHEMA_VERSION,
            pointer_manifest_hash=prev_pointer.pointer_manifest_hash,
            previous_build_id=self._publication_pointer.build_id if self._publication_pointer else None,
            is_rollback=True,
            rollback_reason=reason,
        )

        self._publication_pointer = rollback_pointer
        return rollback_pointer

    @classmethod
    def verify_historical_fixture_isolation(
        cls,
        universe_id: str,
        symbols: List[str],
    ) -> None:
        """
        Firewall check: guarantees HISTORICAL_VCP_35_REGRESSION_FIXTURE cannot
        be silently substituted for live published universe authority.
        """
        if universe_id == "HISTORICAL_VCP_35_REGRESSION_FIXTURE":
            raise PolicyViolationError(
                "Historical regression fixture cannot be used as published-universe authority."
            )
        # Check if the symbol list is an unverified clone of the 35 regression fixture
        if set(symbols) == set(HISTORICAL_VCP_35_REGRESSION_FIXTURE) and universe_id != DEFAULT_EXPANSION_NAMESPACE:
            raise PolicyViolationError(
                "Silent substitution of 35-symbol regression fixture detected without isolated expansion manifest."
            )

    @classmethod
    def assert_no_scanner_enrollment(
        cls,
        published_symbols: List[str],
        active_scanner_symbols: Set[str],
    ) -> None:
        """
        Consumer Protection Check: Guarantees published securities are NOT implicitly
        enrolled into active scanner schedule without explicit G-S2-ACT operational activation.
        Deferred VCP Gate 12 (G-VCP-012) does NOT grant scanner enrollment.
        """
        unauthorized = set(published_symbols).intersection(active_scanner_symbols)
        if unauthorized:
            raise PolicyViolationError(
                f"Actionability leakage detected: Symbols {sorted(unauthorized)} enrolled in active scanner "
                "without State 6 (G-VCP-012) and State 7 (G-S2-ACT) qualifying operational authorization."
            )


# ─── 4. Package B V3 Canonical Consumer ──────────────────────────────────────

CANONICAL_PACKAGE_B_V3_DIGEST: str = "0f54c702f0a4ede41f4e54bb2635464919a84270bd9532a4830ad2a0a2b7f4bf"
CANONICAL_PACKAGE_B_V3_BYTE_LENGTH: int = 9972

# Predecessor manifests formally invalidated and superseded by Package B canonical release decision
SUPERSEDED_PREDECESSOR_DIGESTS: Set[str] = {
    "65f5d9ab185056fd8c092ad1601a6db195494063b5ccfce562a11aad8aa063cd",  # Predecessor 1 (Workstream A, 3,489 bytes)
    "8923dad988a10ae07a2b7bda1044127130bf9c90539424d98f25676a9a59f07b",  # Predecessor 2 (Workstream B, 13,833 bytes)
}

# For backwards compatibility with existing references
PACKAGE_B_V3_REFERENCE_DIGEST: str = CANONICAL_PACKAGE_B_V3_DIGEST
PACKAGE_B_V3_RECONCILED_DIGEST: str = CANONICAL_PACKAGE_B_V3_DIGEST


class PackageBV3Consumer:
    """
    Direct consumer of canonical Package B V3 historical ingestion artifacts.
    Enforces a single active producer contract under ARX Closure Model v2:
    - Canonical Manifest ID: MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL
    - Authoritative Digest: 0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF (9,972 bytes)
    - Rejects all superseded predecessor manifests (65F5D9AB..., 8923DAD9...) and tampered manifests fail closed.
    - Candidate cohort: AAPL, GOOG, GOOGL, ABBV.
    - Full-history certification requirements: SIP_CONSOLIDATED, SPLIT_ADJUSTED_ONLY, CLEAN.
    - Ingestion window: 2025-08-27 to 2026-10-19 (288 NYSE trading sessions).
    - Operating MICs: XNAS (AAPL, GOOG, GOOGL), XNYS (ABBV).
    - Strict separation between MANIFEST_VALIDATED, DATA_INGESTED, DATA_CERTIFIED, PUBLICATION_ELIGIBLE, and PUBLISHED.
    """

    @classmethod
    def load_manifest(
        cls,
        manifest_source: Any,
        expected_digest: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Loads and verifies the canonical Package B V3 pilot manifest from a file path or dict.
        Returns (manifest_dict, file_digest).
        Fails closed on any digest mismatch, superseded predecessor digest, or schema violation.
        """
        # 1. Enforce rejection if caller requests a superseded predecessor digest
        target_expected = (expected_digest or CANONICAL_PACKAGE_B_V3_DIGEST).strip().lower()
        if target_expected in SUPERSEDED_PREDECESSOR_DIGESTS:
            raise PolicyViolationError(
                f"Package B manifest uses superseded predecessor digest '{target_expected}'. "
                f"Only canonical V3 digest '{CANONICAL_PACKAGE_B_V3_DIGEST}' is authorized. Failing closed."
            )

        if isinstance(manifest_source, (str, Path)):
            path = Path(manifest_source)
            if not path.is_file():
                raise PolicyViolationError(f"Package B V3 manifest file not found: {path}")
            raw_bytes = path.read_bytes()
            computed_digest = hashlib.sha256(raw_bytes).hexdigest().lower()
            if computed_digest in SUPERSEDED_PREDECESSOR_DIGESTS:
                raise PolicyViolationError(
                    f"Package B manifest uses superseded predecessor digest '{computed_digest}'. "
                    f"Only canonical V3 digest '{CANONICAL_PACKAGE_B_V3_DIGEST}' is authorized. Failing closed."
                )
            try:
                manifest_dict = json.loads(raw_bytes.decode("utf-8"))
            except Exception as err:
                raise PolicyViolationError(f"Failed to parse Package B V3 manifest JSON: {err}")
        elif isinstance(manifest_source, dict):
            manifest_dict = manifest_source
            computed_digest = canonical_json_digest(manifest_dict)
            if computed_digest in SUPERSEDED_PREDECESSOR_DIGESTS:
                raise PolicyViolationError(
                    f"Package B manifest uses superseded predecessor digest '{computed_digest}'. "
                    f"Only canonical V3 digest '{CANONICAL_PACKAGE_B_V3_DIGEST}' is authorized. Failing closed."
                )
        else:
            raise PolicyViolationError(f"Invalid manifest source type: {type(manifest_source)}")

        # Enforce exact digest verification against canonical authority
        if computed_digest != target_expected:
            raise PolicyViolationError(
                f"Package B V3 manifest digest mismatch: computed '{computed_digest}' != expected '{target_expected}'. "
                "Failing closed under ARX decision-first security policy."
            )

        # Validate canonical V3 schema and contracts
        cls.validate_manifest_contract(manifest_dict)
        return manifest_dict, computed_digest

    @classmethod
    def validate_manifest_contract(cls, manifest: Dict[str, Any]) -> None:
        """Validates canonical Package B V3 manifest structure, feed specification, and window."""
        manifest_id = manifest.get("manifest_id", "")
        if manifest_id != "MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL":
            raise PolicyViolationError(
                f"Invalid manifest_id '{manifest_id}'. Must be 'MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL'."
            )

        provider_cfg = manifest.get("provider_configuration", {})
        tape_feed = provider_cfg.get("tape_feed", "").lower()
        if tape_feed != "sip":
            raise PolicyViolationError(
                f"Package B V3 tape_feed must be 'sip' (Consolidated Tape), got: '{tape_feed}'"
            )

        adj = provider_cfg.get("adjustment_convention", "").lower()
        if adj != "split":
            raise PolicyViolationError(
                f"Package B V3 adjustment must be 'split' (SPLIT_ADJUSTED_ONLY), got: '{adj}'"
            )

        window = manifest.get("acquisition_window", {})
        if window.get("requested_start_date") != "2025-08-27":
            raise PolicyViolationError(
                f"Package B V3 start_date must be '2025-08-27', got: '{window.get('requested_start_date')}'"
            )
        if window.get("requested_end_date") != "2026-10-19":
            raise PolicyViolationError(
                f"Package B V3 end_date must be '2026-10-19', got: '{window.get('requested_end_date')}'"
            )

        candidates = manifest.get("pilot_candidates", [])
        if not candidates or len(candidates) != 4:
            raise PolicyViolationError(
                f"Package B V3 manifest must declare exactly 4 pilot candidates, got {len(candidates)}."
            )

    @classmethod
    def build_candidates_from_manifest(
        cls,
        manifest: Dict[str, Any],
        market_data_fixture: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[IsolatedUniverseCandidate]:
        """
        Builds IsolatedUniverseCandidate objects directly from canonical Package B V3 manifest.
        
        CRITICAL ARCHITECTURAL BOUNDARY (Phase 4):
        Separates MANIFEST_VALIDATED from DATA_CERTIFIED.
        - The manifest is a PROPOSAL (authorization_state: PROPOSED_NOT_AUTHORIZED).
          Manifest validation alone grants NO market data qualification or certification.
        - Without an explicit market_data_fixture, candidates are constructed in state
          MANIFEST_VALIDATED with candle_count=0, uncertified, and unprovenanced.
        - For local tests, a market_data_fixture may be supplied, but it MUST declare
          `_fixture_nature="SYNTHETIC_TEST_FIXTURE"` to confirm it is not production data.
        """
        pilot_candidates = manifest.get("pilot_candidates", [])
        if not pilot_candidates:
            raise PolicyViolationError("Package B V3 manifest contains empty pilot_candidates.")

        # If a market data fixture is provided, enforce explicit synthetic labeling
        if market_data_fixture is not None:
            if market_data_fixture.get("_fixture_nature") != "SYNTHETIC_TEST_FIXTURE":
                raise PolicyViolationError(
                    "Market data fixture must be explicitly declared with _fixture_nature='SYNTHETIC_TEST_FIXTURE'. "
                    "Manifest intent alone cannot confer data certification."
                )

        candidates: List[IsolatedUniverseCandidate] = []

        for candidate_entry in pilot_candidates:
            symbol = candidate_entry["symbol"]
            sec_id = candidate_entry["durable_security_id"]
            assoc_id = candidate_entry.get("listing_association_id")
            mic = candidate_entry.get("operating_mic", "XNAS")

            # Default state: MANIFEST_VALIDATED only (NOT data-certified)
            candle_count = 0
            has_live_price = False
            provenance_feed = "UNVERIFIED"
            adjustment_state = "UNVERIFIED"
            structural_quality = "UNVERIFIED"
            data_as_of = None
            is_qualified = False

            # If an explicit test fixture is provided, apply test market data
            if market_data_fixture is not None and symbol in market_data_fixture:
                sym_md = market_data_fixture[symbol]
                candle_count = sym_md.get("candle_count", 0)
                has_live_price = sym_md.get("has_live_price", False)
                provenance_feed = sym_md.get("provenance_feed", "SIP_CONSOLIDATED")
                adjustment_state = sym_md.get("adjustment_state", "SPLIT_ADJUSTED_ONLY")
                structural_quality = sym_md.get("structural_quality", "CLEAN")
                data_as_of = sym_md.get("data_as_of", "2026-10-19")
                is_qualified = sym_md.get("is_qualified_candidate", True)

            cand = IsolatedUniverseCandidate(
                internal_security_id=sec_id,
                symbol=symbol,
                operating_mic=mic,
                association_id=assoc_id,
                security_type="COMMON_STOCK",
                asset_class="EQUITY",
                listing_status="ACTIVE",
                currency="USD",
                primary_listing=True,
                alpaca_symbol=symbol,
                candle_count=candle_count,
                has_live_price=has_live_price,
                adjustment_state=adjustment_state,
                structural_quality=structural_quality,
                provenance_feed=provenance_feed,
                data_as_of=data_as_of,
                is_qualified_candidate=is_qualified,
            )
            candidates.append(cand)

        return candidates

    @classmethod
    def verify_multi_share_class_isolation(
        cls,
        goog_cand: IsolatedUniverseCandidate,
        googl_cand: IsolatedUniverseCandidate,
        goog_prices: Optional[List[float]] = None,
        googl_prices: Optional[List[float]] = None,
        test_fixture_alias_collision_check: bool = False,
    ) -> None:
        """
        Enforces dual-class capital structure separation between GOOG (Class C) and GOOGL (Class A):
        1. Distinct internal security IDs (ARX_SEC_A11A24D442E2 vs ARX_SEC_B7EF9ED7A35A)
        2. Distinct canonical association IDs
        3. Distinct provider symbol locators
        4. Rejection of genuine alias or association collision.
        
        NOTE ON PRICE EQUALITY (Phase 7):
        Temporary or coincidental equality of market prices (e.g. goog_prices == googl_prices)
        is NOT sufficient by itself to establish identity corruption. Authoritative identity
        and source-record linkage form the fail-closed criteria. Identical prices do NOT
        trigger failure.
        """
        if goog_cand.internal_security_id == googl_cand.internal_security_id:
            raise PolicyViolationError(
                f"Share class collision: GOOG and GOOGL share identical internal_security_id: {goog_cand.internal_security_id}"
            )
        if goog_cand.association_id and googl_cand.association_id:
            if goog_cand.association_id == googl_cand.association_id:
                raise PolicyViolationError(
                    f"Association ID collision: GOOG and GOOGL share identical association_id: {goog_cand.association_id}"
                )
        if goog_cand.symbol == googl_cand.symbol:
            raise PolicyViolationError(
                f"Symbol collision: GOOG and GOOGL share identical symbol: {goog_cand.symbol}"
            )
        if goog_cand.alpaca_symbol and googl_cand.alpaca_symbol and goog_cand.alpaca_symbol == googl_cand.alpaca_symbol:
            raise PolicyViolationError(
                f"Provider symbol collision: GOOG and GOOGL share identical provider symbol: {goog_cand.alpaca_symbol}"
            )


# ─── 5. Scanner Publication Membership Consumer (S2-07) ──────────────────────

class ScannerPublicationMembershipConsumer:
    """
    Sprint 2 Track C / S2-07: Offline Scanner Universe Membership Consumer.

    Provides the fail-closed interface allowing market-wide scanners (e.g. MINERVINI_VCP,
    SMART_MONEY, VALUE_GARP) to safely consume committed, authorized universe publications.

    Invariants Enforced:
    1. Only committed, authorized publication pointers may supply scanner membership.
    2. Candidate previews and dry-run evaluations strictly prohibited from activating scanners.
    3. Invalid, tampered, or stale publication identity fails closed.
    4. Membership counts strictly agree with the committed pointer.
    5. Rollback and pointer transitions immediately invalidate stale memberships.
    6. Missing publication evidence produces truthful unavailable/pending capability states.
    7. Pure offline execution: zero background scanner runs, scheduler triggers, network I/O, or DB writes.
    """

    @classmethod
    def consume_membership(
        cls,
        pointer: Any,
        expected_universe_id: Optional[str] = None,
        current_engine: Optional[IsolatedPublicationEngine] = None,
    ) -> List[str]:
        """
        Consumes scannable symbol membership from an authorized UniversePublicationPointer.
        Fails closed if given dry-run results, uncommitted structures, invalid namespaces,
        malformed digests, or superseded/stale pointers.
        """
        # Invariant 2: Candidate previews and dry-run evaluations strictly prohibited
        if isinstance(pointer, dict):
            if pointer.get("is_dry_run"):
                raise PolicyViolationError(
                    "DRY_RUN_SCANNER_ACTIVATION_PROHIBITED: Candidate evaluations and dry-runs "
                    "cannot activate scanners or supply scanner membership."
                )
            raise PolicyViolationError(
                f"INVALID_POINTER_TYPE: Expected UniversePublicationPointer, got dict: {list(pointer.keys())}"
            )

        # Invariant 1: Only committed, authorized publication pointers may supply membership
        if not isinstance(pointer, UniversePublicationPointer):
            raise PolicyViolationError(
                f"UNCOMMITTED_PUBLICATION: Scanner membership can only be supplied from an authorized, "
                f"committed UniversePublicationPointer. Received: {type(pointer).__name__ if pointer is not None else 'None'}"
            )

        # Invariant 3: Validate publication identity and formats
        if not pointer.universe_id or not re.match(r"^ARX_[A-Z0-9_]+$", pointer.universe_id):
            raise PolicyViolationError(
                f"INVALID_UNIVERSE_IDENTITY: universe_id '{pointer.universe_id}' does not match canonical namespace regex."
            )

        if expected_universe_id and pointer.universe_id != expected_universe_id:
            raise PolicyViolationError(
                f"UNIVERSE_ID_MISMATCH: Expected '{expected_universe_id}', got '{pointer.universe_id}'."
            )

        if (
            not pointer.publication_digest
            or len(pointer.publication_digest) != 64
            or not re.match(r"^[0-9a-fA-F]{64}$", pointer.publication_digest)
        ):
            raise PolicyViolationError(
                f"INVALID_PUBLICATION_DIGEST: Digest '{pointer.publication_digest}' must be a 64-character hexadecimal SHA-256."
            )

        if not pointer.pointer_manifest_hash or len(pointer.pointer_manifest_hash) != 64:
            raise PolicyViolationError(
                f"INVALID_MANIFEST_HASH: Missing or invalid pointer_manifest_hash on build '{pointer.build_id}'."
            )

        # Invariant 4: Membership validation and count agreement
        if not pointer.published_symbols:
            raise PolicyViolationError(
                f"EMPTY_MEMBERSHIP: Publication pointer '{pointer.build_id}' contains zero published symbols."
            )

        # Invariant 5: Stale & Superseded pointer validation against current engine
        if current_engine is not None:
            active_ptr = current_engine.current_pointer
            if active_ptr is None:
                raise PolicyViolationError(
                    f"STALE_POINTER_REJECTED: Engine currently has no committed active pointer "
                    f"(build '{pointer.build_id}' is orphaned or uncommitted)."
                )
            if active_ptr.build_id != pointer.build_id:
                raise PolicyViolationError(
                    f"STALE_SUPERSEDED_POINTER: Build '{pointer.build_id}' has been superseded by active build '{active_ptr.build_id}'."
                )
            if active_ptr.pointer_manifest_hash != pointer.pointer_manifest_hash:
                raise PolicyViolationError(
                    "MANIFEST_HASH_DESYNC: Pointer manifest hash does not match active engine pointer."
                )

        return sorted(list(pointer.published_symbols))

    @classmethod
    def get_scanner_capability_status(
        cls,
        pointer: Optional[UniversePublicationPointer],
        scanner_id: str = "MINERVINI_VCP",
    ) -> Dict[str, Any]:
        """
        Translates publication pointer state into truthful radar capability disclosures,
        matching frontend/app/radar/page.tsx and api/routes/screener.py contracts.
        Missing publication produces truthful PIPELINE_PENDING.
        """
        if pointer is None or not isinstance(pointer, UniversePublicationPointer) or not pointer.published_symbols:
            return {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "buildId": None,
                "universeId": None,
                "reason": "NO_COMMITTED_PUBLICATION_POINTER",
                "rationale": f"Market-wide screening for {scanner_id} is pending authorized universe publication pointer commit.",
            }

        return {
            "status": "AVAILABLE",
            "universeScreening": "AVAILABLE",
            "singleAssetAnalysis": "AVAILABLE",
            "matchedCount": len(pointer.published_symbols),
            "buildId": pointer.build_id,
            "universeId": pointer.universe_id,
            "digest": pointer.publication_digest,
            "publishedAt": pointer.published_at_utc,
            "isRollback": pointer.is_rollback,
            "rationale": f"Committed publication pointer '{pointer.build_id}' active with {len(pointer.published_symbols)} scannable symbols.",
        }

    @classmethod
    def verify_dispatch_authorization(
        cls,
        symbol: str,
        scanner_id: str,
        evidence: Optional[ScannerDispatchEvidence],
    ) -> Tuple[bool, str]:
        """
        Verifies whether a symbol within a published universe possesses explicit State 7 (G-S2-ACT)
        operational dispatch authorization.
        Fails closed: deferred VCP Gate 12 (G-VCP-012) does NOT grant operational scanner dispatch.
        """
        if evidence is None:
            return False, f"DISPATCH_EVIDENCE_MISSING: Operational scanner dispatch evidence absent for {symbol}."

        try:
            evidence.validate()
        except Exception as e:
            return False, f"DISPATCH_VALIDATION_FAILED: {str(e)}"

        if not evidence.g_s2_act_authorized:
            return False, f"G_S2_ACT_UNAUTHORIZED: Operational authorization G-S2-ACT absent for {symbol}."

        return True, f"DISPATCH_AUTHORIZED: {symbol} authorized for scanner {scanner_id} under G-S2-ACT."

    @classmethod
    def build_radar_coverage_disclosure(
        cls,
        pointer: Optional[UniversePublicationPointer],
        scanner_id: str = "MINERVINI_VCP",
        strategy_admission_evidence: Optional[StrategyAdmissionEvidence] = None,
        scanner_dispatch_evidence: Optional[ScannerDispatchEvidence] = None,
        active_snapshot: Optional[Dict[str, Any]] = None,
        deployment_hold: bool = True,
    ) -> Dict[str, Any]:
        """
        Sprint 2 S2-09: Authoritative Radar Coverage Disclosure Engine.
        Traces and resolves the exact 5-tier lifecycle distinction:
        1. NO_COMMITTED_PUBLICATION: No publication pointer committed.
        2. COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED: Publication pointer committed, but scanner not strategy-admitted.
        3. SCANNER_ENROLLED_DISPATCH_UNAUTHORIZED: Strategy-admitted, but State 7 G-S2-ACT operational dispatch unauthorized.
        4. SCANNER_OPERATIONAL: State 7 authorized, active snapshot produced and available.
        5. STALE_OR_ROLLED_BACK_PUBLICATION: Stale or rolled-back pointer detected.

        Invariant Enforcement:
        - AVAILABLE at the membership layer is NEVER translated into live scanner availability.
        - Zero numerical candidate count (0) permitted ONLY when scanner is actively operational and produced 0 matches.
        - For all pending/unauthorized states, matchedCount is strictly None.
        - Every displayed population denominator is strictly bound to universe_build_id, universe_id, and digest.
        """
        # Tier 1: No committed publication pointer
        if pointer is None or not isinstance(pointer, UniversePublicationPointer) or not pointer.published_symbols:
            return {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "populationDenominator": None,
                "universeBuildId": None,
                "universeId": None,
                "publicationDigest": None,
                "isRollback": False,
                "rollbackReason": None,
                "membershipStatus": "UNAVAILABLE",
                "lifecycleState": "NO_COMMITTED_PUBLICATION",
                "reason": "NO_COMMITTED_PUBLICATION_POINTER",
                "rationale": f"Market-wide screening for {scanner_id} is pending authorized universe publication pointer commit.",
            }

        population_denominator = len(pointer.published_symbols)
        build_id = pointer.build_id
        universe_id = pointer.universe_id
        digest = pointer.publication_digest
        is_rollback = pointer.is_rollback
        rollback_reason = pointer.rollback_reason

        # Tier 5 / Degraded: Rolled-back publication pointer cannot imply operational scanner availability (Invariant 6)
        if is_rollback:
            return {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "populationDenominator": population_denominator,
                "universeBuildId": build_id,
                "universeId": universe_id,
                "publicationDigest": digest,
                "isRollback": True,
                "rollbackReason": rollback_reason,
                "membershipStatus": "AVAILABLE",
                "lifecycleState": "STALE_OR_ROLLED_BACK_PUBLICATION",
                "reason": "ROLLED_BACK_PUBLICATION_POINTER",
                "rationale": (
                    f"Universe publication '{build_id}' was rolled back ({rollback_reason or 'No reason specified'}). "
                    f"Scanner {scanner_id} operational screening is suspended pending fresh publication authorization."
                ),
            }

        # Tier 2: Committed publication pointer with valid membership, but scanner not admitted (State 5 -> State 6 pending)
        is_strategy_admitted = False
        if strategy_admission_evidence is not None:
            try:
                strategy_admission_evidence.validate()
                is_strategy_admitted = strategy_admission_evidence.admission_approved
            except Exception:
                is_strategy_admitted = False

        if not is_strategy_admitted:
            return {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "populationDenominator": population_denominator,
                "universeBuildId": build_id,
                "universeId": universe_id,
                "publicationDigest": digest,
                "isRollback": is_rollback,
                "rollbackReason": rollback_reason,
                "membershipStatus": "AVAILABLE",
                "lifecycleState": "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED",
                "reason": "SCANNER_NOT_ENROLLED",
                "rationale": (
                    f"Universe publication '{build_id}' active with {population_denominator} scannable symbols; "
                    f"strategy admission pending for {scanner_id}."
                ),
            }

        # Tier 3: Scanner admitted to strategy, but operational dispatch unauthorized (State 6 -> State 7 pending or deployment hold)
        is_dispatch_authorized = False
        if scanner_dispatch_evidence is not None and not deployment_hold:
            try:
                scanner_dispatch_evidence.validate()
                is_dispatch_authorized = scanner_dispatch_evidence.g_s2_act_authorized and scanner_dispatch_evidence.operational_dispatch_authorized
            except Exception:
                is_dispatch_authorized = False

        if not is_dispatch_authorized:
            hold_reason = "OPERATIONAL_DEPLOYMENT_HOLD" if deployment_hold else "DISPATCH_UNAUTHORIZED_NO_G_S2_ACT"
            hold_detail = (
                f"Scanner {scanner_id} strategy admitted; operational dispatch held under deployment hold/outage."
                if deployment_hold
                else f"Operational dispatch unauthorized for {scanner_id}: requires G-S2-ACT operational authorization."
            )
            return {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "populationDenominator": population_denominator,
                "universeBuildId": build_id,
                "universeId": universe_id,
                "publicationDigest": digest,
                "isRollback": is_rollback,
                "rollbackReason": rollback_reason,
                "membershipStatus": "AVAILABLE",
                "lifecycleState": "SCANNER_ENROLLED_DISPATCH_UNAUTHORIZED",
                "reason": hold_reason,
                "rationale": (
                    f"Universe publication '{build_id}' active ({population_denominator} symbols); "
                    f"{hold_detail}"
                ),
            }

        # Tier 4: Scanner authorized and operational
        snapshot_status = active_snapshot.get("status") if active_snapshot else None
        raw_matched_count = active_snapshot.get("matched_count") if active_snapshot else None
        # Strict Invariants 1, 3, 4: matchedCount must be sourced from completed scanner result (int >= 0, not bool).
        # Incomplete, boolean, negative, or malformed matched_count cannot fabricate 0.
        if snapshot_status == "AVAILABLE" and type(raw_matched_count) is int and raw_matched_count >= 0:
            freshness_info = active_snapshot.get("freshness", {})
            return {
                "status": "AVAILABLE",
                "universeScreening": "AVAILABLE",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": raw_matched_count,
                "populationDenominator": population_denominator,
                "universeBuildId": build_id,
                "universeId": universe_id,
                "publicationDigest": digest,
                "isRollback": False,
                "rollbackReason": None,
                "membershipStatus": "AVAILABLE",
                "lifecycleState": "SCANNER_OPERATIONAL",
                "snapshotId": active_snapshot.get("snapshot_id"),
                "dataAsOf": freshness_info.get("data_as_of") or active_snapshot.get("data_as_of"),
                "generatedAt": freshness_info.get("generated_at") or active_snapshot.get("generated_at"),
                "freshnessStatus": freshness_info.get("status", "LIVE"),
                "reason": "OPERATIONAL_ACTIVE",
                "rationale": (
                    f"Authentic {scanner_id} universe screening active with deterministic publication integrity "
                    f"across {population_denominator} scannable symbols (build '{build_id}')."
                ),
            }

        # Operational dispatch authorized, but snapshot pending or error
        return {
            "status": "PIPELINE_PENDING",
            "universeScreening": "PIPELINE_PENDING",
            "singleAssetAnalysis": "AVAILABLE",
            "matchedCount": None,
            "populationDenominator": population_denominator,
            "universeBuildId": build_id,
            "universeId": universe_id,
            "publicationDigest": digest,
            "isRollback": is_rollback,
            "rollbackReason": rollback_reason,
            "membershipStatus": "AVAILABLE",
            "lifecycleState": "SCANNER_ENROLLED_AWAITING_FIRST_RUN",
            "reason": "AWAITING_FIRST_SCAN_RUN",
            "rationale": (
                f"Scanner {scanner_id} authorized under G-S2-ACT; awaiting initial market-wide execution cycle "
                f"across {population_denominator} scannable symbols."
            ),
        }


