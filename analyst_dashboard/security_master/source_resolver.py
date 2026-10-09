"""
analyst_dashboard/security_master/source_resolver.py

Pure, Deterministic Source Resolution, Reconciliation, Accounting Closure,
and Lifecycle Manager for ARX Security Master Sprint 2A.

Invariants Enforced:
- UNACCOUNTED_RAW_RECORDS = 0
- raw_source_record_count = resolved_count + unresolved_count + quarantined_count
- Admissibility before precedence.
- Closed decision input hash: NO_HIDDEN_SEMANTIC_INPUTS (zero ambient clock reads).
- S2/S3 conflict boundaries strictly enforced.
- Candidate generation is never current truth until validated and atomically promoted.
- Stale promotion CAS prevents race overwrites.
- Last-good generation preservation on validation failure.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from .source_governance_models import (
    canonical_hash,
    canonical_json_dumps,
    RawSourceRecord,
    RawSourceSnapshot,
    CanonicalIssuer,
    CanonicalSecurity,
    CanonicalListing,
    ProviderInstrumentRecord,
    CanonicalFieldDecision,
    SourceConflictRecord,
    MembershipEvent,
    CanonicalGeneration,
    SnapshotStatus,
    ConflictSeverity,
    ConflictResolution,
    MembershipState,
    ListingState,
    MembershipTransitionType,
    PromotionStatus,
    ReasonCode,
    PointInTimeStatus,
    HistoricalMembershipAuthority,
    HistoricalUniverseQueryResult,
    HistoricalMembershipUnavailableError,
)
from .source_governance_policy import (
    FieldAuthorityPolicyRegistry,
    SourceConflictClassifier,
    normalize_symbol_string,
    normalize_exchange_mic,
    CANONICAL_SECURITY_TYPES,
)


class StaleCanonicalPromotionError(RuntimeError):
    """Raised when an attempt is made to promote a candidate generation against a stale predecessor."""
    pass


class ReconciliationIntegrityError(RuntimeError):
    """Raised when raw record accounting or integrity checks fail."""
    pass


class SourceReconciliationEngine:
    """
    Pure, deterministic source reconciliation and canonical generation engine.
    Transforms raw source evidence into auditable canonical decisions.
    """

    def __init__(
        self,
        policy_registry: Optional[FieldAuthorityPolicyRegistry] = None,
        implementation_sha: str = "c868115c7b31b8de6daf4daca47ede049b5bb23b",
    ):
        self.policy_registry = policy_registry or FieldAuthorityPolicyRegistry
        self.implementation_sha = implementation_sha

    def reconcile(
        self,
        source_snapshot: RawSourceSnapshot,
        reference_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        as_of: str = "2026-10-09T00:00:00Z",
        predecessor_generation: Optional[CanonicalGeneration] = None,
        candidate_generation_id: str = "GEN_CANDIDATE_001",
    ) -> CanonicalGeneration:
        """
        Executes complete reconciliation over source_snapshot.
        Guarantees:
        - raw_source_record_count == resolved + unresolved + quarantined
        - unaccounted_raw_records == 0
        - zero ambient clock reads
        """
        ref_data = reference_evidence or {}

        # Section 14 check: Validate enrichment generation coherence
        enrichment_gen_ids = set()
        for k, v in ref_data.items():
            if isinstance(v, dict) and "enrichment_generation_id" in v:
                enrichment_gen_ids.add(v["enrichment_generation_id"])
        if len(enrichment_gen_ids) > 1:
            raise ReconciliationIntegrityError(
                f"MIXED_ENRICHMENT_GENERATIONS_REJECTED: candidate reconciliation cannot mix "
                f"incompatible enrichment generations: {sorted(list(enrichment_gen_ids))}"
            )

        policy_hash = self.policy_registry.compute_policy_hash()

        reconciled_listings: Dict[str, CanonicalListing] = {}
        reconciled_securities: Dict[str, CanonicalSecurity] = {}
        reconciled_issuers: Dict[str, CanonicalIssuer] = {}
        field_decisions: List[CanonicalFieldDecision] = []
        conflicts: List[SourceConflictRecord] = []
        membership_events: List[MembershipEvent] = []

        resolved_count = 0
        unresolved_count = 0
        quarantined_count = 0

        # Collision detection tracking: maps provider_id -> canonical_listing_id
        provider_mapping_registry: Dict[str, str] = {}

        for raw_rec in source_snapshot.records:
            provider_id = raw_rec.provider_record_id
            payload = raw_rec.raw_payload

            # Step 1: Validate raw record schema & integrity
            if not provider_id or not payload:
                quarantined_count += 1
                conflicts.append(SourceConflictRecord(
                    source_conflict_id=f"CONF_CORRUPT_{raw_rec.raw_record_hash[:8]}",
                    canonical_field="provider_record_id",
                    source_a=raw_rec.source_id,
                    value_a=provider_id,
                    evidence_a_hash=raw_rec.raw_record_hash,
                    source_b="SCHEMA_VALIDATOR",
                    value_b="VALID_NON_EMPTY_ID",
                    evidence_b_hash=raw_rec.raw_record_hash,
                    resolution=ConflictResolution.UNRESOLVED,
                    canonical_value=None,
                    severity=ConflictSeverity.S4_INTEGRITY_FAILURE,
                    identity_impact="DEFINITE",
                    denominator_impact="DEFINITE",
                    eligibility_impact="DEFINITE",
                    readiness_impact="DEFINITE",
                    policy_id=FieldAuthorityPolicyRegistry.POLICY_ID,
                    policy_version=FieldAuthorityPolicyRegistry.POLICY_VERSION,
                    decision_hash=canonical_hash({"corrupt": raw_rec.raw_record_hash}),
                ))
                continue

            # Step 2 & 3: Normalization
            raw_sym = str(payload.get("symbol", raw_rec.provider_symbol))
            norm_symbol = normalize_symbol_string(raw_sym)
            raw_exchange = payload.get("exchange", "")
            norm_mic = normalize_exchange_mic(raw_exchange)
            raw_status = str(payload.get("status", "unknown")).upper()
            norm_listing_state = ListingState.ACTIVE if raw_status == "ACTIVE" else (
                ListingState.INACTIVE if raw_status == "INACTIVE" else ListingState.UNKNOWN
            )

            # Reference data (e.g. OpenFIGI enrichment)
            ref_record = ref_data.get(norm_symbol, {})
            ref_sec_type = ref_record.get("security_type", "UNKNOWN")
            ref_composite_figi = ref_record.get("composite_figi")
            ref_share_class_figi = ref_record.get("share_class_figi")
            ref_cik = ref_record.get("cik")

            # Deterministic Identity derivation
            listing_id = f"LST_{norm_mic}_{norm_symbol}"
            sec_id = f"SEC_{ref_share_class_figi}" if ref_share_class_figi else f"SEC_{norm_symbol}"
            issuer_id = f"ISS_{ref_cik}" if ref_cik else f"ISS_{norm_symbol}"

            # Step 4: Cardinality Collision Check
            # 1 provider instrument -> at most 1 canonical listing
            if provider_id in provider_mapping_registry:
                prior_listing = provider_mapping_registry[provider_id]
                if prior_listing != listing_id:
                    # Identity collision detected! Fails closed as S3 Blocking
                    unresolved_count += 1
                    conflicts.append(SourceConflictRecord(
                        source_conflict_id=f"CONF_COLLISION_{provider_id[:8]}",
                        canonical_field="canonical_listing_id",
                        canonical_listing_id=listing_id,
                        source_a=raw_rec.source_id,
                        value_a=listing_id,
                        evidence_a_hash=raw_rec.raw_record_hash,
                        source_b=raw_rec.source_id,
                        value_b=prior_listing,
                        evidence_b_hash=raw_rec.raw_record_hash,
                        resolution=ConflictResolution.UNRESOLVED,
                        canonical_value=None,
                        severity=ConflictSeverity.S3_BLOCKING,
                        identity_impact="DEFINITE",
                        denominator_impact="DEFINITE",
                        eligibility_impact="DEFINITE",
                        readiness_impact="DEFINITE",
                        policy_id=FieldAuthorityPolicyRegistry.POLICY_ID,
                        policy_version=FieldAuthorityPolicyRegistry.POLICY_VERSION,
                        decision_hash=canonical_hash({"collision": provider_id, "a": listing_id, "b": prior_listing}),
                    ))
                    continue
            else:
                provider_mapping_registry[provider_id] = listing_id

            # Step 5: Field Decisions with Closed Decision Input Hash
            # Decision 1: Symbol
            sym_input_hash = canonical_hash({
                "raw_sym": raw_sym,
                "norm_sym": norm_symbol,
                "policy": "POL_SYMBOL_V1",
                "as_of": as_of,
            })
            d_sym = CanonicalFieldDecision(
                field_decision_id=f"DEC_SYM_{listing_id}",
                canonical_listing_id=listing_id,
                canonical_security_id=sec_id,
                canonical_field="symbol",
                canonical_value=norm_symbol,
                winning_source_id=raw_rec.source_id,
                winning_snapshot_id=source_snapshot.source_snapshot_id,
                winning_record_hash=raw_rec.raw_record_hash,
                competing_source_ids=[raw_rec.source_id],
                competing_record_hashes=[raw_rec.raw_record_hash],
                field_policy_id="POL_SYMBOL_V1",
                field_policy_version="1.0.0",
                field_policy_hash=policy_hash,
                rule_id="RULE_SYMBOL_NORM_V1",
                severity=ConflictSeverity.S0_INFO,
                as_of=as_of,
                decision_input_hash=sym_input_hash,
                implementation_sha=self.implementation_sha,
            )
            field_decisions.append(d_sym)

            # Decision 2: Primary Exchange
            exch_input_hash = canonical_hash({
                "raw_exchange": raw_exchange,
                "norm_mic": norm_mic,
                "policy": "POL_EXCHANGE_V1",
                "as_of": as_of,
            })
            d_exch = CanonicalFieldDecision(
                field_decision_id=f"DEC_EXCH_{listing_id}",
                canonical_listing_id=listing_id,
                canonical_security_id=sec_id,
                canonical_field="primary_exchange",
                canonical_value=norm_mic,
                winning_source_id=raw_rec.source_id,
                winning_snapshot_id=source_snapshot.source_snapshot_id,
                winning_record_hash=raw_rec.raw_record_hash,
                competing_source_ids=[raw_rec.source_id],
                competing_record_hashes=[raw_rec.raw_record_hash],
                field_policy_id="POL_EXCHANGE_V1",
                field_policy_version="1.0.0",
                field_policy_hash=policy_hash,
                rule_id="RULE_EXCHANGE_MIC_NORM_V1",
                severity=ConflictSeverity.S0_INFO,
                as_of=as_of,
                decision_input_hash=exch_input_hash,
                implementation_sha=self.implementation_sha,
            )
            field_decisions.append(d_exch)

            # Decision 3: Listing Status
            status_input_hash = canonical_hash({
                "raw_status": raw_status,
                "norm_status": norm_listing_state.value,
                "policy": "POL_LISTING_STATUS_V1",
                "as_of": as_of,
            })
            d_status = CanonicalFieldDecision(
                field_decision_id=f"DEC_STATUS_{listing_id}",
                canonical_listing_id=listing_id,
                canonical_security_id=sec_id,
                canonical_field="listing_status",
                canonical_value=norm_listing_state.value,
                winning_source_id=raw_rec.source_id,
                winning_snapshot_id=source_snapshot.source_snapshot_id,
                winning_record_hash=raw_rec.raw_record_hash,
                competing_source_ids=[raw_rec.source_id],
                competing_record_hashes=[raw_rec.raw_record_hash],
                field_policy_id="POL_LISTING_STATUS_V1",
                field_policy_version="1.0.0",
                field_policy_hash=policy_hash,
                rule_id="RULE_LISTING_STATUS_V1",
                severity=ConflictSeverity.S0_INFO,
                as_of=as_of,
                decision_input_hash=status_input_hash,
                implementation_sha=self.implementation_sha,
            )
            field_decisions.append(d_status)

            # Decision 4: Security Type (OpenFIGI is primary authority; Alpaca broad class is non-authoritative)
            # Check conflict if Alpaca says us_equity while OpenFIGI says Common Stock, ETF, ADR, etc.
            figi_policy = FieldAuthorityPolicyRegistry.POLICIES["security_type"]
            c_sev, c_res, winning_val = SourceConflictClassifier.classify_conflict(
                field_name="security_type",
                value_a=ref_sec_type,
                value_b="us_equity",
                policy=figi_policy,
                source_a="OPENFIGI_V3_MAPPING",
                source_b="ALPACA_ASSET_DIRECTORY",
            )
            if c_sev != ConflictSeverity.S0_INFO:
                conflicts.append(SourceConflictRecord(
                    source_conflict_id=f"CONF_TYPE_{norm_symbol}",
                    canonical_field="security_type",
                    canonical_listing_id=listing_id,
                    canonical_security_id=sec_id,
                    source_a="OPENFIGI_V3_MAPPING",
                    value_a=ref_sec_type,
                    evidence_a_hash=canonical_hash(ref_record),
                    source_b="ALPACA_ASSET_DIRECTORY",
                    value_b="us_equity",
                    evidence_b_hash=raw_rec.raw_record_hash,
                    resolution=c_res,
                    canonical_value=winning_val,
                    severity=c_sev,
                    identity_impact="NONE",
                    denominator_impact="NONE",
                    eligibility_impact="DEFINITE",
                    readiness_impact="NONE",
                    policy_id="POL_SECURITY_TYPE_V1",
                    policy_version="1.0.0",
                    decision_hash=canonical_hash({"sym": norm_symbol, "type_res": winning_val}),
                ))

            # Security type value
            canonical_sec_type = winning_val if winning_val in CANONICAL_SECURITY_TYPES else "UNKNOWN"

            type_input_hash = canonical_hash({
                "ref_sec_type": ref_sec_type,
                "alpaca_class": payload.get("class"),
                "winning_type": canonical_sec_type,
                "policy": "POL_SECURITY_TYPE_V1",
                "as_of": as_of,
            })
            d_type = CanonicalFieldDecision(
                field_decision_id=f"DEC_TYPE_{sec_id}",
                canonical_security_id=sec_id,
                canonical_field="security_type",
                canonical_value=canonical_sec_type,
                winning_source_id="OPENFIGI_V3_MAPPING" if ref_sec_type != "UNKNOWN" else raw_rec.source_id,
                winning_snapshot_id=source_snapshot.source_snapshot_id,
                winning_record_hash=raw_rec.raw_record_hash,
                competing_source_ids=["OPENFIGI_V3_MAPPING", raw_rec.source_id],
                competing_record_hashes=[canonical_hash(ref_record), raw_rec.raw_record_hash],
                field_policy_id="POL_SECURITY_TYPE_V1",
                field_policy_version="1.0.0",
                field_policy_hash=policy_hash,
                rule_id="RULE_SECURITY_TYPE_V1",
                severity=c_sev,
                as_of=as_of,
                decision_input_hash=type_input_hash,
                implementation_sha=self.implementation_sha,
            )
            field_decisions.append(d_type)

            # Construct Canonical Entities
            c_listing = CanonicalListing(
                canonical_listing_id=listing_id,
                canonical_security_id=sec_id,
                symbol=norm_symbol,
                canonical_mic=norm_mic,
                listing_status=norm_listing_state,
                composite_figi=ref_composite_figi,
                effective_from=as_of,  # Valid time anchors strictly to explicit as_of
            )
            reconciled_listings[listing_id] = c_listing

            if sec_id not in reconciled_securities:
                raw_class = payload.get("class", "us_equity")
                c_sec = CanonicalSecurity(
                    canonical_security_id=sec_id,
                    canonical_issuer_id=issuer_id,
                    security_type=canonical_sec_type,
                    provider_asset_class=raw_class.upper() if raw_class else "US_EQUITY",
                    enrichment_status="ENRICHED" if canonical_sec_type != "UNKNOWN" else "AWAITING_ENRICHMENT",
                    share_class_figi=ref_share_class_figi,
                )
                reconciled_securities[sec_id] = c_sec

            # Step 6: Temporal Membership Event
            # Transition evaluation against predecessor
            if predecessor_generation and listing_id in predecessor_generation.reconciled_listings:
                pred_listing = predecessor_generation.reconciled_listings[listing_id]
                if pred_listing.listing_status != norm_listing_state:
                    trans_type = MembershipTransitionType.LISTING_STATUS_CHANGED
                elif pred_listing.canonical_mic != norm_mic:
                    trans_type = MembershipTransitionType.EXCHANGE_TRANSFERRED
                else:
                    trans_type = MembershipTransitionType.ADDED_NEW_LISTING
            else:
                trans_type = MembershipTransitionType.ADDED_NEW_LISTING

            m_event = MembershipEvent(
                membership_event_id=f"EVT_{listing_id}_{as_of}",
                canonical_listing_id=listing_id,
                source_snapshot_id=source_snapshot.source_snapshot_id,
                source_authority_id=raw_rec.source_id,
                membership_state=MembershipState.PRESENT,
                listing_state=norm_listing_state,
                effective_from=as_of,
                observed_at=raw_rec.observed_at,
                transition_type=trans_type,
                reason_code=ReasonCode.IDENTITY_RESOLVED,
                source_record_hash=raw_rec.raw_record_hash,
                decision_hash=c_listing.listing_hash,
            )
            membership_events.append(m_event)

            # Classify raw record reconciliation state
            if canonical_sec_type == "UNKNOWN":
                # Partial enrichment: in source population but subtype unresolved
                unresolved_count += 1
            else:
                resolved_count += 1

        # Check predecessor for removed listings (listings that disappeared)
        if predecessor_generation:
            current_listing_ids = set(reconciled_listings.keys())
            for pred_id, pred_listing in predecessor_generation.reconciled_listings.items():
                if pred_id not in current_listing_ids:
                    # Listing disappeared from snapshot. MUST NOT assume delisted without proof!
                    m_removal = MembershipEvent(
                        membership_event_id=f"EVT_REM_{pred_id}_{as_of}",
                        canonical_listing_id=pred_id,
                        source_snapshot_id=source_snapshot.source_snapshot_id,
                        source_authority_id=source_snapshot.source_id,
                        membership_state=MembershipState.ABSENT,
                        listing_state=ListingState.UNKNOWN,
                        effective_from=as_of,
                        observed_at=as_of,
                        transition_type=MembershipTransitionType.UNRESOLVED_REMOVAL,
                        reason_code=ReasonCode.UNRESOLVED_REMOVAL,
                        source_record_hash="REMOVAL_FROM_SNAPSHOT",
                        decision_hash=canonical_hash({"removed": pred_id, "as_of": as_of}),
                    )
                    membership_events.append(m_removal)

        # Enforce Accounting Closure
        raw_count = len(source_snapshot.records)
        unaccounted = raw_count - (resolved_count + unresolved_count + quarantined_count)
        if unaccounted != 0:
            raise ReconciliationIntegrityError(
                f"ACCOUNTING CLOSURE FAILURE: raw={raw_count}, resolved={resolved_count}, "
                f"unresolved={unresolved_count}, quarantined={quarantined_count}, unaccounted={unaccounted}"
            )

        accounting = {
            "raw_source_record_count": raw_count,
            "resolved_record_count": resolved_count,
            "unresolved_record_count": unresolved_count,
            "quarantined_record_count": quarantined_count,
            "unaccounted_raw_records": 0,
            "canonical_security_count": len(reconciled_securities),
            "canonical_listing_count": len(reconciled_listings),
        }

        # Build candidate generation (starts as CANDIDATE, never active before validation)
        generation = CanonicalGeneration(
            generation_id=candidate_generation_id,
            predecessor_generation_id=predecessor_generation.generation_id if predecessor_generation else None,
            source_snapshot_id=source_snapshot.source_snapshot_id,
            policy_generation_id=f"{FieldAuthorityPolicyRegistry.POLICY_ID}_{FieldAuthorityPolicyRegistry.POLICY_VERSION}",
            as_of=as_of,
            reconciled_listings=reconciled_listings,
            reconciled_securities=reconciled_securities,
            field_decisions=field_decisions,
            conflicts=conflicts,
            membership_events=membership_events,
            accounting_summary=accounting,
            validation_status="PENDING",
            promotion_status=PromotionStatus.CANDIDATE,
        )

        build_hash = generation.compute_build_hash()
        object.__setattr__(generation, "build_hash", build_hash)
        return generation

    def query_point_in_time_universe(
        self,
        requested_as_of: str,
        historical_authority_coverage_start: str = "2026-10-09T00:00:00Z",
        active_generation: Optional[CanonicalGeneration] = None,
        fail_closed: bool = False,
    ) -> HistoricalUniverseQueryResult:
        """
        Queries point-in-time universe under Sprint 2A Section 8 invariants.
        
        Hard Invariant (Sprint 2A Section 8):
        UNKNOWN_HISTORICAL_POPULATION != EMPTY_HISTORICAL_POPULATION
        - If requested_as_of < historical_authority_coverage_start:
          returns point_in_time_status = NOT_AVAILABLE with
          authoritative_denominator = None (never 0) and listings = None (never []).
        - If fail_closed is True, raises HistoricalMembershipUnavailableError.
        """
        if requested_as_of < historical_authority_coverage_start:
            if fail_closed:
                raise HistoricalMembershipUnavailableError(
                    f"HISTORICAL_MEMBERSHIP_UNAVAILABLE: requested_as_of '{requested_as_of}' "
                    f"precedes historical authority coverage start '{historical_authority_coverage_start}'. "
                    f"UNKNOWN_HISTORICAL_POPULATION != EMPTY_HISTORICAL_POPULATION."
                )
            return HistoricalUniverseQueryResult(
                requested_as_of=requested_as_of,
                historical_membership_authority=HistoricalMembershipAuthority.CURRENT_ONLY,
                point_in_time_status=PointInTimeStatus.NOT_AVAILABLE,
                authoritative_denominator=None,
                listings=None,
                reason="HISTORICAL_MEMBERSHIP_UNAVAILABLE: as_of precedes coverage start. UNKNOWN != EMPTY.",
            )

        if active_generation is None:
            return HistoricalUniverseQueryResult(
                requested_as_of=requested_as_of,
                historical_membership_authority=HistoricalMembershipAuthority.CURRENT_ONLY,
                point_in_time_status=PointInTimeStatus.NOT_AVAILABLE,
                authoritative_denominator=None,
                listings=None,
                reason="NO_ACTIVE_CANONICAL_GENERATION_AVAILABLE",
            )

        active_listings = [
            lst for lst in active_generation.reconciled_listings.values()
            if lst.listing_status == ListingState.ACTIVE
        ]
        return HistoricalUniverseQueryResult(
            requested_as_of=requested_as_of,
            historical_membership_authority=HistoricalMembershipAuthority.POINT_IN_TIME_VERIFIED,
            point_in_time_status=PointInTimeStatus.AVAILABLE,
            authoritative_denominator=len(active_listings),
            listings=active_listings,
            reason="POINT_IN_TIME_AVAILABLE",
        )


class GenerationLifecycleManager:
    """
    Coordinates Candidate -> Validate -> Atomic Promotion lifecycle with Compare-And-Swap (CAS).
    Preserves last-good generation on failure.
    """

    def __init__(self):
        self.active_generation: Optional[CanonicalGeneration] = None
        self.last_good_generation: Optional[CanonicalGeneration] = None

    def validate_candidate(self, candidate: CanonicalGeneration) -> bool:
        """
        Validates candidate generation integrity:
        - Accounting closure: unaccounted == 0
        - S4 integrity failure count == 0
        - S3 blocking conflicts == 0 without resolution
        """
        accounting = candidate.accounting_summary
        if accounting.get("unaccounted_raw_records", 999) != 0:
            object.__setattr__(candidate, "validation_status", "FAILED_ACCOUNTING")
            return False

        # Check S4 or unresolved S3
        for conf in candidate.conflicts:
            if conf.severity == ConflictSeverity.S4_INTEGRITY_FAILURE:
                object.__setattr__(candidate, "validation_status", "FAILED_S4_INTEGRITY")
                return False
            if conf.severity == ConflictSeverity.S3_BLOCKING and conf.resolution == ConflictResolution.UNRESOLVED:
                object.__setattr__(candidate, "validation_status", "FAILED_UNRESOLVED_S3")
                return False

        object.__setattr__(candidate, "validation_status", "PASSED")
        return True

    def promote_candidate(
        self,
        candidate: CanonicalGeneration,
        expected_predecessor_generation_id: Optional[str] = None,
        promoted_at: str = "2026-10-09T00:00:00Z",
    ) -> bool:
        """
        Atomically promotes validated candidate using Compare-and-Swap (CAS).
        Fails if expected_predecessor_generation_id does not match active_generation.generation_id.
        Preserves last_good_generation if promotion or validation fails.
        """
        if candidate.validation_status != "PASSED":
            # Attempt validation first
            if not self.validate_candidate(candidate):
                object.__setattr__(candidate, "promotion_status", PromotionStatus.REJECTED)
                return False

        current_active_id = self.active_generation.generation_id if self.active_generation else None
        if expected_predecessor_generation_id != current_active_id:
            object.__setattr__(candidate, "promotion_status", PromotionStatus.STALE_REJECTED)
            raise StaleCanonicalPromotionError(
                f"STALE_PROMOTION_REJECTED: expected predecessor '{expected_predecessor_generation_id}' "
                f"does not match current active generation '{current_active_id}'."
            )

        # Atomic promotion
        object.__setattr__(candidate, "promotion_status", PromotionStatus.PROMOTED)
        object.__setattr__(candidate, "promoted_at", promoted_at)

        self.last_good_generation = self.active_generation
        self.active_generation = candidate
        return True
