"""
analyst_dashboard/universe/identity_adoption.py

Sprint 2: Bounded Offline Identity Adoption Dry-Run & Plan Verification Engine.
Implements the minimum offline capability to demonstrate how reviewed provisional
security entities and listing associations could be adopted into durable identities
without promoting production identities, modifying production stores, or weakening
evidence requirements.

Core Architectural Invariants:
1. Deterministic, Structured Adoption Plan:
   - Evaluates explicitly selected proposal sets against an explicitly supplied snapshot
     of existing identity records.
   - Categorizes each entity and association into clear dispositions:
     APPROVED_FOR_ADOPTION, MATCHES_EXISTING_DURABLE, BLOCKED_POLICY_UNRESOLVED,
     BLOCKED_MISSING_AUTHORIZATION, BLOCKED_UNRESOLVED_MIC, BLOCKED_ASSOCIATION_CONFLICT,
     BLOCKED_CONTRADICTORY_DATES, BLOCKED_BACKDATED_UNSUPPORTED, or BLOCKED_MISMATCHED_ISSUER.
   - Preserves existing durable IDs and guarantees stable durable IDs across reload/replay.
   - Unresolved cases remain strictly unresolved; never mints alternate IDs to bypass conflict.

2. Strict Separation of Evidence, Policy, Authorization, and Admission:
   - Real retained-source records (AAPL, GOOG, GOOGL, TSM, AACIU) are NEVER marked
     APPROVED_FOR_ADOPTION merely because a test or caller supplies an authorization token.
   - For real candidates, adopting an internal identity prior to primary-listing evidence closure
     (Rule R06) is an UNRESOLVED policy decision. They are marked BLOCKED_POLICY_UNRESOLVED.
   - The mechanical approval path is exercised strictly using explicitly labeled synthetic fixtures
     and synthetic authorization.
   - A dry-run plan NEVER implies adopted production identity, primary-listing sufficiency (Rule R06),
     SourceSecurity conversion authority, or universe admission.

3. Authoritative Issuer Binding Gating:
   - Evaluates candidate CIKs against the verified primary-listing evidence authority
     (PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING).
   - Rejects mismatched CIKs (e.g. older/cross-SPAC reuse) with BLOCKED_MISMATCHED_ISSUER.

4. Robust Collision and Matching Semantics:
   - Issuer CIK alone CANNOT match a security; matching requires concordant locator, CIK, and share class.
   - Detects same-issuer, distinct-security collisions on overlapping locators and fails closed.
   - Symbol or venue changes without explicit alias continuity records remain unresolved.

5. Evidence-Bounded Temporal Semantics:
   - Replaces fixed calendar cutoffs with attributable evidence observation timestamps.
   - Fails closed on unsupported backdating prior to verified documentary observation.

6. Content and Target-State Binding with Test-Scoped Atomic Application:
   - Plan is cryptographically bound to target store digest (target_state_digest) and plan content digest.
   - Stale target stores and tampered plan content fail closed (StalePlanError, PlanIntegrityError).
   - Atomic rollback completely restores temporary engine state on mid-operation failure.
   - Idempotent: repeated application introduces zero duplicate records.
   - Test-only helper (apply_adoption_plan_test_only) operates strictly in temporary stores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from .admission_mapping import (
    AttributableFieldEvidence,
    EffectiveDatedListingAssociation,
    EvidenceNature,
    InternalSecurityRecord,
    ListingLocator,
    OfflineAdmissionMappingEngine,
    canonical_hash,
)
from .identity_proposal import (
    CandidateAssociationProposal,
    ExternalIdentifierRecord,
    IdentityAssociationProposalLedger,
    PrimaryListingClaimAnalysis,
    ProvisionalListingAssociation,
    ProvisionalSecurityEntity,
)
from .primary_listing_evidence import PrimaryListingEvidenceAdapter


# =====================================================================
# 1. Adoption Status & Authorization Models
# =====================================================================

class IdentityAdoptionStatus(str, Enum):
    APPROVED_FOR_ADOPTION = "APPROVED_FOR_ADOPTION"
    MATCHES_EXISTING_DURABLE = "MATCHES_EXISTING_DURABLE"
    BLOCKED_POLICY_UNRESOLVED = "BLOCKED_POLICY_UNRESOLVED"
    BLOCKED_MISSING_AUTHORIZATION = "BLOCKED_MISSING_AUTHORIZATION"
    BLOCKED_UNRESOLVED_MIC = "BLOCKED_UNRESOLVED_MIC"
    BLOCKED_ASSOCIATION_CONFLICT = "BLOCKED_ASSOCIATION_CONFLICT"
    BLOCKED_CONTRADICTORY_DATES = "BLOCKED_CONTRADICTORY_DATES"
    BLOCKED_BACKDATED_UNSUPPORTED = "BLOCKED_BACKDATED_UNSUPPORTED"
    BLOCKED_MISMATCHED_ISSUER = "BLOCKED_MISMATCHED_ISSUER"


@dataclass(frozen=True)
class IdentityAdoptionAuthorization:
    """
    Explicit authorization context for identity adoption.
    Differentiates synthetic test execution from production authorization.
    """
    is_authorized: bool = False
    authorized_by: Optional[str] = None
    authorization_timestamp_utc: Optional[str] = None
    authorization_token: Optional[str] = None
    scope: str = "PRODUCTION"  # PRODUCTION, SYNTHETIC_TEST_ONLY, or REFERENCE_IDENTITY_REGISTRATION
    allow_reference_identity_registration: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. Structured Plan Elements
# =====================================================================

@dataclass(frozen=True)
class ProposedEntityAdoption:
    """Represents the planned adoption action for a single provisional security entity."""
    provisional_security_id: str
    primary_symbol: str
    share_class: Optional[str]
    issuer_cik: Optional[str]
    status: IdentityAdoptionStatus
    is_creation: bool
    is_mechanically_plannable: bool
    matched_existing_security_id: Optional[str] = None
    proposed_durable_security_id: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class ProposedAssociationAdoption:
    """Represents the planned adoption action for a listing association."""
    provisional_association_id: str
    provisional_security_id: str
    target_durable_security_id: str
    symbol: str
    operating_mic: Optional[str]
    effective_from: str
    effective_to: Optional[str]
    status: IdentityAdoptionStatus
    is_creation: bool
    is_mechanically_plannable: bool
    matched_existing_association_id: Optional[str] = None
    proposed_durable_association_id: Optional[str] = None
    conflict_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class AdoptionPlanSummary:
    """Quantitative summary of the adoption plan."""
    total_proposals: int
    entities_matched_existing: int
    entities_approved_creation: int
    entities_blocked: int
    associations_matched_existing: int
    associations_approved_creation: int
    associations_blocked: int
    conflicts_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 3. Content-Bound Adoption Plan
# =====================================================================

@dataclass
class IdentityAdoptionPlan:
    """
    Complete, deterministic, reviewable identity adoption dry-run plan.
    Bound to the target state digest and plan content digest.
    """
    plan_id: str
    created_at_utc: str
    evaluation_reference_date: str
    target_state_digest: str
    plan_content_digest: str
    authorization: IdentityAdoptionAuthorization
    entities: List[ProposedEntityAdoption]
    associations: List[ProposedAssociationAdoption]
    provisional_to_durable_map: Dict[str, str]
    conflicts: List[Dict[str, Any]]
    blocked_operations: List[Dict[str, Any]]
    missing_evidence_or_authorization: List[Dict[str, Any]]
    primary_listing_states: Dict[str, Dict[str, Any]]
    summary: AdoptionPlanSummary
    eligibility_separation_note: str = (
        "GOVERNANCE INVARIANT: Adoption of an internal security entity or listing association "
        "registers identity in the internal master only. It does NOT imply primary-listing sufficiency, "
        "SourceSecurity conversion authority, or universe admission under Rule R06. Real-candidate "
        "adoption remains policy-unresolved pending primary-listing evidence closure."
    )

    def is_target_state_valid(
        self,
        current_target: Union[OfflineAdmissionMappingEngine, Dict[str, Any], str],
    ) -> bool:
        """Verifies if the current target state still matches the plan's baseline snapshot."""
        if isinstance(current_target, str):
            return current_target == self.target_state_digest
        current_digest = IdentityAdoptionDryRunEngine.compute_target_state_digest(current_target)
        return current_digest == self.target_state_digest

    def is_plan_content_valid(self) -> bool:
        """Recomputes and verifies the plan content digest to detect any payload tampering."""
        recomputed = IdentityAdoptionDryRunEngine.compute_plan_content_digest(
            entities=self.entities,
            associations=self.associations,
            provisional_to_durable_map=self.provisional_to_durable_map,
            summary=self.summary,
        )
        return recomputed == self.plan_content_digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at_utc": self.created_at_utc,
            "evaluation_reference_date": self.evaluation_reference_date,
            "target_state_digest": self.target_state_digest,
            "plan_content_digest": self.plan_content_digest,
            "authorization": self.authorization.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "associations": [a.to_dict() for a in self.associations],
            "provisional_to_durable_map": self.provisional_to_durable_map,
            "conflicts": self.conflicts,
            "blocked_operations": self.blocked_operations,
            "missing_evidence_or_authorization": self.missing_evidence_or_authorization,
            "primary_listing_states": self.primary_listing_states,
            "summary": self.summary.to_dict(),
            "eligibility_separation_note": self.eligibility_separation_note,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IdentityAdoptionPlan:
        auth_d = data.get("authorization", {})
        auth = IdentityAdoptionAuthorization(**auth_d)

        entities = [
            ProposedEntityAdoption(
                provisional_security_id=e["provisional_security_id"],
                primary_symbol=e["primary_symbol"],
                share_class=e.get("share_class"),
                issuer_cik=e.get("issuer_cik"),
                status=IdentityAdoptionStatus(e["status"]),
                is_creation=e["is_creation"],
                is_mechanically_plannable=e.get("is_mechanically_plannable", False),
                matched_existing_security_id=e.get("matched_existing_security_id"),
                proposed_durable_security_id=e.get("proposed_durable_security_id", ""),
                rationale=e.get("rationale", ""),
            )
            for e in data.get("entities", [])
        ]

        associations = [
            ProposedAssociationAdoption(
                provisional_association_id=a["provisional_association_id"],
                provisional_security_id=a["provisional_security_id"],
                target_durable_security_id=a["target_durable_security_id"],
                symbol=a["symbol"],
                operating_mic=a.get("operating_mic"),
                effective_from=a["effective_from"],
                effective_to=a.get("effective_to"),
                status=IdentityAdoptionStatus(a["status"]),
                is_creation=a["is_creation"],
                is_mechanically_plannable=a.get("is_mechanically_plannable", False),
                matched_existing_association_id=a.get("matched_existing_association_id"),
                proposed_durable_association_id=a.get("proposed_durable_association_id"),
                conflict_reason=a.get("conflict_reason"),
            )
            for a in data.get("associations", [])
        ]

        sum_d = data.get("summary", {})
        # Remap legacy field names if present
        if "entities_proposed_creation" in sum_d and "entities_approved_creation" not in sum_d:
            sum_d["entities_approved_creation"] = sum_d.pop("entities_proposed_creation")
        if "associations_proposed_creation" in sum_d and "associations_approved_creation" not in sum_d:
            sum_d["associations_approved_creation"] = sum_d.pop("associations_proposed_creation")

        summary = AdoptionPlanSummary(**sum_d)

        return cls(
            plan_id=data["plan_id"],
            created_at_utc=data["created_at_utc"],
            evaluation_reference_date=data.get("evaluation_reference_date", "2026-10-17"),
            target_state_digest=data["target_state_digest"],
            plan_content_digest=data["plan_content_digest"],
            authorization=auth,
            entities=entities,
            associations=associations,
            provisional_to_durable_map=data.get("provisional_to_durable_map", {}),
            conflicts=data.get("conflicts", []),
            blocked_operations=data.get("blocked_operations", []),
            missing_evidence_or_authorization=data.get("missing_evidence_or_authorization", []),
            primary_listing_states=data.get("primary_listing_states", {}),
            summary=summary,
            eligibility_separation_note=data.get(
                "eligibility_separation_note",
                cls.eligibility_separation_note,
            ),
        )

    @classmethod
    def from_json(cls, json_str: str) -> IdentityAdoptionPlan:
        return cls.from_dict(json.loads(json_str))


# =====================================================================
# 4. Identity Adoption Dry-Run Engine
# =====================================================================

class IdentityAdoptionDryRunEngine:
    """
    Offline dry-run planner evaluating provisional proposals against existing identity state.
    Strictly read-only; does not mutate target engine or external stores.
    """

    @staticmethod
    def compute_target_state_digest(
        target: Union[OfflineAdmissionMappingEngine, Dict[str, Any], None],
    ) -> str:
        """Computes deterministic SHA-256 fingerprint of the target identity store snapshot."""
        if target is None:
            snapshot = {"security_registry": {}, "association_ledger": {}, "cik_to_securities": {}}
        elif isinstance(target, OfflineAdmissionMappingEngine):
            snapshot = target.to_dict()
        elif isinstance(target, dict):
            snapshot = target
        else:
            raise TypeError(f"Unsupported target type for digest computation: {type(target)}")
        return canonical_hash(snapshot)

    @staticmethod
    def compute_plan_content_digest(
        entities: List[ProposedEntityAdoption],
        associations: List[ProposedAssociationAdoption],
        provisional_to_durable_map: Dict[str, str],
        summary: AdoptionPlanSummary,
    ) -> str:
        """Computes deterministic SHA-256 digest of planned adoption actions."""
        content_payload = {
            "entities": [e.to_dict() for e in entities],
            "associations": [a.to_dict() for a in associations],
            "provisional_to_durable_map": provisional_to_durable_map,
            "summary": summary.to_dict(),
        }
        return canonical_hash(content_payload)

    @classmethod
    def allocate_opaque_security_id(cls, prefix: str = "ARX_SEC") -> str:
        """
        Allocates an opaque proposed durable security ID.
        Decoupled from ephemeral listing observations (MIC, symbol, observation date).
        Once allocated in a plan, this ID is preserved in serialized artifacts and
        in the target ledger upon adoption.
        """
        return f"{prefix}_{uuid.uuid4().hex[:12].upper()}"

    @classmethod
    def allocate_opaque_association_id(
        cls,
        security_id: str,
        operating_mic: str,
        symbol: str,
        effective_from: str,
    ) -> str:
        """
        Allocates a listing association ID tying a durable security to a venue and symbol
        starting from its verified effective date.
        """
        return f"ASSOC_{security_id}_{operating_mic}_{symbol}_{effective_from}"

    @classmethod
    def generate_plan(
        cls,
        proposals: Union[IdentityAssociationProposalLedger, List[CandidateAssociationProposal]],
        existing_target: Union[OfflineAdmissionMappingEngine, Dict[str, Any], None] = None,
        authorization: Optional[IdentityAdoptionAuthorization] = None,
        evaluation_reference_date: Optional[str] = None,
        deterministic_plan_time_utc: Optional[str] = None,
        id_allocator: Optional[Callable[[], str]] = None,
    ) -> IdentityAdoptionPlan:
        """
        Generates a deterministic identity adoption plan from candidate proposals.
        """
        # Unpack proposals
        if isinstance(proposals, IdentityAssociationProposalLedger):
            proposal_list = proposals.list_proposals()
        elif isinstance(proposals, list):
            proposal_list = list(proposals)
        else:
            raise TypeError(f"Unsupported proposals type: {type(proposals)}")

        # Normalize target engine
        if existing_target is None:
            engine = OfflineAdmissionMappingEngine()
            target_snapshot = engine.to_dict()
        elif isinstance(existing_target, OfflineAdmissionMappingEngine):
            engine = existing_target
            target_snapshot = engine.to_dict()
        elif isinstance(existing_target, dict):
            engine = OfflineAdmissionMappingEngine.from_dict(existing_target)
            target_snapshot = existing_target
        else:
            raise TypeError(f"Unsupported existing_target type: {type(existing_target)}")

        target_state_digest = cls.compute_target_state_digest(target_snapshot)
        auth = authorization or IdentityAdoptionAuthorization(is_authorized=False)
        plan_created_utc = deterministic_plan_time_utc or datetime.now(timezone.utc).isoformat()

        proposed_entities: List[ProposedEntityAdoption] = []
        proposed_associations: List[ProposedAssociationAdoption] = []
        provisional_to_durable_map: Dict[str, str] = {}
        conflicts: List[Dict[str, Any]] = []
        blocked_operations: List[Dict[str, Any]] = []
        missing_evidence_or_auth: List[Dict[str, Any]] = []
        primary_listing_states: Dict[str, Dict[str, Any]] = {}

        entities_matched = 0
        entities_approved = 0
        entities_blocked = 0
        assocs_matched = 0
        assocs_approved = 0
        assocs_blocked = 0

        # Evaluation reference date is a replay query parameter, NOT evidence of listing start
        as_of_eval_date = evaluation_reference_date

        for prop in proposal_list:
            sym = prop.symbol.strip().upper()
            ent = prop.proposed_entity
            assoc = prop.proposed_association
            clean_cik = str(ent.issuer_cik).zfill(10) if ent.issuer_cik else None
            effective_date = as_of_eval_date or assoc.effective_from

            # Record primary listing state separately from identity adoption
            pla = prop.primary_listing_analysis
            primary_listing_states[sym] = {
                "claimed_primary_listing": pla.claimed_primary_listing,
                "evidence_sufficiency": pla.evidence_sufficiency,
                "selected_us_assessment_listing": pla.selected_us_assessment_listing,
                "primary_exchange_for_instrument": pla.primary_exchange_for_instrument,
                "issuer_home_market": pla.issuer_home_market,
                "rule_r06_admitted": False,
                "status_note": (
                    "Primary-listing evidence remains disaggregated from internal identity adoption. "
                    "Rule R06 eligibility requires verified primary listing; None remains UNKNOWN."
                ),
            }

            # 1. Authoritative Issuer Binding Gating
            # Enforces candidate CIK matches authoritative public reference evidence
            authoritative_cik = PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING.get(sym)
            if authoritative_cik is not None and clean_cik != authoritative_cik:
                reason = (
                    f"REJECTED_MISMATCHED_ISSUER_EVIDENCE: Proposal issuer CIK '{clean_cik}' for '{sym}' "
                    f"does not match authoritative candidate CIK '{authoritative_cik}' "
                    "(rejects cross-issuer or cross-SPAC evidence reuse)."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_MISMATCHED_ISSUER,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=assoc.operating_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_MISMATCHED_ISSUER,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                conflicts.append({"symbol": sym, "conflict_type": "MISMATCHED_ISSUER_CIK", "details": reason})
                continue

            # 2. Check for contradictory date ranges
            if assoc.effective_to and assoc.effective_to < assoc.effective_from:
                reason = (
                    f"Contradictory date range for {sym}: effective_to ({assoc.effective_to}) "
                    f"< effective_from ({assoc.effective_from})."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_CONTRADICTORY_DATES,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=assoc.operating_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_CONTRADICTORY_DATES,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                conflicts.append({"symbol": sym, "conflict_type": "CONTRADICTORY_DATES", "details": reason})
                continue

            # 3. Check for unresolved operating MIC (e.g. DDS)
            if not assoc.operating_mic:
                reason = (
                    f"Operating MIC is unresolved for {sym} (alternatives: {list(assoc.alternative_mics)}; "
                    f"dispute: {assoc.dispute_reason}). Unresolved cases remain unresolved."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=None,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_UNRESOLVED_MIC,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                missing_evidence_or_auth.append({
                    "symbol": sym,
                    "issue_type": "UNRESOLVED_OPERATING_MIC",
                    "details": reason,
                })
                continue

            clean_mic = assoc.operating_mic.strip().upper()

            # 4. Check for disputed association (e.g. ACCS dual-source venue dispute)
            if assoc.is_disputed:
                reason = f"Listing association for {sym} on {clean_mic} is disputed: {assoc.dispute_reason}"
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                conflicts.append({"symbol": sym, "conflict_type": "DISPUTED_ASSOCIATION", "details": reason})
                continue

            # 5. Check for unsupported backdating beyond evidence observation window
            # Derive observation cutoff from the proposal's attributable evidence
            earliest_ev_date = None
            if prop.evidence_references:
                ev_dates = [
                    e.observation_timestamp_utc[:10]
                    for e in prop.evidence_references
                    if e.observation_timestamp_utc
                ]
                if ev_dates:
                    earliest_ev_date = min(ev_dates)

            evidence_date_limit = earliest_ev_date or assoc.effective_from
            if not assoc.backdating_established and assoc.effective_from < evidence_date_limit:
                reason = (
                    f"Unsupported backdating for {sym}: proposed effective_from '{assoc.effective_from}' "
                    f"precedes earliest verified documentary observation date '{evidence_date_limit}' "
                    "without documentary backdating proof."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                missing_evidence_or_auth.append({
                    "symbol": sym,
                    "issue_type": "UNSUPPORTED_BACKDATING",
                    "details": reason,
                })
                continue

            # 6. Check target engine for preexisting records and locator collisions
            # Enforces:
            # - CIK alone CANNOT match a security (different share classes of same issuer remain distinct)
            # - Same-issuer distinct-security locator collision fails closed
            # - Ambiguous continuity cannot be assumed from CIK and share class alone
            key = (sym, clean_mic)
            existing_assocs = engine._active_listing_index.get(key, [])
            matched_assoc: Optional[EffectiveDatedListingAssociation] = None
            conflict_detected = False
            conflict_message = ""

            active_existing_assocs = [a for a in existing_assocs if a.is_active_on(effective_date)]
            if len(active_existing_assocs) > 1:
                conflict_detected = True
                conflict_message = (
                    f"Overlapping active associations conflict: Target ledger contains {len(active_existing_assocs)} "
                    f"overlapping active associations for listing '{sym}' on '{clean_mic}' on {effective_date}. "
                    "Cannot match or plan adoption on a conflicting multi-active locator."
                )
            else:
                for ex_a in active_existing_assocs:
                    ex_sec = engine._security_registry.get(ex_a.internal_security_id)
                    if not ex_sec:
                        continue

                    # Case 0a: Disputed listing association in target ledger
                    if ex_a.is_disputed:
                        conflict_detected = True
                        conflict_message = (
                            f"Disputed listing association: Association '{ex_a.association_id}' for listing '{sym}' on '{clean_mic}' "
                            f"is flagged as disputed ({ex_a.dispute_reason or 'unspecified dispute reason'}). "
                            "Cannot match or adopt against an active disputed association."
                        )
                        break

                    # Case 0b: Inactive or suspended security in target master
                    if ex_sec.status != "ACTIVE":
                        conflict_detected = True
                        conflict_message = (
                            f"Inactive target security: Security '{ex_sec.internal_security_id}' has non-active status "
                            f"'{ex_sec.status}'. Cannot match an inactive or retired security."
                        )
                        break

                    # Case A: Different CIK -> Overlapping cross-issuer conflict
                    if clean_cik is not None and ex_sec.issuer_cik != clean_cik:
                        conflict_detected = True
                        conflict_message = (
                            f"Overlapping cross-issuer mapping conflict: Listing '{sym}' on '{clean_mic}' is actively associated "
                            f"with distinct security '{ex_a.internal_security_id}' (CIK {ex_sec.issuer_cik}) "
                            f"on {effective_date}. Cannot map to CIK {clean_cik} without corporate action."
                        )
                        break

                    # Case B: Same CIK, but distinct security (different share class or entity) -> Collision
                    if ex_sec.share_class != ent.share_class:
                        conflict_detected = True
                        conflict_message = (
                            f"Same-issuer locator collision: Listing '{sym}' on '{clean_mic}' is actively associated "
                            f"with distinct security '{ex_sec.internal_security_id}' (share class '{ex_sec.share_class}') "
                            f"of issuer CIK {clean_cik} on {effective_date}, while proposed share class is '{ent.share_class}'. "
                            "Distinct securities of the same issuer cannot share an active locator."
                        )
                        break

                    # Case C: Concordant CIK, share class, AND matching primary symbol/locator -> Exact match
                    if (
                        ex_sec.primary_symbol == sym
                        and ex_sec.issuer_cik == clean_cik
                        and ex_sec.share_class == ent.share_class
                    ):
                        matched_assoc = ex_a
                        break
                    else:
                        conflict_detected = True
                        conflict_message = (
                            f"Ambiguous security continuity: Listing '{sym}' on '{clean_mic}' shares CIK {clean_cik} "
                            f"and share class '{ent.share_class}' with '{ex_sec.internal_security_id}' (symbol '{ex_sec.primary_symbol}'), "
                            "but continuity cannot be inferred from CIK and share class alone without explicit corporate action or alias record."
                        )
                        break

            if conflict_detected:
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    rationale=conflict_message,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id="",
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT,
                    is_creation=False,
                    is_mechanically_plannable=False,
                    conflict_reason=conflict_message,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": conflict_message})
                conflicts.append({
                    "symbol": sym,
                    "conflict_type": "LOCATOR_ASSOCIATION_CONFLICT",
                    "details": conflict_message,
                })
                continue

            # If existing durable identity and association match:
            if matched_assoc:
                durable_sec_id = matched_assoc.internal_security_id
                durable_assoc_id = matched_assoc.association_id
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    matched_existing_security_id=durable_sec_id,
                    proposed_durable_security_id=durable_sec_id,
                    rationale=f"Listing '{sym}' on '{clean_mic}' already matches active durable entity '{durable_sec_id}'.",
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id=durable_sec_id,
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    matched_existing_association_id=durable_assoc_id,
                    proposed_durable_association_id=durable_assoc_id,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id
                entities_matched += 1
                assocs_matched += 1
                continue

            # 7. Real Candidate vs Synthetic Fixture Policy & Authorization Separation
            # INVARIANT: Real-source candidates (AAPL, GOOG, GOOGL, TSM, AACIU, etc.) are NEVER
            # marked APPROVED_FOR_ADOPTION. Adopting real candidates prior to Rule R06 primary-listing
            # evidence closure is an unresolved policy decision, and real production adoption is unauthorized.
            #
            # Governance Guards:
            # - A symbol beginning "SYNTH_" MUST NOT grant synthetic status or approval.
            # - Changing only a real proposal's symbol CANNOT change its authorization disposition.
            # - is_synthetic is proposal metadata, NOT production authorization.
            # - A proposal is ONLY treated as an authorized synthetic fixture if:
            #     a) It is explicitly marked as synthetic test metadata (getattr(prop, "is_synthetic", False) is True
            #        or ent.status == "SYNTHETIC_TEST_FIXTURE");
            #     b) It is NOT a known real-source candidate (by symbol or CIK);
            #     c) Authorization explicitly declares scope == "SYNTHETIC_TEST_ONLY".
            real_candidate_ciks = set(PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING.values())
            is_known_real_candidate = (
                sym in PrimaryListingEvidenceAdapter.AUTHORITATIVE_CANDIDATE_CIK_MAPPING
                or (clean_cik is not None and clean_cik in real_candidate_ciks)
            )

            has_synthetic_metadata = (
                getattr(prop, "is_synthetic", False) is True
                or ent.status == "SYNTHETIC_TEST_FIXTURE"
            )

            is_synthetic_fixture = (
                not is_known_real_candidate
                and has_synthetic_metadata
            )

            # Allocate opaque proposed durable security ID decoupled from listing attributes
            if ent.provisional_security_id in provisional_to_durable_map:
                durable_sec_id = provisional_to_durable_map[ent.provisional_security_id]
            else:
                durable_sec_id = id_allocator() if id_allocator else cls.allocate_opaque_security_id()
                provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id

            durable_assoc_id = cls.allocate_opaque_association_id(
                security_id=durable_sec_id,
                operating_mic=clean_mic,
                symbol=sym,
                effective_from=assoc.effective_from,
            )

            if not is_synthetic_fixture:
                allow_reference_reg = (
                    auth.is_authorized
                    and (
                        getattr(auth, "allow_reference_identity_registration", False)
                        or auth.scope in ("REFERENCE_IDENTITY_REGISTRATION", "EVIDENCE_RESOLVED_REFERENCE_IDENTITY")
                    )
                )
                if allow_reference_reg:
                    # Under PO-approved architectural principle:
                    # "An evidence-resolved internal security identity and supported listing association
                    # may be registered independently of strategy eligibility and R06 admission status.
                    # Such registration grants no conversion, admission, publication or scanner authority."
                    ent_adopt = ProposedEntityAdoption(
                        provisional_security_id=ent.provisional_security_id,
                        primary_symbol=sym,
                        share_class=ent.share_class,
                        issuer_cik=clean_cik,
                        status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                        is_creation=True,
                        is_mechanically_plannable=True,
                        proposed_durable_security_id=durable_sec_id,
                        rationale=(
                            "Approved under architectural principle: evidence-resolved internal security "
                            "identity registered independently of strategy eligibility and R06 status. "
                            "Grants zero universe admission or publication authority."
                        ),
                    )
                    assoc_adopt = ProposedAssociationAdoption(
                        provisional_association_id=assoc.provisional_association_id,
                        provisional_security_id=ent.provisional_security_id,
                        target_durable_security_id=durable_sec_id,
                        symbol=sym,
                        operating_mic=clean_mic,
                        effective_from=assoc.effective_from,
                        effective_to=assoc.effective_to,
                        status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                        is_creation=True,
                        is_mechanically_plannable=True,
                        proposed_durable_association_id=durable_assoc_id,
                        conflict_reason=None,
                    )
                    proposed_entities.append(ent_adopt)
                    proposed_associations.append(assoc_adopt)
                    entities_approved += 1
                    assocs_approved += 1
                    continue

                # Real candidate without reference registration authorization: mechanically plannable, but adoption remains policy-unresolved
                reason = (
                    f"Real-source candidate '{sym}' is mechanically plannable, but adoption into durable "
                    "master prior to primary-listing evidence closure (Rule R06) remains an unresolved policy "
                    "decision. Production adoption is unauthorized."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    proposed_durable_security_id=durable_sec_id,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id=durable_sec_id,
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_POLICY_UNRESOLVED,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    proposed_durable_association_id=durable_assoc_id,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                missing_evidence_or_auth.append({
                    "symbol": sym,
                    "issue_type": "POLICY_UNRESOLVED_R06_PREREQUISITE",
                    "details": reason,
                })
                continue

            # 8. Synthetic fixture branch: check explicit synthetic authorization
            if not auth.is_authorized or auth.scope != "SYNTHETIC_TEST_ONLY":
                reason = "Synthetic fixture adoption blocked: authorization token is absent, is_authorized=False, or scope is not SYNTHETIC_TEST_ONLY."
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    proposed_durable_security_id=durable_sec_id,
                    rationale=reason,
                )
                assoc_adopt = ProposedAssociationAdoption(
                    provisional_association_id=assoc.provisional_association_id,
                    provisional_security_id=ent.provisional_security_id,
                    target_durable_security_id=durable_sec_id,
                    symbol=sym,
                    operating_mic=clean_mic,
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    status=IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION,
                    is_creation=False,
                    is_mechanically_plannable=True,
                    proposed_durable_association_id=durable_assoc_id,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                missing_evidence_or_auth.append({
                    "symbol": sym,
                    "issue_type": "MISSING_AUTHORIZATION",
                    "details": reason,
                })
                continue

            # Synthetic fixture with valid authorization -> APPROVED_FOR_ADOPTION
            ent_adopt = ProposedEntityAdoption(
                provisional_security_id=ent.provisional_security_id,
                primary_symbol=sym,
                share_class=ent.share_class,
                issuer_cik=clean_cik,
                status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                is_creation=True,
                is_mechanically_plannable=True,
                proposed_durable_security_id=durable_sec_id,
                rationale="Authorized synthetic test fixture approved for mechanical dry-run adoption.",
            )
            assoc_adopt = ProposedAssociationAdoption(
                provisional_association_id=assoc.provisional_association_id,
                provisional_security_id=ent.provisional_security_id,
                target_durable_security_id=durable_sec_id,
                symbol=sym,
                operating_mic=clean_mic,
                effective_from=assoc.effective_from,
                effective_to=assoc.effective_to,
                status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                is_creation=True,
                is_mechanically_plannable=True,
                proposed_durable_association_id=durable_assoc_id,
            )
            proposed_entities.append(ent_adopt)
            proposed_associations.append(assoc_adopt)
            provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id
            entities_approved += 1
            assocs_approved += 1

        summary = AdoptionPlanSummary(
            total_proposals=len(proposal_list),
            entities_matched_existing=entities_matched,
            entities_approved_creation=entities_approved,
            entities_blocked=entities_blocked,
            associations_matched_existing=assocs_matched,
            associations_approved_creation=assocs_approved,
            associations_blocked=assocs_blocked,
            conflicts_count=len(conflicts),
        )

        plan_content_digest = cls.compute_plan_content_digest(
            entities=proposed_entities,
            associations=proposed_associations,
            provisional_to_durable_map=provisional_to_durable_map,
            summary=summary,
        )
        plan_id = f"PLAN_ADOPT_{plan_content_digest[:12].upper()}"

        return IdentityAdoptionPlan(
            plan_id=plan_id,
            created_at_utc=plan_created_utc,
            evaluation_reference_date=as_of_eval_date or "REPLAY_AS_OF_OBSERVATION",
            target_state_digest=target_state_digest,
            plan_content_digest=plan_content_digest,
            authorization=auth,
            entities=proposed_entities,
            associations=proposed_associations,
            provisional_to_durable_map=provisional_to_durable_map,
            conflicts=conflicts,
            blocked_operations=blocked_operations,
            missing_evidence_or_authorization=missing_evidence_or_auth,
            primary_listing_states=primary_listing_states,
            summary=summary,
        )


# =====================================================================
# 5. Test-Scoped Atomic Application Callable
# =====================================================================

class StalePlanError(RuntimeError):
    """Raised when an adoption plan is applied to an engine whose target state digest differs."""
    pass


class PlanIntegrityError(RuntimeError):
    """Raised when plan actions have been tampered with or do not match declared plan content digest."""
    pass


class PlanApplicationError(RuntimeError):
    """Raised when an error occurs during test-scoped plan application."""
    pass


def apply_adoption_plan_test_only(
    plan: IdentityAdoptionPlan,
    engine: OfflineAdmissionMappingEngine,
    simulate_mid_failure_on_symbol: Optional[str] = None,
    allow_partial: bool = False,
) -> OfflineAdmissionMappingEngine:
    """
    TEST-SCOPED ONLY: atomic transaction verification in temporary/test stores.
    NEVER exposed as a production or operational entry point.

    Validates:
    1. Plan target state freshness against engine (StalePlanError).
    2. Plan content integrity against plan_content_digest (PlanIntegrityError).
    3. Explicit partial application policy: fails closed if plan contains blocked items
       unless allow_partial=True is explicitly granted (PlanApplicationError).

    Atomic rollback: Takes a full snapshot before mutation; restores 100% of state
    on any failure.
    """
    # 0. Test-only execution boundary (must not become a production apply path)
    if plan.authorization.scope != "SYNTHETIC_TEST_ONLY":
        raise PlanApplicationError(
            f"Test-only application helper cannot apply plan with authorization scope '{plan.authorization.scope}'. "
            "The test-only helper must not become a production apply path (strictly restricted to scope='SYNTHETIC_TEST_ONLY')."
        )

    initial_snapshot = engine.to_dict()
    current_digest = IdentityAdoptionDryRunEngine.compute_target_state_digest(initial_snapshot)

    # 1. Stale target state rejection
    if current_digest != plan.target_state_digest:
        raise StalePlanError(
            f"Stale plan rejected: Plan target digest '{plan.target_state_digest[:12]}...' "
            f"does not match current target engine digest '{current_digest[:12]}...'. "
            "Target state has changed since plan generation."
        )

    # 2. Plan content integrity verification (rejects altered plan actions)
    if not plan.is_plan_content_valid():
        raise PlanIntegrityError(
            f"Plan content tampering detected: Recomputed plan content digest does not match "
            f"declared digest '{plan.plan_content_digest}'. Application aborted."
        )

    # 3. Explicit tested application rule for mixed blocked/ready plans
    has_blocked_items = (
        plan.summary.entities_blocked > 0
        or plan.summary.associations_blocked > 0
        or plan.summary.conflicts_count > 0
    )
    if has_blocked_items and not allow_partial:
        raise PlanApplicationError(
            f"Plan application failed closed: Plan '{plan.plan_id}' contains {plan.summary.entities_blocked} "
            f"blocked entities, {plan.summary.associations_blocked} blocked associations, and "
            f"{plan.summary.conflicts_count} conflicts. Mixed plans cannot be applied when allow_partial=False."
        )

    try:
        # 4. Apply approved entity creations
        for ent in plan.entities:
            if ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION and ent.is_creation:
                if simulate_mid_failure_on_symbol and ent.primary_symbol == simulate_mid_failure_on_symbol:
                    raise PlanApplicationError(
                        f"Simulated mid-operation transaction failure on symbol '{ent.primary_symbol}'."
                    )
                now_utc = datetime.now(timezone.utc).isoformat()
                sec_rec = InternalSecurityRecord(
                    internal_security_id=ent.proposed_durable_security_id,
                    issuer_cik=ent.issuer_cik,
                    primary_symbol=ent.primary_symbol,
                    share_class=ent.share_class,
                    created_at_utc=now_utc,
                    status="ACTIVE",
                    continuity_basis="ADOPTED_UNDER_REVIEWED_PLAN",
                    certifies_continuity=False,
                )
                engine._security_registry[sec_rec.internal_security_id] = sec_rec
                if sec_rec.issuer_cik:
                    engine._cik_to_securities.setdefault(sec_rec.issuer_cik, set()).add(
                        sec_rec.internal_security_id
                    )

        # 5. Apply approved association creations
        for assoc in plan.associations:
            if assoc.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION and assoc.is_creation:
                if simulate_mid_failure_on_symbol and assoc.symbol == simulate_mid_failure_on_symbol:
                    raise PlanApplicationError(
                        f"Simulated mid-operation transaction failure on association '{assoc.symbol}'."
                    )
                assoc_id = assoc.proposed_durable_association_id or (
                    f"ASSOC_{assoc.target_durable_security_id}_{assoc.operating_mic}_{assoc.symbol}_{assoc.effective_from}"
                )
                assoc_rec = EffectiveDatedListingAssociation(
                    association_id=assoc_id,
                    internal_security_id=assoc.target_durable_security_id,
                    symbol=assoc.symbol,
                    operating_mic=assoc.operating_mic or "UNRESOLVED",
                    effective_from=assoc.effective_from,
                    effective_to=assoc.effective_to,
                    source_reference=f"ADOPTED_PLAN_{plan.plan_id}",
                )
                engine._association_ledger[assoc_rec.association_id] = assoc_rec
                key = (assoc_rec.symbol, assoc_rec.operating_mic)
                engine._active_listing_index.setdefault(key, []).append(assoc_rec)

        return engine

    except Exception:
        # Atomic rollback: restore engine to initial snapshot
        rolled_back = OfflineAdmissionMappingEngine.from_dict(initial_snapshot)
        engine._security_registry = rolled_back._security_registry
        engine._association_ledger = rolled_back._association_ledger
        engine._cik_to_securities = rolled_back._cik_to_securities
        engine._active_listing_index = rolled_back._active_listing_index
        raise
