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
     APPROVED_FOR_ADOPTION, MATCHES_EXISTING_DURABLE, BLOCKED_MISSING_AUTHORIZATION,
     BLOCKED_UNRESOLVED_MIC, BLOCKED_ASSOCIATION_CONFLICT, BLOCKED_CONTRADICTORY_DATES,
     or BLOCKED_BACKDATED_UNSUPPORTED.
   - Preserves existing durable IDs and guarantees stable durable IDs across reload/replay.
   - Unresolved cases remain strictly unresolved; never mints alternate IDs to bypass conflict.

2. Content and Target-State Binding:
   - The adoption plan is bound to both its proposed content (plan_content_digest)
     and the target state snapshot (target_state_digest).
   - If the target state mutates, stale plans fail closed.

3. Strict Separation of Evidence, Authorization, and Admission:
   - Mechanical identity adoption evaluates internal security master registration only.
   - Does NOT imply production adoption, primary-listing sufficiency (Rule R06),
     SourceSecurity conversion authority, or universe admission.
   - Real-source records (AAPL, GOOG, GOOGL) retain primary_listing = None (UNKNOWN)
     and remain ineligible for universe admission unless newer qualifying evidence exists.

4. Test-Scoped Atomic Application:
   - Test-only application callable (apply_adoption_plan_test_only) operates strictly
     in temporary/in-memory stores.
   - Guarantees full rollback on mid-operation failure and idempotent re-application.
   - NEVER exposed as a production CLI, API route, startup hook, or scheduled job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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


# =====================================================================
# 1. Adoption Status & Authorization Models
# =====================================================================

class IdentityAdoptionStatus(str, Enum):
    APPROVED_FOR_ADOPTION = "APPROVED_FOR_ADOPTION"
    MATCHES_EXISTING_DURABLE = "MATCHES_EXISTING_DURABLE"
    BLOCKED_MISSING_AUTHORIZATION = "BLOCKED_MISSING_AUTHORIZATION"
    BLOCKED_UNRESOLVED_MIC = "BLOCKED_UNRESOLVED_MIC"
    BLOCKED_ASSOCIATION_CONFLICT = "BLOCKED_ASSOCIATION_CONFLICT"
    BLOCKED_CONTRADICTORY_DATES = "BLOCKED_CONTRADICTORY_DATES"
    BLOCKED_BACKDATED_UNSUPPORTED = "BLOCKED_BACKDATED_UNSUPPORTED"


@dataclass(frozen=True)
class IdentityAdoptionAuthorization:
    """
    Explicit authorization context for identity adoption.
    Without an authorized token and affirmative flag, proposals cannot be adopted.
    """
    is_authorized: bool = False
    authorized_by: Optional[str] = None
    authorization_timestamp_utc: Optional[str] = None
    authorization_token: Optional[str] = None
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
    entities_proposed_creation: int
    entities_blocked: int
    associations_matched_existing: int
    associations_proposed_creation: int
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
        "SourceSecurity conversion authority, or universe admission under Rule R06."
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at_utc": self.created_at_utc,
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
                matched_existing_association_id=a.get("matched_existing_association_id"),
                proposed_durable_association_id=a.get("proposed_durable_association_id"),
                conflict_reason=a.get("conflict_reason"),
            )
            for a in data.get("associations", [])
        ]

        sum_d = data.get("summary", {})
        summary = AdoptionPlanSummary(**sum_d)

        return cls(
            plan_id=data["plan_id"],
            created_at_utc=data["created_at_utc"],
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

    @classmethod
    def generate_plan(
        cls,
        proposals: Union[IdentityAssociationProposalLedger, List[CandidateAssociationProposal]],
        existing_target: Union[OfflineAdmissionMappingEngine, Dict[str, Any], None] = None,
        authorization: Optional[IdentityAdoptionAuthorization] = None,
        as_of_date: Optional[str] = None,
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
        plan_created_utc = datetime.now(timezone.utc).isoformat()

        proposed_entities: List[ProposedEntityAdoption] = []
        proposed_associations: List[ProposedAssociationAdoption] = []
        provisional_to_durable_map: Dict[str, str] = {}
        conflicts: List[Dict[str, Any]] = []
        blocked_operations: List[Dict[str, Any]] = []
        missing_evidence_or_auth: List[Dict[str, Any]] = []
        primary_listing_states: Dict[str, Dict[str, Any]] = {}

        entities_matched = 0
        entities_created = 0
        entities_blocked = 0
        assocs_matched = 0
        assocs_created = 0
        assocs_blocked = 0

        for prop in proposal_list:
            sym = prop.symbol.strip().upper()
            ent = prop.proposed_entity
            assoc = prop.proposed_association
            clean_cik = str(ent.issuer_cik).zfill(10) if ent.issuer_cik else None
            effective_date = as_of_date or assoc.effective_from

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

            # 1. Check for contradictory date ranges
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
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                conflicts.append({"symbol": sym, "conflict_type": "CONTRADICTORY_DATES", "details": reason})
                continue

            # 2. Check for unresolved operating MIC (e.g. DDS)
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

            # 3. Check for disputed association (e.g. ACCS)
            if assoc.is_disputed:
                reason = f"Listing association for {sym} on {clean_mic} is disputed: {assoc.dispute_reason}"
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_ASSOCIATION_CONFLICT,
                    is_creation=False,
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
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                conflicts.append({"symbol": sym, "conflict_type": "DISPUTED_ASSOCIATION", "details": reason})
                continue

            # 4. Check for unsupported backdating beyond evidence window
            if not assoc.backdating_established and assoc.effective_from < "2026-10-17":
                reason = (
                    f"Observation date {assoc.effective_from} for {sym} is backdated prior to verified "
                    "evidence window without documentary inception proof."
                )
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_BACKDATED_UNSUPPORTED,
                    is_creation=False,
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

            # 5. Check against target engine for existing durable records or conflicts
            key = (sym, clean_mic)
            existing_assocs = engine._active_listing_index.get(key, [])
            matched_assoc: Optional[EffectiveDatedListingAssociation] = None
            conflict_detected = False
            conflict_message = ""

            for ex_a in existing_assocs:
                if ex_a.is_active_on(effective_date):
                    ex_sec = engine._security_registry.get(ex_a.internal_security_id)
                    if ex_sec and (clean_cik is None or ex_sec.issuer_cik == clean_cik):
                        matched_assoc = ex_a
                        break
                    else:
                        conflict_detected = True
                        conflict_message = (
                            f"Overlapping mapping conflict: Listing '{sym}' on '{clean_mic}' is actively associated "
                            f"with distinct security '{ex_a.internal_security_id}' (CIK {ex_sec.issuer_cik if ex_sec else 'UNKNOWN'}) "
                            f"on {effective_date}. Cannot adopt candidate with CIK {clean_cik} without corporate action."
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
                    conflict_reason=conflict_message,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": conflict_message})
                conflicts.append({
                    "symbol": sym,
                    "conflict_type": "OVERLAPPING_ACTIVE_ASSOCIATION",
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
                    matched_existing_association_id=durable_assoc_id,
                    proposed_durable_association_id=durable_assoc_id,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id
                entities_matched += 1
                assocs_matched += 1
                continue

            # 6. Check authorization for new adoption
            if not auth.is_authorized:
                reason = "Proposal adoption blocked: explicit authorization token is absent or is_authorized=False."
                ent_adopt = ProposedEntityAdoption(
                    provisional_security_id=ent.provisional_security_id,
                    primary_symbol=sym,
                    share_class=ent.share_class,
                    issuer_cik=clean_cik,
                    status=IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION,
                    is_creation=False,
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
                    status=IdentityAdoptionStatus.BLOCKED_MISSING_AUTHORIZATION,
                    is_creation=False,
                    conflict_reason=reason,
                )
                proposed_entities.append(ent_adopt)
                proposed_associations.append(assoc_adopt)
                entities_blocked += 1
                assocs_blocked += 1
                blocked_operations.append({"symbol": sym, "reason": reason})
                missing_evidence_or_auth.append({
                    "symbol": sym,
                    "issue_type": "MISSING_AUTHORIZATION",
                    "details": reason,
                })
                continue

            # 7. Authorized new creation: compute deterministic durable IDs
            # Reuses released OfflineAdmissionMappingEngine hash formula
            id_material = f"{clean_cik or 'NOCIK'}:{clean_mic}:{sym}:{effective_date}"
            sec_id_hash = hashlib.sha256(id_material.encode("utf-8")).hexdigest()[:12].upper()
            durable_sec_id = f"ARX_SEC_{sec_id_hash}"
            durable_assoc_id = f"ASSOC_{durable_sec_id}_{clean_mic}_{sym}_{effective_date}"

            ent_adopt = ProposedEntityAdoption(
                provisional_security_id=ent.provisional_security_id,
                primary_symbol=sym,
                share_class=ent.share_class,
                issuer_cik=clean_cik,
                status=IdentityAdoptionStatus.APPROVED_FOR_ADOPTION,
                is_creation=True,
                proposed_durable_security_id=durable_sec_id,
                rationale="Authorized for durable identity adoption under dry-run plan.",
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
                proposed_durable_association_id=durable_assoc_id,
            )
            proposed_entities.append(ent_adopt)
            proposed_associations.append(assoc_adopt)
            provisional_to_durable_map[ent.provisional_security_id] = durable_sec_id
            entities_created += 1
            assocs_created += 1

        summary = AdoptionPlanSummary(
            total_proposals=len(proposal_list),
            entities_matched_existing=entities_matched,
            entities_proposed_creation=entities_created,
            entities_blocked=entities_blocked,
            associations_matched_existing=assocs_matched,
            associations_proposed_creation=assocs_created,
            associations_blocked=assocs_blocked,
            conflicts_count=len(conflicts),
        )

        # Compute content digest
        content_payload = {
            "entities": [e.to_dict() for e in proposed_entities],
            "associations": [a.to_dict() for a in proposed_associations],
            "provisional_to_durable_map": provisional_to_durable_map,
            "summary": summary.to_dict(),
        }
        plan_content_digest = canonical_hash(content_payload)
        plan_id = f"PLAN_ADOPT_{plan_content_digest[:12].upper()}"

        return IdentityAdoptionPlan(
            plan_id=plan_id,
            created_at_utc=plan_created_utc,
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


class PlanApplicationError(RuntimeError):
    """Raised when an error occurs during test-scoped plan application."""
    pass


def apply_adoption_plan_test_only(
    plan: IdentityAdoptionPlan,
    engine: OfflineAdmissionMappingEngine,
    simulate_mid_failure_on_symbol: Optional[str] = None,
) -> OfflineAdmissionMappingEngine:
    """
    TEST-SCOPED ONLY: atomic transaction verification in temporary/test stores.
    NEVER exposed as a production or operational entry point.

    Validates plan freshness against target state digest.
    Takes a full snapshot before mutation.
    Applies approved entity and association creations atomically.
    If an error occurs or simulate_mid_failure_on_symbol matches, rolls back to snapshot completely.
    """
    initial_snapshot = engine.to_dict()
    current_digest = IdentityAdoptionDryRunEngine.compute_target_state_digest(initial_snapshot)

    # 1. Stale plan invalidation
    if current_digest != plan.target_state_digest:
        raise StalePlanError(
            f"Stale plan rejected: Plan target digest '{plan.target_state_digest[:12]}...' "
            f"does not match current target engine digest '{current_digest[:12]}...'. "
            "Target state has changed since plan generation."
        )

    try:
        # 2. Apply approved entity creations
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

        # 3. Apply approved association creations
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
