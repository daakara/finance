"""
analyst_dashboard/universe/identity_proposal.py

Sprint 2: Bounded Offline Identity Association & Primary-Listing Evidence Proposal Engine.
Assembles reviewable offline association proposals for the seven-candidate real-source sample:
(AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU).

Core Architectural Invariants:
1. Reuses Existing Identity Model:
   - Integrates with ListingLocator, InternalSecurityRecord, EffectiveDatedListingAssociation,
     and OfflineAdmissionMappingEngine.
   - Internal entities remain distinct from ephemeral locators (MIC + symbol + date).
   - Generated IDs are explicitly marked PROVISIONAL/NON_PRODUCTION and never adopted as production IDs.
   - Multi-symbol issuers (GOOG vs GOOGL under CIK 0001652044) are kept strictly separate.
   - TSM Depositary Receipt (ADR) is kept strictly separate from underlying Taiwan ordinary shares.
   - AACIU Unit structure is preserved explicitly (not conflated with ordinary shares or warrants).
   - Entities sharing a CIK, name, or history are NEVER merged automatically.

2. Explicitly Typed External Identifiers:
   - Preserves OpenFIGI levels (compositeFIGI, shareClassFIGI, listing-level FIGI).
   - Preserves CIK as entity-level identifier (not security-level surrogate).
   - Preserves Alpaca asset directory ID and trading status.
   - External identifiers do not independently prove corporate continuity or exchange validity.

3. Temporal Bounding & Non-Backdating:
   - Validity intervals are bounded to verified evidence observation dates (2026-10-17).
   - Observation timestamps are NOT backdated to IPO or listing inception without documentary proof.

4. Independent Primary-Listing Evidence Tracing:
   - Distinguishes issuer home market, underlying ordinary listing, depositary receipt listing,
     primary venue for the specific instrument, and selected US assessment listing.
   - Preserves unknown primary_listing as None (no defaulting to False or True).
   - Documents exact claim needed by contracts.py Rule R06 and specific missing evidence.

5. Guarded Proposal Handoff & Atomic Revocation:
   - Proposal simulation is strictly isolated from production adoption claims.
   - Atomic whole-batch revocation is preserved; Universe Builder remains strictly NOT_RUN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .admission_mapping import (
    AttributableFieldEvidence,
    EffectiveDatedListingAssociation,
    EvidenceNature,
    InternalSecurityRecord,
    ListingLocator,
)
from .validator import (
    UniverseInputValidator,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)


# =====================================================================
# 1. External Identifier Typing & Provisional Entity Models
# =====================================================================

class ExternalIdentifierType(str, Enum):
    COMPOSITE_FIGI = "COMPOSITE_FIGI"
    SHARE_CLASS_FIGI = "SHARE_CLASS_FIGI"
    LISTING_FIGI = "LISTING_FIGI"
    CIK_ENTITY_LEVEL = "CIK_ENTITY_LEVEL"
    ALPACA_ASSET_ID = "ALPACA_ASSET_ID"
    YAHOO_QUOTE_SYMBOL = "YAHOO_QUOTE_SYMBOL"


@dataclass(frozen=True)
class ExternalIdentifierRecord:
    """Explicitly typed external identifier with provenance metadata."""
    id_type: ExternalIdentifierType
    id_value: str
    provider: str
    retrieval_timestamp_utc: str
    is_entity_level: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id_type": self.id_type.value,
            "id_value": self.id_value,
            "provider": self.provider,
            "retrieval_timestamp_utc": self.retrieval_timestamp_utc,
            "is_entity_level": self.is_entity_level,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ProvisionalSecurityEntity:
    """
    Tier B Proposal: Provisional internal security entity.
    Explicitly marked as non-production; CANNOT be adopted into production master without authorization.
    Reuses released InternalSecurityRecord structure without writing to production master.
    """
    provisional_security_id: str
    primary_symbol: str
    share_class: Optional[str]
    instrument_structure: str  # COMMON_STOCK, DEPOSITARY_RECEIPT, UNIT
    issuer_cik: Optional[str]
    company_name: Optional[str]
    external_identifiers: Tuple[ExternalIdentifierRecord, ...]
    created_at_utc: str
    status: str = "PROVISIONAL_NON_PRODUCTION"
    is_adopted_production_id: bool = False
    certifies_continuity: bool = False

    def __post_init__(self):
        if not self.provisional_security_id.startswith("PROV_SEC_"):
            raise ValueError(
                f"Provisional security ID must start with 'PROV_SEC_', got '{self.provisional_security_id}'"
            )
        if self.is_adopted_production_id:
            raise ValueError(
                "ProvisionalSecurityEntity cannot declare is_adopted_production_id=True."
            )

    def to_internal_security_record(self) -> InternalSecurityRecord:
        """
        Wraps provisional entity into released InternalSecurityRecord model.
        Maintains provisional status without mutating production master.
        """
        return InternalSecurityRecord(
            internal_security_id=self.provisional_security_id,
            issuer_cik=self.issuer_cik,
            primary_symbol=self.primary_symbol,
            share_class=self.share_class,
            created_at_utc=self.created_at_utc,
            status="PROVISIONAL_NON_PRODUCTION",
            continuity_basis="PROVISIONAL_OFFLINE_PROPOSAL",
            certifies_continuity=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provisional_security_id": self.provisional_security_id,
            "primary_symbol": self.primary_symbol,
            "share_class": self.share_class,
            "instrument_structure": self.instrument_structure,
            "issuer_cik": self.issuer_cik,
            "company_name": self.company_name,
            "external_identifiers": [i.to_dict() for i in self.external_identifiers],
            "created_at_utc": self.created_at_utc,
            "status": self.status,
            "is_adopted_production_id": self.is_adopted_production_id,
            "certifies_continuity": self.certifies_continuity,
        }


@dataclass(frozen=True)
class ProvisionalListingAssociation:
    """
    Proposed effective-dated listing association.
    Enforces observation-window temporal bounds and prohibits unevidenced inception backdating.
    Reuses released ListingLocator and EffectiveDatedListingAssociation semantics.
    """
    provisional_association_id: str
    provisional_security_id: str
    symbol: str
    operating_mic: Optional[str] = None
    alternative_mics: Tuple[str, ...] = ()
    effective_from: str = "2026-10-17"  # YYYY-MM-DD
    effective_to: Optional[str] = None  # None indicates open/active
    validity_basis: str = "EVIDENCE_OBSERVATION_WINDOW"
    backdating_established: bool = False
    temporal_limitations: str = (
        "Effective interval is bounded to the empirical observation snapshot date; "
        "historical inception or IPO continuity is NOT certified."
    )
    is_disputed: bool = False
    dispute_reason: Optional[str] = None

    def __post_init__(self):
        if not self.provisional_association_id.startswith("PROV_ASSOC_"):
            raise ValueError(
                f"Provisional association ID must start with 'PROV_ASSOC_', got '{self.provisional_association_id}'"
            )
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError(
                f"Contradictory date range: effective_to ({self.effective_to}) < effective_from ({self.effective_from})"
            )

    def is_active_on(self, target_date: str) -> bool:
        if self.effective_from > target_date:
            return False
        if self.effective_to and self.effective_to < target_date:
            return False
        return True

    def to_listing_locator(self, as_of_date: Optional[str] = None) -> ListingLocator:
        """
        Constructs a released ListingLocator from this association.
        Fails closed if operating_mic is unresolved.
        """
        if not self.operating_mic:
            raise ValueError(
                f"Cannot construct ListingLocator for {self.symbol}: "
                f"operating_mic is unresolved ({self.dispute_reason})"
            )
        target_date = as_of_date or self.effective_from
        return ListingLocator(operating_mic=self.operating_mic, symbol=self.symbol, as_of_date=target_date)

    def to_effective_dated_association(
        self,
        internal_security_id: Optional[str] = None,
        source_reference: str = "OFFLINE_PROPOSAL_LEDGER",
    ) -> EffectiveDatedListingAssociation:
        """
        Wraps provisional association into released EffectiveDatedListingAssociation model.
        Fails closed if operating_mic is unresolved.
        """
        if not self.operating_mic:
            raise ValueError(
                f"Cannot construct EffectiveDatedListingAssociation for {self.symbol}: "
                f"operating_mic is unresolved ({self.dispute_reason})"
            )
        sec_id = internal_security_id or self.provisional_security_id
        return EffectiveDatedListingAssociation(
            association_id=self.provisional_association_id,
            internal_security_id=sec_id,
            symbol=self.symbol,
            operating_mic=self.operating_mic,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            source_reference=source_reference,
            is_disputed=self.is_disputed,
            dispute_reason=self.dispute_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alternative_mics"] = list(self.alternative_mics)
        return d


# =====================================================================
# 2. Independent Primary-Listing Evidence Tracing Model
# =====================================================================

@dataclass(frozen=True)
class PrimaryListingClaimAnalysis:

    """
    Disaggregates the 5 independent listing dimensions from single-field primary_listing.
    Enforces that unevidenced primary_listing remains None (unknown) without defaulting.
    """
    symbol: str
    exact_claim_needed: str
    issuer_home_market: str
    underlying_ordinary_share_listing: Optional[str]
    depositary_receipt_listing: Optional[str]
    primary_exchange_for_instrument: str
    selected_us_assessment_listing: Optional[bool]
    claimed_primary_listing: Optional[bool]
    evidence_sufficiency: str  # SUFFICIENT, INSUFFICIENT, UNRESOLVED_CONFLICT
    missing_evidence_details: str
    contract_ambiguity_notes: str
    candidate_evidence_sources: str = (
        "Candidate evidence sources include SEC filing cover pages (Form 10-K / 20-F check-boxes), "
        "official exchange directory master feeds, or exchange listing certifications. "
        "Exclusive §12(b) registration is not mandated by existing contracts."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 3. Candidate Association Proposal
# =====================================================================

@dataclass
class CandidateAssociationProposal:
    """Complete offline association proposal for a candidate."""
    symbol: str
    observed_venue: str
    instrument_classification: str
    available_figis: Dict[str, Optional[str]]
    provider_asset_identifiers: Dict[str, Any]
    issuer_cik: Optional[str]
    issuer_name: Optional[str]
    evidence_references: List[AttributableFieldEvidence]
    proposed_entity: ProvisionalSecurityEntity
    proposed_association: ProvisionalListingAssociation
    primary_listing_analysis: PrimaryListingClaimAnalysis
    ambiguities_and_missing_evidence: List[str]
    review_disposition: str  # READY_FOR_REVIEW, BLOCKED_ON_VENUE_CONFLICT, BLOCKED_ON_CONFLICT, MISSING_EVIDENCE
    archived_parent_venue: Optional[str] = None
    provider_claimed_venue: Optional[str] = None
    normalized_venue: Optional[str] = None
    alternative_mics: Tuple[str, ...] = ()
    evidence_nature_summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "observed_venue": self.observed_venue,
            "instrument_classification": self.instrument_classification,
            "available_figis": self.available_figis,
            "provider_asset_identifiers": self.provider_asset_identifiers,
            "issuer_cik": self.issuer_cik,
            "issuer_name": self.issuer_name,
            "evidence_references": [e.to_dict() for e in self.evidence_references],
            "proposed_entity": self.proposed_entity.to_dict(),
            "proposed_association": self.proposed_association.to_dict(),
            "primary_listing_analysis": self.primary_listing_analysis.to_dict(),
            "ambiguities_and_missing_evidence": self.ambiguities_and_missing_evidence,
            "review_disposition": self.review_disposition,
            "archived_parent_venue": self.archived_parent_venue,
            "provider_claimed_venue": self.provider_claimed_venue,
            "normalized_venue": self.normalized_venue,
            "alternative_mics": list(self.alternative_mics),
            "evidence_nature_summary": self.evidence_nature_summary,
        }


# =====================================================================
# 4. Isolated Task-Local Proposal Ledger
# =====================================================================

class IdentityAssociationProposalLedger:

    """
    Isolated in-memory / task-local ledger for association proposals.
    Guarantees:
    - ID stability across serialization and reload (no regeneration of IDs).
    - Prevents overlapping active associations for distinct entities.
    - Rejects duplicate conflicting proposals.
    - Keeps provisional IDs strictly quarantine-labeled from production master.
    """

    def __init__(self):
        self._proposals: Dict[str, CandidateAssociationProposal] = {}
        self._provisional_entities: Dict[str, ProvisionalSecurityEntity] = {}
        self._provisional_associations: Dict[str, ProvisionalListingAssociation] = {}
        self._active_index: Dict[Tuple[str, str], List[ProvisionalListingAssociation]] = {}

    def add_proposal(self, proposal: CandidateAssociationProposal) -> None:
        sym = proposal.symbol.strip().upper()
        ent = proposal.proposed_entity
        assoc = proposal.proposed_association

        # Overlap and conflict detection
        mic_key = assoc.operating_mic or "UNRESOLVED_MIC"
        key = (assoc.symbol, mic_key)
        for existing in self._active_index.get(key, []):
            if existing.is_active_on(assoc.effective_from):
                if existing.provisional_security_id != assoc.provisional_security_id:
                    raise ValueError(
                        f"Overlapping proposal conflict on {key}: "
                        f"Active association '{existing.provisional_association_id}' "
                        f"for entity '{existing.provisional_security_id}' conflicts with "
                        f"'{assoc.provisional_association_id}' for entity "
                        f"'{assoc.provisional_security_id}' on {assoc.effective_from}."
                    )

        self._proposals[sym] = proposal
        self._provisional_entities[ent.provisional_security_id] = ent
        self._provisional_associations[assoc.provisional_association_id] = assoc
        self._active_index.setdefault(key, []).append(assoc)

    def get_proposal(self, symbol: str) -> Optional[CandidateAssociationProposal]:
        return self._proposals.get(symbol.strip().upper())

    def list_proposals(self) -> List[CandidateAssociationProposal]:
        return [self._proposals[s] for s in sorted(self._proposals.keys())]

    def to_dict(self) -> Dict[str, Any]:
        """Full deterministic serialization of the proposal ledger."""
        return {
            "ledger_type": "ISOLATED_OFFLINE_ASSOCIATION_PROPOSAL_LEDGER",
            "schema_version": "1.0.0",
            "production_adoption_authorized": False,
            "proposals": {k: v.to_dict() for k, v in sorted(self._proposals.items())},
            "total_proposals": len(self._proposals),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IdentityAssociationProposalLedger:
        """Reloads ledger preserving exact provisional IDs, entities, and associations."""
        ledger = cls()
        for _, p_dict in data.get("proposals", {}).items():
            # Reconstruct external identifiers
            ext_ids = []
            for ext in p_dict["proposed_entity"]["external_identifiers"]:
                ext_ids.append(ExternalIdentifierRecord(
                    id_type=ExternalIdentifierType(ext["id_type"]),
                    id_value=ext["id_value"],
                    provider=ext["provider"],
                    retrieval_timestamp_utc=ext["retrieval_timestamp_utc"],
                    is_entity_level=ext.get("is_entity_level", False),
                    notes=ext.get("notes"),
                ))

            # Reconstruct entity
            ent_d = p_dict["proposed_entity"]
            ent = ProvisionalSecurityEntity(
                provisional_security_id=ent_d["provisional_security_id"],
                primary_symbol=ent_d["primary_symbol"],
                share_class=ent_d.get("share_class"),
                instrument_structure=ent_d["instrument_structure"],
                issuer_cik=ent_d.get("issuer_cik"),
                company_name=ent_d.get("company_name"),
                external_identifiers=tuple(ext_ids),
                created_at_utc=ent_d["created_at_utc"],
                status=ent_d.get("status", "PROVISIONAL_NON_PRODUCTION"),
                is_adopted_production_id=False,
                certifies_continuity=ent_d.get("certifies_continuity", False),
            )

            # Reconstruct association
            assoc_d = p_dict["proposed_association"]
            assoc = ProvisionalListingAssociation(
                provisional_association_id=assoc_d["provisional_association_id"],
                provisional_security_id=assoc_d["provisional_security_id"],
                symbol=assoc_d["symbol"],
                operating_mic=assoc_d.get("operating_mic"),
                alternative_mics=tuple(assoc_d.get("alternative_mics", ())),
                effective_from=assoc_d["effective_from"],
                effective_to=assoc_d.get("effective_to"),
                validity_basis=assoc_d.get("validity_basis", "EVIDENCE_OBSERVATION_WINDOW"),
                backdating_established=assoc_d.get("backdating_established", False),
                temporal_limitations=assoc_d.get("temporal_limitations", ""),
                is_disputed=assoc_d.get("is_disputed", False),
                dispute_reason=assoc_d.get("dispute_reason"),
            )

            # Reconstruct primary listing analysis
            pla_d = p_dict["primary_listing_analysis"]
            pla = PrimaryListingClaimAnalysis(
                symbol=pla_d["symbol"],
                exact_claim_needed=pla_d["exact_claim_needed"],
                issuer_home_market=pla_d["issuer_home_market"],
                underlying_ordinary_share_listing=pla_d.get("underlying_ordinary_share_listing"),
                depositary_receipt_listing=pla_d.get("depositary_receipt_listing"),
                primary_exchange_for_instrument=pla_d["primary_exchange_for_instrument"],
                selected_us_assessment_listing=pla_d.get("selected_us_assessment_listing"),
                claimed_primary_listing=pla_d.get("claimed_primary_listing"),
                evidence_sufficiency=pla_d["evidence_sufficiency"],
                missing_evidence_details=pla_d["missing_evidence_details"],
                contract_ambiguity_notes=pla_d["contract_ambiguity_notes"],
                candidate_evidence_sources=pla_d.get(
                    "candidate_evidence_sources",
                    "Candidate evidence sources include SEC filing cover pages (Form 10-K / 20-F check-boxes), "
                    "official exchange directory master feeds, or exchange listing certifications. "
                    "Exclusive §12(b) registration is not mandated by existing contracts.",
                ),
            )

            # Reconstruct evidence references
            ev_refs = []
            for ev in p_dict.get("evidence_references", []):
                ev_refs.append(AttributableFieldEvidence(
                    field_name=ev["field_name"],
                    asserted_value=ev["asserted_value"],
                    evidence_nature=EvidenceNature(ev["evidence_nature"]),
                    source_identifier=ev["source_identifier"],
                    source_record_locator=ev["source_record_locator"],
                    source_content_hash=ev.get("source_content_hash"),
                    observation_timestamp_utc=ev.get("observation_timestamp_utc"),
                    transformation_rule=ev.get("transformation_rule"),
                    mapping_version=ev.get("mapping_version", "1.0.0"),
                    does_certify_claim=ev.get("does_certify_claim", False),
                ))

            cand_proposal = CandidateAssociationProposal(
                symbol=p_dict["symbol"],
                observed_venue=p_dict["observed_venue"],
                instrument_classification=p_dict["instrument_classification"],
                available_figis=p_dict["available_figis"],
                provider_asset_identifiers=p_dict["provider_asset_identifiers"],
                issuer_cik=p_dict.get("issuer_cik"),
                issuer_name=p_dict.get("issuer_name"),
                evidence_references=ev_refs,
                proposed_entity=ent,
                proposed_association=assoc,
                primary_listing_analysis=pla,
                ambiguities_and_missing_evidence=p_dict.get("ambiguities_and_missing_evidence", []),
                review_disposition=p_dict.get("review_disposition", "READY_FOR_REVIEW"),
                archived_parent_venue=p_dict.get("archived_parent_venue"),
                provider_claimed_venue=p_dict.get("provider_claimed_venue"),
                normalized_venue=p_dict.get("normalized_venue"),
                alternative_mics=tuple(p_dict.get("alternative_mics", ())),
                evidence_nature_summary=p_dict.get("evidence_nature_summary", {}),
            )

            ledger.add_proposal(cand_proposal)

        return ledger

    @classmethod
    def from_json(cls, json_str: str) -> IdentityAssociationProposalLedger:

        return cls.from_dict(json.loads(json_str))


# =====================================================================
# 5. Guarded Proposal-to-Validation Handoff
# =====================================================================

@dataclass
class ProposalValidationOutcome:
    """Outcome of proposal validation and bounded simulation."""
    is_simulation: bool
    production_adoption_allowed: bool
    total_evaluated: int
    validated_records_count: int
    validation_passed_count: int
    validation_failed_count: int
    conversion_blocked_count: int
    batch_valid: bool
    builder_status: str
    remaining_blockers: List[str]
    disposition_by_symbol: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_simulation": self.is_simulation,
            "production_adoption_allowed": self.production_adoption_allowed,
            "total_evaluated": self.total_evaluated,
            "validated_records_count": self.validated_records_count,
            "validation_passed_count": self.validation_passed_count,
            "validation_failed_count": self.validation_failed_count,
            "conversion_blocked_count": self.conversion_blocked_count,
            "batch_valid": self.batch_valid,
            "builder_status": self.builder_status,
            "remaining_blockers": self.remaining_blockers,
            "disposition_by_symbol": self.disposition_by_symbol,
        }


class GuardedProposalHandoff:
    """
    Guarded adapter connecting proposed associations to validation.
    Enforces separation between:
    A. Actual evidence-backed state (production identity adoption DISABLED).
    B. Clearly labeled proposal simulation (PROPOSAL_SIMULATION).
    C. Concrete remaining blockers preventing real-world admission.
    """

    @classmethod
    def evaluate_real_source_state(
        cls,
        ledger: IdentityAssociationProposalLedger,
        requirements: Optional[ValidationRequirements] = None,
    ) -> ProposalValidationOutcome:
        """
        State A: Evaluates actual real-source candidate state with production adoption disabled.
        Fails closed on missing production security_id and unevidenced primary_listing.
        """
        reqs = requirements or ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )

        proposals = ledger.list_proposals()
        disposition = {}
        failed_count = 0
        passed_count = 0

        for prop in proposals:
            # Under real state, security_id is omitted (unauthorized), and primary_listing is None
            raw_rec = {
                "symbol": prop.symbol,
                "exchange": prop.observed_venue,
                "security_type": prop.instrument_classification,
                "listing_status": "ACTIVE",
                "currency": "USD",
                "listing_country": "USA",
                "incorporation_country": prop.proposed_entity.issuer_cik,
                "selected_us_assessment_listing": True,
                "primary_listing": None,  # Preserved as UNKNOWN
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            }
            # security_id omitted
            res = UniverseInputValidator.validate_record(raw_rec, reqs)
            is_valid = (res.status == ValidationStatus.VALIDATION_PASSED)
            if is_valid:
                passed_count += 1
            else:
                failed_count += 1

            disposition[prop.symbol] = {
                "validation_status": res.status.value,
                "issues": [i.to_dict() for i in res.issues],
                "production_id_adopted": False,
                "primary_listing_state": "UNKNOWN",
                "conversion_permitted": False,
            }

        return ProposalValidationOutcome(
            is_simulation=False,
            production_adoption_allowed=False,
            total_evaluated=len(proposals),
            validated_records_count=len(proposals),
            validation_passed_count=passed_count,
            validation_failed_count=failed_count,
            conversion_blocked_count=len(proposals),
            batch_valid=False,  # Atomic fail-closed
            builder_status="NOT_RUN",
            remaining_blockers=[
                "Missing production security_id authority (provisional proposal IDs are non-production).",
                "Unknown primary_listing prohibits SourceSecurity conversion under explicit profile.",
                "Atomic batch revocation revokes downstream admission.",
            ],
            disposition_by_symbol=disposition,
        )

    @classmethod
    def simulate_validation_handoff(
        cls,
        ledger: IdentityAssociationProposalLedger,
        requirements: Optional[ValidationRequirements] = None,
        simulate_primary_listing_resolution: bool = False,
    ) -> ProposalValidationOutcome:
        """
        State B: Clearly labeled proposal simulation.
        Passes proposed provisional IDs into validation to exercise schema pathways.
        Explicitly labeled as simulation; DOES NOT authorize real production admission.
        """
        reqs = requirements or ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )

        proposals = ledger.list_proposals()
        disposition = {}
        passed_count = 0
        failed_count = 0
        blocked_conversion_count = 0

        for prop in proposals:
            primary_val = (
                True if simulate_primary_listing_resolution
                else prop.primary_listing_analysis.claimed_primary_listing
            )

            # Assign valid incorporation country string for simulation
            if prop.symbol in ("AAPL", "GOOG", "GOOGL", "DDS", "ACCS"):
                inc_country = "USA"
            elif prop.symbol == "TSM":
                inc_country = "TAIWAN"
            else:
                inc_country = "CAYMAN_ISLANDS"

            raw_rec = {
                "security_id": prop.proposed_entity.provisional_security_id,
                "symbol": prop.symbol,
                "exchange": prop.observed_venue,
                "security_type": prop.instrument_classification,
                "listing_status": "ACTIVE",
                "currency": "USD",
                "listing_country": "USA",
                "incorporation_country": inc_country,
                "selected_us_assessment_listing": True,
                "primary_listing": primary_val,
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            }

            res = UniverseInputValidator.validate_record(raw_rec, reqs)
            val_passed = (res.status == ValidationStatus.VALIDATION_PASSED)
            if val_passed:
                passed_count += 1
            else:
                failed_count += 1

            # Test SourceSecurity conversion
            can_convert = False
            conversion_error = None
            if val_passed:
                # INVARIANT: If proposal is disputed or venue is unresolved, conversion is strictly BLOCKED
                if prop.proposed_association.is_disputed or prop.proposed_association.operating_mic is None:
                    can_convert = False
                    conversion_error = (
                        f"Proposal handoff blocked by unresolved conflict: "
                        f"{prop.proposed_association.dispute_reason}"
                    )
                    blocked_conversion_count += 1
                else:
                    try:
                        to_source_security(res)
                        can_convert = True
                    except ValueError as ve:
                        can_convert = False
                        conversion_error = str(ve)
                        blocked_conversion_count += 1
            else:
                blocked_conversion_count += 1

            disposition[prop.symbol] = {
                "validation_status": res.status.value,
                "issues": [i.to_dict() for i in res.issues],
                "provisional_security_id_tested": prop.proposed_entity.provisional_security_id,
                "can_convert_to_source_security": can_convert,
                "conversion_error": conversion_error,
            }

        batch_valid = (failed_count == 0 and blocked_conversion_count == 0)
        builder_status = "READY_TO_EVALUATE" if batch_valid else "BLOCKED_ADMISSION_FAILED"

        blockers = []
        if blocked_conversion_count > 0:
            if not simulate_primary_listing_resolution:
                blockers.append("SourceSecurity conversion blocked: primary_listing remains UNKNOWN (None).")
            if any(p.proposed_association.is_disputed for p in proposals):
                blockers.append(
                    "SourceSecurity conversion blocked: association conflicts remain unresolved (DDS/ACCS)."
                )

        # Static rule compatibility observations (builder NOT_RUN)
        if any(p.symbol == "ACCS" for p in proposals):
            blockers.append(
                "ACCS venue 'AMEX': static rule compatibility observation against Rule R04 allowlist "
                "(NOT an executed eligibility failure; builder NOT_RUN)."
            )
        if any(p.symbol == "DDS" for p in proposals):
            blockers.append(
                "DDS venue 'TXSE': static rule compatibility observation against Rule R04 allowlist "
                "(NOT an executed eligibility failure; builder NOT_RUN)."
            )
        if any(p.symbol == "TSM" and p.instrument_classification == "Depositary Receipt" for p in proposals):
            blockers.append(
                "TSM structure 'Depositary Receipt': static rule compatibility observation against Rule R02 "
                "(COMMON_STOCK) — NOT an executed eligibility failure (builder NOT_RUN)."
            )
        if any(p.symbol == "AACIU" and p.instrument_classification == "Unit" for p in proposals):
            blockers.append(
                "AACIU structure 'Unit': static rule compatibility observation against Rule R02 "
                "(COMMON_STOCK) — NOT an executed eligibility failure (builder NOT_RUN)."
            )

        return ProposalValidationOutcome(

            is_simulation=True,
            production_adoption_allowed=False,
            total_evaluated=len(proposals),
            validated_records_count=len(proposals),
            validation_passed_count=passed_count,
            validation_failed_count=failed_count,
            conversion_blocked_count=blocked_conversion_count,
            batch_valid=batch_valid,
            builder_status=builder_status,
            remaining_blockers=blockers,
            disposition_by_symbol=disposition,
        )


# =====================================================================
# 6. Proposal Assembly for the Seven Candidates
# =====================================================================

def build_seven_candidate_proposal(research_root: Optional[Path] = None) -> IdentityAssociationProposalLedger:
    """
    Constructs the offline association proposal for the 7 canonical candidates:
    AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU.
    Loads locally retained research evidence if available; otherwise handles missing files cleanly.
    """
    ledger = IdentityAssociationProposalLedger()
    root = research_root or Path(__file__).resolve().parent.parent.parent / "data" / "research"

    openfigi_data = {}
    figi_path = root / "openfigi_sample" / "seven_candidates_openfigi_response.json"
    if figi_path.exists():
        try:
            f_raw = json.loads(figi_path.read_text(encoding="utf-8"))
            req_params = f_raw.get("request_parameters", [])
            for idx, item in enumerate(f_raw.get("response_payload", [])):
                req_sym = req_params[idx].get("idValue") if idx < len(req_params) else None
                if req_sym and "data" in item:
                    openfigi_data[req_sym] = item["data"]
        except Exception:
            openfigi_data = {}

    alpaca_data = {}
    alpaca_path = root / "alpaca_sample" / "seven_candidates_alpaca_assets.json"
    if alpaca_path.exists():
        try:
            alpaca_data = json.loads(alpaca_path.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            alpaca_data = {}

    as_of_date = "2026-10-17"
    now_utc = "2026-10-17T07:51:57Z"

    candidate_specs = [
        {
            "symbol": "AAPL",
            "observed_venue": "NASDAQ",
            "archived_parent_venue": "NASDAQ",
            "provider_claimed_venue": "NASDAQ",
            "normalized_venue": "NASDAQ",
            "mic": "XNAS",
            "alternative_mics": (),
            "cik": "0000320193",
            "company_name": "Apple Inc.",
            "instrument_structure": "Common Stock",
            "share_class": None,
            "home_market": "US (NASDAQ)",
            "primary_exchange": "NASDAQ",
            "disputed": False,
            "dispute_reason": None,
        },
        {
            "symbol": "GOOGL",
            "observed_venue": "NASDAQ",
            "archived_parent_venue": "NASDAQ",
            "provider_claimed_venue": "NASDAQ",
            "normalized_venue": "NASDAQ",
            "mic": "XNAS",
            "alternative_mics": (),
            "cik": "0001652044",
            "company_name": "Alphabet Inc.",
            "instrument_structure": "Common Stock",
            "share_class": "CLASS_A",
            "home_market": "US (NASDAQ)",
            "primary_exchange": "NASDAQ",
            "disputed": False,
            "dispute_reason": None,
        },
        {
            "symbol": "GOOG",
            "observed_venue": "NASDAQ",
            "archived_parent_venue": "NASDAQ",
            "provider_claimed_venue": "NASDAQ",
            "normalized_venue": "NASDAQ",
            "mic": "XNAS",
            "alternative_mics": (),
            "cik": "0001652044",
            "company_name": "Alphabet Inc.",
            "instrument_structure": "Common Stock",
            "share_class": "CLASS_C",
            "home_market": "US (NASDAQ)",
            "primary_exchange": "NASDAQ",
            "disputed": False,
            "dispute_reason": None,
        },
        {
            "symbol": "TSM",
            "observed_venue": "NYSE",
            "archived_parent_venue": "NYSE (Other Listed: N, SEC: NYSE)",
            "provider_claimed_venue": "NYSE",
            "normalized_venue": "NYSE",
            "mic": "XNYS",
            "alternative_mics": (),
            "cik": "0001046179",
            "company_name": "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD",
            "instrument_structure": "Depositary Receipt",
            "share_class": "ADR",
            "home_market": "TAIWAN (TWSE: 2330)",
            "primary_exchange": "NYSE",
            "disputed": False,
            "dispute_reason": None,
        },
        {
            "symbol": "ACCS",
            "observed_venue": "AMEX",
            "archived_parent_venue": "AMEX (Other Listed: A, SEC: NYSE)",
            "provider_claimed_venue": "AMEX",
            "normalized_venue": "AMEX",
            "mic": "XASE",
            "alternative_mics": (),
            "cik": "0000843006",
            "company_name": "ACCESS Newswire Inc.",
            "instrument_structure": "Common Stock",
            "share_class": None,
            "home_market": "US (AMEX / NYSE American)",
            "primary_exchange": "AMEX",
            "disputed": True,
            "dispute_reason": (
                "Dual-source venue discordance: SEC company tickers reports 'NYSE', whereas Nasdaq Other Listed "
                "directory reports 'A' (AMEX Tape B). Normalized to AMEX (XASE), but falls outside R04 allowlist."
            ),
        },
        {
            "symbol": "DDS",
            "observed_venue": "TXSE",
            "archived_parent_venue": "TXSE (SEC: TXSE, Other Listed: F)",
            "provider_claimed_venue": "NYSE (OpenFIGI: UN, Yahoo: NYQ, Alpaca: NYSE)",
            "normalized_venue": "TXSE (PROVISIONAL_UNRESOLVED, operating status NOT_ESTABLISHED)",
            "mic": None,  # UNRESOLVED: DO NOT EMIT CONCRETE XNYS ASSOCIATION
            "alternative_mics": ("XTXS", "XNYS"),
            "cik": "0000028917",
            "company_name": "DILLARD'S, INC.",
            "instrument_structure": "Common Stock",
            "share_class": "CLASS_A",
            "home_market": "US (TXSE / NYSE conflict)",
            "primary_exchange": "UNRESOLVED_VENUE_CONFLICT",
            "disputed": True,
            "dispute_reason": (
                "Cross-source venue conflict: Official 2026-10-16 parent feeds (SEC: TXSE, Other Listed: F) "
                "concordantly record Texas Stock Exchange (operating status NOT_ESTABLISHED), whereas later "
                "provider enrichment (OpenFIGI: UN, Yahoo: NYQ, Alpaca: NYSE) claims NYSE. Concrete MIC association "
                "is omitted pending operational venue confirmation."
            ),
        },
        {
            "symbol": "AACIU",
            "observed_venue": "NASDAQ",
            "archived_parent_venue": "NASDAQ",
            "provider_claimed_venue": "NASDAQ",
            "normalized_venue": "NASDAQ",
            "mic": "XNAS",
            "alternative_mics": (),
            "cik": "0002092897",
            "company_name": "Armada Acquisition Corp. II",
            "instrument_structure": "Unit",
            "share_class": "UNIT",
            "home_market": "CAYMAN_ISLANDS / US_LISTED",
            "primary_exchange": "NASDAQ",
            "disputed": False,
            "dispute_reason": None,
        },
    ]

    for spec in candidate_specs:
        sym = spec["symbol"]
        cik = spec["cik"]

        figi_matches = openfigi_data.get(sym, [])
        comp_figi = None
        sc_figi = None
        listing_figi = None
        if figi_matches:
            for m in figi_matches:
                if m.get("ticker") == sym:
                    comp_figi = m.get("compositeFIGI")
                    sc_figi = m.get("shareClassFIGI")
                    listing_figi = m.get("figi")
                    break
            if not comp_figi and len(figi_matches) > 0:
                comp_figi = figi_matches[0].get("compositeFIGI")
                sc_figi = figi_matches[0].get("shareClassFIGI")
                listing_figi = figi_matches[0].get("figi")

        alp_info = alpaca_data.get(sym, {})
        alp_id = alp_info.get("id") if isinstance(alp_info, dict) else None

        ext_ids = []
        if comp_figi:
            ext_ids.append(ExternalIdentifierRecord(
                id_type=ExternalIdentifierType.COMPOSITE_FIGI,
                id_value=comp_figi,
                provider="OpenFIGI",
                retrieval_timestamp_utc="2026-10-17T07:49:09Z",
                is_entity_level=False,
                notes="OpenFIGI composite instrument identifier.",
            ))
        if sc_figi:
            ext_ids.append(ExternalIdentifierRecord(
                id_type=ExternalIdentifierType.SHARE_CLASS_FIGI,
                id_value=sc_figi,
                provider="OpenFIGI",
                retrieval_timestamp_utc="2026-10-17T07:49:09Z",
                is_entity_level=False,
                notes="OpenFIGI share class identifier.",
            ))
        if listing_figi:
            ext_ids.append(ExternalIdentifierRecord(
                id_type=ExternalIdentifierType.LISTING_FIGI,
                id_value=listing_figi,
                provider="OpenFIGI",
                retrieval_timestamp_utc="2026-10-17T07:49:09Z",
                is_entity_level=False,
                notes="Venue-specific listing FIGI.",
            ))
        if cik:
            ext_ids.append(ExternalIdentifierRecord(
                id_type=ExternalIdentifierType.CIK_ENTITY_LEVEL,
                id_value=cik,
                provider="SEC_EDGAR",
                retrieval_timestamp_utc="2026-10-17T07:51:57Z",
                is_entity_level=True,
                notes="Corporate issuer CIK. Identifies entity, NOT individual security line.",
            ))
        if alp_id:
            ext_ids.append(ExternalIdentifierRecord(
                id_type=ExternalIdentifierType.ALPACA_ASSET_ID,
                id_value=alp_id,
                provider="Alpaca",
                retrieval_timestamp_utc="2026-10-17T07:51:57Z",
                is_entity_level=False,
                notes="Brokerage routing directory asset ID.",
            ))

        mic_key = spec["mic"] or "UNRESOLVED"
        id_material = f"{cik}:{sym}:{mic_key}:{spec.get('share_class') or 'BASE'}"
        id_hash = hashlib.sha256(id_material.encode("utf-8")).hexdigest()[:10].upper()
        prov_sec_id = f"PROV_SEC_{id_hash}"
        clean_date = as_of_date.replace("-", "")
        if spec["mic"]:
            prov_assoc_id = f"PROV_ASSOC_{prov_sec_id}_{spec['mic']}_{sym}_{clean_date}"
            val_basis = "EVIDENCE_OBSERVATION_WINDOW"
            temp_limits = (
                "Valid for observation date 2026-10-17. Inception / IPO historical backdating is NOT evidenced."
            )
        else:
            prov_assoc_id = f"PROV_ASSOC_UNRESOLVED_{sym}_{clean_date}"
            val_basis = "UNRESOLVED_VENUE_CONFLICT"
            temp_limits = (
                "Venue identity unresolved between archived parent observations (TXSE) and secondary provider "
                "claims (NYSE). Alternative candidate MICs: XTXS vs XNYS."
            )

        entity = ProvisionalSecurityEntity(
            provisional_security_id=prov_sec_id,
            primary_symbol=sym,
            share_class=spec["share_class"],
            instrument_structure=spec["instrument_structure"],
            issuer_cik=cik,
            company_name=spec["company_name"],
            external_identifiers=tuple(ext_ids),
            created_at_utc=now_utc,
            status="PROVISIONAL_NON_PRODUCTION",
            is_adopted_production_id=False,
            certifies_continuity=False,
        )

        assoc = ProvisionalListingAssociation(
            provisional_association_id=prov_assoc_id,
            provisional_security_id=prov_sec_id,
            symbol=sym,
            operating_mic=spec["mic"],
            alternative_mics=spec.get("alternative_mics", ()),
            effective_from=as_of_date,
            effective_to=None,
            validity_basis=val_basis,
            backdating_established=False,
            temporal_limitations=temp_limits,
            is_disputed=spec["disputed"],
            dispute_reason=spec["dispute_reason"],
        )

        exact_claim = (
            "contracts.SourceSecurity.primary_listing == True "
            "(Rule R06: 'Must be primary listing, not secondary or cross-listing')"
        )
        missing_doc = (
            "Parent feeds (Nasdaq Listed / Other Listed / SEC) omit explicit boolean primary_listing column. "
            "Candidate evidence sources include SEC filing cover pages (Form 10-K / 20-F check-boxes), "
            "official exchange directory master feeds, or exchange listing certifications."
        )

        if sym == "TSM":
            ambiguity_note = (
                "TSM global home market is Taiwan (TWSE: 2330). On NYSE, TSM trades as a Depositary Receipt. "
                "Rule R06 requires primary listing, but existing contracts do not specify whether ADR primary "
                "trading venue satisfies R06 when the underlying ordinary share is listed abroad. "
                "Field remains None (unknown) pending explicit policy qualification."
            )
        elif sym in ("GOOG", "GOOGL"):
            other_sym = "GOOGL" if sym == "GOOG" else "GOOG"
            ambiguity_note = (
                f"Alphabet Inc. issues Class A and Class C ({sym} vs {other_sym}). Both trade natively on NASDAQ. "
                "Directory feeds do not declare whether non-voting Class C constitutes a secondary listing."
            )
        elif sym == "AACIU":
            ambiguity_note = (
                "AACIU is a Unit consisting of underlying components. It trades on NASDAQ, but fails "
                "static rule compatibility observation against Rule R02 (COMMON_STOCK). "
                "Parent feeds omit explicit primary_listing boolean."
            )
        elif sym == "ACCS":
            ambiguity_note = (
                "SEC reports NYSE; Other Listed directory reports AMEX (Tape B). Dual-source venue discordance "
                "prevents unambiguous primary venue certification. Normalized venue AMEX (XASE) fails "
                "static rule compatibility observation against Rule R04 allowlist."
            )
        elif sym == "DDS":
            ambiguity_note = (
                "Official 2026-10-16 parent feeds (SEC: TXSE, Other Listed: F) concordantly designate "
                "Texas Stock Exchange (operating status NOT_ESTABLISHED), whereas later provider enrichment "
                "(OpenFIGI: UN, Yahoo: NYQ, Alpaca: NYSE) claims NYSE. "
                "Cross-source venue conflict remains unresolved; concrete MIC association cannot be emitted."
            )
        else:
            ambiguity_note = (
                "AAPL trades natively on NASDAQ as its sole global listing. However, parent directory feeds omit "
                "an attributable boolean primary_listing column."
            )

        pla = PrimaryListingClaimAnalysis(
            symbol=sym,
            exact_claim_needed=exact_claim,
            issuer_home_market=spec["home_market"],
            underlying_ordinary_share_listing=None,
            depositary_receipt_listing="NYSE (Tape A)" if sym == "TSM" else None,
            primary_exchange_for_instrument=spec["primary_exchange"],
            selected_us_assessment_listing=True,
            claimed_primary_listing=None,  # STRICT UNKNOWN PRESERVATION
            evidence_sufficiency="INSUFFICIENT_IN_RAW_FEEDS",
            missing_evidence_details=missing_doc,
            contract_ambiguity_notes=ambiguity_note,
        )

        ambiguities = []
        if spec["disputed"]:
            ambiguities.append(spec["dispute_reason"])
        if not comp_figi:
            ambiguities.append(f"Composite FIGI unevidenced for {sym}.")
        ambiguities.append("Parent feeds omit explicit boolean primary_listing column.")
        ambiguities.append("Inception / IPO date not established from single-day observation.")

        disposition = (
            "BLOCKED_ON_VENUE_CONFLICT"
            if sym == "DDS"
            else ("BLOCKED_ON_CONFLICT" if spec["disputed"] else "READY_FOR_REVIEW")
        )

        cand_proposal = CandidateAssociationProposal(
            symbol=sym,
            observed_venue=spec["observed_venue"],
            instrument_classification=spec["instrument_structure"],
            available_figis={
                "composite_figi": comp_figi,
                "share_class_figi": sc_figi,
                "listing_figi": listing_figi,
            },
            provider_asset_identifiers={
                "alpaca_id": alp_id,
                "alpaca_status": alp_info.get("status") if isinstance(alp_info, dict) else None,
                "alpaca_class": alp_info.get("asset_class") if isinstance(alp_info, dict) else None,
            },
            issuer_cik=cik,
            issuer_name=spec["company_name"],
            evidence_references=[],
            proposed_entity=entity,
            proposed_association=assoc,
            primary_listing_analysis=pla,
            ambiguities_and_missing_evidence=ambiguities,
            review_disposition=disposition,
            archived_parent_venue=spec.get("archived_parent_venue"),
            provider_claimed_venue=spec.get("provider_claimed_venue"),
            normalized_venue=spec.get("normalized_venue"),
            alternative_mics=spec.get("alternative_mics", ()),
            evidence_nature_summary={
                "venue_evidence": "REAL_OBSERVATION" if not spec["disputed"] else "UNRESOLVED_CONFLICT",
                "cik_evidence": "REAL_OBSERVATION" if cik else "UNAVAILABLE",
                "figi_evidence": "DERIVED_PROPOSAL" if comp_figi else "UNAVAILABLE",
                "classification_evidence": "REAL_OBSERVATION",
                "primary_listing_evidence": "UNKNOWN_PRESERVED",
            },
        )

        ledger.add_proposal(cand_proposal)

    return ledger
