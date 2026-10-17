"""
analyst_dashboard/universe/primary_listing_evidence.py

Sprint 2: Bounded Primary-Listing Evidence Ingestion & Offline Replay Engine.
Ingests attributable evidence addressing the primary_listing requirement for the 7-candidate sample:
(AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU).

Core Architectural Invariants:
1. Exact Claim Definition & Non-Inference:
   - primary_listing traces through SourceSecurity -> ValidationRequirements -> Rule R06.
   - Rule R06 requires primary listing, not secondary or cross-listing.
   - A filing's Section 12(b) exchange registration establishes registration on a named exchange
     without proving every possible interpretation of primary listing.
   - Does NOT infer True from selected_us_assessment_listing.
   - Does NOT infer False from foreign incorporation or an underlying share's foreign home market.
   - Does NOT infer primary listing from a FIGI, ticker match, or provider active status.
   - Preserves unknown primary_listing as None when evidence establishes only narrower facts
     or where contract semantics require an explicit policy decision.

2. Source & Temporal Provenance:
   - For every artifact, retains source provider, exact URL, accession, report date, raw response hash,
     exact table/section locator, stated exchange, symbol, and temporal limitations.
   - Does NOT equate valid address syntax with verified operator ownership.
   - Respects SEC fair access rate limits, finite timeouts, and non-invention rule.

3. Candidate Separation & Structural Fidelity:
   - Distinguishes multi-class issuers (GOOGL Class A voting vs GOOG Class C non-voting).
   - Distinguishes depositary receipts (TSM ADR on NYSE vs underlying ordinary shares on TWSE).
   - Preserves Unit structure (AACIU Unit consisting of share + warrant).
   - Preserves dual-source discordance (ACCS SEC: NYSE vs Other Listed: AMEX) without clearing.
   - Preserves cross-source venue conflict (DDS Parent: TXSE vs Providers/Filing: NYSE) without clearing.

4. Bounded Replay & Quarantine:
   - Zero production identity adoption (PROVISIONAL_ID_PROMOTION = NO).
   - Clearly labels simulation runs as PROPOSAL_SIMULATION.
   - Zero database writes, zero UniverseStore writes.
   - Atomic whole-batch conversion fails closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .admission_mapping import (
    AttributableFieldEvidence,
    EvidenceNature,
)
from .identity_proposal import (
    IdentityAssociationProposalLedger,
    PrimaryListingClaimAnalysis,
)
from .validator import (
    UniverseInputValidator,
    ValidationRequirements,
    ValidationStatus,
    to_source_security,
)


# =====================================================================
# 1. Primary-Listing Evidence Classification Taxonomy
# =====================================================================

class PrimaryListingEvidenceClassification(str, Enum):
    """
    Taxonomy of primary-listing evidence sufficiency per candidate.
    Distinguishes documentary proof of the defined claim from narrower listing facts,
    conflicts, or unresolved contract semantics.
    """
    EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM = (
        "EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM"
    )
    EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY = (
        "EVIDENCE_ESTABLISHES_NARROWER_LISTING_FACT_ONLY"
    )
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION = (
        "CONTRACT_SEMANTICS_REQUIRE_SPECIFIC_DECISION"
    )


# =====================================================================
# 2. Attributable Primary-Listing Evidence Record
# =====================================================================

@dataclass(frozen=True)
class AttributablePrimaryListingEvidenceRecord:
    """
    Attributable reference evidence record for an instrument's listing claim.
    Retains complete provenance, accession numbers, response hashes, and locators.
    """
    symbol: str
    cik: Optional[str]
    instrument_title: str
    instrument_structure: str
    share_class: Optional[str]
    source_provider: str
    source_url: str
    request_locator: str
    retrieval_timestamp_utc: str
    filing_accession: Optional[str]
    filing_report_date: Optional[str]
    filing_form: Optional[str]
    raw_response_sha256: str
    section_table_locator: str
    stated_exchange: str
    stated_symbol: str
    operating_mic: Optional[str]
    is_multi_class: bool
    effective_as_of_date: str
    temporal_limitations: str
    classification: PrimaryListingEvidenceClassification
    claim_supported: str
    inferences_prohibited: Tuple[str, ...]
    assigned_primary_listing: Optional[bool]
    contract_ambiguity_notes: str
    source_artifact: Optional[str] = None
    source_claim: Optional[str] = None
    publisher_primary_listing_claim: Optional[str] = None
    interpretation: Optional[str] = None
    prospective_lookup_target: Optional[str] = None
    inspected_filing_accession: Optional[str] = None
    inspected_filing_report_date: Optional[str] = None
    inspected_filing_artifact: Optional[str] = None
    inspected_filing_sha256: Optional[str] = None
    inspected_filing_url: Optional[str] = None
    inspected_table_locator: Optional[str] = None
    unfetched_filing_targets: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        d["inferences_prohibited"] = list(self.inferences_prohibited)
        return d

    def to_attributable_field_evidence(self) -> AttributableFieldEvidence:
        """
        Converts this record to the released AttributableFieldEvidence model.
        """
        is_real = (self.assigned_primary_listing is True)
        ev_nature = (
            EvidenceNature.DIRECT_PROVIDER_OBSERVATION
            if is_real
            else EvidenceNature.POLICY_DERIVED_ASSERTION
        )
        rule = (
            "NASDAQ_EXCHANGE_DIRECTORY_OBSERVATION"
            if self.filing_form == "NASDAQ_LISTED_DIRECTORY"
            else "SEC_EDGAR_SUBMISSIONS_API_METADATA_OBSERVATION"
        )
        return AttributableFieldEvidence(
            field_name="primary_listing",
            asserted_value=self.assigned_primary_listing,
            evidence_nature=ev_nature,
            source_identifier=self.source_provider,
            source_record_locator=self.section_table_locator,
            source_content_hash=self.raw_response_sha256,
            observation_timestamp_utc=self.retrieval_timestamp_utc,
            transformation_rule=rule,
            mapping_version="1.0.0",
            does_certify_claim=is_real,
        )


# =====================================================================
# 3. Source Adapter & SEC Contact Verification
# =====================================================================

class PrimaryListingEvidenceAdapter:
    """
    Bounded source adapter for primary-listing reference evidence.
    Loads authentic public filing reference metadata and enforces strict contact validation.
    """

    DEFAULT_EVIDENCE_PATH = Path("data/research/primary_listing_evidence/seven_candidates_primary_listing_evidence.json")
    DEFAULT_THREE_CANDIDATES_EVIDENCE_PATH = Path("data/research/primary_listing_evidence/three_candidates_listing_evidence.json")
    DEFAULT_NASDAQ_DIRECTORY_PATH = Path("data/research/exchange_reference/nasdaqlisted_20261017.txt")

    AUTHORITATIVE_CANDIDATE_CIK_MAPPING = {
        "AAPL": "0000320193",
        "GOOGL": "0001652044",
        "GOOG": "0001652044",
        "TSM": "0001046179",
        "ACCS": "0000843006",
        "DDS": "0000028917",
        "AACIU": "0002092897",
    }

    @classmethod
    def validate_candidate_issuer_binding(
        cls,
        symbol: str,
        evidence: AttributablePrimaryListingEvidenceRecord,
        expected_cik: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that an attributable evidence record matches the candidate's verified issuer CIK.
        Strictly rejects evidence reuse from another issuer, similarly named SPAC, or historical ticker.
        """
        authoritative_cik = expected_cik or cls.AUTHORITATIVE_CANDIDATE_CIK_MAPPING.get(symbol)
        if authoritative_cik is None:
            return True, None
        if evidence.cik != authoritative_cik:
            return (
                False,
                f"REJECTED_MISMATCHED_ISSUER_EVIDENCE: Evidence CIK '{evidence.cik}' does not match "
                f"verified candidate CIK '{authoritative_cik}' for '{symbol}' (rejects cross-issuer or cross-SPAC evidence reuse)",
            )
        return True, None

    @classmethod
    def get_operator_sec_contact_status(cls) -> Dict[str, Any]:
        """
        Inspects environment for a verified operator-supplied SEC contact identity.
        Enforces SEC fair access non-invention rule:
        Valid address syntax alone is NOT equated with verified operator ownership.
        """
        sec_ua = os.environ.get("SEC_USER_AGENT", "").strip()
        sec_email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()

        # Check for genuine operator configuration
        is_verified = bool(sec_ua and sec_email and ("@" in sec_email) and ("example.com" not in sec_email))

        if is_verified:
            return {
                "verified": True,
                "user_agent_declared": True,
                "access_blocker": None,
                "operator_contact": sec_email,
            }
        else:
            return {
                "verified": False,
                "user_agent_declared": bool(sec_ua),
                "access_blocker": (
                    "OPERATOR_SEC_CONTACT_UNVERIFIED: Genuine operator-supplied SEC contact identity "
                    "(SEC_USER_AGENT / SEC_CONTACT_EMAIL) is not configured in local environment. "
                    "Live EDGAR retrieval is bounded/blocked under SEC Fair Access Compliance to prevent "
                    "unverified operator attribution. Authentically archived public reference metadata is reused."
                ),
                "operator_contact": None,
            }

    @classmethod
    def load_reference_evidence(
        cls,
        evidence_path: Optional[Path] = None,
    ) -> Dict[str, AttributablePrimaryListingEvidenceRecord]:
        """
        Loads attributable reference evidence from authentic JSON dataset.
        Returns empty dict on missing files without synthesizing data.
        """
        target_path = evidence_path or cls.DEFAULT_EVIDENCE_PATH
        if not target_path.exists():
            return {}

        try:
            content = target_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception:
            return {}

        records: Dict[str, AttributablePrimaryListingEvidenceRecord] = {}
        for sym, rec_dict in data.get("evidence_records", {}).items():
            records[sym] = AttributablePrimaryListingEvidenceRecord(
                symbol=rec_dict["symbol"],
                cik=rec_dict.get("cik"),
                instrument_title=rec_dict["instrument_title"],
                instrument_structure=rec_dict["instrument_structure"],
                share_class=rec_dict.get("share_class"),
                source_provider=rec_dict["source_provider"],
                source_url=rec_dict["source_url"],
                request_locator=rec_dict["request_locator"],
                retrieval_timestamp_utc=rec_dict["retrieval_timestamp_utc"],
                filing_accession=rec_dict.get("filing_accession"),
                filing_report_date=rec_dict.get("filing_report_date"),
                filing_form=rec_dict.get("filing_form"),
                raw_response_sha256=rec_dict["raw_response_sha256"],
                section_table_locator=rec_dict["section_table_locator"],
                stated_exchange=rec_dict["stated_exchange"],
                stated_symbol=rec_dict["stated_symbol"],
                operating_mic=rec_dict.get("operating_mic"),
                is_multi_class=rec_dict.get("is_multi_class", False),
                effective_as_of_date=rec_dict["effective_as_of_date"],
                temporal_limitations=rec_dict["temporal_limitations"],
                classification=PrimaryListingEvidenceClassification(rec_dict["classification"]),
                claim_supported=rec_dict["claim_supported"],
                inferences_prohibited=tuple(rec_dict.get("inferences_prohibited", ())),
                assigned_primary_listing=rec_dict.get("assigned_primary_listing"),
                contract_ambiguity_notes=rec_dict["contract_ambiguity_notes"],
                source_artifact=rec_dict.get("source_artifact"),
                source_claim=rec_dict.get("source_claim"),
                publisher_primary_listing_claim=rec_dict.get("publisher_primary_listing_claim"),
                interpretation=rec_dict.get("interpretation"),
                prospective_lookup_target=rec_dict.get("prospective_lookup_target"),
                inspected_filing_accession=rec_dict.get("inspected_filing_accession"),
                inspected_filing_report_date=rec_dict.get("inspected_filing_report_date"),
                inspected_filing_artifact=rec_dict.get("inspected_filing_artifact"),
                inspected_filing_sha256=rec_dict.get("inspected_filing_sha256"),
                inspected_filing_url=rec_dict.get("inspected_filing_url"),
                inspected_table_locator=rec_dict.get("inspected_table_locator"),
                unfetched_filing_targets=rec_dict.get("unfetched_filing_targets"),
            )

        return records

    @classmethod
    def load_three_candidate_evidence(
        cls,
        evidence_path: Optional[Path] = None,
    ) -> Dict[str, AttributablePrimaryListingEvidenceRecord]:
        """
        Loads authentic three-candidate listing evidence (AAPL, GOOGL, GOOG).
        """
        target = evidence_path or cls.DEFAULT_THREE_CANDIDATES_EVIDENCE_PATH
        return cls.load_reference_evidence(evidence_path=target)

    @classmethod
    def enrich_candidate_proposals(
        cls,
        ledger: IdentityAssociationProposalLedger,
        evidence_records: Dict[str, AttributablePrimaryListingEvidenceRecord],
    ) -> IdentityAssociationProposalLedger:
        """
        Enriches candidate proposals in the proposal ledger with attributable primary-listing evidence.
        Updates PrimaryListingClaimAnalysis with exact evidence sufficiency and claims.
        """
        for sym, ev in evidence_records.items():
            prop = ledger.get_proposal(sym)
            if not prop:
                continue

            # Validate issuer binding before attributing
            is_valid_binding, binding_err = cls.validate_candidate_issuer_binding(
                sym, ev, expected_cik=prop.proposed_entity.issuer_cik
            )
            if not is_valid_binding:
                prop.ambiguities_and_missing_evidence.append(binding_err or "MISMATCHED_ISSUER_CIK")
                continue

            # Attach attributable field evidence
            ev_field = ev.to_attributable_field_evidence()
            prop.evidence_references.append(ev_field)

            # Update primary listing claim analysis with empirical evidence
            suff_str = (
                "SUFFICIENT_FOR_DEFINED_CLAIM"
                if ev.classification == PrimaryListingEvidenceClassification.EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM
                else ("CONFLICTING_EVIDENCE" if ev.classification == PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE else "INSUFFICIENT_OR_AMBIGUOUS")
            )

            updated_pla = PrimaryListingClaimAnalysis(
                symbol=sym,
                exact_claim_needed=prop.primary_listing_analysis.exact_claim_needed,
                issuer_home_market=prop.primary_listing_analysis.issuer_home_market,
                underlying_ordinary_share_listing=prop.primary_listing_analysis.underlying_ordinary_share_listing,
                depositary_receipt_listing=prop.primary_listing_analysis.depositary_receipt_listing,
                primary_exchange_for_instrument=ev.stated_exchange,
                selected_us_assessment_listing=prop.primary_listing_analysis.selected_us_assessment_listing,
                claimed_primary_listing=ev.assigned_primary_listing,
                evidence_sufficiency=suff_str,
                missing_evidence_details=ev.claim_supported,
                contract_ambiguity_notes=ev.contract_ambiguity_notes,
            )
            prop.primary_listing_analysis = updated_pla

            # Update evidence nature summary
            prop.evidence_nature_summary["primary_listing_evidence"] = (
                "REAL_DOCUMENTARY_OBSERVATION"
                if ev.assigned_primary_listing is not None
                else "NARROWER_FACT_OR_AMBIGUOUS_PRESERVED"
            )

        return ledger


# =====================================================================
# 4. Bounded Replay Engine & Candidate-Level Results
# =====================================================================

@dataclass
class CandidateReplayResult:
    """
    Individual candidate replay audit result.
    Explicitly labeled as an isolated contract probe without admission authority.
    """
    symbol: str
    provisional_security_id: str
    instrument_structure: str
    observed_venue: str
    operating_mic: Optional[str]
    evidence_classification: PrimaryListingEvidenceClassification
    primary_listing_value: Optional[bool]
    newly_evidenced_fields: List[str]
    remaining_uncertainty: List[str]
    validation_status: str
    validation_issues: List[str]
    conversion_permitted: bool
    conversion_error: Optional[str]
    builder_status: str
    blocking_reason: str
    is_isolated_probe: bool = True
    has_admission_authority: bool = False
    batch_admitted: bool = False
    isolated_probe_note: str = "Isolated simulation probe only; carries zero admission authority."

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_classification"] = self.evidence_classification.value
        return d


@dataclass
class BatchReplayReport:
    """
    Complete batch replay outcome enforcing atomic whole-batch conversion gating.
    Separates isolated contract probes from actual whole-batch admission.
    """
    is_simulation: bool
    production_adoption_allowed: bool
    total_evaluated: int
    newly_evidenced_symbols: List[str]
    validation_passed_count: int
    validation_failed_count: int
    conversion_count: int  # Count of isolated probes that passed conversion
    conversion_blocked_count: int  # Count of isolated probes blocked
    is_batch_valid: bool
    builder_status: str
    candidate_results: Dict[str, CandidateReplayResult]
    blocking_summary: List[str]
    batch_conversion_count: int = 0  # Actual SourceSecurity count from whole-batch gate
    converted_source_securities: List[Any] = field(default_factory=list)
    batch_admitted_count: int = 0
    builder_handoff_permitted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_simulation": self.is_simulation,
            "production_adoption_allowed": self.production_adoption_allowed,
            "total_evaluated": self.total_evaluated,
            "newly_evidenced_symbols": self.newly_evidenced_symbols,
            "validation_passed_count": self.validation_passed_count,
            "validation_failed_count": self.validation_failed_count,
            "conversion_count": self.conversion_count,
            "conversion_blocked_count": self.conversion_blocked_count,
            "batch_conversion_count": self.batch_conversion_count,
            "batch_admitted_count": self.batch_admitted_count,
            "is_batch_valid": self.is_batch_valid,
            "builder_status": self.builder_status,
            "builder_handoff_permitted": self.builder_handoff_permitted,
            "candidate_results": {k: v.to_dict() for k, v in self.candidate_results.items()},
            "blocking_summary": self.blocking_summary,
        }


class PrimaryListingReplayEngine:
    """
    Executes bounded offline replay with newly ingested primary-listing evidence.
    Enforces strict separation between:
    A. Isolated per-record synthetic/proposal contract probes.
    B. Actual whole-batch admission and conversion using the existing batch gate.
    """

    @classmethod
    def run_replay(
        cls,
        ledger: IdentityAssociationProposalLedger,
        evidence_records: Dict[str, AttributablePrimaryListingEvidenceRecord],
        is_simulation: bool = False,
    ) -> BatchReplayReport:
        """
        Executes bounded offline replay enforcing atomic batch gating.
        """
        reqs = ValidationRequirements.create_listing_jurisdiction_profile(
            allowed_listing_countries={"USA"},
            require_selected_us_assessment=True,
        )

        candidate_results: Dict[str, CandidateReplayResult] = {}
        passed_count = 0
        failed_count = 0
        probe_converted_count = 0
        probe_conversion_blocked = 0
        newly_evidenced: List[str] = []
        blocking_reasons: List[str] = []
        raw_records: List[Dict[str, Any]] = []

        proposals = ledger.list_proposals()

        # Step 1: Isolated per-record probes
        for prop in proposals:
            sym = prop.symbol
            ev = evidence_records.get(sym)

            # Check issuer binding
            is_valid_binding, binding_err = (
                PrimaryListingEvidenceAdapter.validate_candidate_issuer_binding(
                    sym, ev, expected_cik=prop.proposed_entity.issuer_cik
                )
                if ev
                else (True, None)
            )

            if not is_valid_binding:
                ev_class = PrimaryListingEvidenceClassification.CONFLICTING_EVIDENCE
                assigned_pl = None
            else:
                ev_class = ev.classification if ev else PrimaryListingEvidenceClassification.EVIDENCE_UNAVAILABLE
                assigned_pl = ev.assigned_primary_listing if ev else None

            if assigned_pl is not None:
                newly_evidenced.append(sym)

            # Determine security_id used
            if is_simulation and is_valid_binding:
                sec_id = prop.proposed_entity.provisional_security_id
            else:
                sec_id = None  # Production adoption DISABLED or mismatched issuer rejected

            # Build raw record for validation
            raw_rec = {
                "symbol": sym,
                "exchange": prop.observed_venue,
                "security_type": prop.instrument_classification,
                "listing_status": "ACTIVE",
                "currency": "USD",
                "listing_country": "USA",
                "incorporation_country": prop.proposed_entity.issuer_cik if is_valid_binding else "INVALID_ISSUER_CIK",
                "selected_us_assessment_listing": True,
                "primary_listing": assigned_pl,
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            }
            if sec_id:
                raw_rec["security_id"] = sec_id

            raw_records.append(raw_rec)

            val_res = UniverseInputValidator.validate_record(raw_rec, reqs)
            val_status = val_res.status.value
            val_issues = [f"{i.field_name}: {i.issue_code.value}" for i in val_res.issues]
            if not is_valid_binding:
                val_issues.append(f"issuer_cik: {binding_err}")

            if val_res.status == ValidationStatus.VALIDATION_PASSED and is_valid_binding:
                passed_count += 1
            else:
                failed_count += 1

            # Isolated probe of SourceSecurity conversion
            can_convert = False
            conv_err = None
            if val_res.status == ValidationStatus.VALIDATION_PASSED and is_valid_binding:
                try:
                    to_source_security(val_res)
                    can_convert = True
                    probe_converted_count += 1
                except Exception as e:
                    can_convert = False
                    conv_err = str(e)
                    probe_conversion_blocked += 1
            else:
                probe_conversion_blocked += 1
                if not is_valid_binding:
                    conv_err = binding_err
                else:
                    conv_err = f"Validation failed ({val_status})"

            # Remaining uncertainty
            uncertainties = list(prop.ambiguities_and_missing_evidence)
            if not is_valid_binding:
                uncertainties.append(binding_err or "REJECTED_MISMATCHED_ISSUER_EVIDENCE")
            elif ev:
                uncertainties.append(ev.contract_ambiguity_notes)

            # Blocking reason determination
            if not is_valid_binding:
                block_reason = binding_err or "REJECTED_MISMATCHED_ISSUER_EVIDENCE"
            elif not is_simulation:
                block_reason = "PRODUCTION_IDENTITY_ADOPTION_DISABLED: security_id unadopted"
            elif assigned_pl is None:
                block_reason = f"UNKNOWN_PRIMARY_LISTING: {ev_class.value}"
            elif prop.proposed_association.is_disputed:
                block_reason = f"UNRESOLVED_VENUE_CONFLICT: {prop.proposed_association.dispute_reason}"
            elif not can_convert:
                block_reason = f"CONVERSION_BLOCKED: {conv_err}"
            else:
                block_reason = "ISOLATED_PROBE_ONLY: NO_BATCH_ADMISSION_AUTHORITY"

            blocking_reasons.append(f"{sym}: {block_reason}")

            candidate_results[sym] = CandidateReplayResult(
                symbol=sym,
                provisional_security_id=prop.proposed_entity.provisional_security_id,
                instrument_structure=prop.instrument_classification,
                observed_venue=prop.observed_venue,
                operating_mic=prop.proposed_association.operating_mic,
                evidence_classification=ev_class,
                primary_listing_value=assigned_pl,
                newly_evidenced_fields=["primary_listing"] if assigned_pl is not None else [],
                remaining_uncertainty=uncertainties,
                validation_status=val_status,
                validation_issues=val_issues,
                conversion_permitted=can_convert,
                conversion_error=conv_err,
                builder_status="NOT_RUN",
                blocking_reason=block_reason,
                is_isolated_probe=True,
                has_admission_authority=False,
                batch_admitted=False,
            )

        # Step 2: Whole-Batch Admission and Conversion Gate
        batch_val_res = UniverseInputValidator.validate_batch(raw_records, reqs)
        is_batch_valid = batch_val_res.is_batch_valid

        converted_securities: List[Any] = []
        batch_conv_count = 0
        batch_admitted = 0
        builder_status = "NOT_RUN"
        builder_permitted = False

        if is_batch_valid:
            if is_simulation:
                try:
                    converted_securities = batch_val_res.to_source_securities()
                    batch_conv_count = len(converted_securities)
                    batch_admitted = batch_conv_count
                    builder_status = "READY_FOR_BUILDER"
                    builder_permitted = True
                    for c in candidate_results.values():
                        c.batch_admitted = True
                        c.builder_status = "READY_FOR_BUILDER"
                except Exception:
                    # Atomic whole-batch failure: if any member fails conversion (e.g. primary_listing is None),
                    # whole-batch conversion strictly fails closed with zero admitted.
                    converted_securities = []
                    batch_conv_count = 0
                    batch_admitted = 0
                    builder_status = "NOT_RUN"
                    builder_permitted = False
                    is_batch_valid = False
                    for c in candidate_results.values():
                        c.batch_admitted = False
                        c.builder_status = "NOT_RUN"
            else:
                # Production adoption disabled
                converted_securities = []
                batch_conv_count = 0
                batch_admitted = 0
                builder_status = "NOT_RUN"
                builder_permitted = False
        else:
            # Batch failed: Atomic whole-batch contract strictly blocks partial conversion
            converted_securities = []
            batch_conv_count = 0
            batch_admitted = 0
            builder_status = "NOT_RUN"
            builder_permitted = False
            for c in candidate_results.values():
                c.batch_admitted = False
                c.builder_status = "NOT_RUN"

        return BatchReplayReport(
            is_simulation=is_simulation,
            production_adoption_allowed=False,
            total_evaluated=len(candidate_results),
            newly_evidenced_symbols=newly_evidenced,
            validation_passed_count=passed_count,
            validation_failed_count=failed_count,
            conversion_count=probe_converted_count,
            conversion_blocked_count=probe_conversion_blocked,
            is_batch_valid=is_batch_valid,
            builder_status=builder_status,
            candidate_results=candidate_results,
            blocking_summary=blocking_reasons,
            batch_conversion_count=batch_conv_count,
            converted_source_securities=converted_securities,
            batch_admitted_count=batch_admitted,
            builder_handoff_permitted=builder_permitted,
        )
