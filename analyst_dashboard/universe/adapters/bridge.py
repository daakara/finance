"""
analyst_dashboard/universe/adapters/bridge.py

Adapter-to-Validator Bridge for ARX Terminal universe inputs.

Strict Non-Inference Invariants:
- Passes only actually observed or explicitly derived source values to UniverseInputValidator.
- Preserves evidence references across all participating source feeds.
- Leaves unsupported required fields missing/omitted; NEVER fabricates passing defaults.
- Respects whole-batch failure semantics: if any candidate or batch-level requirement fails,
  downstream admission is strictly revoked for all batch members.
- Differentiates parser syntax validity from complete universe field certification.
- Pure in-memory execution: zero file reads, zero network requests, zero store writes.
"""

from __future__ import annotations

from typing import Any

from ..validator import (
    EvidenceReference,
    TemporalEvaluationContext,
    UniverseInputValidator,
    ValidationRequirements,
)
from .models import (
    AdapterValidationReport,
    FieldCertificationStatus,
    MatchedUniverseCandidate,
    OpenFigiMatchStatus,
    SourceMatchingAccounting,
)


class AdapterValidatorBridge:
    """
    Bridges offline parsed source observations into UniverseInputValidator.
    Surfaces explicit evidence gaps as expected validation failures rather than
    masking them with uncertified defaults.
    """

    @staticmethod
    def build_candidate_raw_record(
        candidate: MatchedUniverseCandidate,
        caller_security_id_mapping: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], list[EvidenceReference]]:
        """
        Extracts only actually observed source attributes from a candidate.
        Fields not present in primary source bytes are strictly omitted.

        Identity Policy Boundary:
        - Built-in surrogate-key generation (e.g. CIK_{cik}_{symbol}) is removed.
        - Production instrument identity scheme remains unresolved by policy.
        - OpenFIGI returns (figi, compositeFIGI, shareClassFIGI) are preserved distinctly
          as observed external identifiers without automatically designating any as security_id.
        - If caller provides caller_security_id_mapping, it is accepted as an external pass-through
          and does NOT constitute certified primary source attestation.
        """
        raw: dict[str, Any] = {}
        evidences: list[EvidenceReference] = []

        raw["symbol"] = candidate.join_key

        # 1. SEC Observations
        if candidate.sec_row and not candidate.sec_row.is_malformed:
            sec_vals = candidate.sec_row.raw_values
            if sec_vals.get("company_name"):
                raw["company_name"] = sec_vals["company_name"]
            if sec_vals.get("sec_exchange"):
                raw["exchange"] = sec_vals["sec_exchange"]
            if "issuer_cik" in sec_vals:
                raw["issuer_cik"] = sec_vals["issuer_cik"]

            sec_id = candidate.sec_source_identifier or "company_tickers_exchange.json"
            row_idx = candidate.sec_row.row_index
            evidences.append(
                EvidenceReference(
                    source_authority="SEC_EDGAR",
                    source_identifier=f"{sec_id}#row={row_idx}",
                    source_hash=candidate.sec_content_hash,
                    is_external_attestation=True,
                )
            )

        # 2. Directory Observations
        if candidate.directory_row and not candidate.directory_row.is_malformed:
            dir_vals = candidate.directory_row.raw_values
            if "security_name" in dir_vals and "company_name" not in raw:
                raw["company_name"] = dir_vals["security_name"]
            if dir_vals.get("venue_exchange"):
                raw["exchange"] = dir_vals["venue_exchange"]
            if "is_etf" in dir_vals:
                raw["is_etf"] = dir_vals["is_etf"]

            dir_src = candidate.directory_source_identifier or dir_vals.get("source_directory", "nasdaq_directory")
            row_idx = candidate.directory_row.row_index
            evidences.append(
                EvidenceReference(
                    source_authority="NASDAQ_TRADER",
                    source_identifier=f"{dir_src}#row={row_idx}",
                    source_hash=candidate.directory_content_hash,
                    is_external_attestation=True,
                )
            )

        # 3. OpenFIGI Observations (only single, unambiguous, associated matches populate fields)
        if (
            candidate.figi_result
            and candidate.figi_result.matches
            and candidate.figi_result.match_status == OpenFigiMatchStatus.SINGLE_MATCH
            and not candidate.is_ambiguous
            and not any("OPENFIGI_UNASSOCIATED" in c for c in candidate.join_conflicts)
        ):
            match = candidate.figi_result.matches[0]
            # Preserve distinct FIGI levels without automatically designating security_id:
            if match.get("figi"):
                raw["figi"] = match.get("figi")
            if match.get("listing_figi"):
                raw["listing_figi"] = match.get("listing_figi")
            if match.get("composite_figi"):
                raw["composite_figi"] = match.get("composite_figi")
            if match.get("share_class_figi"):
                raw["share_class_figi"] = match.get("share_class_figi")
            if match.get("security_type2"):
                raw["security_type"] = match.get("security_type2")

            figi_src = candidate.figi_source_identifier or "openfigi_mapping"
            req_idx = candidate.figi_result.request_index
            evidences.append(
                EvidenceReference(
                    source_authority="OPENFIGI",
                    source_identifier=f"{figi_src}#request={req_idx}",
                    source_hash=candidate.figi_content_hash,
                    is_external_attestation=True,
                )
            )

        # 4. Caller-supplied security_id mapping (external policy boundary)
        if caller_security_id_mapping and candidate.join_key in caller_security_id_mapping:
            raw["security_id"] = caller_security_id_mapping[candidate.join_key]

        # 5. Raw payload for full provenance retention
        raw["raw_payload"] = {
            "join_key": candidate.join_key,
            "has_sec": candidate.sec_row is not None,
            "has_directory": candidate.directory_row is not None,
            "has_figi": candidate.figi_result is not None,
            "conflicts": candidate.join_conflicts,
        }

        # CRITICAL INVARIANT: The following fields are NOT present in raw directory files:
        # - security_id: production identity scheme unresolved; omitted unless mapped by caller
        # - currency: absent in SEC and Nasdaq directories (cannot be inferred from positive pricing)
        # - country: absent in directory files
        # - primary_listing: directory inclusion does not verify primary listing status
        # - listing_date / delisting_date: historical temporal dates absent in current snapshot
        # - trading_suspension_date: real-time halt feed absent
        # These are left OMITTED so UniverseInputValidator correctly fails closed.

        return raw, evidences


    @staticmethod
    def assess_field_certifications(
        candidates: list[MatchedUniverseCandidate],
    ) -> dict[str, FieldCertificationStatus]:
        """
        Produces field-by-field certification diagnostic across all candidate observations.
        Identifies fields supported by empirical source evidence versus open policy gaps.
        """
        has_figi = any(c.figi_result and c.figi_result.selected_figi for c in candidates)
        has_sec = any(c.sec_row is not None for c in candidates)
        has_dir = any(c.directory_row is not None for c in candidates)

        return {
            "symbol": FieldCertificationStatus(
                field_name="symbol",
                is_certified=has_sec or has_dir,
                source_authorities=["SEC_EDGAR", "NASDAQ_TRADER"],
                limitation_reason=None,
            ),
            "company_name": FieldCertificationStatus(
                field_name="company_name",
                is_certified=has_sec or has_dir,
                source_authorities=["SEC_EDGAR", "NASDAQ_TRADER"],
                limitation_reason=None,
            ),
            "exchange": FieldCertificationStatus(
                field_name="exchange",
                is_certified=has_sec or has_dir,
                source_authorities=["SEC_EDGAR", "NASDAQ_TRADER"],
                limitation_reason="Venue representation differences require authoritative venue code mapping.",
            ),
            "security_id": FieldCertificationStatus(
                field_name="security_id",
                is_certified=False,
                source_authorities=[],
                limitation_reason=(
                    "Production instrument identity scheme (Listing FIGI vs Composite FIGI vs Share Class FIGI vs CIK) "
                    "remains unresolved by policy; security_id is not certified by directory or raw OpenFIGI feeds."
                ),
            ),
            "security_type": FieldCertificationStatus(
                field_name="security_type",
                is_certified=has_figi and not any(c.is_ambiguous for c in candidates),
                source_authorities=["OPENFIGI"] if (has_figi and not any(c.is_ambiguous for c in candidates)) else [],
                limitation_reason=(
                    None
                    if (has_figi and not any(c.is_ambiguous for c in candidates))
                    else "Absent in primary directory feeds or uncertified due to ambiguous/unassociated mapping; ETF=N does not prove COMMON_STOCK."
                ),
            ),
            "currency": FieldCertificationStatus(
                field_name="currency",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Absent across SEC and directory feeds. Prohibits USD inference from positive price.",
            ),
            "country": FieldCertificationStatus(
                field_name="country",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Absent in directory directory feeds.",
            ),
            "primary_listing": FieldCertificationStatus(
                field_name="primary_listing",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Directory presence does not resolve primary vs secondary/cross-listing status.",
            ),
            "listing_date": FieldCertificationStatus(
                field_name="listing_date",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Historical point-in-time listing date absent in current directory snapshot.",
            ),
            "delisting_date": FieldCertificationStatus(
                field_name="delisting_date",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Historical point-in-time delisting date absent in active directory snapshot.",
            ),
            "trading_suspension_date": FieldCertificationStatus(
                field_name="trading_suspension_date",
                is_certified=False,
                source_authorities=[],
                limitation_reason="Real-time halt/suspension feed is not integrated into directory feeds.",
            ),
        }

    @classmethod
    def validate_candidates(
        cls,
        candidates: list[MatchedUniverseCandidate],
        requirements: ValidationRequirements,
        temporal_context: TemporalEvaluationContext | None = None,
        caller_security_id_mapping: dict[str, str] | None = None,
        matching_accounting: SourceMatchingAccounting | None = None,
        allow_ambiguous_candidates: bool = False,
        allow_incomplete_source_files: bool = False,
    ) -> AdapterValidationReport:
        """
        Builds raw input records and validates the complete batch through UniverseInputValidator.
        Returns full diagnostic report containing batch outcome, field certification matrix,
        and end-to-end batch admission protection.
        """
        field_certs = cls.assess_field_certifications(candidates)

        # 1. Guard against empty candidate set: an empty batch cannot produce a certified universe
        if not candidates:
            return AdapterValidationReport(
                total_candidates=0,
                validation_passed_count=0,
                validation_failed_count=0,
                is_batch_valid=False,
                field_certifications=field_certs,
                raw_records=[],
                batch_validation_result=None,
                matching_accounting=matching_accounting,
                batch_rejection_reason="EMPTY_CANDIDATE_SET: Empty input candidate batch cannot produce a certified universe.",
            )

        # 2. Check for caller-specified coverage expectations and source parsing failures
        if matching_accounting and matching_accounting.required_sources_present is False:
            batch_rejection = "REQUIRED_SOURCES_MISSING: Caller-required input source feeds were not all supplied."
        elif matching_accounting and matching_accounting.required_coverage_established is False:
            batch_rejection = "REQUIRED_COVERAGE_NOT_MET: Supplied candidate count did not satisfy caller-required coverage threshold."
        elif matching_accounting and matching_accounting.malformed_source_row_count > 0 and not allow_incomplete_source_files:
            batch_rejection = (
                f"SOURCE_PARSE_FAILED: {matching_accounting.malformed_source_row_count} malformed rows detected across input sources."
            )
        elif any(c.is_ambiguous for c in candidates) and not allow_ambiguous_candidates:
            ambig_count = sum(1 for c in candidates if c.is_ambiguous)
            batch_rejection = (
                f"UNRESOLVED_SOURCE_AMBIGUITY: Batch contains {ambig_count} ambiguous candidate(s) with unresolvable collisions."
            )
        else:
            batch_rejection = None

        raw_records: list[dict[str, Any]] = []
        evidence_map: dict[str, list[EvidenceReference]] = {}

        for cand in candidates:
            rec, ev_list = cls.build_candidate_raw_record(
                cand,
                caller_security_id_mapping=caller_security_id_mapping,
            )
            raw_records.append(rec)
            # Map evidence by symbol or security_id
            key = rec.get("security_id") or rec.get("symbol")
            if key:
                evidence_map[key] = ev_list

        batch_res = UniverseInputValidator.validate_batch(
            raw_records=raw_records,
            requirements=requirements,
            temporal_context=temporal_context,
            evidence_map=evidence_map,
        )

        is_valid = batch_res.is_batch_valid and (batch_rejection is None)
        if not batch_res.is_batch_valid and not batch_rejection:
            batch_rejection = f"VALIDATION_REQUIREMENTS_VIOLATED: {batch_res.failed_count} candidate(s) failed validation requirements."

        return AdapterValidationReport(
            total_candidates=len(candidates),
            validation_passed_count=batch_res.passed_count,
            validation_failed_count=batch_res.failed_count,
            is_batch_valid=is_valid,
            field_certifications=field_certs,
            raw_records=raw_records,
            batch_validation_result=batch_res,
            matching_accounting=matching_accounting,
            batch_rejection_reason=batch_rejection,
        )

