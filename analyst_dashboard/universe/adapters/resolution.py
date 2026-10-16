"""
analyst_dashboard/universe/adapters/resolution.py

Sprint 2 Package 2 (S2-03): Security Identity & Required Evidence Resolution Engine.
Resolves and correlates multi-source listing directory observations (SEC EDGAR,
Nasdaq Listed, Other Listed, and offline OpenFIGI results) into evidence-backed,
validator-ready input candidate structures without inventing synthetic security
identities or manufacturing unevidenced attributes.

Core Invariants:
1. Strict Identity Non-Inference:
   - CIK is an issuer identifier, NEVER an instrument/security identifier.
   - Multiple share classes (e.g. GOOG/GOOGL, FOX/FOXA) sharing one CIK remain distinct candidates.
   - Synthetic surrogate keys (e.g. CIK_{cik}_{symbol}, MKT_{symbol}) are strictly PROHIBITED.
   - If production security_id scheme is unresolved by policy, security_id remains explicitly UNRESOLVED.
2. OpenFIGI Multi-Match Disambiguation:
   - Arbitrary first-match selection on MULTI_MATCH responses is strictly PROHIBITED.
   - Multi-match results remain explicitly AMBIGUOUS and uncertified.
3. Strict Evidence Absence Preservation:
   - Currency, country, primary listing, historical listing/delisting dates, and real-time
     suspension dates are absent in raw directory feeds. They are strictly left OMITTED unless
     explicitly provided with authoritative provenance in caller evidence overlays.
4. Venue Conflict Preservation:
   - Observed venue mismatches (e.g. SEC EDGAR vs Nasdaq Other Listed) retain original raw
     venue values from each source without being cleared through unsupported equivalence mappings.
5. Population Accounting & Foreign Issuer Integrity:
   - Foreign US-listed issuers (e.g. BABA, TSM, SE, SPOT, ASML) remain fully accounted for in
     the population candidate pool. Rule R02 common-stock vs ADR distinctions remain downstream
     strategy eligibility decisions, not upstream population exclusions.
6. Pure In-Memory Execution:
   - Zero filesystem writes, zero database queries, zero network calls, zero scanner/cron activity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..validator import (
    BatchValidationResult,
    EvidenceReference,
    FieldValidationIssue,
    TemporalEvaluationContext,
    UniverseInputValidator,
    ValidationIssueCode,
    ValidationRequirements,
    ValidationStatus,
)
from .bridge import AdapterValidatorBridge
from .matching import match_sources_with_accounting
from .models import (
    FieldCertificationStatus,
    MatchedUniverseCandidate,
    OpenFigiMatchStatus,
    ParsedOpenFigiResult,
    ParsedSourceFile,
    ParsedSourceRow,
    SourceMatchingAccounting,
)


class IdentityResolutionStatus(str, Enum):
    """Classification status for instrument/security identity resolution."""
    IDENTIFIED = "IDENTIFIED"           # External caller-asserted security_id mapping (pass-through; not publisher-certified)
    UNRESOLVED = "UNRESOLVED"           # Production security_id uncertified; synthetic IDs avoided
    AMBIGUOUS = "AMBIGUOUS"             # Multi-match or unresolved collision prevents identity resolution


class ResolutionRecordStatus(str, Enum):
    """Evidence resolution outcome status for an individual candidate record."""
    RESOLVED = "RESOLVED"               # All required evidence fields present and verified
    UNRESOLVED = "UNRESOLVED"           # Missing one or more required evidence fields
    CONFLICTED = "CONFLICTED"           # Material source conflict (e.g. venue disagreement)
    AMBIGUOUS = "AMBIGUOUS"             # Unresolvable collision or OpenFIGI multi-match


@dataclass
class ResolvedSecurityRecord:
    """
    Evidence-reconciled candidate representation ready for downstream validation.
    Maintains full provenance, raw source values, and explicit resolution diagnostics.
    """
    symbol: str
    join_key: str
    security_id: str | None
    issuer_cik: str | None
    company_name: str | None
    exchange: str | None
    sec_exchange: str | None
    directory_exchange: str | None
    security_type: str | None
    is_etf: bool | None
    is_multi_share_class: bool  # Qualified alias: shared CIK establishes multi-symbol issuer; share-class requires instrument evidence
    share_class_symbols: list[str]
    openfigi_match_status: OpenFigiMatchStatus | None
    figi_identifiers: dict[str, str | None]
    foreign_issuer_indicator: bool | None
    identity_status: IdentityResolutionStatus
    record_status: ResolutionRecordStatus
    resolution_issues: list[str]
    raw_record: dict[str, Any]
    evidence_references: list[EvidenceReference]
    is_multi_symbol_issuer: bool = False
    multi_symbol_issuer_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "join_key": self.join_key,
            "security_id": self.security_id,
            "issuer_cik": self.issuer_cik,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "sec_exchange": self.sec_exchange,
            "directory_exchange": self.directory_exchange,
            "security_type": self.security_type,
            "is_etf": self.is_etf,
            "is_multi_share_class": self.is_multi_share_class,
            "share_class_symbols": self.share_class_symbols,
            "is_multi_symbol_issuer": self.is_multi_symbol_issuer,
            "multi_symbol_issuer_symbols": self.multi_symbol_issuer_symbols,
            "openfigi_match_status": self.openfigi_match_status.value if self.openfigi_match_status else None,
            "figi_identifiers": self.figi_identifiers,
            "foreign_issuer_indicator": self.foreign_issuer_indicator,
            "identity_status": self.identity_status.value,
            "record_status": self.record_status.value,
            "resolution_issues": self.resolution_issues,
            "raw_record": self.raw_record,
            "evidence_reference_count": len(self.evidence_references),
        }


@dataclass
class SecurityEvidenceAccounting:
    """
    Comprehensive population accounting across all resolution dimensions.
    Enforces mathematical reconciliation across resolved, unresolved, conflicted,
    and ambiguous outcomes.
    """
    total_candidates: int
    fully_evidenced_count: int
    identity_resolved_count: int
    identity_unresolved_count: int
    synthetic_identities_avoided_count: int
    multi_share_class_cik_count: int
    multi_share_class_symbol_count: int
    venue_concordant_count: int
    venue_conflicted_count: int
    openfigi_single_match_count: int
    openfigi_multi_match_count: int
    openfigi_no_match_count: int
    foreign_issuer_count: int
    missing_evidence_field_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)
    multi_symbol_issuer_cik_count: int = 0
    multi_symbol_issuer_symbol_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityEvidenceResolutionOutcome:
    """
    Complete outcome container returned by SecurityEvidenceResolver.
    Holds partitioned candidate lists, population accounting, and field certifications.
    """
    total_candidates: int
    records: list[ResolvedSecurityRecord]
    resolved_records: list[ResolvedSecurityRecord]
    unresolved_records: list[ResolvedSecurityRecord]
    conflicted_records: list[ResolvedSecurityRecord]
    ambiguous_records: list[ResolvedSecurityRecord]
    accounting: SecurityEvidenceAccounting
    field_certifications: dict[str, FieldCertificationStatus]
    matching_accounting: SourceMatchingAccounting | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "resolved_count": len(self.resolved_records),
            "unresolved_count": len(self.unresolved_records),
            "conflicted_count": len(self.conflicted_records),
            "ambiguous_count": len(self.ambiguous_records),
            "accounting": self.accounting.to_dict(),
            "field_certifications": {k: v.to_dict() for k, v in self.field_certifications.items()},
            "matching_accounting": self.matching_accounting.to_dict() if self.matching_accounting else None,
        }

    def to_raw_records_and_evidence(self) -> tuple[list[dict[str, Any]], dict[str, list[EvidenceReference]]]:
        """
        Extracts raw record dictionaries and keyed evidence lists for UniverseInputValidator.
        """
        raw_list: list[dict[str, Any]] = []
        ev_map: dict[str, list[EvidenceReference]] = {}
        for r in self.records:
            raw_list.append(r.raw_record)
            k = r.raw_record.get("security_id") or r.raw_record.get("symbol")
            if k:
                ev_map[k] = r.evidence_references
        return raw_list, ev_map

    def validate_batch(
        self,
        requirements: ValidationRequirements,
        temporal_context: TemporalEvaluationContext | None = None,
        allow_incomplete_source_files: bool = False,
    ) -> BatchValidationResult:
        """
        Directly executes UniverseInputValidator.validate_batch over all resolved candidate records.
        Propagates whole-batch admission requirements and source parsing constraints.
        """
        # Guard against empty candidate set
        if not self.records:
            return UniverseInputValidator.validate_batch(
                raw_records=[],
                requirements=requirements,
                temporal_context=temporal_context,
                evidence_map={},
            )

        raw_records, evidence_map = self.to_raw_records_and_evidence()
        batch_res = UniverseInputValidator.validate_batch(
            raw_records=raw_records,
            requirements=requirements,
            temporal_context=temporal_context,
            evidence_map=evidence_map,
        )

        # Propagate source matching accounting constraints if present
        if self.matching_accounting:
            extra_issues: list[FieldValidationIssue] = []
            if self.matching_accounting.required_sources_present is False:
                extra_issues.append(
                    FieldValidationIssue(
                        field_name="__batch__",
                        issue_code=ValidationIssueCode.MISSING_REQUIRED_FIELD,
                        message="REQUIRED_SOURCES_MISSING: Caller-required input source feeds were not all supplied.",
                    )
                )
            if self.matching_accounting.required_coverage_established is False:
                extra_issues.append(
                    FieldValidationIssue(
                        field_name="__batch__",
                        issue_code=ValidationIssueCode.BATCH_ADMISSION_BLOCKED,
                        message="REQUIRED_COVERAGE_NOT_MET: Supplied candidate count did not satisfy caller-required coverage threshold.",
                    )
                )
            if self.matching_accounting.malformed_source_row_count > 0 and not allow_incomplete_source_files:
                extra_issues.append(
                    FieldValidationIssue(
                        field_name="__batch__",
                        issue_code=ValidationIssueCode.MALFORMED_VALUE,
                        message=f"SOURCE_PARSE_FAILED: {self.matching_accounting.malformed_source_row_count} malformed rows detected across input sources.",
                    )
                )

            if extra_issues:
                # Demote batch fail-closed under atomic batch contract
                all_issues = list(batch_res.batch_issues) + extra_issues
                for r in batch_res.records:
                    if r.status == ValidationStatus.VALIDATION_PASSED:
                        r.status = ValidationStatus.VALIDATION_FAILED
                        r.downstream_admission_permitted = False
                        r.validated_values = None
                        r.issues.extend(extra_issues)

                return BatchValidationResult(
                    total_input_count=batch_res.total_input_count,
                    passed_count=0,
                    failed_count=batch_res.total_input_count,
                    unresolved_count=batch_res.unresolved_count,
                    records=batch_res.records,
                    batch_issues=all_issues,
                    is_batch_valid=False,
                )

        return batch_res


class SecurityEvidenceResolver:
    """
    Deterministic offline evidence resolution engine for ARX Terminal Radar universe inputs.
    Correlates SEC EDGAR, Nasdaq Trader, Other Listed directories, and OpenFIGI mapping
    observations into structured, evidence-grounded candidate records.
    """

    # Substrings indicative of foreign depositary receipts or non-domestic entities
    _FOREIGN_NAME_INDICATORS = (
        " ADR",
        " ADS",
        " AMERICAN DEPOSITARY",
        " GDR",
        " SPONSORED ADR",
        " NY REGISTRY SHS",
    )

    @classmethod
    def resolve_candidates(
        cls,
        candidates: list[MatchedUniverseCandidate],
        caller_security_id_mapping: dict[str, str] | None = None,
        caller_field_evidence_overlay: dict[str, dict[str, Any]] | None = None,
        caller_evidence_references: dict[str, list[EvidenceReference]] | None = None,
        require_venue_match: bool = False,
        matching_accounting: SourceMatchingAccounting | None = None,
    ) -> SecurityEvidenceResolutionOutcome:
        """
        Resolves a list of MatchedUniverseCandidate instances into ResolvedSecurityRecord objects.

        Enforces:
        - Multi-share class CIK detection and symbol separation
        - Strict avoidance of synthetic surrogate keys (CIK_{cik}_{symbol})
        - OpenFIGI multi-match disambiguation protection (reject first-match selection)
        - Raw venue disagreement preservation
        - Foreign issuer retention without population exclusion
        - Full missing-field evidence accounting against validator requirements
        """
        field_certs = AdapterValidatorBridge.assess_field_certifications(candidates)

        # 1. Multi-share class CIK index across all candidate observations
        cik_to_symbols: dict[str, list[str]] = {}
        for c in candidates:
            if c.sec_row and not c.sec_row.is_malformed:
                cik = c.sec_row.raw_values.get("issuer_cik")
                if cik:
                    cik_to_symbols.setdefault(cik, []).append(c.join_key)

        records: list[ResolvedSecurityRecord] = []
        resolved_records: list[ResolvedSecurityRecord] = []
        unresolved_records: list[ResolvedSecurityRecord] = []
        conflicted_records: list[ResolvedSecurityRecord] = []
        ambiguous_records: list[ResolvedSecurityRecord] = []

        # Accounting counters
        identity_resolved_cnt = 0
        identity_unresolved_cnt = 0
        synthetic_avoided_cnt = 0
        venue_concordant_cnt = 0
        venue_conflicted_cnt = 0
        openfigi_single_cnt = 0
        openfigi_multi_cnt = 0
        openfigi_none_cnt = 0
        foreign_issuer_cnt = 0
        missing_counts: dict[str, int] = {
            "security_id": 0,
            "security_type": 0,
            "currency": 0,
            "country": 0,
            "primary_listing": 0,
            "asset_class": 0,
            "listing_status": 0,
        }

        for c in candidates:
            sym = c.join_key
            issues: list[str] = list(c.join_conflicts)
            evidences: list[EvidenceReference] = []

            # ── A. Issuer CIK & Multi-share class evaluation ──────────────────
            issuer_cik: str | None = None
            company_name: str | None = None
            sec_exchange: str | None = None

            if c.sec_row and not c.sec_row.is_malformed:
                sec_vals = c.sec_row.raw_values
                issuer_cik = sec_vals.get("issuer_cik")
                company_name = sec_vals.get("company_name")
                sec_exchange = sec_vals.get("sec_exchange")

                sec_id = c.sec_source_identifier or "company_tickers_exchange.json"
                evidences.append(
                    EvidenceReference(
                        source_authority="SEC_EDGAR",
                        source_identifier=f"{sec_id}#row={c.sec_row.row_index}",
                        source_hash=c.sec_content_hash,
                        is_external_attestation=True,
                    )
                )

            # Evaluate multi-symbol issuer structure (shared CIK)
            # NOTE (Check C.A): Shared CIK establishes multi-symbol issuer relationship;
            # distinct share classes require explicit instrument-level evidence.
            is_multi_sym = False
            symbols_under_cik: list[str] = [sym]
            if issuer_cik and issuer_cik in cik_to_symbols:
                all_syms = sorted(list(set(cik_to_symbols[issuer_cik])))
                if len(all_syms) > 1:
                    is_multi_sym = True
                    symbols_under_cik = all_syms
                    issues.append(
                        f"MULTI_SYMBOL_ISSUER_CIK: Issuer CIK {issuer_cik} has {len(all_syms)} "
                        f"distinct listed securities: {all_syms}"
                    )

            is_multi_share = is_multi_sym  # Kept as qualified alias for backwards compatibility
            share_classes = symbols_under_cik

            # ── B. Directory observations ─────────────────────────────────────
            dir_exchange: str | None = None
            is_etf: bool | None = None

            if c.directory_row and not c.directory_row.is_malformed:
                dir_vals = c.directory_row.raw_values
                dir_exchange = dir_vals.get("venue_exchange")
                is_etf = dir_vals.get("is_etf")
                if "security_name" in dir_vals and not company_name:
                    company_name = dir_vals["security_name"]

                dir_src = c.directory_source_identifier or dir_vals.get("source_directory", "nasdaq_directory")
                evidences.append(
                    EvidenceReference(
                        source_authority="NASDAQ_TRADER",
                        source_identifier=f"{dir_src}#row={c.directory_row.row_index}",
                        source_hash=c.directory_content_hash,
                        is_external_attestation=True,
                    )
                )

            # ── C. Venue Concordance & Conflict Analysis ──────────────────────
            # Preserve raw exchange attributes; check concordance without speculative equivalence
            exchange: str | None = dir_exchange or sec_exchange
            is_venue_conflicted = any("VENUE_MISMATCH" in conf for conf in c.join_conflicts)
            if sec_exchange and dir_exchange:
                # Compare raw representations
                if is_venue_conflicted:
                    venue_conflicted_cnt += 1
                else:
                    venue_concordant_cnt += 1
            elif dir_exchange or sec_exchange:
                venue_concordant_cnt += 1

            # ── D. OpenFIGI Matching & Disambiguation ─────────────────────────
            figi_ids: dict[str, str | None] = {
                "listing_figi": None,
                "composite_figi": None,
                "share_class_figi": None,
            }
            security_type: str | None = None
            openfigi_status: OpenFigiMatchStatus | None = None

            if c.figi_result:
                openfigi_status = c.figi_result.match_status
                if openfigi_status == OpenFigiMatchStatus.SINGLE_MATCH:
                    openfigi_single_cnt += 1
                    # Single unambiguous match
                    if c.figi_result.matches and not c.is_ambiguous:
                        m = c.figi_result.matches[0]
                        figi_ids["listing_figi"] = m.get("listing_figi") or m.get("figi")
                        figi_ids["composite_figi"] = m.get("composite_figi")
                        figi_ids["share_class_figi"] = m.get("share_class_figi")
                        security_type = m.get("security_type2") or m.get("security_type")

                        figi_src = c.figi_source_identifier or "openfigi_mapping"
                        evidences.append(
                            EvidenceReference(
                                source_authority="OPENFIGI",
                                source_identifier=f"{figi_src}#request={c.figi_result.request_index}",
                                source_hash=c.figi_content_hash,
                                is_external_attestation=True,
                            )
                        )
                elif openfigi_status == OpenFigiMatchStatus.MULTI_MATCH:
                    openfigi_multi_cnt += 1
                    # INVARIANT: First-match selection is strictly rejected
                    issues.append(
                        f"OPENFIGI_MULTI_MATCH_UNRESOLVED: Received {len(c.figi_result.matches)} matches; "
                        f"arbitrary first-match selection is prohibited."
                    )
                elif openfigi_status == OpenFigiMatchStatus.NO_MATCH:
                    openfigi_none_cnt += 1
                    issues.append("OPENFIGI_NO_MATCH: No mapping returned for instrument.")

            # ── E. Foreign Issuer Retention & Classification ──────────────────
            # NOTE (Check C.B): Domicile cannot be inferred from US exchange listing or
            # positive pricing. Without explicit country evidence, domicile status is UNKNOWN (None).
            # If explicit ADR / Depositary Receipt structure is observed, instrument structure is foreign ADR.
            is_foreign: bool | None = None
            name_check = (company_name or "").upper()
            if any(ind in name_check for ind in cls._FOREIGN_NAME_INDICATORS):
                is_foreign = True
            elif security_type and any(term in security_type.upper() for term in ("DEPOSITARY", "ADR", "GDR")):
                is_foreign = True
            # Check caller overlay for explicit country evidence if provided
            if caller_field_evidence_overlay and sym in caller_field_evidence_overlay:
                overlay_country = caller_field_evidence_overlay[sym].get("country")
                if overlay_country:
                    is_foreign = (str(overlay_country).strip().upper() != "USA")

            if is_foreign is True:
                foreign_issuer_cnt += 1

            # ── F. Identity Resolution Boundary ──────────────────────────────
            # Check if caller provided explicit authoritative security_id mapping
            security_id: str | None = None
            if caller_security_id_mapping and sym in caller_security_id_mapping:
                security_id = caller_security_id_mapping[sym]
                identity_status = IdentityResolutionStatus.IDENTIFIED
                identity_resolved_cnt += 1
            else:
                # INVARIANT: Do NOT fabricate synthetic keys like CIK_{cik}_{symbol} or MKT_{symbol}
                security_id = None
                identity_status = IdentityResolutionStatus.UNRESOLVED
                identity_unresolved_cnt += 1
                synthetic_avoided_cnt += 1
                issues.append(
                    "UNRESOLVED_IDENTITY: Production instrument identity scheme remains unresolved by policy; "
                    "synthetic surrogate keys (CIK+symbol) are prohibited."
                )

            # ── G. Build Raw Input Record for Validator ───────────────────────
            raw_record: dict[str, Any] = {"symbol": sym}
            if company_name:
                raw_record["company_name"] = company_name
            if exchange:
                raw_record["exchange"] = exchange
            if security_type:
                raw_record["security_type"] = security_type
            if is_etf is not None:
                raw_record["is_etf"] = is_etf
            if issuer_cik:
                raw_record["issuer_cik"] = issuer_cik
            if security_id:
                raw_record["security_id"] = security_id

            # Add preserved FIGI levels without automatically designating security_id
            for figi_k, figi_val in figi_ids.items():
                if figi_val:
                    raw_record[figi_k] = figi_val

            # Preserve unadjudicated source conflicts or ambiguity in raw_record (Check A & D)
            # An overlay cannot silently clear source ambiguity or venue conflicts
            if is_venue_conflicted or c.is_ambiguous or (openfigi_status == OpenFigiMatchStatus.MULTI_MATCH):
                raw_record["__source_conflict__"] = issues

            # Incorporate caller-supplied field evidence overlays if present
            if caller_field_evidence_overlay and sym in caller_field_evidence_overlay:
                overlay = caller_field_evidence_overlay[sym]
                for k, v in overlay.items():
                    raw_record[k] = v

            # Incorporate caller-supplied explicit evidence references if present
            if caller_evidence_references and sym in caller_evidence_references:
                evidences.extend(caller_evidence_references[sym])

            # Raw diagnostic provenance tracking
            raw_record["raw_payload"] = {
                "join_key": sym,
                "issuer_cik": issuer_cik,
                "sec_exchange": sec_exchange,
                "directory_exchange": dir_exchange,
                "is_multi_share_class": is_multi_share,
                "is_multi_symbol_issuer": is_multi_sym,
                "foreign_issuer_indicator": is_foreign,
                "openfigi_status": openfigi_status.value if openfigi_status else None,
                "conflicts": issues,
            }

            # ── H. Missing Evidence Accounting & Record Status Determination ──
            # Evaluate against mandatory validator requirements
            record_missing: list[str] = []
            for req_field in (
                "security_id",
                "symbol",
                "exchange",
                "security_type",
                "listing_status",
                "currency",
                "country",
                "primary_listing",
                "asset_class",
            ):
                if req_field not in raw_record or raw_record[req_field] is None:
                    record_missing.append(req_field)
                    missing_counts[req_field] = missing_counts.get(req_field, 0) + 1

            if record_missing:
                issues.append(f"MISSING_REQUIRED_EVIDENCE: Fields omitted from raw feeds: {sorted(record_missing)}")

            # Determine Record Status
            if is_venue_conflicted:
                rec_status = ResolutionRecordStatus.CONFLICTED
                conflicted_records.append(None)
            elif c.is_ambiguous or (openfigi_status == OpenFigiMatchStatus.MULTI_MATCH):
                rec_status = ResolutionRecordStatus.AMBIGUOUS
                ambiguous_records.append(None)
            elif record_missing:
                rec_status = ResolutionRecordStatus.UNRESOLVED
                unresolved_records.append(None)
            else:
                rec_status = ResolutionRecordStatus.RESOLVED
                resolved_records.append(None)

            resolved_rec = ResolvedSecurityRecord(
                symbol=sym,
                join_key=sym,
                security_id=security_id,
                issuer_cik=issuer_cik,
                company_name=company_name,
                exchange=exchange,
                sec_exchange=sec_exchange,
                directory_exchange=dir_exchange,
                security_type=security_type,
                is_etf=is_etf,
                is_multi_share_class=is_multi_share,
                share_class_symbols=share_classes,
                is_multi_symbol_issuer=is_multi_sym,
                multi_symbol_issuer_symbols=symbols_under_cik,
                openfigi_match_status=openfigi_status,
                figi_identifiers=figi_ids,
                foreign_issuer_indicator=is_foreign,
                identity_status=identity_status,
                record_status=rec_status,
                resolution_issues=issues,
                raw_record=raw_record,
                evidence_references=evidences,
            )

            records.append(resolved_rec)
            # Replace placeholder in partitioned list
            if rec_status == ResolutionRecordStatus.AMBIGUOUS:
                ambiguous_records[-1] = resolved_rec
            elif rec_status == ResolutionRecordStatus.CONFLICTED:
                conflicted_records[-1] = resolved_rec
            elif rec_status == ResolutionRecordStatus.UNRESOLVED:
                unresolved_records[-1] = resolved_rec
            else:
                resolved_records[-1] = resolved_rec

        # 2. Multi-symbol issuer (shared CIK) accounting
        multi_sym_ciks = [cik for cik, syms in cik_to_symbols.items() if len(syms) > 1]
        multi_sym_symbols = sum(len(syms) for cik, syms in cik_to_symbols.items() if len(syms) > 1)

        accounting = SecurityEvidenceAccounting(
            total_candidates=len(records),
            fully_evidenced_count=len(resolved_records),
            identity_resolved_count=identity_resolved_cnt,
            identity_unresolved_count=identity_unresolved_cnt,
            synthetic_identities_avoided_count=synthetic_avoided_cnt,
            multi_share_class_cik_count=len(multi_sym_ciks),
            multi_share_class_symbol_count=multi_sym_symbols,
            multi_symbol_issuer_cik_count=len(multi_sym_ciks),
            multi_symbol_issuer_symbol_count=multi_sym_symbols,
            venue_concordant_count=venue_concordant_cnt,
            venue_conflicted_count=venue_conflicted_cnt,
            openfigi_single_match_count=openfigi_single_cnt,
            openfigi_multi_match_count=openfigi_multi_cnt,
            openfigi_no_match_count=openfigi_none_cnt,
            foreign_issuer_count=foreign_issuer_cnt,
            missing_evidence_field_counts=missing_counts,
            notes=[
                "Evidence resolution accounts strictly for supplied observations across parsed input files.",
                "Production instrument identity scheme remains unresolved by policy; zero synthetic IDs generated.",
                "Directory feeds omit currency, country, primary_listing, and trading_suspension_date.",
                "Downstream admission through UniverseInputValidator strictly requires explicit evidence.",
            ],
        )

        return SecurityEvidenceResolutionOutcome(
            total_candidates=len(records),
            records=records,
            resolved_records=resolved_records,
            unresolved_records=unresolved_records,
            conflicted_records=conflicted_records,
            ambiguous_records=ambiguous_records,
            accounting=accounting,
            field_certifications=field_certs,
            matching_accounting=matching_accounting,
        )

    @classmethod
    def resolve_from_source_files(
        cls,
        sec_file: ParsedSourceFile | None = None,
        directory_files: list[ParsedSourceFile] | None = None,
        openfigi_results: list[ParsedOpenFigiResult] | None = None,
        caller_security_id_mapping: dict[str, str] | None = None,
        caller_field_evidence_overlay: dict[str, dict[str, Any]] | None = None,
        caller_evidence_references: dict[str, list[EvidenceReference]] | None = None,
        require_venue_match: bool = False,
        required_sources: list[str] | None = None,
        expected_minimum_coverage: int | None = None,
    ) -> SecurityEvidenceResolutionOutcome:
        """
        High-level orchestrator that joins parsed source files and resolves candidate evidence.
        Pure in-memory execution.
        """
        candidates, matching_accounting = match_sources_with_accounting(
            sec_file=sec_file,
            directory_files=directory_files,
            openfigi_results=openfigi_results,
            require_venue_match=require_venue_match,
            required_sources=required_sources,
            expected_minimum_coverage=expected_minimum_coverage,
        )

        return cls.resolve_candidates(
            candidates=candidates,
            caller_security_id_mapping=caller_security_id_mapping,
            caller_field_evidence_overlay=caller_field_evidence_overlay,
            caller_evidence_references=caller_evidence_references,
            require_venue_match=require_venue_match,
            matching_accounting=matching_accounting,
        )
