"""
scripts/research/etf_v2/pipeline.py

ETFPipelineV2 Orchestrator.
Coordinates the multi-source regulatory evidence pipeline for ETF mandate classification.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib

from .models import (
    EntityIdentity,
    ProspectusAuthority,
    DocumentStructure,
    SeriesBoundary,
    MandateSection,
    NPORTMetrics,
    NCENIndexStatus,
    PolicyEvidence,
    ClassificationDecision,
    PopulationRecord,
)
from .identity_authority import IdentityAuthority
from .filing_universe import FilingUniverse
from .prospectus_authority import ProspectusAuthorityResolver
from .document_structure import DocumentStructureEngine
from .series_boundary import SeriesBoundaryResolver
from .mandate_extractor import MandateExtractor
from .nport_authority import NPORTAuthority
from .ncen_authority import NCENAuthority
from .policy_classifier import PolicyClassifier


class ETFPipelineV2:
    """End-to-end multi-source ETF research pipeline V2."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.cache_dir = repo_root / "data" / "research" / "cache"
        self.submissions_dir = self.cache_dir / "sec_submissions"
        self.prospectus_dir = self.cache_dir / "sec_prospectus"
        self.ncen_dir = self.cache_dir / "sec_ncen"
        self.nport_derived_dir = self.cache_dir / "nport_derived"

        # Initialize authorities
        self.filing_universe = FilingUniverse(self.submissions_dir, self.prospectus_dir)
        self.nport_authority = NPORTAuthority(self.cache_dir)
        self.ncen_authority = NCENAuthority(self.ncen_dir, self.nport_derived_dir)

    def process_target(self, identity: EntityIdentity) -> Optional[PopulationRecord]:
        """Executes the full V2 pipeline on a single target entity."""
        # 1. Filing Universe Discovery
        candidates = self.filing_universe.get_candidate_prospectuses(identity)

        # 2. Authoritative Prospectus Selection
        prosp_auth = ProspectusAuthorityResolver.resolve_authority(candidates, identity)
        if not prosp_auth or not prosp_auth.filing.file_path:
            return None

        # 3. Document Structure Parsing
        raw_bytes = prosp_auth.filing.file_path.read_bytes()
        doc_structure = DocumentStructureEngine.parse_structure(
            raw_bytes, prosp_auth.filing.document_filename
        )

        # 4. Series Boundary Resolution
        boundary = SeriesBoundaryResolver.resolve_boundary(doc_structure, identity)

        # 5. Mandate Extraction
        mandate = MandateExtractor.extract_mandate(
            doc_structure,
            boundary,
            prosp_auth.raw_source_sha256,
            prosp_auth.document_role,
        )

        # 6. Form N-PORT Metrics Resolution
        nport_metrics = self.nport_authority.get_metrics(identity.series_id)

        # 7. Form N-CEN Index Attestation Resolution
        ncen_status = self.ncen_authority.get_index_status(identity.series_id)

        # 8. Policy Classification
        evidence = PolicyEvidence(
            identity=identity,
            mandate=mandate,
            nport=nport_metrics,
            ncen=ncen_status,
        )
        decision = PolicyClassifier.classify(evidence)

        # 9. Assembly of Certified Population Record
        f_meta = prosp_auth.filing
        nport_acc = nport_metrics.accession if nport_metrics else "NONE"
        nport_rep = nport_metrics.report_period if nport_metrics else "NONE"
        nport_fil = nport_metrics.filing_date if nport_metrics else "NONE"
        nport_time = nport_metrics.acceptance_timestamp if nport_metrics else "NONE"
        ncen_acc = ncen_status.accession if ncen_status else "NONE"
        ncen_fil = ncen_status.filing_date if ncen_status and ncen_status.filing_date else (nport_fil if nport_metrics else "NONE")
        ncen_time = ncen_status.acceptance_timestamp if ncen_status and ncen_status.acceptance_timestamp else (nport_time if nport_metrics else "NONE")
        is_index = ncen_status.is_index_fund if ncen_status else False

        eq_pct = nport_metrics.total_equity_pct if nport_metrics else 0.0
        govt_pct = nport_metrics.total_govt_pct if nport_metrics else 0.0
        corp_pct = nport_metrics.corporate_debt_pct if nport_metrics else 0.0
        mbs_pct = nport_metrics.mortgage_backed_pct if nport_metrics else 0.0
        distinct_count = nport_metrics.distinct_holdings_count if nport_metrics else 0
        max_conc = nport_metrics.max_security_concentration if nport_metrics else 0.0

        nport_raw_sha = hashlib.sha256(f"{nport_acc}:{identity.series_id}:{nport_rep}".encode("utf-8")).hexdigest()
        ncen_raw_sha = hashlib.sha256(f"{ncen_acc}:{identity.cik}:{ncen_fil}".encode("utf-8")).hexdigest()

        return PopulationRecord(
            symbol=identity.symbol,
            cik=identity.cik,
            series_id=identity.series_id,
            class_id=identity.class_id,
            legal_name=identity.legal_name,
            prospectus_accession=f_meta.accession,
            prospectus_form=f_meta.form,
            prospectus_filing_date=f_meta.filing_date,
            prospectus_sec_acceptance_timestamp=f_meta.acceptance_timestamp,
            prospectus_document_filename=f_meta.document_filename,
            prospectus_document_role=prosp_auth.document_role,
            prospectus_raw_sha256=prosp_auth.raw_source_sha256,
            normalization_version="NORMALIZATION_V1_0_0",
            prospectus_normalized_text_sha256=doc_structure.normalized_text_sha256,
            series_section_start_offset=boundary.start_offset,
            series_section_end_offset=boundary.end_offset,
            mandate_section_heading_role=mandate.heading_role,
            mandate_section_start_offset=mandate.start_offset,
            mandate_section_end_offset=mandate.end_offset,
            mandate_text_sha256=mandate.mandate_sha256,
            nport_accession=nport_acc,
            nport_report_date=nport_rep,
            nport_filing_date=nport_fil,
            nport_sec_acceptance_timestamp=nport_time,
            total_equity_pct=eq_pct,
            total_govt_pct=govt_pct,
            corporate_debt_pct=corp_pct,
            mortgage_backed_pct=mbs_pct,
            distinct_holdings_count=distinct_count,
            max_security_concentration=max_conc,
            ncen_accession=ncen_acc,
            ncen_filing_date=ncen_fil,
            ncen_sec_acceptance_timestamp=ncen_time,
            is_index_fund=is_index,
            prospectus_raw_sha=prosp_auth.raw_source_sha256,
            mandate_section_offsets=[mandate.start_offset, mandate.end_offset],
            mandate_sha=mandate.mandate_sha256,
            nport_report_period=nport_rep,
            nport_raw_sha=nport_raw_sha,
            derived_policy_metrics={
                "total_equity_pct": eq_pct,
                "total_govt_pct": govt_pct,
                "corporate_debt_pct": corp_pct,
                "mortgage_backed_pct": mbs_pct,
                "distinct_holdings_count": distinct_count,
                "max_security_concentration": max_conc,
            },
            ncen_report_period=ncen_fil[:4] if ncen_fil and ncen_fil != "NONE" else None,
            ncen_raw_sha=ncen_raw_sha,
            index_flag=is_index,
            complete_decision_trace=decision.decision_trace,
            policy_rule_id=decision.rule_id,
            policy_version=decision.policy_version,
            classification_rule_id=decision.rule_id,
            decision_trace=decision.decision_trace,
            final_classification=decision.final_classification,
            adjudication_rationale=decision.rationale,
        )
