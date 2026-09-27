"""
scripts/research/etf_v2/mandate_extractor.py

Mandate Extractor for Pipeline V2.
Enforces statutory heading precedence:
Principal Investment Strategies (Item 4)
> Statutory Equivalent
> Investment Objective (Item 2) fallback ONLY when Item 4 is genuinely absent.

Item 2 is strictly prohibited from displacing an existing Item 4.
"""

import re
import hashlib
from typing import Optional
from .models import DocumentStructure, SeriesBoundary, MandateSection
from .document_structure import DocumentStructureEngine


class MandateExtractor:
    """Extracts target mandate section under statutory precedence invariants."""

    @classmethod
    def extract_mandate(
        cls,
        doc_structure: DocumentStructure,
        boundary: SeriesBoundary,
        raw_source_sha256: str,
        source_role: str,
    ) -> MandateSection:
        """Extracts mandate text respecting series boundaries and section precedence."""
        raw_text = doc_structure.raw_text
        target_block = raw_text[boundary.start_offset: boundary.end_offset]

        # 1. Search for Item 4 Principal Strategies within target block
        m_strat = re.search(
            r"(?:Principal\s+Investment\s+Strateg(?:ies|y)|Principal\s+Strategies)(.*?)(?:Principal\s+(?:Investment\s+)?Risks|Fee\s+Table|Annual\s+Fund\s+Operating|Portfolio\s+Management|<hr|\Z)",
            target_block,
            re.IGNORECASE | re.DOTALL,
        )

        if m_strat and len(m_strat.group(1).strip()) >= 100:
            strat_raw = m_strat.group(1)
            strat_norm = DocumentStructureEngine.normalize_text(strat_raw)
            strat_start = boundary.start_offset + m_strat.start()
            strat_end = boundary.start_offset + m_strat.end()
            mandate_sha = hashlib.sha256(strat_norm.encode("utf-8")).hexdigest()

            return MandateSection(
                text=strat_norm,
                heading_role="PRINCIPAL_STRATEGY",
                start_offset=strat_start,
                end_offset=strat_end,
                completeness_state="COMPLETE",
                source_role=source_role,
                raw_source_sha256=raw_source_sha256,
                normalized_text_sha256=doc_structure.normalized_text_sha256,
                mandate_sha256=mandate_sha,
            )

        # 2. Fallback to Item 2 Investment Objective ONLY if Item 4 is genuinely absent
        m_obj = re.search(
            r"(?:Investment\s+Objective|Investment\s+Goal)(.*?)(?:Fee\s+Table|Principal\s+Investment\s+Strateg|Principal\s+Risks|<hr|\Z)",
            target_block,
            re.IGNORECASE | re.DOTALL,
        )

        if m_obj and len(m_obj.group(1).strip()) >= 50:
            strat_raw = m_obj.group(1)
            strat_norm = DocumentStructureEngine.normalize_text(strat_raw)
            strat_start = boundary.start_offset + m_obj.start()
            strat_end = boundary.start_offset + m_obj.end()
            mandate_sha = hashlib.sha256(strat_norm.encode("utf-8")).hexdigest()

            return MandateSection(
                text=strat_norm,
                heading_role="INVESTMENT_OBJECTIVE",
                start_offset=strat_start,
                end_offset=strat_end,
                completeness_state="FALLBACK_OBJECTIVE",
                source_role=source_role,
                raw_source_sha256=raw_source_sha256,
                normalized_text_sha256=doc_structure.normalized_text_sha256,
                mandate_sha256=mandate_sha,
            )

        # 3. Unresolved / Empty mandate
        return MandateSection(
            text="",
            heading_role="NONE",
            start_offset=-1,
            end_offset=-1,
            completeness_state="NONE",
            source_role=source_role,
            raw_source_sha256=raw_source_sha256,
            normalized_text_sha256=doc_structure.normalized_text_sha256,
            mandate_sha256="NONE",
        )
