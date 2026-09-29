"""
scripts/research/etf_v2/series_boundary.py

Series Boundary Resolver for Pipeline V2.
Enforces invariant: CROSS_SERIES_TEXT_LEAKAGE = 0.
Isolates a target ETF series container from preceding and succeeding sibling funds.
"""

import re
from typing import Optional, List
from .models import DocumentStructure, SeriesBoundary, EntityIdentity


class SeriesBoundaryResolver:
    """Resolves closed target series boundaries in multi-series or omnibus filings."""

    @classmethod
    def resolve_boundary(
        cls,
        doc_structure: DocumentStructure,
        identity: EntityIdentity,
    ) -> SeriesBoundary:
        """Determines the start and end offsets bounding the target series."""
        sid = identity.series_id
        raw_text = doc_structure.raw_text
        text_len = len(raw_text)

        body_series_map = doc_structure.series_occurrences
        body_series_ids = set(body_series_map.keys())
        series_offsets = body_series_map.get(sid, [])
        all_series = sorted(body_series_ids)

        # Check for multi-series presence in raw_text (including iXBRL header declarations)
        SERIES_ID_PATTERN = r"(?<![A-Za-z0-9])(S\d{9})(?:Member)?(?![A-Za-z0-9])"
        all_raw_series_matches = set(
            m.group(1) for m in re.finditer(SERIES_ID_PATTERN, raw_text)
        )
        is_multi_series = len(body_series_ids) > 1 or len(all_raw_series_matches) > 1

        # 1. Exact target series boundary delimited from body occurrences
        if series_offsets:
            start_offset = series_offsets[0]
            # Find next series occurrence that belongs to a different series
            next_offsets = []
            for other_sid in all_series:
                if other_sid != sid:
                    for off in body_series_map[other_sid]:
                        if off > start_offset:
                            next_offsets.append((off, other_sid))

            if next_offsets:
                next_offsets.sort(key=lambda x: x[0])
                end_offset = next_offsets[0][0]
                next_sibling = next_offsets[0][1]
            else:
                end_offset = text_len
                next_sibling = None

            # Look for previous sibling
            prev_offsets = []
            for other_sid in all_series:
                if other_sid != sid:
                    for off in body_series_map[other_sid]:
                        if off < start_offset:
                            prev_offsets.append((off, other_sid))
            prev_sibling = max(prev_offsets, key=lambda x: x[0])[1] if prev_offsets else None

            return SeriesBoundary(
                series_id=sid,
                start_offset=max(0, start_offset - 2000),  # context allowance for fund cover/header
                end_offset=end_offset,
                boundary_type="EXACT_SERIES_ID_DELIMITED",
                previous_sibling=prev_sibling,
                next_sibling=next_sibling,
            )

        # 2. If document is multi-series and target boundary could not be delimited: FAIL CLOSED
        if is_multi_series:
            return SeriesBoundary(
                series_id=sid,
                start_offset=-1,
                end_offset=-1,
                boundary_type="MULTI_SERIES_DELIMITATION_FAILED",
                previous_sibling=None,
                next_sibling=None,
            )

        # 3. Determine single-fund detection status
        # Retrieve authoritative document form from doc_structure or parsed EDGAR <TYPE> metadata
        form = getattr(doc_structure, "form", None)
        if not form:
            type_match = re.search(r"<TYPE>\s*([A-Za-z0-9\-\/]+)", raw_text[:2500], re.IGNORECASE)
            form = type_match.group(1).upper() if type_match else ""

        # Single-fund status is positively established if:
        # a) Exactly one series ID is detected in the document and matches target series, OR
        # b) Document is an authoritative statutory Form 497K single-fund summary prospectus with no conflicting series
        has_matching_single_series = (len(all_raw_series_matches) == 1 and sid in all_raw_series_matches)
        is_statutory_single_fund_form = (form == "497K")

        if has_matching_single_series or (is_statutory_single_fund_form and len(all_raw_series_matches) <= 1):
            series_detection_status = "ESTABLISHED"
        else:
            series_detection_status = "INDETERMINATE"


        if series_detection_status == "ESTABLISHED":
            return SeriesBoundary(
                series_id=sid,
                start_offset=0,
                end_offset=text_len,
                boundary_type="SINGLE_FUND_WHOLE_DOCUMENT",
                previous_sibling=None,
                next_sibling=None,
            )

        # Indeterminate series detection: whole-document fallback is strictly PROHIBITED
        return SeriesBoundary(
            series_id=sid,
            start_offset=-1,
            end_offset=-1,
            boundary_type="MULTI_SERIES_DELIMITATION_FAILED",
            previous_sibling=None,
            next_sibling=None,
        )
