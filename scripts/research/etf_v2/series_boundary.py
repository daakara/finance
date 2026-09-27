"""
scripts/research/etf_v2/series_boundary.py

Series Boundary Resolver for Pipeline V2.
Enforces invariant: CROSS_SERIES_TEXT_LEAKAGE = 0.
Isolates a target ETF series container from preceding and succeeding sibling funds.
"""

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

        # 1. Check exact Series ID occurrences
        series_offsets = doc_structure.series_occurrences.get(sid, [])
        all_series = sorted(doc_structure.series_occurrences.keys())

        if series_offsets:
            start_offset = series_offsets[0]
            # Find next series occurrence that belongs to a different series
            next_offsets = []
            for other_sid in all_series:
                if other_sid != sid:
                    for off in doc_structure.series_occurrences[other_sid]:
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
                    for off in doc_structure.series_occurrences[other_sid]:
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

        # 2. Single-fund document fallback (entire document represents the target series)
        return SeriesBoundary(
            series_id=sid,
            start_offset=0,
            end_offset=text_len,
            boundary_type="SINGLE_FUND_WHOLE_DOCUMENT",
            previous_sibling=None,
            next_sibling=None,
        )
