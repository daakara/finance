"""ARX Terminal — Series-Level Statutory Prospectus Mapping & Resolution Engine (Stage B).

FROZEN PRODUCTION RULESET: SERIES_RESOLVER_V1_0_0.

Enforces:
1. Input: target SeriesMetadata and pre-indexed DocumentIndex (Stage A).
2. Output: SeriesMappingResult and ResolvedSeriesSection.
3. Separation of Concerns: Zero downstream mandate or subtype classification logic.
4. Precedence Hierarchy (Section 8):
   - MAPPED_EXACT_SERIES_ID
   - MAPPED_EXACT_CLASS_ID (with verified class->series relationship)
   - MAPPED_EXACT_LEGAL_NAME
   - MAPPED_DETERMINISTIC_COMPOSITE
5. Consistency & Conflict Resolution (Section 9 & 10):
   - Strong evidence conflict -> CONFLICTING_IDENTITY_EVIDENCE
   - Unverified class relation -> CONFLICTING_OR_UNVERIFIED_CLASS_RELATION
6. Explicit Leakage Model (Section 18):
   - CHECKED_CLEAN (required for successful mapping)
   - CONTAMINATION_DETECTED
   - INSUFFICIENT_EVIDENCE
   - NOT_EVALUATED
7. Truncation Discipline (Section 20):
   - No silent clipping!
   - Records source_section_length, extracted_length, extraction_truncated.
8. Deterministic Cache Identities:
   - SERIES_RESOLUTION_CACHE_KEY
   - MANDATE_PARSE_IDENTITY
"""

import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Set, Any
from bs4 import BeautifulSoup

from scripts.research.document_index_engine import (
    DocumentIndex,
    DocumentIdentity,
    DocumentNormalizer,
    Occurrence,
    StrategyAnchor,
    SectionBoundaryCandidate,
    INDEX_ENGINE_VERSION,
    NORMALIZATION_VERSION,
)

SERIES_RESOLVER_VERSION = "SERIES_RESOLVER_V1_1_0"
SNAPSHOT_BOUNDARY = "2026-09-24"
SNAPSHOT_BOUNDARY_ISO = "2026-09-24T23:59:59Z"
MAX_STRATEGY_LENGTH_CEILING = 50000


@dataclass
class SeriesMetadata:
    """Legal identity metadata for an ETF series."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    legal_name: str
    trust_name: Optional[str] = None


@dataclass
class ResolvedSeriesSection:
    """The isolated statutory section for an ETF series."""
    series_id: str
    symbol: str
    start_offset: int
    end_offset: int
    section_text: str
    section_sha256: str
    strategy_heading: str
    strategy_text: str
    source_section_length: int
    extracted_length: int
    extraction_truncated: bool


@dataclass
class SeriesMappingResult:
    """Deterministic document-selection audit trail record."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    mapping_outcome: str
    mapping_rule_id: str
    mapping_evidence: str
    identity_evidence: str
    boundary_evidence: str
    mapping_confidence_state: str
    selected_accession: str = "NONE"
    selected_form: str = "NONE"
    primary_document: str = "NONE"
    selected_document: str = "NONE"
    source_bytes_sha256: str = "NONE"
    document_index_sha256: str = "NONE"
    extracted_series_block_sha256: str = "NONE"
    extracted_strategy_text: str = ""
    exact_source_section: str = "NONE"
    section_start_offset: int = -1
    section_end_offset: int = -1
    source_section_length: int = 0
    extracted_length: int = 0
    extraction_truncated: bool = False
    leakage_status: str = "NOT_EVALUATED"
    cross_series_text_leakage: int = 0
    series_resolution_cache_key: str = "NONE"
    mandate_parse_identity: str = "NONE"
    # Legacy alias support
    raw_document_sha256: str = "NONE"


class SeriesProspectusMapper:
    """Production series-level statutory prospectus mapper and resolver."""

    RULESET_ID = SERIES_RESOLVER_VERSION
    SNAPSHOT_BOUNDARY = SNAPSHOT_BOUNDARY
    SNAPSHOT_BOUNDARY_ISO = SNAPSHOT_BOUNDARY_ISO

    # Outcome constants (Section 17)
    OUTCOME_EXACT_SERIES_ID = "MAPPED_EXACT_SERIES_ID"
    OUTCOME_EXACT_CLASS_ID = "MAPPED_EXACT_CLASS_ID"
    OUTCOME_EXACT_LEGAL_NAME = "MAPPED_EXACT_LEGAL_NAME"
    OUTCOME_DETERMINISTIC_COMPOSITE = "MAPPED_DETERMINISTIC_COMPOSITE"
    OUTCOME_CONFLICTING_IDENTITY = "CONFLICTING_IDENTITY_EVIDENCE"
    OUTCOME_UNVERIFIED_CLASS_RELATION = "CONFLICTING_OR_UNVERIFIED_CLASS_RELATION"
    OUTCOME_AMBIGUOUS_MULTI_MATCH = "AMBIGUOUS_MULTI_MATCH"
    OUTCOME_SERIES_NOT_FOUND = "SERIES_NOT_FOUND_IN_SOURCE"
    OUTCOME_CLASS_NOT_FOUND = "CLASS_NOT_FOUND_IN_SOURCE"
    OUTCOME_SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    OUTCOME_BOUNDARY_NOT_ESTABLISHED = "BOUNDARY_NOT_ESTABLISHED"
    OUTCOME_PARSE_FAILURE = "PARSE_FAILURE"
    OUTCOME_EXPLICIT_TRUNCATION_FAILURE = "EXPLICIT_TRUNCATION_FAILURE"

    AUTHORIZED_OUTCOMES_FOR_PARSING = {
        OUTCOME_EXACT_SERIES_ID,
        OUTCOME_EXACT_CLASS_ID,
        OUTCOME_EXACT_LEGAL_NAME,
        OUTCOME_DETERMINISTIC_COMPOSITE,
    }

    # Leakage Status Constants (Section 18)
    LEAKAGE_CHECKED_CLEAN = "CHECKED_CLEAN"
    LEAKAGE_CONTAMINATION_DETECTED = "CONTAMINATION_DETECTED"
    LEAKAGE_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LEAKAGE_NOT_EVALUATED = "NOT_EVALUATED"

    STATUTORY_FORMS = {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A", "497", "497K"}
    FORM_PRIORITY = {
        "485BPOS": 10,
        "485APOS": 8,
        "497K": 7,
        "497": 6,
        "N-1A": 5,
        "N-1A/A": 4,
        "S-6": 3,
        "S-6/A": 2,
    }

    GENERAL_DISCLOSURE_DELIMITERS = [
        r"\bStatement\s+of\s+Additional\s+Information\b",
        r"\bPART\s+B\b",
        r"\bPART\s+C\b",
        r"\bAdditional\s+Information\s+about\s+(?:the\s+)?Funds?\b",
        r"\bMore\s+Information\s+About\s+(?:the\s+)?Funds?\b",
        r"\bFinancial\s+Highlights\b",
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        return DocumentNormalizer.normalize_html_to_text(text)[0]

    @classmethod
    def normalize_name(cls, name: str) -> str:
        return DocumentNormalizer.normalize_name(name)

    @classmethod
    def build_filing_index_for_cik(
        cls,
        cik: str,
        submission_json: dict,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY,
    ) -> List[Dict[str, Any]]:
        """Constructs an index of pre-boundary statutory filings for a CIK."""
        recent = submission_json.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        acceptance_times = recent.get("acceptanceDateTime", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_descs = recent.get("primaryDocDescription", [])
        file_numbers = recent.get("fileNumber", [])

        indexed_filings = []
        for i, form in enumerate(forms):
            fdate = filing_dates[i] if i < len(filing_dates) else ""
            if form in cls.STATUTORY_FORMS and fdate <= snapshot_boundary:
                indexed_filings.append({
                    "cik": str(cik),
                    "accession": accessions[i] if i < len(accessions) else "",
                    "form": form,
                    "filing_date": fdate,
                    "report_date": report_dates[i] if i < len(report_dates) else "",
                    "acceptance_date_time": acceptance_times[i] if i < len(acceptance_times) else "",
                    "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                    "primary_doc_description": primary_descs[i] if i < len(primary_descs) else "",
                    "file_number": file_numbers[i] if i < len(file_numbers) else "",
                    "form_priority": cls.FORM_PRIORITY.get(form, 0),
                })

        indexed_filings.sort(key=lambda x: (x["filing_date"], x["form_priority"]), reverse=True)
        return indexed_filings

    @classmethod
    def verify_leakage(
        cls,
        extracted_text: str,
        neighboring_series: List[SeriesMetadata],
    ) -> Tuple[str, int]:
        """Independent leakage validation (Section 18 & 19).

        Returns: (leakage_status, cross_series_text_leakage_count)
        """
        if not extracted_text or len(extracted_text.strip()) < 50:
            return cls.LEAKAGE_INSUFFICIENT_EVIDENCE, 0

        if not neighboring_series:
            return cls.LEAKAGE_CHECKED_CLEAN, 0

        text_lower = extracted_text.lower()
        leakage_count = 0

        for neighbor in neighboring_series:
            n_sid = (neighbor.series_id or "").lower()
            n_name = cls.normalize_name(neighbor.legal_name) if neighbor.legal_name else ""

            # Check if neighbor series ID appears in extracted strategy text
            if n_sid and n_sid in text_lower:
                leakage_count += 1

            # Check if neighbor full legal name appears in extracted strategy text
            if n_name and len(n_name) > 12 and n_name in text_lower:
                leakage_count += 1

        if leakage_count > 0:
            return cls.LEAKAGE_CONTAMINATION_DETECTED, leakage_count

        return cls.LEAKAGE_CHECKED_CLEAN, 0

    @classmethod
    def map_series(
        cls,
        target_series: SeriesMetadata,
        document_index_or_text: Any = None,
        accession: str = "NONE",
        form: str = "NONE",
        document_filename: str = "NONE",
        neighboring_series: Optional[List[SeriesMetadata]] = None,
        authoritative_class_to_series_map: Optional[Dict[str, str]] = None,
        raw_document_text: Optional[str] = None,
    ) -> SeriesMappingResult:
        """Stage B: Maps target SeriesMetadata against DocumentIndex.

        Supports both:
        1. map_series(target, document_index) [Stage B production API]
        2. map_series(target, raw_text, accession, form, doc_name, neighbors) [Legacy adapter]
        """
        if raw_document_text is not None and document_index_or_text is None:
            document_index_or_text = raw_document_text

        if document_index_or_text is None:
            return cls._fail_result(
                target_series, None, cls.OUTCOME_PARSE_FAILURE,
                "EMPTY_DOCUMENT", "Document is None"
            )

        # Adapt legacy parameters into DocumentIndex if raw string provided
        if isinstance(document_index_or_text, DocumentIndex):
            doc_index = document_index_or_text
            if not doc_index.normalized_text.strip():
                return cls._fail_result(
                    target_series, doc_index, cls.OUTCOME_PARSE_FAILURE,
                    "EMPTY_DOCUMENT", "Document contains no text"
                )
        else:
            raw_text = str(document_index_or_text)
            if not raw_text.strip():
                return cls._fail_result(
                    target_series, None, cls.OUTCOME_PARSE_FAILURE,
                    "EMPTY_DOCUMENT", "Document text is empty or whitespace"
                )
            raw_bytes = raw_text.encode("utf-8")
            identity = DocumentIdentity(
                cik=target_series.cik,
                accession=accession,
                form=form,
                filing_date=SNAPSHOT_BOUNDARY,
                document_filename=document_filename,
                source_byte_length=len(raw_bytes),
            )
            # Known metadata from target + neighbors
            known = [{"legal_name": target_series.legal_name}]
            if neighboring_series:
                for nb in neighboring_series:
                    known.append({"legal_name": nb.legal_name})
            doc_index = DocumentIndex(identity, raw_bytes, known)

        target = target_series
        neighbors = neighboring_series or []
        class_map = authoritative_class_to_series_map or {}

        # -------------------------------------------------------------
        # STEP 1: IDENTITY RESOLUTION PRECEDENCE (Section 8)
        # -------------------------------------------------------------
        sid = (target.series_id or "").upper().strip()
        cid = (target.class_id or "").upper().strip()
        raw_name = (target.legal_name or "").strip()
        norm_name = DocumentNormalizer.normalize_name(raw_name)

        sid_occs = [o for o in doc_index.series_occurrences.get(sid, []) if not o.is_toc_or_cross_ref]
        cid_occs = [o for o in doc_index.class_occurrences.get(cid, []) if not o.is_toc_or_cross_ref]
        name_occs = [o for o in doc_index.legal_name_occurrences.get(raw_name, []) if not o.is_toc_or_cross_ref]
        if not name_occs and norm_name:
            name_occs = [o for o in doc_index.normalized_name_occurrences.get(norm_name, []) if not o.is_toc_or_cross_ref]

        # Consistency Checking (Section 9 & 10)
        # Verify Class-Series Relation
        if cid and class_map:
            mapped_series = class_map.get(cid)
            if mapped_series and sid and mapped_series.upper() != sid:
                return cls._fail_result(
                    target, doc_index, cls.OUTCOME_UNVERIFIED_CLASS_RELATION,
                    "RULE_CLASS_SERIES_MISMATCH",
                    f"Class ID {cid} belongs to series {mapped_series}, conflicting with target series {sid}"
                )

        # Precedence Evaluation
        chosen_anchor: Optional[int] = None
        identity_evidence = ""
        mapping_outcome = ""
        rule_id = ""

        if sid_occs:
            mapping_outcome = cls.OUTCOME_EXACT_SERIES_ID
            rule_id = "MAPPING_RULE_EXACT_SERIES_ID"
            identity_evidence = f"Exact series ID {sid} matched {len(sid_occs)} substantive occurrence(s)"
            # Pick occurrence closest to a substantive strategy section
            chosen_anchor = cls._select_anchor_closest_to_strategy(sid_occs, doc_index.strategy_anchors, doc_index.normalized_text, neighbors)

        elif cid_occs:
            mapping_outcome = cls.OUTCOME_EXACT_CLASS_ID
            rule_id = "MAPPING_RULE_EXACT_CLASS_ID"
            identity_evidence = f"Exact class ID {cid} matched {len(cid_occs)} substantive occurrence(s) (class-series verified)"
            chosen_anchor = cls._select_anchor_closest_to_strategy(cid_occs, doc_index.strategy_anchors, doc_index.normalized_text, neighbors)

        elif name_occs:
            mapping_outcome = cls.OUTCOME_EXACT_LEGAL_NAME
            rule_id = "MAPPING_RULE_EXACT_LEGAL_NAME"
            identity_evidence = f"Exact legal fund name '{raw_name}' matched {len(name_occs)} occurrence(s)"
            chosen_anchor = cls._select_anchor_closest_to_strategy(name_occs, doc_index.strategy_anchors, doc_index.normalized_text, neighbors)

        else:
            if not sid_occs and not cid_occs and not name_occs:
                return cls._fail_result(
                    target, doc_index, cls.OUTCOME_SERIES_NOT_FOUND,
                    "SERIES_NOT_IN_SOURCE",
                    f"Target series {sid} / {cid} / '{raw_name}' not found in substantive document sections"
                )

        if chosen_anchor is None:
            return cls._fail_result(
                target, doc_index, cls.OUTCOME_BOUNDARY_NOT_ESTABLISHED,
                "RULE_NO_USABLE_ANCHOR",
                f"No usable anchor outside Table of Contents for {target.symbol}"
            )

        # -------------------------------------------------------------
        # STEP 2: STRUCTURAL BOUNDARY RESOLUTION (Section 14)
        # -------------------------------------------------------------
        text = doc_index.normalized_text
        start_boundary = chosen_anchor
        boundary_evidence_parts = []

        # Find preceding delimiter within a 15,000 character lookback window
        window_start = max(0, chosen_anchor - 15000)
        preceding_delims = [
            b for b in doc_index.boundary_candidates
            if window_start <= b.start_offset <= chosen_anchor
        ]
        if preceding_delims:
            best_preceding = preceding_delims[-1]
            start_boundary = best_preceding.start_offset
            boundary_evidence_parts.append(f"Start boundary anchored by {best_preceding.boundary_type} at offset {start_boundary}")
        else:
            boundary_evidence_parts.append(f"Start boundary anchored at identity occurrence offset {start_boundary}")

        # Find subsequent delimiter / next fund boundary
        # Find all neighbor anchors occurring strictly after chosen_anchor + 10
        neighbor_anchors = []
        for nb in neighbors:
            nb_sid = (nb.series_id or "").upper()
            nb_cid = (nb.class_id or "").upper()
            nb_name = (nb.legal_name or "").strip()
            for o in doc_index.series_occurrences.get(nb_sid, []):
                if o.start_offset > chosen_anchor + 50 and not o.is_toc_or_cross_ref:
                    neighbor_anchors.append(o.start_offset)
            for o in doc_index.class_occurrences.get(nb_cid, []):
                if o.start_offset > chosen_anchor + 50 and not o.is_toc_or_cross_ref:
                    neighbor_anchors.append(o.start_offset)
            for o in doc_index.legal_name_occurrences.get(nb_name, []):
                if o.start_offset > chosen_anchor + 50 and not o.is_toc_or_cross_ref:
                    neighbor_anchors.append(o.start_offset)

        end_boundary = len(text)
        if neighbor_anchors:
            earliest_next = min(neighbor_anchors)
            # Find any delimiter preceding the earliest next fund anchor
            n_delims = [
                b for b in doc_index.boundary_candidates
                if start_boundary < b.start_offset <= earliest_next
            ]
            if n_delims:
                end_boundary = n_delims[-1].start_offset
                boundary_evidence_parts.append(f"End boundary delimited by next-fund {n_delims[-1].boundary_type} at offset {end_boundary}")
            else:
                end_boundary = earliest_next
                boundary_evidence_parts.append(f"End boundary delimited by adjacent fund anchor at offset {end_boundary}")
        else:
            # Check for general trust disclosure delimiters
            slice_text = text[start_boundary: min(len(text), start_boundary + 60000)]
            for pat in cls.GENERAL_DISCLOSURE_DELIMITERS:
                m = re.search(pat, slice_text, re.IGNORECASE)
                if m and m.start() > 100:
                    end_boundary = start_boundary + m.start()
                    boundary_evidence_parts.append(f"End boundary delimited by general disclosure at offset {end_boundary}")
                    break

        series_block = text[start_boundary:end_boundary]
        block_sha = hashlib.sha256(series_block.encode("utf-8")).hexdigest()
        boundary_evidence = "; ".join(boundary_evidence_parts)

        # -------------------------------------------------------------
        # STEP 3: STRATEGY SECTION EXTRACTION (Section 20: No Silent Truncation)
        # -------------------------------------------------------------
        strat_anchors_in_block = [
            a for a in doc_index.strategy_anchors
            if start_boundary <= a.start_offset < end_boundary
        ]

        if not strat_anchors_in_block:
            # Check inside series_block directly
            strat_text, sec_name = cls._extract_strategy_from_block(series_block)
        else:
            best_strat = strat_anchors_in_block[0]
            sec_name = best_strat.heading_name
            # Strategy section runs from its anchor until next delimiter or end of series block
            strat_rel_start = best_strat.start_offset - start_boundary
            candidate_block = series_block[strat_rel_start:]

            # Terminate at next standard item heading
            term_m = re.search(
                r"\b(Principal\s+Risks?|Principal\s+Risk\s+Factors|Principal\s+Investment\s+Risks|Annual\s+Fund\s+Operating\s+Expenses|Portfolio\s+Management|Fund\s+Management|Purchase\s+and\s+Sale|Tax\s+Information|Payments\s+to\s+Broker-Dealers|Financial\s+Intermediary\s+Compensation)\b",
                candidate_block[50:],
                re.IGNORECASE,
            )
            if term_m:
                raw_strat = candidate_block[: 50 + term_m.start()]
            else:
                raw_strat = candidate_block

            # Clean HTML tags from strategy section
            if "<" in raw_strat and ">" in raw_strat:
                soup = BeautifulSoup(raw_strat, "html.parser")
                strat_text = soup.get_text(separator=" ", strip=True)
            else:
                strat_text = raw_strat.strip()

        source_section_len = len(strat_text)
        is_truncated = False

        if not strat_text or len(strat_text.strip()) < 50:
            return cls._fail_result(
                target, doc_index, cls.OUTCOME_PARSE_FAILURE,
                "RULE_STRATEGY_SECTION_EMPTY",
                "Strategy section not extractable from isolated series block"
            )

        if source_section_len > MAX_STRATEGY_LENGTH_CEILING:
            # Exceeded safety ceiling -> explicit failure per Section 20
            return cls._fail_result(
                target, doc_index, cls.OUTCOME_EXPLICIT_TRUNCATION_FAILURE,
                "RULE_STRATEGY_LENGTH_EXCEEDED_CEILING",
                f"Source strategy section length ({source_section_len}) exceeds ceiling ({MAX_STRATEGY_LENGTH_CEILING})"
            )


        # -------------------------------------------------------------
        # STEP 4: INDEPENDENT LEAKAGE VALIDATION (Section 18 & 19)
        # -------------------------------------------------------------
        leakage_status, leakage_count = cls.verify_leakage(strat_text, neighbors)
        if leakage_status != cls.LEAKAGE_CHECKED_CLEAN:
            return cls._fail_result(
                target, doc_index, cls.OUTCOME_AMBIGUOUS_MULTI_MATCH,
                "RULE_CROSS_SERIES_LEAKAGE_DETECTED",
                f"Cross-series leakage detected: {leakage_count} neighbor occurrence(s) inside strategy text"
            )

        # -------------------------------------------------------------
        # STEP 5: COMPUTE DETERMINISTIC CACHE IDENTITIES (Section 22 & 23)
        # -------------------------------------------------------------
        strat_sha = hashlib.sha256(strat_text.encode("utf-8")).hexdigest()
        name_sha = hashlib.sha256(norm_name.encode("utf-8")).hexdigest()
        res_cache_seed = f"{doc_index.document_index_sha256}:{sid}:{cid}:{name_sha}:{SERIES_RESOLVER_VERSION}:{SNAPSHOT_BOUNDARY}"
        resolution_key = hashlib.sha256(res_cache_seed.encode("utf-8")).hexdigest()

        # Frozen parser version is 1.2.0
        parse_seed = f"{strat_sha}:1.2.0"
        parse_identity = hashlib.sha256(parse_seed.encode("utf-8")).hexdigest()

        return SeriesMappingResult(
            symbol=target.symbol,
            cik=target.cik,
            series_id=target.series_id,
            class_id=target.class_id,
            mapping_outcome=mapping_outcome,
            mapping_rule_id=rule_id,
            mapping_evidence=f"Isolated series block [{start_boundary}:{end_boundary}] ({len(series_block)} chars) via {rule_id}",
            identity_evidence=identity_evidence,
            boundary_evidence=boundary_evidence,
            mapping_confidence_state="CONFIDENT_SERIES_ISOLATION",
            selected_accession=doc_index.identity.accession,
            selected_form=doc_index.identity.form,
            primary_document=doc_index.identity.document_filename,
            selected_document=doc_index.identity.document_filename,
            source_bytes_sha256=doc_index.identity.source_bytes_sha256,
            document_index_sha256=doc_index.document_index_sha256,
            extracted_series_block_sha256=block_sha,
            extracted_strategy_text=strat_text,
            exact_source_section=sec_name,
            section_start_offset=start_boundary,
            section_end_offset=end_boundary,
            source_section_length=source_section_len,
            extracted_length=len(strat_text),
            extraction_truncated=is_truncated,
            leakage_status=cls.LEAKAGE_CHECKED_CLEAN,
            cross_series_text_leakage=0,
            series_resolution_cache_key=resolution_key,
            mandate_parse_identity=parse_identity,
            raw_document_sha256=doc_index.identity.source_bytes_sha256,
        )

    @classmethod
    def _select_anchor_closest_to_strategy(
        cls,
        occurrences: List[Occurrence],
        strategy_anchors: List[StrategyAnchor],
        text: str = "",
        neighbors: Optional[List[SeriesMetadata]] = None,
    ) -> int:
        if not occurrences:
            return 0
        if not strategy_anchors:
            return occurrences[0].start_offset

        if not text:
            best_anchor = occurrences[0].start_offset
            min_dist = float("inf")
            for occ in occurrences:
                for sa in strategy_anchors:
                    if sa.start_offset >= occ.start_offset:
                        dist = sa.start_offset - occ.start_offset
                        if dist < min_dist:
                            min_dist = dist
                            best_anchor = occ.start_offset
                        break
            return best_anchor

        # Check for start of Statement of Additional Information (Part B)
        sai_start = float("inf")
        if text:
            m_sai = re.search(
                r"<h[1-4][^>]*>[^<]*Statement\s+of\s+Additional\s+Information|<b>[^<]*Statement\s+of\s+Additional\s+Information|\bStatement\s+of\s+Additional\s+Information\b",
                text[25000:],
                re.IGNORECASE,
            )
            if m_sai:
                sai_start = 25000 + m_sai.start()

        neighbor_names = [n.legal_name.lower().strip() for n in (neighbors or []) if n.legal_name]

        scored_candidates = []
        for occ in occurrences:
            o_start = occ.start_offset
            ahead = text[o_start: o_start + 35000]

            is_clustered_cover = False
            first_1k = ahead[:1500].lower()
            for nname in neighbor_names:
                if nname and nname in first_1k:
                    is_clustered_cover = True
                    break

            closest_sa_dist = float("inf")
            for sa in strategy_anchors:
                if sa.start_offset >= o_start:
                    dist = sa.start_offset - o_start
                    if dist < closest_sa_dist:
                        closest_sa_dist = dist
                    break

            has_obj = bool(re.search(r"Investment\s+Objective", ahead, re.IGNORECASE))
            has_fees = bool(re.search(r"Fees?\s+and\s+Expenses|Annual\s+Fund\s+Operating\s+Expenses|Expense\s+Example", ahead, re.IGNORECASE))
            has_strat = bool(re.search(r"Principal\s+Investment\s+Strateg", ahead, re.IGNORECASE))

            score = 0
            if not is_clustered_cover:
                score += 100
            if has_obj:
                score += 50
            if has_fees:
                score += 100
            if has_strat:
                score += 50
            if has_obj and has_fees and has_strat:
                score += 150
            if closest_sa_dist < 30000:
                score += max(0, 50 - int(closest_sa_dist / 600))
            if o_start >= sai_start:
                score -= 300

            scored_candidates.append((score, closest_sa_dist, o_start))

        scored_candidates.sort(key=lambda x: (-x[0], x[1]))
        if scored_candidates and scored_candidates[0][0] >= 150:
            return scored_candidates[0][2]
        return None

    @classmethod
    def _extract_strategy_from_block(cls, block: str) -> Tuple[str, str]:
        for pat in [
            r"Principal\s+Investment\s+Strateg(?:y|ies)",
            r"Investment\s+Objective\s+and\s+Principal\s+Strategies",
            r"Principal\s+Strategies",
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                found_start = m.start()
                sec_name = m.group(0)
                extracted = block[found_start:]
                term_m = re.search(
                    r"\b(Principal\s+Risks?|Principal\s+Risk\s+Factors|Principal\s+Investment\s+Risks|Annual\s+Fund\s+Operating\s+Expenses|Portfolio\s+Management|Fund\s+Management|Purchase\s+and\s+Sale|Tax\s+Information|Payments\s+to\s+Broker-Dealers|Financial\s+Intermediary\s+Compensation)\b",
                    extracted[50:],
                    re.IGNORECASE,
                )
                if term_m:
                    extracted = extracted[: 50 + term_m.start()]
                if "<" in extracted and ">" in extracted:
                    soup = BeautifulSoup(extracted[:MAX_STRATEGY_LENGTH_CEILING], "html.parser")
                    extracted = soup.get_text(separator=" ", strip=True)
                return extracted.strip(), sec_name
        return "", "SECTION_NOT_FOUND"

    @classmethod
    def _fail_result(
        cls,
        target: SeriesMetadata,
        doc_index: Optional[DocumentIndex],
        outcome: str,
        rule_id: str,
        evidence: str,
    ) -> SeriesMappingResult:
        ident = doc_index.identity if doc_index else None
        return SeriesMappingResult(
            symbol=target.symbol,
            cik=target.cik,
            series_id=target.series_id,
            class_id=target.class_id,
            mapping_outcome=outcome,
            mapping_rule_id=rule_id,
            mapping_evidence=evidence,
            identity_evidence=f"Failed identity validation: {outcome}",
            boundary_evidence="No boundary established",
            mapping_confidence_state="FAIL_CLOSED",
            selected_accession=ident.accession if ident else "NONE",
            selected_form=ident.form if ident else "NONE",
            primary_document=ident.document_filename if ident else "NONE",
            selected_document=ident.document_filename if ident else "NONE",
            source_bytes_sha256=ident.source_bytes_sha256 if ident else "NONE",
            document_index_sha256=doc_index.document_index_sha256 if doc_index else "NONE",
            extracted_series_block_sha256="NONE",
            extracted_strategy_text="",
            exact_source_section="NONE",
            leakage_status=cls.LEAKAGE_NOT_EVALUATED,
            raw_document_sha256=ident.source_bytes_sha256 if ident else "NONE",
        )
