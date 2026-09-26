"""ARX Terminal — Series-Level Statutory Prospectus Mapping Engine (Policy v1.1.0).

FROZEN PRE-POPULATION EXECUTION RULESET: SERIES_MAPPING_V1_0_0_FROZEN.

Enforces:
1. Deterministic series-level statutory prospectus mapping for multi-series trusts.
2. Mapping hierarchy (Section 6 & 11):
   - MAPPED_EXACT_SERIES_ID
   - MAPPED_EXACT_CLASS_ID
   - MAPPED_EXACT_LEGAL_NAME
   - MAPPED_DETERMINISTIC_COMPOSITE
3. Fail-closed states:
   - SOURCE_NOT_FOUND
   - SERIES_NOT_FOUND_IN_SOURCE
   - AMBIGUOUS_MULTI_MATCH
   - PARSE_FAILURE
4. Series-delimited omnibus document segmentation with CROSS_SERIES_TEXT_LEAKAGE = 0.
5. Strict snapshot boundary: 2026-09-24T23:59:59Z (POST_BOUNDARY_FILINGS_USED = 0).
6. Complete document-selection audit trail per Section 10.
"""

import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Set, Any
from bs4 import BeautifulSoup


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
class SeriesMappingResult:
    """Deterministic document-selection audit trail record (Section 10 & 11)."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    mapping_outcome: str
    mapping_rule_id: str
    mapping_evidence: str
    mapping_confidence_state: str
    selected_accession: str = "NONE"
    selected_form: str = "NONE"
    primary_document: str = "NONE"
    selected_document: str = "NONE"
    raw_document_sha256: str = "NONE"
    extracted_series_block_sha256: str = "NONE"
    extracted_strategy_text: str = ""
    exact_source_section: str = "NONE"
    cross_series_text_leakage: int = 0


class SeriesProspectusMapper:
    """Frozen series-level statutory prospectus mapping engine."""

    RULESET_ID = "SERIES_MAPPING_V1_0_0_FROZEN"
    SNAPSHOT_BOUNDARY = "2026-09-24"
    SNAPSHOT_BOUNDARY_ISO = "2026-09-24T23:59:59Z"

    # Outcome constants (Section 11)
    OUTCOME_EXACT_SERIES_ID = "MAPPED_EXACT_SERIES_ID"
    OUTCOME_EXACT_CLASS_ID = "MAPPED_EXACT_CLASS_ID"
    OUTCOME_EXACT_LEGAL_NAME = "MAPPED_EXACT_LEGAL_NAME"
    OUTCOME_DETERMINISTIC_COMPOSITE = "MAPPED_DETERMINISTIC_COMPOSITE"
    OUTCOME_SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    OUTCOME_SERIES_NOT_FOUND = "SERIES_NOT_FOUND_IN_SOURCE"
    OUTCOME_AMBIGUOUS_MULTI_MATCH = "AMBIGUOUS_MULTI_MATCH"
    OUTCOME_PARSE_FAILURE = "PARSE_FAILURE"

    AUTHORIZED_OUTCOMES_FOR_PARSING = {
        OUTCOME_EXACT_SERIES_ID,
        OUTCOME_EXACT_CLASS_ID,
        OUTCOME_EXACT_LEGAL_NAME,
        OUTCOME_DETERMINISTIC_COMPOSITE,
    }

    # Policy v1.1.0 authorized statutory forms
    STATUTORY_FORMS = {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A", "497", "497K"}

    # Form preference hierarchy (485BPOS preferred over 485APOS, then 497K/497, then N-1A)
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

    # Strategy heading patterns
    STRATEGY_PATTERNS = [
        r"Principal\s+Investment\s+Strateg(?:y|ies)",
        r"Investment\s+Objective\s+and\s+Principal\s+Strategies",
        r"Principal\s+Strategies",
        r"Principal\s+Investment\s+Policies\s+and\s+Strategies",
        r"Principal\s+Risks\s+and\s+Strategies",
    ]

    # Delimiters marking the end of a series summary section or start of general disclosures
    GENERAL_DISCLOSURE_DELIMITERS = [
        r"\bStatement\s+of\s+Additional\s+Information\b",
        r"\bPART\s+B\b",
        r"\bPART\s+C\b",
        r"\bAdditional\s+Information\s+about\s+(?:the\s+)?Funds?\b",
        r"\bInvestment\s+Objectives,\s+Strategies\s+and\s+Risks\b",
        r"\bMore\s+Information\s+About\s+(?:the\s+)?Funds?\b",
        r"\bGeneral\s+Information\b",
        r"\bFinancial\s+Highlights\b",
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"&nbsp;", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"&amp;", "&", clean, flags=re.IGNORECASE)
        clean = re.sub(r"&#160;", " ", clean)
        clean = re.sub(r"&#8212;", "—", clean)
        clean = re.sub(r"&#8217;", "'", clean)
        clean = re.sub(r"&rsquo;", "'", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """Normalize legal fund name for robust comparison."""
        clean = cls.clean_text(name).lower()
        clean = re.sub(r"[^\w\s]", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def build_filing_index_for_cik(
        cls,
        cik: str,
        submission_json: dict,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY
    ) -> List[Dict[str, Any]]:
        """Constructs an index of pre-boundary statutory filings for a CIK (Section 5).

        Enforces:
        - POST_BOUNDARY_FILINGS_USED = 0
        - Form filtering to STATUTORY_FORMS
        - Priority-based sorting (latest date, then form priority)
        """
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
                acc = accessions[i] if i < len(accessions) else ""
                pdoc = primary_docs[i] if i < len(primary_docs) else ""
                pdesc = primary_descs[i] if i < len(primary_descs) else ""
                rdate = report_dates[i] if i < len(report_dates) else ""
                atime = acceptance_times[i] if i < len(acceptance_times) else ""
                fnum = file_numbers[i] if i < len(file_numbers) else ""

                indexed_filings.append({
                    "cik": str(cik),
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "report_date": rdate,
                    "acceptance_date_time": atime,
                    "primary_document": pdoc,
                    "primary_doc_description": pdesc,
                    "file_number": fnum,
                    "form_priority": cls.FORM_PRIORITY.get(form, 0),
                })

        # Sort: latest filing_date first, then form_priority descending
        indexed_filings.sort(key=lambda x: (x["filing_date"], x["form_priority"]), reverse=True)
        return indexed_filings

    @classmethod
    def verify_cross_series_leakage(
        cls,
        extracted_block: str,
        neighboring_series: List[SeriesMetadata]
    ) -> int:
        """Verifies that the extracted series section does not contain strategy text

        belonging to adjacent funds (Section 9).
        Requires CROSS_SERIES_TEXT_LEAKAGE = 0.
        """
        if not neighboring_series or not extracted_block:
            return 0

        leakage_count = 0
        block_clean = cls.clean_text(extracted_block).lower()

        for neighbor in neighboring_series:
            n_sid = neighbor.series_id.lower() if neighbor.series_id else ""
            n_name = cls.normalize_name(neighbor.legal_name) if neighbor.legal_name else ""

            # Check if neighbor's series ID appears alongside a strategy section
            if n_sid and n_sid in block_clean:
                pos = block_clean.find(n_sid)
                snippet = block_clean[pos: pos + 5000]
                for pat in cls.STRATEGY_PATTERNS:
                    if re.search(pat, snippet, re.IGNORECASE):
                        leakage_count += 1
                        break

            # Check if neighbor's normalized legal name appears alongside a strategy section
            if n_name and len(n_name) > 10 and n_name in block_clean:
                pos = block_clean.find(n_name)
                snippet = block_clean[pos: pos + 5000]
                for pat in cls.STRATEGY_PATTERNS:
                    if re.search(pat, snippet, re.IGNORECASE):
                        leakage_count += 1
                        break

        return leakage_count

    @classmethod
    def isolate_series_block(
        cls,
        html_or_text: str,
        target_series: SeriesMetadata,
        neighboring_series: Optional[List[SeriesMetadata]] = None
    ) -> Tuple[str, str, str, str]:
        """Isolates the deterministic statutory strategy block for target series (Section 8).

        Returns:
            (series_block, mapping_outcome, rule_id, evidence)
        """
        if not html_or_text or len(html_or_text.strip()) < 50:
            return "", cls.OUTCOME_PARSE_FAILURE, "EMPTY_DOCUMENT", "Filing document is empty"

        text = html_or_text
        neighboring = neighboring_series or []

        # If document is pure single-series (no neighbors), verify target is mentioned
        if not neighboring:
            sid = target_series.series_id
            cid = target_series.class_id
            name = target_series.legal_name
            has_sid = bool(sid and sid.lower() in text.lower())
            has_cid = bool(cid and cid.lower() in text.lower())
            has_name = bool(name and cls.normalize_name(name) in cls.normalize_name(text))

            if has_sid:
                return text, cls.OUTCOME_EXACT_SERIES_ID, "MAPPING_RULE_SINGLE_SERIES_SERIES_ID", f"Single-series filing verified by series ID {sid}"
            elif has_cid:
                return text, cls.OUTCOME_EXACT_CLASS_ID, "MAPPING_RULE_SINGLE_SERIES_CLASS_ID", f"Single-series filing verified by class ID {cid}"
            elif has_name:
                return text, cls.OUTCOME_EXACT_LEGAL_NAME, "MAPPING_RULE_SINGLE_SERIES_LEGAL_NAME", f"Single-series filing verified by legal name {name}"
            else:
                return "", cls.OUTCOME_SERIES_NOT_FOUND, "SERIES_NOT_IN_SOURCE", f"Target {sid}/{cid}/{name} not found in single-series filing"

        # --- MULTI-SERIES OMNIBUS DOCUMENT SEGMENTATION ---
        sid = target_series.series_id
        cid = target_series.class_id
        raw_name = target_series.legal_name

        # Hierarchy: 1) explicit series ID, 2) explicit class ID, 3) legal name
        sid_matches = list(re.finditer(re.escape(sid), text, re.IGNORECASE)) if sid else []
        cid_matches = list(re.finditer(re.escape(cid), text, re.IGNORECASE)) if cid else []
        
        name_matches = []
        if raw_name and len(raw_name.strip()) > 5:
            words = [re.escape(w) for w in re.split(r"\s+", raw_name.strip()) if len(w) > 1]
            if len(words) >= 2:
                name_pat = r"\s+(?:<[^>]+>\s*)*".join(words)
                name_matches = list(re.finditer(name_pat, text, re.IGNORECASE))

        target_anchors = []
        outcome = None
        rule_id = None

        if sid_matches:
            target_anchors = [m.start() for m in sid_matches]
            outcome = cls.OUTCOME_EXACT_SERIES_ID
            rule_id = "MAPPING_RULE_EXACT_SERIES_ID"
        elif cid_matches:
            target_anchors = [m.start() for m in cid_matches]
            outcome = cls.OUTCOME_EXACT_CLASS_ID
            rule_id = "MAPPING_RULE_EXACT_CLASS_ID"
        elif name_matches:
            target_anchors = [m.start() for m in name_matches]
            outcome = cls.OUTCOME_EXACT_LEGAL_NAME
            rule_id = "MAPPING_RULE_EXACT_LEGAL_NAME"
        else:
            return "", cls.OUTCOME_SERIES_NOT_FOUND, "SERIES_NOT_IN_SOURCE", (
                f"Target series {sid} / {cid} / '{raw_name}' not found in document"
            )

        # In documents with XBRL headers, choose anchor closest to a strategy section
        best_anchor = None
        best_strategy_dist = float("inf")

        strat_positions = []
        for pat in cls.STRATEGY_PATTERNS:
            strat_positions.extend([m.start() for m in re.finditer(pat, text, re.IGNORECASE)])
        strat_positions.sort()

        if strat_positions:
            for anc in target_anchors:
                for sp in strat_positions:
                    if sp >= anc and (sp - anc) < best_strategy_dist:
                        best_strategy_dist = sp - anc
                        best_anchor = anc
                        break

        if best_anchor is None:
            ix_header_end = text.lower().find("</ix:header>")
            if ix_header_end != -1:
                post_ix_anchors = [a for a in target_anchors if a > ix_header_end]
                best_anchor = post_ix_anchors[0] if post_ix_anchors else target_anchors[-1]
            else:
                best_anchor = target_anchors[0]

        # Determine start boundary of target series section
        window_start = max(0, best_anchor - 15000)
        preceding = text[window_start:best_anchor]

        start_boundary = best_anchor
        delimiter_patterns = [
            r"<hr\s*/?>",
            r"<div[^>]*class=[\"'][^\"']*(?:fund-summary|fund-section|summary|summary-section)[\"']",
            r"Fund\s+Summary\b",
            r"Summary\s+Prospectus\b",
            r"Fund\s+Overview\b",
            r"SUMMARY\s+SECTION\b",
        ]
        if raw_name:
            words = [re.escape(w) for w in re.split(r"\s+", raw_name.strip()) if len(w) > 1]
            if len(words) >= 2:
                delimiter_patterns.insert(0, r"\s+(?:<[^>]+>\s*)*".join(words))

        for pat in delimiter_patterns:
            matches = list(re.finditer(pat, preceding, re.IGNORECASE))
            if matches:
                start_boundary = window_start + matches[-1].start()
                break

        # Determine end boundary: the earliest anchor of any neighboring series AFTER best_anchor + 10
        neighbor_anchors = []
        for neighbor in neighboring:
            if neighbor.series_id:
                for m in re.finditer(re.escape(neighbor.series_id), text, re.IGNORECASE):
                    if m.start() > best_anchor + 10:
                        neighbor_anchors.append(m.start())
            if neighbor.class_id:
                for m in re.finditer(re.escape(neighbor.class_id), text, re.IGNORECASE):
                    if m.start() > best_anchor + 10:
                        neighbor_anchors.append(m.start())
            if neighbor.legal_name and len(neighbor.legal_name.strip()) > 5:
                n_words = [re.escape(w) for w in re.split(r"\s+", neighbor.legal_name.strip()) if len(w) > 1]
                if len(n_words) >= 2:
                    n_pat = r"\s+(?:<[^>]+>\s*)*".join(n_words)
                    for m in re.finditer(n_pat, text, re.IGNORECASE):
                        if m.start() > best_anchor + 10:
                            neighbor_anchors.append(m.start())

        end_boundary = len(text)
        if neighbor_anchors:
            earliest_neighbor = min(neighbor_anchors)
            n_win_start = max(start_boundary, earliest_neighbor - 10000)
            n_preceding = text[n_win_start:earliest_neighbor]
            n_delims = []
            for pat in [
                r"<hr\s*/?>",
                r"<div[^>]*class=[\"'][^\"']*(?:fund-summary|fund-section|summary|summary-section)[\"']",
                r"Fund\s+Summary\b",
                r"Summary\s+Prospectus\b",
                r"SUMMARY\s+SECTION\b"
            ]:
                n_delims.extend([m.start() for m in re.finditer(pat, n_preceding, re.IGNORECASE)])
            if n_delims:
                end_boundary = n_win_start + n_delims[-1]
            else:
                end_boundary = earliest_neighbor

        # Also check for general trust disclosure delimiters
        block_prelim = text[start_boundary:end_boundary]
        for delim_pat in cls.GENERAL_DISCLOSURE_DELIMITERS:
            m = re.search(delim_pat, block_prelim, re.IGNORECASE)
            if m and m.start() > 100:
                end_boundary = start_boundary + m.start()
                break

        series_block = text[start_boundary:end_boundary]

        # Verify cross-series leakage (must be 0)
        leakage = cls.verify_cross_series_leakage(series_block, neighboring)
        if leakage > 0:
            evidence = f"Cross-series leakage detected: {leakage} neighboring series strategy sections present in isolated block"
            return "", cls.OUTCOME_AMBIGUOUS_MULTI_MATCH, "AMBIGUOUS_CROSS_SERIES_LEAKAGE", evidence

        evidence = (
            f"Isolated series block [{start_boundary}:{end_boundary}] ({len(series_block)} chars) "
            f"for {sid} via {rule_id}"
        )
        return series_block, outcome, rule_id, evidence

    @classmethod
    def extract_series_strategy(
        cls,
        series_block: str,
        target_series: SeriesMetadata
    ) -> Tuple[str, str]:
        """Extracts the Principal Investment Strategies section from the isolated series block."""
        if not series_block or len(series_block.strip()) < 50:
            return "", "EMPTY_SERIES_BLOCK"

        if "<html" in series_block.lower() or "<div" in series_block.lower() or "<p" in series_block.lower():
            soup = BeautifulSoup(series_block[:1000000], "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        else:
            text = series_block

        found_start = -1
        section_name = ""
        for pat in cls.STRATEGY_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found_start = m.start()
                section_name = m.group(0)
                break

        if found_start == -1:
            return "", "SECTION_NOT_FOUND"

        # Bounded extract: up to 15,000 characters from strategy start
        extracted = text[found_start: found_start + 15000]
        return extracted, section_name

    @classmethod
    def map_series(
        cls,
        target_series: SeriesMetadata,
        raw_document_text: str,
        accession: str,
        form: str,
        document_filename: str,
        neighboring_series: Optional[List[SeriesMetadata]] = None
    ) -> SeriesMappingResult:
        """Executes full deterministic series mapping and strategy extraction."""
        raw_sha = hashlib.sha256(raw_document_text.encode("utf-8")).hexdigest()

        # Step 1: Isolate series block
        block, outcome, rule_id, evidence = cls.isolate_series_block(
            html_or_text=raw_document_text,
            target_series=target_series,
            neighboring_series=neighboring_series
        )

        if outcome not in cls.AUTHORIZED_OUTCOMES_FOR_PARSING:
            return SeriesMappingResult(
                symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                mapping_outcome=outcome,
                mapping_rule_id=rule_id,
                mapping_evidence=evidence,
                mapping_confidence_state="FAIL_CLOSED",
                selected_accession=accession,
                selected_form=form,
                primary_document=document_filename,
                selected_document=document_filename,
                raw_document_sha256=raw_sha,
                extracted_series_block_sha256="NONE",
                extracted_strategy_text="",
                exact_source_section="NONE",
                cross_series_text_leakage=0
            )

        block_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()

        # Step 2: Extract strategy section from isolated block
        strat_text, sec_name = cls.extract_series_strategy(block, target_series)
        if not strat_text or len(strat_text.strip()) < 50:
            return SeriesMappingResult(
                symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                mapping_outcome=cls.OUTCOME_PARSE_FAILURE,
                mapping_rule_id="RULE_STRATEGY_SECTION_EMPTY",
                mapping_evidence="Strategy section not extractable from isolated series block",
                mapping_confidence_state="PARSE_FAILED",
                selected_accession=accession,
                selected_form=form,
                primary_document=document_filename,
                selected_document=document_filename,
                raw_document_sha256=raw_sha,
                extracted_series_block_sha256=block_sha,
                extracted_strategy_text="",
                exact_source_section=sec_name,
                cross_series_text_leakage=0
            )

        return SeriesMappingResult(
            symbol=target_series.symbol,
            cik=target_series.cik,
            series_id=target_series.series_id,
            class_id=target_series.class_id,
            mapping_outcome=outcome,
            mapping_rule_id=rule_id,
            mapping_evidence=evidence,
            mapping_confidence_state="CONFIDENT_SERIES_ISOLATION",
            selected_accession=accession,
            selected_form=form,
            primary_document=document_filename,
            selected_document=document_filename,
            raw_document_sha256=raw_sha,
            extracted_series_block_sha256=block_sha,
            extracted_strategy_text=strat_text,
            exact_source_section=sec_name,
            cross_series_text_leakage=0
        )
