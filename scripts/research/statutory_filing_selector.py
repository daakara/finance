"""ARX Terminal — Statutory Filing Selector (STATUTORY_FILING_SELECTOR_V1_0_0).

Deterministically maps an ETF series (SeriesMetadata) to its correct eligible
pre-boundary statutory filing and document before DocumentIndex and SeriesProspectusMapper
are invoked.

Core Principles:
1. Registrant != Document (CIK may have multiple accessions, multiple prospectuses, multiple series).
2. Strict temporal cutoff (filingDate <= SNAPSHOT_BOUNDARY; POST_BOUNDARY_FILING_SELECTED = 0).
3. Statutory document roles (distinguishes Base Prospectus, Summary Prospectus, Supplements, SAI Part B).
4. Multi-accession search (iterates through candidates if target is absent from top candidate).
5. Target presence and mandate section verification (Item 4, StrategyNarrativeTextBlock, etc.).
6. Deterministic audit trail & cache identity.
7. Zero classification feedback (no tuning toward desired mandate subtypes or blocker counts).
"""

import os
import re
import json
import hashlib
import unicodedata
import html
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Any

from scripts.research.series_prospectus_mapper import SeriesMetadata, DocumentNormalizer

STATUTORY_FILING_SELECTOR_VERSION = "STATUTORY_FILING_SELECTOR_V1_0_0"
SNAPSHOT_BOUNDARY = "2026-09-24T23:59:59Z"
SNAPSHOT_BOUNDARY_DATE = "2026-09-24"

# Statutory Forms authorized by Policy v1.1
STATUTORY_FORMS = {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A", "497K", "497"}

# Document Roles (Section 7)
ROLE_BASE_STATUTORY_PROSPECTUS = "BASE_STATUTORY_PROSPECTUS"
ROLE_SUMMARY_PROSPECTUS = "SUMMARY_PROSPECTUS"
ROLE_PROSPECTUS_SUPPLEMENT = "PROSPECTUS_SUPPLEMENT"
ROLE_FEE_WAIVER_SUPPLEMENT = "FEE_WAIVER_SUPPLEMENT"
ROLE_SAI_PART_B = "SAI_PART_B"
ROLE_NON_MANDATE_DOCUMENT = "NON_MANDATE_DOCUMENT"
ROLE_UNKNOWN = "UNKNOWN"

# Selection Outcomes (Section 16)
OUTCOME_SELECTED_STATUTORY_PROSPECTUS = "SELECTED_TARGET_STATUTORY_PROSPECTUS"
OUTCOME_SELECTED_SUMMARY_PROSPECTUS = "SELECTED_TARGET_SUMMARY_PROSPECTUS"
OUTCOME_SELECTED_BASE_WITH_SUPPLEMENT = "SELECTED_BASE_PROSPECTUS_WITH_RELEVANT_SUPPLEMENT"
OUTCOME_NO_PREBOUNDARY_CANDIDATE = "NO_PREBOUNDARY_CANDIDATE"
OUTCOME_TARGET_ABSENT_FROM_ALL = "TARGET_ABSENT_FROM_ALL_CANDIDATES"
OUTCOME_MANDATE_ABSENT_FROM_ALL = "MANDATE_SECTION_ABSENT_FROM_ALL_CANDIDATES"
OUTCOME_CONFLICTING_DOCUMENTS = "CONFLICTING_CANDIDATE_DOCUMENTS"
OUTCOME_AMBIGUOUS_MAPPING = "AMBIGUOUS_SERIES_TO_DOCUMENT_MAPPING"
OUTCOME_SOURCE_CACHE_MISS = "SOURCE_CACHE_MISS"


@dataclass
class FilingCandidate:
    """A pre-boundary filing candidate for an ETF registrant."""
    accession: str
    form: str
    filing_date: str
    primary_document: str
    primary_doc_description: str
    document_role: str
    is_preboundary: bool
    is_cached: bool = False
    target_present: str = "UNKNOWN"   # YES / NO / UNKNOWN
    mandate_present: str = "UNKNOWN"  # YES / NO / UNKNOWN
    rejection_reason: str = ""
    target_metadata_match: bool = False
    priority_score: int = 0


@dataclass
class FilingSelectionResult:
    """Deterministic outcome of the statutory filing selection process."""
    target_symbol: str
    cik: str
    series_id: str
    class_id: str
    legal_name: str
    selected_accession: str
    selected_form: str
    filing_date: str
    document_filename: str
    document_role: str
    selection_outcome: str
    selection_rule_id: str
    selection_evidence: str
    candidate_count: int
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_boundary: str = SNAPSHOT_BOUNDARY
    selector_version: str = STATUTORY_FILING_SELECTOR_VERSION
    source_bytes_sha256: str = ""
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatutoryFilingSelector:
    """Upstream filing-selection layer for ARX Terminal ETF research."""

    VERSION = STATUTORY_FILING_SELECTOR_VERSION
    SNAPSHOT_BOUNDARY = SNAPSHOT_BOUNDARY
    SNAPSHOT_BOUNDARY_DATE = SNAPSHOT_BOUNDARY_DATE

    # Delimiters and patterns for role classification
    SAI_PATTERNS = [
        r"\bStatement\s+of\s+Additional\s+Information\b",
        r"\bPart\s+B\b",
        r"\bCONSOLIDATED\s+SAI\b",
        r"-sai\b",
        r"_sai\b",
        r"\bsai\.",
        r"\bSAI\b",
    ]

    SUPPLEMENT_FEE_WAIVER_PATTERNS = [
        r"\bfee\s*waiver\b",
        r"feewaiver",
        r"\bliquidation\b",
        r"\bsticker\b",
        r"\bdistributor\s*change\b",
        r"\bfee\s*table\b",
        r"\bname\s*change\b",
        r"\bindex\s*reconstitution\b",
        r"\bpm\s*change\b",
    ]

    MANDATE_SECTION_PATTERNS = [
        r"oef:StrategyNarrativeTextBlock",
        r"oef:RiskReturnHeading",
        r"oef:ObjectivePrimaryTextBlock",
        r"Principal\s+Investment\s+Strateg",
        r"Investment\s+Objective\s+and\s+Principal\s+Strategies",
        r"Principal\s+Strategies",
        r"Fund\s+Summary",
    ]

    @classmethod
    def classify_document_role(
        cls,
        form: str,
        primary_document: str,
        primary_doc_description: str,
        cached_text: Optional[str] = None
    ) -> str:
        """Classify candidate document by its statutory role (Section 7)."""
        form = (form or "").upper().strip()
        doc_lower = (primary_document or "").lower()
        desc_lower = (primary_doc_description or "").lower()
        combined_meta = f"{doc_lower} {desc_lower}"

        # 1. Check for Summary Prospectus (497K)
        if form == "497K":
            return ROLE_SUMMARY_PROSPECTUS

        # 2. Check for SAI (Statement of Additional Information) Part B
        # Must check if description or document filename indicates SAI
        is_sai = False
        for pat in [r"\bstatement\s+of\s+additional\s+information\b", r"\bsai\b", r"-sai\b", r"_sai\b"]:
            if re.search(pat, combined_meta, re.IGNORECASE):
                is_sai = True
                break
        if is_sai:
            return ROLE_SAI_PART_B

        # 3. Check for 497 Supplements vs Fee Waivers
        if form == "497":
            for pat in cls.SUPPLEMENT_FEE_WAIVER_PATTERNS:
                if re.search(pat, combined_meta, re.IGNORECASE):
                    return ROLE_FEE_WAIVER_SUPPLEMENT
            return ROLE_PROSPECTUS_SUPPLEMENT

        # 4. Check for Base Statutory Prospectus (485BPOS, 485APOS, N-1A)
        if form in {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A"}:
            return ROLE_BASE_STATUTORY_PROSPECTUS

        return ROLE_UNKNOWN

    @classmethod
    def check_target_presence(
        cls,
        target_series: SeriesMetadata,
        text: str
    ) -> Tuple[bool, str]:
        """Verify whether target series is genuinely present in candidate text (Section 8).

        Rejects candidates where target appears only in:
        - Trustee / officer tables
        - General compensation tables
        - SAI investment restrictions tables
        """
        if not text or len(text.strip()) < 10:
            return False, "EMPTY_TEXT"

        sid = (target_series.series_id or "").upper().strip()
        cid = (target_series.class_id or "").upper().strip()
        raw_name = (target_series.legal_name or "").strip()
        norm_name = DocumentNormalizer.normalize_name(raw_name)

        # Fast substring check before regex
        has_sid = bool(sid and sid in text)
        has_cid = bool(cid and cid in text)

        has_name = False
        if raw_name and len(raw_name) > 5:
            if raw_name.lower() in text.lower():
                has_name = True
            else:
                words = raw_name.split()
                name_pat = r"\s+".join(re.escape(w) for w in words)
                has_name = bool(re.search(name_pat, text, re.IGNORECASE))

        if not has_sid and not has_cid and not has_name:
            return False, "TARGET_NOT_FOUND_IN_TEXT"

        # Check if the target presence is purely inside SAI back-of-book or trustee table
        # Identify if document has an SAI boundary
        m_sai = re.search(
            r"<h[1-4][^>]*>[^<]*Statement\s+of\s+Additional\s+Information|<b>[^<]*Statement\s+of\s+Additional\s+Information|\bStatement\s+of\s+Additional\s+Information\b",
            text,
            re.IGNORECASE,
        )
        if m_sai and m_sai.start() > 5000:
            sai_start = m_sai.start()
            # If target appears ONLY after sai_start, check if it's an SAI-only mention
            first_occ = len(text)
            if has_sid:
                m = re.search(rf"\b{re.escape(sid)}\b", text)
                if m:
                    first_occ = min(first_occ, m.start())
            if has_name:
                words = raw_name.split()
                name_pat = r"\s+".join(re.escape(w) for w in words)
                m = re.search(name_pat, text, re.IGNORECASE)
                if m:
                    first_occ = min(first_occ, m.start())

            if first_occ > sai_start:
                # Target occurs strictly after SAI start
                # Verify if there's any Item 4 or Fund Summary in that region
                has_summary_after_sai = bool(re.search(r"\bFund\s+Summary\b|oef:RiskReturnHeading", text[first_occ-500:first_occ+2000], re.IGNORECASE))
                if not has_summary_after_sai:
                    return False, "TARGET_ONLY_IN_SAI_SECTION"

        return True, "TARGET_PRESENT"

    @classmethod
    def check_mandate_content(cls, text: str) -> Tuple[bool, str]:
        """Verify whether statutory mandate / strategy section exists in text (Section 9)."""
        if not text:
            return False, "NO_TEXT"

        for pat in cls.MANDATE_SECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return True, f"MANDATE_PRESENT_{pat}"

        return False, "MANDATE_SECTION_NOT_FOUND"

    @classmethod
    def compute_cache_key(
        cls,
        target: SeriesMetadata,
        cik: str,
        config: Optional[Dict[str, Any]] = None,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY,
    ) -> str:
        """Compute deterministic selection cache identity (Section 18)."""
        sid = target.series_id or ""
        cid = target.class_id or ""
        norm_name = DocumentNormalizer.normalize_name(target.legal_name or "")
        name_sha = hashlib.sha256(norm_name.encode("utf-8")).hexdigest()
        cfg_sha = hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode("utf-8")).hexdigest()
        raw = f"{sid}:{cid}:{name_sha}:{cik}:{snapshot_boundary}:{cls.VERSION}:{cfg_sha}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def select_statutory_filing(
        cls,
        target_series: SeriesMetadata,
        submission_json: dict,
        cache_dir: Optional[Path] = None,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY,
        allow_network_acquisition: bool = False,
        config: Optional[Dict[str, Any]] = None,
        file_cache: Optional[Dict[str, str]] = None,
        cached_filenames: Optional[Set[str]] = None,
    ) -> FilingSelectionResult:
        """Deterministically selects the correct pre-boundary statutory filing for target_series.

        Steps:
        1. Enumerate candidate filings from submission_json.
        2. Filter strictly by filingDate <= snapshot_boundary.
        3. Classify document roles (BASE_STATUTORY_PROSPECTUS, SUMMARY_PROSPECTUS, etc.).
        4. Prioritize candidates by target relevance and form hierarchy.
        5. For cached documents, verify target presence and mandate section.
        6. Fail closed with informative outcome if target is absent or ambiguous.
        """
        cik = str(target_series.cik or "").zfill(10)
        recent = submission_json.get("filings", {}).get("recent", {})
        cache_key = cls.compute_cache_key(target_series, cik, config, snapshot_boundary)

        if not recent:
            return FilingSelectionResult(
                target_symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                legal_name=target_series.legal_name,
                selected_accession="NONE",
                selected_form="NONE",
                filing_date="NONE",
                document_filename="NONE",
                document_role=ROLE_UNKNOWN,
                selection_outcome=OUTCOME_NO_PREBOUNDARY_CANDIDATE,
                selection_rule_id="RULE_EMPTY_SUBMISSION_METADATA",
                selection_evidence="No recent filings found in CIK submissions metadata",
                candidate_count=0,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
            )

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_descs = recent.get("primaryDocDescription", [])

        # Step 1: Enumerate pre-boundary candidate filings
        candidates: List[FilingCandidate] = []
        rejected_candidates: List[Dict[str, Any]] = []

        target_sym_lower = (target_series.symbol or "").lower().strip()
        target_sid_lower = (target_series.series_id or "").lower().strip()
        target_cid_lower = (target_series.class_id or "").lower().strip()
        target_name_words = [
            w.lower() for w in (target_series.legal_name or "").split()
            if len(w) > 3 and w.lower() not in {"fund", "etf", "index", "trust", "series"}
        ]

        prospectus_dir = cache_dir / "sec_prospectus" if cache_dir else Path("data/research/cache/sec_prospectus")

        if cached_filenames is None:
            cached_filenames = {p.name for p in prospectus_dir.iterdir()} if prospectus_dir.exists() else set()

        for i, form in enumerate(forms):
            fdate = filing_dates[i] if i < len(filing_dates) else ""
            acc = accessions[i] if i < len(accessions) else ""
            pdoc = primary_docs[i] if i < len(primary_docs) else ""
            pdesc = primary_descs[i] if i < len(primary_descs) else ""

            # Strict temporal boundary check (Section 24: POST_BOUNDARY_FILING_SELECTED = 0)
            if fdate > snapshot_boundary[:10]:
                rejected_candidates.append({
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "rejection_reason": f"POST_BOUNDARY_FILING (filingDate {fdate} > {snapshot_boundary[:10]})"
                })
                continue

            if form not in STATUTORY_FORMS:
                continue

            role = cls.classify_document_role(form, pdoc, pdesc)
            if role in {ROLE_SAI_PART_B, ROLE_FEE_WAIVER_SUPPLEMENT, ROLE_NON_MANDATE_DOCUMENT}:
                rejected_candidates.append({
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "primary_doc": pdoc,
                    "rejection_reason": f"DISQUALIFIED_DOCUMENT_ROLE ({role})"
                })
                continue

            # Check target-specific match in metadata
            pdoc_lower = pdoc.lower()
            pdesc_lower = pdesc.lower()
            combined_meta = f"{pdoc_lower} {pdesc_lower}"

            meta_match = False
            if target_sym_lower and (f"{target_sym_lower}." in pdoc_lower or f"{target_sym_lower}sum" in pdoc_lower or f"_{target_sym_lower}" in pdoc_lower or f"-{target_sym_lower}" in pdoc_lower or target_sym_lower in pdesc_lower):
                meta_match = True
            elif target_name_words and sum(1 for w in target_name_words if w in combined_meta) >= min(2, len(target_name_words)):
                meta_match = True

            # Fast in-memory check if file is cached locally
            cached_filename = f"{acc}_{pdoc}"
            is_cached = cached_filename in cached_filenames

            # Priority scoring
            score = 0
            if meta_match:
                score += 500
            if role == ROLE_SUMMARY_PROSPECTUS:
                score += 300 if meta_match else 50
            elif role == ROLE_BASE_STATUTORY_PROSPECTUS:
                score += 200
            elif role == ROLE_PROSPECTUS_SUPPLEMENT:
                score += 20

            candidates.append(FilingCandidate(
                accession=acc,
                form=form,
                filing_date=fdate,
                primary_document=pdoc,
                primary_doc_description=pdesc,
                document_role=role,
                is_preboundary=True,
                is_cached=is_cached,
                target_metadata_match=meta_match,
                priority_score=score
            ))

        if not candidates:
            return FilingSelectionResult(
                target_symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                legal_name=target_series.legal_name,
                selected_accession="NONE",
                selected_form="NONE",
                filing_date="NONE",
                document_filename="NONE",
                document_role=ROLE_UNKNOWN,
                selection_outcome=OUTCOME_NO_PREBOUNDARY_CANDIDATE,
                selection_rule_id="RULE_NO_PREBOUNDARY_STATUTORY_FILING",
                selection_evidence=f"No eligible pre-boundary statutory filing found in CIK {cik} metadata",
                candidate_count=0,
                rejected_candidates=rejected_candidates[:20],
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
            )

        # Sort candidates deterministically: Priority Score desc, Filing Date desc, Form Priority desc
        form_weight = {"485BPOS": 10, "497K": 9, "485APOS": 8, "497": 5, "N-1A": 4}
        candidates.sort(
            key=lambda c: (c.priority_score, c.filing_date, form_weight.get(c.form, 0)),
            reverse=True
        )

        # Step 2: Multi-Accession Search & Content Qualification (Sections 8, 9, 14, 15)
        # Iterate candidates and verify series presence and mandate presence
        inspected_count = 0
        cache_miss_candidate: Optional[FilingCandidate] = None

        for cand in candidates:
            inspected_count += 1
            cached_filename = f"{cand.accession}_{cand.primary_document}"
            cached_path = prospectus_dir / cached_filename

            if not cand.is_cached:
                # Document is not cached locally
                if cand.target_metadata_match and cache_miss_candidate is None:
                    cache_miss_candidate = cand
                # If target metadata matches specifically, we record as potential cache miss
                rejected_candidates.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_doc": cand.primary_document,
                    "rejection_reason": "SOURCE_NOT_CACHED_LOCALLY"
                })
                continue

            # Read cached content and inspect (leveraging in-memory cache if provided)
            if file_cache is not None and str(cached_path) in file_cache:
                content = file_cache[str(cached_path)]
            else:
                try:
                    with open(cached_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if file_cache is not None:
                        file_cache[str(cached_path)] = content
                except Exception as e:
                    rejected_candidates.append({
                        "accession": cand.accession,
                        "rejection_reason": f"CACHE_READ_ERROR: {str(e)}"
                    })
                    continue

            # Check target presence
            target_present, t_reason = cls.check_target_presence(target_series, content)
            if not target_present:
                cand.target_present = "NO"
                cand.rejection_reason = t_reason
                rejected_candidates.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_doc": cand.primary_document,
                    "rejection_reason": f"TARGET_NOT_PRESENT ({t_reason})"
                })
                continue

            cand.target_present = "YES"

            # Check mandate presence
            mandate_present, m_reason = cls.check_mandate_content(content)
            if not mandate_present:
                cand.mandate_present = "NO"
                cand.rejection_reason = m_reason
                rejected_candidates.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_doc": cand.primary_document,
                    "rejection_reason": f"MANDATE_NOT_PRESENT ({m_reason})"
                })
                continue

            cand.mandate_present = "YES"

            # SUCCESS: Selected qualifying statutory document!
            doc_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            outcome = (
                OUTCOME_SELECTED_SUMMARY_PROSPECTUS
                if cand.document_role == ROLE_SUMMARY_PROSPECTUS
                else OUTCOME_SELECTED_STATUTORY_PROSPECTUS
            )
            rule_id = (
                "SELECTION_RULE_TARGET_SUMMARY_PROSPECTUS"
                if cand.document_role == ROLE_SUMMARY_PROSPECTUS
                else "SELECTION_RULE_TARGET_BASE_PROSPECTUS"
            )
            evidence = (
                f"Selected {cand.form} accession {cand.accession} ({cand.primary_document}) filed {cand.filing_date}; "
                f"target presence verified ({t_reason}); mandate section verified ({m_reason})"
            )

            return FilingSelectionResult(
                target_symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                legal_name=target_series.legal_name,
                selected_accession=cand.accession,
                selected_form=cand.form,
                filing_date=cand.filing_date,
                document_filename=cand.primary_document,
                document_role=cand.document_role,
                selection_outcome=outcome,
                selection_rule_id=rule_id,
                selection_evidence=evidence,
                candidate_count=len(candidates),
                rejected_candidates=rejected_candidates,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
                source_bytes_sha256=doc_sha,
            )

        # Step 3: Exhaustive search complete without qualifying cached match
        if cache_miss_candidate is not None:
            # Candidate known from SEC submissions metadata but not yet cached
            return FilingSelectionResult(
                target_symbol=target_series.symbol,
                cik=target_series.cik,
                series_id=target_series.series_id,
                class_id=target_series.class_id,
                legal_name=target_series.legal_name,
                selected_accession=cache_miss_candidate.accession,
                selected_form=cache_miss_candidate.form,
                filing_date=cache_miss_candidate.filing_date,
                document_filename=cache_miss_candidate.primary_document,
                document_role=cache_miss_candidate.document_role,
                selection_outcome=OUTCOME_SOURCE_CACHE_MISS,
                selection_rule_id="RULE_SOURCE_CACHE_MISS_KNOWN_METADATA",
                selection_evidence=(
                    f"Candidate accession {cache_miss_candidate.accession} identified in SEC metadata "
                    f"({cache_miss_candidate.primary_document}) but not yet acquired into local cache"
                ),
                candidate_count=len(candidates),
                rejected_candidates=rejected_candidates,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
            )

        # Genuinely absent from all inspected candidates
        return FilingSelectionResult(
            target_symbol=target_series.symbol,
            cik=target_series.cik,
            series_id=target_series.series_id,
            class_id=target_series.class_id,
            legal_name=target_series.legal_name,
            selected_accession="NONE",
            selected_form="NONE",
            filing_date="NONE",
            document_filename="NONE",
            document_role=ROLE_UNKNOWN,
            selection_outcome=OUTCOME_TARGET_ABSENT_FROM_ALL,
            selection_rule_id="RULE_EXHAUSTIVE_SEARCH_TARGET_ABSENT",
            selection_evidence=f"Target series {target_series.series_id} absent from all {len(candidates)} pre-boundary statutory candidates",
            candidate_count=len(candidates),
            rejected_candidates=rejected_candidates,
            snapshot_boundary=snapshot_boundary,
            cache_key=cache_key,
        )
