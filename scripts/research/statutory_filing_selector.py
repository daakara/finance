"""ARX Terminal — Statutory Filing Selector (STATUTORY_FILING_SELECTOR_V1_2_0).

Deterministically maps an ETF series (SeriesMetadata) to its correct eligible
pre-boundary statutory filing and document before DocumentIndex and SeriesProspectusMapper
are invoked.

V1.2.0 Systematic Remediation:
1. DEFECT_A: Load complete SEC submission history (recent + all filings.files).
2. DEFECT_B: Defined-outcome / buffer month discrimination (no generic "500"+"buffer" collision).
3. DEFECT_C: Context-aware ticker matching in target presence check.
4. DEFECT_D: Concatenated ticker filename support (e.g. precidian-armhasmhandsthhs.htm).
5. DEFECT_E: Omnibus base prospectus in-text evaluation (inspects cached 485BPOS even with generic metadata).
6. DEFECT_F: Distinctive name token order & punctuation normalization (handles HTML entities like &#38;, &#58;, &#8482;).
7. Strict affirmative cache-miss semantics (weak metadata matches never trigger SOURCE_CACHE_MISS).
8. Share-class safety (mutual-fund Admiral/Investor classes never match ETF targets).
9. Per-CIK normalized filing history indexing and deterministic cache identity.
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

STATUTORY_FILING_SELECTOR_VERSION = "STATUTORY_FILING_SELECTOR_V1_4_0"
STATUTORY_FILING_SELECTOR_V1_2_0 = "STATUTORY_FILING_SELECTOR_V1_2_0"
STATUTORY_FILING_SELECTOR_V1_3_0 = "STATUTORY_FILING_SELECTOR_V1_3_0"
STATUTORY_FILING_SELECTOR_V1_3_1 = "STATUTORY_FILING_SELECTOR_V1_3_1"
STATUTORY_FILING_SELECTOR_V1_4_0 = "STATUTORY_FILING_SELECTOR_V1_4_0"
SNAPSHOT_BOUNDARY = "2026-09-24T23:59:59Z"
SNAPSHOT_BOUNDARY_DATE = "2026-09-24"

# Statutory Forms authorized by Policy v1.1
STATUTORY_FORMS = {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A", "497K", "497"}

# Token Classification Hierarchy (Sections 12 & 13)
ISSUER_TOKENS = {
    "goldman", "sachs", "vanguard", "spdr", "state", "street", "ssga", "ishares",
    "invesco", "schwab", "fidelity", "xtrackers", "first", "global", "proshares",
    "direxion", "vaneck", "franklin", "templeton", "jpmorgan", "blackrock",
    "wisdomtree", "dimensional", "pimco", "simplify", "amplify", "roundhill",
    "defiance", "yieldmax", "innovator", "ft", "dws", "dbx", "select", "sector",
    "guinness", "atkinson", "morgan", "stanley", "pathway", "harbor", "hotchkis",
    "wiley", "alpha", "architect", "ea", "bridges", "precidian", "smartetfs", "matthews"
}

GENERIC_PRODUCT_TOKENS = {
    "fund", "funds", "etf", "etfs", "index", "indexes", "trust", "series", "shares",
    "portfolio", "portfolios", "capital", "management", "asset", "assets", "advisor",
    "advisors", "adviser", "advisers", "investment", "investments", "company", "holdings",
    "sp", "spi", "summary", "prospectus", "annual", "update"
}

MUTUAL_FUND_CLASS_TOKENS = {
    "admiral", "investor shares", "institutional shares", "instl shares",
    "class a", "class c", "class i", "class r", "class y", "investor class",
    "institutional class", "admiral shares"
}

STRATEGY_FAMILY_TOKENS = {
    "buffer", "buffered", "laddered", "hedged", "defined", "outcome", "target", "floor",
    "500", "100", "1000", "2000", "large", "mid", "small", "cap", "core", "blend",
    "rate", "treasury", "volatility", "esg", "ultra", "short", "long", "term"
}

MONTH_TOKENS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
}

MONTH_MAP = {
    "january": "jan", "jan": "jan",
    "february": "feb", "feb": "feb",
    "march": "mar", "mar": "mar",
    "april": "apr", "apr": "apr",
    "may": "may",
    "june": "jun", "jun": "jun",
    "july": "jul", "july": "jul", "jul": "jul",
    "august": "aug", "aug": "aug",
    "september": "sep", "sep": "sep",
    "october": "oct", "oct": "oct",
    "november": "nov", "nov": "nov",
    "december": "dec", "dec": "dec"
}

NON_TICKER_FOUR_LETTER_WORDS = {
    "fund", "post", "stat", "supp", "form", "base", "part", "hold", "rate",
    "corp", "curr", "term", "debt", "risk", "tech", "core", "port", "grow",
    "real", "inst", "stmt", "file", "text", "page", "item", "html", "head",
    "body", "font", "span", "desc", "docs", "doc1", "doc2", "type", "size"
}

# Document Roles (Section 7 & 17)
ROLE_BASE_STATUTORY_PROSPECTUS = "BASE_STATUTORY_PROSPECTUS"
ROLE_SUMMARY_PROSPECTUS = "SUMMARY_PROSPECTUS"
ROLE_PROSPECTUS_SUPPLEMENT = "PROSPECTUS_SUPPLEMENT"
ROLE_FEE_WAIVER_SUPPLEMENT = "FEE_WAIVER_SUPPLEMENT"
ROLE_SAI_PART_B = "SAI_PART_B"
ROLE_NON_MANDATE_DOCUMENT = "NON_MANDATE_DOCUMENT"
ROLE_UNKNOWN = "UNKNOWN"

# Specialized Candidate Roles (Section 17)
ROLE_TARGET_SUMMARY_PROSPECTUS = "TARGET_SUMMARY_PROSPECTUS"
ROLE_TARGET_BASE_STATUTORY_PROSPECTUS = "TARGET_BASE_STATUTORY_PROSPECTUS"
ROLE_TARGET_PROSPECTUS_SUPPLEMENT = "TARGET_PROSPECTUS_SUPPLEMENT"
ROLE_NON_TARGET_PROSPECTUS = "NON_TARGET_PROSPECTUS"
ROLE_PART_C_ONLY = "PART_C_ONLY"
ROLE_SAI = "SAI"
ROLE_FEE_WAIVER_ONLY = "FEE_WAIVER_ONLY"
ROLE_ANCILLARY = "ANCILLARY"

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
class NormalizedFilingRecord:
    """A canonical normalized SEC submission filing record (Section 6)."""
    cik: str
    accession: str
    form: str
    filing_date: str
    primary_document: str
    primary_doc_description: str
    source_history_file: str
    snapshot_eligible: bool


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
    candidate_selection_trace: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_boundary: str = SNAPSHOT_BOUNDARY
    selector_version: str = STATUTORY_FILING_SELECTOR_VERSION
    source_bytes_sha256: str = ""
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatutoryFilingSelector:
    """Upstream filing-selection layer for ARX Terminal ETF research (V1.2.0)."""

    VERSION = STATUTORY_FILING_SELECTOR_VERSION
    SNAPSHOT_BOUNDARY = SNAPSHOT_BOUNDARY
    SNAPSHOT_BOUNDARY_DATE = SNAPSHOT_BOUNDARY_DATE

    # In-memory candidate history cache per CIK (Section 23: builds per CIK = 1)
    _normalized_history_cache: Dict[str, Tuple[List[NormalizedFilingRecord], str]] = {}
    _history_build_counts: Dict[str, int] = {}
    _series_directory: Optional[Dict[str, Dict[str, Any]]] = None

    SUPPLEMENT_DISQUALIFY_PATTERNS = [
        r"\bfee\s*waiver\b",
        r"feewaiver",
        r"\bliquidation\b",
        r"\bsticker\b",
        r"\bdistributor\s*change\b",
        r"\bfee\s*table\b",
        r"\bname\s*change\b",
        r"\bindex\s*reconstitution\b",
        r"\bpm\s*change\b",
        r"\breorganization\b",
        r"\breorgani\b",
        r"\bclosing\b",
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
    def get_series_directory(cls, directory_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """Load or return the authoritative SEC series accession directory (V1.3.0)."""
        if cls._series_directory is not None:
            return cls._series_directory
        path = directory_path or Path("data/research/sec_series_accession_directory_v1.json")
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cls._series_directory = json.load(f)
            except Exception:
                cls._series_directory = {}
        else:
            cls._series_directory = {}
        return cls._series_directory

    @classmethod
    def set_series_directory(cls, series_dir: Optional[Dict[str, Dict[str, Any]]]):
        """Set or override the authoritative SEC series accession directory."""
        cls._series_directory = series_dir

    @classmethod
    def reset_history_cache(cls):
        """Reset the in-memory normalized submission history cache."""
        cls._normalized_history_cache.clear()
        cls._history_build_counts.clear()
        cls._series_directory = None

    @classmethod
    def extract_months_from_text(cls, text: str) -> Set[str]:
        """Extract canonical month tokens from text."""
        words = re.findall(r"[a-zA-Z]+", (text or "").lower())
        return {MONTH_MAP[w] for w in words if w in MONTH_MAP}

    @classmethod
    def load_normalized_submission_history(
        cls,
        cik: str,
        submission_json: dict,
        submissions_dir: Optional[Path] = None,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY,
    ) -> Tuple[List[NormalizedFilingRecord], str]:
        """Load and normalize complete SEC submission history (Sections 5, 6, 7, 23).

        Merges filings.recent + all filings.files, deduplicating accessions deterministically.
        Computes deterministic cache identity including file SHAs.
        """
        cik_str = str(cik).zfill(10)
        dir_key = str(submissions_dir.resolve()) if submissions_dir else "default"
        cache_lookup_key = f"{cik_str}:{dir_key}:{snapshot_boundary}:{cls.VERSION}"

        if cache_lookup_key in cls._normalized_history_cache:
            return cls._normalized_history_cache[cache_lookup_key]

        cls._history_build_counts[cik_str] = cls._history_build_counts.get(cik_str, 0) + 1

        if submissions_dir is None:
            submissions_dir = Path("data/research/cache/sec_submissions")

        hasher = hashlib.sha256()
        hasher.update(cik_str.encode("utf-8"))
        hasher.update(cls.VERSION.encode("utf-8"))
        hasher.update(snapshot_boundary.encode("utf-8"))

        recent = submission_json.get("filings", {}).get("recent", {})
        recent_bytes = json.dumps(recent, sort_keys=True).encode("utf-8")
        hasher.update(recent_bytes)

        seen_accessions: Set[str] = set()
        normalized_records: List[NormalizedFilingRecord] = []

        # 1. Process filings.recent
        if recent:
            forms = recent.get("form", [])
            fdates = recent.get("filingDate", [])
            accs = recent.get("accessionNumber", [])
            pdocs = recent.get("primaryDocument", [])
            pdescs = recent.get("primaryDocDescription", [])
            count = len(accs)

            for i in range(count):
                acc = accs[i] if i < len(accs) else ""
                if not acc or acc in seen_accessions:
                    continue
                seen_accessions.add(acc)

                form = forms[i] if i < len(forms) else ""
                fdate = fdates[i] if i < len(fdates) else ""
                pdoc = pdocs[i] if i < len(pdocs) else ""
                pdesc = pdescs[i] if i < len(pdescs) else ""

                eligible = bool(fdate and fdate <= snapshot_boundary[:10])
                normalized_records.append(NormalizedFilingRecord(
                    cik=cik_str,
                    accession=acc,
                    form=form,
                    filing_date=fdate,
                    primary_document=pdoc,
                    primary_doc_description=pdesc,
                    source_history_file="recent",
                    snapshot_eligible=eligible,
                ))

        # 2. Process all historical files in filings.files (Section 5)
        hist_files = submission_json.get("filings", {}).get("files", [])
        for file_info in hist_files:
            fname = file_info.get("name", "")
            if not fname:
                continue

            fpath = submissions_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required historical submission file: {fpath}")

            with open(fpath, "rb") as fp:
                file_bytes = fp.read()
            hasher.update(fname.encode("utf-8"))
            hasher.update(hashlib.sha256(file_bytes).digest())

            hist_data = json.loads(file_bytes.decode("utf-8"))
            h_forms = hist_data.get("form", [])
            h_fdates = hist_data.get("filingDate", [])
            h_accs = hist_data.get("accessionNumber", [])
            h_pdocs = hist_data.get("primaryDocument", [])
            h_pdescs = hist_data.get("primaryDocDescription", [])
            h_count = len(h_accs)

            for i in range(h_count):
                acc = h_accs[i] if i < len(h_accs) else ""
                if not acc or acc in seen_accessions:
                    continue
                seen_accessions.add(acc)

                form = h_forms[i] if i < len(h_forms) else ""
                fdate = h_fdates[i] if i < len(h_fdates) else ""
                pdoc = h_pdocs[i] if i < len(h_pdocs) else ""
                pdesc = h_pdescs[i] if i < len(h_pdescs) else ""

                eligible = bool(fdate and fdate <= snapshot_boundary[:10])
                normalized_records.append(NormalizedFilingRecord(
                    cik=cik_str,
                    accession=acc,
                    form=form,
                    filing_date=fdate,
                    primary_document=pdoc,
                    primary_doc_description=pdesc,
                    source_history_file=fname,
                    snapshot_eligible=eligible,
                ))

        history_sha = hasher.hexdigest()
        cls._normalized_history_cache[cache_lookup_key] = (normalized_records, history_sha)
        return normalized_records, history_sha

    @classmethod
    def classify_document_role(
        cls,
        form: str,
        primary_document: str,
        primary_doc_description: str,
    ) -> str:
        """Classify candidate document by its statutory role (Section 7)."""
        form = (form or "").upper().strip()
        doc_lower = (primary_document or "").lower()
        desc_lower = (primary_doc_description or "").lower()
        combined_meta = f"{doc_lower} {desc_lower}"

        # 1. Summary Prospectus (497K) vs 497K Supplements
        if form == "497K":
            for pat in cls.SUPPLEMENT_DISQUALIFY_PATTERNS:
                if re.search(pat, combined_meta, re.IGNORECASE):
                    return ROLE_FEE_WAIVER_SUPPLEMENT
            if re.search(r"\bsupplement\b|\bcap\s*summary\b|\bcap\s*range\b|\bsticker\b|\bsupp\b", combined_meta, re.IGNORECASE):
                return ROLE_PROSPECTUS_SUPPLEMENT
            return ROLE_SUMMARY_PROSPECTUS

        # 2. Statement of Additional Information (SAI Part B)
        is_sai = False
        for pat in [r"\bstatement\s+of\s+additional\s+information\b", r"\bsai\b", r"-sai\b", r"_sai\b"]:
            if re.search(pat, combined_meta, re.IGNORECASE):
                is_sai = True
                break
        if is_sai:
            return ROLE_SAI_PART_B

        # 3. 497 Supplements vs Disqualified Fee/Reorganization Supplements
        if form == "497":
            for pat in cls.SUPPLEMENT_DISQUALIFY_PATTERNS:
                if re.search(pat, combined_meta, re.IGNORECASE):
                    return ROLE_FEE_WAIVER_SUPPLEMENT
            return ROLE_PROSPECTUS_SUPPLEMENT

        # 4. Base Statutory Prospectus (485BPOS, 485APOS, N-1A)
        if form in {"485BPOS", "485APOS", "N-1A", "N-1A/A", "S-6", "S-6/A"}:
            return ROLE_BASE_STATUTORY_PROSPECTUS

        return ROLE_UNKNOWN

    @classmethod
    def match_target_metadata(
        cls,
        target_series: SeriesMetadata,
        pdoc: str,
        pdesc: str
    ) -> bool:
        """Check target-specific match in metadata (Sections 8, 9, 12, 13, 14, 15, 20).

        Returns boolean indicating affirmative target relevance.
        """
        pdoc_lower = (pdoc or "").lower()
        pdesc_lower = (pdesc or "").lower()
        combined_meta = f"{pdoc_lower} {pdesc_lower}"

        # 1. Share-class safety (Section 20): reject mutual-fund share class filings for ETF targets
        target_name_lower = (target_series.legal_name or "").lower()
        is_etf_target = "etf" in target_name_lower or "shares" in target_name_lower
        if is_etf_target:
            for mf_tok in MUTUAL_FUND_CLASS_TOKENS:
                if mf_tok in pdesc_lower:
                    return False

        # 2. Exact Series ID / Class ID match
        target_sid_lower = (target_series.series_id or "").lower().strip()
        target_cid_lower = (target_series.class_id or "").lower().strip()
        if target_sid_lower and target_sid_lower in combined_meta:
            return True
        if target_cid_lower and target_cid_lower in combined_meta:
            return True

        # 3. Context-Aware Ticker Matching & Concatenated Tickers (Sections 8 & 9)
        sym = (target_series.symbol or "").lower().strip()
        if sym:
            # (a) Ticker with delimiters or context in description
            if (
                f"({sym})" in pdesc_lower
                or f" {sym} " in f" {pdesc_lower} "
                or re.search(rf"\b(?:ticker|symbol|trading\s+symbol)\s*[:\-–—]?\s*{re.escape(sym)}\b", pdesc_lower)
            ):
                return True

            # (b) Delimited ticker in document filename
            if (
                f"_{sym}." in pdoc_lower
                or f"-{sym}." in pdoc_lower
                or f"_{sym}_" in pdoc_lower
                or f"-{sym}-" in pdoc_lower
                or f"{sym}sum." in pdoc_lower
                or f"{sym}." in pdoc_lower
            ):
                return True

            # (c) Concatenated Ticker matching (Section 9)
            # For 4+ letter tickers not in generic English words, allow substring in filename stem
            if len(sym) >= 4 and sym not in NON_TICKER_FOUR_LETTER_WORDS:
                stem = pdoc_lower.rsplit(".", 1)[0]
                if sym in stem:
                    return True

        # 4. Month & Strategy Number Discrimination for Defined-Outcome / Buffer ETFs (Section 12)
        target_months = cls.extract_months_from_text(target_name_lower)
        meta_months = cls.extract_months_from_text(combined_meta)

        # Buffer number discrimination (e.g. Buffer 12 vs Buffer 20)
        target_has_12 = bool(re.search(r"\b12\b|\bbuffer\s*12\b", target_name_lower))
        target_has_20 = bool(re.search(r"\b20\b|\bbuffer\s*20\b", target_name_lower))
        meta_has_12 = bool(re.search(r"\b12\b|\bbuffer\s*12\b", combined_meta))
        meta_has_20 = bool(re.search(r"\b20\b|\bbuffer\s*20\b", combined_meta))

        if target_has_12 and meta_has_20 and not meta_has_12:
            return False
        if target_has_20 and meta_has_12 and not meta_has_20:
            return False

        if target_months:
            # If target has a month (e.g. August), candidate MUST NOT contain a conflicting month
            if meta_months and not (target_months & meta_months):
                return False
            # If candidate does not specify any month, generic strategy words alone CANNOT qualify
            if not meta_months:
                return False
            # If month matches, and strategy matches, qualify
            if target_months & meta_months:
                if any(w in combined_meta for w in ["buffer", "defined", "outcome", "pgim", "500", "100", "2000"]):
                    return True

        # 5. Distinctive Name Token Matching (Sections 13, 14, 15)
        raw_words = re.findall(r"[a-z0-9]+", target_name_lower)
        active_issuer = set(ISSUER_TOKENS)
        if target_series.trust_name:
            active_issuer |= set(re.findall(r"[a-z0-9]+", target_series.trust_name.lower()))
        substantive_words = [
            w for w in raw_words
            if len(w) > 2
            and w not in active_issuer
            and w not in GENERIC_PRODUCT_TOKENS
            and w not in MONTH_TOKENS
        ]

        if substantive_words:
            matched_words = []
            for w in substantive_words:
                stem = w.rstrip("s")
                if w in combined_meta or (len(stem) > 3 and stem in combined_meta):
                    matched_words.append(w)

            distinctive_only = [w for w in substantive_words if w not in STRATEGY_FAMILY_TOKENS]
            if distinctive_only:
                matched_distinctive = [w for w in matched_words if w in distinctive_only]
                # If target has distinctive brand/name tokens, require at least 1 distinctive match
                # and total matched count >= min(2, len(substantive_words))
                if len(matched_distinctive) >= 1 and len(matched_words) >= min(2, len(substantive_words)):
                    return True
            else:
                # Target has only strategy family words (e.g. Large Cap ETF)
                if len(matched_words) >= min(2, len(substantive_words)):
                    return True

        # 6. Normalized full legal name match (ignoring generic entity words)
        norm_name = DocumentNormalizer.normalize_name(target_series.legal_name or "")
        norm_meta = DocumentNormalizer.normalize_name(combined_meta)
        if len(norm_name) > 8 and norm_name in norm_meta:
            return True

        return False

    @classmethod
    def check_target_presence(
        cls,
        target_series: SeriesMetadata,
        text: str,
        form: str = ""
    ) -> Tuple[bool, str]:
        """Verify whether target series is genuinely present in candidate text (Sections 8, 10, 14, 17).

        Checks:
        - series_id, class_id
        - context-aware ticker / symbol
        - full normalized legal name & distinctive token sets
        Rejects candidates where target appears only in:
        - Part C / Item 28 / exhibit lists / signatures
        - SAI back-of-book tables without Fund Summary/Item 4
        - Short Form 497 supplements lacking substantive strategy sections
        """
        if not text or len(text.strip()) < 10:
            return False, "EMPTY_TEXT"

        # Short Form 497 Supplement check (Section 17):
        # 1-2 page fee waivers or stickers lack substantive strategy
        if form == "497" and len(text) < 30000:
            has_strat = bool(re.search(
                r"oef:StrategyNarrativeTextBlock|oef:RiskReturnHeading|Principal\s+Investment\s+Strateg|Investment\s+Objective\s+and\s+Principal\s+Strategies|Fund\s+Summary\b",
                text,
                re.IGNORECASE
            ))
            if not has_strat:
                return False, "SUPPLEMENT_LACKS_SUBSTANTIVE_STRATEGY"

        # Unescape HTML entities (converts &#38; -> &, &#58; -> :, &#8482; -> ™, etc.)
        clean_text = html.unescape(text)

        # Create plain text (tags stripped, normalized whitespace) for fast, robust label & ticker matching
        plain_text = re.sub(r"<[^>]+>", " ", clean_text)
        plain_text = re.sub(r"\s+", " ", plain_text)

        sid = (target_series.series_id or "").upper().strip()
        cid = (target_series.class_id or "").upper().strip()
        raw_name = (target_series.legal_name or "").strip()
        sym = (target_series.symbol or "").upper().strip()

        # Cross-Series Contradiction Rule (Section 9)
        # If candidate is a single-fund statutory document (497K summary prospectus or short filing)
        # and explicitly declares exact Series IDs, but target series ID is not among them:
        if sid and (form == "497K" or len(clean_text) < 120000):
            found_sids = set(re.findall(r"\bS0000\d{5}\b", clean_text, re.IGNORECASE))
            if found_sids:
                upper_sids = {s.upper() for s in found_sids}
                if sid not in upper_sids:
                    return False, f"CONFLICTING_EXACT_SERIES_ID (Document series {sorted(upper_sids)} != target {sid})"

        # 1. Series ID & Class ID
        has_sid = bool(sid and sid in clean_text)
        has_cid = bool(cid and cid in clean_text)
        if has_sid or has_cid:
            return cls._validate_sai_boundary(clean_text, sid, cid, sym, raw_name, has_sid, has_cid, form=form)

        # 2. Context-Aware Ticker Evidence (Sections 8 & 10)
        has_ticker = False
        if sym and len(sym) >= 2:
            ticker_patterns = [
                rf"\({re.escape(sym)}\)",
                rf"\b(?:Ticker\s+Symbol|Trading\s+Symbol|Ticker|Symbol|NASDAQ|NYSE(?:\s+Arca)?|Cboe(?:\s+BZX)?)\s*[:\-–—]?\s*{re.escape(sym)}\b",
                rf"\b(?:Shares|Class|ETF)\s*\(?{re.escape(sym)}\b",
            ]
            for pat in ticker_patterns:
                if len(sym) <= 3:
                    if re.search(pat, plain_text):
                        has_ticker = True
                        break
                else:
                    if re.search(pat, plain_text, re.IGNORECASE):
                        has_ticker = True
                        break

            if not has_ticker and len(sym) >= 4 and sym not in NON_TICKER_FOUR_LETTER_WORDS:
                if re.search(rf"\b{re.escape(sym)}\b\s*Exchange", plain_text, re.IGNORECASE):
                    has_ticker = True

        # 3. Name Evidence with Plain Text Matching (Section 14)
        has_name = False
        if raw_name and len(raw_name) > 5:
            if raw_name.lower() in plain_text.lower():
                has_name = True
            else:
                words = [w for w in re.split(r"[\s&–—\-,]+", raw_name) if len(w) > 2]
                if len(words) >= 2:
                    pat = r"\s+(?:&|and|[–—\-])?\s*".join(re.escape(w) for w in words)
                    if re.search(pat, plain_text, re.IGNORECASE):
                        has_name = True

                if not has_name:
                    # Check distinctive tokens (ignoring generic issuer tokens)
                    distinctive_words = [
                        w for w in words
                        if w.lower() not in ISSUER_TOKENS
                        and w.lower() not in GENERIC_PRODUCT_TOKENS
                        and w.lower() not in STRATEGY_FAMILY_TOKENS
                    ]
                    if len(distinctive_words) >= 2:
                        pat_disc = r"\s+(?:&|and|[–—\-])?\s*".join(re.escape(w) for w in distinctive_words)
                        if re.search(pat_disc, plain_text, re.IGNORECASE):
                            has_name = True

        if not has_sid and not has_cid and not has_ticker and not has_name:
            return False, "TARGET_NOT_FOUND_IN_TEXT"

        return cls._validate_sai_boundary(clean_text, sid, cid, sym, raw_name, has_sid, has_cid, form=form)

    @classmethod
    def _validate_sai_boundary(
        cls,
        text: str,
        sid: str,
        cid: str,
        sym: str,
        raw_name: str,
        has_sid: bool,
        has_cid: bool,
        form: str = ""
    ) -> Tuple[bool, str]:
        """Check if target presence occurs only after an SAI boundary or Part C boundary."""
        # Summary prospectuses (Form 497K) never contain an SAI or Part C
        if form == "497K":
            return True, "TARGET_PRESENT"

        first_occ = len(text)
        if has_sid:
            idx = text.find(sid)
            if idx >= 0:
                first_occ = min(first_occ, idx)
        if has_cid:
            idx = text.find(cid)
            if idx >= 0:
                first_occ = min(first_occ, idx)
        if sym and len(sym) >= 2:
            m_tick = re.search(
                rf"\({re.escape(sym)}\)|\b(?:Ticker\s+Symbol|Trading\s+Symbol|Ticker|Symbol)\s*[:\-–—]?\s*{re.escape(sym)}\b",
                text,
                re.IGNORECASE if len(sym) >= 4 else 0,
            )
            if m_tick:
                first_occ = min(first_occ, m_tick.start())
            elif len(sym) >= 4 and sym not in NON_TICKER_FOUR_LETTER_WORDS and text.find(sym) >= 0:
                first_occ = min(first_occ, text.find(sym))
        if raw_name:
            idx_name = text.lower().find(raw_name.lower())
            if idx_name >= 0:
                first_occ = min(first_occ, idx_name)
            else:
                words = [w for w in re.split(r"[\s&–—\-,]+", raw_name) if len(w) > 2]
                distinctive_words = [
                    w for w in words
                    if w.lower() not in ISSUER_TOKENS
                    and w.lower() not in GENERIC_PRODUCT_TOKENS
                    and w.lower() not in STRATEGY_FAMILY_TOKENS
                ]
                if len(distinctive_words) >= 2:
                    pat_disc = r"\s+(?:&|and|[–—\-])?\s*".join(re.escape(w) for w in distinctive_words)
                    m_disc = re.search(pat_disc, text, re.IGNORECASE)
                    if m_disc:
                        first_occ = min(first_occ, m_disc.start())
                if len(words) >= 2 and first_occ == len(text):
                    pat_w = r"\s+(?:&|and|[–—\-])?\s*".join(re.escape(w) for w in words[:3])
                    m_w = re.search(pat_w, text, re.IGNORECASE)
                    if m_w:
                        first_occ = min(first_occ, m_w.start())

        # 1. Part C Boundary Check (Item 28 Exhibits / Signatures / Other Information)
        m_part_c = re.search(
            r"<h[1-4][^>]*>[^<]*Part\s+C\b|<(?:b|strong|p|div)[^>]*align=['\"]center['\"][^>]*>[^<]*Part\s+C\b|<(?:b|strong|p|div)[^>]*>[^<]*PART\s+C\b|\bPART\s+C\b\s*[-–—]?\s*(?:OTHER\s+INFORMATION|OTHER|REGISTRATION)|\bItem\s+28\.\s*Exhibits|\bItem\s+28\b",
            text,
            re.IGNORECASE,
        )
        if m_part_c:
            part_c_start = m_part_c.start()
            if first_occ > part_c_start:
                return False, "TARGET_ONLY_IN_PART_C_OR_ANCILLARY"

        # 2. SAI Boundary Check
        m_sai = re.search(
            r"<h[1-4][^>]*>[^<]*Statement\s+of\s+Additional\s+Information|<(?:b|strong|p|div)[^>]*align=['\"]center['\"][^>]*>[^<]*Statement\s+of\s+Additional\s+Information|\bPart\s+B\b",
            text,
            re.IGNORECASE,
        )
        if m_sai and m_sai.start() > 5000:
            sai_start = m_sai.start()
            if first_occ > sai_start:
                has_summary_after_sai = bool(
                    re.search(r"\bFund\s+Summary\b|oef:RiskReturnHeading|oef:StrategyNarrativeTextBlock",
                              text[max(0, first_occ - 500):first_occ + 2000], re.IGNORECASE)
                )
                if not has_summary_after_sai:
                    return False, "TARGET_ONLY_IN_SAI_SECTION"

        return True, "TARGET_PRESENT"

    @classmethod
    def check_mandate_content(cls, text: str, form: str = "") -> Tuple[bool, str]:
        """Verify whether statutory mandate / strategy section exists in text (Section 9 & V1.4.0).

        Rejects partial supplement amendments (e.g. 'replaces the disclosure in the section titled...')
        that lack complete statutory Principal Investment Strategies sections.
        """
        if not text:
            return False, "NO_TEXT"

        is_supp = bool(re.search(
            r"\bsupplement\s+dated\b|\bsupplement\s+to\s+(?:the\s+)?(?:summary\s+)?prospectus\b|\bplease\s+retain\s+this\s+supplement\b",
            text, re.IGNORECASE
        ))

        # Check for XBRL narrative text blocks which are always authoritative
        if re.search(r"oef:StrategyNarrativeTextBlock", text, re.IGNORECASE):
            return True, "MANDATE_PRESENT_oef:StrategyNarrativeTextBlock"

        matched_pat = None
        for pat in cls.MANDATE_SECTION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # Context check: ignore references that are merely amendment instructions
                pre = text[max(0, m.start() - 100): m.start()].lower()
                if re.search(r"\b(?:replaces|amends|amended|section\s+titled|heading\s+titled|under\s+the\s+heading)\b", pre):
                    continue
                matched_pat = pat
                break
            if matched_pat:
                break

        if not matched_pat:
            return False, "MANDATE_SECTION_NOT_FOUND"

        # If document is an abbreviated supplement (< 35KB) that merely amends part of the strategy
        if is_supp and len(text) < 35000:
            if re.search(r"\b(?:replaces\s+the\s+disclosure|following\s+is\s+added|amended\s+as\s+follows|the\s+following\s+replaces)\b", text, re.IGNORECASE):
                return False, "SUPPLEMENT_PARTIAL_AMENDMENT_NOT_FULL_MANDATE"

        return True, f"MANDATE_PRESENT_{matched_pat}"

    @classmethod
    def compute_cache_key(
        cls,
        target: SeriesMetadata,
        cik: str,
        config: Optional[Dict[str, Any]] = None,
        snapshot_boundary: str = SNAPSHOT_BOUNDARY,
        history_sha: str = "",
    ) -> str:
        """Compute deterministic selection cache identity (Sections 7 & 18)."""
        sid = target.series_id or ""
        cid = target.class_id or ""
        norm_name = DocumentNormalizer.normalize_name(target.legal_name or "")
        name_sha = hashlib.sha256(norm_name.encode("utf-8")).hexdigest()
        cfg_sha = hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode("utf-8")).hexdigest()
        raw = f"{sid}:{cid}:{name_sha}:{cik}:{snapshot_boundary}:{cls.VERSION}:{cfg_sha}:{history_sha}"
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
        """Deterministically selects the correct pre-boundary statutory filing for target_series (V1.2.0).

        Steps:
        1. Load normalized filing history (recent + historical submissions).
        2. Filter strictly by filingDate <= snapshot_boundary.
        3. Classify document roles (BASE_STATUTORY_PROSPECTUS, SUMMARY_PROSPECTUS, etc.).
        4. Prioritize candidates by target relevance and form hierarchy.
        5. For cached documents, verify target presence and mandate section.
        6. Fail closed with informative outcome if target is absent or ambiguous.
        """
        cik = str(target_series.cik or "").zfill(10)
        submissions_dir = cache_dir / "sec_submissions" if cache_dir else Path("data/research/cache/sec_submissions")
        prospectus_dir = cache_dir / "sec_prospectus" if cache_dir else Path("data/research/cache/sec_prospectus")

        if cached_filenames is None:
            cached_filenames = {p.name for p in prospectus_dir.iterdir()} if prospectus_dir.exists() else set()

        # Step 1: Load complete normalized filing history
        try:
            records, history_sha = cls.load_normalized_submission_history(
                cik=cik,
                submission_json=submission_json,
                submissions_dir=submissions_dir,
                snapshot_boundary=snapshot_boundary,
            )
        except Exception as e:
            cache_key = cls.compute_cache_key(target_series, cik, config, snapshot_boundary)
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
                selection_rule_id="RULE_HISTORY_LOAD_ERROR",
                selection_evidence=f"Failed to load historical submission records for CIK {cik}: {str(e)}",
                candidate_count=0,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
            )

        cache_key = cls.compute_cache_key(target_series, cik, config, snapshot_boundary, history_sha)

        if not records:
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
                selection_evidence="No filings found in CIK submissions metadata",
                candidate_count=0,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
            )

        # Step 2: Enumerate pre-boundary candidate filings
        series_dir = cls.get_series_directory()
        target_sid = (target_series.series_id or "").strip()
        target_dir_entry = series_dir.get(target_sid)

        max_preboundary_year = 0
        for r in records:
            if r.snapshot_eligible and r.filing_date and len(r.filing_date) >= 4:
                try:
                    yr = int(r.filing_date[:4])
                    if yr > max_preboundary_year:
                        max_preboundary_year = yr
                except ValueError:
                    pass

        candidates: List[FilingCandidate] = []
        rejected_candidates: List[Dict[str, Any]] = []

        for rec in records:
            # Strict temporal boundary check (Section 32: POST_BOUNDARY_SELECTED = 0)
            if not rec.snapshot_eligible:
                rejected_candidates.append({
                    "accession": rec.accession,
                    "form": rec.form,
                    "filing_date": rec.filing_date,
                    "rejection_reason": f"POST_BOUNDARY_FILING (filingDate {rec.filing_date} > {snapshot_boundary[:10]})"
                })
                continue

            if rec.form not in STATUTORY_FORMS:
                continue

            role = cls.classify_document_role(rec.form, rec.primary_document, rec.primary_doc_description)
            if role in {ROLE_SAI_PART_B, ROLE_FEE_WAIVER_SUPPLEMENT, ROLE_NON_MANDATE_DOCUMENT}:
                rejected_candidates.append({
                    "accession": rec.accession,
                    "form": rec.form,
                    "filing_date": rec.filing_date,
                    "primary_doc": rec.primary_document,
                    "rejection_reason": f"DISQUALIFIED_DOCUMENT_ROLE ({role})"
                })
                continue

            # Check target-specific match in metadata
            meta_match = cls.match_target_metadata(
                target_series, rec.primary_document, rec.primary_doc_description
            )

            # Fast in-memory check if file is cached locally
            cached_filename = f"{rec.accession}_{rec.primary_document}"
            is_cached = cached_filename in cached_filenames

            # Priority scoring (Section 21 & V1.3.0 remediation)
            score = 0
            is_dir_accession_match = bool(target_dir_entry and rec.accession == target_dir_entry.get("accession"))
            if is_dir_accession_match:
                score += 2000
                if target_dir_entry.get("primary_document") == rec.primary_document:
                    score += 500

            if meta_match:
                score += 500
            if role == ROLE_SUMMARY_PROSPECTUS:
                score += 300 if meta_match else 50
            elif role == ROLE_BASE_STATUTORY_PROSPECTUS:
                score += 250
            elif role == ROLE_PROSPECTUS_SUPPLEMENT:
                score += 20 if meta_match else 5

            # V1.3.0 Ancient Filing Penalty (-1000 points)
            # If registrant has modern pre-boundary filings (>= 2015), penalize ancient filings (< 2010)
            if max_preboundary_year >= 2015 and rec.filing_date and len(rec.filing_date) >= 4:
                try:
                    rec_year = int(rec.filing_date[:4])
                    if rec_year < 2010:
                        score -= 1000
                except ValueError:
                    pass

            candidates.append(FilingCandidate(
                accession=rec.accession,
                form=rec.form,
                filing_date=rec.filing_date,
                primary_document=rec.primary_document,
                primary_doc_description=rec.primary_doc_description,
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

        # Sort candidates deterministically: Directory Match desc, Priority Score desc, Filing Date desc, Form Priority desc (Section 10)
        form_weight = {"485BPOS": 10, "497K": 9, "485APOS": 8, "497": 5, "N-1A": 4}
        candidates.sort(
            key=lambda c: (
                1 if (target_dir_entry and c.accession == target_dir_entry.get("accession")) else 0,
                c.priority_score,
                c.filing_date,
                form_weight.get(c.form, 0)
            ),
            reverse=True
        )

        # Step 3: Multi-Accession Search & Content Qualification (Sections 8, 11, 16)
        inspected_count = 0
        cache_miss_candidate: Optional[FilingCandidate] = None
        candidate_selection_trace: List[Dict[str, Any]] = []

        for cand in candidates:
            inspected_count += 1
            cached_filename = f"{cand.accession}_{cand.primary_document}"
            cached_path = prospectus_dir / cached_filename

            if not cand.is_cached:
                # Document is not cached locally
                # Only affirmative target relevance triggers SOURCE_CACHE_MISS (Section 16)
                is_dir_match = bool(target_dir_entry and cand.accession == target_dir_entry.get("accession"))
                if (cand.target_metadata_match or is_dir_match) and cache_miss_candidate is None:
                    cache_miss_candidate = cand

                rejected_candidates.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_doc": cand.primary_document,
                    "rejection_reason": "SOURCE_NOT_CACHED_LOCALLY"
                })
                candidate_selection_trace.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_document": cand.primary_document,
                    "priority_score": cand.priority_score,
                    "target_present": "NO",
                    "mandate_present": "UNKNOWN",
                    "rejection_reason": "SOURCE_NOT_CACHED_LOCALLY",
                    "selected": False,
                })
                continue

            # Read cached content (leveraging in-memory cache if provided)
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
                    candidate_selection_trace.append({
                        "accession": cand.accession,
                        "form": cand.form,
                        "filing_date": cand.filing_date,
                        "primary_document": cand.primary_document,
                        "priority_score": cand.priority_score,
                        "target_present": "NO",
                        "mandate_present": "UNKNOWN",
                        "rejection_reason": f"CACHE_READ_ERROR: {str(e)}",
                        "selected": False,
                    })
                    continue

            # Check target presence
            target_present, t_reason = cls.check_target_presence(target_series, content, form=cand.form)
            if not target_present:
                cand.target_present = "NO"
                cand.rejection_reason = t_reason
                rej_desc = t_reason if t_reason.startswith("CONFLICTING_EXACT_SERIES_ID") else f"TARGET_NOT_PRESENT ({t_reason})"
                rejected_candidates.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_doc": cand.primary_document,
                    "rejection_reason": rej_desc
                })
                candidate_selection_trace.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_document": cand.primary_document,
                    "priority_score": cand.priority_score,
                    "target_present": "NO",
                    "mandate_present": "UNKNOWN",
                    "rejection_reason": rej_desc,
                    "selected": False,
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
                candidate_selection_trace.append({
                    "accession": cand.accession,
                    "form": cand.form,
                    "filing_date": cand.filing_date,
                    "primary_document": cand.primary_document,
                    "priority_score": cand.priority_score,
                    "target_present": "YES",
                    "mandate_present": "NO",
                    "rejection_reason": f"MANDATE_NOT_PRESENT ({m_reason})",
                    "selected": False,
                })
                continue

            cand.mandate_present = "YES"
            candidate_selection_trace.append({
                "accession": cand.accession,
                "form": cand.form,
                "filing_date": cand.filing_date,
                "primary_document": cand.primary_document,
                "priority_score": cand.priority_score,
                "target_present": "YES",
                "mandate_present": "YES",
                "rejection_reason": "",
                "selected": True,
            })

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
                candidate_selection_trace=candidate_selection_trace,
                snapshot_boundary=snapshot_boundary,
                cache_key=cache_key,
                source_bytes_sha256=doc_sha,
            )

        # Step 4: Exhaustive search complete without qualifying cached match
        if cache_miss_candidate is not None:
            # Candidate known from SEC submissions metadata with affirmative target match but not yet cached
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
                candidate_selection_trace=candidate_selection_trace,
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
            candidate_selection_trace=candidate_selection_trace,
            snapshot_boundary=snapshot_boundary,
            cache_key=cache_key,
        )
