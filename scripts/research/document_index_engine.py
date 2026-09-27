"""ARX Terminal — Document Indexing Engine (Stage A).

Production implementation of Stage A: Document Indexing for multi-series statutory prospectuses.

Enforces:
1. Input: original acquired document bytes, document identity, metadata snapshot.
2. Output: deterministic DocumentIndex with zero mandate/subtype classification logic.
3. Hashes preserved:
   - SOURCE_BYTES_SHA256 (hash of original acquired raw bytes)
   - NORMALIZED_TEXT_SHA256 (hash of normalized representation)
   - DOCUMENT_INDEX_SHA256 (hash of serialized deterministic index)
4. Versioning:
   - INDEX_ENGINE_VERSION = DOC_INDEX_V1_0_0
   - NORMALIZATION_VERSION = NORMALIZATION_V1_0_0
   - INDEX_SCHEMA_VERSION = SCHEMA_V1_0_0
5. Structural detection:
   - Series ID occurrences (S\\d{9})
   - Class ID occurrences (C\\d{9})
   - Fund summary markers, delimiters, <hr>, headings
   - Strategy section headings
   - Table of contents (TOC) / cross-reference detection & exclusion
"""

import re
import html
import unicodedata
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Set, Tuple, Any


INDEX_ENGINE_VERSION = "DOC_INDEX_V1_3_1"
NORMALIZATION_VERSION = "NORMALIZATION_V1_3_1"
INDEX_SCHEMA_VERSION = "SCHEMA_V1_3_1"


@dataclass
class DocumentIdentity:
    """Legal provenance and source identity for an acquired SEC document."""
    cik: str
    accession: str
    form: str
    filing_date: str
    effective_date: str = ""
    document_filename: str = ""
    source_url: str = ""
    source_byte_length: int = 0
    source_bytes_sha256: str = ""


@dataclass
class Occurrence:
    """An identity occurrence within a document."""
    matched_term: str
    term_type: str  # SERIES_ID, CLASS_ID, LEGAL_NAME, NORMALIZED_NAME
    start_offset: int
    end_offset: int
    is_toc_or_cross_ref: bool = False
    context_snippet: str = ""


@dataclass
class SectionBoundaryCandidate:
    """A structural boundary candidate within a document."""
    start_offset: int
    end_offset: int
    boundary_type: str  # HR_DELIMITER, FUND_SUMMARY_DIV, HEADING, NEXT_FUND_START
    delimiter_text: str = ""


@dataclass
class StrategyAnchor:
    """Location of a Principal Investment Strategies section."""
    start_offset: int
    heading_name: str
    length: int


class DocumentNormalizer:
    """Versioned, deterministic document normalizer (NORMALIZATION_V1_2_0)."""

    VERSION = NORMALIZATION_VERSION

    # Bounded financial abbreviations derived from audited SEC statutory corpus
    FINANCIAL_ABBREVIATIONS = {
        "tech": "technology",
        "technology": "technology",
        "discretion": "discretionary",
        "discretionary": "discretionary",
        "div": "dividend",
        "dividend": "dividend",
        "divs": "dividends",
        "dividends": "dividends",
        "smcp": "small cap",
        "smcap": "small cap",
        "idx": "index",
        "index": "index",
        "wld": "world",
        "world": "world",
        "eq": "equity",
        "equity": "equity",
        "intl": "international",
        "international": "international",
        "corp": "corporate",
        "corporat": "corporate",
        "corporate": "corporate",
        "govt": "government",
        "government": "government",
        "ftseeuropean": "ftse european",
        "adrhedged": "adr hedged",
        "market": "markets",
        "markets": "markets",
    }

    @classmethod
    def get_word_equivalence_pattern(cls, word: str) -> str:
        """Returns bounded regex pattern for financial equivalence variants."""
        w_low = word.lower()
        if w_low in ("tech", "technology"):
            return r"(?:technolog(?:y|ies)|tech)"
        if w_low in ("discretion", "discretionary"):
            return r"(?:discretionar(?:y|ies)|discretion)"
        if w_low in ("div", "dividend"):
            return r"(?:dividend|div)"
        if w_low in ("divs", "dividends"):
            return r"(?:dividends|divs)"
        if w_low in ("smcp", "smcap", "small-cap", "smallcap"):
            return r"(?:small[- ]?cap|sm[- ]?cap|smcap|smcp)"
        if w_low in ("idx", "index"):
            return r"(?:index|idx)"
        if w_low in ("wld", "world"):
            return r"(?:world|wld)"
        if w_low in ("eq", "equity"):
            return r"(?:equit(?:y|ies)|eq)"
        if w_low in ("intl", "international"):
            return r"(?:international|intl)"
        if w_low in ("corp", "corporat", "corporate"):
            return r"(?:corporat(?:e|ion)?|corp)"
        if w_low in ("govt", "government"):
            return r"(?:government|govt)"
        if w_low == "ftseeuropean":
            return r"(?:ftse\s+european|ftseeuropean)"
        if w_low == "adrhedged":
            return r"(?:adr\s+hedged|adrhedged)"
        if w_low in ("market", "markets"):
            return r"(?:markets?)"
        if w_low in ("etf", "fund"):
            # ETF <-> Fund equivalence: resolves DOCUMENT_INDEX_DEFECT where manifest
            # legal_name carries 'Growth Fund' but source filing says 'Growth ETF'
            # or vice versa (IWF, IWV, IWO; ETF_INDEX_FUND_SUFFIX alias authority cases).
            return r"(?:ETF|Fund)"
        if w_low in ("&", "and", "&amp;"):
            return r"(?:&|&amp;|and)"
        if "-" in word:
            parts = [re.escape(p) for p in word.split("-")]
            return r"[- ]?".join(parts)
        return re.escape(word)

    @classmethod
    def normalize_html_to_text(cls, raw_html: str) -> Tuple[str, str]:
        """Normalizes raw HTML while recording whitespace, entity unescaping, and unicode folding.

        Returns: (normalized_text, normalized_text_sha256)
        """
        if not raw_html:
            return "", hashlib.sha256(b"").hexdigest()

        # Step 1: Unicode NFKC normalization
        norm = unicodedata.normalize("NFKC", raw_html)

        # Step 2: HTML entity unescaping
        norm = html.unescape(norm)
        norm = norm.replace("\xa0", " ")
        norm = norm.replace("&nbsp;", " ")
        norm = norm.replace("&amp;", "&")
        norm = norm.replace("&#160;", " ")
        norm = norm.replace("&#8212;", "—")
        norm = norm.replace("&#8217;", "'")
        norm = norm.replace("&rsquo;", "'")
        norm = norm.replace("&ldquo;", '"')
        norm = norm.replace("&rdquo;", '"')

        # Step 3: Whitespace normalization (collapse multi-spaces while preserving line break structure)
        norm = re.sub(r"[ \t]+", " ", norm)
        norm = re.sub(r"[\r\n]+", "\n", norm)

        norm_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return norm, norm_sha

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """Standardized legal fund name normalization with bounded financial equivalences."""
        if not name:
            return ""
        norm = unicodedata.normalize("NFKC", name)
        norm = html.unescape(norm)
        norm = norm.lower()
        norm = re.sub(r"[^\w\s]", " ", norm)
        tokens = norm.split()
        normalized_tokens = []
        for t in tokens:
            if t in cls.FINANCIAL_ABBREVIATIONS:
                normalized_tokens.append(cls.FINANCIAL_ABBREVIATIONS[t])
            else:
                normalized_tokens.append(t)
        norm = " ".join(normalized_tokens)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm


class DocumentIndex:
    """Deterministic structural index for a single SEC statutory filing."""

    # V1.3.0: expanded to capture SPDR "Investment Objective", Tidal "Investment Goal",
    # and Schwab/Fidelity "Principal Investment Strategy" (singular) variants.
    # Ordered by specificity — more specific multi-word patterns first to avoid
    # "Investment Objective and Principal Strategies" being consumed by the shorter
    # "Investment Objective" pattern in the wrong order.
    STRATEGY_PATTERNS = [
        # --- V1.2.0 patterns (preserved, reordered for specificity) ---
        r"Principal\s+Investment\s+Strateg(?:y|ies)",
        r"Principal\s+Investment\s+Policies\s+and\s+Strategies",
        r"Principal\s+Risks\s+and\s+Strategies",
        # --- V1.3.0 additions ---
        # "Investment Objective and Principal Strategies" (SPDR combo heading)
        r"Investment\s+Objective\s+and\s+Principal\s+(?:Investment\s+)?Strateg(?:y|ies)",
        # "Investment Objective" standalone — SPDR/Schwab table-embedded heading (52 targets)
        r"Investment\s+Objective",
        # "The Fund's Investment Goal" / "Investment Goal" — Tidal/Touchstone (7 targets)
        # V1.3.1: Extended quote char class to [\u2019\u2018\u201c\u201d\u0094\u0093] to cover
        # Windows-1252 byte 0x94 (right double quotation mark, used as apostrophe in Touchstone
        # 497K filings after NFKC normalization leaves C1 control as-is).
        r"The\s+Fund[\u2019\u2018\u201c\u201d\u0094\u0093']?s?\s+Investment\s+Goal",
        r"Investment\s+Goal",
        # "Principal Strategies" standalone (some SPDR combined forms)
        r"Principal\s+Strategies",
        # "Investment Strategy" (singular, Fidelity Schwab variants)
        r"Investment\s+Strategy(?:\s+and\s+Policy)?",
    ]

    DELIMITER_PATTERNS = [
        (r"<hr\s*/?>", "HR_DELIMITER"),
        (r"<div[^>]*class=[\"'][^\"']*(?:fund-summary|fund-section|summary|summary-section)[\"']", "FUND_SUMMARY_DIV"),
        (r"<h[1-4][^>]*>\s*(?:Fund\s+Summary|Summary\s+Prospectus|Fund\s+Overview|SUMMARY\s+SECTION)\b", "FUND_SUMMARY_HEADING"),
        (r"\bFund\s+Summary\b", "TEXT_FUND_SUMMARY"),
        (r"\bSummary\s+Prospectus\b", "TEXT_SUMMARY_PROSPECTUS"),
        (r"\bSUMMARY\s+SECTION\b", "TEXT_SUMMARY_SECTION"),
        # V1.3.0: iShares combined prospectus format (9 BOUNDARY_NOT_ESTABLISHED targets:
        # FLTB/MBBB/MIG/EQL/IGIB/IYC/IYK/EQIN/TBIL). These documents use zero <hr> delimiters
        # and instead demarcate fund groups with a "Fund Group:" header in a bold table cell.
        # Matching directly on "Fund Group:" is safe — this token is only used as a
        # section header in combined trust 485BPOS documents, never in TOC rows.
        (r"\bFund\s+Group\s*:", "FUND_GROUP_TABLE"),
    ]

    TOC_PATTERNS = [
        r"Table\s+of\s+Contents",
        r"Index\s+to\s+(?:Summary\s+)?Prospectus",
        r"Index\s+to\s+Financial\s+Statements",
        r"\bTABLE\s+OF\s+CONTENTS\b",
        r"<table[^>]*class=[\"'][^\"']*toc[\"']",
        r"<div[^>]*class=[\"'][^\"']*toc[\"']",
    ]

    def __init__(
        self,
        identity: DocumentIdentity,
        raw_bytes: bytes,
        known_series_metadata: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
        alias_legal_names: Optional[List[str]] = None,
    ):
        """Build a deterministic DocumentIndex.

        Args:
            identity: Document provenance.
            raw_bytes: Original acquired document bytes (never mutated).
            known_series_metadata: List of dicts, each with at least 'legal_name' key.
            config: Optional configuration overrides (reserved, currently unused).
            alias_legal_names: V1.3.0 — Optional list of historical legal-name strings from
                ETF_HISTORICAL_IDENTITY_ALIAS_AUTHORITY_V1.  These are indexed as ALIAS_NAME
                occurrences to supplement LEGAL_NAME matching for 13 MANIFEST_IDENTITY_DEFECT
                targets.  The manifest is NOT mutated; aliases are a supplementary lookup path.
        """
        self.identity = identity
        self.engine_version = INDEX_ENGINE_VERSION
        self.normalization_version = NORMALIZATION_VERSION
        self.schema_version = INDEX_SCHEMA_VERSION

        # Step 1: Hash original raw acquired bytes
        self.identity.source_byte_length = len(raw_bytes)
        self.identity.source_bytes_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        # Step 2: Decode and normalize text
        raw_text = raw_bytes.decode("utf-8", errors="ignore")
        self.normalized_text, self.normalized_text_sha256 = DocumentNormalizer.normalize_html_to_text(raw_text)

        # Step 3: Index structural components
        self.series_occurrences: Dict[str, List[Occurrence]] = {}
        self.class_occurrences: Dict[str, List[Occurrence]] = {}
        self.legal_name_occurrences: Dict[str, List[Occurrence]] = {}
        self.normalized_name_occurrences: Dict[str, List[Occurrence]] = {}
        self.alias_name_occurrences: Dict[str, List[Occurrence]] = {}  # V1.3.0
        self.strategy_anchors: List[StrategyAnchor] = []
        self.boundary_candidates: List[SectionBoundaryCandidate] = []
        self.toc_ranges: List[Tuple[int, int]] = []

        self._build_toc_ranges(self.normalized_text)
        self._build_strategy_anchors(self.normalized_text)
        self._build_boundary_candidates(self.normalized_text)
        self._index_known_metadata(self.normalized_text, known_series_metadata or [], alias_legal_names or [])

        # Step 4: Serialize deterministic index representation and hash
        self.index_dict = self._to_deterministic_dict()
        serialized_bytes = json.dumps(self.index_dict, sort_keys=True).encode("utf-8")
        self.document_index_sha256 = hashlib.sha256(serialized_bytes).hexdigest()

        # Step 5: Compute cache key
        meta_sha = hashlib.sha256(
            json.dumps(known_series_metadata or [], sort_keys=True).encode("utf-8")
        ).hexdigest()
        cfg_sha = hashlib.sha256(
            json.dumps(config or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_seed = f"{self.identity.source_bytes_sha256}:{self.engine_version}:{self.normalization_version}:{self.schema_version}:{meta_sha}:{cfg_sha}"
        self.cache_key = hashlib.sha256(cache_seed.encode("utf-8")).hexdigest()

    def _build_toc_ranges(self, text: str):
        """Identifies Table of Contents / Index regions to reject TOC false positives (Section 15)."""
        for pat in self.TOC_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                start = max(0, m.start() - 50)
                rest = text[m.end(): m.end() + 10000]
                table_end = rest.lower().find("</table>")
                div_end = rest.lower().find("</div>")
                if table_end != -1 and ("<table" in text[start:m.end()] or "<table" in rest[:table_end]):
                    end = m.end() + table_end + 8
                elif div_end != -1 and "<div" in text[start:m.end()]:
                    end = m.end() + div_end + 6
                else:
                    h_match = re.search(r"<h[1-3][^>]*>(?:(?!table\s+of\s+contents).)*?</h[1-3]>|\bFund\s+Summary\b", rest, re.IGNORECASE)
                    if h_match and h_match.start() > 100:
                        end = m.end() + h_match.start()
                    else:
                        end = min(len(text), m.end() + 1500)
                self.toc_ranges.append((start, end))

    def _is_in_toc(self, offset: int) -> bool:
        for start, end in self.toc_ranges:
            if start <= offset <= end:
                return True
        return False

    def _is_structural_heading(self, text: str, m_start: int, m_end: int, m_text: str) -> bool:
        """Determines whether a strategy heading match is a genuine structural heading rather than prose.

        V1.3.0: Added table-cell boundary detection. When the 120-char pre-context contains
        an opening <td or <th tag, the match is inside a table cell and qualifies as a
        structural heading. This resolves 52 SPDR/Schwab/iShares table-embedded targets.
        """
        # 1. Reject all-lowercase prose matches that continue into lowercase narrative sentences (Defect E / INVN)
        post = text[m_end: m_end + 30]
        if m_text.islower() and re.match(r"^\s+[a-z]", post):
            return False

        pre = text[max(0, m_start - 120): m_start]
        last_boundary = max(pre.rfind('>'), pre.rfind('\n'))
        if last_boundary != -1:
            prefix_text = pre[last_boundary + 1:].strip()
        else:
            prefix_text = pre.strip()

        # V1.3.0: Table-cell heading detection.
        # If the pre-context contains an opening <td or <th tag, this heading sits inside
        # a table cell. SPDR/Schwab/Schwab prospectuses format fund sections as table rows
        # where each cell starts with the strategy heading. Treat as structural.
        if re.search(r"<t[dh][\s>]", pre, re.IGNORECASE):
            # Still reject if there is substantive prose between the last <td/<th and the heading
            td_start = max(
                (m.start() for m in re.finditer(r"<t[dh][\s>]", pre, re.IGNORECASE)),
                default=-1,
            )
            if td_start != -1:
                cell_pre = pre[td_start:].lstrip()[3:]  # skip tag opener itself
                # Skip any HTML attributes / close angle bracket of the opening tag
                attr_end = cell_pre.find('>')
                if attr_end != -1:
                    cell_pre = cell_pre[attr_end + 1:].strip()
                # If what remains is only whitespace, nested tags, or empty -> heading is first content
                stripped_cell_pre = re.sub(r"<[^>]+>", "", cell_pre).strip()
                if not stripped_cell_pre or len(stripped_cell_pre) <= 10:
                    return True

        # If prefix_text is empty, it started immediately after a tag or newline -> Structural Heading!
        if not prefix_text:
            return True

        # If prefix_text is a section or item number (e.g. 'Item 4.', '4.', 'Section 2.', 'A.') -> Heading!
        if re.match(r'^(?:item\s+\d+\.?|\d+\.?|[A-Z]\.?|\*|\u2022)\s*$', prefix_text, re.IGNORECASE):
            return True

        # Supplement / amendment headings (e.g. 'under the heading "', 'entitled "')
        if re.search(r'\b(?:heading|caption|section|entitled)\b', prefix_text, re.IGNORECASE):
            return True

        # Punctuation / quote boundary
        if prefix_text and prefix_text[-1] in ('"', "'", ":", ".", ";", "-", "—"):
            return True

        # If prefix_text contains narrative words like 'with the fund\'s', reject as mid-sentence
        if re.search(r'\b(?:the|its|our|their|with|to|of|inconsistent\s+with)\b', prefix_text, re.IGNORECASE):
            return False

        return True

    def _build_strategy_anchors(self, text: str):
        """Locates all strategy section headers in the document."""
        for pat in self.STRATEGY_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # Verify not in TOC
                if not self._is_in_toc(m.start()):
                    if not self._is_structural_heading(text, m.start(), m.end(), m.group(0)):
                        continue
                    self.strategy_anchors.append(
                        StrategyAnchor(
                            start_offset=m.start(),
                            heading_name=m.group(0),
                            length=m.end() - m.start(),
                        )
                    )
        self.strategy_anchors.sort(key=lambda a: a.start_offset)

    def _build_boundary_candidates(self, text: str):
        """Identifies potential fund delimiters (<hr>, fund-summary class divs, etc.)."""
        for pat, b_type in self.DELIMITER_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                if not self._is_in_toc(m.start()):
                    self.boundary_candidates.append(
                        SectionBoundaryCandidate(
                            start_offset=m.start(),
                            end_offset=m.end(),
                            boundary_type=b_type,
                            delimiter_text=m.group(0)[:100],
                        )
                    )
        self.boundary_candidates.sort(key=lambda b: b.start_offset)

    def _index_known_metadata(
        self,
        text: str,
        known_metadata: List[Dict[str, str]],
        alias_legal_names: List[str] = None,
    ):
        """Indexes series IDs, class IDs, and legal fund names across the document.

        V1.3.0: Added alias_legal_names parameter.  When provided, alias names are scanned
        with the same resilience pattern as manifest legal names and stored in
        alias_name_occurrences.  This allows the series resolver to fall back to historical
        source names from ETF_HISTORICAL_IDENTITY_ALIAS_AUTHORITY_V1 when the manifest name
        fails to match.  Series isolation invariant is preserved: each alias is consumed only
        within the series block identified by its Series ID or Class ID primary key.
        """
        if alias_legal_names is None:
            alias_legal_names = []

        # 1. Fast regex scan for all Series IDs in text
        for m in re.finditer(r"\b(S\d{9})\b", text, re.IGNORECASE):
            sid = m.group(1).upper()
            occ = Occurrence(
                matched_term=sid,
                term_type="SERIES_ID",
                start_offset=m.start(),
                end_offset=m.end(),
                is_toc_or_cross_ref=self._is_in_toc(m.start()),
                context_snippet=text[max(0, m.start() - 50): min(len(text), m.end() + 50)],
            )
            self.series_occurrences.setdefault(sid, []).append(occ)

        # 2. Fast regex scan for all Class IDs in text
        for m in re.finditer(r"\b(C\d{9})\b", text, re.IGNORECASE):
            cid = m.group(1).upper()
            occ = Occurrence(
                matched_term=cid,
                term_type="CLASS_ID",
                start_offset=m.start(),
                end_offset=m.end(),
                is_toc_or_cross_ref=self._is_in_toc(m.start()),
                context_snippet=text[max(0, m.start() - 50): min(len(text), m.end() + 50)],
            )
            self.class_occurrences.setdefault(cid, []).append(occ)

        # Helper: build flexible legal-name regex and scan for occurrences.
        # V1.3.0: Added hyphen (\-) and Unicode dashes (–, —, &#8211;, &#8212;, &ndash;, &mdash;)
        # to the separator so that 'Nasdaq-100' in source matches 'Nasdaq 100' in the manifest
        # (QSIX SERIES_RESOLVER_DEFECT: cover page uses 'Nasdaq-100' vs manifest 'Nasdaq 100').
        sep = r"(?:<[^>]+>|\s|-|–|—|&#8211;|&#8212;|&ndash;|&mdash;|&#174;|&reg;|&#8482;|&trade;|[®™]|\([Rr]\)|\([Tt][Mm]\))+"

        def _scan_name(raw_name: str, term_type: str, target_dict: Dict[str, List[Occurrence]]):
            if not raw_name or len(raw_name.strip()) <= 5:
                return
            words = raw_name.strip().split()
            word_patterns = [DocumentNormalizer.get_word_equivalence_pattern(w) for w in words]
            name_pat = sep.join(word_patterns)
            try:
                for m in re.finditer(name_pat, text, re.IGNORECASE):
                    occ = Occurrence(
                        matched_term=raw_name,
                        term_type=term_type,
                        start_offset=m.start(),
                        end_offset=m.end(),
                        is_toc_or_cross_ref=self._is_in_toc(m.start()),
                        context_snippet=text[max(0, m.start() - 50): min(len(text), m.end() + 50)],
                    )
                    target_dict.setdefault(raw_name, []).append(occ)
                    if term_type == "LEGAL_NAME":
                        name_clean = DocumentNormalizer.normalize_name(raw_name)
                        if name_clean:
                            self.normalized_name_occurrences.setdefault(name_clean, []).append(occ)
            except re.error:
                pass

        # 3. Known legal names scan (resilient to whitespace, HTML tags, font tags, and trademark glyphs)
        for meta in known_metadata:
            _scan_name(meta.get("legal_name", ""), "LEGAL_NAME", self.legal_name_occurrences)

        # 4. V1.3.0: Alias legal names scan (from ETF_HISTORICAL_IDENTITY_ALIAS_AUTHORITY_V1).
        # Stored in alias_name_occurrences.  The series resolver consults this dict as a
        # supplementary identity signal when LEGAL_NAME occurrences return zero hits.
        # Alias lookup is keyed by series_id at the resolver layer — no target-specific logic here.
        for alias_name in alias_legal_names:
            _scan_name(alias_name, "ALIAS_NAME", self.alias_name_occurrences)


    def _to_deterministic_dict(self) -> Dict[str, Any]:
        """Converts index into a serializable deterministic dictionary."""
        return {
            "identity": asdict(self.identity),
            "engine_version": self.engine_version,
            "normalization_version": self.normalization_version,
            "schema_version": self.schema_version,
            "source_bytes_sha256": self.identity.source_bytes_sha256,
            "normalized_text_sha256": self.normalized_text_sha256,
            "series_count": len(self.series_occurrences),
            "class_count": len(self.class_occurrences),
            "strategy_anchor_count": len(self.strategy_anchors),
            "boundary_candidate_count": len(self.boundary_candidates),
            "indexed_series_ids": sorted(list(self.series_occurrences.keys())),
            "indexed_class_ids": sorted(list(self.class_occurrences.keys())),
        }
