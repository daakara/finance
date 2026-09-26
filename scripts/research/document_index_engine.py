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


INDEX_ENGINE_VERSION = "DOC_INDEX_V1_0_0"
NORMALIZATION_VERSION = "NORMALIZATION_V1_0_0"
INDEX_SCHEMA_VERSION = "SCHEMA_V1_0_0"


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
    """Versioned, deterministic document normalizer (NORMALIZATION_V1_0_0)."""

    VERSION = NORMALIZATION_VERSION

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
        """Standardized legal fund name normalization."""
        if not name:
            return ""
        norm = unicodedata.normalize("NFKC", name)
        norm = html.unescape(norm)
        norm = norm.lower()
        norm = re.sub(r"[^\w\s]", "", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm


class DocumentIndex:
    """Deterministic structural index for a single SEC statutory filing."""

    STRATEGY_PATTERNS = [
        r"Principal\s+Investment\s+Strateg(?:y|ies)",
        r"Investment\s+Objective\s+and\s+Principal\s+Strategies",
        r"Principal\s+Strategies",
        r"Principal\s+Investment\s+Policies\s+and\s+Strategies",
        r"Principal\s+Risks\s+and\s+Strategies",
    ]

    DELIMITER_PATTERNS = [
        (r"<hr\s*/?>", "HR_DELIMITER"),
        (r"<div[^>]*class=[\"'][^\"']*(?:fund-summary|fund-section|summary|summary-section)[\"']", "FUND_SUMMARY_DIV"),
        (r"<h[1-4][^>]*>\s*(?:Fund\s+Summary|Summary\s+Prospectus|Fund\s+Overview|SUMMARY\s+SECTION)\b", "FUND_SUMMARY_HEADING"),
        (r"\bFund\s+Summary\b", "TEXT_FUND_SUMMARY"),
        (r"\bSummary\s+Prospectus\b", "TEXT_SUMMARY_PROSPECTUS"),
        (r"\bSUMMARY\s+SECTION\b", "TEXT_SUMMARY_SECTION"),
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
    ):
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
        self.strategy_anchors: List[StrategyAnchor] = []
        self.boundary_candidates: List[SectionBoundaryCandidate] = []
        self.toc_ranges: List[Tuple[int, int]] = []

        self._build_toc_ranges(self.normalized_text)
        self._build_strategy_anchors(self.normalized_text)
        self._build_boundary_candidates(self.normalized_text)
        self._index_known_metadata(self.normalized_text, known_series_metadata or [])

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

    def _build_strategy_anchors(self, text: str):
        """Locates all strategy section headers in the document."""
        for pat in self.STRATEGY_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # Verify not in TOC
                if not self._is_in_toc(m.start()):
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

    def _index_known_metadata(self, text: str, known_metadata: List[Dict[str, str]]):
        """Indexes series IDs, class IDs, and legal fund names across the document."""
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

        # 3. Known legal names scan (linear string find, avoids catastrophic backtracking)
        text_lower = text.lower()
        for meta in known_metadata:
            raw_name = meta.get("legal_name", "")
            if raw_name and len(raw_name.strip()) > 5:
                name_clean = DocumentNormalizer.normalize_name(raw_name)
                # Search occurrences using case-insensitive find
                pos = 0
                norm_lower = raw_name.lower().strip()
                while True:
                    idx = text_lower.find(norm_lower, pos)
                    if idx == -1:
                        break
                    occ = Occurrence(
                        matched_term=raw_name,
                        term_type="LEGAL_NAME",
                        start_offset=idx,
                        end_offset=idx + len(raw_name),
                        is_toc_or_cross_ref=self._is_in_toc(idx),
                        context_snippet=text[max(0, idx - 50): min(len(text), idx + len(raw_name) + 50)],
                    )
                    self.legal_name_occurrences.setdefault(raw_name, []).append(occ)
                    if name_clean:
                        self.normalized_name_occurrences.setdefault(name_clean, []).append(occ)
                    pos = idx + len(norm_lower)

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
