"""
scripts/research/etf_v2/document_structure.py

Document Structure Engine for Pipeline V2.
Builds structural representations from statutory prospectuses (HTML/XML/DOM).
Enforces:
1. Canonical normalization under NORMALIZATION_V1_0_0.
2. Structural distinction between Item 2 (Investment Objective) and Item 4 (Principal Investment Strategies).
3. Rejection of mid-sentence prose occurrences from masquerading as structural headings.
"""

import re
import hashlib
from typing import List, Dict, Optional, Tuple
from .models import DocumentStructure, StructuralSection

NORMALIZATION_VERSION = "NORMALIZATION_V1_0_0"


class DocumentStructureEngine:
    """Parses statutory filings into structured document representations."""

    STRATEGY_PATTERNS = [
        r"(?:item\s+4\.?\s*)?principal\s+investment\s+strategies",
        r"(?:item\s+4\.?\s*)?principal\s+strategies",
        r"(?:item\s+4\.?\s*)?principal\s+investment\s+strategy",
        r"(?:item\s+9\.?\s*)?principal\s+investment\s+strategies",
        r"investment\s+objective\s+and\s+principal\s+strategies",
    ]

    OBJECTIVE_PATTERNS = [
        r"(?:item\s+2\.?\s*)?investment\s+objective",
        r"(?:item\s+2\.?\s*)?investment\s+goal",
    ]

    TERMINATION_PATTERNS = [
        r"(?:item\s+5\.?\s*)?principal\s+(?:investment\s+)?risks",
        r"(?:item\s+3\.?\s*)?fee\s+table",
        r"(?:item\s+6\.?\s*)?annual\s+fund\s+operating\s+expenses",
        r"(?:item\s+8\.?\s*)?portfolio\s+management",
        r"<hr[^>]*>",
    ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Canonical normalization procedure according to ETF_SOURCE_PROVENANCE_CONTRACT_V1.json."""
        if not text:
            return ""
        # 2. Line breaks
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        # 3. HTML entities
        t = t.replace("&nbsp;", " ").replace("&#160;", " ")
        t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        t = t.replace("&mdash;", "—").replace("&#8212;", "—").replace("&ndash;", "–").replace("&#8211;", "–")
        # 4. Text extraction (strip tags but preserve block boundaries)
        t = re.sub(r"<(?:p|div|tr|br|h[1-6])[^>]*>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"<[^>]+>", " ", t)
        # 5. Whitespace collapse
        lines = []
        for line in t.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @classmethod
    def is_structural_heading(cls, text: str, start: int, end: int, matched_text: str) -> bool:
        """Verifies whether a match is a true structural heading rather than mid-sentence narrative prose."""
        # 1. Reject lowercase prose matches that continue into narrative sentences
        post = text[end: end + 30]
        if matched_text.islower() and re.match(r"^\s+[a-z]", post):
            return False

        # 2. Inspect prefix text
        pre = text[max(0, start - 120): start]
        last_boundary = max(pre.rfind(">"), pre.rfind("\n"))
        prefix_text = pre[last_boundary + 1:].strip() if last_boundary != -1 else pre.strip()

        if not prefix_text:
            return True

        if re.match(r"^(?:item\s+\d+\.?|\d+\.?|[A-Z]\.?|\*|\u2022)\s*$", prefix_text, re.IGNORECASE):
            return True

        if re.search(r"\b(?:heading|caption|section|entitled)\b", prefix_text, re.IGNORECASE):
            return True

        if prefix_text and prefix_text[-1] in ('"', "'", ":", ".", ";", "-", "—"):
            return True

        if re.search(r"\b(?:the|its|our|their|with|to|of|inconsistent\s+with)\b", prefix_text, re.IGNORECASE):
            return False

        return True

    @staticmethod
    def _find_matching_div_end(text: str, start_pos: int) -> int:
        open_tag_end = text.find(">", start_pos)
        if open_tag_end == -1:
            return -1
        pos = open_tag_end + 1
        depth = 1
        text_len = len(text)
        while depth > 0 and pos < text_len:
            next_open = text.find("<div", pos)
            next_close = text.find("</div>", pos)
            if next_close == -1:
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = text.find(">", next_open) + 1
                if pos == 0:
                    break
            else:
                depth -= 1
                pos = next_close + 6
        return pos

    @classmethod
    def parse_structure(cls, raw_bytes: bytes, filename: str) -> DocumentStructure:
        """Parses raw document bytes into a DocumentStructure."""
        raw_text = raw_bytes.decode("utf-8", errors="ignore")
        norm_text = cls.normalize_text(raw_text)
        norm_sha = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        sections = []

        # Find Item 4 Strategy Headings
        for pat in cls.STRATEGY_PATTERNS:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                if cls.is_structural_heading(raw_text, m.start(), m.end(), m.group(0)):
                    sections.append(
                        StructuralSection(
                            heading=m.group(0),
                            heading_role="PRINCIPAL_STRATEGY",
                            start_offset=m.start(),
                            end_offset=m.end(),
                        )
                    )

        # Find Item 2 Objective Headings
        for pat in cls.OBJECTIVE_PATTERNS:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                if cls.is_structural_heading(raw_text, m.start(), m.end(), m.group(0)):
                    sections.append(
                        StructuralSection(
                            heading=m.group(0),
                            heading_role="INVESTMENT_OBJECTIVE",
                            start_offset=m.start(),
                            end_offset=m.end(),
                        )
                    )

        # Tag-range detection for iXBRL header, XML context, and hidden metadata blocks
        excluded_spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"<ix:header\b.*?</ix:header>", raw_text, re.DOTALL | re.IGNORECASE):
            excluded_spans.append((m.start(), m.end()))
        for m in re.finditer(r"<xbrli:context\b.*?</xbrli:context>", raw_text, re.DOTALL | re.IGNORECASE):
            excluded_spans.append((m.start(), m.end()))
        for m in re.finditer(r"<ix:hidden\b.*?</ix:hidden>", raw_text, re.DOTALL | re.IGNORECASE):
            excluded_spans.append((m.start(), m.end()))
        for m in re.finditer(r"<div\b[^>]*display:\s*none[^>]*>", raw_text, re.IGNORECASE):
            end_pos = cls._find_matching_div_end(raw_text, m.start())
            if end_pos != -1:
                excluded_spans.append((m.start(), end_pos))

        # Find Series ID occurrences (bounded regex matching plain and iXBRL Member forms)
        series_occurrences: Dict[str, List[int]] = {}
        SERIES_ID_PATTERN = r"(?<![A-Za-z0-9])(S\d{9})(?:Member)?(?![A-Za-z0-9])"
        for m in re.finditer(SERIES_ID_PATTERN, raw_text):
            sid = m.group(1)
            pos = m.start()
            if not any(s <= pos < e for s, e in excluded_spans):
                series_occurrences.setdefault(sid, []).append(pos)

        sections.sort(key=lambda s: s.start_offset)

        # Extract authoritative EDGAR form from parsed <TYPE> metadata header
        type_match = re.search(r"<TYPE>\s*([A-Za-z0-9\-\/]+)", raw_text[:2500], re.IGNORECASE)
        edgar_form = type_match.group(1).upper() if type_match else ""

        doc_struct = DocumentStructure(
            document_filename=filename,
            raw_text=raw_text,
            normalized_text=norm_text,
            normalized_text_sha256=norm_sha,
            sections=sections,
            series_occurrences=series_occurrences,
        )
        doc_struct.form = edgar_form
        return doc_struct
