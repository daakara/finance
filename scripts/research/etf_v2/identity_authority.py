"""
scripts/research/etf_v2/identity_authority.py

Identity Authority for Pipeline V2.
Enforces the invariant: WEAK_TEXTUAL_IDENTITY_MAY_OVERRIDE_EXACT_IDS = NO.
Exact CIK, Series ID, and Class ID dominate fuzzy name and ticker heuristics.
"""

import html
from pathlib import Path
import re
from typing import Dict, Any, Optional
from .models import EntityIdentity


class IdentityAuthority:
    """Validates and enforces authoritative entity identity."""

    @staticmethod
    def create_identity(
        symbol: str,
        cik: str,
        series_id: str,
        class_id: str,
        legal_name: str,
        historical_aliases: Optional[list] = None,
    ) -> EntityIdentity:
        """Constructs and validates an authoritative EntityIdentity."""
        # Clean formatting
        sym = symbol.strip().upper()
        clean_cik = str(cik).strip().zfill(10)
        clean_sid = series_id.strip()
        clean_cid = class_id.strip()
        clean_name = re.sub(r"\s+", " ", legal_name).strip()

        # Invariant checks: Format validation
        if not re.match(r"^S\d{9}$", clean_sid):
            raise ValueError(f"Invalid Series ID format: {clean_sid}")
        if not re.match(r"^C\d{9}$", clean_cid):
            raise ValueError(f"Invalid Class ID format: {clean_cid}")
        if not re.match(r"^\d{10}$", clean_cik):
            raise ValueError(f"Invalid CIK format: {clean_cik}")

        identity = EntityIdentity(
            symbol=sym,
            cik=clean_cik,
            series_id=clean_sid,
            class_id=clean_cid,
            legal_name=clean_name,
            historical_aliases=historical_aliases or [],
        )

        assert identity.validate(), f"Failed identity validation for {sym}"
        return identity

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """
        Canonical text normalization primitive (Architecture D).
        Performs bounded fixed-point HTML entity decoding (k <= 5),
        typographical punctuation canonicalization, non-printing/trademark
        mark removal, and whitespace collapsing.
        """
        if not text:
            return ""
        prev = text
        for _ in range(5):
            curr = html.unescape(prev)
            if curr == prev:
                break
            prev = curr

        # Typographical punctuation canonicalization
        curr = re.sub(r"[–—−‐\x96\x97]", "-", curr)
        curr = re.sub(r"[’‘`′]", "'", curr)
        curr = re.sub(r'[“”″]', '"', curr)
        curr = curr.replace("&#47;", "/").replace("&#58;", ":")

        # Zero-width / BOM & mark removal
        curr = re.sub(r"[\u200b\ufeff]", "", curr)
        curr = re.sub(r"[®™©]", "", curr)
        curr = re.sub(r"\((?:r|tm)\)", "", curr, flags=re.IGNORECASE)

        # Whitespace collapsing & trimming
        curr = re.sub(r"\s+", " ", curr).strip()
        return curr

    _lane_a_symbols: Optional[set] = None

    @classmethod
    def _get_lane_a_symbols(cls) -> set:
        if cls._lane_a_symbols is None:
            repo_root = Path(__file__).resolve().parents[3]
            ledger_path = repo_root / "docs" / "research" / "ETF_V2_RESIDUAL_FAILURE_MODE_LEDGER.json"
            if ledger_path.exists():
                try:
                    import json
                    with open(ledger_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cls._lane_a_symbols = {
                        r["symbol"] for r in data.get("cohort_a_records", [])
                        if r.get("remediation_class") == "NORMALIZATION_REMEDIATION_CANDIDATE"
                    }
                except Exception:
                    cls._lane_a_symbols = set()
            else:
                cls._lane_a_symbols = set()
        return cls._lane_a_symbols

    @classmethod
    def match_identity(
        cls,
        text: str,
        identity: EntityIdentity,
    ) -> Dict[str, Any]:
        """
        Evaluates presence of authoritative identity indicators in statutory text.
        Exact Series ID and Class ID take absolute precedence over legal name.
        """
        has_series = identity.series_id in text
        has_class = identity.class_id in text
        symbols_to_check = [identity.symbol] + (identity.historical_aliases or [])
        has_symbol = any(bool(re.search(rf"\b{re.escape(sym)}\b", text, re.IGNORECASE)) for sym in symbols_to_check)

        has_name = False
        if not has_series:
            # Baseline textual name match
            words = identity.legal_name.split()
            sep = r"(?:<[^>]+>|\s|&#174;|&reg;|&#8482;|&trade;|[®™]|\([Rr]\)|\([Tt][Mm]\))+"
            word_patterns = []
            for w in words:
                if w.lower() in ("&", "and", "&amp;", "&#38;", "&#x26;"):
                    word_patterns.append(r"(?:&|&amp;|&#38;|&#x26;|and)")
                else:
                    word_patterns.append(re.escape(w))
            name_pattern = sep.join(word_patterns)
            has_name = bool(re.search(name_pattern, text, re.IGNORECASE))

            # Canonical normalized name matching (Architecture D) for Lane A authorized cohort
            if not has_name and (identity.symbol in cls._get_lane_a_symbols()):
                norm_name = cls.normalize_for_matching(identity.legal_name)
                norm_text = cls.normalize_for_matching(text)
                norm_words = norm_name.split()
                norm_sep = r"(?:<[^>]+>|\s)+"
                norm_word_patterns = [r"(?:&|and)" if w.lower() in ("&", "and") else re.escape(w) for w in norm_words]
                norm_pattern = norm_sep.join(norm_word_patterns)
                has_name = bool(re.search(norm_pattern, norm_text, re.IGNORECASE))
        else:
            has_name = identity.legal_name.lower() in text.lower()

        # Identity strength scoring
        # Series ID > Class ID > Exact Name > Symbol
        confidence = 0.0
        if has_series:
            confidence += 0.50
        if has_class:
            confidence += 0.25
        if has_name:
            confidence += 0.20
        if has_symbol:
            confidence += 0.05

        is_qualified = bool(
            has_series
            or (has_class and has_name)
            or (has_name and (has_symbol or len(identity.legal_name) >= 15))
        )

        return {
            "has_series_id": has_series,
            "has_class_id": has_class,
            "has_name": has_name,
            "has_symbol": has_symbol,
            "confidence": confidence,
            "is_qualified": is_qualified,
        }

