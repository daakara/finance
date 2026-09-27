"""
scripts/research/etf_v2/identity_authority.py

Identity Authority for Pipeline V2.
Enforces the invariant: WEAK_TEXTUAL_IDENTITY_MAY_OVERRIDE_EXACT_IDS = NO.
Exact CIK, Series ID, and Class ID dominate fuzzy name and ticker heuristics.
"""

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
    def match_identity(
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

        # Word-split resilient name match (handles HTML tags, entities like &#38;, trademark symbols)
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
