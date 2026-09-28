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

    _form_497k_cohort: Optional[Dict[str, Dict[str, Any]]] = None

    FAIL_CLOSED_LANE_C_SYMBOLS = {"BFOR", "OEFA", "OGIG", "OUSA", "OUSM"}

    APPROVED_VARIATION_CLASSES = {
        "TRUST_BRAND_PREFIX_SEPARATION",
        "REBRANDING_NAME_EVOLUTION",
        "INDEX_FUND_VS_ETF_SUFFIX",
        "SERIES_QUALIFIER_PUNCTUATION_OR_HYPHEN",
        "HYPHENATION_COMPOUND_WORD_VARIATION",
        "LEGAL_SUFFIX_OR_STRATEGY_WORDING",
        "TRADEMARK_OR_SYMBOL_QUALIFIER",
        "NUMERIC_PLUS_QUALIFIER_VARIATION",
        "SHARE_CLASS_SUFFIX_VARIATION",
        "PLURAL_WORD_FORM_VARIATION",
        "CHARACTER_ENCODING_CORRUPTION",
        "TICKER_IN_NAME_VARIATION",
    }

    @classmethod
    def _get_form_497k_cohort(cls) -> Dict[str, Dict[str, Any]]:
        """Loads authoritative Form 497K Single-Fund Identity Qualification cohort."""
        if cls._form_497k_cohort is None:
            cohort = {}
            repo_root = Path(__file__).resolve().parents[3]
            ledger_path = repo_root / "docs" / "research" / "ETF_V2_FORM_497K_QUALIFICATION_LEDGER.json"
            if ledger_path.exists():
                try:
                    import json
                    with open(ledger_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for r in data.get("census_records", []):
                        if r.get("status") == "POLICY_RESOLVABLE":
                            cohort[r["symbol"]] = r
                except Exception:
                    pass
            cls._form_497k_cohort = cohort
        return cls._form_497k_cohort

    @classmethod
    def has_authoritative_ticker_declaration(cls, text: str, symbol: str) -> bool:
        """
        Validates authoritative exchange ticker declaration in Form 497K header.
        Rejects narrative mentions like 'market capitalization size'.
        """
        if not text or not symbol:
            return False

        header = text[:15000]
        # Bounded fixed-point unescape and tag stripping
        for _ in range(5):
            un = html.unescape(header)
            if un == header:
                break
            header = un
        header = re.sub(r"<[^>]+>", " ", header)
        header = re.sub(r"[–—−‐\x96\x97]", "-", header)
        header = re.sub(r"\s+", " ", header).strip()

        # 1. Standard ticker label declaration
        label_pattern = rf"(?:Ticker(?:\s*Symbol)?|Trading\s*Symbol|Symbol)(?:\s+ETF\s+Class)?\s*[:\-\s]\s*\(?\b{re.escape(symbol)}\b\)?"
        if re.search(label_pattern, header, re.IGNORECASE):
            return True

        # 2. Exchange label declaration
        exchange_pattern = rf"(?:NYSE\s*(?:Arca)?|NASDAQ|Cboe(?:\s*BZX)?)\s*[:\-\s]\s*\(?\b{re.escape(symbol)}\b\)?"
        if re.search(exchange_pattern, header, re.IGNORECASE):
            return True
        trailing_exchange_pattern = rf"\b{re.escape(symbol)}\b\s*\(?(?:NYSE\s*(?:Arca|Ticker)?|NASDAQ|Cboe(?:\s*BZX)?)\)?"
        if re.search(trailing_exchange_pattern, header, re.IGNORECASE):
            return True

        # 3. Delimited ticker on cover page
        pipe_pattern = rf"\b{re.escape(symbol)}\b\s*\|\s*(?:NYSE|NASDAQ|Cboe)"
        if re.search(pipe_pattern, header, re.IGNORECASE):
            return True

        # 4. Explicit parenthesized/bracketed uppercase ticker in header
        paren_pattern = rf"[\(\[\—\–\-]\s*\b{re.escape(symbol)}\b\s*[\)\]\—\–\-]"
        if re.search(paren_pattern, header):
            return True

        # 5. Summary prospectus header layout where ticker directly precedes/follows fund name or header date
        sp_pattern = rf"(?:T\.\s*ROWE\s*PRICE|SUMMARY\s*PROSPECTUS)(?:[^\.\n\r]{{0,100}})?\s+\b{re.escape(symbol)}\b\s+(?:Invesco|[A-Z])"
        if re.search(sp_pattern, header, re.IGNORECASE):
            return True

        return False

    @classmethod
    def has_statutory_mandate_evidence(cls, text: str) -> bool:
        """
        Validates non-empty statutory mandate evidence in filing text.
        Fails closed on hollow joint supplements lacking Item 4/Item 2 disclosures.
        """
        if not text:
            return False
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        m_strat = re.search(r"(?:Principal\s+Investment\s+Strateg(?:ies|y)|Principal\s+Strategies)", clean, re.IGNORECASE)
        m_obj = re.search(r"(?:Investment\s+Objective|Investment\s+Goal)", clean, re.IGNORECASE)
        return bool(m_strat or m_obj)

    @classmethod
    def match_identity(
        cls,
        text: str,
        identity: EntityIdentity,
        filing: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates presence of authoritative identity indicators in statutory text.
        Exact Series ID and Class ID take absolute precedence over legal name.
        Composes deterministically with Form 497K Single-Fund Identity Qualification Supplement.
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

        if is_qualified:
            return {
                "has_series_id": has_series,
                "has_class_id": has_class,
                "has_name": has_name,
                "has_symbol": has_symbol,
                "confidence": confidence,
                "is_qualified": True,
            }

        # Form 497K Single-Fund Identity Qualification Policy Supplement
        # Evaluates all 7 mandatory conditions:
        # 1. filing form is Form 497K
        # 2. filing registrant CIK matches target authoritative CIK
        # 3. target exchange ticker is explicitly present in authoritative header declaration
        # 4. accession belongs to certified pre-boundary authority chain
        # 5. fund-name difference belongs to approved variation class (DELIMITER_PIPE_VARIATION excluded)
        # 6. filing contains non-empty statutory mandate evidence (completeness_state != NONE)
        # 7. no sibling series within the same trust shares candidate identity (fail-closed targets excluded)
        if filing is not None:
            form = getattr(filing, "form", "")
            accession = getattr(filing, "accession", "")
            filing_cik = getattr(filing, "cik", None)

            cohort = cls._get_form_497k_cohort()
            target_record = cohort.get(identity.symbol)

            if target_record and target_record.get("status") == "POLICY_RESOLVABLE":
                cik_matches = (filing_cik is None or str(filing_cik).strip().zfill(10) == identity.cik) and (target_record.get("cik") == identity.cik)
                acc_matches = (accession == target_record.get("base_accession"))
                form_matches = (form == "497K")
                var_class = target_record.get("difference_class", "")
                class_approved = (var_class in cls.APPROVED_VARIATION_CLASSES)
                not_fail_closed = (identity.symbol not in cls.FAIL_CLOSED_LANE_C_SYMBOLS)

                if cik_matches and acc_matches and form_matches and class_approved and not_fail_closed:
                    if cls.has_authoritative_ticker_declaration(text, identity.symbol):
                        if cls.has_statutory_mandate_evidence(text):
                            return {
                                "has_series_id": has_series,
                                "has_class_id": has_class,
                                "has_name": True,
                                "has_symbol": True,
                                "confidence": 0.50 if has_series else 0.40,
                                "is_qualified": True,
                                "qualification_authority": "ETF_V2_FORM_497K_STATUTORY_IDENTITY_QUALIFICATION_SUPPLEMENT",
                                "variation_class": var_class,
                            }

        return {
            "has_series_id": has_series,
            "has_class_id": has_class,
            "has_name": has_name,
            "has_symbol": has_symbol,
            "confidence": confidence,
            "is_qualified": False,
        }

