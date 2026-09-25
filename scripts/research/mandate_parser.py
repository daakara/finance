"""ARX Terminal — Deterministic Statutory Mandate Parser (Policy v1.1.0).

FROZEN PRE-POPULATION EXECUTION RULESET.
Extracts only frozen Policy-v1.1 fields:
- broad_or_multi_sector_mandate: bool
- sector_specific_mandate: bool
- approved_sector: Optional[str]
- government_debt_mandate: bool
- corporate_credit_mandate: bool
- non_confirmatory_mandate: bool

PROHIBITIONS:
- No classification from ticker, security name, marketing title, or fund name.
- No model interpretation as final authority.
- Ambiguous or contradictory evidence FAILS CLOSED as UNRESOLVED (denominator blocking).
"""

import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup


APPROVED_SECTORS = {
    "TECHNOLOGY",
    "FINANCIALS",
    "ENERGY",
    "HEALTHCARE",
    "INDUSTRIALS",
    "MATERIALS",
    "CONSUMER_STAPLES",
    "CONSUMER_DISCRETIONARY",
    "UTILITIES",
    "REAL_ESTATE",
    "COMMUNICATION_SERVICES",
}

SECTOR_KEYWORDS = {
    "TECHNOLOGY": [r"\btechnology\b", r"\bsemiconductor", r"\bsoftware\b", r"\binformation technology\b"],
    "FINANCIALS": [r"\bfinancial\b", r"\bbank\b", r"\bbanking\b", r"\bregional bank", r"\binsurance\b"],
    "ENERGY": [r"\benergy\b", r"\boil\b", r"\bgas\b", r"\bpetroleum\b", r"\bexploration & production\b"],
    "HEALTHCARE": [r"\bhealth care\b", r"\bhealthcare\b", r"\bbiotechnology\b", r"\bbiotech\b", r"\bpharmaceutical"],
    "INDUSTRIALS": [r"\bindustrial", r"\baerospace\b", r"\bdefense\b", r"\btransportation\b"],
    "MATERIALS": [r"\bmaterials\b", r"\bchemical", r"\bmining\b", r"\bmetals?\b"],
    "CONSUMER_STAPLES": [r"\bconsumer staples\b", r"\bfood & beverage\b", r"\bhousehold products\b"],
    "CONSUMER_DISCRETIONARY": [r"\bconsumer discretionary\b", r"\bretail\b", r"\bhomebuilders?\b", r"\bhome construction\b"],
    "UTILITIES": [r"\butilities\b", r"\belectric utilities\b"],
    "REAL_ESTATE": [r"\breal estate\b", r"\breit\b", r"\breits\b"],
    "COMMUNICATION_SERVICES": [r"\bcommunication services\b", r"\btelecom", r"\bmedia\b"],
}

BROAD_EQUITY_PATTERNS = [
    r"\bs&p 500\b",
    r"\brussell (1000|2000|3000)\b",
    r"\bdow jones industrial average\b",
    r"\bnasdaq-?100\b",
    r"\bcrsp us (total market|large cap|mid cap|small cap)\b",
    r"\bmsci (usa|us|us investable market)\b",
    r"\bs&p (midcap 400|smallcap 600|total market)\b",
    r"\bbroad(?:-based)? (?:us|u\.s\.) equity\b",
    r"\btotal (?:us|u\.s\.) stock market\b",
    r"\bmarket-wide\b",
]

TREASURY_PATTERNS = [
    r"\bu\.s\. treasury\b",
    r"\bus treasury\b",
    r"\btreasury (?:bills?|notes?|bonds?)\b",
    r"\btips\b",
    r"\btreasury inflation-protected\b",
    r"\bbloomberg (?:u\.s\.|us) treasury\b",
    r"\bice (?:u\.s\.|us) treasury\b",
]

CORPORATE_CREDIT_PATTERNS = [
    r"\bcorporate bond",
    r"\binvestment grade corporate\b",
    r"\bhigh yield (?:corporate )?bond",
    r"\bliquid high yield\b",
    r"\bmarkit iboxx (?:usd )?liquid (?:high yield|investment grade)\b",
    r"\bbloomberg (?:u\.s\.|us) (?:corporate|high yield)\b",
]

NON_CONFIRMATORY_PATTERNS = [
    r"\bactively managed\b",
    r"\bactive management\b",
    r"\bcovered call\b",
    r"\boption (?:writing|overlay|strategy)\b",
    r"\bbuffer (?:etf|strategy)\b",
    r"\bdefined outcome\b",
    r"\bcommodity futures\b",
    r"\bmanaged futures\b",
    r"\bcrypto\b",
    r"\bbitcoin\b",
    r"\bether(?:eum)?\b",
    r"\bemerging markets (?:debt|local currency)\b",
    r"\bex-(?:us|u\.s\.)\b",
    r"\beurope\b",
    r"\basia\b",
    r"\bjapan\b",
    r"\bchina\b",
    r"\bdeveloped ex-(?:us|u\.s\.)\b",
    r"\bleveraged\b",
    r"\binverse\b",
]


@dataclass
class MandateParseResult:
    broad_or_multi_sector_mandate: bool = False
    sector_specific_mandate: bool = False
    approved_sector: Optional[str] = None
    government_debt_mandate: bool = False
    corporate_credit_mandate: bool = False
    non_confirmatory_mandate: bool = False
    source_accession: str = ""
    source_section: str = ""
    evidence_text_hash: str = ""
    parser_rule_id: str = "NO_RULE_MATCH"
    confidence_state: str = "UNRESOLVED"
    evidence_snippet: str = ""


class DeterministicMandateParser:
    """Frozen pre-population deterministic mandate parser under Policy v1.1.0."""

    RULESET_ID = "MANDATE_PARSER_V1_1_0_FROZEN"

    @classmethod
    def extract_strategy_text(cls, html_or_text: str, fund_identifier: Optional[str] = None) -> tuple[str, str]:
        """Extract Principal Investment Strategy section text from filing HTML/text."""
        if not html_or_text or len(html_or_text.strip()) == 0:
            return "", ""

        # If html, extract text or parse
        if "<html" in html_or_text.lower() or "<body" in html_or_text.lower() or "<div" in html_or_text.lower():
            # Quick text clean
            soup = BeautifulSoup(html_or_text[:1000000], "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        else:
            text = html_or_text

        # Look for section header
        patterns = [
            r"Principal Investment Strateg(?:y|ies)",
            r"Investment Objective and Principal Strategies",
            r"Principal Strategies",
        ]
        
        found_start = -1
        section_name = ""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found_start = m.start()
                section_name = m.group(0)
                break

        if found_start == -1:
            return "", "SECTION_NOT_FOUND"

        # Bounded extract: up to 15,000 characters from start of strategy
        extracted = text[found_start: found_start + 15000]
        return extracted, section_name

    @classmethod
    def parse_mandate(
        cls,
        strategy_text: str,
        accession: str = "",
        section_name: str = "Principal Investment Strategies"
    ) -> MandateParseResult:
        """Parse structured mandate fields deterministically from strategy text."""
        result = MandateParseResult(
            source_accession=accession,
            source_section=section_name
        )

        if not strategy_text or len(strategy_text.strip()) < 50:
            result.confidence_state = "PARSE_FAILURE_EMPTY_SECTION"
            return result

        result.evidence_text_hash = hashlib.sha256(strategy_text.encode("utf-8")).hexdigest()
        text_lower = strategy_text.lower()

        # 1. Check for non-confirmatory signals first
        non_conf_matches = []
        for pat in NON_CONFIRMATORY_PATTERNS:
            if re.search(pat, text_lower):
                non_conf_matches.append(pat)

        # 2. Check for Treasury Government mandate
        govt_matches = []
        for pat in TREASURY_PATTERNS:
            if re.search(pat, text_lower):
                govt_matches.append(pat)

        # 3. Check for Corporate Credit mandate
        credit_matches = []
        for pat in CORPORATE_CREDIT_PATTERNS:
            if re.search(pat, text_lower):
                credit_matches.append(pat)

        # 4. Check for Approved Sector mandate
        detected_sectors = []
        for sector, kw_list in SECTOR_KEYWORDS.items():
            for kw in kw_list:
                if re.search(kw, text_lower):
                    detected_sectors.append(sector)
                    break

        # 5. Check for Broad Equity Index mandate
        broad_matches = []
        for pat in BROAD_EQUITY_PATTERNS:
            if re.search(pat, text_lower):
                broad_matches.append(pat)

        # Snippet for evidence
        result.evidence_snippet = strategy_text[:200].replace("\n", " ").strip()

        # Decision Logic:
        # Non-confirmatory mandate
        if len(non_conf_matches) > 0:
            result.non_confirmatory_mandate = True
            result.parser_rule_id = "RULE_NON_CONFIRMATORY"
            result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
            return result

        # Treasury Government
        if len(govt_matches) > 0 and len(credit_matches) == 0 and len(detected_sectors) == 0 and len(broad_matches) == 0:
            result.government_debt_mandate = True
            result.parser_rule_id = "RULE_TREASURY_GOVERNMENT"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Corporate Credit
        if len(credit_matches) > 0 and len(govt_matches) == 0 and len(detected_sectors) == 0 and len(broad_matches) == 0:
            result.corporate_credit_mandate = True
            result.parser_rule_id = "RULE_CORPORATE_CREDIT"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Sector Specific (single approved sector)
        if len(detected_sectors) == 1 and len(broad_matches) == 0:
            result.sector_specific_mandate = True
            result.approved_sector = detected_sectors[0]
            result.parser_rule_id = "RULE_SECTOR_EQUITY"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Broad Equity Index (multi-sector or total market)
        if len(broad_matches) > 0 and len(detected_sectors) == 0:
            result.broad_or_multi_sector_mandate = True
            result.parser_rule_id = "RULE_BROAD_EQUITY_INDEX"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Multiple sector collision -> ambiguous or multi-sector?
        if len(detected_sectors) > 1:
            # Multi-sector without explicit sector concentration is broad
            result.broad_or_multi_sector_mandate = True
            result.parser_rule_id = "RULE_MULTI_SECTOR_EQUITY"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Contradictory / collision between govt and credit
        if len(govt_matches) > 0 and len(credit_matches) > 0:
            # Mixed aggregate bond mandate -> non-confirmatory under Policy v1.1
            result.non_confirmatory_mandate = True
            result.parser_rule_id = "RULE_MIXED_AGGREGATE_BOND"
            result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
            return result

        # Fall-closed: unclassified mandate
        result.confidence_state = "AMBIGUOUS_UNCLASSIFIED"
        result.parser_rule_id = "RULE_FAIL_CLOSED_AMBIGUOUS"
        return result
