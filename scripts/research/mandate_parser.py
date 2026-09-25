"""ARX Terminal — Deterministic Statutory Mandate Parser (Policy v1.1.0).

FROZEN PRE-POPULATION EXECUTION RULESET: MANDATE_PARSER_V1_2_0_FROZEN.
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
- Ambiguous, multi-asset fund-of-funds, or contradictory evidence FAILS CLOSED as UNRESOLVED.
- Incidental sector vocabulary (financial condition, energy costs, technology platforms) REJECTED.
- Foreign, international, ex-US funds REJECTED from US confirmatory equity hypotheses.
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

# Affirmative sector concentration / tracking patterns
# Requires explicit syntactic connection asserting investment/tracking of the specific sector
AFFIRMATIVE_SECTOR_PATTERNS = {
    "TECHNOLOGY": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:technology|semiconductor|software|information\s+technology)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:technology|semiconductor|information\s+technology)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?(?:Technology|Information\s+Technology|Semiconductor)\s+Index\b",
    ],
    "FINANCIALS": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:financials?|financial\s+services|banking|banks|regional\s+banks?|insurance)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:financials?|financial\s+services|regional\s+bank(?:ing)?)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?(?:Financials?|Financial\s+Services|Banks?)\s+Index\b",
        r"\bKBW\s+(?:Bank|Regional\s+Banking)\s+Index\b",
    ],
    "ENERGY": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:energy|oil\s+(?:&|and)\s+gas|petroleum|oil\s+services|exploration\s+(?:&|and)\s+production)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:energy|oil\s+(?:&|and)\s+gas|oil\s+services)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Energy\s+Index\b",
    ],
    "HEALTHCARE": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:health\s*care|healthcare|biotechnology|biotech|pharmaceuticals?|medical\s+devices?)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:health\s*care|biotechnology|biotech|pharmaceuticals?)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Health\s*Care\s+Index\b",
    ],
    "INDUSTRIALS": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:industrials?|aerospace|defense|transportation|airlines?)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:industrials?|aerospace\s+(?:&|and)\s+defense|transportation)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Industrials?\s+Index\b",
    ],
    "MATERIALS": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:materials?|basic\s+materials|chemicals?|mining|metals?\s+(?:&|and)\s+mining)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:materials?|basic\s+materials|metals?\s+(?:&|and)\s+mining)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Materials?\s+Index\b",
    ],
    "CONSUMER_STAPLES": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:consumer\s+staples|food\s+(?:&|and)\s+beverage|household\s+products)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:consumer\s+staples|food\s+(?:&|and)\s+beverage)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Consumer\s+Staples\s+Index\b",
    ],
    "CONSUMER_DISCRETIONARY": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:consumer\s+discretionary|retail|homebuilders?|home\s+construction)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:consumer\s+discretionary|retail|homebuilders?)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Consumer\s+Discretionary\s+Index\b",
    ],
    "UTILITIES": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:utilities|electric\s+utilities|water\s+utilities)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:utilities|electric\s+utilities)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Utilities\s+Index\b",
    ],
    "REAL_ESTATE": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:real\s+estate|reits?)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:real\s+estate|reits?)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\b(?:MSCI\s+US\s+REIT\s+Index|Dow\s+Jones\s+U\.S\.\s+Real\s+Estate\s+Index)\b",
    ],
    "COMMUNICATION_SERVICES": [
        r"\b(?:invests|investing|concentrat\w*|tracks?|replicates?|seeks\s+to\s+track)\b[^.;\n]{0,100}?\b(?:in|of)\b[^.;\n]{0,80}?(?:the\s+)?(?:communication\s+services|telecom(?:munications)?|media\s+(?:&|and)\s+entertainment)\s+(?:select\s+sector|sector|industry|sub-industry)\b",
        r"\b(?:communication\s+services|telecom)\s+(?:select\s+sector|sector\s+index|industry\s+index)\b",
        r"\bS&P\s+(?:500\s+)?Communication\s+Services\s+Index\b",
    ],
}

BROAD_EQUITY_PATTERNS = [
    r"\bs&p 500\b",
    r"\brussell (?:1000|2000|3000)\b",
    r"\bdow jones industrial average\b",
    r"\bnasdaq-?100\b",
    r"\bnasdaq\s+composite\b",
    r"\bcrsp us (?:total market|large cap|mid cap|small cap)\b",
    r"\bmsci (?:usa|us|us investable market)\b",
    r"\bs&p (?:midcap 400|smallcap 600|total market)\b",
    r"\bbroad(?:-based)? (?:us|u\.s\.) equity\b",
    r"\btotal (?:us|u\.s\.) stock market\b",
    r"\bmarket-wide\b",
    r"\bwilshire 5000\b",
    r"\bnyse composite\b",
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

# Non-confirmatory product structure / strategy exclusions
PRODUCT_EXCLUSION_PATTERNS = [
    r"\bcovered call\b",
    r"\boption (?:writing|overlay|strategy)\b",
    r"\bbuffer (?:etf|strategy)\b",
    r"\bdefined outcome\b",
    r"\bcommodity futures\b",
    r"\bmanaged futures\b",
    r"\bcrypto\b",
    r"\bbitcoin\b",
    r"\bether(?:eum)?\b",
    r"\bleveraged\b",
    r"\binverse\b",
    r"\b2x\b",
    r"\b3x\b",
    r"\b-1x\b",
    r"\b-2x\b",
    r"\bdaily target\b",
]

# Geography Invariant: Ex-US / International / Foreign mandates rejected from US confirmatory equity
GEOGRAPHY_EX_US_PATTERNS = [
    r"\b(?:ex[- ](?:us|u\.s\.|united states)|developed ex[- ](?:us|u\.s\.|united states))\b",
    r"\b(?:outside (?:the )?(?:us|u\.s\.|united states))\b",
    r"\b(?:non[- ](?:us|u\.s\.|united states) (?:companies|issuers|securities|stocks|equities|markets|investments|countries))\b",
    r"\b(?:international (?:equity|equities|stocks|developed|markets|index|fund|companies|securities))\b",
    r"\b(?:foreign (?:companies|issuers|securities|stocks|equities|markets|countries))\b",
    r"\b(?:emerging markets|emerging market)\b",
    r"\b(?:global excluding (?:the )?(?:us|u\.s\.|united states))\b",
    r"\b(?:europe|asia|japan|china|latin america|pacific|emea|asia-pacific|australia|united kingdom|canada)\b",
    r"\bftse (?:developed|all cap|global) ex[- ](?:us|u\.s\.)\b",
    r"\bmsci (?:eafe|em|emerging|world ex|acwi ex|acwi)\b",
]

# Affirmative active management pattern (excludes negative qualifiers like "differs from an actively managed fund")
ACTIVE_MANAGEMENT_PATTERN = re.compile(
    r"(?<!differs from an )(?<!unlike an )(?<!not an )(?<!rather than an )"
    r"\b(?:(?:the fund|it)\s+is\s+(?:an?\s+)?actively managed|is\s+(?:an?\s+)?actively managed|actively manages?|employs\s+(?:an?\s+)?active management)\b",
    re.IGNORECASE
)

# Balanced multi-asset / fund-of-funds patterns
BALANCED_OR_FOF_PATTERNS = [
    r"\b(?:balanced approach|balanced fund|core balanced)\b",
    r"\b(?:50%-75% in equity securities, at least 25% in debt)\b",
    r"\b(?:allocation to debt securities through .*? (?:underlying )?funds)\b",
    r"\b(?:invests in (?:one or more|proprietary|underlying) (?:underlying )?funds)\b",
    r"\b(?:fund of funds)\b",
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

    RULESET_ID = "MANDATE_PARSER_V1_2_0_FROZEN"

    @classmethod
    def extract_strategy_text(cls, html_or_text: str, fund_identifier: Optional[str] = None) -> tuple[str, str]:
        """Extract Principal Investment Strategy section text from filing HTML/text."""
        if not html_or_text or len(html_or_text.strip()) == 0:
            return "", ""

        # If html, extract text or parse
        if "<html" in html_or_text.lower() or "<body" in html_or_text.lower() or "<div" in html_or_text.lower():
            soup = BeautifulSoup(html_or_text[:1000000], "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        else:
            text = html_or_text

        # Look for section header with flexible whitespace (\s+)
        patterns = [
            r"Principal\s+Investment\s+Strateg(?:y|ies)",
            r"Investment\s+Objective\s+and\s+Principal\s+Strategies",
            r"Principal\s+Strategies",
            r"Principal\s+Investment\s+Policies\s+and\s+Strategies",
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
        result.evidence_snippet = strategy_text[:200].replace("\n", " ").strip()

        # Step 1: Check for Non-confirmatory Product Structure / Exclusions
        for pat in PRODUCT_EXCLUSION_PATTERNS:
            if re.search(pat, text_lower):
                result.non_confirmatory_mandate = True
                result.parser_rule_id = "RULE_NON_CONFIRMATORY"
                result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
                return result

        # Step 2: Check Geography Invariant (Ex-US / International / Foreign Scope)
        for pat in GEOGRAPHY_EX_US_PATTERNS:
            if re.search(pat, text_lower):
                result.non_confirmatory_mandate = True
                result.parser_rule_id = "RULE_EX_US_OR_INTERNATIONAL"
                result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
                return result

        # Step 3: Check Balanced / Fund-of-Funds Ambiguity
        for pat in BALANCED_OR_FOF_PATTERNS:
            if re.search(pat, text_lower):
                result.parser_rule_id = "RULE_FUND_OF_FUNDS_OR_BALANCED"
                result.confidence_state = "UNRESOLVED_POLICY_EXECUTABILITY"
                return result

        # Step 4: Check Affirmative Active Management
        if ACTIVE_MANAGEMENT_PATTERN.search(strategy_text):
            result.non_confirmatory_mandate = True
            result.parser_rule_id = "RULE_ACTIVE_MANAGEMENT"
            result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
            return result

        # Step 5: Check Treasury Government Mandate
        govt_matches = [pat for pat in TREASURY_PATTERNS if re.search(pat, text_lower)]

        # Step 6: Check Corporate Credit Mandate
        credit_matches = [pat for pat in CORPORATE_CREDIT_PATTERNS if re.search(pat, text_lower)]

        # Step 7: Check Affirmative Approved Sector Mandate
        detected_sectors = []
        for sector, pat_list in AFFIRMATIVE_SECTOR_PATTERNS.items():
            for pat in pat_list:
                if re.search(pat, strategy_text, re.IGNORECASE):
                    detected_sectors.append(sector)
                    break

        # Step 8: Check Broad US Equity Index Mandate
        broad_matches = [pat for pat in BROAD_EQUITY_PATTERNS if re.search(pat, text_lower)]

        # Step 9: Contradictory / Mixed Aggregate Bond Collision
        if len(govt_matches) > 0 and len(credit_matches) > 0:
            result.non_confirmatory_mandate = True
            result.parser_rule_id = "RULE_MIXED_AGGREGATE_BOND"
            result.confidence_state = "CONFIDENT_NON_CONFIRMATORY"
            return result

        # Step 10: Treasury Government (Pure)
        if len(govt_matches) > 0 and len(credit_matches) == 0 and len(detected_sectors) == 0 and len(broad_matches) == 0:
            result.government_debt_mandate = True
            result.parser_rule_id = "RULE_TREASURY_GOVERNMENT"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Step 11: Corporate Credit (Pure)
        if len(credit_matches) > 0 and len(govt_matches) == 0 and len(detected_sectors) == 0 and len(broad_matches) == 0:
            result.corporate_credit_mandate = True
            result.parser_rule_id = "RULE_CORPORATE_CREDIT"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Step 12: Broad Equity Index (Pure or taking precedence over broad multi-sector)
        if len(broad_matches) > 0 and len(detected_sectors) == 0:
            result.broad_or_multi_sector_mandate = True
            result.parser_rule_id = "RULE_BROAD_EQUITY_INDEX"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Step 13: Sector Specific (Single Approved Sector)
        if len(detected_sectors) == 1 and len(broad_matches) == 0:
            result.sector_specific_mandate = True
            result.approved_sector = detected_sectors[0]
            result.parser_rule_id = "RULE_SECTOR_EQUITY"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Step 14: Multiple Approved Sectors
        if len(detected_sectors) > 1 and len(broad_matches) == 0:
            result.broad_or_multi_sector_mandate = True
            result.parser_rule_id = "RULE_MULTI_SECTOR_EQUITY"
            result.confidence_state = "CONFIDENT_CONFIRMATORY"
            return result

        # Step 15: Fail-closed unclassified mandate
        result.confidence_state = "AMBIGUOUS_UNCLASSIFIED"
        result.parser_rule_id = "RULE_FAIL_CLOSED_AMBIGUOUS"
        return result
