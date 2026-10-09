"""ARX VCP Domain Glossary.

Sprint 2B Domain-Authority Resolution.
Explicit, frozen semantic definitions for all VCP and stage analysis terms.
Enforces that every term has an authoritative basis, explicit observations,
prohibited interpretations, and frozen status.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class TermStatus(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    SUPPORTED_INTERPRETATION = "SUPPORTED_INTERPRETATION"
    ARX_EXTENSION = "ARX_EXTENSION"
    UNRESOLVED = "UNRESOLVED"
    PROHIBITED = "PROHIBITED"


@dataclass(frozen=True)
class DomainTermDefinition:
    term_id: str
    term_name: str
    version: str
    semantic_definition: str
    domain_authority_sources: List[str]
    required_observations: List[str]
    prohibited_interpretations: List[str]
    boundary_ambiguity: str
    status: TermStatus


GLOSSARY_TERMS: List[DomainTermDefinition] = [
    DomainTermDefinition(
        term_id="TERM-VCP",
        term_name="VCP",
        version="1.0.0",
        semantic_definition=(
            "Volatility Contraction Pattern: A constructive equity consolidation structure occurring "
            "within an established Stage 2 primary uptrend, characterized by a series of successive "
            "contractions of decreasing percentage depth accompanied by drying volume, leading to supply exhaustion "
            "and an actionable low-risk pivot entry point."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        required_observations=[
            "Stage 2 primary uptrend / Trend Template",
            "At least 2 distinct contraction waves",
            "Progressive contraction in percentage wave depth",
            "Volume dry-up on the final contraction",
            "Identifiable pivot resistance ceiling",
        ],
        prohibited_interpretations=[
            "Treating any random multi-day price pullback as a VCP",
            "Assigning VCP to an asset in a Stage 4 markdown or declining 200-day moving average",
            "Claiming VCP confirmation without verifying volume dry-up",
            "Using future price advance to retroactively declare a valid VCP",
        ],
        boundary_ambiguity="Discretionary visual wave identification in overlapping or choppy intraday noise.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-CONTRACTION",
        term_name="contraction",
        version="1.0.0",
        semantic_definition=(
            "A distinct downward corrective wave or swing within the base, bounded by a local peak (swing high) "
            "and subsequent local reaction trough (swing low), measuring the percentage drawdown from that peak."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=["Swing high price", "Swing low price", "Percentage depth", "Wave duration in bars"],
        prohibited_interpretations=[
            "Single daily bar fluctuations treated as independent contraction waves",
            "Overlapping multi-month unrelated macro cycles treated as single contractions",
        ],
        boundary_ambiguity="Exact bar identifying local trough in flat-bottom consolidations.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-CANDIDATE-CONTRACTION",
        term_name="candidate contraction",
        version="1.0.0",
        semantic_definition=(
            "An algorithmically detected peak-to-trough price wave meeting preliminary duration and depth "
            "filtering before domain-level structural validation."
        ),
        domain_authority_sources=["SRC-ARX-SPEC-2026"],
        required_observations=["Local peak index", "Local trough index", "Raw peak-to-trough range"],
        prohibited_interpretations=["Equating algorithmic candidate detection with authoritative VCP confirmation"],
        boundary_ambiguity="Sensitivity of rolling peak detection window parameter.",
        status=TermStatus.ARX_EXTENSION,
    ),
    DomainTermDefinition(
        term_id="TERM-VALID-CONTRACTION",
        term_name="valid contraction",
        version="1.0.0",
        semantic_definition=(
            "A candidate contraction that satisfies minimum depth (>= 2%), maximum depth (<= 45%), "
            "and minimum duration (>= 4 trading sessions) criteria."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-ARX-SPEC-2026"],
        required_observations=["Validated depth percentage", "Validated duration in trading days"],
        prohibited_interpretations=["Accepting micro-swings (< 2% depth) as macro base contractions"],
        boundary_ambiguity="Transitions between 3% and 2% noise floor.",
        status=TermStatus.SUPPORTED_INTERPRETATION,
    ),
    DomainTermDefinition(
        term_id="TERM-CONTRACTION-SEQUENCE",
        term_name="contraction sequence",
        version="1.0.0",
        semantic_definition=(
            "The ordered chronological set of valid contractions in the base, typically comprising 2 to 4 waves "
            "(denoted 2T, 3T, 4T, up to 6T in rare large bases)."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=["Wave count (T-count)", "Chronological ordering of wave start/end points"],
        prohibited_interpretations=["Permitting non-chronological wave reordering", "Accepting 1-wave pullbacks as VCP"],
        boundary_ambiguity="Determining whether a small pause before breakout counts as a distinct T-wave.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-PROGRESSIVE-TIGHTENING",
        term_name="progressive tightening",
        version="1.0.0",
        semantic_definition=(
            "The mathematical invariant that each successive contraction wave in the sequence exhibits a strictly "
            "smaller percentage depth than the preceding wave: Depth_k < Depth_{k-1}."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=["Sequential wave depths: [Depth_1, Depth_2, ..., Depth_n]"],
        prohibited_interpretations=[
            "Accepting widening formations (megaphones) or expanding volatility as VCP",
            "Allowing an intermediate wave deeper than the initial wave",
        ],
        boundary_ambiguity="Equal-depth waves within 0.25% margin of error.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-VOLATILITY-CONTRACTION",
        term_name="volatility contraction",
        version="1.0.0",
        semantic_definition=(
            "The macro phenomenon where the absolute price range over rolling windows and average true range "
            "substantially compress as the consolidation matures toward the right side of the base."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=["ATR compression ratio", "Rolling standard deviation compression"],
        prohibited_interpretations=["Confusing low liquidity/inactivity with volatility contraction"],
        boundary_ambiguity="Lookback window length for volatility baseline.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-VOLUME-DRY-UP",
        term_name="volume dry-up",
        version="1.0.0",
        semantic_definition=(
            "A marked reduction in trading volume during pullbacks within the base, and especially on the "
            "final contraction wave, reaching levels significantly below the 50-day average volume."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-ONEIL-2009"],
        required_observations=["Daily volume", "50-day SMA of volume", "Final wave average volume ratio"],
        prohibited_interpretations=["Allowing heavy selling volume during the final contraction wave"],
        boundary_ambiguity="Holiday-shortened or half-day trading volume distortion.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-PRIOR-TREND",
        term_name="prior trend",
        version="1.0.0",
        semantic_definition=(
            "A preceding advance of at least +30% (ideally +100% or more) from a prior base or major low "
            "establishing that the base is a continuation consolidation rather than a bottom reversal attempt."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=["Prior run-up magnitude percentage", "Prior run-up duration in months"],
        prohibited_interpretations=["Seeking VCP in an asset with no prior primary uptrend"],
        boundary_ambiguity="Exact starting point of the prior impulse run.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-TREND-TEMPLATE",
        term_name="trend template",
        version="1.0.0",
        semantic_definition=(
            "Mark Minervini's 8-criteria technical trend filter: 1) Price > 150 SMA & 200 SMA; 2) 150 SMA > 200 SMA; "
            "3) 200 SMA trending up >= 22 sessions; 4) 50 SMA > 150 & 200 SMA; 5) Price > 50 SMA; "
            "6) Price >= 30% above 52-week low; 7) Price within 25% of 52-week high; 8) RS rank >= 70 if available."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013"],
        required_observations=[
            "Close price", "50 SMA", "150 SMA", "200 SMA", "200 SMA 22-day slope",
            "52-week High", "52-week Low",
        ],
        prohibited_interpretations=[
            "Accepting assets trading below their 200-day moving average",
            "Accepting assets with declining 200-day moving averages",
        ],
        boundary_ambiguity="Whipsaws around 50 SMA during base pullbacks.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-PIVOT",
        term_name="pivot",
        version="1.0.0",
        semantic_definition=(
            "The specific resistance price level at the high of the final narrow contraction wave ('cheat' or handle high) "
            "where an upward penetration indicates supply absorption and triggers a low-risk tactical buy point."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-MINERVINI-2017"],
        required_observations=["Final wave swing high price", "Proximity of current price to pivot level"],
        prohibited_interpretations=[
            "Arbitrary round numbers treated as pivots",
            "Using moving average levels as breakout pivots when price is far below base high",
        ],
        boundary_ambiguity="Distinguishing high-handle pivot from absolute base high.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-BASE",
        term_name="base",
        version="1.0.0",
        semantic_definition=(
            "A multi-week price consolidation area (typically 3 to 45 weeks) where institutional investors absorb "
            "floating supply following an advance, prior to resuming the Stage 2 uptrend."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-ONEIL-2009", "SRC-WEINSTEIN-1988"],
        required_observations=["Base duration in weeks", "Base depth (peak-to-trough)", "Base high and base low"],
        prohibited_interpretations=["Calling a 3-day pullback a base"],
        boundary_ambiguity="Base-on-base formations where a second base begins before breakout from first.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-BREAKOUT",
        term_name="breakout",
        version="1.0.0",
        semantic_definition=(
            "The decisive upward price cross above the pivot point on expanding volume (typically >= 40-50% "
            "above 50-day average volume), signaling the conclusion of the consolidation."
        ),
        domain_authority_sources=["SRC-MINERVINI-2013", "SRC-ONEIL-2009"],
        required_observations=["Cross above pivot price", "Day's volume relative to 50-day average"],
        prohibited_interpretations=["Declaring breakout on weak or below-average volume"],
        boundary_ambiguity="Intraday breakout that closes back below pivot on heavy volume (reversal).",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-FAILED-SETUP",
        term_name="failed setup",
        version="1.0.0",
        semantic_definition=(
            "A setup where price breaks below the lower boundary of the final contraction wave, breaches the 50-day SMA "
            "on heavy volume, or falls more than 7-8% below the pivot point post-entry."
        ),
        domain_authority_sources=["SRC-MINERVINI-2017"],
        required_observations=["Stop loss price level", "Invalidation event trigger"],
        prohibited_interpretations=["Holding through severe breakdowns in hopes of recovery"],
        boundary_ambiguity="Brief false breakdown that immediately reverses back into base (shakeout).",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-STAGE-1",
        term_name="Stage 1",
        version="1.0.0",
        semantic_definition=(
            "Weinstein Basing Phase: A sideways accumulation channel following a prolonged downtrend; "
            "the 30-week / 200-day moving average flattens out after declining; price oscillates around the flat average."
        ),
        domain_authority_sources=["SRC-WEINSTEIN-1988"],
        required_observations=["Prior downtrend", "Flat 200 SMA (slope near zero)", "Sideways price channel"],
        prohibited_interpretations=["Initiating long breakout positions before Stage 2 breakout occurs"],
        boundary_ambiguity="Transition zone between late Stage 4 and early Stage 1.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-STAGE-2",
        term_name="Stage 2",
        version="1.0.0",
        semantic_definition=(
            "Weinstein / Minervini Advancing Phase: Secular bull trend where price trades above a rising "
            "30-week / 200-day moving average, creating higher highs and higher lows on expanding volume on rallies."
        ),
        domain_authority_sources=["SRC-WEINSTEIN-1988", "SRC-MINERVINI-2013"],
        required_observations=[
            "Price above rising 200 SMA", "Rising 200 SMA slope", "Expanding volume on up moves",
        ],
        prohibited_interpretations=["Calling a stock in a declining 200 SMA Stage 2"],
        boundary_ambiguity="Deep pullbacks to testing 200 SMA during general market corrections.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-STAGE-3",
        term_name="Stage 3",
        version="1.0.0",
        semantic_definition=(
            "Weinstein Topping Phase: Distribution area where upward momentum stalls, volatility increases, "
            "the 30-week / 200-day moving average begins to flatten, and price swings wildly with churning volume."
        ),
        domain_authority_sources=["SRC-WEINSTEIN-1988"],
        required_observations=["Flattening 200 SMA after advance", "Increased volatility", "Churning volume"],
        prohibited_interpretations=["Interpreting high-volatility wide swings as constructive VCP bases"],
        boundary_ambiguity="Differentiating Stage 3 distribution from deep constructive consolidations.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-STAGE-4",
        term_name="Stage 4",
        version="1.0.0",
        semantic_definition=(
            "Weinstein Declining Phase: Secular bear trend where price trades below a declining "
            "30-week / 200-day moving average, making lower highs and lower lows. Long positions strictly prohibited."
        ),
        domain_authority_sources=["SRC-WEINSTEIN-1988"],
        required_observations=["Price below declining 200 SMA", "Negative 200 SMA slope", "Lower lows"],
        prohibited_interpretations=["Searching for VCP long setups in Stage 4 assets"],
        boundary_ambiguity="Violent bear market rallies testing declining 50 SMA or 200 SMA.",
        status=TermStatus.AUTHORITATIVE,
    ),
    DomainTermDefinition(
        term_id="TERM-INSUFFICIENT-EVIDENCE",
        term_name="insufficient evidence",
        version="1.0.0",
        semantic_definition=(
            "Evaluation outcome when the available historical session data at evaluation_as_of is fewer than "
            "required lookback thresholds (e.g., < 200 sessions for 200 SMA, < 50 sessions for volume analysis), "
            "requiring fail-closed abstention from domain classification."
        ),
        domain_authority_sources=["SRC-ARX-SPEC-2026"],
        required_observations=["Available session count", "Required lookback threshold"],
        prohibited_interpretations=["Coercing missing historical data to FALSE or TRUE"],
        boundary_ambiguity="Assets with exactly 199 vs 200 sessions.",
        status=TermStatus.ARX_EXTENSION,
    ),
    DomainTermDefinition(
        term_id="TERM-DOMAIN-UNRESOLVED",
        term_name="domain unresolved",
        version="1.0.0",
        semantic_definition=(
            "Evaluation outcome when data is complete, but contradictory structural patterns or unadjudicated boundary "
            "ambiguities prevent an affirmative domain classification under frozen predicates."
        ),
        domain_authority_sources=["SRC-ARX-SPEC-2026"],
        required_observations=["Predicate evaluation vector containing unresolved states"],
        prohibited_interpretations=["Coercing unresolved domain state to affirmative PASS or FAIL"],
        boundary_ambiguity="Borderline cases where wave count or depth contraction is borderline.",
        status=TermStatus.ARX_EXTENSION,
    ),
    DomainTermDefinition(
        term_id="TERM-NOT-APPLICABLE",
        term_name="not applicable",
        version="1.0.0",
        semantic_definition=(
            "Evaluation outcome for predicates whose preconditions are unsatisfied (e.g., evaluating pivot proximity "
            "when no valid contraction sequence exists)."
        ),
        domain_authority_sources=["SRC-ARX-SPEC-2026"],
        required_observations=["Status of prerequisite predicates"],
        prohibited_interpretations=["Evaluating downstream execution rules on non-qualifying setups"],
        boundary_ambiguity="None.",
        status=TermStatus.ARX_EXTENSION,
    ),
]


class VCPDomainGlossary:
    """Singleton authority glossary for VCP domain terminology."""

    GLOSSARY_ID = "ARX_VCP_DOMAIN_GLOSSARY"
    VERSION = "1.0.0"

    def __init__(self, terms: Optional[List[DomainTermDefinition]] = None):
        self.terms: Dict[str, DomainTermDefinition] = {
            t.term_id: t for t in (terms or GLOSSARY_TERMS)
        }
        self._name_index: Dict[str, DomainTermDefinition] = {
            t.term_name.strip().lower(): t for t in self.terms.values()
        }

    def get_term(self, term_id: str) -> Optional[DomainTermDefinition]:
        return self.terms.get(term_id)

    def find_by_name(self, name: str) -> Optional[DomainTermDefinition]:
        return self._name_index.get(name.strip().lower())

    def list_terms(self) -> List[DomainTermDefinition]:
        return sorted(self.terms.values(), key=lambda t: t.term_id)

    def count_terms_without_definition(self) -> int:
        count = 0
        for t in self.terms.values():
            if not t.semantic_definition or len(t.semantic_definition.strip()) == 0:
                count += 1
        return count

    def compute_glossary_hash(self) -> str:
        serialized = []
        for t in sorted(self.terms.values(), key=lambda x: x.term_id):
            serialized.append({
                "term_id": t.term_id,
                "term_name": t.term_name,
                "version": t.version,
                "def": t.semantic_definition,
                "sources": sorted(t.domain_authority_sources),
                "obs": sorted(t.required_observations),
                "proh": sorted(t.prohibited_interpretations),
                "amb": t.boundary_ambiguity,
                "status": t.status.value,
            })
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "glossary_id": self.GLOSSARY_ID,
            "version": self.VERSION,
            "hash": self.compute_glossary_hash(),
            "terms": [asdict(t) for t in self.list_terms()],
        }
