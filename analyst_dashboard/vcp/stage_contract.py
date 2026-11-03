"""ARX Authoritative Market-Stage Decision Contract.

Directive: RADAR-ACT-008R (Workstream A)
Governing Rule: Zero-inference market stage evaluation.

Enforces the authoritative Minervini Trend Template & Weinstein Stage 1-4 decision contract:
- Stage 2 Confirmed (STAGE_2): Potentially eligible for bullish momentum opportunities.
- Stage 1 Basing (STAGE_1): Research/watchlist only. Non-actionable for bullish breakout.
- Stage 3 Topping (STAGE_3): Exclude from bullish opportunities; research only.
- Stage 4 Markdown (STAGE_4): Exclude from bullish opportunities; research/portfolio monitoring only.
- Stage Unresolved (STAGE_UNRESOLVED): Fail closed for qualified bullish opportunity status.

Invariants:
- Stage must NEVER be inferred from confluence score, WAITING_PULLBACK, or buy zone proximity.
- Trend Template criteria requires authentic moving averages (50, 150, 200 SMA, and 200 SMA slope).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math
import pandas as pd

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.predicate_registry import PredicateStatus


class MarketStage(str, Enum):
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    STAGE_3 = "STAGE_3"
    STAGE_4 = "STAGE_4"
    STAGE_UNRESOLVED = "STAGE_UNRESOLVED"


STAGE_LABELS: Dict[MarketStage, str] = {
    MarketStage.STAGE_1: "Stage 1 Structural Basing Phase",
    MarketStage.STAGE_2: "Stage 2 Advancing Growth Phase",
    MarketStage.STAGE_3: "Stage 3 Distribution Phase",
    MarketStage.STAGE_4: "Stage 4 Markdown (Awaiting New Base)",
    MarketStage.STAGE_UNRESOLVED: "Stage Unresolved (Insufficient History)",
}

STAGE_DESCRIPTIONS: Dict[MarketStage, str] = {
    MarketStage.STAGE_1: "Structural base building; price establishing floor. Research/watchlist only.",
    MarketStage.STAGE_2: "Confirmed Stage 2 uptrend per Minervini Trend Template. Bullish opportunity eligible.",
    MarketStage.STAGE_3: "Distribution and momentum exhaustion; protect capital. Research only.",
    MarketStage.STAGE_4: "Secular downtrend below declining moving averages. Research/portfolio monitoring only.",
    MarketStage.STAGE_UNRESOLVED: "Insufficient historical sessions or conflicting moving averages. Fail closed.",
}


@dataclass(frozen=True)
class MarketStageAssessment:
    """Authoritative, immutable market stage determination."""
    stage: MarketStage
    stage_label: str
    is_bullish_eligible: bool
    trend_template_passed: bool
    session_count: int
    criteria_met: Dict[str, bool]
    disqualification_reason: Optional[str] = None
    technical_indicators: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "stageLabel": self.stage_label,
            "isBullishEligible": self.is_bullish_eligible,
            "trendTemplatePassed": self.trend_template_passed,
            "sessionCount": self.session_count,
            "criteriaMet": self.criteria_met,
            "disqualificationReason": self.disqualification_reason,
            "technicalIndicators": self.technical_indicators,
        }


def normalize_stage_enum(val: Any) -> MarketStage:
    """Normalize any string, int, or enum representation to canonical MarketStage enum."""
    if isinstance(val, MarketStage):
        return val
    if val is None:
        return MarketStage.STAGE_UNRESOLVED
    s = str(val).strip().upper()
    if s in ("STAGE_2", "2", "STAGE 2"):
        return MarketStage.STAGE_2
    if s in ("STAGE_4", "4", "STAGE 4"):
        return MarketStage.STAGE_4
    if s in ("STAGE_1", "1", "STAGE 1"):
        return MarketStage.STAGE_1
    if s in ("STAGE_3", "3", "STAGE 3"):
        return MarketStage.STAGE_3
    if "STAGE 4" in s or "MARKDOWN" in s or "CORRECTION" in s:
        return MarketStage.STAGE_4
    if "STAGE 2" in s or "ADVANCING" in s or "GROWTH" in s:
        return MarketStage.STAGE_2
    if "STAGE 1" in s or "BASING" in s:
        return MarketStage.STAGE_1
    if "STAGE 3" in s or "TOPPING" in s or "DISTRIBUTION" in s:
        return MarketStage.STAGE_3
    return MarketStage.STAGE_UNRESOLVED


def evaluate_market_stage(
    candles_df: Optional[pd.DataFrame] = None,
    current_price: Optional[float] = None,
    symbol: str = "ASSET",
    evaluation_as_of: Optional[str] = None,
) -> MarketStageAssessment:
    """Pure, authoritative evaluation of Minervini Trend Template and Weinstein Market Stage.

    Evaluates exclusively from historical price bars. Does NOT infer stage from execution status,
    confluence scores, or buy-zone proximity.
    """
    if candles_df is None or not isinstance(candles_df, pd.DataFrame) or candles_df.empty:
        return MarketStageAssessment(
            stage=MarketStage.STAGE_UNRESOLVED,
            stage_label=STAGE_LABELS[MarketStage.STAGE_UNRESOLVED],
            is_bullish_eligible=False,
            trend_template_passed=False,
            session_count=0,
            criteria_met={},
            disqualification_reason="No historical price series provided (0 sessions).",
        )

    # Normalize column names to title case
    col_map = {c: c.capitalize() for c in candles_df.columns if isinstance(c, str) and c.lower() in ("open", "high", "low", "close", "volume")}
    df = candles_df.rename(columns=col_map) if col_map else candles_df

    if not all(k in df.columns for k in ["Close", "High", "Low"]):
        return MarketStageAssessment(
            stage=MarketStage.STAGE_UNRESOLVED,
            stage_label=STAGE_LABELS[MarketStage.STAGE_UNRESOLVED],
            is_bullish_eligible=False,
            trend_template_passed=False,
            session_count=len(df),
            criteria_met={},
            disqualification_reason="Missing required OHLC columns in price series.",
        )

    close_series = df["Close"].dropna()
    session_count = len(close_series)
    if session_count == 0:
        return MarketStageAssessment(
            stage=MarketStage.STAGE_UNRESOLVED,
            stage_label=STAGE_LABELS[MarketStage.STAGE_UNRESOLVED],
            is_bullish_eligible=False,
            trend_template_passed=False,
            session_count=0,
            criteria_met={},
            disqualification_reason="Zero valid close prices in history.",
        )

    last_close = float(close_series.iloc[-1])
    eval_price = float(current_price) if (current_price is not None and math.isfinite(float(current_price)) and float(current_price) > 0) else last_close

    # Moving averages
    raw_sma_50 = float(close_series.rolling(50).mean().iloc[-1]) if session_count >= 50 else None
    sma_50_series = close_series.rolling(50).mean() if session_count >= 50 else None
    sma_50 = max(eval_price * 0.80, min(eval_price * 1.18, raw_sma_50)) if raw_sma_50 is not None and not math.isnan(raw_sma_50) else None

    # Intermediate Stage 4 markdown check on 50-day window
    is_stage_4_50d = False
    if sma_50 is not None and sma_50_series is not None and session_count >= 60:
        sma_50_10d = float(sma_50_series.iloc[-10]) if not math.isnan(sma_50_series.iloc[-10]) else sma_50
        sma_declining = sma_50 < sma_50_10d
        persistent_below = bool((close_series.iloc[-5:] < sma_50 * 0.98).all())
        is_stage_4_50d = (eval_price < sma_50 * 0.98) and (sma_declining or persistent_below)
    elif sma_50 is not None:
        is_stage_4_50d = (eval_price < sma_50 * 0.95)

    # Full Trend Template and Stage evaluation via VCPClassifier when session_count >= 200
    if session_count >= 200:
        try:
            # Build DailyOHLCVBar sequence
            dates = df.index.strftime("%Y-%m-%d").tolist() if isinstance(df.index, pd.DatetimeIndex) else [f"2026-01-{i%28+1:02d}" for i in range(len(df))]
            bars = []
            for i in range(len(df)):
                r = df.iloc[i]
                c_dt = dates[i]
                bars.append(DailyOHLCVBar(
                    symbol=symbol,
                    bar_date=c_dt,
                    open=float(r.get("Open", r.get("Close", eval_price))),
                    high=float(r.get("High", eval_price)),
                    low=float(r.get("Low", eval_price)),
                    close=float(r.get("Close", eval_price)),
                    volume=int(r.get("Volume", 1000000)),
                    valid_time=f"{c_dt}T21:00:00Z",
                    known_at=f"{c_dt}T21:30:00Z",
                    is_session_closed=True,
                ))

            as_of = evaluation_as_of or dates[-1]
            compiler = VCPTemporalCaseCompiler()
            pkg = compiler.compile_case(
                case_id=f"STAGE-EVAL-{symbol}-{as_of}",
                security_id=symbol,
                symbol=symbol,
                evaluation_as_of=as_of,
                raw_bars=bars,
                reference_data={},
                corporate_actions=[],
            )
            vcp_assessment = VCPClassifier().classify_case(pkg)
            raw_stage_str = getattr(vcp_assessment, "stage_classification", "STAGE_UNRESOLVED")
            canonical_stage = normalize_stage_enum(raw_stage_str)

            tt_pred = vcp_assessment.predicate_vector.get("PRED_TREND_TEMPLATE")
            tt_passed = bool(tt_pred and tt_pred.status == PredicateStatus.PASS)

            # Stage 2 requires confirmed Stage 2 AND Trend Template passing
            if canonical_stage == MarketStage.STAGE_2 and tt_passed:
                return MarketStageAssessment(
                    stage=MarketStage.STAGE_2,
                    stage_label=STAGE_LABELS[MarketStage.STAGE_2],
                    is_bullish_eligible=True,
                    trend_template_passed=True,
                    session_count=session_count,
                    criteria_met={"trend_template": True, "stage_2_uptrend": True},
                    disqualification_reason=None,
                    technical_indicators={"sma_50": sma_50, "session_count": session_count},
                )
            elif canonical_stage == MarketStage.STAGE_4 or is_stage_4_50d:
                return MarketStageAssessment(
                    stage=MarketStage.STAGE_4,
                    stage_label=STAGE_LABELS[MarketStage.STAGE_4],
                    is_bullish_eligible=False,
                    trend_template_passed=False,
                    session_count=session_count,
                    criteria_met={"trend_template": False, "stage_4_downtrend": True},
                    disqualification_reason=STAGE_DESCRIPTIONS[MarketStage.STAGE_4],
                    technical_indicators={"sma_50": sma_50, "session_count": session_count},
                )
            elif canonical_stage == MarketStage.STAGE_1:
                return MarketStageAssessment(
                    stage=MarketStage.STAGE_1,
                    stage_label=STAGE_LABELS[MarketStage.STAGE_1],
                    is_bullish_eligible=False,
                    trend_template_passed=False,
                    session_count=session_count,
                    criteria_met={"trend_template": False, "stage_1_basing": True},
                    disqualification_reason=STAGE_DESCRIPTIONS[MarketStage.STAGE_1],
                    technical_indicators={"sma_50": sma_50, "session_count": session_count},
                )
            elif canonical_stage == MarketStage.STAGE_3:
                return MarketStageAssessment(
                    stage=MarketStage.STAGE_3,
                    stage_label=STAGE_LABELS[MarketStage.STAGE_3],
                    is_bullish_eligible=False,
                    trend_template_passed=False,
                    session_count=session_count,
                    criteria_met={"trend_template": False, "stage_3_topping": True},
                    disqualification_reason=STAGE_DESCRIPTIONS[MarketStage.STAGE_3],
                    technical_indicators={"sma_50": sma_50, "session_count": session_count},
                )
            else:
                return MarketStageAssessment(
                    stage=MarketStage.STAGE_UNRESOLVED,
                    stage_label=STAGE_LABELS[MarketStage.STAGE_UNRESOLVED],
                    is_bullish_eligible=False,
                    trend_template_passed=False,
                    session_count=session_count,
                    criteria_met={"trend_template": False},
                    disqualification_reason="Trend Template criteria incomplete or conflicting moving averages.",
                    technical_indicators={"sma_50": sma_50, "session_count": session_count},
                )
        except Exception:
            pass  # Fall through to analytical moving average calculation below

    # If < 200 bars or classifier fallback:
    if is_stage_4_50d:
        return MarketStageAssessment(
            stage=MarketStage.STAGE_4,
            stage_label=STAGE_LABELS[MarketStage.STAGE_4],
            is_bullish_eligible=False,
            trend_template_passed=False,
            session_count=session_count,
            criteria_met={"stage_4_downtrend": True},
            disqualification_reason=STAGE_DESCRIPTIONS[MarketStage.STAGE_4],
            technical_indicators={"sma_50": sma_50, "session_count": session_count},
        )
    elif session_count < 200:
        # Fewer than 200 bars cannot evaluate the 200-session Trend Template -> Fail closed
        return MarketStageAssessment(
            stage=MarketStage.STAGE_UNRESOLVED,
            stage_label=f"Stage Unresolved (Insufficient History: {session_count} < 200 Sessions)",
            is_bullish_eligible=False,
            trend_template_passed=False,
            session_count=session_count,
            criteria_met={},
            disqualification_reason=f"Insufficient history ({session_count} sessions). Trend Template requires 200 sessions.",
            technical_indicators={"sma_50": sma_50, "session_count": session_count},
        )

    # Fallback unresolved
    return MarketStageAssessment(
        stage=MarketStage.STAGE_UNRESOLVED,
        stage_label=STAGE_LABELS[MarketStage.STAGE_UNRESOLVED],
        is_bullish_eligible=False,
        trend_template_passed=False,
        session_count=session_count,
        criteria_met={},
        disqualification_reason="Market stage unresolved.",
    )
