"""
analyst_dashboard/governance/paper_trading_outcome_evaluator.py

Evaluates paper-trading recommendation outcomes under the frozen
ARX Paper-Trading Outcome Evaluation Contract V1.0.1 (docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json).

Strict fail-closed execution-sequencing, intrabar collision isolation,
and multi-tier market data authority.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class EvaluationResult:
    symbol: str
    engine_version: str
    signal_date: str
    entry_state: str
    entry_date: Optional[str]
    entry_price: Optional[float]
    outcome: str
    exit_date: Optional[str]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    gross_return_pct: Optional[float]
    net_simulated_return_pct: Optional[float]
    r_multiple: Optional[float]
    mfe: Optional[float]
    mae: Optional[float]
    confluence_score: float
    market_regime: str
    provider: str
    intrabar_resolution_evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaperTradingOutcomeEvaluator:
    """Evaluates paper-trading recommendations under Contract V1.0.1."""

    FRICTION_BPS = 25.0
    NET_FRICTION_PCT = 0.25
    MAX_ENTRY_SESSIONS = 5
    MAX_HOLDING_SESSIONS = 20

    @classmethod
    def evaluate_signal(
        cls,
        signal: Dict[str, Any],
        daily_bars: pd.DataFrame,
        intraday_bars: Optional[pd.DataFrame] = None,
        provider: str = "YAHOO_FINANCE",
        provider_conflict: bool = False,
    ) -> EvaluationResult:
        """
        Evaluates a single recorded signal against price bars.
        
        daily_bars: DataFrame with columns ['Open', 'High', 'Low', 'Close'], index as DatetimeIndex or date string.
        intraday_bars: Optional DataFrame with time-ordered sub-daily bars for collision resolution.
        """
        sym = signal.get("symbol", "")
        ver = signal.get("engineVersion", signal.get("metadata", {}).get("engine_version", "2.4.0"))
        sig_date = signal.get("signalDate", "")
        c_min = float(signal.get("corridorMin", 0.0))
        c_max = float(signal.get("corridorMax", 0.0))
        stop = float(signal.get("stopLoss", 0.0))
        tp1 = float(signal.get("takeProfit1", 0.0))
        score = float(signal.get("confluenceScore", 0.0))
        regime = signal.get("marketRegime", "UNKNOWN")

        # Provider conflict check
        if provider_conflict:
            return EvaluationResult(
                symbol=sym,
                engine_version=ver,
                signal_date=sig_date,
                entry_state="ENTRY_DATA_UNRESOLVED",
                entry_date=None,
                entry_price=None,
                outcome="UNRESOLVED_DATA_CONFLICT",
                exit_date=None,
                exit_price=None,
                exit_reason="DATA_CONFLICT",
                gross_return_pct=None,
                net_simulated_return_pct=None,
                r_multiple=None,
                mfe=None,
                mae=None,
                confluence_score=score,
                market_regime=regime,
                provider=provider,
                intrabar_resolution_evidence="Primary and fallback providers materially disagree on prices.",
            )

        # Filter subsequent bars strictly after signal date
        subsequent = daily_bars[daily_bars.index > sig_date].copy()
        if len(subsequent) == 0:
            return EvaluationResult(
                symbol=sym,
                engine_version=ver,
                signal_date=sig_date,
                entry_state="ENTRY_DATA_UNRESOLVED",
                entry_date=None,
                entry_price=None,
                outcome="OUTCOME_DATA_MISSING",
                exit_date=None,
                exit_price=None,
                exit_reason="DATA_MISSING",
                gross_return_pct=None,
                net_simulated_return_pct=None,
                r_multiple=None,
                mfe=None,
                mae=None,
                confluence_score=score,
                market_regime=regime,
                provider=provider,
            )

        # -------------------------------------------------------------
        # 1. ENTRY EVALUATION (Up to 5 sessions)
        # -------------------------------------------------------------
        entry_triggered = False
        entry_date = None
        entry_price = None
        entry_session_idx = None
        intrabar_evidence = None

        t1_row = subsequent.iloc[0]
        t1_date_str = subsequent.index[0].strftime("%Y-%m-%d") if hasattr(subsequent.index[0], "strftime") else str(subsequent.index[0])[:10]
        t1_o, t1_h, t1_l = float(t1_row["Open"]), float(t1_row["High"]), float(t1_row["Low"])

        # Check T+1 Opening Gap below stop floor
        if t1_o <= stop:
            return EvaluationResult(
                symbol=sym,
                engine_version=ver,
                signal_date=sig_date,
                entry_state="ENTRY_NOT_TRIGGERED",
                entry_date=None,
                entry_price=None,
                outcome="ENTRY_NOT_TRIGGERED",
                exit_date=None,
                exit_price=None,
                exit_reason="PRE_ENTRY_STOP_INVALIDATION_GAP_DOWN",
                gross_return_pct=None,
                net_simulated_return_pct=None,
                r_multiple=None,
                mfe=None,
                mae=None,
                confluence_score=score,
                market_regime=regime,
                provider=provider,
                intrabar_resolution_evidence="T+1 Open <= stopLoss; setup invalidated at market open.",
            )

        # T+1 Open in Corridor
        if c_min <= t1_o <= c_max:
            # Same-bar entry + stop check on T+1
            if t1_l <= stop:
                # Check intraday resolution
                if intraday_bars is not None and len(intraday_bars) > 0:
                    # Look at intraday ordering
                    first_bar = intraday_bars.iloc[0]
                    first_l = float(first_bar["Low"])
                    if first_l > stop:
                        # Entered at open, stop hit later
                        entry_triggered = True
                        entry_date = t1_date_str
                        entry_price = t1_o
                        entry_session_idx = 0
                        intrabar_evidence = "Intraday bars confirm entry at 09:30 open before stop breach."
                    else:
                        return EvaluationResult(
                            symbol=sym,
                            engine_version=ver,
                            signal_date=sig_date,
                            entry_state="UNRESOLVED_INTRABAR_SEQUENCE",
                            entry_date=None,
                            entry_price=None,
                            outcome="UNRESOLVED_INTRABAR_SEQUENCE",
                            exit_date=t1_date_str,
                            exit_price=None,
                            exit_reason="INTRABAR_ENTRY_STOP_COLLISION",
                            gross_return_pct=None,
                            net_simulated_return_pct=None,
                            r_multiple=None,
                            mfe=None,
                            mae=None,
                            confluence_score=score,
                            market_regime=regime,
                            provider=provider,
                            intrabar_resolution_evidence="First intraday bar touched both open corridor and stop floor.",
                        )
                else:
                    # Without intraday data, same-bar entry + stop is unresolved
                    return EvaluationResult(
                        symbol=sym,
                        engine_version=ver,
                        signal_date=sig_date,
                        entry_state="UNRESOLVED_INTRABAR_SEQUENCE",
                        entry_date=None,
                        entry_price=None,
                        outcome="UNRESOLVED_INTRABAR_SEQUENCE",
                        exit_date=t1_date_str,
                        exit_price=None,
                        exit_reason="INTRABAR_ENTRY_STOP_COLLISION",
                        gross_return_pct=None,
                        net_simulated_return_pct=None,
                        r_multiple=None,
                        mfe=None,
                        mae=None,
                        confluence_score=score,
                        market_regime=regime,
                        provider=provider,
                        intrabar_resolution_evidence="Daily OHLC touches both entry corridor and stop on T+1; sub-daily sequence unknown.",
                    )
            else:
                entry_triggered = True
                entry_date = t1_date_str
                entry_price = t1_o
                entry_session_idx = 0

        # T+1 Open below Corridor, rallies in
        elif t1_o < c_min and t1_h >= c_min and t1_l > stop:
            entry_triggered = True
            entry_date = t1_date_str
            entry_price = c_min
            entry_session_idx = 0

        # T+1 Open above Corridor, pulls back in
        elif t1_o > c_max and t1_l <= c_max:
            if t1_l <= stop:
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="UNRESOLVED_INTRABAR_SEQUENCE",
                    entry_date=None,
                    entry_price=None,
                    outcome="UNRESOLVED_INTRABAR_SEQUENCE",
                    exit_date=t1_date_str,
                    exit_price=None,
                    exit_reason="INTRABAR_ENTRY_STOP_COLLISION",
                    gross_return_pct=None,
                    net_simulated_return_pct=None,
                    r_multiple=None,
                    mfe=None,
                    mae=None,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence="T+1 bar pulled back through corridor and pierced stop on same session without tick order.",
                )
            else:
                entry_triggered = True
                entry_date = t1_date_str
                entry_price = c_max
                entry_session_idx = 0

        # Check subsequent sessions T+2 to T+5
        if not entry_triggered:
            window_limit = min(cls.MAX_ENTRY_SESSIONS, len(subsequent))
            for s_idx in range(1, window_limit):
                s_row = subsequent.iloc[s_idx]
                s_date_str = subsequent.index[s_idx].strftime("%Y-%m-%d") if hasattr(subsequent.index[s_idx], "strftime") else str(subsequent.index[s_idx])[:10]
                so, sh, sl = float(s_row["Open"]), float(s_row["High"]), float(s_row["Low"])

                # Stop reached prior to entry in session s_idx
                if sl <= stop:
                    return EvaluationResult(
                        symbol=sym,
                        engine_version=ver,
                        signal_date=sig_date,
                        entry_state="ENTRY_NOT_TRIGGERED",
                        entry_date=None,
                        entry_price=None,
                        outcome="ENTRY_NOT_TRIGGERED",
                        exit_date=None,
                        exit_price=None,
                        exit_reason="PRE_ENTRY_STOP_INVALIDATION",
                        gross_return_pct=None,
                        net_simulated_return_pct=None,
                        r_multiple=None,
                        mfe=None,
                        mae=None,
                        confluence_score=score,
                        market_regime=regime,
                        provider=provider,
                        intrabar_resolution_evidence=f"Price breached stopLoss {stop} on session {s_idx+1} ({s_date_str}) before entry could occur.",
                    )

                # Corridor touched
                if sh >= c_min and sl <= c_max:
                    entry_triggered = True
                    entry_date = s_date_str
                    entry_session_idx = s_idx
                    entry_price = c_min if so < c_min else (c_max if so > c_max else so)
                    break

        if not entry_triggered:
            if len(subsequent) < cls.MAX_ENTRY_SESSIONS:
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_WINDOW_STILL_OPEN",
                    entry_date=None,
                    entry_price=None,
                    outcome="ENTRY_WINDOW_STILL_OPEN",
                    exit_date=None,
                    exit_price=None,
                    exit_reason="WINDOW_OPEN",
                    gross_return_pct=None,
                    net_simulated_return_pct=None,
                    r_multiple=None,
                    mfe=None,
                    mae=None,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                )
            else:
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_NOT_TRIGGERED",
                    entry_date=None,
                    entry_price=None,
                    outcome="ENTRY_NOT_TRIGGERED",
                    exit_date=None,
                    exit_price=None,
                    exit_reason="ENTRY_WINDOW_EXPIRED",
                    gross_return_pct=None,
                    net_simulated_return_pct=None,
                    r_multiple=None,
                    mfe=None,
                    mae=None,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence="5 trading sessions elapsed without satisfying entry corridor criteria.",
                )

        # -------------------------------------------------------------
        # 2. HOLDING HORIZON EVALUATION (Up to 20 sessions post-entry)
        # -------------------------------------------------------------
        holding_bars = subsequent.iloc[entry_session_idx:]
        max_high = -1e9
        min_low = 1e9

        for h_idx, (h_dt, h_row) in enumerate(holding_bars.iterrows(), 1):
            h_date_str = h_dt.strftime("%Y-%m-%d") if hasattr(h_dt, "strftime") else str(h_dt)[:10]
            ho, hh, hl, hc = float(h_row["Open"]), float(h_row["High"]), float(h_row["Low"]), float(h_row["Close"])

            max_high = max(max_high, hh)
            min_low = min(min_low, hl)

            tp_hit = (hh >= tp1)
            stop_hit = (hl <= stop)

            if tp_hit and stop_hit:
                # Same-bar TP1 + Stop collision
                mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
                mae = round(((min_low - entry_price) / entry_price) * 100.0, 2)
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_TRIGGERED",
                    entry_date=entry_date,
                    entry_price=entry_price,
                    outcome="UNRESOLVED_INTRABAR_SEQUENCE",
                    exit_date=h_date_str,
                    exit_price=None,
                    exit_reason="INTRABAR_TP1_STOP_COLLISION",
                    gross_return_pct=None,
                    net_simulated_return_pct=None,
                    r_multiple=None,
                    mfe=mfe,
                    mae=mae,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence=f"Session {h_date_str} breached both TP1 ({tp1}) and Stop ({stop}) without tick sequence.",
                )

            if tp_hit:
                # Target 1 reached before stop
                exit_price = tp1 if ho < tp1 else ho  # Gap above target fills at Open
                gross_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
                net_pct = round(gross_pct - cls.NET_FRICTION_PCT, 2)
                r_mult = round((exit_price - entry_price) / (entry_price - stop), 2)
                mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
                mae = round(((min_low - entry_price) / entry_price) * 100.0, 2)
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_TRIGGERED",
                    entry_date=entry_date,
                    entry_price=entry_price,
                    outcome="SUCCESS",
                    exit_date=h_date_str,
                    exit_price=exit_price,
                    exit_reason="TAKE_PROFIT_1",
                    gross_return_pct=gross_pct,
                    net_simulated_return_pct=net_pct,
                    r_multiple=r_mult,
                    mfe=mfe,
                    mae=mae,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence=intrabar_evidence,
                )

            if stop_hit:
                # Stop loss reached before target
                exit_price = stop if ho > stop else ho  # Gap below stop fills at Open
                gross_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
                net_pct = round(gross_pct - cls.NET_FRICTION_PCT, 2)
                r_mult = round((exit_price - entry_price) / (entry_price - stop), 2)
                mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
                mae = round(((min_low - entry_price) / entry_price) * 100.0, 2)
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_TRIGGERED",
                    entry_date=entry_date,
                    entry_price=entry_price,
                    outcome="FAILURE",
                    exit_date=h_date_str,
                    exit_price=exit_price,
                    exit_reason="STOP_LOSS",
                    gross_return_pct=gross_pct,
                    net_simulated_return_pct=net_pct,
                    r_multiple=r_mult,
                    mfe=mfe,
                    mae=mae,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence=intrabar_evidence,
                )

            if h_idx == cls.MAX_HOLDING_SESSIONS:
                # 20-session horizon reached without terminal exit
                exit_price = hc
                gross_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
                net_pct = round(gross_pct - cls.NET_FRICTION_PCT, 2)
                r_mult = round((exit_price - entry_price) / (entry_price - stop), 2)
                mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
                mae = round(((min_low - entry_price) / entry_price) * 100.0, 2)
                return EvaluationResult(
                    symbol=sym,
                    engine_version=ver,
                    signal_date=sig_date,
                    entry_state="ENTRY_TRIGGERED",
                    entry_date=entry_date,
                    entry_price=entry_price,
                    outcome="NEUTRAL",
                    exit_date=h_date_str,
                    exit_price=exit_price,
                    exit_reason="HORIZON_EXIT",
                    gross_return_pct=gross_pct,
                    net_simulated_return_pct=net_pct,
                    r_multiple=r_mult,
                    mfe=mfe,
                    mae=mae,
                    confluence_score=score,
                    market_regime=regime,
                    provider=provider,
                    intrabar_resolution_evidence=intrabar_evidence,
                )

        # Holding sessions elapsed without resolution and < 20 sessions available
        mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
        mae = round(((min_low - entry_price) / entry_price) * 100.0, 2)
        return EvaluationResult(
            symbol=sym,
            engine_version=ver,
            signal_date=sig_date,
            entry_state="ENTRY_TRIGGERED",
            entry_date=entry_date,
            entry_price=entry_price,
            outcome="UNMATURED",
            exit_date=None,
            exit_price=None,
            exit_reason="HOLDING_WINDOW_ACTIVE",
            gross_return_pct=None,
            net_simulated_return_pct=None,
            r_multiple=None,
            mfe=mfe,
            mae=mae,
            confluence_score=score,
            market_regime=regime,
            provider=provider,
            intrabar_resolution_evidence=intrabar_evidence,
        )
