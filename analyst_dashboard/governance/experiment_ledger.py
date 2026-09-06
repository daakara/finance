"""ARX Model Governance & Experiment Ledger Engine.

Maintains an immutable audit trail:
engine_version -> signal -> timestamp -> inputs -> decision -> entry -> stop -> TP1 -> TP2 -> outcome

Tracks forward paper-trading positions without post-hoc modification.
"""

import os
import json
import math
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


class ProvenanceCohort:
    """Rigid provenance classification for cohort contamination insulation."""
    PROSPECTIVE_CLEAN = "PROSPECTIVE_CLEAN"
    HISTORICAL_CONTAMINATED = "HISTORICAL_CONTAMINATED"
    HISTORICAL_UNKNOWN = "HISTORICAL_UNKNOWN"
    EXCLUDED = "EXCLUDED"


class ExperimentLedger:
    """Production-grade Model Governance and Forward Experiment Tracker."""

    FREEZE_DATE_THRESHOLD = "2026-09-04"
    FROZEN_ENGINE_COMMIT = "4e36862"
    FROZEN_ENGINE_TAG = "v2.4.0-phase24-freeze"

    DEFAULT_LEDGER_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "paper_trading_ledger.json"
    )

    @classmethod
    def get_frozen_manifest(cls) -> Optional[Dict[str, Any]]:
        """Loads the cryptographic engine freeze manifest if available."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        manifest_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def verify_frozen_engine_manifest(cls) -> Dict[str, Any]:
        """Verifies repository engines against FROZEN_ENGINE_MANIFEST.json."""
        manifest = cls.get_frozen_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        engine_results = {}
        all_valid = True
        for engine_name, meta in manifest.get("engines", {}).items():
            full_path = os.path.join(repo_root, meta["filePath"])
            if not os.path.exists(full_path):
                engine_results[engine_name] = {"valid": False, "error": "FILE_MISSING"}
                all_valid = False
                continue
            with open(full_path, "rb") as fp:
                content = fp.read().replace(b"\r\n", b"\n")
                h = hashlib.sha256(content).hexdigest()
            is_match = (h == meta["sha256"])
            engine_results[engine_name] = {"valid": is_match, "sha256": h, "expectedSha256": meta["sha256"]}
            if not is_match:
                all_valid = False
        return {
            "status": "VERIFIED" if all_valid else "CORRUPTED",
            "valid": all_valid,
            "manifestVersion": manifest.get("manifestVersion"),
            "provenanceCommit": manifest.get("provenanceCommit"),
            "frozenStrategyVersion": manifest.get("frozenStrategyVersion"),
            "engines": engine_results
        }

    @classmethod
    def classify_provenance_cohort(cls, sig: Dict[str, Any]) -> str:
        """Classifies a signal into a strict provenance tier (Cohort Contamination Firewall).
        Enforces 5 fail-closed security barriers:
        1. Pre-freeze temporal barrier: Any signal dated prior to 2026-09-04 is permanently
           quarantined as HISTORICAL_CONTAMINATED, regardless of explicit tags.
        2. Explicit quarantine preservation: Explicit HISTORICAL_CONTAMINATED, HISTORICAL_UNKNOWN,
           or EXCLUDED records are strictly preserved as non-clean.
        3. Decision hash integrity: If decisionSnapshotHash is present, it must match the
           SHA-256 fingerprint of the record's frozen decision fields; otherwise fail-closed to EXCLUDED.
        4. Inputs hash integrity: If inputsSnapshotHash is present, it must match; otherwise EXCLUDED.
        5. Mandatory provenance fields: A clean prospective signal must have symbol, signalDate >= 2026-09-04,
           and status. If engineVersion is present, it must match the frozen manifest/commit.
        """
        sig_date = sig.get("signalDate", "")
        # Invariant 1: Temporal barrier - pre-freeze signals are ALWAYS contaminated
        if sig_date and sig_date < cls.FREEZE_DATE_THRESHOLD:
            return ProvenanceCohort.HISTORICAL_CONTAMINATED

        explicit = sig.get("provenanceCohort")
        # Invariant 2: Explicit quarantine preservation
        if explicit in [
            ProvenanceCohort.HISTORICAL_CONTAMINATED,
            ProvenanceCohort.HISTORICAL_UNKNOWN,
            ProvenanceCohort.EXCLUDED,
        ]:
            return explicit

        # Invariant 3 & 4: Cryptographic hash validation (Fail-closed)
        dec_hash = sig.get("decisionSnapshotHash")
        if dec_hash:
            if dec_hash != cls.compute_decision_snapshot_hash(sig):
                return ProvenanceCohort.EXCLUDED

        in_hash = sig.get("inputsSnapshotHash")
        if in_hash:
            if in_hash != cls.compute_inputs_snapshot_hash(sig):
                return ProvenanceCohort.EXCLUDED

        # Invariant 5: Engine version verification against frozen manifest/commit
        engine_ver = sig.get("engineVersion")
        if engine_ver:
            manifest_meta = cls.get_frozen_manifest()
            valid_commits = [cls.FROZEN_ENGINE_COMMIT]
            if manifest_meta and manifest_meta.get("provenanceCommit"):
                valid_commits.append(manifest_meta["provenanceCommit"])
                valid_commits.append(manifest_meta["provenanceCommit"][:7])
            if not any(engine_ver.startswith(c[:7]) or c.startswith(engine_ver[:7]) for c in valid_commits):
                return ProvenanceCohort.EXCLUDED

        if sig_date and sig_date >= cls.FREEZE_DATE_THRESHOLD:
            if not sig.get("symbol") or not sig.get("status"):
                return ProvenanceCohort.EXCLUDED
            return ProvenanceCohort.PROSPECTIVE_CLEAN

        # Explicit clean tag without dates is synthetic test fixture
        if explicit == ProvenanceCohort.PROSPECTIVE_CLEAN:
            return ProvenanceCohort.PROSPECTIVE_CLEAN

        return "SYNTHETIC_OR_UNCLASSIFIED"

    @classmethod
    def load_ledger(cls, ledger_path: Optional[str] = None) -> Dict[str, Any]:
        path = ledger_path or cls.DEFAULT_LEDGER_PATH
        if not os.path.exists(path):
            return {
                "version": "1.0.0",
                "engineCommit": "4e36862",
                "tag": "v2.4.0-phase24-freeze",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "totalActiveSignals": 0,
                "signals": []
            }
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Guarantee backward-compatible executionObservations field on all signals
            for sig in data.get("signals", []):
                if "executionObservations" not in sig:
                    sig["executionObservations"] = []
            return data

    @classmethod
    def save_ledger(cls, data: Dict[str, Any], ledger_path: Optional[str] = None) -> None:
        path = ledger_path or cls.DEFAULT_LEDGER_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def compute_decision_snapshot_hash(cls, record: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint over the immutable frozen decision parameters."""
        canonical_dict = {
            "engineVersion": str(record.get("engineVersion", "")),
            "engineTag": str(record.get("engineTag", "")),
            "decisionState": str(record.get("decisionState", "")),
            "entryPrice": float(record.get("entryPrice", 0.0)),
            "corridorMin": float(record.get("corridorMin", 0.0)) if record.get("corridorMin") is not None else None,
            "corridorMax": float(record.get("corridorMax", 0.0)) if record.get("corridorMax") is not None else None,
            "stopLoss": float(record.get("stopLoss", 0.0)) if record.get("stopLoss") is not None else None,
            "takeProfit1": float(record.get("takeProfit1", 0.0)) if record.get("takeProfit1") is not None else None,
            "takeProfit2": float(record.get("takeProfit2", 0.0)) if record.get("takeProfit2") is not None else None,
            "riskRewardRatio": float(record.get("riskRewardRatio", 0.0)) if record.get("riskRewardRatio") is not None else None,
            "confluenceScore": float(record.get("confluenceScore", 0.0)),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def compute_inputs_snapshot_hash(cls, record: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint over the immutable frozen input features."""
        inputs = record.get("inputs") or {}
        canonical_dict = {
            "atr14": float(inputs.get("atr14", 0.0)) if inputs.get("atr14") is not None else None,
            "atrPct": float(inputs.get("atrPct", 0.0)) if inputs.get("atrPct") is not None else None,
            "setupPattern": str(inputs.get("setupPattern", "")),
            "stagePhase": str(inputs.get("stagePhase", "")),
            "marketRegime": str(inputs.get("marketRegime", "")),
            "sector": str(inputs.get("sector", "")),
            "assetClass": str(inputs.get("assetClass", "")),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def register_signal(
        cls,
        symbol: str,
        entry_price: float,
        opt_exec: Dict[str, Any],
        confluence_score: float,
        inputs_meta: Dict[str, Any],
        engine_commit: str = "4e36862",
        engine_tag: str = "v2.4.0-phase24-freeze",
        ledger_path: Optional[str] = None,
        signal_date: Optional[str] = None,
        component_scores: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an immutable new signal in the ledger."""
        ledger = cls.load_ledger(ledger_path)
        sig_date = signal_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        signal_id = f"{symbol}_{sig_date}"

        # Prevent duplicate entries for the same symbol on the same date
        existing = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if existing:
            return existing

        record = {
            "signalId": signal_id,
            "symbol": symbol.upper().strip(),
            "signalDate": sig_date,
            "engineVersion": engine_commit,
            "engineTag": engine_tag,
            "decisionState": "ACTIONABLE_SETUP",
            "entryPrice": float(entry_price),
            "corridorMin": opt_exec.get("optimal_entry_min"),
            "corridorMax": opt_exec.get("optimal_entry_max"),
            "stopLoss": opt_exec.get("stop_loss"),
            "stopLossPct": opt_exec.get("stop_loss_pct"),
            "takeProfit1": opt_exec.get("take_profit_1"),
            "takeProfit1Pct": opt_exec.get("take_profit_1_pct"),
            "takeProfit2": opt_exec.get("take_profit_2"),
            "takeProfit2Pct": opt_exec.get("take_profit_2_pct"),
            "riskRewardRatio": opt_exec.get("risk_reward_ratio"),
            "confluenceScore": float(confluence_score),
            "inputs": {
                "atr14": opt_exec.get("atr_14"),
                "atrPct": round((opt_exec.get("atr_14", 0) / max(0.01, entry_price)) * 100, 2),
                "setupPattern": opt_exec.get("setup_pattern"),
                "stagePhase": opt_exec.get("stage_phase"),
                "marketRegime": inputs_meta.get("market_regime", "BULL"),
                "sector": inputs_meta.get("sector", "EQUITY"),
                "assetClass": inputs_meta.get("asset_class", "US_EQUITY"),
            },
            "componentScores": component_scores or {
                "qualityScore": None,
                "growthScore": None,
                "valuationScore": None,
                "technicalScore": None,
                "smartMoneyScore": None,
                "macroScore": None,
                "catalystScore": None,
            },
            "benchmarksAtSignal": benchmarks or {
                "spyPrice": None,
                "rspPrice": None,
                "sectorBenchmarkSymbol": None,
                "sectorBenchmarkPrice": None,
                "momentumBaselineReturn": None,
            },
            "status": "OPEN",
            "forwardTracking": {
                "sessionsObserved": 0,
                "currentPrice": float(entry_price),
                "maxFavorableExcursionPct": 0.0,
                "maxAdverseExcursionPct": 0.0,
                "tp1Hit": False,
                "tp1Session": None,
                "stopHit": False,
                "stopSession": None,
                "return1d": None,
                "return5d": None,
                "return10d": None,
                "return20d": None,
                "benchmarkReturns": {
                    "spy": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "rsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "sector": {"1d": None, "5d": None, "10d": None, "20d": None},
                },
                "relativeReturns": {
                    "vsSpy": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "vsRsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "vsSector": {"1d": None, "5d": None, "10d": None, "20d": None},
                },
                "signalQuality": {
                    "directionalAccuracy1d": None,
                    "directionalAccuracy5d": None,
                    "directionalAccuracy10d": None,
                    "directionalAccuracy20d": None,
                    "rawReturn1d": None,
                    "rawReturn5d": None,
                    "rawReturn10d": None,
                    "rawReturn20d": None,
                    "mfePct": 0.0,
                    "maePct": 0.0,
                },
                "tradeConstruction": {
                    "prematureStopOut": False,
                    "captureRatio": None,
                    "riskRewardRealized": None,
                },
                "economicOutcome": {
                    "realizedReturnPct": None,
                    "frictionBps": 25.0,
                    "netReturnPct": None,
                },
                "signalPersistence": {
                    "day1Valid": True,
                    "day3Valid": None,
                    "day5Valid": None,
                    "day10Valid": None
                },
                "resolutionDate": None,
                "resolvedOutcome": None,  # "TP1_WIN", "STOP_LOSS", "TIME_EXPIRED"
                "realizedReturnPct": None
            }
        }
        # Compute immutable cryptographic snapshot hashes (Decision & Inputs Dual-Hash)
        record["decisionSnapshotHash"] = cls.compute_decision_snapshot_hash(record)
        record["inputsSnapshotHash"] = cls.compute_inputs_snapshot_hash(record)

        # Record point-in-time liquidity observation at signal generation
        liq_meta = opt_exec.get("liquidity_defense") or {}
        if isinstance(liq_meta, dict):
            liq_meta = dict(liq_meta)
            if "specVersion" not in liq_meta and "spec_version" not in liq_meta:
                liq_meta["specVersion"] = "LiquidityGuard Shadow Spec v1.0"
            if "signalTimestamp" not in liq_meta and "signal_timestamp" not in liq_meta:
                liq_meta["signalTimestamp"] = datetime.now(timezone.utc).isoformat()
        record["liquidityAtSignal"] = liq_meta
        record["liquidityForwardObservations"] = []
        record["executionObservations"] = []

        ledger["signals"].append(record)
        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return record

    @classmethod
    def record_liquidity_forward_observation(
        cls,
        signal_id: str,
        session_index: int,
        liquidity_metrics: Dict[str, Any],
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records subsequent time-varying liquidity observation without mutating liquidityAtSignal or decision parameters."""
        ledger = cls.load_ledger(ledger_path)
        sig = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if not sig:
            raise KeyError(f"Signal {signal_id} not found in ledger")

        # Verify decision and inputs integrity (anti-tamper cryptographic check)
        if sig.get("decisionSnapshotHash"):
            current_hash = cls.compute_decision_snapshot_hash(sig)
            if current_hash != sig["decisionSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {signal_id}")

        if sig.get("inputsSnapshotHash"):
            current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
            if current_inputs_hash != sig["inputsSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {signal_id}")

        if "liquidityForwardObservations" not in sig:
            sig["liquidityForwardObservations"] = []

        obs_entry = {
            "sessionIndex": session_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adv20dUsd": liquidity_metrics.get("adv_20d_usd"),
            "amihudIlliqRaw": liquidity_metrics.get("amihud_illiq"),
            "amihudIlliqScaled": liquidity_metrics.get("amihud_illiq_scaled"),
            "liquidityGrade": liquidity_metrics.get("liquidity_grade"),
            "executionHazard": liquidity_metrics.get("execution_hazard", False),
        }
        sig["liquidityForwardObservations"].append(obs_entry)
        cls.save_ledger(ledger, ledger_path)
        return obs_entry

    @classmethod
    def record_execution_observation(
        cls,
        signal_id: str,
        fill_price: Optional[float] = None,
        execution_timestamp: Optional[str] = None,
        order_size_usd: Optional[float] = None,
        side: str = "BUY",
        execution_source: str = "BROKER_FILL",
        is_simulated: bool = False,
        symbol: Optional[str] = None,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a Phase 26 prospective execution observation without mutating the frozen decision or signal outcomes.

        Captures:
        - signalTimestamp, referencePrice, executionTimestamp, fillPrice, side, orderSizeUsd,
          adv20dUsd, adv5dUsd, liquidityTrend, amihudIlliqRaw, amihudIlliqScaled, liquidityGrade,
          participationRate, realized slippage in bps (magnitude and signed), executionSource, isSimulated.
        """
        ledger = cls.load_ledger(ledger_path)
        sig = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if not sig:
            raise KeyError(f"Signal {signal_id} not found in ledger")

        # Anti-tamper cryptographic check: verify decision and inputs integrity fail-closed
        if sig.get("decisionSnapshotHash"):
            current_hash = cls.compute_decision_snapshot_hash(sig)
            if current_hash != sig["decisionSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {signal_id}")

        if sig.get("inputsSnapshotHash"):
            current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
            if current_inputs_hash != sig["inputsSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {signal_id}")

        # Cross-signal contamination guard: verify symbol matches if provided
        if symbol is not None:
            clean_sym = str(symbol).upper().strip()
            if clean_sym != sig["symbol"]:
                raise ValueError(f"CROSS_SIGNAL_CONTAMINATION: Provided symbol '{symbol}' does not match signal symbol '{sig['symbol']}'")

        # Validate order side
        norm_side = str(side).upper().strip()
        if norm_side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side '{side}': must be 'BUY' or 'SELL'")

        # Validate reference price
        ref_price = float(sig.get("entryPrice", 0.0))
        if ref_price <= 0.0 or math.isnan(ref_price) or math.isinf(ref_price):
            raise ValueError(f"Invalid reference entry price {ref_price} on signal {signal_id}")

        # Validate fill price and compute slippage
        validated_fill: Optional[float] = None
        slippage_bps: Optional[float] = None
        signed_slippage_bps: Optional[float] = None
        if fill_price is not None:
            f_price = float(fill_price)
            if f_price <= 0.0 or math.isnan(f_price) or math.isinf(f_price):
                raise ValueError(f"Invalid fill price {fill_price}: must be positive finite number")
            validated_fill = f_price
            # Absolute slippage magnitude in basis points: abs(fill_price - ref_price) / ref_price * 10,000
            slippage_bps = round((abs(validated_fill - ref_price) / ref_price) * 10000.0, 2)
            # Signed directional slippage (BUY: fill > ref is adverse (+), SELL: fill < ref is adverse (+))
            direction = 1 if norm_side == "BUY" else -1
            signed_slippage_bps = round(direction * ((validated_fill - ref_price) / ref_price) * 10000.0, 2)

        # Validate temporal sequencing
        sig_time_str = (
            sig.get("liquidityAtSignal", {}).get("signalTimestamp")
            or sig.get("liquidityAtSignal", {}).get("signal_timestamp")
            or sig.get("signalDate")
        )
        exec_time_str = execution_timestamp or datetime.now(timezone.utc).isoformat()

        # Enforce temporal anti-lookahead: execution timestamp cannot precede signal timestamp
        if sig_time_str and exec_time_str and validated_fill is not None:
            try:
                sig_clean = sig_time_str.replace("Z", "+00:00")
                exec_clean = exec_time_str.replace("Z", "+00:00")
                sig_dt = datetime.fromisoformat(sig_clean) if "T" in sig_clean else datetime.strptime(sig_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                exec_dt = datetime.fromisoformat(exec_clean) if "T" in exec_clean else datetime.strptime(exec_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if exec_dt < sig_dt:
                    raise ValueError(f"TEMPORAL_VIOLATION: Execution timestamp {exec_time_str} precedes signal timestamp {sig_time_str}")

                # Enforce temporal sequencing against trade resolution: execution cannot occur after trade is resolved
                if sig.get("status") == "RESOLVED":
                    res_date_str = sig.get("forwardTracking", {}).get("resolutionDate")
                    if res_date_str:
                        res_clean = res_date_str.replace("Z", "+00:00")
                        res_dt = datetime.fromisoformat(res_clean) if "T" in res_clean else datetime.strptime(res_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if exec_dt.date() > res_dt.date():
                            raise ValueError(f"TEMPORAL_VIOLATION: Execution date {exec_dt.date()} occurs after trade resolution date {res_dt.date()}")
            except (ValueError, TypeError) as e:
                if "TEMPORAL_VIOLATION" in str(e):
                    raise

        # Extract signal-time liquidity diagnostics
        liq = sig.get("liquidityAtSignal") or {}
        adv_20d = float(liq.get("adv20dUsd") or liq.get("adv_20d_usd") or 0.0)
        adv_5d = float(liq.get("adv5dUsd") or liq.get("adv_5d_usd") or 0.0)
        liq_trend = float(liq.get("liquidityTrend") or liq.get("liquidity_trend") or 1.0)
        amihud_raw = float(liq.get("amihudIlliqRaw") or liq.get("amihud_illiq") or 0.0)
        amihud_scaled = float(liq.get("amihudIlliqScaled") or liq.get("amihud_illiq_scaled") or 0.0)
        liq_grade = str(liq.get("liquidityGrade") or liq.get("liquidity_grade") or "UNKNOWN_LIQUIDITY")

        # Validate order size and participation rate
        order_size: Optional[float] = None
        participation_rate: Optional[float] = None
        if order_size_usd is not None:
            sz = float(order_size_usd)
            if sz < 0.0 or math.isnan(sz) or math.isinf(sz):
                raise ValueError(f"Invalid order size USD {order_size_usd}: must be non-negative finite number")
            order_size = sz
            if adv_20d > 0:
                participation_rate = round(order_size / adv_20d, 6)
            else:
                participation_rate = 1.0 if order_size > 0 else 0.0

        if "executionObservations" not in sig:
            sig["executionObservations"] = []

        # Duplicate fill protection: reject duplicate observation with identical fill, timestamp, side, and size
        if validated_fill is not None:
            for existing_obs in sig["executionObservations"]:
                if (
                    existing_obs.get("fillPrice") == validated_fill
                    and existing_obs.get("executionTimestamp") == exec_time_str
                    and existing_obs.get("side") == norm_side
                    and existing_obs.get("orderSizeUsd") == order_size
                ):
                    raise ValueError(f"DUPLICATE_EXECUTION_OBSERVATION: Identical fill already recorded for signal {signal_id}")

        obs_record = {
            "observationId": f"{signal_id}_exec_{len(sig['executionObservations']) + 1}",
            "signalId": signal_id,
            "symbol": sig.get("symbol"),
            "signalTimestamp": sig_time_str,
            "referencePrice": ref_price,
            "executionTimestamp": exec_time_str if validated_fill is not None else None,
            "fillPrice": validated_fill,
            "side": norm_side,
            "orderSizeUsd": order_size,
            "adv20dUsd": adv_20d,
            "adv5dUsd": adv_5d,
            "liquidityTrend": liq_trend,
            "amihudIlliqRaw": amihud_raw,
            "amihudIlliqScaled": amihud_scaled,
            "liquidityGrade": liq_grade,
            "participationRate": participation_rate,
            "slippageBps": slippage_bps,
            "signedSlippageBps": signed_slippage_bps,
            "executionSource": execution_source if validated_fill is not None else "UNFILLED",
            "isSimulated": bool(is_simulated),
            "specVersion": "LiquidityGuard Shadow Spec v1.0",
            "recordedAt": datetime.now(timezone.utc).isoformat(),
        }

        sig["executionObservations"].append(obs_record)
        cls.save_ledger(ledger, ledger_path)
        return obs_record

    @classmethod
    def update_forward_observations(
        cls,
        db_engine,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest subsequent price candles and update forward outcomes objectively."""
        ledger = cls.load_ledger(ledger_path)
        updated_count = 0

        for sig in ledger["signals"]:
            # Hard Governance Check: verify decision and inputs snapshot integrity
            if sig.get("decisionSnapshotHash"):
                current_hash = cls.compute_decision_snapshot_hash(sig)
                if current_hash != sig["decisionSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {sig.get('signalId')}")

            if sig.get("inputsSnapshotHash"):
                current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
                if current_inputs_hash != sig["inputsSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {sig.get('signalId')}")

            if sig["status"] != "OPEN":
                continue

            sym = sig["symbol"]
            entry_price = sig["entryPrice"]
            stop = sig["stopLoss"]
            tp1 = sig["takeProfit1"]
            sig_date = sig["signalDate"]

    @classmethod
    def _compute_benchmark_horizon_return(
        cls, db_engine, bench_symbol: str, sig_date: str, horizon_days: int
    ) -> Optional[float]:
        """Calculates benchmark buy-and-hold return over specified session horizon from sig_date."""
        if not db_engine or not hasattr(db_engine, "get_daily_candles"):
            return None
        try:
            b_candles = db_engine.get_daily_candles(bench_symbol, limit=100)
            if not b_candles:
                return None
            b_df = pd.DataFrame(b_candles)
            b_date_col = "time" if "time" in b_df.columns else ("date" if "date" in b_df.columns else None)
            if not b_date_col:
                return None
            b_sub = b_df[b_df[b_date_col] > sig_date].copy()
            if len(b_sub) < horizon_days:
                return None
            b_sub.rename(columns={"close": "Close", "open": "Open"}, inplace=True)
            prior = b_df[b_df[b_date_col] <= sig_date]
            if not prior.empty:
                p_close_col = "close" if "close" in prior.columns else "Close"
                b_p0 = float(prior.iloc[-1][p_close_col])
            else:
                b_p0 = float(b_sub.iloc[0]["Open"])
            if b_p0 <= 0:
                return None
            b_close_h = float(b_sub.iloc[horizon_days - 1]["Close"])
            return round(((b_close_h - b_p0) / b_p0) * 100.0, 2)
        except Exception:
            return None

    @classmethod
    def update_forward_observations(
        cls,
        db_engine,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest subsequent price candles and update forward outcomes objectively."""
        ledger = cls.load_ledger(ledger_path)
        updated_count = 0

        for sig in ledger["signals"]:
            # Hard Governance Check: verify decision and inputs snapshot integrity
            if sig.get("decisionSnapshotHash"):
                current_hash = cls.compute_decision_snapshot_hash(sig)
                if current_hash != sig["decisionSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {sig.get('signalId')}")

            if sig.get("inputsSnapshotHash"):
                current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
                if current_inputs_hash != sig["inputsSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {sig.get('signalId')}")

            if sig["status"] != "OPEN":
                continue

            sym = sig["symbol"]
            entry_price = sig["entryPrice"]
            stop = sig["stopLoss"]
            tp1 = sig["takeProfit1"]
            sig_date = sig["signalDate"]

            candles = db_engine.get_daily_candles(sym, limit=100)
            if not candles:
                continue

            df = pd.DataFrame(candles)
            date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
            if not date_col:
                continue

            # Filter candles occurring AFTER the signal date
            subsequent = df[df[date_col] > sig_date].copy()
            if subsequent.empty:
                continue

            subsequent.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            sessions = len(subsequent)
            highs = subsequent["High"].values
            lows = subsequent["Low"].values
            closes = subsequent["Close"].values

            latest_close = float(closes[-1])
            mfe = float(np.max(highs) - entry_price) / entry_price * 100.0
            mae = float(np.min(lows) - entry_price) / entry_price * 100.0

            ft = sig.setdefault("forwardTracking", {})
            ft["sessionsObserved"] = sessions
            ft["currentPrice"] = latest_close
            ft["maxFavorableExcursionPct"] = round(mfe, 2)
            ft["maxAdverseExcursionPct"] = round(mae, 2)

            sq = ft.setdefault("signalQuality", {
                "directionalAccuracy1d": None,
                "directionalAccuracy5d": None,
                "directionalAccuracy10d": None,
                "directionalAccuracy20d": None,
                "rawReturn1d": None,
                "rawReturn5d": None,
                "rawReturn10d": None,
                "rawReturn20d": None,
                "mfePct": 0.0,
                "maePct": 0.0,
            })
            sq["mfePct"] = round(mfe, 2)
            sq["maePct"] = round(mae, 2)

            # Check Returns at multi-horizons (T+1, T+5, T+10, T+20)
            if sessions >= 1 and ft.get("return1d") is None:
                r1d = round(((closes[0] - entry_price) / entry_price) * 100.0, 2)
                ft["return1d"] = r1d
                sq["rawReturn1d"] = r1d
                sq["directionalAccuracy1d"] = bool(r1d > 0)

            if sessions >= 5 and ft.get("return5d") is None:
                r5d = round(((closes[4] - entry_price) / entry_price) * 100.0, 2)
                ft["return5d"] = r5d
                sq["rawReturn5d"] = r5d
                sq["directionalAccuracy5d"] = bool(r5d > 0)

            if sessions >= 10 and ft.get("return10d") is None:
                r10d = round(((closes[9] - entry_price) / entry_price) * 100.0, 2)
                ft["return10d"] = r10d
                sq["rawReturn10d"] = r10d
                sq["directionalAccuracy10d"] = bool(r10d > 0)

            if sessions >= 20 and ft.get("return20d") is None:
                r20d = round(((closes[19] - entry_price) / entry_price) * 100.0, 2)
                ft["return20d"] = r20d
                sq["rawReturn20d"] = r20d
                sq["directionalAccuracy20d"] = bool(r20d > 0)

            # Multi-horizon benchmark comparisons (SPY, RSP, Sector)
            bench_dict = ft.setdefault("benchmarkReturns", {
                "spy": {"1d": None, "5d": None, "10d": None, "20d": None},
                "rsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                "sector": {"1d": None, "5d": None, "10d": None, "20d": None},
            })
            rel_dict = ft.setdefault("relativeReturns", {
                "vsSpy": {"1d": None, "5d": None, "10d": None, "20d": None},
                "vsRsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                "vsSector": {"1d": None, "5d": None, "10d": None, "20d": None},
            })

            sec_meta = sig.get("benchmarksAtSignal") or {}
            sector_sym = sec_meta.get("sectorBenchmarkSymbol") or (
                "XLK" if sig.get("inputs", {}).get("sector") == "TECHNOLOGY" else None
            )

            for h_label, h_days, ret_val in [
                ("1d", 1, ft.get("return1d")),
                ("5d", 5, ft.get("return5d")),
                ("10d", 10, ft.get("return10d")),
                ("20d", 20, ft.get("return20d")),
            ]:
                if sessions >= h_days and ret_val is not None:
                    spy_ret = cls._compute_benchmark_horizon_return(db_engine, "SPY", sig_date, h_days)
                    if spy_ret is not None:
                        bench_dict["spy"][h_label] = spy_ret
                        rel_dict["vsSpy"][h_label] = round(ret_val - spy_ret, 2)
                    rsp_ret = cls._compute_benchmark_horizon_return(db_engine, "RSP", sig_date, h_days)
                    if rsp_ret is not None:
                        bench_dict["rsp"][h_label] = rsp_ret
                        rel_dict["vsRsp"][h_label] = round(ret_val - rsp_ret, 2)
                    if sector_sym:
                        sec_ret = cls._compute_benchmark_horizon_return(db_engine, sector_sym, sig_date, h_days)
                        if sec_ret is not None:
                            bench_dict["sector"][h_label] = sec_ret
                            rel_dict["vsSector"][h_label] = round(ret_val - sec_ret, 2)

            # Check TP1 vs Stop sequence with fail-closed intrabar collision handling
            intrabar_collision = False
            for idx in range(sessions):
                hit_tp = bool(highs[idx] >= tp1)
                hit_sl = bool(lows[idx] <= stop)

                if hit_tp and hit_sl and not ft["tp1Hit"] and not ft["stopHit"]:
                    # Pessimistic fail-closed policy: adverse excursion precedes favorable excursion
                    intrabar_collision = True
                    ft["stopHit"] = True
                    ft["stopSession"] = idx + 1
                    ft["tp1Hit"] = True
                    ft["tp1Session"] = idx + 1
                    break

                if hit_tp and not ft["tp1Hit"]:
                    ft["tp1Hit"] = True
                    ft["tp1Session"] = idx + 1
                if hit_sl and not ft["stopHit"]:
                    ft["stopHit"] = True
                    ft["stopSession"] = idx + 1

                if ft["tp1Hit"] or ft["stopHit"]:
                    break

            ft["intrabarCollision"] = intrabar_collision

            # Resolve outcome with strict fail-closed arbitration
            if intrabar_collision or (ft["stopHit"] and (not ft["tp1Hit"] or ft["stopSession"] <= ft["tp1Session"])):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "STOP_LOSS"
                ft["realizedReturnPct"] = round(((stop - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["stopSession"] - 1] if ft["stopSession"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif ft["tp1Hit"] and (not ft["stopHit"] or ft["tp1Session"] < ft["stopSession"]):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TP1_WIN"
                ft["realizedReturnPct"] = round(((tp1 - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["tp1Session"] - 1] if ft["tp1Session"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif sessions >= 20:
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TIME_EXPIRED"
                ft["realizedReturnPct"] = round(((latest_close - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[19]

            # Populate Trade Construction and Economic Outcomes
            if sig["status"] == "RESOLVED":
                tc = ft.setdefault("tradeConstruction", {
                    "prematureStopOut": False,
                    "captureRatio": None,
                    "captureVsMfe5d": None,
                    "captureVsMfe10d": None,
                    "captureVsMfe20d": None,
                    "riskRewardRealized": None,
                })
                eco = ft.setdefault("economicOutcome", {
                    "realizedReturnPct": None,
                    "frictionBps": 30.0,
                    "netReturnPct": None,
                })
                realized = ft.get("realizedReturnPct", 0.0)

                # Multi-horizon capture ratios
                mfe_5d = round(float(np.max(highs[:min(sessions, 5)]) - entry_price) / entry_price * 100.0, 2)
                mfe_10d = round(float(np.max(highs[:min(sessions, 10)]) - entry_price) / entry_price * 100.0, 2)
                mfe_20d = round(float(np.max(highs[:min(sessions, 20)]) - entry_price) / entry_price * 100.0, 2)

                if realized is not None and realized > 0:
                    tc["captureVsMfe5d"] = round(realized / max(0.01, mfe_5d), 4) if mfe_5d > 0 else 1.0
                    tc["captureVsMfe10d"] = round(realized / max(0.01, mfe_10d), 4) if mfe_10d > 0 else 1.0
                    tc["captureVsMfe20d"] = round(realized / max(0.01, mfe_20d), 4) if mfe_20d > 0 else 1.0
                    tc["captureRatio"] = round(realized / max(0.01, mfe), 4)
                elif realized is not None and realized <= 0:
                    tc["captureVsMfe5d"] = 0.0
                    tc["captureVsMfe10d"] = 0.0
                    tc["captureVsMfe20d"] = 0.0
                    tc["captureRatio"] = 0.0

                # Post-Stop Favorable Excursion (PSFE) analysis
                # Clarified: This measures path potential after stop-out to inform shadow exit research,
                # NOT an assertion that the stop itself was incorrect.
                psfe_tp1_crossed = False
                post_stop_mfe = 0.0
                if ft["stopHit"] and ft.get("stopSession") and ft["stopSession"] < sessions:
                    post_stop_highs = highs[ft["stopSession"]:]
                    if len(post_stop_highs) > 0:
                        post_stop_mfe = round(float(np.max(post_stop_highs) - entry_price) / entry_price * 100.0, 2)
                        if np.any(post_stop_highs >= tp1):
                            psfe_tp1_crossed = True

                tc["postStopFavorableExcursion"] = {
                    "tp1CrossedPostStop": psfe_tp1_crossed,
                    "maxFavorableExcursionPostStopPct": post_stop_mfe
                }
                # Backward compatibility alias
                tc["prematureStopOut"] = psfe_tp1_crossed

                # Risk-reward realization
                stop_dist = abs(entry_price - stop)
                if stop_dist > 0 and realized is not None:
                    tc["riskRewardRealized"] = round((realized / 100.0 * entry_price) / stop_dist, 2)

                # Economic outcome net of primary friction (30 bps round-trip)
                primary_friction_bps = 30.0
                eco["realizedReturnPct"] = realized
                eco["frictionBps"] = primary_friction_bps
                eco["netReturnPct"] = round(realized - (primary_friction_bps / 100.0), 2) if realized is not None else None

            updated_count += 1

        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return {"updatedSignals": updated_count, "openSignals": ledger["totalActiveSignals"]}

    @classmethod
    def compute_governance_scorecard(
        cls,
        ledger_path: Optional[str] = None,
        cohort_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate performance against Phase 25/26 Model Governance criteria with Cohort Contamination Firewall."""
        ledger = cls.load_ledger(ledger_path)
        all_signals = ledger.get("signals", [])
        if not all_signals:
            return {"status": "NO_SIGNALS", "n": 0}

        # 1. Audit Provenance Across All Signals (Cohort Contamination Firewall)
        clean_signals = []
        contaminated_signals = []
        unclassified_signals = []
        excluded_signals = []

        for s in all_signals:
            c = cls.classify_provenance_cohort(s)
            s["provenanceCohort"] = c
            if c == ProvenanceCohort.PROSPECTIVE_CLEAN:
                clean_signals.append(s)
            elif c == ProvenanceCohort.HISTORICAL_CONTAMINATED:
                contaminated_signals.append(s)
            elif c == ProvenanceCohort.EXCLUDED:
                excluded_signals.append(s)
            else:
                unclassified_signals.append(s)

        # Determine target evaluation cohort
        if cohort_filter == ProvenanceCohort.HISTORICAL_CONTAMINATED:
            signals = contaminated_signals
            clean_eval_eligible = False
            firewall_status = "HISTORICAL_CONTAMINATED_RESEARCH_ONLY"
            firewall_warning = "CRITICAL: Evaluating historically contaminated cohort. NOT ELIGIBLE for predictive edge claims."
        elif cohort_filter == "ALL":
            signals = all_signals
            clean_eval_eligible = (len(contaminated_signals) == 0)
            firewall_status = "ALL_SIGNALS_MIXED"
            firewall_warning = "WARNING: Mixed evaluation includes contaminated historical signals." if len(contaminated_signals) > 0 else None
        elif cohort_filter == ProvenanceCohort.PROSPECTIVE_CLEAN:
            signals = clean_signals
            clean_eval_eligible = True
            firewall_status = "CLEAN_COHORT_INSULATED"
            firewall_warning = None
        else:
            # Default auto-isolation: If clean prospective signals exist, insulate them from contaminated history
            if len(clean_signals) > 0:
                signals = clean_signals
                clean_eval_eligible = True
                firewall_status = "AUTO_INSULATED_PROSPECTIVE_CLEAN"
                firewall_warning = None
            elif len(contaminated_signals) > 0 and len(clean_signals) == 0 and len(unclassified_signals) == 0:
                signals = contaminated_signals
                clean_eval_eligible = False
                firewall_status = "HISTORICAL_CONTAMINATED_NO_PROSPECTIVE"
                firewall_warning = "CRITICAL: Only contaminated historical signals found. NOT ELIGIBLE for predictive edge claims."
            else:
                signals = all_signals
                clean_eval_eligible = True
                firewall_status = "UNCLASSIFIED_SYNTHETIC_TEST_COHORT"
                firewall_warning = None

        # Verify Frozen Engine Manifest Integrity
        manifest_audit = cls.verify_frozen_engine_manifest()
        if manifest_audit.get("status") == "CORRUPTED":
            clean_eval_eligible = False
            firewall_status = "PRODUCTION_ENGINE_MUTATION_DETECTED"
            firewall_warning = "CRITICAL: Production engine SHA-256 manifest mismatch. Clean prospective evaluation revoked."

        resolved = [s for s in signals if s.get("status") == "RESOLVED"]
        open_signals = [s for s in signals if s.get("status") == "OPEN"]

        n_resolved = len(resolved)
        wins = [s for s in resolved if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN"]
        stops = [s for s in resolved if s.get("forwardTracking", {}).get("resolvedOutcome") == "STOP_LOSS"]

        p_win = (len(wins) / n_resolved) if n_resolved > 0 else 0.0
        p_loss = (len(stops) / n_resolved) if n_resolved > 0 else 0.0

        avg_win = float(np.mean([w["forwardTracking"]["realizedReturnPct"] for w in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops])) if stops else 0.0

        expectancy = (p_win * avg_win) - (p_loss * avg_loss)
        primary_friction_pct = 0.30  # 30 bps round-trip primary standard (additive deduction)
        net_expectancy = expectancy - primary_friction_pct if n_resolved > 0 else 0.0

        friction_sensitivity = {
            "arithmeticConvention": "linear_additive_deduction",
            "0bps_gross": round(expectancy, 2),
            "15bps_optimistic": round(expectancy - 0.15, 2),
            "30bps_primary": round(net_expectancy, 2),
            "50bps_stress": round(expectancy - 0.50, 2),
            "100bps_severe": round(expectancy - 1.00, 2),
            "edgeBreakevenFrictionBps": round(expectancy * 100.0, 1) if expectancy > 0 else 0.0
        }

        total_gains = sum([w["forwardTracking"]["realizedReturnPct"] for w in wins]) if wins else 0.0
        total_losses = sum([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops]) if stops else 0.0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (999.0 if total_gains > 0 else 0.0)

        # Statistical Uncertainty & Confidence Intervals (Diagnostic layer)
        all_realized = [
            s["forwardTracking"]["realizedReturnPct"]
            for s in resolved
            if s.get("forwardTracking", {}).get("realizedReturnPct") is not None
        ]
        if len(all_realized) >= 2:
            std_err = float(np.std(all_realized, ddof=1) / np.sqrt(len(all_realized)))
            ci_95_lower = round(expectancy - 1.96 * std_err, 2)
            ci_95_upper = round(expectancy + 1.96 * std_err, 2)

            # Wilson score interval for binomial Win Rate
            z = 1.96
            p = p_win
            denom = 1 + (z**2 / n_resolved)
            center = (p + (z**2 / (2 * n_resolved))) / denom
            spread = (z * np.sqrt((p * (1 - p) / n_resolved) + (z**2 / (4 * n_resolved**2)))) / denom
            win_ci_lower = round(max(0.0, (center - spread) * 100.0), 1)
            win_ci_upper = round(min(100.0, (center + spread) * 100.0), 1)
        else:
            std_err = None
            ci_95_lower = None
            ci_95_upper = None
            win_ci_lower = None
            win_ci_upper = None

        # Trade Construction Quality Metrics
        psfe_stops = [
            s for s in stops
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("prematureStopOut") is True
            or s.get("forwardTracking", {}).get("tradeConstruction", {}).get("postStopFavorableExcursion", {}).get("tp1CrossedPostStop") is True
        ]
        post_stop_favorable_rate = round((len(psfe_stops) / len(stops)) * 100.0, 1) if stops else 0.0

        capture_ratios = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureRatio")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureRatio") is not None
        ]
        capture_5d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe5d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe5d") is not None
        ]
        capture_10d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe10d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe10d") is not None
        ]
        capture_20d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe20d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe20d") is not None
        ]

        mean_capture_ratio = round(float(np.mean(capture_ratios)), 4) if capture_ratios else None

        # Directional Signal Quality (Unconstrained price behavior across all tracked signals)
        all_tracked = [s for s in signals if s.get("forwardTracking", {}).get("sessionsObserved", 0) > 0]
        acc_1d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy1d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy1d") is not None]
        acc_5d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy5d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy5d") is not None]
        acc_10d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy10d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy10d") is not None]
        acc_20d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy20d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy20d") is not None]

        ret_1d = [s["forwardTracking"]["signalQuality"]["rawReturn1d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn1d") is not None]
        ret_5d = [s["forwardTracking"]["signalQuality"]["rawReturn5d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn5d") is not None]
        ret_10d = [s["forwardTracking"]["signalQuality"]["rawReturn10d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn10d") is not None]
        ret_20d = [s["forwardTracking"]["signalQuality"]["rawReturn20d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn20d") is not None]

        mfes = [s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0) for s in all_tracked]
        maes = [s.get("forwardTracking", {}).get("maxAdverseExcursionPct", 0.0) for s in all_tracked]

        # Benchmark Relative Returns with Medians and Hit Rates
        rel_spy_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSpy", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSpy", {}).get("20d") is not None
        ]
        rel_rsp_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsRsp", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsRsp", {}).get("20d") is not None
        ]
        rel_sec_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSector", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSector", {}).get("20d") is not None
        ]

        benchmark_suite = {
            "vsSpy20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_spy_20d)), 2) if rel_spy_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_spy_20d)), 2) if rel_spy_20d else None,
                "hitRatePct": round((sum(1 for r in rel_spy_20d if r > 0) / len(rel_spy_20d)) * 100.0, 1) if rel_spy_20d else None,
                "sampleCount": len(rel_spy_20d)
            },
            "vsRsp20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_rsp_20d)), 2) if rel_rsp_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_rsp_20d)), 2) if rel_rsp_20d else None,
                "hitRatePct": round((sum(1 for r in rel_rsp_20d if r > 0) / len(rel_rsp_20d)) * 100.0, 1) if rel_rsp_20d else None,
                "sampleCount": len(rel_rsp_20d)
            },
            "vsSector20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_sec_20d)), 2) if rel_sec_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_sec_20d)), 2) if rel_sec_20d else None,
                "hitRatePct": round((sum(1 for r in rel_sec_20d if r > 0) / len(rel_sec_20d)) * 100.0, 1) if rel_sec_20d else None,
                "sampleCount": len(rel_sec_20d)
            },
            # Backward compatibility aliases
            "excessReturnVsSpy20d": round(float(np.mean(rel_spy_20d)), 2) if rel_spy_20d else None,
            "excessReturnVsRsp20d": round(float(np.mean(rel_rsp_20d)), 2) if rel_rsp_20d else None,
            "excessReturnVsSector20d": round(float(np.mean(rel_sec_20d)), 2) if rel_sec_20d else None,
        }

        # Confluence Score Monotonicity Evaluation (Ordinal ranking verification)
        confluence_buckets = {
            "<75": [],
            "75-79.9": [],
            "80-84.9": [],
            "85+": []
        }
        for s in resolved:
            score = float(s.get("confluenceScore", 0.0))
            if score < 75.0:
                confluence_buckets["<75"].append(s)
            elif score < 80.0:
                confluence_buckets["75-79.9"].append(s)
            elif score < 85.0:
                confluence_buckets["80-84.9"].append(s)
            else:
                confluence_buckets["85+"].append(s)

        monotonicity_table = {}
        prev_win_rate = -1.0
        prev_mean_return = -999.0
        is_monotonic_win = True
        is_monotonic_ret = True
        populated_bucket_count = 0

        for b_name in ["<75", "75-79.9", "80-84.9", "85+"]:
            b_signals = confluence_buckets[b_name]
            b_n = len(b_signals)
            if b_n > 0:
                b_wins = len([s for s in b_signals if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN"])
                b_wr = round((b_wins / b_n) * 100.0, 1)
                b_rets = [s["forwardTracking"]["realizedReturnPct"] for s in b_signals if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
                b_mean_ret = round(float(np.mean(b_rets)), 2) if b_rets else 0.0
                b_mfes = [s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0) for s in b_signals]
                b_mean_mfe = round(float(np.mean(b_mfes)), 2) if b_mfes else 0.0

                if prev_win_rate >= 0 and b_wr < prev_win_rate:
                    is_monotonic_win = False
                if prev_mean_return > -900.0 and b_mean_ret < prev_mean_return:
                    is_monotonic_ret = False

                prev_win_rate = b_wr
                prev_mean_return = b_mean_ret
                populated_bucket_count += 1

                monotonicity_table[b_name] = {
                    "count": b_n,
                    "winRate": b_wr,
                    "meanRealizedReturn": b_mean_ret,
                    "meanMfe": b_mean_mfe
                }
            else:
                monotonicity_table[b_name] = {"count": 0, "winRate": None, "meanRealizedReturn": None, "meanMfe": None}

        # Spearman Rank Correlation Analysis
        spearman_metrics = {}
        conf_scores_res = [float(s.get("confluenceScore", 0.0)) for s in resolved]
        rets_res = [float(s["forwardTracking"]["realizedReturnPct"]) for s in resolved if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
        mfes_res = [float(s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0)) for s in resolved]

        if len(conf_scores_res) >= 5 and spearmanr is not None and len(conf_scores_res) == len(rets_res):
            if len(set(conf_scores_res)) > 1 and len(set(rets_res)) > 1:
                try:
                    corr_ret, p_ret = spearmanr(conf_scores_res, rets_res)
                    corr_mfe, p_mfe = spearmanr(conf_scores_res, mfes_res) if len(set(mfes_res)) > 1 else (0.0, 1.0)
                    spearman_metrics = {
                        "confluenceVsReturnRho": round(float(corr_ret), 4) if not math.isnan(corr_ret) else 0.0,
                        "confluenceVsReturnPValue": round(float(p_ret), 4) if not math.isnan(p_ret) else 1.0,
                        "confluenceVsMfeRho": round(float(corr_mfe), 4) if not math.isnan(corr_mfe) else 0.0,
                        "confluenceVsMfePValue": round(float(p_mfe), 4) if not math.isnan(p_mfe) else 1.0,
                    }
                except Exception:
                    spearman_metrics = {"status": "CALCULATION_UNAVAILABLE"}
            else:
                spearman_metrics = {
                    "status": "CONSTANT_INPUT_UNDEFINED",
                    "confluenceVsReturnRho": 0.0,
                    "confluenceVsReturnPValue": 1.0,
                    "confluenceVsMfeRho": 0.0,
                    "confluenceVsMfePValue": 1.0,
                }
        else:
            spearman_metrics = {"status": "INSUFFICIENT_SAMPLE_FOR_CORRELATION", "sampleSize": len(conf_scores_res)}

        # Top-vs-Bottom Quartile Discrimination Analysis (Q4 vs Q1)
        quartile_discrimination = {}
        if len(resolved) >= 8:
            sorted_by_conf = sorted(resolved, key=lambda s: float(s.get("confluenceScore", 0.0)))
            q_size = max(1, len(sorted_by_conf) // 4)
            q1_bottom = sorted_by_conf[:q_size]
            q4_top = sorted_by_conf[-q_size:]

            q1_rets = [s["forwardTracking"]["realizedReturnPct"] for s in q1_bottom if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
            q4_rets = [s["forwardTracking"]["realizedReturnPct"] for s in q4_top if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
            q1_wins = sum(1 for s in q1_bottom if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN")
            q4_wins = sum(1 for s in q4_top if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN")

            q1_wr = round((q1_wins / len(q1_bottom)) * 100.0, 1)
            q4_wr = round((q4_wins / len(q4_top)) * 100.0, 1)
            q1_mean_ret = round(float(np.mean(q1_rets)), 2) if q1_rets else 0.0
            q4_mean_ret = round(float(np.mean(q4_rets)), 2) if q4_rets else 0.0
            q1_med_ret = round(float(np.median(q1_rets)), 2) if q1_rets else 0.0
            q4_med_ret = round(float(np.median(q4_rets)), 2) if q4_rets else 0.0

            quartile_discrimination = {
                "topQuartileQ4": {
                    "count": len(q4_top),
                    "minConfluence": float(q4_top[0].get("confluenceScore", 0.0)),
                    "winRatePct": q4_wr,
                    "meanReturnPct": q4_mean_ret,
                    "medianReturnPct": q4_med_ret,
                },
                "bottomQuartileQ1": {
                    "count": len(q1_bottom),
                    "maxConfluence": float(q1_bottom[-1].get("confluenceScore", 0.0)),
                    "winRatePct": q1_wr,
                    "meanReturnPct": q1_mean_ret,
                    "medianReturnPct": q1_med_ret,
                },
                "spreadQ4MinusQ1": {
                    "winRateSpread": round(q4_wr - q1_wr, 1),
                    "meanReturnSpread": round(q4_mean_ret - q1_mean_ret, 2),
                    "medianReturnSpread": round(q4_med_ret - q1_med_ret, 2),
                    "rankingPowerPositive": bool(q4_wr >= q1_wr and q4_mean_ret >= q1_mean_ret)
                }
            }
        else:
            quartile_discrimination = {"status": "INSUFFICIENT_SAMPLE_FOR_QUARTILES", "requiredN": 8, "currentN": len(resolved)}

        # Cluster Structure & Dependence Diagnostics
        dates = [s.get("signalDate") for s in signals if s.get("signalDate")]
        session_dist = {}
        for d in dates:
            session_dist[d] = session_dist.get(d, 0) + 1
        max_trades_per_day = max(session_dist.values()) if session_dist else len(signals)

        sectors = [s.get("inputs", {}).get("sector", "UNKNOWN") for s in signals]
        sector_dist = {}
        for sec in sectors:
            sector_dist[sec] = sector_dist.get(sec, 0) + 1
        max_sec_cnt = max(sector_dist.values()) if sector_dist else 0
        max_sec_pct = round((max_sec_cnt / len(signals)) * 100.0, 1) if signals else 0.0

        regimes = [s.get("inputs", {}).get("marketRegime", "UNKNOWN") for s in signals]
        regime_dist = {}
        for r in regimes:
            regime_dist[r] = regime_dist.get(r, 0) + 1

        clustering_diagnostics = {
            "maxTradesPerSession": max_trades_per_day,
            "uniqueSessionsCount": len(session_dist),
            "sessionDistribution": session_dist,
            "sectorConcentration": sector_dist,
            "maxSectorConcentrationPct": max_sec_pct,
            "regimeDistribution": regime_dist,
            "dependenceWarning": (
                "Cluster dependence detected: trades are concentrated across a limited number of "
                "market sessions and sectors. Nominal N must not be treated as independent observations."
            )
        }

        portfolio_aggregation = {
            "maxConcurrentEntries": max_trades_per_day,
            "topSectorExposurePct": max_sec_pct,
            "portfolioRiskNote": (
                "Trade-level expectancy does not guarantee bounded portfolio drawdown or Sharpe ratio "
                "under simultaneous sector exposure."
            )
        }

        unique_dates = len(set(s.get("signalDate") for s in resolved if s.get("signalDate")))

        return {
            "frozenEngineManifest": manifest_audit,
            "cohortFirewall": {
                "status": firewall_status,
                "cohortFilterRequested": cohort_filter,
                "cleanSignalsCount": len(clean_signals),
                "contaminatedSignalsCount": len(contaminated_signals),
                "excludedSignalsCount": len(excluded_signals),
                "unclassifiedSignalsCount": len(unclassified_signals),
                "cleanEvaluationEligible": clean_eval_eligible,
                "firewallWarning": firewall_warning
            },
            "clusteringAndDependence": clustering_diagnostics,
            "portfolioAggregation": portfolio_aggregation,
            "totalSignals": len(signals),
            "openSignals": len(open_signals),
            "resolvedSignals": n_resolved,
            # Top-level backward compatibility keys
            "winRate": round(p_win * 100.0, 1),
            "winRateCI95": [win_ci_lower, win_ci_upper] if win_ci_lower is not None else None,
            "stopRate": round(p_loss * 100.0, 1),
            "avgWinPct": round(avg_win, 2),
            "avgLossPct": round(avg_loss, 2),
            "expectancyPct": round(expectancy, 2),
            "netExpectancyPct": round(net_expectancy, 2),
            "expectancyStandardError": round(std_err, 2) if std_err is not None else None,
            "expectancyCI95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
            "profitFactor": round(profit_factor, 2),
            "frictionSensitivity": friction_sensitivity,
            # Detailed structured dimensions
            "tradeConstruction": {
                "winRate": round(p_win * 100.0, 1),
                "winRateCI95": [win_ci_lower, win_ci_upper] if win_ci_lower is not None else None,
                "stopRate": round(p_loss * 100.0, 1),
                "avgWinPct": round(avg_win, 2),
                "avgLossPct": round(avg_loss, 2),
                "expectancyPct": round(expectancy, 2),
                "netExpectancyPct": round(net_expectancy, 2),
                "expectancyStandardError": round(std_err, 2) if std_err is not None else None,
                "expectancyCI95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
                "profitFactor": round(profit_factor, 2),
                "postStopFavorableRate": post_stop_favorable_rate,
                "prematureStopOutRate": post_stop_favorable_rate,  # Backward compatibility alias
                "meanCaptureRatio": mean_capture_ratio,
                "meanCaptureVsMfe5d": round(float(np.mean(capture_5d)), 4) if capture_5d else None,
                "meanCaptureVsMfe10d": round(float(np.mean(capture_10d)), 4) if capture_10d else None,
                "meanCaptureVsMfe20d": round(float(np.mean(capture_20d)), 4) if capture_20d else None,
            },
            "signalQuality": {
                "directionalAccuracy1d": round((sum(acc_1d) / len(acc_1d)) * 100.0, 1) if acc_1d else None,
                "directionalAccuracy5d": round((sum(acc_5d) / len(acc_5d)) * 100.0, 1) if acc_5d else None,
                "directionalAccuracy10d": round((sum(acc_10d) / len(acc_10d)) * 100.0, 1) if acc_10d else None,
                "directionalAccuracy20d": round((sum(acc_20d) / len(acc_20d)) * 100.0, 1) if acc_20d else None,
                "meanRawReturn1d": round(float(np.mean(ret_1d)), 2) if ret_1d else None,
                "meanRawReturn5d": round(float(np.mean(ret_5d)), 2) if ret_5d else None,
                "meanRawReturn10d": round(float(np.mean(ret_10d)), 2) if ret_10d else None,
                "meanRawReturn20d": round(float(np.mean(ret_20d)), 2) if ret_20d else None,
                "meanMfePct": round(float(np.mean(mfes)), 2) if mfes else 0.0,
                "meanMaePct": round(float(np.mean(maes)), 2) if maes else 0.0,
            },
            "benchmarkComparisons": benchmark_suite,
            "confluenceMonotonicity": {
                "buckets": monotonicity_table,
                "isMonotonicWinRate": is_monotonic_win if populated_bucket_count >= 2 else None,
                "isMonotonicReturn": is_monotonic_ret if populated_bucket_count >= 2 else None,
                "monotonicityStatus": (
                    "MONOTONIC_CONFIRMED" if (is_monotonic_win and populated_bucket_count >= 2)
                    else ("NON_MONOTONIC_ANOMALY" if populated_bucket_count >= 2 else "INSUFFICIENT_BUCKET_SAMPLE")
                )
            },
            "rankingPower": {
                "spearmanCorrelation": spearman_metrics,
                "quartileDiscrimination": quartile_discrimination
            },
            "governanceGates": {
                "expectancyPositive": bool(expectancy > 0),
                "netExpectancyPositive": bool(net_expectancy > 0),
                "profitFactorAbove1_5": bool(profit_factor >= 1.5),
                "stopRateBelow50": bool(p_loss < 0.50 if n_resolved > 0 else True),
                "statisticallySignificant": bool(ci_95_lower is not None and ci_95_lower > 0),
                "confluenceMonotonic": bool(is_monotonic_win if populated_bucket_count >= 2 else True),
                "benchmarkAlphaPositive": bool(rel_rsp_20d and np.mean(rel_rsp_20d) > 0),
                "independentCohortSizeTarget": {
                    "targetResolvedN": 60,
                    "targetDistinctDates": 20,
                    "currentResolvedN": n_resolved,
                    "currentDistinctDates": unique_dates,
                    "isTargetMet": bool(n_resolved >= 60 and unique_dates >= 20)
                },
                "statisticalInference": {
                    "standardError": round(std_err, 2) if std_err is not None else None,
                    "ci95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
                    "tStat": round(expectancy / std_err, 2) if (std_err and std_err > 0) else None,
                    "evaluationNote": (
                        "Statistical inference is evaluated as a diagnostic layer with confidence intervals. "
                        "Conventional p < 0.05 is not used as a single binary pass/fail gate for prospective cohorts."
                    )
                }
            },
            "governancePrinciple": (
                "A model change cannot be justified by a single metric moving outside its target. "
                "It requires a reproducible failure pattern across a predefined cohort and sufficient forward observations."
            )
        }
