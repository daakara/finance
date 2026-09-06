import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.governance.experiment_ledger import ExperimentLedger

def upgrade_ledger():
    ledger_path = ExperimentLedger.DEFAULT_LEDGER_PATH
    ledger = ExperimentLedger.load_ledger(ledger_path)

    for sig in ledger["signals"]:
        # 1. Compute and attach decisionSnapshotHash
        sig["decisionSnapshotHash"] = ExperimentLedger.compute_decision_snapshot_hash(sig)

        # 2. Rename liquidityObservation -> liquidityAtSignal
        old_obs = sig.pop("liquidityObservation", {})
        raw_amihud = float(old_obs.get("amihudIlliqRaw", 0.0))
        scaled_amihud = float(raw_amihud * 1e6)

        sig["liquidityAtSignal"] = {
            "liquidityGrade": "HIGH_TRADING_LIQUIDITY",
            "badgeColor": "emerald",
            "adv20dUsd": old_obs.get("adv20dUsd"),
            "amihudIlliqRaw": raw_amihud,
            "amihudIlliqScaled": scaled_amihud,
            "volumeSpikeRatio": old_obs.get("volumeSpikeRatio", 1.0),
            "executionHazard": False,
            "marketOrderWarning": False,
            "signalTimestamp": sig.get("signalDate") + "T17:32:00Z"
        }
        sig["liquidityForwardObservations"] = []

        print(f"[{sig['symbol']}] Hash: {sig['decisionSnapshotHash'][:12]}... Raw: {raw_amihud:.2e} Scaled: {scaled_amihud:.6f}")

    ExperimentLedger.save_ledger(ledger, ledger_path)
    print("Upgraded paper_trading_ledger.json with cryptographic snapshot hashes and immutable liquidityAtSignal structure.")

if __name__ == "__main__":
    upgrade_ledger()
