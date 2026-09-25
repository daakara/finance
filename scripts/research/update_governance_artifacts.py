"""Synchronizes ETF Denominator Blocker Ledger and Evidence Completeness Matrix after Track A Mandate Resolution."""

import json
from pathlib import Path
import pandas as pd

LEDGER_PATH = Path("docs/research/ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet")
MATRIX_PATH = Path("docs/research/ETF_EVIDENCE_COMPLETENESS_MATRIX_V1.parquet")
SNAPSHOT_PATH = Path("docs/research/ETF_SURVIVING_UNIVERSE_V1.parquet")
MANIFEST_PATH = Path("docs/research/ETF_SURVIVING_UNIVERSE_V1_MANIFEST.json")
MANDATE_EVIDENCE_PATH = Path("data/research/etf_mandate_evidence_v1.json")

def sync_governance_artifacts():
    print("Synchronizing governance artifacts...")
    df_snap = pd.read_parquet(SNAPSHOT_PATH)
    df_ledger = pd.read_parquet(LEDGER_PATH)
    df_matrix = pd.read_parquet(MATRIX_PATH)

    with open(MANDATE_EVIDENCE_PATH, "r", encoding="utf-8") as f:
        mandate_db = json.load(f)
    mandate_map = {e["symbol"]: e for e in mandate_db["entries"]}
    snap_map = df_snap.set_index("symbol").to_dict("index")

    # 1. Update Blocker Ledger
    updated_ledger = df_ledger.copy()
    for idx, row in updated_ledger.iterrows():
        sym = row["symbol"]
        b_type = row["blocker_type"]
        if b_type == "MANDATE_BLOCKED":
            snap_info = snap_map.get(sym)
            if snap_info:
                sub = snap_info["research_subtype"]
                if sub != "UNRESOLVED":
                    updated_ledger.at[idx, "blocker_type"] = "RESOLVED"
                    if sub in ["EQUITY_INDEX", "EQUITY_SECTOR", "FIXED_INCOME_GOVERNMENT", "FIXED_INCOME_CREDIT", "COMMODITY_PHYSICAL"]:
                        updated_ledger.at[idx, "blocker_reason"] = f"RESOLVED_{sub}"
                        updated_ledger.at[idx, "resolution_status"] = "RESOLVED_CONFIRMATORY_CANDIDATE"
                    else:
                        updated_ledger.at[idx, "blocker_reason"] = "RESOLVED_NON_CONFIRMATORY_MANDATE"
                        updated_ledger.at[idx, "resolution_status"] = "RESOLVED_NON_CONFIRMATORY_OTHER_ETF"
                    updated_ledger.at[idx, "final_subtype"] = sub
                    updated_ledger.at[idx, "mandate_available"] = True
                    updated_ledger.at[idx, "denominator_blocking"] = False
                    updated_ledger.at[idx, "resolution_evidence"] = snap_info["classification_evidence"]
                else:
                    m_info = mandate_map.get(sym, {})
                    status = m_info.get("mandate_status", "STATUTORY_PROSPECTUS_NOT_CACHED")
                    updated_ledger.at[idx, "resolution_evidence"] = f"MANDATE_EXECUTION_{status}"

    updated_ledger.to_parquet(LEDGER_PATH, index=False)
    print(f"Updated {LEDGER_PATH}: {len(updated_ledger)} rows, {updated_ledger['denominator_blocking'].sum()} blocking")

    # 2. Update Evidence Completeness Matrix
    updated_matrix = df_matrix.copy()
    for idx, row in updated_matrix.iterrows():
        sym = row["symbol"]
        snap_info = snap_map.get(sym)
        if snap_info:
            sub = snap_info["research_subtype"]
            updated_matrix.at[idx, "final_subtype"] = sub
            if sub != "UNRESOLVED":
                updated_matrix.at[idx, "all_applicable_rules_evaluable"] = True
                updated_matrix.at[idx, "denominator_blocking"] = False
                if sub in ["EQUITY_INDEX", "EQUITY_SECTOR", "FIXED_INCOME_GOVERNMENT", "FIXED_INCOME_CREDIT", "COMMODITY_PHYSICAL"]:
                    updated_matrix.at[idx, "resolution_status"] = "RESOLVED_CONFIRMATORY_CANDIDATE"
                    updated_matrix.at[idx, "blocker_reason"] = f"RESOLVED_{sub}"
                else:
                    updated_matrix.at[idx, "resolution_status"] = "RESOLVED_NON_CONFIRMATORY_OTHER_ETF"
                    updated_matrix.at[idx, "blocker_reason"] = "RESOLVED_NON_CONFIRMATORY_MANDATE"
                updated_matrix.at[idx, "mandate_available"] = True
            else:
                updated_matrix.at[idx, "all_applicable_rules_evaluable"] = False
                updated_matrix.at[idx, "denominator_blocking"] = True
                updated_matrix.at[idx, "resolution_status"] = "UNRESOLVED_BLOCKING"

    updated_matrix.to_parquet(MATRIX_PATH, index=False)
    print(f"Updated {MATRIX_PATH}: {len(updated_matrix)} rows, {updated_matrix['denominator_blocking'].sum()} blocking")

    # 3. Assert exact parity
    ledger_blocking = int((updated_ledger["denominator_blocking"] == True).sum())
    matrix_blocking = int((updated_matrix["denominator_blocking"] == True).sum())
    snapshot_unresolved = int((df_snap["research_subtype"] == "UNRESOLVED").sum())
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_blocking = manifest["remaining_denominator_blockers"]

    assert ledger_blocking == matrix_blocking == snapshot_unresolved == manifest_blocking == 3580, (
        f"Parity mismatch: ledger={ledger_blocking}, matrix={matrix_blocking}, snap={snapshot_unresolved}, manifest={manifest_blocking}"
    )
    print("Exact 4-artifact parity verified: 3580 denominator blockers across all artifacts.")

if __name__ == "__main__":
    sync_governance_artifacts()
