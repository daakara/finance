import json
from pathlib import Path
import pandas as pd
import numpy as np

def run():
    df_fund = pd.read_parquet('data/research/cache/nport_derived/target_fund_info.parquet')
    df_h = pd.read_parquet('data/research/cache/nport_derived/target_holdings.parquet')

    # Load N-CEN latest series info
    # We built latest_ncen earlier, let's load or rebuild
    import zipfile
    ncen_dir = Path('data/research/cache/sec_ncen')
    all_ncen = []
    for b_name in ['2025q1_ncen.zip', '2025q2_ncen.zip', '2025q3_ncen.zip', '2025q4_ncen.zip', '2026q1_ncen.zip', '2026q2_ncen.zip']:
        with zipfile.ZipFile(ncen_dir / b_name) as z:
            with z.open('SUBMISSION.tsv') as f:
                df_sub = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'FILING_DATE'])
                df_sub['FILING_DATE_PARSED'] = pd.to_datetime(df_sub['FILING_DATE'], format='%d-%b-%Y')
            with z.open('FUND_REPORTED_INFO.tsv') as f:
                df_f = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'SERIES_ID', 'IS_INDEX'], low_memory=False)
            m = pd.merge(df_f, df_sub, on='ACCESSION_NUMBER')
            m['BATCH'] = b_name
            all_ncen.append(m)
    df_ncen_all = pd.concat(all_ncen, ignore_index=True)
    df_ncen_all = df_ncen_all[df_ncen_all['FILING_DATE_PARSED'] <= pd.Timestamp('2026-09-24 23:59:59')]
    df_ncen_latest = df_ncen_all.sort_values(by=['SERIES_ID', 'FILING_DATE_PARSED']).drop_duplicates(subset=['SERIES_ID'], keep='last')

    ncen_map = dict(zip(df_ncen_latest['SERIES_ID'], df_ncen_latest['IS_INDEX']))

    # Compute portfolio stats per accession
    # Group df_h by ACCESSION_NUMBER
    print("Computing portfolio aggregations...")
    fund_stats = {}
    
    # Pre-calculate masks
    is_equity = df_h['ASSET_CAT'].isin(['EC', 'EP', 'DE'])
    is_govt = df_h['ISSUER_TYPE'].isin(['UST', 'USGA', 'USGSE'])
    is_corp_debt = (df_h['ISSUER_TYPE'] == 'CORP') & df_h['ASSET_CAT'].isin(['DBT', 'LON', 'ABS-CBDO', 'ABS-APCP', 'SN'])
    is_mbs = df_h['ASSET_CAT'] == 'ABS-MBS'

    df_h['is_eq'] = is_equity
    df_h['is_gov'] = is_govt
    df_h['is_corp'] = is_corp_debt
    df_h['is_mbs'] = is_mbs

    # Percentage in df_h is percentage points (e.g. 5.0 for 5%)
    # Scale to decimal (0.05 for 5%)
    df_h['pct_dec'] = df_h['PERCENTAGE'] / 100.0

    # Vectorized calculation columns
    df_h['eq_pct'] = np.where(is_equity, df_h['pct_dec'], 0.0)
    df_h['gov_pct'] = np.where(is_govt, df_h['pct_dec'], 0.0)
    df_h['corp_pct'] = np.where(is_corp_debt, df_h['pct_dec'], 0.0)
    df_h['mbs_pct'] = np.where(is_mbs, df_h['pct_dec'], 0.0)
    df_h['eq_count'] = np.where(is_equity, 1, 0)

    agg = df_h.groupby('ACCESSION_NUMBER').agg(
        total_holding_val=('CURRENCY_VALUE', 'sum'),
        total_pct=('pct_dec', 'sum'),
        total_equity_pct=('eq_pct', 'sum'),
        distinct_equity_count=('eq_count', 'sum'),
        max_concentration=('pct_dec', 'max'),
        total_govt_pct=('gov_pct', 'sum'),
        corporate_debt_pct=('corp_pct', 'sum'),
        mortgage_backed_pct=('mbs_pct', 'sum'),
        distinct_total_holdings=('HOLDING_ID', 'count')
    ).reset_index()

    print("Aggregation complete!")
    print(agg.head(5))

    # Merge with fund info
    merged = pd.merge(df_fund, agg, on='ACCESSION_NUMBER', how='left')
    merged['is_index_ncen'] = merged['SERIES_ID'].map(lambda s: ncen_map.get(s) == 'Y')
    
    # Check reconciliation
    # reconciliation residual = abs(net_assets - total_holding_val - cash) / net_assets
    total_investments_and_cash = merged['total_holding_val'] + merged['CASH_NOT_RPTD_IN_C_OR_D'].fillna(0.0)
    merged['reconciliation_diff'] = merged['NET_ASSETS'] - total_investments_and_cash
    merged['reconciliation_ratio'] = total_investments_and_cash / merged['NET_ASSETS']
    
    # Save
    out_path = Path('data/research/cache/nport_derived/portfolio_metrics.parquet')
    merged.to_parquet(out_path)
    print(f"Saved portfolio metrics to {out_path}")

    # Summary
    print(f"Total merged funds: {len(merged)}")
    print(f"Funds with equity >= 80%: {(merged['total_equity_pct'] >= 0.80).sum()}")
    print(f"Funds with govt >= 80%: {(merged['total_govt_pct'] >= 0.80).sum()}")
    print(f"Funds with corporate debt >= 50%: {(merged['corporate_debt_pct'] >= 0.50).sum()}")

if __name__ == '__main__':
    run()
