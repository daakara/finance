import zipfile
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.research.build_etf_dataset import (
    fetch_nasdaq_traded_directory,
    load_sec_mf_directory,
    ClassificationAuthorityEngine,
    CACHE_DIR
)

def main():
    t0 = time.time()
    # 1. Structure-eligible universe
    df_disc, _ = fetch_nasdaq_traded_directory(CACHE_DIR)
    test_issue_col = 'Test Issue' if 'Test Issue' in df_disc.columns else 'TestIssue'
    df_clean = df_disc[df_disc[test_issue_col] == 'N'] if test_issue_col in df_disc.columns else df_disc
    etf_col = 'ETF' if 'ETF' in df_clean.columns else 'etf'
    df_etfs = df_clean[df_clean[etf_col].astype(str).str.strip().str.upper() == 'Y'].copy()
    sec_mf_map = load_sec_mf_directory(CACHE_DIR)

    series_to_sym = {}
    for _, row in df_etfs.iterrows():
        sym = row.get('Symbol', '')
        sec_info = sec_mf_map.get(str(sym).strip().upper())
        if sec_info and sec_info.get('series_id'):
            series_to_sym[sec_info['series_id']] = sym

    # 2. Latest N-PORT submission & fund info
    nport_dir = Path('data/research/cache/sec_nport')
    all_nport_info = []
    for b_name in ['2025q3_nport.zip', '2025q4_nport.zip', '2026q1_nport.zip', '2026q2_nport.zip']:
        b_path = nport_dir / b_name
        with zipfile.ZipFile(b_path) as z:
            with z.open('SUBMISSION.tsv') as f:
                df_sub = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'REPORT_DATE'])
                df_sub['FILING_DATE_PARSED'] = pd.to_datetime(df_sub['FILING_DATE'], format='%d-%b-%Y')
                df_sub['REPORT_DATE_PARSED'] = pd.to_datetime(df_sub['REPORT_DATE'], format='%d-%b-%Y')
            with z.open('FUND_REPORTED_INFO.tsv') as f:
                df_fund = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'SERIES_ID', 'NET_ASSETS', 'TOTAL_ASSETS', 'CASH_NOT_RPTD_IN_C_OR_D'], low_memory=False)
            m = pd.merge(df_fund, df_sub, on='ACCESSION_NUMBER', how='inner')
            m['NPORT_BATCH'] = b_name
            all_nport_info.append(m)

    df_all = pd.concat(all_nport_info, ignore_index=True)
    df_all = df_all[df_all['FILING_DATE_PARSED'] <= pd.Timestamp('2026-09-24 23:59:59')]
    # Latest per series
    df_latest = df_all.sort_values(by=['SERIES_ID', 'FILING_DATE_PARSED', 'REPORT_DATE_PARSED']).drop_duplicates(subset=['SERIES_ID'], keep='last')
    df_target = df_latest[df_latest['SERIES_ID'].isin(series_to_sym.keys())].copy()

    print(f"Target series in latest N-PORT: {len(df_target)}")
    target_accessions = set(df_target['ACCESSION_NUMBER'])
    target_acc_to_batch = dict(zip(df_target['ACCESSION_NUMBER'], df_target['NPORT_BATCH']))

    # Group target accessions by batch to only stream necessary zips
    batch_to_accs = {}
    for acc, b in target_acc_to_batch.items():
        batch_to_accs.setdefault(b, set()).add(acc)

    print("Batch distribution for target accessions:", {k: len(v) for k, v in batch_to_accs.items()})

    # 3. Process holdings
    # Aggregate metrics per accession
    acc_holdings_agg = []
    
    for b_name, acc_set in batch_to_accs.items():
        b_path = nport_dir / b_name
        print(f"Processing holdings from {b_name} for {len(acc_set)} accessions...")
        with zipfile.ZipFile(b_path) as z:
            with z.open('FUND_REPORTED_HOLDING.tsv') as f:
                chunks = pd.read_csv(
                    f,
                    sep='\t',
                    usecols=['ACCESSION_NUMBER', 'HOLDING_ID', 'ISSUER_NAME', 'PERCENTAGE', 'CURRENCY_VALUE', 'ASSET_CAT', 'ISSUER_TYPE'],
                    chunksize=250000,
                    low_memory=False
                )
                chunk_num = 0
                for chunk in chunks:
                    chunk_num += 1
                    target_chunk = chunk[chunk['ACCESSION_NUMBER'].isin(acc_set)]
                    if not target_chunk.empty:
                        acc_holdings_agg.append(target_chunk)
                    if chunk_num % 10 == 0:
                        print(f"  Processed {chunk_num * 250000} rows ({time.time() - t0:.1f}s)...")

    df_holdings = pd.concat(acc_holdings_agg, ignore_index=True)
    print(f"Total matched holdings rows: {len(df_holdings)} in {time.time() - t0:.1f}s")

    # 4. Save intermediate parquet for ultra-fast replay
    out_dir = Path('data/research/cache/nport_derived')
    out_dir.mkdir(parents=True, exist_ok=True)
    df_holdings.to_parquet(out_dir / 'target_holdings.parquet')
    df_target.to_parquet(out_dir / 'target_fund_info.parquet')
    print(f"Saved target holdings and fund info to {out_dir}")

if __name__ == '__main__':
    main()
