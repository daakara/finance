import zipfile
import pandas as pd
from pathlib import Path
from scripts.research.build_etf_dataset import (
    fetch_nasdaq_traded_directory,
    load_sec_mf_directory,
    ClassificationAuthorityEngine,
    CACHE_DIR
)

def run():
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

    # Latest N-PORT
    nport_dir = Path('data/research/cache/sec_nport')
    all_nport_info = []
    for b_name in ['2025q3_nport.zip', '2025q4_nport.zip', '2026q1_nport.zip', '2026q2_nport.zip']:
        with zipfile.ZipFile(nport_dir / b_name) as z:
            with z.open('SUBMISSION.tsv') as f:
                df_sub = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'REPORT_DATE'])
                df_sub['FILING_DATE_PARSED'] = pd.to_datetime(df_sub['FILING_DATE'], format='%d-%b-%Y')
                df_sub['REPORT_DATE_PARSED'] = pd.to_datetime(df_sub['REPORT_DATE'], format='%d-%b-%Y')
            with z.open('FUND_REPORTED_INFO.tsv') as f:
                df_fund = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'SERIES_ID', 'NET_ASSETS', 'TOTAL_ASSETS'], low_memory=False)
            m = pd.merge(df_fund, df_sub, on='ACCESSION_NUMBER', how='inner')
            m['NPORT_BATCH'] = b_name
            all_nport_info.append(m)

    df_all = pd.concat(all_nport_info, ignore_index=True)
    df_all = df_all[df_all['FILING_DATE_PARSED'] <= pd.Timestamp('2026-09-24 23:59:59')]
    df_latest = df_all.sort_values(by=['SERIES_ID', 'FILING_DATE_PARSED', 'REPORT_DATE_PARSED']).drop_duplicates(subset=['SERIES_ID'], keep='last')

    df_target = df_latest[df_latest['SERIES_ID'].isin(series_to_sym.keys())].copy()
    print(f'Target series in latest N-PORT: {len(df_target)}')
    print(f'Unique accessions for target series: {df_target["ACCESSION_NUMBER"].nunique()}')
    print('Accessions per batch:')
    print(df_target['NPORT_BATCH'].value_counts())

if __name__ == '__main__':
    run()
