import pandas as pd
from pathlib import Path
from scripts.research.build_etf_dataset import evaluate_liquidity_and_history, CACHE_DIR

def run():
    candidates = [
        'AAAU', 'GLD', 'IAU', 'OUNZ', 'PALL', 'PPLT', 'SGOL', 'SIVR', 'SLV',
        'SPY', 'DIA', 'QQQ', 'IWM', 'VOO', 'IVV', 'VTI', 'SCHX', 'RSP', 'IJH', 'IJR', 'VB', 'VO',
        'SCHD', 'VUG', 'VTV', 'IEFA', 'IEMG',
        'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB',
        'XOP', 'XBI', 'SMH', 'VNQ', 'IYR', 'ITB', 'XHB', 'KRE', 'KBE', 'IBB',
        'TLT', 'IEF', 'SHY', 'IEI', 'GOVT', 'VGSH', 'VGIT', 'VGLT', 'SCHO', 'SCHR', 'SPTL',
        'BIL', 'TIP', 'SHV',
        'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'USIG', 'FLOT', 'SJNK', 'HYLB', 'VUSB'
    ]

    price_dict = {}
    for sym in candidates:
        df = pd.read_parquet(CACHE_DIR / f'{sym}_adj.parquet')
        price_dict[sym] = df

    all_dates = pd.DatetimeIndex([])
    for df in price_dict.values():
        all_dates = all_dates.union(df.index)
    eval_calendar = all_dates[(all_dates <= pd.Timestamp('2026-09-24')) & (all_dates >= pd.Timestamp('2025-01-01'))].sort_values()

    res = evaluate_liquidity_and_history(
        symbols=candidates,
        price_data_adj=price_dict,
        trading_calendar=eval_calendar,
        adv_window=60,
        adv_percentile=0.80,
        min_history=250
    )

    as_of_res = res[res['observation_date'] == res['observation_date'].max()].sort_values(by='adv60', ascending=False)
    latest_date = as_of_res['observation_date'].iloc[0]
    print(f'Evaluation date: {latest_date}')
    print(f'Total candidates evaluated: {len(as_of_res)}')
    adv80_val = as_of_res['adv80_threshold'].iloc[0]
    print(f'ADV80 Threshold: ${adv80_val:,.2f}')
    liquid = as_of_res[as_of_res['is_liquid']]
    illiquid = as_of_res[~as_of_res['is_liquid']]
    print(f'ADV_PASS: {len(liquid)}')
    print(f'ADV_FAIL: {len(illiquid)}')
    print('\nEligible Symbols (ADV_PASS):')
    print(liquid[['symbol', 'adv60']].to_string())

if __name__ == '__main__':
    run()
