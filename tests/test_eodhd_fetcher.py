"""Tests for EODHD Market Data Fetcher & Failover Pipeline."""

from analyst_dashboard.data.eodhd_fetcher import EODHDMarketFetcher

def test_eodhd_fetcher_realtime_quote():
    fetcher = EODHDMarketFetcher()
    quote = fetcher.fetch_realtime_quote("AAPL")
    assert quote is not None
    assert "close" in quote
    assert quote["close"] > 0

def test_eodhd_fetcher_historical_candles():
    fetcher = EODHDMarketFetcher()
    df = fetcher.fetch_historical_candles("AAPL")
    assert df is not None
    assert not df.empty
    assert "Close" in df.columns
    assert len(df) > 50
