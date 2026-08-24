"""Celery Background Task Queue Worker Definitions."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def background_prefetch_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Background task to pre-fetch price histories and update DiskCache."""
    from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline

    logger.info(f"Starting background market data prefetch for {len(symbols)} symbols...")
    pipeline = MultiAssetDataPipeline()
    successful = 0
    failed = 0

    for sym in symbols:
        try:
            pipeline.fetch_stock_data(sym, period="1y")
            successful += 1
        except Exception as e:
            logger.error(f"Failed prefetch for {sym}: {e}")
            failed += 1

    return {
        "symbols_processed": len(symbols),
        "successful": successful,
        "failed": failed,
    }

