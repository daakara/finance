"""
Asset Data Manager - Handles data fetching and processing for different asset types
Focused on data acquisition, validation, and preparation
"""

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd

logger = logging.getLogger(__name__)

class AssetDataManager:
    """Manages data fetching and processing for different asset types."""
    
    def get_asset_data(self, ticker: str, asset_type: str, period: str) -> Dict[str, Any]:
        """Get comprehensive data based on asset type."""
        try:
            if asset_type == "Stock":
                return self._get_stock_data(ticker, period)
            elif asset_type == "ETF":
                return self._get_etf_data(ticker, period)
            else:  # Cryptocurrency
                return self._get_crypto_data(ticker, period)
        except Exception as e:
            logger.error(f"Error getting asset data for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def _get_stock_data(self, ticker: str, period: str) -> Dict[str, Any]:
        """Get comprehensive stock data."""
        results = {}
        
        try:
            # Import stock fetcher
            from data.fetchers import stock_fetcher
            
            # Get price data
            price_data = stock_fetcher.get_stock_data(ticker, period)
            results['price_data'] = price_data
            
            # Get stock info
            stock_info = stock_fetcher.get_stock_info(ticker)
            results['stock_info'] = stock_info
            
            if not price_data.empty:
                # Import analysis modules
                from analysis.technical import technical_analysis
                from analysis.fundamental import fundamental_analysis
                
                # Technical analysis
                tech_analysis = technical_analysis.analyze_stock(ticker, price_data)
                results['technical_analysis'] = tech_analysis
                
                # Fundamental analysis
                fund_analysis = fundamental_analysis.analyze_stock(ticker, stock_info, price_data)
                results['fundamental_analysis'] = fund_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def _get_etf_data(self, ticker: str, period: str) -> Dict[str, Any]:
        """Get comprehensive ETF data."""
        try:
            from analysis.etf import etf_analyzer
            return etf_analyzer.get_etf_data(ticker, period)
        except Exception as e:
            logger.error(f"Error getting ETF data for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def _get_crypto_data(self, ticker: str, period: str) -> Dict[str, Any]:
        """Get comprehensive cryptocurrency data."""
        try:
            from analysis.crypto import crypto_analyzer
            return crypto_analyzer.get_crypto_data(ticker, period)
        except Exception as e:
            logger.error(f"Error getting crypto data for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def get_multiple_assets_data(self, tickers: List[str], asset_type: str, period: str) -> Dict[str, Dict[str, Any]]:
        """Get data for multiple assets."""
        results = {}
        
        for ticker in tickers:
            try:
                data = self.get_asset_data(ticker, asset_type, period)
                if 'error' not in data:
                    results[ticker] = data
                else:
                    logger.warning(f"Could not load data for {ticker}: {data['error']}")
            except Exception as e:
                logger.warning(f"Could not load data for {ticker}: {str(e)}")
        
        return results
    
    # Compatibility methods for existing workflow calls
    def fetch_stock_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch stock price data - compatibility method."""
        try:
            from data.fetchers import stock_fetcher
            return stock_fetcher.get_stock_data(ticker, period)
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info data - compatibility method."""
        try:
            from data.fetchers import stock_fetcher
            return stock_fetcher.get_stock_info(ticker)
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {str(e)}")
            return {}
    
    def fetch_etf_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch ETF price data - compatibility method."""
        try:
            from analysis.etf import etf_analyzer
            data = etf_analyzer.get_etf_data(ticker, period)
            return data.get('price_data', pd.DataFrame())
        except Exception as e:
            logger.error(f"Error fetching ETF data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_etf_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch ETF info data - compatibility method."""
        try:
            from analysis.etf import etf_analyzer
            data = etf_analyzer.get_etf_data(ticker, '1y')
            return data.get('etf_info', {})
        except Exception as e:
            logger.error(f"Error fetching ETF info for {ticker}: {str(e)}")
            return {}
    
    def fetch_crypto_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch crypto price data - compatibility method."""
        try:
            from analysis.crypto import crypto_analyzer
            data = crypto_analyzer.get_crypto_data(ticker, period)
            return data.get('price_data', pd.DataFrame())
        except Exception as e:
            logger.error(f"Error fetching crypto data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_crypto_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch crypto info data - compatibility method."""
        try:
            from analysis.crypto import crypto_analyzer
            data = crypto_analyzer.get_crypto_data(ticker, '1y')
            return data.get('crypto_info', {})
        except Exception as e:
            logger.error(f"Error fetching crypto info for {ticker}: {str(e)}")
            return {}
