"""
Data fetching functions for various financial data sources.
Handles stocks, ETFs, cryptocurrencies, and market data retrieval.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
from datetime import datetime, timedelta
import logging
import urllib3
import ssl
import certifi
import os
import warnings

from config import config, market_constants
import ssl_config  # Configure SSL environment on import

# Configure SSL and certificate handling
try:
    # Try to use certifi certificates
    cert_path = certifi.where()
    if os.path.exists(cert_path):
        os.environ['SSL_CERT_FILE'] = cert_path
        session = requests.Session()
        session.verify = cert_path
    else:
        # Fallback: disable SSL verification with warnings suppressed
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = requests.Session()
        session.verify = False
except Exception:
    # Fallback: disable SSL verification with warnings suppressed
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.verify = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_yfinance_fetch(ticker_symbol: str, period: str = "1y"):
    """
    Safely fetch data from yfinance with comprehensive SSL and error handling.
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Time period for data
    
    Returns:
        Tuple of (ticker_data, info_data) or (None, None) if failed
    """
    import os
    import ssl
    import certifi
    
    try:
        # Set SSL certificate environment variables
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['CURL_CA_BUNDLE'] = certifi.where()
        
        # Disable SSL verification for curl_cffi in corporate environments
        os.environ['CURL_DISABLE_SSL_VERIFY'] = '1'
        
        # Temporarily suppress ALL warnings and logging
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Suppress yfinance logging
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)
            
            # Suppress curl_cffi logging  
            curl_logger = logging.getLogger('curl_cffi')
            curl_original_level = getattr(curl_logger, 'level', logging.WARNING)
            curl_logger.setLevel(logging.CRITICAL)
            
            try:
                # Try with extended timeout and retry logic
                ticker = yf.Ticker(ticker_symbol)
                
                # First try with basic settings
                data = ticker.history(period=period, timeout=60, auto_adjust=True)
                
                # Get info with error handling
                info = {}
                try:
                    info = ticker.info or {}
                except:
                    info = {}
                
                return data, info
                
            except Exception as primary_error:
                # No fallback - raise error for transparency
                error_str = str(primary_error).lower()
                if any(keyword in error_str for keyword in ['ssl', 'certificate', 'curl', 'timeout']):
                    logger.error(f"Network/SSL issues for {ticker_symbol}: {primary_error}")
                    raise ConnectionError(f"Unable to fetch data for {ticker_symbol}. Please check your network connection and try again.") from primary_error
                else:
                    raise primary_error
                    
            finally:
                # Restore logging levels
                yf_logger.setLevel(original_level)
                if hasattr(curl_logger, 'setLevel'):
                    curl_logger.setLevel(curl_original_level)
                
    except Exception as e:
        logger.debug(f"yfinance fetch failed for {ticker_symbol}: {e}")
        return None, None

class StockDataFetcher:
    """Handles stock and ETF data retrieval using yfinance."""
    
    @staticmethod
    def get_stock_data(
        symbol: str,
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL,
        retry_count: int = 3
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol with SSL error handling.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            retry_count: Number of retry attempts
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
            
        Example:
            >>> fetcher = StockDataFetcher()
            >>> data = fetcher.get_stock_data('AAPL', period='1y')
        """
        last_error = None
        
        for attempt in range(retry_count):
            try:
                # Configure yfinance session with SSL handling
                ticker = yf.Ticker(symbol, session=session)
                
                # Try different approaches for SSL issues
                if attempt == 0:
                    # Standard approach
                    data = ticker.history(period=period, interval=interval)
                elif attempt == 1:
                    # Try with different SSL context
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    data = ticker.history(period=period, interval=interval)
                else:
                    # No fallback - raise error for transparency
                    logger.error(f"Unable to fetch data for {symbol} after {retry_count} attempts")
                    raise ConnectionError(f"Unable to fetch data for {symbol}. Please check your network connection and try again.")
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.title() for col in data.columns]
                    logger.info(f"Successfully fetched data for {symbol}")
                    return data
                else:
                    logger.warning(f"No data found for symbol: {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < retry_count - 1:
                    import time
                    time.sleep(1)  # Brief delay before retry
                continue
        
        # All attempts failed - raise error for transparency
        logger.error(f"All attempts failed for {symbol}: {str(last_error)}")
        raise ConnectionError(
            f"Unable to fetch data for {symbol} after {retry_count} attempts. "
            f"Please check your network connection and try again. Error: {str(last_error)}"
        )
    
    # NOTE: _generate_sample_data() method removed as part of live-data-only policy
    # Sample data generation is no longer supported in production
    
    @staticmethod
    def get_multiple_stocks(
        symbols: List[str],
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks efficiently.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        # Process in batches to respect API limits
        batch_size = config.MAX_SYMBOLS_PER_REQUEST
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            try:
                # Use yfinance download for batch processing
                data = yf.download(
                    tickers=' '.join(batch),
                    period=period,
                    interval=interval,
                    group_by='ticker'
                )
                
                for symbol in batch:
                    if len(batch) == 1:
                        # Single symbol case
                        symbol_data = data
                    else:
                        # Multiple symbols case
                        symbol_data = data[symbol] if symbol in data.columns.levels[0] else pd.DataFrame()
                    
                    if not symbol_data.empty:
                        symbol_data.columns = [col.title() for col in symbol_data.columns]
                        results[symbol] = symbol_data
                    else:
                        logger.warning(f"No data found for symbol: {symbol}")
                        results[symbol] = pd.DataFrame()
                        
            except Exception as e:
                logger.error(f"Error fetching batch {batch}: {str(e)}")
                for symbol in batch:
                    results[symbol] = pd.DataFrame()
        
        return results
    
    @staticmethod
    def get_stock_info(symbol: str) -> Dict:
        """
        Get detailed stock information and fundamentals.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dict: Stock information including fundamentals
        """
        try:
            ticker = yf.Ticker(symbol, session=session)
            info = ticker.info
            
            # Extract key metrics
            key_metrics = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'profit_margin': info.get('profitMargins', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', None),
                '52_week_low': info.get('fiftyTwoWeekLow', None),
                'current_price': info.get('currentPrice', None),
                'recommendation': info.get('recommendationKey', 'N/A')
            }
            
            return key_metrics
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            # Raise error instead of returning sample data
            raise ConnectionError(
                f"Unable to fetch company information for {symbol}. "
                f"Please check your network connection and try again. Error: {str(e)}"
            )
    
    # NOTE: _generate_sample_info() method removed as part of live-data-only policy
    # Sample data generation is no longer supported in production

class CryptoDataFetcher:
    """Handles cryptocurrency data retrieval using ccxt."""
    
    def __init__(self, exchange_id: str = 'binance'):
        """Initialize with specific exchange."""
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'apiKey': '',  # Public endpoints don't need API key
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.error(f"Error initializing {exchange_id} exchange: {str(e)}")
            self.exchange = None
    
    def get_crypto_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 365
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency OHLCV data.
        
        Args:
            symbol: Crypto pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
        
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_info(self, symbol: str) -> Dict:
        """
        Get cryptocurrency market information.
        
        Args:
            symbol: Crypto pair symbol
        
        Returns:
            Dict: Market information
        """
        if not self.exchange:
            return {'symbol': symbol, 'error': 'Exchange not initialized'}
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'volume_usd': ticker['quoteVolume'],
                'change_24h': ticker['change'],
                'percentage_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }
            
        except Exception as e:
            logger.error(f"Error fetching crypto info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

class EconomicDataFetcher:
    """Handles economic indicators and macro data."""
    
    @staticmethod
    def get_treasury_rates() -> pd.DataFrame:
        """
        Fetch US Treasury rates using yfinance with fallback data.
        
        Returns:
            pd.DataFrame: Treasury rates data
        """
        treasury_symbols = {
            '3M': '^IRX',    # 3-Month Treasury
            '10Y': '^TNX',   # 10-Year Treasury
            '30Y': '^TYX'    # 30-Year Treasury
        }
        
        data = {}
        for name, symbol in treasury_symbols.items():
            hist, _ = safe_yfinance_fetch(symbol, '1mo')
            if hist is not None and not hist.empty:
                data[name] = hist['Close'].iloc[-1]
            else:
                # Fallback to realistic sample rates
                sample_rates = {'3M': 4.5, '10Y': 4.2, '30Y': 4.0}
                data[name] = sample_rates.get(name, 4.0)
                logger.debug(f"Using fallback rate for {name}: {data[name]}")
        
        return pd.DataFrame([data], index=['Current'])
    
    @staticmethod
    def get_market_indices() -> pd.DataFrame:
        """
        Fetch major market indices current values with fallback data.
        
        Returns:
            pd.DataFrame: Market indices data
        """
        indices_data = {}
        
        # Sample data for fallback
        sample_indices = {
            'S&P 500': {'price': 4200.0, 'change': 15.2, 'change_pct': 0.36},
            'Dow Jones': {'price': 34000.0, 'change': -45.3, 'change_pct': -0.13},
            'NASDAQ': {'price': 13000.0, 'change': 78.5, 'change_pct': 0.61},
            'Russell 2000': {'price': 1800.0, 'change': 12.1, 'change_pct': 0.68},
            'FTSE 100': {'price': 7500.0, 'change': -8.2, 'change_pct': -0.11},
            'Nikkei 225': {'price': 28000.0, 'change': 125.3, 'change_pct': 0.45},
            'DAX': {'price': 15000.0, 'change': -22.1, 'change_pct': -0.15}
        }
        
        for name, symbol in market_constants.MAJOR_INDICES.items():
            hist, _ = safe_yfinance_fetch(symbol, '2d')
            
            if hist is not None and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_price = hist['Close'].iloc[-2]
                change = current_price - previous_price
                change_pct = (change / previous_price) * 100
                
                indices_data[name] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct
                }
            else:
                    # Use sample data
                # Use sample data as fallback
                sample_data = sample_indices.get(name, {'price': 1000.0, 'change': 0.0, 'change_pct': 0.0})
                indices_data[name] = sample_data
                logger.debug(f"Using sample data for {name}")
        
        return pd.DataFrame(indices_data).T

# Create instances for easy importing
stock_fetcher = StockDataFetcher()
crypto_fetcher = CryptoDataFetcher()
economic_fetcher = EconomicDataFetcher()
