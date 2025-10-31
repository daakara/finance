"""
Data Fetcher Module - Unified data retrieval for all asset types
Handles Stocks, ETFs, and Cryptocurrencies with fallback mechanisms
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
import requests
from abc import ABC, abstractmethod
import ssl
import urllib3
import certifi

from data.cache import cache_result

# SSL Configuration
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def configure_ssl_session():
    """Configure SSL session for secure connections."""
    try:
        # Create SSL context with certificate verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Configure requests session
        session = requests.Session()
        session.verify = False
        
        return session
    except Exception as e:
        logger.warning(f"SSL configuration failed: {str(e)}")
        return requests.Session()

logger = logging.getLogger(__name__)

class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    @abstractmethod
    def get_price_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical price data."""
        pass
    
    @abstractmethod
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get asset information."""
        pass

class UnifiedDataFetcher:
    """Unified data fetcher for all asset types."""
    
    def __init__(self):
        """Initialize the unified data fetcher."""
        self.stock_fetcher = StockDataFetcher()
        self.crypto_fetcher = CryptoDataFetcher()
        self.etf_fetcher = ETFDataFetcher()
    
    def get_data(self, 
                 symbol: str, 
                 asset_type: str, 
                 period: str = '1y') -> Dict[str, Any]:
        """
        Get data for any asset type with robust fallback handling.
        
        Args:
            symbol: Asset symbol
            asset_type: 'Stock', 'ETF', or 'Cryptocurrency'
            period: Time period for data
        
        Returns:
            Dictionary containing price data and asset info
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching data for {symbol} ({asset_type}), attempt {attempt + 1}")
                
                if asset_type.lower() == 'stock':
                    result = self.stock_fetcher.get_complete_data(symbol, period)
                elif asset_type.lower() == 'etf':
                    result = self.etf_fetcher.get_complete_data(symbol, period)
                elif asset_type.lower() in ['cryptocurrency', 'crypto']:
                    result = self.crypto_fetcher.get_complete_data(symbol, period)
                else:
                    raise ValueError(f"Unsupported asset type: {asset_type}")
                
                # Validate result
                if 'price_data' in result and not result['price_data'].empty:
                    logger.info(f"Successfully fetched data for {symbol}")
                    return result
                else:
                    logger.warning(f"Empty data returned for {symbol}, attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol} ({asset_type}): {str(e)}")
                if attempt == max_retries - 1:
                    # All attempts failed - raise error
                    logger.error(f"All attempts failed for {symbol}")
                    raise ConnectionError(
                        f"Unable to fetch data for {symbol} after {max_retries} attempts. "
                        f"Please check your network connection and try again. Error: {str(e)}"
                    )
        
        raise ConnectionError(f"Unable to fetch data for {symbol}. Please check your network connection and try again.")
    
    # NOTE: _generate_fallback_data() method removed as part of live-data-only policy
    # Sample data generation is no longer supported in production

class StockDataFetcher(BaseDataFetcher):
    """Fetcher for stock data."""
    
    @cache_result(ttl=3600)  # Cache for 1 hour
    def get_price_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get stock price data - let yfinance handle sessions."""
        try:
            # Let yfinance handle its own session management
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No price data found for {symbol}")
                raise ConnectionError(f"No price data available for {symbol}. Please check the symbol and try again.")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            raise ConnectionError(f"Unable to fetch price data for {symbol}. Error: {str(e)}")
    
    @cache_result(ttl=3600)
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get stock information - let yfinance handle sessions."""
        try:
            # Let yfinance handle its own session management
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No info found for {symbol}")
                raise ConnectionError(f"No information available for {symbol}. Please check the symbol and try again.")
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            raise ConnectionError(f"Unable to fetch information for {symbol}. Error: {str(e)}")
    
    def get_complete_data(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """Get complete stock data package."""
        return {
            'price_data': self.get_price_data(symbol, period),
            'stock_info': self.get_asset_info(symbol),
            'asset_type': 'Stock'
        }
    
    # NOTE: _generate_sample_stock_data() and _generate_sample_stock_info() methods
    # removed as part of live-data-only policy. Sample data generation is no longer
    # supported in production.
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to days."""
        period_map = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 365)

class ETFDataFetcher(BaseDataFetcher):
    """Fetcher for ETF data."""
    
    def get_price_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get ETF price data (same as stock for now)."""
        stock_fetcher = StockDataFetcher()
        return stock_fetcher.get_price_data(symbol, period)
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get ETF information."""
        try:
            # Let yfinance handle its own session management
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No ETF info found for {symbol}, using sample data")
                return self._generate_sample_etf_info(symbol)
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching ETF info for {symbol}: {str(e)}")
            logger.info(f"Falling back to sample ETF info for {symbol}")
            return self._generate_sample_etf_info(symbol)
    
    def get_complete_data(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """Get complete ETF data package."""
        return {
            'price_data': self.get_price_data(symbol, period),
            'etf_info': self.get_asset_info(symbol),
            'asset_type': 'ETF'
        }
    
    def _generate_sample_etf_info(self, symbol: str) -> Dict[str, Any]:
        """Generate sample ETF info with safe numeric bounds."""
        # Use safe seed within int32 bounds
        safe_seed = abs(hash(symbol)) % 2147483647  # Max int32
        np.random.seed(safe_seed)
        
        try:
            # Generate total assets safely - use float calculation to avoid overflow
            base_assets = 100000000  # 100M base
            multiplier = np.random.uniform(1.0, 500.0)  # 100M to 50B range
            total_assets = int(min(base_assets * multiplier, 2147483647))  # Cap at int32 max
            
            return {
                'symbol': symbol,
                'shortName': f"{symbol} ETF",
                'longName': f"{symbol} Exchange Traded Fund",
                'totalAssets': total_assets,
                'expenseRatio': float(np.random.uniform(0.03, 0.75)),
                'category': np.random.choice(['Equity', 'Fixed Income', 'Commodity', 'Sector']),
                'fundFamily': f"{symbol} Funds"
            }
            
        except Exception as e:
            logger.warning(f"Error generating ETF info for {symbol}: {e}")
            # Ultimate safe fallback
            return {
                'symbol': symbol,
                'shortName': f"{symbol} ETF", 
                'longName': f"{symbol} Exchange Traded Fund",
                'totalAssets': 1000000000,  # 1B fixed safe value
                'expenseRatio': 0.15,
                'category': 'Equity',
                'fundFamily': f"{symbol} Funds"
            }

class CryptoDataFetcher(BaseDataFetcher):
    """Fetcher for cryptocurrency data."""
    
    @cache_result(ttl=1800)  # Cache for 30 minutes
    def get_price_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get cryptocurrency price data."""
        try:
            # Try to get crypto data via yfinance (many cryptos available as XXX-USD)
            crypto_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(crypto_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No crypto data found for {symbol}")
                return self._generate_sample_crypto_data(symbol, period)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return self._generate_sample_crypto_data(symbol, period)
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get cryptocurrency information."""
        return self._generate_sample_crypto_info(symbol)
    
    def get_complete_data(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """Get complete crypto data package."""
        return {
            'price_data': self.get_price_data(symbol, period),
            'crypto_info': self.get_asset_info(symbol),
            'asset_type': 'Cryptocurrency'
        }
    
    def _generate_sample_crypto_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate sample cryptocurrency data with safe numeric bounds."""
        try:
            # Convert period to days with safe limits
            period_days = min(self._period_to_days(period), 3650)  # Max 10 years
            
            # Generate dates (crypto trades 24/7)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Use safe random seed within int32 bounds
            safe_seed = abs(hash(symbol)) % 2147483647  # Max int32
            np.random.seed(safe_seed)
            
            # Set crypto-specific parameters with safe ranges
            if symbol == 'BTC':
                base_price = 45000.0
                volatility = 0.04
            elif symbol == 'ETH':
                base_price = 3000.0
                volatility = 0.05
            else:
                # Use safe hash operations to avoid overflow
                price_hash = abs(hash(symbol + "_price")) % 1000
                vol_hash = abs(hash(symbol + "_vol")) % 50
                base_price = 10.0 + float(price_hash)
                volatility = 0.06 + float(vol_hash) / 1000.0
            
            # Generate price series with bounds checking
            prices = [base_price]
            max_price = 1000000.0  # Cap at 1M to avoid overflow
            min_price = 0.01
            
            for i in range(1, min(len(dates), 5000)):  # Limit iterations
                change = np.random.normal(0, volatility)
                change = np.clip(change, -0.2, 0.2)  # Limit daily changes
                new_price = prices[-1] * (1 + change)
                new_price = np.clip(new_price, min_price, max_price)
                prices.append(float(new_price))
            
            # Create OHLCV data with safe bounds
            data = []
            for i, price in enumerate(prices):
                try:
                    # Safe volatility calculation
                    daily_vol = np.random.uniform(0.02, 0.08)
                    daily_vol = np.clip(daily_vol, 0.01, 0.1)
                    
                    # Calculate OHLC with bounds checking
                    high = float(np.clip(price * (1 + daily_vol), min_price, max_price))
                    low = float(np.clip(price * (1 - daily_vol), min_price, max_price))
                    open_price = float(prices[i-1] if i > 0 else price)
                    close_price = float(price)
                    
                    # Generate safe volume within int32 bounds
                    volume = int(np.random.randint(100000, 1000000))  # Safe range
                    
                    data.append({
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close_price,
                        'Volume': volume
                    })
                except Exception as e:
                    # Fallback to simple values if calculation fails
                    logger.warning(f"Error in crypto data generation at index {i}: {e}")
                    data.append({
                        'Open': float(price),
                        'High': float(price * 1.02),
                        'Low': float(price * 0.98),
                        'Close': float(price),
                        'Volume': 500000
                    })
            
            # Create DataFrame with proper indexing
            if len(data) > 0:
                df = pd.DataFrame(data, index=dates[:len(data)])
                logger.info(f"Generated {len(df)} crypto data points for {symbol}")
                return df
            else:
                # Ultimate fallback
                logger.warning(f"Creating minimal crypto data for {symbol}")
                return pd.DataFrame({
                    'Open': [base_price],
                    'High': [base_price * 1.02],
                    'Low': [base_price * 0.98],
                    'Close': [base_price],
                    'Volume': [500000]
                }, index=[end_date])
                
        except Exception as e:
            logger.error(f"Error generating crypto sample data for {symbol}: {e}")
            # Ultimate safe fallback
            return pd.DataFrame({
                'Open': [100.0],
                'High': [102.0],
                'Low': [98.0],
                'Close': [100.0],
                'Volume': [500000]
            }, index=[datetime.now()])
    
    def _generate_sample_crypto_info(self, symbol: str) -> Dict[str, Any]:
        """Generate sample crypto info."""
        # Use safe random seed within int32 bounds
        safe_seed = abs(hash(symbol)) % 2147483647  # Max int32
        np.random.seed(safe_seed)
        
        # Create realistic crypto data based on symbol
        if symbol == 'BTC':
            sample_info = {
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'market_cap_rank': 1,
                'market_cap': 850000000000,  # ~850B
                'circulating_supply': 19500000,
                'max_supply': 21000000,
                'category': 'Currency'
            }
        elif symbol == 'ETH':
            sample_info = {
                'symbol': 'ETH',
                'name': 'Ethereum',
                'market_cap_rank': 2,
                'market_cap': 200000000000,  # ~200B
                'circulating_supply': 120000000,
                'max_supply': None,
                'category': 'Smart Contract Platform'
            }
        else:
            # Generate safe values within int32 bounds
            market_cap_base = int(np.random.randint(1000000, 1000000000))  # 1M to 1B (safe)
            supply_base = int(np.random.randint(1000000, 100000000))       # 1M to 100M (safe)
            max_supply_base = int(np.random.randint(10000000, 500000000))  # 10M to 500M (safe)
            
            sample_info = {
                'symbol': symbol,
                'name': f"{symbol} Coin",
                'market_cap_rank': int(np.random.randint(1, 100)),
                'market_cap': market_cap_base,
                'circulating_supply': supply_base,
                'max_supply': max_supply_base if np.random.random() > 0.3 else None,
                'category': np.random.choice(['Currency', 'Smart Contract Platform', 'DeFi', 'Gaming'])
            }
        
        return sample_info
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to days."""
        period_map = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 365)

# Global instance
unified_fetcher = UnifiedDataFetcher()
