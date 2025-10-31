"""
Data processing and transformation utilities.
Handles data cleaning, normalization, and feature engineering.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for financial data."""
    
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        clean_df = df.copy()
        
        # Ensure standard column names
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close'
        }
        
        clean_df.columns = [column_mapping.get(col.lower(), col) for col in clean_df.columns]
        
        # Remove rows with all NaN values
        clean_df = clean_df.dropna(how='all')
        
        # Forward fill missing values (common in financial data)
        numeric_columns = clean_df.select_dtypes(include=[np.number]).columns
        clean_df[numeric_columns] = clean_df[numeric_columns].fillna(method='ffill')
        
        # Validate OHLC relationships
        if all(col in clean_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            clean_df = DataProcessor._validate_ohlc_relationships(clean_df)
        
        # Remove extreme outliers (prices that are 0 or negative)
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in clean_df.columns]
        
        for col in available_price_cols:
            clean_df = clean_df[clean_df[col] > 0]
        
        return clean_df
    
    @staticmethod
    def _validate_ohlc_relationships(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that High >= Low and High >= Open, Close; Low <= Open, Close.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            pd.DataFrame: Validated data
        """
        # Create boolean masks for invalid data
        invalid_high = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close'])
        invalid_low = (df['Low'] > df['High']) | (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        
        # Log warnings for invalid data
        if invalid_high.any():
            logger.warning(f"Found {invalid_high.sum()} rows with invalid High values")
        if invalid_low.any():
            logger.warning(f"Found {invalid_low.sum()} rows with invalid Low values")
        
        # Remove invalid rows
        valid_rows = ~(invalid_high | invalid_low)
        return df[valid_rows]
    
    @staticmethod
    def normalize_prices(df: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
        """
        Normalize price data to a base value for comparison.
        
        Args:
            df: DataFrame with price data
            base_value: Base value for normalization (default: 100)
        
        Returns:
            pd.DataFrame: Normalized data
        """
        if df.empty:
            return df
        
        normalized_df = df.copy()
        
        # Find the first valid close price
        if 'Close' in df.columns:
            first_close = df['Close'].dropna().iloc[0] if not df['Close'].dropna().empty else 1
            
            price_columns = ['Open', 'High', 'Low', 'Close']
            available_cols = [col for col in price_columns if col in df.columns]
            
            for col in available_cols:
                normalized_df[col] = (df[col] / first_close) * base_value
        
        return normalized_df
    
    @staticmethod
    def calculate_returns(
        prices: pd.Series,
        method: str = 'simple',
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log' returns
            periods: Number of periods for return calculation
        
        Returns:
            pd.Series: Returns series
        """
        if method == 'simple':
            returns = prices.pct_change(periods=periods)
        elif method == 'log':
            returns = np.log(prices / prices.shift(periods))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns
    
    @staticmethod
    def calculate_volatility(
        returns: pd.Series,
        window: int = 30,
        annualized: bool = True
    ) -> pd.Series:
        """
        Calculate rolling volatility from returns.
        
        Args:
            returns: Returns series
            window: Rolling window size
            annualized: Whether to annualize volatility
        
        Returns:
            pd.Series: Volatility series
        """
        vol = returns.rolling(window=window).std()
        
        if annualized:
            # Assume 252 trading days per year
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        frequency: str,
        agg_method: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            df: DataFrame with time series data
            frequency: Resampling frequency ('D', 'W', 'M', etc.)
            agg_method: Dictionary specifying aggregation method for each column
        
        Returns:
            pd.DataFrame: Resampled data
        """
        if df.empty:
            return df
        
        # Default aggregation methods for OHLCV data
        default_agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Adj Close': 'last'
        }
        
        if agg_method is None:
            agg_method = default_agg
        
        # Only use aggregation methods for columns that exist
        agg_dict = {col: method for col, method in agg_method.items() if col in df.columns}
        
        # For any remaining numeric columns, use 'last'
        remaining_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                         if col not in agg_dict]
        for col in remaining_cols:
            agg_dict[col] = 'last'
        
        try:
            resampled = df.resample(frequency).agg(agg_dict)
            return resampled.dropna()
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            return df

class MultiAssetProcessor:
    """Processor for handling multiple assets simultaneously."""
    
    @staticmethod
    def align_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames to common date index.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
        
        Returns:
            Dict[str, pd.DataFrame]: Aligned data
        """
        if not data_dict:
            return {}
        
        # Find common date range
        all_indices = [df.index for df in data_dict.values() if not df.empty]
        if not all_indices:
            return data_dict
        
        # Get intersection of all indices
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        # Align each DataFrame to common index
        aligned_data = {}
        for symbol, df in data_dict.items():
            if not df.empty and len(common_index) > 0:
                aligned_data[symbol] = df.reindex(common_index)
            else:
                aligned_data[symbol] = df
        
        return aligned_data
    
    @staticmethod
    def create_correlation_matrix(
        data_dict: Dict[str, pd.DataFrame],
        column: str = 'Close'
    ) -> pd.DataFrame:
        """
        Create correlation matrix from multiple assets.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            column: Column to use for correlation calculation
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Extract the specified column from each DataFrame
        price_data = {}
        for symbol, df in data_dict.items():
            if not df.empty and column in df.columns:
                price_data[symbol] = df[column]
        
        if not price_data:
            return pd.DataFrame()
        
        # Create combined DataFrame
        combined_df = pd.DataFrame(price_data)
        
        # Calculate correlation matrix
        correlation_matrix = combined_df.corr()
        
        return correlation_matrix
    
    @staticmethod
    def calculate_portfolio_returns(
        data_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        column: str = 'Close'
    ) -> pd.Series:
        """
        Calculate portfolio returns given weights.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            weights: Dictionary of symbol -> weight
            column: Price column to use
        
        Returns:
            pd.Series: Portfolio returns
        """
        # Align data first
        aligned_data = MultiAssetProcessor.align_data(data_dict)
        
        # Extract returns for each asset
        returns_data = {}
        for symbol, df in aligned_data.items():
            if not df.empty and column in df.columns and symbol in weights:
                returns = DataProcessor.calculate_returns(df[column])
                returns_data[symbol] = returns
        
        if not returns_data:
            return pd.Series()
        
        # Create combined returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=returns_df.index)
        
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight
        
        return portfolio_returns

# Create instances for easy importing
data_processor = DataProcessor()
multi_asset_processor = MultiAssetProcessor()
