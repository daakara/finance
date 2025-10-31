"""
Input validation functions for the financial analysis platform.
Ensures data integrity and prevents errors from invalid inputs.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Main input validation class."""
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        Validate stock symbol format.
        
        Args:
            symbol: Stock symbol to validate
        
        Returns:
            Tuple of (is_valid, cleaned_symbol)
        """
        if not symbol or not isinstance(symbol, str):
            return False, ""
        
        # Clean the symbol
        cleaned = symbol.strip().upper()
        
        # Check length (reasonable range for stock symbols)
        if len(cleaned) < 1 or len(cleaned) > 10:
            return False, ""
        
        # Check for valid characters (letters, numbers, dots, hyphens)
        if not re.match(r'^[A-Z0-9.-]+$', cleaned):
            return False, ""
        
        # Additional checks for common patterns
        if cleaned.startswith('.') or cleaned.endswith('.'):
            return False, ""
        
        return True, cleaned
    
    @staticmethod
    def validate_symbol_list(symbols: List[str], max_symbols: int = 50) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a list of stock symbols.
        
        Args:
            symbols: List of symbols to validate
            max_symbols: Maximum number of symbols allowed
        
        Returns:
            Tuple of (is_valid, valid_symbols, invalid_symbols)
        """
        if not symbols or not isinstance(symbols, list):
            return False, [], []
        
        if len(symbols) > max_symbols:
            logger.warning(f"Too many symbols provided: {len(symbols)} > {max_symbols}")
            return False, [], symbols
        
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            is_valid, cleaned = InputValidator.validate_symbol(symbol)
            if is_valid:
                if cleaned not in valid_symbols:  # Avoid duplicates
                    valid_symbols.append(cleaned)
            else:
                invalid_symbols.append(symbol)
        
        return len(valid_symbols) > 0, valid_symbols, invalid_symbols
    
    @staticmethod
    def validate_date_range(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        max_years: int = 20
    ) -> Tuple[bool, datetime, datetime]:
        """
        Validate date range for data fetching.
        
        Args:
            start_date: Start date
            end_date: End date
            max_years: Maximum years allowed in range
        
        Returns:
            Tuple of (is_valid, validated_start, validated_end)
        """
        try:
            # Convert to datetime if strings
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date)
            else:
                start_dt = start_date
            
            if isinstance(end_date, str):
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = end_date
            
            # Check if start is before end
            if start_dt >= end_dt:
                return False, start_dt, end_dt
            
            # Check if dates are not too far in the future
            today = datetime.now()
            if end_dt > today:
                end_dt = today
            
            # Check if range is not too large
            date_range = end_dt - start_dt
            if date_range.days > (max_years * 365):
                return False, start_dt, end_dt
            
            # Check if start date is not too far in the past (reasonable limit)
            min_date = datetime(1900, 1, 1)
            if start_dt < min_date:
                return False, start_dt, end_dt
            
            return True, start_dt, end_dt
            
        except Exception as e:
            logger.error(f"Date validation error: {str(e)}")
            return False, datetime.now() - timedelta(days=365), datetime.now()
    
    @staticmethod
    def validate_period_string(period: str) -> Tuple[bool, str]:
        """
        Validate period string for data fetching.
        
        Args:
            period: Period string (e.g., '1y', '6mo', '5d')
        
        Returns:
            Tuple of (is_valid, validated_period)
        """
        if not period or not isinstance(period, str):
            return False, "1y"
        
        period = period.lower().strip()
        
        valid_periods = {
            '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        }
        
        return period in valid_periods, period
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        Validate portfolio weights dictionary.
        
        Args:
            weights: Dictionary of symbol -> weight
        
        Returns:
            Tuple of (is_valid, normalized_weights)
        """
        if not weights or not isinstance(weights, dict):
            return False, {}
        
        validated_weights = {}
        
        # Validate each weight
        for symbol, weight in weights.items():
            # Validate symbol
            is_valid_symbol, cleaned_symbol = InputValidator.validate_symbol(symbol)
            if not is_valid_symbol:
                continue
            
            # Validate weight
            try:
                weight_float = float(weight)
                if weight_float < 0:
                    logger.warning(f"Negative weight for {symbol}: {weight_float}")
                    continue
                
                validated_weights[cleaned_symbol] = weight_float
                
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight for {symbol}: {weight}")
                continue
        
        if not validated_weights:
            return False, {}
        
        # Check if weights sum to reasonable range
        total_weight = sum(validated_weights.values())
        if total_weight <= 0:
            return False, {}
        
        # Normalize weights if they don't sum to 1 (within tolerance)
        if abs(total_weight - 1.0) > 0.01:
            logger.info(f"Normalizing weights (sum was {total_weight})")
            validated_weights = {k: v / total_weight for k, v in validated_weights.items()}
        
        return True, validated_weights
    
    @staticmethod
    def validate_numeric_parameter(
        value: Any,
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default_value: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Validate numeric parameter with optional bounds.
        
        Args:
            value: Value to validate
            param_name: Parameter name for logging
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default_value: Default value if validation fails
        
        Returns:
            Tuple of (is_valid, validated_value)
        """
        try:
            numeric_value = float(value)
            
            # Check bounds
            if min_value is not None and numeric_value < min_value:
                logger.warning(f"{param_name} below minimum: {numeric_value} < {min_value}")
                return False, default_value if default_value is not None else min_value
            
            if max_value is not None and numeric_value > max_value:
                logger.warning(f"{param_name} above maximum: {numeric_value} > {max_value}")
                return False, default_value if default_value is not None else max_value
            
            return True, numeric_value
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid {param_name}: {value}")
            return False, default_value if default_value is not None else 0.0
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if len(df) < min_rows:
            return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for all NaN columns
        for col in required_columns:
            if df[col].isna().all():
                return False, f"Column '{col}' contains only NaN values"
        
        return True, "Valid DataFrame"
    
    @staticmethod
    def validate_time_series_data(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        check_monotonic: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate time series data structure.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of date column (if None, assumes DatetimeIndex)
            check_monotonic: Whether to check if dates are monotonic
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic DataFrame validation
        is_valid, error_msg = InputValidator.validate_dataframe(df, [], min_rows=2)
        if not is_valid:
            return False, error_msg
        
        # Check date index or column
        if date_column is None:
            # Assume index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                return False, "Index is not DatetimeIndex"
            date_series = df.index
        else:
            if date_column not in df.columns:
                return False, f"Date column '{date_column}' not found"
            
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                return False, f"Column '{date_column}' is not datetime type"
            
            date_series = df[date_column]
        
        # Check for monotonic dates if requested
        if check_monotonic and not date_series.is_monotonic_increasing:
            return False, "Dates are not in ascending order"
        
        # Check for duplicate dates
        if date_series.duplicated().any():
            return False, "Duplicate dates found"
        
        return True, "Valid time series data"

class DataSanitizer:
    """Data sanitization functions."""
    
    @staticmethod
    def sanitize_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize OHLCV data by removing invalid values.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Sanitized DataFrame
        """
        if df.empty:
            return df
        
        sanitized_df = df.copy()
        
        # Remove rows where price columns are <= 0
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in sanitized_df.columns]
        
        for col in available_price_cols:
            sanitized_df = sanitized_df[sanitized_df[col] > 0]
        
        # Remove rows where volume is negative
        if 'Volume' in sanitized_df.columns:
            sanitized_df = sanitized_df[sanitized_df['Volume'] >= 0]
        
        # Validate OHLC relationships
        if all(col in sanitized_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Open, Close, Low
            valid_high = (
                (sanitized_df['High'] >= sanitized_df['Open']) &
                (sanitized_df['High'] >= sanitized_df['Close']) &
                (sanitized_df['High'] >= sanitized_df['Low'])
            )
            
            # Low should be <= Open, Close, High
            valid_low = (
                (sanitized_df['Low'] <= sanitized_df['Open']) &
                (sanitized_df['Low'] <= sanitized_df['Close']) &
                (sanitized_df['Low'] <= sanitized_df['High'])
            )
            
            # Keep only rows with valid OHLC relationships
            sanitized_df = sanitized_df[valid_high & valid_low]
        
        # Remove extreme outliers (prices that change by more than 50% in one day)
        if 'Close' in sanitized_df.columns and len(sanitized_df) > 1:
            returns = sanitized_df['Close'].pct_change().abs()
            sanitized_df = sanitized_df[returns <= 0.5]  # Remove >50% daily changes
        
        return sanitized_df.dropna()
    
    @staticmethod
    def sanitize_financial_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Sanitize financial ratios by removing extreme values.
        
        Args:
            ratios: Dictionary of ratio name -> value
        
        Returns:
            Sanitized ratios dictionary
        """
        sanitized_ratios = {}
        
        # Define reasonable bounds for common ratios
        ratio_bounds = {
            'pe_ratio': (0, 1000),
            'peg_ratio': (0, 10),
            'price_to_book': (0, 50),
            'debt_to_equity': (0, 10),
            'current_ratio': (0, 20),
            'quick_ratio': (0, 20),
            'roe': (-100, 100),  # Percentage
            'roa': (-100, 100),  # Percentage
            'profit_margin': (-100, 100),  # Percentage
        }
        
        for ratio_name, value in ratios.items():
            if value is None or pd.isna(value):
                continue
            
            try:
                float_value = float(value)
                
                # Check bounds if defined
                if ratio_name in ratio_bounds:
                    min_val, max_val = ratio_bounds[ratio_name]
                    if min_val <= float_value <= max_val:
                        sanitized_ratios[ratio_name] = float_value
                    else:
                        logger.warning(f"Ratio {ratio_name} out of bounds: {float_value}")
                else:
                    # No specific bounds, just check if it's a reasonable number
                    if abs(float_value) < 1e10:  # Avoid extremely large numbers
                        sanitized_ratios[ratio_name] = float_value
                
            except (ValueError, TypeError):
                logger.warning(f"Invalid ratio value for {ratio_name}: {value}")
                continue
        
        return sanitized_ratios

# Create validator instances for easy importing
input_validator = InputValidator()
data_sanitizer = DataSanitizer()
