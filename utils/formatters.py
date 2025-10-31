"""
Data formatting utilities for the financial analysis platform.
Provides consistent formatting for financial data display.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FinancialFormatter:
    """Formatter for financial data and metrics."""
    
    @staticmethod
    def format_currency(
        amount: Union[float, int],
        currency: str = "USD",
        compact: bool = True,
        decimal_places: int = 2
    ) -> str:
        """
        Format monetary amounts with appropriate scaling.
        
        Args:
            amount: Amount to format
            currency: Currency code
            compact: Whether to use compact notation (K, M, B, T)
            decimal_places: Number of decimal places
        
        Returns:
            Formatted currency string
        """
        if pd.isna(amount) or amount is None:
            return "N/A"
        
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return "N/A"
        
        if currency == "USD":
            symbol = "$"
        elif currency == "EUR":
            symbol = "€"
        elif currency == "GBP":
            symbol = "£"
        elif currency == "JPY":
            symbol = "¥"
        else:
            symbol = f"{currency} "
        
        if compact:
            if abs(amount) >= 1e12:
                return f"{symbol}{amount/1e12:.{decimal_places}f}T"
            elif abs(amount) >= 1e9:
                return f"{symbol}{amount/1e9:.{decimal_places}f}B"
            elif abs(amount) >= 1e6:
                return f"{symbol}{amount/1e6:.{decimal_places}f}M"
            elif abs(amount) >= 1e3:
                return f"{symbol}{amount/1e3:.{decimal_places}f}K"
            else:
                return f"{symbol}{amount:.{decimal_places}f}"
        else:
            return f"{symbol}{amount:,.{decimal_places}f}"
    
    @staticmethod
    def format_percentage(
        value: Union[float, int],
        decimal_places: int = 2,
        multiply_by_100: bool = True
    ) -> str:
        """
        Format percentage values.
        
        Args:
            value: Value to format
            decimal_places: Number of decimal places
            multiply_by_100: Whether to multiply by 100 (for decimal inputs)
        
        Returns:
            Formatted percentage string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            value = float(value)
            if multiply_by_100:
                value *= 100
            return f"{value:.{decimal_places}f}%"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_ratio(
        value: Union[float, int],
        decimal_places: int = 2,
        suffix: str = ""
    ) -> str:
        """
        Format financial ratios.
        
        Args:
            value: Ratio value
            decimal_places: Number of decimal places
            suffix: Optional suffix (e.g., "x" for multiples)
        
        Returns:
            Formatted ratio string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            value = float(value)
            return f"{value:.{decimal_places}f}{suffix}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_large_number(
        number: Union[float, int],
        decimal_places: int = 1
    ) -> str:
        """
        Format large numbers with K, M, B, T suffixes.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
        
        Returns:
            Formatted number string
        """
        if pd.isna(number) or number is None:
            return "N/A"
        
        try:
            number = float(number)
            
            if abs(number) >= 1e12:
                return f"{number/1e12:.{decimal_places}f}T"
            elif abs(number) >= 1e9:
                return f"{number/1e9:.{decimal_places}f}B"
            elif abs(number) >= 1e6:
                return f"{number/1e6:.{decimal_places}f}M"
            elif abs(number) >= 1e3:
                return f"{number/1e3:.{decimal_places}f}K"
            else:
                return f"{number:.{decimal_places}f}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_price(
        price: Union[float, int],
        decimal_places: int = 2
    ) -> str:
        """
        Format stock prices.
        
        Args:
            price: Price value
            decimal_places: Number of decimal places
        
        Returns:
            Formatted price string
        """
        if pd.isna(price) or price is None:
            return "N/A"
        
        try:
            price = float(price)
            return f"${price:.{decimal_places}f}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_change(
        current: Union[float, int],
        previous: Union[float, int],
        as_percentage: bool = True,
        decimal_places: int = 2
    ) -> tuple[str, str]:
        """
        Format price/value changes with direction indicators.
        
        Args:
            current: Current value
            previous: Previous value
            as_percentage: Whether to show as percentage
            decimal_places: Number of decimal places
        
        Returns:
            Tuple of (change_text, direction_indicator)
        """
        if pd.isna(current) or pd.isna(previous) or current is None or previous is None:
            return "N/A", "→"
        
        try:
            current = float(current)
            previous = float(previous)
            
            if previous == 0:
                return "N/A", "→"
            
            change = current - previous
            
            if as_percentage:
                change_pct = (change / previous) * 100
                change_text = f"{change_pct:+.{decimal_places}f}%"
            else:
                change_text = f"{change:+.{decimal_places}f}"
            
            # Direction indicator
            if change > 0:
                direction = "↑"
            elif change < 0:
                direction = "↓"
            else:
                direction = "→"
            
            return change_text, direction
            
        except (ValueError, TypeError):
            return "N/A", "→"
    
    @staticmethod
    def format_volume(volume: Union[float, int]) -> str:
        """
        Format trading volume.
        
        Args:
            volume: Volume value
        
        Returns:
            Formatted volume string
        """
        if pd.isna(volume) or volume is None:
            return "N/A"
        
        try:
            volume = float(volume)
            
            if volume >= 1e9:
                return f"{volume/1e9:.1f}B"
            elif volume >= 1e6:
                return f"{volume/1e6:.1f}M"
            elif volume >= 1e3:
                return f"{volume/1e3:.1f}K"
            else:
                return f"{volume:.0f}"
        except (ValueError, TypeError):
            return "N/A"

class DateTimeFormatter:
    """Formatter for date and time values."""
    
    @staticmethod
    def format_date(
        date: Union[datetime, pd.Timestamp, str],
        format_string: str = "%Y-%m-%d"
    ) -> str:
        """
        Format date values.
        
        Args:
            date: Date to format
            format_string: Python date format string
        
        Returns:
            Formatted date string
        """
        if pd.isna(date) or date is None:
            return "N/A"
        
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if isinstance(date, (datetime, pd.Timestamp)):
                return date.strftime(format_string)
            else:
                return "N/A"
        except Exception:
            return "N/A"
    
    @staticmethod
    def format_datetime(
        dt: Union[datetime, pd.Timestamp, str],
        format_string: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """
        Format datetime values.
        
        Args:
            dt: Datetime to format
            format_string: Python datetime format string
        
        Returns:
            Formatted datetime string
        """
        return DateTimeFormatter.format_date(dt, format_string)
    
    @staticmethod
    def format_time_ago(dt: Union[datetime, pd.Timestamp]) -> str:
        """
        Format datetime as "time ago" string.
        
        Args:
            dt: Datetime to format
        
        Returns:
            Time ago string (e.g., "2 hours ago")
        """
        if pd.isna(dt) or dt is None:
            return "N/A"
        
        try:
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            
            now = datetime.now()
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
            
            diff = now - dt
            
            if diff.days > 365:
                years = diff.days // 365
                return f"{years} year{'s' if years != 1 else ''} ago"
            elif diff.days > 30:
                months = diff.days // 30
                return f"{months} month{'s' if months != 1 else ''} ago"
            elif diff.days > 0:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                return "Just now"
                
        except Exception:
            return "N/A"

class TableFormatter:
    """Formatter for tabular data display."""
    
    @staticmethod
    def format_dataframe_for_display(
        df: pd.DataFrame,
        formatters: Optional[Dict[str, callable]] = None,
        precision: int = 2
    ) -> pd.DataFrame:
        """
        Format DataFrame for display with appropriate formatting.
        
        Args:
            df: DataFrame to format
            formatters: Dictionary of column -> formatter function
            precision: Default decimal precision
        
        Returns:
            Formatted DataFrame
        """
        if df.empty:
            return df
        
        display_df = df.copy()
        
        if formatters is None:
            formatters = {}
        
        for column in display_df.columns:
            if column in formatters:
                # Use custom formatter
                display_df[column] = display_df[column].apply(formatters[column])
            else:
                # Auto-detect and format based on column name and data type
                if display_df[column].dtype in ['float64', 'float32']:
                    if any(keyword in column.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                        display_df[column] = display_df[column].apply(FinancialFormatter.format_price)
                    elif any(keyword in column.lower() for keyword in ['volume']):
                        display_df[column] = display_df[column].apply(FinancialFormatter.format_volume)
                    elif any(keyword in column.lower() for keyword in ['percent', 'change', 'return', 'yield']):
                        display_df[column] = display_df[column].apply(
                            lambda x: FinancialFormatter.format_percentage(x, multiply_by_100=False)
                        )
                    elif any(keyword in column.lower() for keyword in ['ratio']):
                        display_df[column] = display_df[column].apply(FinancialFormatter.format_ratio)
                    elif any(keyword in column.lower() for keyword in ['cap', 'value', 'amount']):
                        display_df[column] = display_df[column].apply(FinancialFormatter.format_currency)
                    else:
                        # Default float formatting
                        display_df[column] = display_df[column].round(precision)
        
        return display_df
    
    @staticmethod
    def create_summary_table(
        data: Dict[str, Any],
        title: str = "Summary"
    ) -> pd.DataFrame:
        """
        Create a formatted summary table from dictionary data.
        
        Args:
            data: Dictionary of metric name -> value
            title: Table title
        
        Returns:
            Formatted DataFrame
        """
        summary_data = []
        
        for metric, value in data.items():
            # Clean up metric name
            display_name = metric.replace('_', ' ').title()
            
            # Format value based on type and name
            if isinstance(value, (int, float)):
                if any(keyword in metric.lower() for keyword in ['price', 'close', 'value']):
                    formatted_value = FinancialFormatter.format_price(value)
                elif any(keyword in metric.lower() for keyword in ['percent', 'ratio', 'yield']):
                    formatted_value = FinancialFormatter.format_percentage(value, multiply_by_100=False)
                elif any(keyword in metric.lower() for keyword in ['volume']):
                    formatted_value = FinancialFormatter.format_volume(value)
                elif any(keyword in metric.lower() for keyword in ['cap', 'market']):
                    formatted_value = FinancialFormatter.format_currency(value)
                else:
                    formatted_value = FinancialFormatter.format_ratio(value)
            else:
                formatted_value = str(value)
            
            summary_data.append({
                'Metric': display_name,
                'Value': formatted_value
            })
        
        return pd.DataFrame(summary_data)

class ColorFormatter:
    """Color and styling formatters for data visualization."""
    
    @staticmethod
    def get_change_color(value: Union[float, int]) -> str:
        """
        Get color based on positive/negative change.
        
        Args:
            value: Change value
        
        Returns:
            Color string (red, green, or gray)
        """
        if pd.isna(value) or value is None:
            return "gray"
        
        try:
            value = float(value)
            if value > 0:
                return "green"
            elif value < 0:
                return "red"
            else:
                return "gray"
        except (ValueError, TypeError):
            return "gray"
    
    @staticmethod
    def get_performance_color(value: Union[float, int], benchmark: float = 0) -> str:
        """
        Get color based on performance relative to benchmark.
        
        Args:
            value: Performance value
            benchmark: Benchmark value
        
        Returns:
            Color string
        """
        if pd.isna(value) or value is None:
            return "gray"
        
        try:
            value = float(value)
            if value > benchmark:
                return "green"
            elif value < benchmark:
                return "red"
            else:
                return "gray"
        except (ValueError, TypeError):
            return "gray"
    
    @staticmethod
    def get_risk_color(risk_level: str) -> str:
        """
        Get color based on risk level.
        
        Args:
            risk_level: Risk level string (low, medium, high)
        
        Returns:
            Color string
        """
        risk_colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red',
            'very_high': 'darkred'
        }
        
        return risk_colors.get(risk_level.lower(), 'gray')

# Create formatter instances for easy importing
financial_formatter = FinancialFormatter()
datetime_formatter = DateTimeFormatter()
table_formatter = TableFormatter()
color_formatter = ColorFormatter()
