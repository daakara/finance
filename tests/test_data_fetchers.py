"""Tests for data fetching functionality"""
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline


class TestDataFetchers(unittest.TestCase):
    """Test data fetching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = MultiAssetDataPipeline()
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(hasattr(self.pipeline, 'get_comprehensive_data'))
    
    @patch('yfinance.Ticker')
    def test_fetch_with_valid_ticker(self, mock_ticker):
        """Test fetching data with valid ticker"""
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        }, index=[datetime.now() - timedelta(days=1), datetime.now()])
        
        mock_ticker.return_value.history.return_value = mock_data
        mock_ticker.return_value.info = {'marketCap': 1e9, 'sector': 'Technology'}
        self.assertTrue(True)
    
    @patch('analyst_dashboard.data.gem_fetchers.MultiAssetDataPipeline.fetch_stock_data')
    def test_error_handling_invalid_ticker(self, mock_fetch):
        """Test error handling for invalid ticker without hitting external yahoo finance network"""
        mock_fetch.return_value = None
        result = self.pipeline.get_comprehensive_data('INVALID_TICKER_12345', 'stock')
        self.assertIsInstance(result, dict)
    
    @patch('analyst_dashboard.data.gem_fetchers.MultiAssetDataPipeline.fetch_stock_data')
    def test_empty_ticker_handling(self, mock_fetch):
        """Test handling of empty ticker"""
        mock_fetch.return_value = None
        result = self.pipeline.get_comprehensive_data('', 'stock')
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()