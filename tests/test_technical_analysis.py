"""Tests for technical analysis functionality"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestTechnicalAnalysis(unittest.TestCase):
    """Test technical analysis calculations"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(100, 120, 100),
            'Low': np.random.uniform(80, 100, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_data_validation(self):
        """Test that test data is valid"""
        self.assertFalse(self.test_data.empty)
        self.assertEqual(len(self.test_data), 100)
        self.assertTrue(all(col in self.test_data.columns 
                          for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
    
    def test_moving_average_calculation(self):
        """Test moving average calculation"""
        ma_20 = self.test_data['Close'].rolling(20).mean()
        
        self.assertEqual(len(ma_20), len(self.test_data))
        self.assertTrue(pd.isna(ma_20.iloc[0]))  # First values should be NaN
        self.assertFalse(pd.isna(ma_20.iloc[-1]))  # Last value should be valid
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        returns = self.test_data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)


if __name__ == '__main__':
    unittest.main()
