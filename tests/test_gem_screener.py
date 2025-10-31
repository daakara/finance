"""Tests for gem screening functionality"""
import unittest
from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria


class TestGemScreener(unittest.TestCase):
    """Test gem screening functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.screener = HiddenGemScreener()
        self.criteria = GemCriteria()
    
    def test_screener_initialization(self):
        """Test screener initializes correctly"""
        self.assertIsNotNone(self.screener)
        self.assertIsNotNone(self.screener.criteria)
    
    def test_custom_criteria(self):
        """Test custom criteria initialization"""
        custom = GemCriteria(
            min_market_cap=100e6,
            max_market_cap=5e9,
            min_revenue_growth=0.30
        )
        
        self.assertEqual(custom.min_market_cap, 100e6)
        self.assertEqual(custom.max_market_cap, 5e9)
        self.assertEqual(custom.min_revenue_growth, 0.30)
    
    def test_empty_universe_screening(self):
        """Test screening with empty universe"""
        results = self.screener.screen_universe([])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
    
    def test_screener_methods_exist(self):
        """Test that required methods exist"""
        self.assertTrue(hasattr(self.screener, 'screen_universe'))
        self.assertTrue(hasattr(self.screener, 'calculate_composite_score'))
        self.assertTrue(callable(self.screener.screen_universe))


if __name__ == '__main__':
    unittest.main()
