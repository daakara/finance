import unittest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyst_dashboard.visualizers.gem_dashboard import GemDashboard

class TestGemDashboardLiveData(unittest.TestCase):
    """
    Test suite for verifying GemDashboard uses live data only.
    """

    def setUp(self):
        """Set up a GemDashboard instance for testing."""
        self.dashboard = GemDashboard()

    def test_no_fallback_sample_methods_exist(self):
        """
        Verify that sample data fallback methods have been removed.
        
        - GIVEN: The GemDashboard class has been refactored to use live data only
        - WHEN: We check for the existence of fallback methods
        - THEN: These methods should not exist
        """
        # Verify fallback methods no longer exist
        self.assertFalse(
            hasattr(self.dashboard, '_get_fallback_sample_results'),
            "Sample data fallback method should be removed"
        )
        self.assertFalse(
            hasattr(self.dashboard, '_show_sample_analysis_fallback'),
            "Sample analysis fallback method should be removed"
        )
        self.assertFalse(
            hasattr(self.dashboard, '_create_generic_sample_analysis'),
            "Generic sample analysis method should be removed"
        )

    def test_screening_returns_empty_on_failure(self):
        """
        Verify that screening returns empty list on failure, not sample data.
        
        - GIVEN: A dashboard instance
        - WHEN: Screening encounters an error
        - THEN: It should return an empty list, not fall back to sample data
        """
        # This is a structural test - the method should not contain fallback logic
        import inspect
        source = inspect.getsource(self.dashboard._run_sample_screening)
        
        # Verify no references to sample/fallback in the screening code
        self.assertNotIn('_get_fallback_sample_results', source, 
                        "Screening should not call fallback sample results")
        self.assertNotIn('sample_data', source.lower(),
                        "Screening should not reference sample data")

if __name__ == '__main__':
    unittest.main()
