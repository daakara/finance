"""
Financial Analyst Dashboard - Streamlined Main Application
Modular financial analysis platform with focused responsibilities
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modular dashboard components
try:
    from dashboard import DashboardManager
except ImportError as e:
    st.error(f"Dashboard module import error: {e}")
    st.stop()

class FinancialAnalystDashboard:
    """Streamlined Financial Analyst Dashboard using modular architecture."""
    
    def __init__(self):
        """Initialize the streamlined dashboard."""
        self.dashboard_manager = DashboardManager()
        
    def run(self):
        """Run the main dashboard application."""
        try:
            self.dashboard_manager.run_dashboard()
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state for debugging."""
        return self.dashboard_manager.get_dashboard_state()

def main():
    """Main application entry point."""
    try:
        dashboard = FinancialAnalystDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
