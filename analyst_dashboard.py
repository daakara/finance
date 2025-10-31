"""
Financial Analyst Dashboard - Streamlined Entry Point
Modern modular dashboard with specialized analysis components.
"""

import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        # Import modular components
        from analyst_dashboard.core.dashboard_manager import AnalystDashboardManager
        
        # Initialize and run dashboard
        dashboard = AnalystDashboardManager()
        dashboard.run_dashboard()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
