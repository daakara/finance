"""
Financial Analyst Dashboard - Streamlined Entry Point
Modern modular dashboard with specialized analysis components.
"""

import streamlit as st
import logging

# Configure page
st.set_page_config(
    page_title="Financial Analyst Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
from analyst_dashboard.core.dashboard_manager import AnalystDashboardManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    try:
        dashboard_manager = AnalystDashboardManager()
        dashboard_manager.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
