"""
Hidden Gems Dashboard - Standalone Entry Point
Run this to access the dedicated Hidden Gems Scanner interface
"""

import streamlit as st
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analyst_dashboard.visualizers.gem_dashboard import run_gem_dashboard

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Hidden Gems Scanner",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run the dashboard
    run_gem_dashboard()
