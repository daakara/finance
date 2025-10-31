"""
Test script to verify all modular components work correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Core modules
        from config import config
        print("✅ Config imported")
        
        from visualizer import financial_visualizer
        print("✅ Visualizer imported")
        
        from analysis_engine import (
            technical_engine, risk_engine, macro_engine, 
            fundamental_engine, portfolio_engine, forecasting_engine
        )
        print("✅ Analysis engines imported")
        
        # Modular components
        from ui_components import (
            SidebarManager, MetricsDisplayManager, ErrorHandler
        )
        print("✅ UI components imported")
        
        from analysis_renderers import (
            MacroeconomicRenderer, FundamentalRenderer
        )
        print("✅ Analysis renderers imported")
        
        from analysis_components import (
            DataProcessors, TechnicalIndicators, RiskMetrics
        )
        print("✅ Analysis components imported")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_component_functionality():
    """Test basic functionality of components."""
    print("\nTesting component functionality...")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        price_data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'High': 101 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Low': 99 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        print("✅ Sample data created")
        
        # Test DataProcessors
        from analysis_components import DataProcessors
        returns = DataProcessors.calculate_returns(price_data)
        print(f"✅ Returns calculated: {len(returns)} data points")
        
        # Test TechnicalIndicators
        from analysis_components import TechnicalIndicators
        rsi = TechnicalIndicators.calculate_rsi(price_data['Close'])
        print(f"✅ RSI calculated: {len(rsi.dropna())} data points")
        
        # Test RiskMetrics
        from analysis_components import RiskMetrics
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
        print(f"✅ Sharpe ratio calculated: {sharpe:.3f}")
        
        # Test technical analysis engine
        from analysis_engine import technical_engine
        tech_analysis = technical_engine.analyze(price_data)
        print(f"✅ Technical analysis completed: {len(tech_analysis)} components")
        
        # Test visualizer
        from visualizer import financial_visualizer
        fig = financial_visualizer.create_volatility_chart(returns)
        print("✅ Volatility chart created")
        
        print("🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("FINANCIAL ANALYSIS PLATFORM - MODULE TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_component_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED - MODULARIZATION SUCCESSFUL!")
            print("✅ Import structure is working correctly")
            print("✅ Modular components are functional")
            print("✅ Analysis engines are operational")
            print("✅ Platform is ready for use")
            print("=" * 50)
        else:
            print("\n❌ Functionality tests failed")
    else:
        print("\n❌ Import tests failed")

if __name__ == "__main__":
    main()
