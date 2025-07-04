import pytest
import pandas as pd
import numpy as np
from financial_ratios.valuation_model import (
    get_steady_state_value,
    get_fair_value_vs_market_price,
    get_price_to_revenue_band,
    get_price_to_eps_band,
    get_price_to_cfo_band,
    get_fcf_yield
)

# Test Data Setup
@pytest.fixture
def time_index():
    return pd.date_range(start='2020-01-01', periods=6, freq='YE')  # 6 years of yearly data

@pytest.fixture
def sample_data(time_index):
    # Create yearly data patterns
    pattern = {
        'eps': [2.0, 0, 1.8, 2.2, 2.5, 2.3],  # Annual EPS
        'wacc': [0.08, 0, 0.085, 0.09, 0.095, 0.088],  # Annual WACC
        'current_price': [50, 52, 48, 55, 58, 53],  # Year-end prices
        'net_income': [100, 120, 90, 110, 130, 115],  # Annual net income
        'total_assets': [1000, 0, 900, 950, 1100, 1050],  # Year-end assets
        'total_liabilities': [400, 420, 380, 410, 450, 430],  # Year-end liabilities
        'total_revenue': [2000, 2100, 1800, 2100, 2300, 2200],  # Annual revenue
        'shares_outstanding': [1000, 0, 1000, 1100, 1000, 1050],  # Year-end shares
        'cfo': [150, 160, 140, 155, 170, 165],  # Annual cash flow
        'fcf': [130, -140, 120, 135, 150, 145]  # Annual free cash flow
    }

    # Create Series with yearly frequency
    return {key: pd.Series(values, index=time_index) for key, values in pattern.items()}

def test_steady_state_value(sample_data):
    """Test steady state value calculation including edge cases."""
    result = get_steady_state_value(sample_data['eps'], sample_data['wacc'], sample_data['current_price'])
    
    # First period should have valid value
    assert pd.notna(result.iloc[0])
    assert isinstance(result.iloc[0], (float, np.floating))
    
    # Second period should be NaN (zero WACC)
    assert pd.isna(result.iloc[1])
    
    # Test remaining periods
    # We expect valid values for periods where we have non-zero WACC and price
    for i in range(2, len(result)):
        if sample_data['wacc'].iloc[i] != 0 and sample_data['current_price'].iloc[i] != 0:
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))

def test_fair_value_vs_market_price(sample_data):
    """Test fair value vs market price calculation including edge cases."""
    result = get_fair_value_vs_market_price(
        sample_data['net_income'],
        sample_data['total_assets'],
        sample_data['total_liabilities'],
        sample_data['eps'],
        sample_data['current_price']
    )
    
    # First 2 years should be NaN (need 3 years of data for P/E average)
    for i in range(2):
        assert pd.isna(result.iloc[i])
    
    # Year with zero assets should be NaN
    assert pd.isna(result.iloc[1])
    
    # Test remaining years
    for i in range(3, len(result)):
        if (sample_data['total_assets'].iloc[i] != 0 and 
            sample_data['eps'].iloc[i] != 0 and 
            sample_data['current_price'].iloc[i] != 0):
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))
            assert result.iloc[i] >= 0  # Should be absolute percentage

def test_price_to_revenue_band(sample_data):
    """Test price to revenue band calculation including edge cases."""
    result = get_price_to_revenue_band(
        sample_data['current_price'],
        sample_data['total_revenue'],
        sample_data['shares_outstanding']
    )
    
    # First 2 years should be NaN (need 3 years for mean/std)
    for i in range(2):
        assert pd.isna(result.iloc[i])
    
    # Year with zero shares should be NaN
    assert pd.isna(result.iloc[1])
    
    # Test remaining years
    for i in range(3, len(result)):
        if (sample_data['shares_outstanding'].iloc[i] != 0 and
            sample_data['total_revenue'].iloc[i] != 0):
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))

def test_price_to_eps_band(sample_data):
    """Test price to EPS band calculation including edge cases."""
    result = get_price_to_eps_band(
        sample_data['current_price'],
        sample_data['eps']
    )
    
    # First 2 years should be NaN (need 3 years for mean/std)
    for i in range(2):
        assert pd.isna(result.iloc[i])
    
    # Year with zero EPS should be NaN
    assert pd.isna(result.iloc[1])
    
    # Test remaining years
    for i in range(3, len(result)):
        if sample_data['eps'].iloc[i] != 0:
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))

def test_price_to_cfo_band(sample_data):
    """Test price to CFO band calculation including edge cases."""
    result = get_price_to_cfo_band(
        sample_data['current_price'],
        sample_data['cfo'],
        sample_data['shares_outstanding']
    )
    
    # First 2 years should be NaN (need 3 years for mean/std)
    for i in range(2):
        assert pd.isna(result.iloc[i])
    
    # Year with zero shares should be NaN
    assert pd.isna(result.iloc[1])
    
    # Test remaining years
    for i in range(3, len(result)):
        if (sample_data['shares_outstanding'].iloc[i] != 0 and
            sample_data['cfo'].iloc[i] != 0):
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))

def test_fcf_yield(sample_data):
    """Test FCF yield calculation including edge cases."""
    result = get_fcf_yield(
        sample_data['fcf'],
        sample_data['current_price'],
        sample_data['shares_outstanding']
    )
    
    # First year should have valid yield
    assert pd.notna(result.iloc[0])
    assert isinstance(result.iloc[0], (float, np.floating))
    assert result.iloc[0] >= 0  # Should be absolute percentage
    
    # Second year should be NaN (zero shares)
    assert pd.isna(result.iloc[1])
    
    # Test remaining years
    for i in range(2, len(result)):
        if (sample_data['shares_outstanding'].iloc[i] != 0 and
            sample_data['current_price'].iloc[i] != 0):
            assert pd.notna(result.iloc[i])
            assert isinstance(result.iloc[i], (float, np.floating))
            assert result.iloc[i] >= 0  # Should be absolute percentage
