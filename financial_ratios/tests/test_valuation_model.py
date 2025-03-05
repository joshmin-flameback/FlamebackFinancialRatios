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
    return pd.date_range(start='2020-01-01', periods=48, freq='M')  # 4 years of monthly data

@pytest.fixture
def sample_data(time_index):
    return {
        'eps': pd.Series([2.0, 0, 1.8, 2.2, 2.5] * 10, index=time_index[:50]),
        'wacc': pd.Series([0.08, 0, 0.085, 0.09, 0.095] * 10, index=time_index[:50]),
        'current_price': pd.Series([50, 52, 48, 55, 58] * 10, index=time_index[:50]),
        'net_income': pd.Series([100, 120, 90, 110, 130] * 10, index=time_index[:50]),
        'total_assets': pd.Series([1000, 0, 900, 950, 1100] * 10, index=time_index[:50]),
        'total_liabilities': pd.Series([400, 420, 380, 410, 450] * 10, index=time_index[:50]),
        'total_revenue': pd.Series([2000, 2100, 1800, 2100, 2300] * 10, index=time_index[:50]),
        'shares_outstanding': pd.Series([1000, 0, 1000, 1100, 1000] * 10, index=time_index[:50]),
        'cfo': pd.Series([150, 160, 140, 155, 170] * 10, index=time_index[:50]),
        'fcf': pd.Series([130, -140, 120, 135, 150] * 10, index=time_index[:50])
    }

def test_steady_state_value(sample_data):
    """Test steady state value calculation including edge cases."""
    result = get_steady_state_value(
        sample_data['eps'],
        sample_data['wacc'],
        sample_data['current_price']
    )
    
    # First period should have valid value
    assert pd.notna(result[0])
    assert isinstance(result[0], (float, np.floating))
    
    # Second period should be NaN (zero WACC)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(2, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_fair_value_vs_market_price(sample_data):
    """Test fair value vs market price calculation including edge cases."""
    result = get_fair_value_vs_market_price(
        sample_data['net_income'],
        sample_data['total_assets'],
        sample_data['total_liabilities'],
        sample_data['eps'],
        sample_data['current_price']
    )
    
    # First period should have valid value
    assert pd.notna(result[0])
    assert isinstance(result[0], (float, np.floating))
    assert result[0] >= 0  # Should be absolute percentage
    
    # Second period should be NaN (zero assets)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(2, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))
        assert result[i] >= 0  # Should be absolute percentage

def test_price_to_revenue_band(sample_data):
    """Test price to revenue band calculation including edge cases."""
    result = get_price_to_revenue_band(
        sample_data['current_price'],
        sample_data['total_revenue'],
        sample_data['shares_outstanding']
    )
    
    # First 11 periods should be NaN (insufficient data)
    for i in range(11):
        assert pd.isna(result[i])
    
    # Second period should be NaN (zero shares)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(12, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_price_to_eps_band(sample_data):
    """Test price to EPS band calculation including edge cases."""
    result = get_price_to_eps_band(
        sample_data['current_price'],
        sample_data['eps']
    )
    
    # First 11 periods should be NaN (insufficient data)
    for i in range(11):
        assert pd.isna(result[i])
    
    # Second period should be NaN (zero EPS)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(12, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_price_to_cfo_band(sample_data):
    """Test price to CFO band calculation including edge cases."""
    result = get_price_to_cfo_band(
        sample_data['current_price'],
        sample_data['cfo'],
        sample_data['shares_outstanding']
    )
    
    # First 11 periods should be NaN (insufficient data)
    for i in range(11):
        assert pd.isna(result[i])
    
    # Second period should be NaN (zero shares)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(12, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_fcf_yield(sample_data):
    """Test FCF yield calculation including edge cases."""
    result = get_fcf_yield(
        sample_data['fcf'],
        sample_data['current_price'],
        sample_data['shares_outstanding']
    )
    
    # First period should have valid yield
    assert pd.notna(result[0])
    assert isinstance(result[0], (float, np.floating))
    assert result[0] >= 0  # Should be absolute percentage
    
    # Second period should be NaN (zero shares)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(2, len(result)):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))
        assert result[i] >= 0  # Should be absolute percentage

# Error Cases
def test_insufficient_data_errors():
    """Test error handling for insufficient data."""
    short_index = pd.date_range(start='2020-01-01', periods=11, freq='M')
    short_data = pd.Series(range(11), index=short_index)
    
    # Test price to revenue band
    with pytest.raises(ValueError, match="Insufficient data"):
        get_price_to_revenue_band(short_data, short_data, short_data)
    
    # Test price to EPS band
    with pytest.raises(ValueError, match="Insufficient data"):
        get_price_to_eps_band(short_data, short_data)
    
    # Test price to CFO band
    with pytest.raises(ValueError, match="Insufficient data"):
        get_price_to_cfo_band(short_data, short_data, short_data)

# Edge Cases
def test_all_zero_values(time_index):
    """Test handling of all zero values."""
    zero_series = pd.Series(0, index=time_index)
    
    # Test steady state value
    result = get_steady_state_value(zero_series, zero_series, zero_series)
    assert all(pd.isna(result))
    
    # Test fair value vs market price
    result = get_fair_value_vs_market_price(zero_series, zero_series, zero_series, zero_series, zero_series)
    assert all(pd.isna(result))
    
    # Test FCF yield
    result = get_fcf_yield(zero_series, zero_series, zero_series)
    assert all(pd.isna(result))

def test_negative_values(time_index):
    """Test handling of negative values."""
    negative_series = pd.Series(-1, index=time_index)
    positive_series = pd.Series(1, index=time_index)
    
    # Test fair value vs market price with negative equity
    result = get_fair_value_vs_market_price(
        negative_series,  # net_income
        positive_series,  # total_assets
        positive_series * 2,  # total_liabilities > assets
        positive_series,  # eps
        positive_series   # current_price
    )
    assert all(pd.notna(result))  # Should handle negative equity gracefully
    
    # Test FCF yield with negative FCF
    result = get_fcf_yield(negative_series, positive_series, positive_series)
    assert all(result >= 0)  # Should return absolute percentage