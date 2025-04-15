import pytest
import pandas as pd
import numpy as np
from financial_ratios.quality_model import (
    get_intrinsic_compounding_rate,
    get_dips_in_profit_over_10yrs,
    get_roic_band,
    get_cfo_band,
    get_negative_dips_in_fcf_over_10yrs,
    get_negative_fcf_years,
    get_fcf_to_net_profit_band
)

# Test Data Setup
@pytest.fixture
def time_index():
    return pd.date_range(start='2020-01-01', periods=12, freq='YE')

@pytest.fixture
def sample_data(time_index):
    return {
        'net_income': pd.Series([100, 120, 90, 110, 130, 140, 135, 145, 150, 160, 155, 165], index=time_index),
        'total_assets': pd.Series([1000, 0, 900, 950, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450], index=time_index),
        'total_liabilities': pd.Series([400, 420, 380, 410, 450, 460, 470, 480, 490, 500, 510, 520], index=time_index),
        'dividend_paid': pd.Series([20, 25, 18, 22, 26, 28, 27, 29, 30, 32, 31, 33], index=time_index),
        'revenue': pd.Series([2000, 2100, 1800, 2100, 2300, 2400, 2350, 2450, 2500, 2600, 2550, 2650], index=time_index),
        'total_expense': pd.Series([1800, 1900, 1700, 1900, 2100, 2200, 2150, 2250, 2300, 2400, 2350, 2450], index=time_index),
        'invested_capital': pd.Series([800, 0, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150], index=time_index),
        'nopat': pd.Series([80, 90, 70, 85, 95, 100, 98, 105, 110, 115, 112, 118], index=time_index),
        'cfo': pd.Series([150, 160, 140, 155, 170, 175, 172, 180, 185, 190, 188, 195], index=time_index),
        'fcf': pd.Series([130, -140, 120, 135, 150, 155, 152, 160, 165, 170, 168, 175], index=time_index),
        'net_profit': pd.Series([100, 0, 90, 110, 130, 140, 135, 145, 150, 160, 155, 165], index=time_index)
    }

def test_intrinsic_compounding_rate(sample_data):
    """Test intrinsic compounding rate calculation including edge cases."""
    result = get_intrinsic_compounding_rate(
        sample_data['net_income'],
        sample_data['total_assets'],
        sample_data['total_liabilities'],
        sample_data['dividend_paid']
    )
    
    # First period should have valid rate
    assert pd.notna(result[0])
    assert isinstance(result[0], (float, np.floating))
    assert result[0] >= 0  # Rate should be positive
    
    # Second period should be NaN (zero assets)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(2, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))
        assert result[i] >= 0  # Rate should be positive

def test_dips_in_profit_over_10yrs(sample_data):
    """Test profit dips calculation including edge cases."""
    result = get_dips_in_profit_over_10yrs(
        sample_data['revenue'],
        sample_data['total_expense']
    )
    
    # First 9 periods should be NaN (insufficient data)
    for i in range(9):
        assert pd.isna(result[i])
    
    # Test remaining periods
    for i in range(9, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (int, float, np.integer))
        assert result[i] >= 0  # Count should be non-negative

def test_roic_band(sample_data):
    """Test ROIC band calculation including edge cases."""
    result = get_roic_band(
        sample_data['invested_capital'],
        sample_data['nopat']
    )
    
    # First 4 periods should be NaN (insufficient data)
    for i in range(4):
        assert pd.isna(result[i])
    
    # Second period should be NaN (zero invested capital)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(5, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_cfo_band(sample_data):
    """Test CFO band calculation including edge cases."""
    result = get_cfo_band(sample_data['cfo'])
    
    # First 4 periods should be NaN (insufficient data)
    for i in range(4):
        assert pd.isna(result[i])
    
    # Test remaining periods
    for i in range(5, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))

def test_negative_dips_in_fcf_over_10yrs(sample_data):
    """Test negative FCF dips calculation including edge cases."""
    result = get_negative_dips_in_fcf_over_10yrs(sample_data['fcf'])
    
    # First 9 periods should be NaN (insufficient data)
    for i in range(9):
        assert pd.isna(result[i])
    
    # Test remaining periods
    for i in range(10, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (int, float, np.integer))
        assert result[i] >= 0  # Count should be non-negative

def test_negative_fcf_years(sample_data):
    """Test negative FCF years calculation including edge cases."""
    result = get_negative_fcf_years(sample_data['fcf'])
    
    # First 9 periods should be NaN (insufficient data)
    for i in range(9):
        assert pd.isna(result[i])
    
    # Test remaining periods
    for i in range(10, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (int, float, np.integer))
        assert result[i] >= 0  # Count should be non-negative

def test_fcf_to_net_profit_band(sample_data):
    """Test FCF to net profit band calculation including edge cases."""
    result = get_fcf_to_net_profit_band(
        sample_data['fcf'],
        sample_data['net_profit']
    )
    
    # First 4 periods should be NaN (insufficient data)
    for i in range(4):
        assert pd.isna(result[i])
    
    # Second period should be NaN (zero net profit)
    assert pd.isna(result[1])
    
    # Test remaining periods
    for i in range(5, 12):
        assert pd.notna(result[i])
        assert isinstance(result[i], (float, np.floating))