import pytest
import pandas as pd
import numpy as np
from financial_ratios.financial_health_model import (
    get_debt_to_equity_ratio,
    get_interest_coverage_ratio,
    get_current_ratio,
    get_cash_conversion_cycle,
    get_altman_z_score
)

# Test Data Setup
@pytest.fixture
def time_index():
    return pd.date_range(start='2020-01-01', periods=4, freq='QE')

@pytest.fixture
def sample_data(time_index):
    return {
        'total_debt': pd.Series([1000, 1200, 800, 900], index=time_index),
        'total_equity': pd.Series([2000, 0, 1500, -100], index=time_index),
        'ebitda': pd.Series([500, 600, 400, 450], index=time_index),
        'interest_expense': pd.Series([100, 0, 80, 90], index=time_index),
        'current_assets': pd.Series([800, 900, 700, 750], index=time_index),
        'current_liabilities': pd.Series([400, 0, 350, 375], index=time_index),
        'inventory': pd.Series([300, 350, 250, 275], index=time_index),
        'cogs': pd.Series([600, 0, 500, 525], index=time_index),
        'accounts_receivable': pd.Series([200, 225, 175, 185], index=time_index),
        'revenue': pd.Series([1000, 0, 800, 850], index=time_index),
        'accounts_payable': pd.Series([150, 175, 125, 135], index=time_index),
        'total_assets': pd.Series([3000, 0, 2500, 2600], index=time_index),
        'ebit': pd.Series([400, 450, 350, 375], index=time_index),
        'diluted_shares_outstanding': pd.Series([100, 100, 100, 100], index=time_index),
        'retained_earnings': pd.Series([800, 900, 700, 750], index=time_index),
        'stock_price': pd.Series([20, 22, 18, 19], index=time_index),
        'total_liabilities': pd.Series([400, 450, 350, 375], index=time_index)
    }

def test_debt_to_equity_ratio(sample_data):
    """Test debt to equity ratio calculation including edge cases."""
    result = get_debt_to_equity_ratio(
        sample_data['total_debt'],
        sample_data['total_equity']
    )
    
    # Normal case
    assert result.iloc[0] == pytest.approx(0.5)  # 1000/2000
    
    # Zero equity - should return NaN
    assert pd.isna(result.iloc[1])
    
    # Normal case
    assert result.iloc[2] == pytest.approx(0.533333, rel=1e-5)  # 800/1500
    
    # Negative equity - should still compute but indicates financial distress
    assert result.iloc[3] == pytest.approx(-9.0)  # 900/-100

def test_interest_coverage_ratio(sample_data):
    """Test interest coverage ratio calculation including edge cases."""
    result = get_interest_coverage_ratio(
        sample_data['ebitda'],
        sample_data['interest_expense']
    )
    
    # Normal case
    assert result.iloc[0] == pytest.approx(5.0)  # 500/100
    
    # Zero interest expense - should return NaN
    assert pd.isna(result.iloc[1])
    
    # Normal cases
    assert result.iloc[2] == pytest.approx(5.0)  # 400/80
    assert result.iloc[3] == pytest.approx(5.0)  # 450/90

def test_current_ratio(sample_data):
    """Test current ratio calculation including edge cases."""
    result = get_current_ratio(
        sample_data['current_assets'],
        sample_data['current_liabilities']
    )
    
    # Normal case
    assert result.iloc[0] == pytest.approx(2.0)  # 800/400
    
    # Zero current liabilities - should return NaN
    assert pd.isna(result.iloc[1])
    
    # Normal cases
    assert result.iloc[2] == pytest.approx(2.0)  # 700/350
    assert result.iloc[3] == pytest.approx(2.0)  # 750/375

def test_cash_conversion_cycle(sample_data):
    """Test cash conversion cycle calculation including edge cases."""
    result = get_cash_conversion_cycle(
        sample_data['inventory'],
        sample_data['cogs'],
        sample_data['accounts_receivable'],
        sample_data['revenue'],
        sample_data['accounts_payable']
    )
    
    # Normal case for first period
    expected_dio = (300 / 600) * 365  # days inventory outstanding
    expected_dso = (200 / 1000) * 365  # days sales outstanding
    expected_dpo = (150 / 600) * 365  # days payables outstanding
    expected_ccc = expected_dio + expected_dso - expected_dpo
    assert result.iloc[0] == pytest.approx(expected_ccc)
    
    # Zero denominators - should return NaN
    assert pd.isna(result.iloc[1])
    
    # Normal cases for remaining periods
    expected_dio = (250 / 500) * 365
    expected_dso = (175 / 800) * 365
    expected_dpo = (125 / 500) * 365
    expected_ccc = expected_dio + expected_dso - expected_dpo
    assert result.iloc[2] == pytest.approx(expected_ccc)

def test_altman_z_score(sample_data):
    """Test Altman Z-Score calculation including edge cases."""
    result = get_altman_z_score(
        sample_data['current_assets'],
        sample_data['current_liabilities'],
        sample_data['total_assets'],
        sample_data['ebit'],
        sample_data['diluted_shares_outstanding'],
        sample_data['revenue'],
        sample_data['total_liabilities'],
        sample_data['retained_earnings'],
        sample_data['stock_price']
    )
    
    # Normal case for first period
    working_capital = 800 - 400
    x1 = 1.2 * (working_capital / 3000)
    x2 = 1.4 * (800 / 3000)
    x3 = 3.3 * (400 / 3000)
    x4 = 0.6 * ((20 * 100) / 400)
    x5 = 1.0 * (1000 / 3000)
    expected_z = x1 + x2 + x3 + x4 + x5
    assert result.iloc[0] == pytest.approx(expected_z)
    
    # Zero denominators - should return NaN
    assert pd.isna(result.iloc[1])
    
    # Test remaining periods
    working_capital = 700 - 350
    x1 = 1.2 * (working_capital / 2500)
    x2 = 1.4 * (700 / 2500)
    x3 = 3.3 * (350 / 2500)
    x4 = 0.6 * ((18 * 100) / 350)
    x5 = 1.0 * (800 / 2500)
    expected_z = x1 + x2 + x3 + x4 + x5
    assert result.iloc[2] == pytest.approx(expected_z)