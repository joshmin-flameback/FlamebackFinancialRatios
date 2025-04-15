import pandas as pd
import pytest

from financial_ratios.earnings_model import (
    get_piotroski_score,
    get_revenue_growth,
    get_eps_growth,
    get_revenue_consecutive_growth,
    get_eps_consecutive_growth,
    get_average_revenue_growth,
    get_average_gross_margin,
    get_average_gross_margin_growth,
    get_average_ebitda,
    get_average_ebitda_growth,
    get_average_eps_growth,
    get_revenue_growth_vs_average_growth,
    get_eps_growth_vs_average_growth,
    get_ebitda_growth_vs_average_growth,
    get_gross_margin_growth_vs_average_growth,
    get_return_on_equity,
    get_roe_vs_average_roe,
    get_return_on_assets,
    get_roa_vs_average_roa,
    get_revenue_vs_estimate,
    get_shares_outstanding_vs_estimate,
    get_free_cash_flow_growth,
    get_free_cash_flow_average_growth
)


# Test Data Setup
@pytest.fixture
def time_index():
    return pd.date_range(start='2020-01-01', periods=4, freq='YE')


@pytest.fixture
def sample_data(time_index):
    return {
        'net_income': pd.Series([100, 0, 90, 110], index=time_index),
        'total_assets': pd.Series([1000, 0, 900, 950], index=time_index),
        'cash_flow_from_operations': pd.Series([150, 160, 140, 155], index=time_index),
        'current_assets': pd.Series([800, 900, 700, 750], index=time_index),
        'current_liabilities': pd.Series([400, 0, 350, 375], index=time_index),
        'long_term_debt': pd.Series([500, 550, 450, 475], index=time_index),
        'shares_outstanding': pd.Series([100, 0, 100, 110], index=time_index),
        'revenue': pd.Series([2000, 0, 1800, 2100], index=time_index),
        'cogs': pd.Series([1200, 0, 1100, 1250], index=time_index),
        'eps': pd.Series([2.0, 0, 1.8, 2.2], index=time_index),
        'ebitda': pd.Series([350, 0, 320, 380], index=time_index),
        'gross_margin': pd.Series([0.4, 0, 0.39, 0.41], index=time_index),
        'shareholders_equity': pd.Series([600, 0, 550, 575], index=time_index),
        'revenue_estimate': pd.Series([1900, 0, 1750, 2000], index=time_index),
        'eps_estimate': pd.Series([1.9, 0, 1.7, 2.1], index=time_index),
        'net_income_estimate': pd.Series([95, 0, 85, 105], index=time_index),
        'free_cash_flow': pd.Series([120, 0, 110, 130], index=time_index)
    }


def test_piotroski_score(sample_data):
    """Test Piotroski F-Score calculation including all 9 criteria and edge cases."""
    result = get_piotroski_score(
        sample_data['net_income'],
        sample_data['total_assets'],
        sample_data['cash_flow_from_operations'],
        sample_data['current_assets'],
        sample_data['current_liabilities'],
        sample_data['long_term_debt'],
        sample_data['shares_outstanding'],
        sample_data['revenue'],
        sample_data['cogs']
    )

    # Test first period - should calculate all criteria except changes
    # 1. ROA > 0 (100/1000 = 0.1)
    # 2. Operating Cash Flow > 0 (150 > 0)
    # 3. ROA Change (N/A for first period)
    # 4. Cash Flow vs Net Income (150 > 100)
    # 5. Long-term Debt Change (N/A for first period)
    # 6. Current Ratio Change (N/A for first period)
    # 7. Shares Outstanding Change (N/A for first period)
    # 8. Gross Margin Change (N/A for first period)
    # 9. Asset Turnover Change (N/A for first period)
    assert result[0] >= 3  # Should get points for criteria 1, 2, and 4

    # Test zero values period - should return NaN
    assert pd.isna(result[1])

    # Test third period - should calculate all criteria
    # 1. ROA > 0 (90/900 = 0.1)
    # 2. Operating Cash Flow > 0 (140 > 0)
    # 3. ROA Change (0.1 vs 0 from prev period)
    # 4. Cash Flow vs Net Income (140 > 90)
    # 5. Long-term Debt Change (450 < 550)
    # 6. Current Ratio Change (700/350 vs 900/0)
    # 7. Shares Outstanding Change (100 <= 0)
    # 8. Gross Margin Change (0.39 vs 0)
    # 9. Asset Turnover Change (1800/900 vs 0/0)
    assert result[2] >= 6  # Should get points for at least 6 criteria

    # Test fourth period - should calculate all criteria
    # 1. ROA > 0 (110/950 > 0)
    # 2. Operating Cash Flow > 0 (155 > 0)
    # 3. ROA Change (110/950 vs 90/900)
    # 4. Cash Flow vs Net Income (155 > 110)
    # 5. Long-term Debt Change (475 > 450, no point)
    # 6. Current Ratio Change (750/375 vs 700/350)
    # 7. Shares Outstanding Change (110 > 100, no point)
    # 8. Gross Margin Change (0.41 > 0.39)
    # 9. Asset Turnover Change (2100/950 vs 1800/900)
    assert result[3] >= 6  # Should get points for at least 6 criteria

    # Test that all scores are within valid range
    valid_scores = result.dropna()
    assert all(0 <= score <= 9 for score in valid_scores)

    # Test that scores are integers
    assert all(float(score).is_integer() for score in valid_scores)


def test_revenue_growth(sample_data):
    """Test revenue growth calculation including edge cases."""
    result = get_revenue_growth(sample_data['revenue'])

    # First period - no prior data
    assert pd.isna(result[0])

    # Zero denominator period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - normal calculation
    expected_growth = (2100 - 1800) / 1800  # = 0.1667
    assert result[3] == pytest.approx(expected_growth)


def test_eps_growth(sample_data):
    """Test EPS growth calculation including edge cases."""
    result = get_eps_growth(sample_data['eps'])

    # First period - no prior data
    assert pd.isna(result[0])

    # Zero denominator period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - normal calculation
    expected_growth = (2.2 - 1.8) / 1.8  # = 0.2222
    assert result[3] == pytest.approx(expected_growth)


def test_revenue_consecutive_growth(sample_data):
    """Test revenue consecutive growth calculation including edge cases."""
    result = get_revenue_consecutive_growth(sample_data['revenue'])

    # First period - no growth history
    assert result[0] == 0

    # Zero value breaks consecutive growth
    assert result[1] == 0

    # Third period - growth from zero
    assert result[2] == 1  # First growth period

    # Fourth period - continued growth
    assert result[3] == 2  # Second consecutive growth


def test_eps_consecutive_growth(sample_data):
    """Test EPS consecutive growth calculation including edge cases."""
    result = get_eps_consecutive_growth(sample_data['eps'])

    # First period - no growth history
    assert result[0] == 0

    # Zero value breaks consecutive growth
    assert result[1] == 0

    # Third period - growth from zero
    assert result[2] == 1  # First growth period

    # Fourth period - continued growth
    assert result[3] == 2  # Second consecutive growth


def test_average_revenue_growth(sample_data):
    """Test average revenue growth calculation including edge cases."""
    result = get_average_revenue_growth(sample_data['revenue'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - valid growth
    expected_growth = (2100 - 1800) / 1800  # Only valid growth rate
    assert result[3] == pytest.approx(expected_growth)


def test_average_gross_margin(sample_data):
    """Test average gross margin calculation including edge cases."""
    result = get_average_gross_margin(sample_data['gross_margin'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - can calculate average
    expected_avg = 0.39  # Only using valid values
    assert result[2] == pytest.approx(expected_avg)

    # Fourth period - updated average
    expected_avg = (0.39 + 0.41) / 2
    assert result[3] == pytest.approx(expected_avg)


def test_average_gross_margin_growth(sample_data):
    """Test average gross margin growth calculation including edge cases."""
    result = get_average_gross_margin_growth(sample_data['gross_margin'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - valid growth
    expected_growth = (0.41 - 0.39) / 0.39  # Only valid growth rate
    assert result[3] == pytest.approx(expected_growth)


def test_average_ebitda(sample_data):
    """Test average EBITDA calculation including edge cases."""
    result = get_average_ebitda(sample_data['ebitda'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - can calculate average
    expected_avg = 320  # Only using valid values
    assert result[2] == pytest.approx(expected_avg)

    # Fourth period - updated average
    expected_avg = (320 + 380) / 2
    assert result[3] == pytest.approx(expected_avg)


def test_average_ebitda_growth(sample_data):
    """Test average EBITDA growth calculation including edge cases."""
    result = get_average_ebitda_growth(sample_data['ebitda'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - valid growth
    expected_growth = (380 - 320) / 320  # Only valid growth rate
    assert result[3] == pytest.approx(expected_growth)


def test_average_eps_growth(sample_data):
    """Test average EPS growth calculation including edge cases."""
    result = get_average_eps_growth(sample_data['eps'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - valid growth
    expected_growth = (2.2 - 1.8) / 1.8  # Only valid growth rate
    assert result[3] == pytest.approx(expected_growth)


def test_revenue_growth_vs_average_growth(sample_data):
    """Test revenue growth vs average growth calculation including edge cases."""
    result = get_revenue_growth_vs_average_growth(sample_data['revenue'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # NaN due to zero division

    # Fourth period - valid comparison
    # Should be ~1 as it's the only growth rate to compare against
    assert result[3] == pytest.approx(1.0)


def test_eps_growth_vs_average_growth(sample_data):
    """Test EPS growth vs average growth calculation including edge cases."""
    result = get_eps_growth_vs_average_growth(sample_data['eps'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # NaN due to zero division

    # Fourth period - valid comparison
    # Should be ~1 as it's the only growth rate to compare against
    assert result[3] == pytest.approx(1.0)


def test_ebitda_growth_vs_average_growth(sample_data):
    """Test EBITDA growth vs average growth calculation including edge cases."""
    result = get_ebitda_growth_vs_average_growth(sample_data['ebitda'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # NaN due to zero division

    # Fourth period - valid comparison
    # Should be ~1 as it's the only growth rate to compare against
    assert result[3] == pytest.approx(1.0)


def test_gross_margin_growth_vs_average_growth(sample_data):
    """Test gross margin growth vs average growth calculation including edge cases."""
    result = get_gross_margin_growth_vs_average_growth(sample_data['gross_margin'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # NaN due to zero division

    # Fourth period - valid comparison
    # Should be ~1 as it's the only growth rate to compare against
    assert result[3] == pytest.approx(1.0)


def test_return_on_equity(sample_data):
    """Test return on equity calculation including edge cases."""
    result = get_return_on_equity(sample_data['net_income'], sample_data['shareholders_equity'])

    # First period - normal calculation
    expected_roe = 100 / 600  # = 0.1667
    assert result[0] == pytest.approx(expected_roe)

    # Zero equity period
    assert pd.isna(result[1])

    # Third period - normal calculation
    expected_roe = 90 / 550  # = 0.1636
    assert result[2] == pytest.approx(expected_roe)

    # Fourth period - normal calculation
    expected_roe = 110 / 575  # ≈ 0.1913
    assert result[3] == pytest.approx(expected_roe)


def test_roe_vs_average_roe(sample_data):
    """Test ROE vs average ROE calculation including edge cases."""
    result = get_roe_vs_average_roe(sample_data['net_income'], sample_data['shareholders_equity'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero equity period
    assert pd.isna(result[1])

    # Third period - can calculate average
    assert result[2] == pytest.approx(1.0)  # Only one valid ROE to compare

    # Fourth period - valid comparison
    expected_ratio = (110 / 575) / ((100 / 600 + 90 / 550) / 2)
    assert result[3] == pytest.approx(expected_ratio)


def test_return_on_assets(sample_data):
    """Test return on assets calculation including edge cases."""
    result = get_return_on_assets(sample_data['net_income'], sample_data['total_assets'])

    # First period - normal calculation
    expected_roa = 100 / 1000  # = 0.1
    assert result[0] == pytest.approx(expected_roa)

    # Zero assets period
    assert pd.isna(result[1])

    # Third period - normal calculation
    expected_roa = 90 / 900  # = 0.1
    assert result[2] == pytest.approx(expected_roa)

    # Fourth period - normal calculation
    expected_roa = 110 / 950  # ≈ 0.1158
    assert result[3] == pytest.approx(expected_roa)


def test_roa_vs_average_roa(sample_data):
    """Test ROA vs average ROA calculation including edge cases."""
    result = get_roa_vs_average_roa(sample_data['net_income'], sample_data['total_assets'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero assets period
    assert pd.isna(result[1])

    # Third period - can calculate average
    assert result[2] == pytest.approx(1.0)  # Only one valid ROA to compare

    # Fourth period - valid comparison
    expected_ratio = (110 / 950) / ((100 / 1000 + 90 / 900) / 2)
    assert result[3] == pytest.approx(expected_ratio)


def test_revenue_vs_estimate(sample_data):
    """Test revenue vs estimate calculation including edge cases."""
    result = get_revenue_vs_estimate(sample_data['revenue'], sample_data['revenue_estimate'])

    # First period - normal calculation
    expected_ratio = 2000 / 1900  # ≈ 1.0526
    assert result[0] == pytest.approx(expected_ratio)

    # Zero estimate period
    assert pd.isna(result[1])

    # Third period - normal calculation
    expected_ratio = 1800 / 1750  # ≈ 1.0286
    assert result[2] == pytest.approx(expected_ratio)

    # Fourth period - normal calculation
    expected_ratio = 2100 / 2000  # = 1.05
    assert result[3] == pytest.approx(expected_ratio)


def test_shares_outstanding_vs_estimate(sample_data):
    """Test shares outstanding vs estimate calculation including edge cases."""
    result = get_shares_outstanding_vs_estimate(sample_data['net_income'], sample_data['eps'],
                                                sample_data['net_income_estimate'], sample_data['eps_estimate'])

    # First period - normal calculation
    expected_ratio = 100 / 95  # = 1.0526
    assert result[0] == pytest.approx(expected_ratio)

    # Zero estimate period
    assert pd.isna(result[1])

    # Third period - normal calculation
    expected_ratio = 90 / 85  # ≈ 1.0588
    assert result[2] == pytest.approx(expected_ratio)

    # Fourth period - normal calculation
    expected_ratio = 110 / 105  # ≈ 1.0476
    assert result[3] == pytest.approx(expected_ratio)


def test_free_cash_flow_growth(sample_data):
    """Test free cash flow growth calculation including edge cases."""
    result = get_free_cash_flow_growth(sample_data['free_cash_flow'])

    # First period - no prior data
    assert pd.isna(result[0])

    # Zero denominator period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - normal calculation
    expected_growth = (130 - 110) / 110  # ≈ 0.1818
    assert result[3] == pytest.approx(expected_growth)


def test_free_cash_flow_average_growth(sample_data):
    """Test free cash flow average growth calculation including edge cases."""
    result = get_free_cash_flow_average_growth(sample_data['free_cash_flow'])

    # First period - insufficient data
    assert pd.isna(result[0])

    # Zero value period
    assert pd.isna(result[1])

    # Third period - growth from zero
    assert pd.isna(result[2])  # Should be NaN due to zero division

    # Fourth period - valid growth
    expected_growth = (130 - 110) / 110  # Only valid growth rate
    assert result[3] == pytest.approx(expected_growth)
