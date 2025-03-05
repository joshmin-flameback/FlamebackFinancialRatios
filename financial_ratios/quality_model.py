import pandas as pd
import numpy as np
from typing import Union

"""
Quality Analysis Module

This module contains functions for calculating various quality metrics and ratios
organized into the following categories:

1. Growth Quality (1 function)
   - get_intrinsic_compounding_rate

2. Earnings Quality (2 functions)
   - get_dips_in_profit_over_10yrs
   - get_roic_band

3. Cash Flow Quality (4 functions)
   - get_cfo_band
   - get_negative_dips_in_fcf_over_10yrs
   - get_negative_fcf_years
   - get_fcf_to_net_profit_band

Total Functions: 7

Note: All functions handle invalid calculations (like division by zero) by returning NaN
values for those specific time periods, maintaining the time series structure.
"""

# ----------------------
# 1. Growth Quality
# ----------------------

def get_intrinsic_compounding_rate(
        net_income: pd.Series,
        total_assets: pd.Series,
        total_liabilities: pd.Series,
        dividend_paid: pd.Series
) -> pd.Series:
    """
    Calculate the Annual Intrinsic Compounding Rate (AICR), which measures a company's
    ability to grow its equity value through retained earnings.

    AICR is determined using the formula:
        AICR = ROE * Retention Ratio
    where,
        ROE = Net Income / Shareholder's Equity
        Retention Ratio = 1 - Dividend Payout Ratio
        Dividend Payout Ratio = Dividends / Net Income

    A higher AICR indicates better quality of growth as it shows the company can
    effectively reinvest its earnings for future growth.

    Args:
        net_income (pd.Series): Time series of Net Income values
        total_assets (pd.Series): Time series of Total Assets values
        total_liabilities (pd.Series): Time series of Total Liabilities values
        dividend_paid (pd.Series): Time series of Dividend Paid values

    Returns:
        pd.Series: Time series of AICR values. Returns NaN for periods with zero
                  shareholder equity or net income, or insufficient data.
    """
    # Handle zero values
    safe_net_income = net_income.replace(0, np.nan)
    
    # Calculate shareholder equity and handle zero values
    shareholder_equity = total_assets - total_liabilities
    safe_shareholder_equity = shareholder_equity.replace(0, np.nan)
    
    # Calculate ROE and retention ratio
    return_on_equity = safe_net_income / safe_shareholder_equity
    dividend_payout_ratio = dividend_paid / safe_net_income
    retention_ratio = 1 - dividend_payout_ratio.clip(lower=0, upper=1)  # Ensure ratio is between 0 and 1
    
    return return_on_equity * retention_ratio

# ----------------------
# 2. Earnings Quality
# ----------------------

def get_dips_in_profit_over_10yrs(
        revenue: pd.Series,
        total_expense: pd.Series
) -> pd.Series:
    """
    Count the number of times the annual profit dips by more than 10% over a 10-year period.
    This metric helps identify the stability and quality of a company's earnings.

    A lower count of dips indicates higher earnings quality and more stable business operations.
    Frequent large dips may signal operational issues or cyclical business risks.

    Args:
        revenue (pd.Series): Time series of revenue values
        total_expense (pd.Series): Time series of total expense values

    Returns:
        pd.Series: Time series of cumulative dip counts. Returns NaN for periods with
                  insufficient data (less than 10 years).
    """
    # Calculate profit and handle NaN values
    profit_series = revenue - total_expense
    
    # Ensure we have enough data
    if len(profit_series) < 10:
        return pd.Series(np.nan, index=profit_series.index)
    
    # Calculate year-over-year change
    profit_change = profit_series.pct_change(periods=10) * 100
    
    # Count large dips and handle NaN values
    large_dips = (profit_change < -10).fillna(False).astype(int)
    return large_dips.cumsum()

def get_roic_band(
        invested_capital: pd.Series,
        nopat: pd.Series
) -> pd.Series:
    """
    Calculate the Return on Invested Capital (ROIC) Band percentage by comparing current ROIC
    to its historical mean and standard deviation over 5-10 years.

    ROIC measures how efficiently a company uses its capital to generate profits.
    The band shows how current ROIC deviates from historical norms, helping identify
    potential improvements or deterioration in capital efficiency.

    Args:
        invested_capital (pd.Series): Time series of invested capital values
        nopat (pd.Series): Time series of Net Operating Profit After Tax values

    Returns:
        pd.Series: Time series of ROIC deviation values (in standard deviations from mean).
                  Returns NaN for periods with zero invested capital or insufficient data.

    Raises:
        ValueError: If less than 5 periods of data are available
    """
    # Handle zero values
    safe_invested_capital = invested_capital.replace(0, np.nan)
    
    # Calculate ROIC
    roic = nopat / safe_invested_capital
    rolling_window = min(10, len(roic))
    
    if rolling_window < 5:
        raise ValueError("Insufficient data: At least 5 periods of ROIC required.")

    # Calculate rolling statistics with NaN handling
    mean_roic = roic.rolling(window=rolling_window, min_periods=5).mean()
    std_roic = roic.rolling(window=rolling_window, min_periods=5).std()
    
    # Handle zero standard deviation
    safe_std_roic = std_roic.replace(0, np.nan)
    
    return (roic - mean_roic) / safe_std_roic

# ------------------------
# 3. Cash Flow Quality
# ------------------------

def get_cfo_band(cfo: pd.Series) -> pd.Series:
    """
    Calculate the Cash Flow from Operations (CFO) Band percentage by comparing current CFO
    to its historical mean and standard deviation over 5-10 years.

    This metric helps identify unusual deviations in operating cash flow, which could
    signal changes in business quality or potential accounting issues.

    Args:
        cfo (pd.Series): Time series of Cash Flow from Operations values

    Returns:
        pd.Series: Time series of CFO deviation values (in standard deviations from mean).
                  Returns NaN for periods with insufficient data or zero standard deviation.

    Raises:
        ValueError: If less than 5 periods of data are available
    """
    rolling_window = min(10, len(cfo))
    
    if rolling_window < 5:
        raise ValueError("Insufficient data: At least 5 periods of CFO required.")

    # Calculate rolling statistics with NaN handling
    mean_cfo = cfo.rolling(window=rolling_window, min_periods=5).mean()
    std_cfo = cfo.rolling(window=rolling_window, min_periods=5).std()
    
    # Handle zero standard deviation
    safe_std_cfo = std_cfo.replace(0, np.nan)
    
    return (cfo - mean_cfo) / safe_std_cfo

def get_negative_dips_in_fcf_over_10yrs(fcf: pd.Series) -> pd.Series:
    """
    Count the number of years with negative year-over-year changes in Free Cash Flow (FCF)
    over a 10-year period.

    Frequent negative dips in FCF may indicate poor cash flow quality or operational issues.
    This metric helps assess the stability and reliability of a company's cash generation.

    Args:
        fcf (pd.Series): Time series of Free Cash Flow values

    Returns:
        pd.Series: Time series of cumulative negative FCF change counts. Returns NaN for
                  periods with insufficient data (less than 10 years).
    """
    # Ensure we have enough data
    if len(fcf) < 10:
        return pd.Series(np.nan, index=fcf.index)
        
    # Calculate year-over-year change
    fcf_change = fcf.pct_change() * 100
    
    # Count negative dips and handle NaN values
    dips = (fcf_change < 0).fillna(False).astype(int)
    return dips.rolling(window=10, min_periods=10).sum()

def get_negative_fcf_years(fcf: pd.Series) -> pd.Series:
    """
    Count the number of years with negative Free Cash Flow (FCF) over a 10-year period.

    Negative FCF years indicate periods where the company consumed rather than generated cash.
    A high count of negative FCF years may signal poor business quality or financial distress.

    Args:
        fcf (pd.Series): Time series of Free Cash Flow values

    Returns:
        pd.Series: Time series of cumulative negative FCF year counts. Returns NaN for
                  periods with insufficient data (less than 10 years).
    """
    # Ensure we have enough data
    if len(fcf) < 10:
        return pd.Series(np.nan, index=fcf.index)
        
    # Count negative years and handle NaN values
    negative_fcf_years = (fcf < 0).fillna(False).astype(int)
    return negative_fcf_years.rolling(window=10, min_periods=10).sum()

def get_fcf_to_net_profit_band(
        fcf: pd.Series,
        net_profit: pd.Series
) -> pd.Series:
    """
    Calculate the Free Cash Flow (FCF) to Net Profit ratio band by comparing the current
    ratio to its historical mean and standard deviation.

    This metric helps identify the quality of earnings by showing how well net profit
    converts to actual cash flow. A ratio consistently below historical norms may
    indicate deteriorating earnings quality or aggressive accounting practices.

    Args:
        fcf (pd.Series): Time series of Free Cash Flow values
        net_profit (pd.Series): Time series of Net Profit values

    Returns:
        pd.Series: Time series of ratio deviation values (in standard deviations from mean).
                  Returns NaN for periods with zero net profit or insufficient data.

    Raises:
        ValueError: If less than 5 periods of data are available
    """
    # Handle zero values
    safe_net_profit = net_profit.replace(0, np.nan)
    
    # Calculate FCF to Net Profit ratio
    ratio = fcf / safe_net_profit
    rolling_window = min(10, len(ratio))
    
    if rolling_window < 5:
        raise ValueError("Insufficient data: At least 5 periods of data required.")

    # Calculate rolling statistics with NaN handling
    mean_ratio = ratio.rolling(window=rolling_window, min_periods=5).mean()
    std_ratio = ratio.rolling(window=rolling_window, min_periods=5).std()
    
    # Handle zero standard deviation
    safe_std_ratio = std_ratio.replace(0, np.nan)
    
    return (ratio - mean_ratio) / safe_std_ratio
