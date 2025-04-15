import pandas as pd
import numpy as np
from typing import Union

"""
Valuation Analysis Module

This module contains functions for calculating various valuation metrics and ratios
organized into the following categories:

1. Intrinsic Value Metrics (2 functions)
   - get_steady_state_value
   - get_fair_value_vs_market_price

2. Price Multiple Bands (3 functions)
   - get_price_to_revenue_band
   - get_price_to_eps_band
   - get_price_to_cfo_band

3. Yield Metrics (1 function)
   - get_fcf_yield

Total Functions: 6

Note: All functions handle invalid calculations (like division by zero) by returning NaN
values for those specific time periods, maintaining the time series structure.
"""

# ---------------------------
# 1. Intrinsic Value Metrics
# ---------------------------

def get_steady_state_value(
        eps: pd.Series,
        wacc: pd.Series,
        current_price: pd.Series
) -> pd.Series:
    """
    Calculate the Steady State Value, which measures how much a stock is over/undervalued
    assuming the company maintains its current earnings power indefinitely.

    A positive percentage indicates potential undervaluation, while a negative percentage
    suggests potential overvaluation. This metric is most useful for stable, mature companies.

    Formula:
        Steady State Value = ((EPS / WACC) / Current Price - 1) * 100
    where:
        - EPS: Earnings Per Share
        - WACC: Weighted Average Cost of Capital

    Args:
        eps (pd.Series): Time series of Earnings Per Share values
        wacc (pd.Series): Time series of Weighted Average Cost of Capital values
        current_price (pd.Series): Time series of current stock prices

    Returns:
        pd.Series: Time series of Steady State Value in percentage. Returns NaN for periods
                  with zero WACC or current price, or insufficient data.
    """
    # Handle zero values
    safe_wacc = wacc.replace(0, np.nan)
    safe_price = current_price.replace(0, np.nan)
    
    # Calculate steady state value
    return ((eps / safe_wacc) / safe_price - 1) * 100

def get_fair_value_vs_market_price(
        net_income: pd.Series,
        total_assets: pd.Series,
        total_liabilities: pd.Series,
        eps: pd.Series,
        current_price: pd.Series
) -> pd.Series:
    """
    Calculate the fair value vs current market price ratio as a percentage, using a
    comprehensive formula that incorporates multiple valuation factors.

    This metric provides a more nuanced view of valuation by considering:
    - Return on Equity (business quality)
    - EPS Growth (growth potential)
    - Historical P/E ratios (market sentiment)

    Formula:
    ((log(1 + ROE) + Recent EPS Growth) * (TTM EPS * (1 + log(TTM EPS))) * Avg 3-year P/E) / Current Price - 1
    where:
    - ROE = Net Income / Shareholder's Equity
    - Shareholder's Equity = Total Assets - Total Liabilities
    - Recent EPS Growth = (Current EPS - Previous EPS) / Previous EPS
    - P/E Ratio = Current Price / EPS
    - Avg 3-year P/E = Rolling 3-year average of P/E ratio

    Args:
        net_income (pd.Series): Time series of net income values
        total_assets (pd.Series): Time series of total assets values
        total_liabilities (pd.Series): Time series of total liabilities values
        eps (pd.Series): Time series of earnings per share values
        current_price (pd.Series): Time series of stock prices

    Returns:
        pd.Series: Time series of fair value vs current market price ratios in absolute percentage.
                  Returns NaN for periods with zero equity, EPS, or current price, or insufficient data.
    """
    # Align all series to common index
    common_index = net_income.index.intersection(total_assets.index)\
        .intersection(total_liabilities.index)\
        .intersection(eps.index)\
        .intersection(current_price.index)
    
    net_income = net_income.loc[common_index].sort_index()
    total_assets = total_assets.loc[common_index].sort_index()
    total_liabilities = total_liabilities.loc[common_index].sort_index()
    eps = eps.loc[common_index].sort_index()
    current_price = current_price.replace(0, np.nan).loc[common_index].sort_index()

    # Calculate components with NaN handling
    shareholder_equity = total_assets - total_liabilities
    safe_equity = shareholder_equity.replace(0, np.nan)
    roe = net_income / safe_equity
    
    safe_eps = eps.replace(0, np.nan)
    eps_growth = (eps - eps.shift(1)) / safe_eps.shift(1)
    
    pe_ratio = current_price / safe_eps
    # Use 3-year average for historical comparison
    avg_pe = pe_ratio.rolling(window=3, min_periods=1).mean()  # Require at least 1 year of data

    # Calculate fair value ratio with NaN handling
    fair_value_ratio = (
        (np.log1p(roe.clip(lower=-0.99)) + eps_growth) *  # Prevent log(0) or log(negative)
        (safe_eps * (1 + np.log1p(safe_eps.abs()))) *
        avg_pe / current_price
    ) - 1

    # Convert to percentage and handle infinities
    return fair_value_ratio.replace([np.inf, -np.inf], np.nan).abs() * 100

# ------------------------
# 2. Price Multiple Bands
# ------------------------

def get_price_to_revenue_band(
        price: pd.Series,
        total_revenue: pd.Series,
        shares_outstanding: pd.Series
) -> pd.Series:
    """
    Calculate the Price to Revenue per Share Band, showing how current P/S ratio
    deviates from its historical average.

    This metric is particularly useful for:
    - Companies with negative earnings where P/E can't be used
    - Comparing companies in the same industry
    - Identifying potential over/undervaluation based on historical trends

    Formula:
        1. Revenue per Share = Total Revenue / Shares Outstanding
        2. Price to Revenue Ratio = Price / Revenue per Share
        3. Band = (Current Ratio - 3-year Mean) / 3-year Standard Deviation

    Args:
        price (pd.Series): Time series of stock price values
        total_revenue (pd.Series): Time series of total revenue values
        shares_outstanding (pd.Series): Time series of shares outstanding values

    Returns:
        pd.Series: Time series of Price to Revenue band values (in standard deviations).
                  Returns NaN for periods with zero revenue or shares, or insufficient data.

    Raises:
        ValueError: If less than 1 year data of data are available
    """
    # Handle zero values
    safe_shares = shares_outstanding.replace(0, np.nan)
    
    # Calculate revenue per share
    revenue_per_share = total_revenue / safe_shares
    safe_revenue_per_share = revenue_per_share.replace(0, np.nan)
    
    # Calculate ratio
    ratio = price / safe_revenue_per_share
    
    # Ensure sufficient data
    if len(ratio) < 1:
        raise ValueError("Insufficient data: At least 12 periods required.")
    
    # Calculate band statistics with NaN handling
    # Use 3-year average for historical comparison
    mean_ratio = ratio.rolling(window=3, min_periods=1).mean()
    std_ratio = ratio.rolling(window=3, min_periods=1).std()
    safe_std_ratio = std_ratio.replace(0, np.nan)
    
    return (ratio - mean_ratio) / safe_std_ratio

def get_price_to_eps_band(
        price: pd.Series,
        eps: pd.Series
) -> pd.Series:
    """
    Calculate the Price to Earnings (P/E) Band, showing how current P/E ratio
    deviates from its historical average.

    This is one of the most widely used valuation metrics as it:
    - Shows how much investors are willing to pay for each dollar of earnings
    - Helps identify if a stock is trading at premium/discount to historical levels
    - Facilitates comparison with industry peers

    Formula:
        1. P/E Ratio = Price / TTM EPS
        2. Band = (Current P/E - 3-year Mean P/E) / 3-year Standard Deviation

    Args:
        price (pd.Series): Time series of stock price values
        eps (pd.Series): Time series of trailing twelve-month EPS values

    Returns:
        pd.Series: Time series of P/E band values (in standard deviations).
                  Returns NaN for periods with zero EPS or insufficient data.

    Raises:
        ValueError: If less than 1 year data are available
    """
    # Handle zero values
    safe_eps = eps.replace(0, np.nan)
    
    # Calculate P/E ratio
    ratio = price / safe_eps
    
    # Ensure sufficient data
    if len(ratio) < 1:
        raise ValueError("Insufficient data: At least 1 year data required.")
    
    # Calculate band statistics with NaN handling
    # Use 3-year average for historical comparison
    mean_ratio = ratio.rolling(window=3, min_periods=1).mean()
    std_ratio = ratio.rolling(window=3, min_periods=1).std()
    safe_std_ratio = std_ratio.replace(0, np.nan)
    
    return (ratio - mean_ratio) / safe_std_ratio

def get_price_to_cfo_band(
        price: pd.Series,
        cfo: pd.Series,
        shares_outstanding: pd.Series
) -> pd.Series:
    """
    Calculate the Price to Cash Flow from Operations (P/CFO) Band, showing how current
    P/CFO ratio deviates from its historical average.

    This metric is valuable because:
    - Cash flows are harder to manipulate than earnings
    - It helps identify companies trading at premium/discount to historical cash generation
    - It's useful for companies with significant non-cash charges

    Formula:
        1. CFO per Share = CFO / Shares Outstanding
        2. P/CFO Ratio = Price / CFO per Share
        3. Band = (Current Ratio - 3-year Mean) / 3-year Standard Deviation

    Args:
        price (pd.Series): Time series of stock price values
        cfo (pd.Series): Time series of Cash Flow from Operations values
        shares_outstanding (pd.Series): Time series of shares outstanding values

    Returns:
        pd.Series: Time series of P/CFO band values (in standard deviations).
                  Returns NaN for periods with zero CFO or shares, or insufficient data.

    Raises:
        ValueError: If less than 1 year data of data are available
    """
    # Handle zero values
    safe_shares = shares_outstanding.replace(0, np.nan)
    
    # Calculate CFO per share
    cfo_per_share = cfo / safe_shares
    safe_cfo_per_share = cfo_per_share.replace(0, np.nan)
    
    # Calculate ratio
    ratio = price / safe_cfo_per_share
    
    # Ensure sufficient data
    if len(ratio) < 1:
        raise ValueError("Insufficient data: At least 1 year data required.")
    
    # Calculate band statistics with NaN handling
    # Use 3-year average for historical comparison
    mean_ratio = ratio.rolling(window=3, min_periods=1).mean()
    std_ratio = ratio.rolling(window=3, min_periods=1).std()
    safe_std_ratio = std_ratio.replace(0, np.nan)
    
    return (ratio - mean_ratio) / safe_std_ratio

# -------------------
# 3. Yield Metrics
# -------------------

def get_fcf_yield(
        fcf: pd.Series,
        price: pd.Series,
        shares_outstanding: pd.Series
) -> pd.Series:
    """
    Calculate the Free Cash Flow (FCF) Yield, which measures how much free cash flow
    a company generates relative to its market value.

    This metric is important because:
    - It shows how much cash is available for dividends, buybacks, or reinvestment
    - Higher yields may indicate undervaluation or strong cash generation ability
    - It's useful for comparing companies across different sectors

    Formula:
        FCF Yield = (Free Cash Flow / Market Capitalization) * 100
    where:
        Market Capitalization = Stock Price × Shares Outstanding

    Args:
        fcf (pd.Series): Time series of Free Cash Flow values
        price (pd.Series): Time series of stock price values
        shares_outstanding (pd.Series): Time series of shares outstanding values

    Returns:
        pd.Series: Time series of FCF Yield in absolute percentage.
                  Returns NaN for periods with zero market cap or insufficient data.
    """
    # Calculate market cap with NaN handling
    market_cap = price * shares_outstanding
    safe_market_cap = market_cap.replace(0, np.nan)
    
    # Calculate FCF yield and convert to percentage
    return (fcf / safe_market_cap * 100).abs()
