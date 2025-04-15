import pandas as pd
import numpy as np
from typing import Union

from financial_ratios.utils.helpers import calculate_growth, get_consecutive_number_of_growth, calculate_average

"""
Financial Ratio Analysis Module

This module contains functions for calculating various financial ratios and metrics
organized into the following categories:

1. Composite Scores (1 function)
   - get_piotroski_score

2. Basic Growth Metrics (2 functions)
   - get_revenue_growth
   - get_eps_growth

3. Consecutive Growth Analysis (2 functions)
   - get_revenue_consecutive_growth
   - get_eps_consecutive_growth

4. Average Growth Analysis (6 functions)
   - get_average_revenue_growth
   - get_average_gross_margin
   - get_average_gross_margin_growth
   - get_average_ebitda
   - get_average_ebitda_growth
   - get_average_eps_growth


5. Growth Comparison Metrics (4 functions)
   - get_revenue_growth_vs_average_growth
   - get_eps_growth_vs_average_growth
   - get_ebitda_growth_vs_average_growth
   - get_gross_margin_growth_vs_average_growth


6. Return Metrics (4 functions)
   - get_return_on_equity
   - get_roe_vs_average_roe
   - get_return_on_assets
   - get_roa_vs_average_roa

7. Estimate Comparison Metrics (2 functions)
   - get_revenue_vs_estimate
   - get_shares_outstanding_vs_estimate

8. Cash Flow Analysis (2 functions)
   - get_free_cash_flow_growth
   - get_free_cash_flow_average_growth

Total Functions: 23

Note: All functions handle invalid calculations (like division by zero) by returning NaN
values for those specific time periods, maintaining the time series structure.
"""

# ----------------------
# 1. Composite Scores
# ----------------------

def get_piotroski_score(
    net_income: pd.Series,
    total_assets: pd.Series,
    cash_flow_from_operations: pd.Series,
    current_assets: pd.Series,
    current_liabilities: pd.Series,
    long_term_debt: pd.Series,
    shares_outstanding: pd.Series,
    revenue: pd.Series,
    cogs: pd.Series
) -> pd.Series:
    """
    Calculate the Piotroski F-Score, a value investing metric that uses nine criteria to determine the financial strength of a company.
    
    The score ranges from 0-9, where 9 indicates the highest financial strength. The criteria are:
    1. Return on Assets (ROA) > 0
    2. Operating Cash Flow > 0
    3. ROA increasing
    4. Operating Cash Flow > Net Income
    5. Decreasing Leverage (Long-term debt)
    6. Increasing Current Ratio
    7. No new shares issued
    8. Increasing Gross Margin
    9. Increasing Asset Turnover
    
    All errors are handled gracefully by returning NaN values to maintain time series integrity.
    
    Parameters
    ----------
    net_income : pd.Series
        Time series of net income values
    total_assets : pd.Series
        Time series of total assets values
    cash_flow_from_operations : pd.Series
        Time series of operating cash flow values
    current_assets : pd.Series
        Time series of current assets values
    current_liabilities : pd.Series
        Time series of current liabilities values
    long_term_debt : pd.Series
        Time series of long-term debt values
    shares_outstanding : pd.Series
        Time series of shares outstanding values
    revenue : pd.Series
        Time series of revenue values
    cogs : pd.Series
        Time series of cost of goods sold values
    
    Returns
    -------
    pd.Series
        Time series of Piotroski F-scores (0-9)
    """
    try:
        # Initialize score series
        score = pd.Series(0, index=net_income.index)
        
        # 1. ROA (Net Income / Total Assets)
        roa = net_income / total_assets.replace(0, np.nan)
        score = score.where(pd.isna(roa), score + (roa > 0))
        
        # 2. Operating Cash Flow
        score = score.where(pd.isna(cash_flow_from_operations), 
                          score + (cash_flow_from_operations > 0))
        
        # 3. ROA Change (requires lag)
        roa_change = roa - roa.shift(1)
        score = score.where(pd.isna(roa_change), score + (roa_change > 0))
        
        # 4. Cash Flow vs Net Income (Quality of Earnings)
        score = score.where(pd.isna(cash_flow_from_operations) | pd.isna(net_income),
                          score + (cash_flow_from_operations > net_income))
        
        # 5. Long-term Debt Change
        debt_change = long_term_debt - long_term_debt.shift(1)
        score = score.where(pd.isna(debt_change), score + (debt_change < 0))
        
        # 6. Current Ratio Change
        current_ratio = current_assets / current_liabilities.replace(0, np.nan)
        curr_ratio_change = current_ratio - current_ratio.shift(1)
        score = score.where(pd.isna(curr_ratio_change), score + (curr_ratio_change > 0))
        
        # 7. Shares Outstanding Change (no new shares issued)
        shares_change = shares_outstanding - shares_outstanding.shift(1)
        score = score.where(pd.isna(shares_change), score + (shares_change <= 0))
        
        # 8. Gross Margin Change
        gross_margin = (revenue - cogs) / revenue.replace(0, np.nan)
        margin_change = gross_margin - gross_margin.shift(1)
        score = score.where(pd.isna(margin_change), score + (margin_change > 0))
        
        # 9. Asset Turnover Change
        asset_turnover = revenue / total_assets.replace(0, np.nan)
        turnover_change = asset_turnover - asset_turnover.shift(1)
        score = score.where(pd.isna(turnover_change), score + (turnover_change > 0))
        
        # Handle edge cases
        score = score.where(~((total_assets == 0) | (current_liabilities == 0)), np.nan)
        score = score.where(~pd.isna(net_income), np.nan)  # If net income is NaN, score is NaN
        
        return score.astype('Int64')  # Convert to nullable integer type
        
    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=net_income.index)

# ------------------------
# 2. Basic Growth Metrics
# ------------------------

def get_revenue_growth(revenue: pd.Series) -> pd.Series:
    """
    Calculate the period-over-period revenue growth rate.

    Parameters
    ----------
    revenue : pd.Series
        Time series of revenue values

    Returns
    -------
    pd.Series
        Time series of revenue growth rates
    """
    try:
        # Handle missing values
        if revenue is None or revenue.empty:
            return pd.Series(dtype=float)

        # Calculate growth rate
        prev_revenue = revenue.shift(1)
        growth = (revenue - prev_revenue) / prev_revenue.abs()
        
        # Cap extreme growth values at 1000% (10x)
        growth = growth.clip(-10, 10)

        # Handle invalid calculations
        growth = growth.replace([np.inf, -np.inf], np.nan)

        return growth

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=revenue.index)


def get_eps_growth(eps: pd.Series) -> pd.Series:
    """
    Calculate the period-over-period EPS growth rate.

    Parameters
    ----------
    eps : pd.Series
        Time series of EPS values

    Returns
    -------
    pd.Series
        Time series of EPS growth rates. Returns NaN for periods with insufficient data,
        zero EPS values in denominator, or other invalid calculations.
    """
    try:
        if eps is None or eps.empty:
            return pd.Series(dtype=float)

        growth = (eps - eps.shift(1)) / eps.shift(1).replace(0, np.nan)
        return growth.replace([np.inf, -np.inf], np.nan)

    except Exception as e:
        return pd.Series(np.nan, index=eps.index)

# --------------------------------
# 3. Consecutive Growth Analysis
# --------------------------------

def get_revenue_consecutive_growth(revenue: pd.Series) -> pd.Series:
    """
    Calculate the number of consecutive periods of revenue growth.

    Parameters
    ----------
    revenue : pd.Series
        Time series of revenue values

    Returns
    -------
    pd.Series
        Time series of consecutive growth periods
    """
    try:
        # Handle missing values
        if revenue is None or revenue.empty:
            return pd.Series(dtype=float)

        # Calculate growth rates
        growth = get_revenue_growth(revenue)

        # Initialize consecutive growth series
        consecutive = pd.Series(0, index=revenue.index)

        # Calculate consecutive periods
        for i in range(1, len(revenue)):
            if pd.isna(growth.iloc[i]) or growth.iloc[i] <= 0:
                consecutive.iloc[i] = 0
            else:
                consecutive.iloc[i] = consecutive.iloc[i - 1] + 1

        return consecutive

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=revenue.index)


def get_eps_consecutive_growth(eps: pd.Series) -> pd.Series:
    """
    Calculate the number of consecutive periods of EPS growth.

    Parameters
    ----------
    eps : pd.Series
        Time series of EPS values

    Returns
    -------
    pd.Series
        Time series of consecutive growth periods
    """
    try:
        # Handle missing values
        if eps is None or eps.empty:
            return pd.Series(dtype=float)

        # Calculate growth rates
        growth = get_eps_growth(eps)

        # Initialize consecutive growth series
        consecutive = pd.Series(0, index=eps.index)

        # Calculate consecutive periods
        for i in range(1, len(eps)):
            if pd.isna(growth.iloc[i]) or growth.iloc[i] <= 0:
                consecutive.iloc[i] = 0
            else:
                consecutive.iloc[i] = consecutive.iloc[i - 1] + 1

        return consecutive

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=eps.index)

# ---------------------------
# 4. Average Growth Analysis
# ---------------------------

def get_average_revenue_growth(revenue: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average revenue growth rate.

    Parameters
    ----------
    revenue : pd.Series
        Time series of revenue values

    Returns
    -------
    pd.Series
        Time series of average revenue growth rates
    """
    try:
        # Handle missing values
        if revenue is None or revenue.empty:
            return pd.Series(dtype=float)

        # Calculate growth rates
        growth = get_revenue_growth(revenue)

        # Calculate rolling average with more lenient min_periods
        avg_growth = growth.rolling(window=20, min_periods=1).mean()

        # Handle invalid calculations
        avg_growth = avg_growth.replace([np.inf, -np.inf], np.nan)

        return avg_growth

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=revenue.index)


def get_average_gross_margin(gross_margin: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average gross margin.

    Parameters
    ----------
    gross_margin : pd.Series
        Time series of gross margin values

    Returns
    -------
    pd.Series
        Time series of average gross margin values
    """
    try:
        # Handle missing values
        if gross_margin is None or gross_margin.empty:
            return pd.Series(dtype=float)

        # Calculate rolling average
        avg_margin = gross_margin.rolling(window=20, min_periods=20).mean()

        # Handle invalid calculations
        avg_margin = avg_margin.replace([np.inf, -np.inf], np.nan)

        return avg_margin

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=gross_margin.index)

def get_average_gross_margin_growth(gross_margin: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average gross margin growth rate.

    Args:
        gross_margin (pd.Series): Time series of gross margin values

    Returns:
        pd.Series: Time series of average gross margin growth rates. Returns NaN for
                  periods with insufficient data or zero gross margin values.
    """
    # Handle zero gross margin values
    safe_gross_margin = gross_margin.replace(0, np.nan)
    growth_rates = calculate_growth(safe_gross_margin, lag=1)
    return calculate_average(growth_rates, window=20)


def get_average_ebitda(ebitda: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average EBITDA.

    Parameters
    ----------
    ebitda : pd.Series
        Time series of EBITDA values

    Returns
    -------
    pd.Series
        Time series of average EBITDA values. Returns NaN for periods with 
        insufficient data or invalid calculations.
    """
    try:
        if ebitda is None or ebitda.empty:
            return pd.Series(dtype=float)
            
        return calculate_average(ebitda, window=20)
        
    except Exception as e:
        return pd.Series(np.nan, index=ebitda.index)

def get_average_ebitda_growth(ebitda: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average EBITDA growth rate.

    Parameters
    ----------
    ebitda : pd.Series
        Time series of EBITDA values

    Returns
    -------
    pd.Series
        Time series of average EBITDA growth rates. Returns NaN for periods with
        insufficient data or invalid calculations.
    """
    try:
        if ebitda is None or ebitda.empty:
            return pd.Series(dtype=float)
            
        growth = calculate_growth(ebitda)
        return calculate_average(growth, window=20)
        
    except Exception as e:
        return pd.Series(np.nan, index=ebitda.index)

def get_average_eps_growth(eps: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average EPS growth rate.

    Parameters
    ----------
    eps : pd.Series
        Time series of EPS values

    Returns
    -------
    pd.Series
        Time series of average EPS growth rates. Returns NaN for periods with
        insufficient data or invalid calculations.
    """
    try:
        if eps is None or eps.empty:
            return pd.Series(dtype=float)
            
        growth = calculate_growth(eps)
        return calculate_average(growth, window=20)
        
    except Exception as e:
        return pd.Series(np.nan, index=eps.index)


# ------------------------------
# 5. Growth Comparison Metrics
# ------------------------------

def get_revenue_growth_vs_average_growth(revenue: pd.Series) -> pd.Series:
    """
    Calculate the ratio of current revenue growth to its 20-period trailing average growth.

    Parameters
    ----------
    revenue : pd.Series
        Time series of revenue values

    Returns
    -------
    pd.Series
        Time series of growth ratios
    """
    try:
        # Handle missing values
        if revenue is None or revenue.empty:
            return pd.Series(dtype=float)

        # Calculate current growth and average growth
        current_growth = get_revenue_growth(revenue)
        avg_growth = get_average_revenue_growth(revenue)

        # Calculate ratio, handling negative averages
        ratio = current_growth / avg_growth.abs().replace(0, np.nan)
        
        # If average growth is negative, flip the ratio sign
        ratio = ratio * (avg_growth / avg_growth.abs()).fillna(1)

        # Handle invalid calculations and cap extreme values
        ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(-10, 10)

        return ratio

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=revenue.index)

def get_eps_growth_vs_average_growth(eps: pd.Series) -> pd.Series:
    """
    Calculate the ratio of current EPS growth to its 20-period trailing average growth.
    This metric helps identify if current EPS growth is above or below historical trends.

    Args:
        eps (pd.Series): Time series of EPS values

    Returns:
        pd.Series: Time series of growth ratios (current growth / average growth)
    """
    # Handle zero EPS values
    safe_eps = eps.replace(0, np.nan)
    growth_rates = calculate_growth(safe_eps, lag=1)
    average_growth_rates = calculate_average(growth_rates, window=20)
    return growth_rates / average_growth_rates


def get_ebitda_growth_vs_average_growth(ebitda: pd.Series) -> pd.Series:
    """
    Calculate the ratio of current EBITDA growth to its 20-period trailing average growth.

    Parameters
    ----------
    ebitda : pd.Series
        Time series of EBITDA values

    Returns
    -------
    pd.Series
        Time series of growth ratios (current growth / average growth). Returns NaN for
        periods with insufficient data or invalid calculations.
    """
    try:
        if ebitda is None or ebitda.empty:
            return pd.Series(dtype=float)
            
        current_growth = calculate_growth(ebitda)
        avg_growth = get_average_ebitda_growth(ebitda)
        return current_growth / avg_growth.replace(0, np.nan)
        
    except Exception as e:
        return pd.Series(np.nan, index=ebitda.index)

def get_gross_margin_growth_vs_average_growth(gross_margin: pd.Series) -> pd.Series:
    """
    Calculate the ratio of current gross margin growth to its 20-period trailing average growth.

    Parameters
    ----------
    gross_margin : pd.Series
        Time series of gross margin values

    Returns
    -------
    pd.Series
        Time series of growth ratios (current growth / average growth). Returns NaN for
        periods with insufficient data or invalid calculations.
    """
    try:
        if gross_margin is None or gross_margin.empty:
            return pd.Series(dtype=float)
            
        current_growth = calculate_growth(gross_margin)
        avg_growth = get_average_gross_margin_growth(gross_margin)
        return current_growth / avg_growth.replace(0, np.nan)
        
    except Exception as e:
        return pd.Series(np.nan, index=gross_margin.index)


# ------------------
# 6. Return Metrics
# ------------------

def get_return_on_equity(net_income: pd.Series, shareholders_equity: pd.Series) -> pd.Series:
    """
    Calculate Return on Equity (ROE).

    Parameters
    ----------
    net_income : pd.Series
        Time series of net income values
    shareholders_equity : pd.Series
        Time series of shareholders' equity values

    Returns
    -------
    pd.Series
        Time series of ROE values
    """
    try:
        # Handle missing values
        if net_income is None or shareholders_equity is None:
            return pd.Series(dtype=float)

        # Calculate ROE
        roe = net_income / shareholders_equity.replace(0, np.nan)

        # Handle invalid calculations
        roe = roe.replace([np.inf, -np.inf], np.nan)

        return roe

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=net_income.index)


def get_roe_vs_average_roe(net_income: pd.Series, shareholders_equity: pd.Series) -> pd.Series:
    """
    Calculate the ratio of current Return on Equity (ROE) to its 20-period trailing average.

    Parameters
    ----------
    net_income : pd.Series
        Time series of net income values
    shareholders_equity : pd.Series
        Time series of shareholders' equity values

    Returns
    -------
    pd.Series
        Time series of ROE ratios (current ROE / average ROE). Returns NaN for
        periods with insufficient data or invalid calculations.
    """
    try:
        if net_income is None or shareholders_equity is None or net_income.empty or shareholders_equity.empty:
            return pd.Series(dtype=float)
            
        current_roe = get_return_on_equity(net_income, shareholders_equity)
        avg_roe = calculate_average(current_roe, window=20)
        return current_roe / avg_roe.replace(0, np.nan)
        
    except Exception as e:
        return pd.Series(np.nan, index=net_income.index)

def get_return_on_assets(net_income: pd.Series, assets: pd.Series) -> pd.Series:
    """
    Calculate Return on Assets (ROA), which measures how efficiently a company uses
    its assets to generate earnings.

    Args:
        net_income (pd.Series): Time series of net income values
        assets (pd.Series): Time series of total assets values

    Returns:
        pd.Series: Time series of ROA values
    """
    # Handle zero assets values
    safe_assets = assets.replace(0, np.nan)
    return net_income / safe_assets


def get_roa_vs_average_roa(net_income: pd.Series, assets: pd.Series) -> pd.Series:
    """Calculate the ratio of current ROA to its average."""
    try:
        if net_income is None or assets is None:
            return pd.Series(dtype=float)

        current_roa = net_income / assets.replace(0, np.nan)
        avg_roa = current_roa.rolling(window=20, min_periods=20).mean()
        ratio = current_roa / avg_roa.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        return ratio
    except Exception as e:
        return pd.Series(np.nan, index=net_income.index)

# -----------------------------
# 7. Estimate Comparison Metrics
# -----------------------------

def get_revenue_vs_estimate(revenue: pd.Series, revenue_estimate: pd.Series) -> pd.Series:
    """
    Calculate the ratio of actual revenue to estimated revenue.

    Parameters
    ----------
    revenue : pd.Series
        Time series of actual revenue values
    revenue_estimate : pd.Series
        Time series of estimated revenue values

    Returns
    -------
    pd.Series
        Time series of revenue ratios
    """
    try:
        # Handle missing values
        if revenue is None or revenue_estimate is None:
            return pd.Series(dtype=float)

        # Calculate ratio
        ratio = revenue / revenue_estimate.replace(0, np.nan)

        # Handle invalid calculations
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        return ratio

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=revenue.index)


def get_shares_outstanding_vs_estimate(
    net_income: pd.Series,
    eps: pd.Series,
    net_income_estimate: pd.Series,
    eps_estimate: pd.Series
) -> pd.Series:
    """
    Calculate the ratio of actual shares outstanding to estimated shares outstanding.
    Shares outstanding is derived from net income / EPS.

    Parameters
    ----------
    net_income : pd.Series
        Time series of actual net income values
    eps : pd.Series
        Time series of actual EPS values
    net_income_estimate : pd.Series
        Time series of estimated net income values
    eps_estimate : pd.Series
        Time series of estimated EPS values

    Returns
    -------
    pd.Series
        Time series of shares outstanding ratios (actual / estimated). Returns NaN for
        periods with insufficient data, zero values in denominators, or invalid calculations.
    """
    try:
        if any(x is None or x.empty for x in [net_income, eps, net_income_estimate, eps_estimate]):
            return pd.Series(dtype=float)
            
        actual_shares = net_income / eps.replace(0, np.nan)
        estimated_shares = net_income_estimate / eps_estimate.replace(0, np.nan)
        return actual_shares / estimated_shares.replace(0, np.nan)
        
    except Exception as e:
        return pd.Series(np.nan, index=net_income.index)

# ----------------------
# 8. Cash Flow Analysis
# ----------------------

def get_free_cash_flow_growth(free_cash_flow: pd.Series) -> pd.Series:
    """
    Calculate the year-over-year free cash flow growth rate.

    Parameters
    ----------
    free_cash_flow : pd.Series
        Time series of free cash flow values

    Returns
    -------
    pd.Series
        Time series of free cash flow growth rates
    """
    try:
        # Handle missing values
        if free_cash_flow is None or free_cash_flow.empty:
            return pd.Series(dtype=float)

        # Calculate growth rate
        growth = (free_cash_flow - free_cash_flow.shift(1)) / free_cash_flow.shift(1).replace(0, np.nan)

        # Handle invalid calculations
        growth = growth.replace([np.inf, -np.inf], np.nan)

        return growth

    except Exception as e:
        # Return NaN series in case of any error
        return pd.Series(np.nan, index=free_cash_flow.index)


def get_free_cash_flow_average_growth(free_cash_flow: pd.Series) -> pd.Series:
    """
    Calculate the 20-period trailing average free cash flow growth rate.

    Parameters
    ----------
    free_cash_flow : pd.Series
        Time series of free cash flow values

    Returns
    -------
    pd.Series
        Time series of average free cash flow growth rates. Returns NaN for periods with
        insufficient data or invalid calculations.
    """
    try:
        if free_cash_flow is None or free_cash_flow.empty:
            return pd.Series(dtype=float)
            
        growth = calculate_growth(free_cash_flow)
        return calculate_average(growth, window=20)
        
    except Exception as e:
        return pd.Series(np.nan, index=free_cash_flow.index)