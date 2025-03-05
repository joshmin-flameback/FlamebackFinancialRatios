import pandas as pd
import numpy as np
from typing import Union

"""
Financial Health Analysis Module

This module contains functions for calculating various financial health metrics
organized into the following categories:

1. Solvency Ratios (2 functions)
   - get_debt_to_equity_ratio
   - get_interest_coverage_ratio

2. Liquidity Ratios (1 function)
   - get_current_ratio

3. Operational Efficiency (1 function)
   - get_cash_conversion_cycle

4. Bankruptcy Risk (1 function)
   - get_altman_z_score

Total Functions: 5

Note: All functions handle invalid calculations (like division by zero) by returning NaN
values for those specific time periods, maintaining the time series structure.
"""

# ---------------------
# 1. Solvency Ratios
# ---------------------

def get_debt_to_equity_ratio(
        total_debt: pd.Series, total_equity: pd.Series) -> pd.Series:
    """
    Calculate the debt to equity ratio (D/E ratio), a solvency ratio that measures the
    proportion of a company's equity that is financed by debt.

    A higher ratio indicates more leverage and higher risk, while a lower ratio indicates
    a more financially stable business.

    Args:
        total_debt (pd.Series): Time series of total debt values
        total_equity (pd.Series): Time series of total equity values

    Returns:
        pd.Series: Time series of debt to equity ratio values. Returns NaN for periods
                  where equity is zero or negative.
    """
    result = total_debt / total_equity
    # Replace infinite values with NaN (occurs when equity is zero)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def get_interest_coverage_ratio(
        ebitda: pd.Series,
        interest_expense: pd.Series) -> pd.Series:
    """
    Calculate the interest coverage ratio (ICR), a solvency ratio that measures a company's
    ability to pay its interest expenses on outstanding debt using its earnings.

    A higher ratio indicates better ability to meet interest payments. Generally,
    a ratio of 2 or higher is considered safe.

    Args:
        ebitda (pd.Series): Time series of EBITDA (Earnings Before Interest, Taxes, 
                           Depreciation, and Amortization) values
        interest_expense (pd.Series): Time series of interest expense values

    Returns:
        pd.Series: Time series of interest coverage ratio values. Returns NaN for periods
                  where interest expense is zero.
    """
    # Handle zero interest expense by replacing with NaN
    safe_interest = interest_expense.replace(0, np.nan)
    result = ebitda / abs(safe_interest)
    return result

# ---------------------
# 2. Liquidity Ratios
# ---------------------

def get_current_ratio(
        current_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
    """
    Calculate the current ratio, a liquidity ratio that measures a company's ability
    to pay off its short-term liabilities with its current assets.

    A ratio above 1 indicates good short-term liquidity, with 2:1 considered a healthy ratio.
    Also known as the working capital ratio.

    Args:
        current_assets (pd.Series): Time series of current assets values
        current_liabilities (pd.Series): Time series of current liabilities values

    Returns:
        pd.Series: Time series of current ratio values. Returns NaN for periods where
                  current liabilities are zero.
    """
    # Handle zero liabilities by replacing with NaN
    safe_liabilities = current_liabilities.replace(0, np.nan)
    return current_assets / safe_liabilities

# ---------------------------
# 3. Operational Efficiency
# ---------------------------

def get_cash_conversion_cycle(
        inventory: pd.Series,
        cogs: pd.Series,
        accounts_receivable: pd.Series,
        revenue: pd.Series,
        accounts_payable: pd.Series,
        days: Union[int, float] = 365) -> pd.Series:
    """
    Calculate the Cash Conversion Cycle (CCC), which measures the time (in days) it takes 
    for a company to convert investments in inventory and other resources into cash flows 
    from sales.

    A lower number of days indicates better efficiency in managing working capital.
    CCC = DIO + DSO - DPO, where:
    - DIO: Days Inventory Outstanding
    - DSO: Days Sales Outstanding
    - DPO: Days Payables Outstanding

    Args:
        inventory (pd.Series): Time series of inventory balances
        cogs (pd.Series): Time series of Cost of Goods Sold values
        accounts_receivable (pd.Series): Time series of accounts receivable balances
        revenue (pd.Series): Time series of revenue values
        accounts_payable (pd.Series): Time series of accounts payable balances
        days (Union[int, float], optional): Number of days in period. Defaults to 365.

    Returns:
        pd.Series: Time series of Cash Conversion Cycle values in days. Returns NaN for
                  periods where any denominator (COGS or revenue) is zero.
    """
    # Handle zero denominators by replacing with NaN
    safe_cogs = cogs.replace(0, np.nan)
    safe_revenue = revenue.replace(0, np.nan)

    days_inventory_outstanding = (inventory / safe_cogs) * days
    days_sales_outstanding = (accounts_receivable / safe_revenue) * days
    days_payables_outstanding = (accounts_payable / safe_cogs) * days

    return days_inventory_outstanding + days_sales_outstanding - days_payables_outstanding

# ----------------------
# 4. Bankruptcy Risk
# ----------------------

def get_altman_z_score(
        current_assets: pd.Series,
        current_liabilities: pd.Series,
        total_assets: pd.Series,
        ebit: pd.Series,
        diluted_shares_outstanding: pd.Series,
        revenue: pd.Series,
        total_liabilities: pd.Series,
        retained_earnings: pd.Series,
        stock_price: pd.Series) -> pd.Series:
    """
    Calculate the Altman Z-Score, a financial metric that predicts the probability of 
    a company going bankrupt within two years.

    The Z-Score uses multiple corporate income and balance sheet values to measure 
    the financial health of a company.

    Formula: Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    where:
    - A = Working Capital / Total Assets
    - B = Retained Earnings / Total Assets
    - C = EBIT / Total Assets
    - D = Market Value of Equity / Total Liabilities
    - E = Sales / Total Assets

    Interpretation:
    - Z > 2.99: "Safe" Zone - Low probability of bankruptcy
    - 1.81 < Z < 2.99: "Grey" Zone - Medium probability of bankruptcy
    - Z < 1.81: "Distress" Zone - High probability of bankruptcy

    Args:
        current_assets (pd.Series): Time series of current assets values
        current_liabilities (pd.Series): Time series of current liabilities values
        total_assets (pd.Series): Time series of total assets values
        ebit (pd.Series): Time series of Earnings Before Interest and Taxes values
        diluted_shares_outstanding (pd.Series): Time series of diluted shares outstanding
        revenue (pd.Series): Time series of revenue values
        total_liabilities (pd.Series): Time series of total liabilities values
        retained_earnings (pd.Series): Time series of retained earnings values
        stock_price (pd.Series): Time series of stock price values

    Returns:
        pd.Series: Time series of Altman Z-Score values. Returns NaN for periods where
                  any denominator (total assets or total liabilities) is zero.
    """
    # Handle zero denominators by replacing with NaN
    safe_assets = total_assets.replace(0, np.nan)
    safe_liabilities = total_liabilities.replace(0, np.nan)

    x_1 = (current_assets - current_liabilities) / safe_assets  # Working Capital ratio
    x_2 = retained_earnings / safe_assets  # Retained Earnings ratio
    x_3 = ebit / safe_assets  # Profitability ratio
    x_4 = (stock_price * diluted_shares_outstanding) / safe_liabilities  # Solvency ratio
    x_5 = revenue / safe_assets  # Asset Turnover ratio

    return (1.2 * x_1) + (1.4 * x_2) + (3.3 * x_3) + (0.6 * x_4) + (1.0 * x_5)
