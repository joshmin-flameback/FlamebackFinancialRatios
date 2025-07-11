"""Helpers Module"""

import inspect
import time
import pandas as pd
import warnings
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Union, Optional, Callable

import numpy as np
import pandas as pd
RETRY_LIMIT = 12

# pylint: disable=comparison-with-itself,too-many-locals



def calculate_growth(
    dataset: pd.Series | pd.DataFrame,
    lag: int | list[int] = 1,
    rounding: int | None = 4,
    axis: str | int = 0,
) -> pd.Series | pd.DataFrame:
    """
    Calculates directional growth using absolute value of previous value as base.
    This avoids misleading % changes when previous value is negative.

    Args:
        dataset (pd.Series | pd.DataFrame): Input time series data.
        lag (int | list[int]): Lag interval(s) to compute growth over.
        rounding (int | None): Decimal rounding for output.
        axis (str | int): Axis for DataFrame abs_pct_change. Default is 0 (index).

    Returns:
        pd.Series | pd.DataFrame: Growth values.
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)

    def abs_pct_change(x, periods=1):
        prev = x.shift(periods)
        return (x - prev) / prev.abs()

    if isinstance(dataset, pd.Series):
        if isinstance(lag, list):
            return pd.concat(
                [abs_pct_change(dataset, l).round(rounding).rename(f"Lag {l}") for l in lag],
                axis=1
            )
        return abs_pct_change(dataset, lag).round(rounding)

    if isinstance(lag, list):
        result = {}
        for l in lag:
            if axis == 1:
                growth = dataset.T.pipe(abs_pct_change, periods=l).T
            else:
                growth = dataset.pipe(abs_pct_change, periods=l)
            result[f"Lag {l}"] = growth.round(rounding)
        return pd.concat(result, axis=1)

    # Single lag
    if axis == 1:
        return dataset.T.pipe(abs_pct_change, periods=lag).T.round(rounding)
    else:
        return dataset.pipe(abs_pct_change, periods=lag).round(rounding)



def calculate_average(
    dataset: pd.Series,
    growth: bool = False,
    trailing: int | None = 20,
    min_periods: int | None = 1,
    rounding: int | None = 4
) -> pd.Series | pd.DataFrame:
    """
    Calculate the average growth over a trailing period for any financial metric.
    
    Args:
        dataset (pd.Series): The time series data to calculate growth for
        growth: (bool) , growth should be calculated or not. Defaults to False.
        trailing (int, optional): Number of periods to calculate the trailing average over. 
            Defaults to 20.
        rounding (int, optional): Number of decimal places to round to.
            Defaults to 4.
    
    Returns:
        pd.DataFrame: DataFrame containing the average growth rates
        
    """
    if growth:
        # Calculate period-over-period growth
        dataset = calculate_growth(dataset, axis=0)
        # handle infinite values by replacing with NaN
        dataset = dataset.replace([float('inf'), float('-inf')], pd.NA)
        # convert the dataset, coercing invalid values like <NA> or 'NaN' into actual np.nan.
        dataset = pd.to_numeric(dataset, errors='coerce')

    # Calculate trailing average of growth rates
    if trailing:
        result = dataset.rolling(window=trailing, min_periods=min_periods).mean()
    else:
        result = dataset
        
    return result.round(rounding)

def get_consecutive_number_of_growth(dataset: pd.Series, period: int = 20) -> pd.Series:
    dataset = dataset.sort_index()  # Ensure time series is sorted
    growth = calculate_growth(dataset, lag=1)

    results = pd.Series(index=dataset.index, dtype=int)

    for i in range(len(dataset)):
        start = max(0, i - period)
        # Lookback last `period` quarters
        growth_window = growth.iloc[start:i]

        # If no data yet (e.g., first element), skip
        if growth_window.empty:
            continue
        # Create boolean Series for positive growth
        growth_streak = growth_window > 0

        max_consecutive_growth = 0
        current_streak = 0

        for is_growth in growth_streak:
            if is_growth:
                current_streak += 1
                max_consecutive_growth = max(max_consecutive_growth, current_streak)
            else:
                current_streak = 0

        # Store max streak for the quarter we are evaluating
        results.iloc[i] = max_consecutive_growth

    return results




class FrequencySelector:
    """
    A utility class that adds frequency selection methods to pandas Series and DataFrames.
    
    This allows for easy extraction of annual data or TTM (Trailing Twelve Months) calculations
    directly from financial time series data using dot notation.
    
    Example:
        equity_data = financial_data['Total Equity']
        annual_equity = equity_data.freq.FY(exchange='NSE')  # Get annual data (fiscal year end values)
        ttm_equity = equity_data.freq.TTM  # Get TTM calculation
    """
    
    def __init__(self, obj):
        self._obj = obj
        

        
    def FY(self, exchange='NSE'):
        """
        Calculates the fiscal year sum for each date based on the exchange's fiscal year end date.
        
        For Indian exchanges (NSE, BSE): April 1st to March 31st
        For US exchanges (NYSE, NASDAQ): January 1st to December 31st
        
        For dates that are fiscal year-end dates, calculates the sum of all values
        in that fiscal year. For other dates, finds the most recent fiscal year sum.
        
        Args:
            exchange (str): Stock exchange code to determine fiscal year end date
                           (e.g., 'NSE', 'BSE', 'NYSE', 'NASDAQ')
        
        Returns:
            pd.Series or pd.DataFrame: Financial year sums for each date
        """
        if isinstance(self._obj, pd.Series) or isinstance(self._obj, pd.DataFrame):
            # Get fiscal year end month and day based on exchange
            fy_end_month, fy_end_day = get_fiscal_year_end(exchange)
            original_index = self._obj.index
            # Convert index to datetime if it's not already
            if not isinstance(self._obj.index, pd.DatetimeIndex):
                obj = self._obj.copy()
                obj.index = pd.to_datetime(obj.index)
            else:
                obj = self._obj
            
            # Ensure data is sorted by date
            obj = obj.sort_index()
            
            # Get all dates from the original data
            all_dates = obj.index
            
            # Create a result container with the same index as original
            if isinstance(obj, pd.Series):
                result = pd.Series(index=all_dates, dtype=obj.dtype)
            else:  # DataFrame
                result = pd.DataFrame(index=all_dates, columns=obj.columns)
            
            # Find all fiscal year ends based on exchange
            fiscal_year_ends = obj.index[(obj.index.month == fy_end_month) & (obj.index.day == fy_end_day)]
            
            # Calculate fiscal year sums for each fiscal year end
            fiscal_year_sums = {}
            
            for fy_end in fiscal_year_ends:
                # Calculate fiscal year start
                if fy_end_month == 3:  # Indian fiscal year (April 1st to March 31st)
                    fy_start = pd.Timestamp(year=fy_end.year-1, month=4, day=1)
                elif fy_end_month == 12:  # US fiscal year (January 1st to December 31st)
                    fy_start = pd.Timestamp(year=fy_end.year, month=1, day=1)
                
                # Get data for this fiscal year
                fy_mask = (obj.index >= fy_start) & (obj.index <= fy_end)
                fy_data = obj[fy_mask]
                
                # Calculate sum for this fiscal year
                if isinstance(obj, pd.Series):
                    fiscal_year_sums[fy_end] = fy_data.sum()
                else:  # DataFrame
                    fiscal_year_sums[fy_end] = fy_data.sum()
            
            # Assign fiscal year sums to result
            for date in all_dates:
                # Check if this date is a fiscal year end
                if date in fiscal_year_ends:
                    if isinstance(result, pd.Series):
                        result[date] = fiscal_year_sums[date]
                    else:  # DataFrame
                        result.loc[date] = fiscal_year_sums[date]
                else:
                    # Find the most recent fiscal year end
                    previous_fy_ends = fiscal_year_ends[fiscal_year_ends <= date]
                    if len(previous_fy_ends) > 0:
                        most_recent_fy_end = previous_fy_ends[-1]
                        if isinstance(result, pd.Series):
                            result[date] = fiscal_year_sums[most_recent_fy_end]
                        else:  # DataFrame
                            result.loc[date] = fiscal_year_sums[most_recent_fy_end]
            
            # Remove NaN values
            if isinstance(result, pd.Series):
                result = result.dropna()
            else:  # DataFrame
                result = result.dropna(how='all')

            if not isinstance(original_index, pd.DatetimeIndex):
                result.index = result.index.date
            return result
        return self._obj
        

    
    @property
    def TTM(self):
        """
        Calculates Trailing Twelve Months (TTM) values for all dates in the timeseries.
        
        For each date, calculates the sum of values from the previous 365 days.
        This is typically used for income statement and cash flow metrics.
        
        Returns:
            pd.Series or pd.DataFrame: TTM calculations for each date with available data
        """
        if isinstance(self._obj, pd.Series) or isinstance(self._obj, pd.DataFrame):
            original_index = self._obj.index
            # Convert index to datetime if it's not already
            if not isinstance(self._obj.index, pd.DatetimeIndex):
                obj = self._obj.copy()
                obj.index = pd.to_datetime(obj.index)
            else:
                obj = self._obj
                
            # Ensure data is sorted by date
            obj = obj.sort_index()
            
            # Get all dates from the original data
            all_dates = obj.index
            
            # Handle differently based on data type
            if isinstance(obj, pd.Series):
                result = pd.Series(index=all_dates, dtype=obj.dtype)
                
                # For each date, calculate TTM by summing values from previous year
                for i, current_date in enumerate(all_dates):
                    # Define the TTM period start date (1 year back)
                    ttm_start_date = current_date - pd.Timedelta(days=365)
                    
                    # Get data points within the TTM period
                    mask = (obj.index > ttm_start_date) & (obj.index <= current_date)
                    ttm_period_data = obj[mask]
                    
                    # Calculate TTM value if we have data points
                    if len(ttm_period_data) > 0:
                        # Sum the values for the trailing 12 months
                        result[current_date] = ttm_period_data.sum()
                
                # Remove NaN values
                result = result.dropna()
                
            else:  # DataFrame
                result = pd.DataFrame(index=all_dates, columns=obj.columns)
                
                # For each date, calculate TTM by summing values from previous year
                for current_date in all_dates:
                    # Define the TTM period start date (1 year back)
                    ttm_start_date = current_date - pd.Timedelta(days=365)
                    
                    # Get data points within the TTM period
                    mask = (obj.index > ttm_start_date) & (obj.index <= current_date)
                    ttm_period_data = obj[mask]
                    
                    # Calculate TTM values if we have data points
                    if len(ttm_period_data) > 0:
                        # Sum the values for the trailing 12 months for each column
                        for column in obj.columns:
                            result.loc[current_date, column] = ttm_period_data[column].sum()
                
                # Remove rows with all NaN values
                result = result.dropna(how='all')

            if not isinstance(original_index, pd.DatetimeIndex):
                result.index = result.index.date
            return result
        return self._obj
        
    def TTM_with_periods(self, periods=4):
        """
        Calculates Trailing Twelve Months (TTM) or other trailing periods based on number of periods.
        
        For each date, calculates the sum of the specified number of most recent periods.
        This is more flexible than the TTM property which uses 365 days.
        
        Args:
            periods: Number of periods to include in the trailing calculation (default 4 for TTM with quarterly data)
            
        Returns:
            pd.Series or pd.DataFrame: Trailing calculations for each date with sufficient history
        """
        if isinstance(self._obj, pd.Series) or isinstance(self._obj, pd.DataFrame):
            original_index = self._obj.index
            # Convert index to datetime if it's not already
            if not isinstance(self._obj.index, pd.DatetimeIndex):
                obj = self._obj.copy()
                obj.index = pd.to_datetime(obj.index)
            else:
                obj = self._obj
                
            # Sort by date
            obj = obj.sort_index()
            
            # Handle differently based on data type:
            if isinstance(obj, pd.Series):
                # For Series: Calculate rolling sum with the specified window
                result = obj.rolling(window=periods, min_periods=periods).sum()
                
                # Filter out NaN values from result
                result = result.dropna()
                
            elif isinstance(obj, pd.DataFrame):
                # For DataFrames: Apply the same logic column by column
                result = pd.DataFrame(index=obj.index)
                
                for column in obj.columns:
                    col_data = obj[column].rolling(window=periods, min_periods=periods).sum()
                    result[column] = col_data
                
                # Filter out rows where all values are NaN
                result = result.dropna(how='all')
            if not isinstance(original_index, pd.DatetimeIndex):
                result.index = result.index.date
            return result
        return self._obj
    
    @property
    def fiscal_year(self):
        """
        Groups data by fiscal year (April 1st to March 31st).
        
        Returns:
            dict: Dictionary with fiscal year as key and corresponding data as value
        """
        if isinstance(self._obj, pd.Series) or isinstance(self._obj, pd.DataFrame):
            # Convert index to datetime if it's not already
            if not isinstance(self._obj.index, pd.DatetimeIndex):
                obj = self._obj.copy()
                obj.index = pd.to_datetime(obj.index)
            else:
                obj = self._obj
            
            # Group by fiscal years
            result = {}
            
            # Sort by date
            obj = obj.sort_index()
            
            for date, value in obj.iteritems() if isinstance(obj, pd.Series) else obj.iterrows():
                # Determine fiscal year: April 1st to March 31st
                # Fiscal year is named by the year it ends in
                fiscal_year = date.year if date.month >= 4 else date.year - 1
                fiscal_year_end = f"FY{fiscal_year+1}"
                
                if fiscal_year_end not in result:
                    result[fiscal_year_end] = []
                
                result[fiscal_year_end].append((date, value))
            
            return result
        return self._obj
    

    
    @property
    def latest_TTM(self):
        """
        Returns the most recent TTM calculation.
        
        Returns:
            The most recent TTM calculation
        """
        ttm_data = self.TTM
        if len(ttm_data) > 0:
            return ttm_data.iloc[-1]
        return None


# Register the accessor with pandas
# Register the accessor with pandas
pd.api.extensions.register_series_accessor("freq")(FrequencySelector)
pd.api.extensions.register_dataframe_accessor("freq")(FrequencySelector)


def get_fiscal_year_end(exchange):
    """
    Determine fiscal year end month and day based on exchange.
    
    Args:
        exchange (str): Stock exchange code (e.g., 'NSE', 'BSE', 'NYSE', 'NASDAQ')
        
    Returns:
        tuple: (month, day) tuple representing fiscal year end date
    """
    # US exchanges (NYSE, NASDAQ) use calendar year (ending December 31)
    if exchange in ['NYSE', 'NASDAQ']:
        return (12, 31)  # December 31
    # Indian exchanges (NSE, BSE) and others use fiscal year ending March 31
    else:
        return (3, 31)   # March 31

class FrequencyType(Enum):
    """Frequency types for financial data calculations."""
    FY = auto()    # Financial Year (varies by exchange: Apr-Mar for Indian, Jan-Dec for US)
    TTM = auto()   # Trailing Twelve Months


# Create a global instance for easy access
freq = FrequencyType





def get_ttm_data(data):
    """
    Calculate Trailing Twelve Months (TTM) values.
    
    Args:
        data: pandas Series or DataFrame with datetime index, typically quarterly data
        
    Returns:
        TTM calculations (sum of last 4 quarters for each date)
    """
    return FrequencySelector(data).TTM





def get_latest_ttm(data):
    """
    Get the most recent TTM calculation.
    
    Args:
        data: pandas Series or DataFrame with datetime index
        
    Returns:
        Most recent TTM calculation
    """
    return FrequencySelector(data).latest_TTM


def handle_errors(func):
    """
    Decorator to handle specific errors that may occur in a function and provide informative messages.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Raises:
        KeyError: If an index name is missing in the provided financial statements.
        ValueError: If an error occurs while running the function, typically due to incomplete financial statements.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            function_name = func.__name__
            print(
                "There is an index name missing in the provided financial statements. "
                f"This is {e}. This is required for the function ({function_name}) "
                "to run. Please fill this column to be able to calculate the ratios."
            )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to a division by zero."
            )
            return pd.Series(dtype="object")
        except IndexError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to missing data."
            )
            return pd.Series(dtype="object")

    # These steps are there to ensure the docstring of the function remains intact
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__module__ = func.__module__

    return wrapper
