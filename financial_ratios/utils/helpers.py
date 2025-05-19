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
    axis: str = "columns",
) -> pd.Series | pd.DataFrame:
    """
    Calculates growth for a given dataset. Defaults to a lag of 1 (i.e. 1 year or 1 quarter).

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the growth values for.
        lag (int | str): the lag to use for the calculation. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    # With Pandas 2.1, pct_change will no longer automatically forward fill
    # given that this has been solved within the code already but the warning
    # still appears, this is a temporary fix to ignore the warning
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if isinstance(lag, list):
        new_index = []
        lag_dict = {f"Lag {lag_value}": lag_value for lag_value in lag}

        if axis == "columns":
            for old_index in dataset.index:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(new_index),
                columns=dataset.columns,
                dtype=np.float64,
            )

            for new_index in dataset_lag.index:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]

                dataset_lag.loc[new_index] = (
                    dataset.loc[other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                )
        else:
            for old_index in dataset.columns:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(new_index),
                index=dataset.index,
                dtype=np.float64,
            )

            for new_index in dataset_lag.columns:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]

                dataset_lag.loc[:, new_index] = (
                    dataset.loc[:, other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                )

        return dataset_lag.round(rounding)

    return dataset.ffill().pct_change(periods=lag, axis=axis).round(rounding)


def calculate_average(
    dataset: pd.Series,
    growth: bool = False,
    trailing: int | None = 20,
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
        dataset = calculate_growth(dataset)
        # handle infinite values by replacing with NaN
        dataset = dataset.replace([float('inf'), float('-inf')], pd.NA)

    # Calculate trailing average of growth rates
    if trailing:
        result = dataset.rolling(window=trailing).mean()
    else:
        result = dataset
        
    return result.round(rounding)

def get_consecutive_number_of_growth(dataset: pd.Series, period: int = 20) -> pd.Series:
    dataset = dataset.sort_index()  # Ensure time series is sorted
    growth = calculate_growth(dataset, lag=1)
    # TODO: need to test
    # growth = item.pct_change()

    results = pd.Series(index=dataset.index, dtype=int)

    for i in range(len(dataset)):
        if i < period:  # Skip initial periods where we don't have full 20 quarters
            continue

        # Lookback last `period` quarters
        growth_window = growth.iloc[i - period:i]

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
        annual_equity = equity_data.annual  # Get annual data (March 31st values)
        ttm_equity = equity_data.ttm  # Get TTM calculation
    """
    
    def __init__(self, obj):
        self._obj = obj
        

        
    @property
    def FY(self):
        """
        Calculates the fiscal year sum (April 1st to March 31st) for each date.
        
        For dates that are March 31st (fiscal year-end), calculates the sum of all values 
        in that fiscal year (from April 1st of previous year to March 31st of current year).
        For other dates, finds the most recent fiscal year sum.
        
        Returns:
            pd.Series or pd.DataFrame: Financial year sums for each date
        """
        if isinstance(self._obj, pd.Series) or isinstance(self._obj, pd.DataFrame):
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
            
            # Find all fiscal year ends (March 31st dates)
            fiscal_year_ends = obj.index[(obj.index.month == 3) & (obj.index.day == 31)]
            
            # Calculate fiscal year sums for each fiscal year end
            fiscal_year_sums = {}
            
            for fy_end in fiscal_year_ends:
                # Calculate fiscal year start (April 1st of previous year)
                fy_start = pd.Timestamp(year=fy_end.year-1, month=4, day=1)
                
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
                
                return result
            elif isinstance(obj, pd.DataFrame):
                # For DataFrames: Apply the same logic column by column
                result = pd.DataFrame(index=obj.index)
                
                for column in obj.columns:
                    col_data = obj[column].rolling(window=periods, min_periods=periods).sum()
                    result[column] = col_data
                
                # Filter out rows where all values are NaN
                result = result.dropna(how='all')
                
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


class FrequencyType(Enum):
    """Frequency types for financial data calculations."""
    FY = auto()    # Financial Year (April to March)
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
