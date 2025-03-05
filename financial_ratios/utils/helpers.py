"""Helpers Module"""

import inspect
import time
import warnings
from io import StringIO

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
