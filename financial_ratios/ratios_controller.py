"""Ratios Module"""

import pandas as pd

from financial_ratios.utils.helpers import calculate_growth, handle_errors, calculate_average
from . import financial_health_model, earnings_model, quality_model, valuation_model


class Ratios:
    """
    The Ratios Module contains financial health ratios that can be used to analyse companies.
    These ratios focus on assessing a company's financial health, solvency, and risk of bankruptcy.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical: pd.DataFrame,
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
    ):
        """
        Initializes the Ratios Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculations.
            historical (pd.DataFrame): The historical data to use for the calculations.
            balance (pd.DataFrame): The balance sheet data to use for the calculations.
            income (pd.DataFrame): The income statement data to use for the calculations.
            cash (pd.DataFrame): The cash flow statement data to use for the calculations.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
        """
        self._tickers = tickers
        
        self._balance_sheet_statement = balance
        self._income_statement = income
        self._cash_flow_statement = cash
        self._historical_data = historical
        self._rounding = rounding
        self._quarterly = quarterly
        
        # Initialize financial health ratios
        self._financial_health_ratios = pd.DataFrame()
        self._financial_health_ratios_growth = pd.DataFrame()
        self._earning_ratios = pd.DataFrame()
        self._earning_ratios_growth = pd.DataFrame()
        self._quality_ratios = pd.DataFrame()
        self._quality_ratios_growth = pd.DataFrame()
        self._valuation_ratios = pd.DataFrame()
        self._valuation_ratios_growth = pd.DataFrame()

    def _validate_data(self, *series_list: pd.Series) -> tuple[pd.Series, ...]:
        """
        Validates and aligns multiple pandas Series to ensure they have matching dates.
        
        Args:
            *series_list: Variable number of pandas Series to validate
            
        Returns:
            tuple[pd.Series, ...]: Tuple of aligned Series with only dates present in all Series
        """
        # Find common dates across all series
        common_dates = None
        for series in series_list:
            if series is None:
                continue
            dates = series.dropna().index
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates &= set(dates)
        
        if not common_dates:
            return tuple(pd.Series(dtype=float) for _ in series_list)  # Return empty series if no common dates
            
        # Filter each series to only include common dates
        aligned_series = []
        for series in series_list:
            if series is None:
                aligned_series.append(pd.Series(dtype=float))
            else:
                aligned_series.append(series[list(common_dates)].sort_index())
                
        return tuple(aligned_series)

    def _process_ratio_result(
        self,
        result: pd.DataFrame,
        growth: bool = False,
        lag: int | list[int] = 1,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Handle common post-processing for ratio calculations.

        Args:
            result (pd.DataFrame): The ratio calculation result to process
            growth (bool): Whether to calculate growth rates
            lag (int | list[int]): Lag periods for growth calculation
            rounding (int | None): Number of decimal places for rounding

        Returns:
            pd.DataFrame: Processed ratio results
        """
        if growth:
            result = calculate_growth(
                result,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )
        else:
            result = result.round(rounding if rounding else self._rounding)
        return result


    ################ Financial Health Model Ratios ###############

    @handle_errors
    def collect_financial_health_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        days: int | float | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Financial Health Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Financial health ratios calculated based on the specified parameters.
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365
        
        # Calculate all financial health ratios
        debt_equity = self.get_debt_to_equity_ratio(trailing=trailing)
        interest_coverage = self.get_interest_coverage_ratio(trailing=trailing)
        current_ratio = self.get_current_ratio(trailing=trailing)
        cash_conversion = self.get_cash_conversion_cycle(trailing=trailing)
        altman_z = self.get_altman_z_score(trailing=trailing)
        self._financial_health_ratios = pd.concat([debt_equity, interest_coverage, current_ratio, cash_conversion,altman_z], axis=1)
        
        # Process and return the results
        return self._process_ratio_result(self._financial_health_ratios, growth, lag, rounding)

    @handle_errors
    def get_debt_to_equity_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the debt to equity ratio, which measures the proportion of
        a company's equity that is financed by debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Debt to equity ratio values.
        """
        # Get series for each financial item where full_name matches
        long_term_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Long-Term Debt']['value']
        short_term_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Short-Term Debt']['value']
        total_debt = long_term_debt + short_term_debt
        total_equity = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Equity']['value']

        if trailing:
            long_term_debt = long_term_debt.T.rolling(trailing).mean().T
            short_term_debt = short_term_debt.T.rolling(trailing).mean().T
            total_debt = long_term_debt + short_term_debt
            total_equity = total_equity.T.rolling(trailing).mean().T

        # Calculate ratio and convert to DataFrame
        result = financial_health_model.get_debt_to_equity_ratio(total_debt, total_equity)
        result_df = result.to_frame(name='Debt to Equity')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the interest coverage ratio, which measures a company's
        ability to pay its interest expenses on outstanding debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Interest coverage ratio values.
        """

        ebitda = self._income_statement[self._income_statement['full_name'] == 'EBITDA']['value']
        interest_expense = self._income_statement[self._income_statement['full_name'] == 'Interest Expense']['value']

        if trailing:
            ebitda = ebitda.T.rolling(trailing).sum().T
            interest_expense = interest_expense.T.rolling(trailing).sum().T

        result = financial_health_model.get_interest_coverage_ratio(ebitda, interest_expense)
        result_df = result.to_frame(name='TTM Interest Coverage')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_current_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the current ratio, which measures a company's ability to pay
        its short-term liabilities with its current assets.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Current ratio values.
        """
        current_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Current Assets']['value']
        current_liabilities = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Current Liabilities']['value']

        if trailing:
            current_assets = current_assets.T.rolling(trailing).mean().T
            current_liabilities = current_liabilities.T.rolling(trailing).mean().T

        result = financial_health_model.get_current_ratio(current_assets, current_liabilities)
        result_df = result.to_frame(name='Current Ratio')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cash_conversion_cycle(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        days: int | float | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Cash Conversion Cycle, which measures how long it takes
        to convert investments in inventory into cash flows from sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Cash conversion cycle values.
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        inventory = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Inventories']['value']
        cogs = self._income_statement[self._income_statement['full_name'] == 'Cost of Goods Sold']['value']
        accounts_receivable = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Accounts Receivable']['value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        accounts_payable = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Accounts Payable']['value']

        if trailing:
            inventory = inventory.T.rolling(trailing).mean().T
            cogs = cogs.T.rolling(trailing).sum().T
            accounts_receivable = accounts_receivable.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).sum().T
            accounts_payable = accounts_payable.T.rolling(trailing).mean().T

        result = financial_health_model.get_cash_conversion_cycle(
            inventory, cogs, accounts_receivable, revenue, accounts_payable, days
        )
        result_df = result.to_frame(name='Cash Conversion Cycle')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_altman_z_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Altman Z-Score, which predicts the probability that a company
        will go bankrupt within two years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Altman Z-Score values.
        """
        current_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Current Assets']['value']
        current_liabilities = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Current Liabilities']['value']
        total_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Assets']['value']
        ebit = self._income_statement[self._income_statement['full_name'] == 'EBIT']['value']
        diluted_shares = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding (Diluted Average)']['value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        total_liabilities = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Liabilities']['value']
        retained_earnings = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Retained Earnings']['value']
        stock_price = self._historical_data['close']

        if trailing:
            current_assets = current_assets.T.rolling(trailing).mean().T
            current_liabilities = current_liabilities.T.rolling(trailing).mean().T
            total_assets = total_assets.T.rolling(trailing).mean().T
            ebit = ebit.T.rolling(trailing).sum().T
            diluted_shares = diluted_shares.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).sum().T
            total_liabilities = total_liabilities.T.rolling(trailing).mean().T
            retained_earnings = retained_earnings.T.rolling(trailing).mean().T

        result = financial_health_model.get_altman_z_score(
            current_assets, current_liabilities, total_assets,
            ebit, diluted_shares, revenue, total_liabilities, retained_earnings, stock_price
        )
        result_df = result.to_frame(name='Altman Z-Score')
        return self._process_ratio_result(result_df, growth, lag, rounding)


    ################ Earnings Model Ratios ###############

    @handle_errors
    def collect_earning_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Earnings-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Earnings ratios calculated based on the specified parameters.
        """
        # Calculate all earnings ratios
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        eps_basic = self._income_statement[self._income_statement['full_name'] == 'Basic EPS']['value']
        gross_margin = self._income_statement[self._income_statement['full_name'] == 'Gross Margin %']['value']
        ebitda = self._income_statement[self._income_statement['full_name'] == 'EBITDA']['value']

        piotroski_score = self.get_piotroski_score_ratio(trailing=trailing)
        revenue_growth = calculate_growth(revenue, lag=1)
        eps_growth = calculate_growth(eps_basic, lag=1)
        consecutive_revenue_growth = self.get_consecutive_number_of_growth_ratio(revenue, period=20)
        avg_revenue_growth = calculate_average(revenue, growth=True, trailing=20)
        avg_gross_margin_growth = calculate_average(gross_margin, growth=True, trailing=20)
        avg_gross_margin = calculate_average(gross_margin, trailing=20)
        avg_ebitda_growth = calculate_average(ebitda, growth=True, trailing=20)
        avg_ebitda = calculate_average(ebitda, trailing=20)
        avg_eps_growth = calculate_average(eps_basic, growth=True, trailing=20)
        consecutive_eps_growth = self.get_consecutive_number_of_growth_ratio(eps_basic, period=20)


        ebitda_margin = self.get_ebitda_margin_ratio(trailing=trailing)
        roe = self.get_roe_ratio(trailing=trailing)
        fcf_growth = self.get_fcf_growth_ratio(trailing=trailing)

        # Combine all ratios
        self._earning_ratios = pd.concat([
             piotroski_score, revenue_growth, eps_growth, ebitda_margin, roe, fcf_growth
        ], axis=1)

        # Process and return the results
        return self._process_ratio_result(self._earning_ratios, growth, lag, rounding)

    @handle_errors
    def get_piotroski_score_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Piotroski F-Score, a comprehensive scoring system that evaluates
        a company's financial health across 9 criteria.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Piotroski F-Score values ranging from 0-9
        """
        # Get required metrics
        net_income = self._income_statement[self._income_statement['full_name'] == 'Net Income']['value']
        operating_cash_flow = \
        self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Cash Flow from Operations']['value']
        total_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Assets'][
            'value']
        total_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Debt']['value']
        current_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Current Assets'][
            'value']
        current_liabilities = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Current Liabilities']['value']
        shares_outstanding = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding'][
            'value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        cogs = self._income_statement[self._income_statement['full_name'] == 'Cost of Goods Sold']['value']

        # Apply trailing if specified
        if trailing:
            net_income = net_income.T.rolling(trailing).mean().T
            operating_cash_flow = operating_cash_flow.T.rolling(trailing).mean().T
            total_assets = total_assets.T.rolling(trailing).mean().T
            total_debt = total_debt.T.rolling(trailing).mean().T
            current_assets = current_assets.T.rolling(trailing).mean().T
            current_liabilities = current_liabilities.T.rolling(trailing).mean().T
            shares_outstanding = shares_outstanding.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).mean().T
            cogs = cogs.T.rolling(trailing).mean().T

        # Calculate Piotroski score using earnings model
        result = earnings_model.get_piotroski_score(
            net_income=net_income,
            total_assets=total_assets,
            cash_flow_from_operations=operating_cash_flow,
            current_assets=current_assets,
            current_liabilities=current_liabilities,
            long_term_debt=total_debt,
            shares_outstanding=shares_outstanding,
            revenue=revenue,
            cogs=cogs
        )

        # Convert to DataFrame with appropriate name
        result_df = result.to_frame(name='Piotroski F-Score')

        # Process and return the results
        return self._process_ratio_result(result_df, growth, lag, rounding)


    def get_consecutive_number_of_growth_ratio(
        self,
        dataset: pd.Series,
        period: int = 20,
        growth: bool = False,
        lag: int | list[int] = 1,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the number of consecutive periods with positive growth for a given financial metric.

        Args:
            dataset (pd.Series): The financial metric to analyze for consecutive growth
            period (int, optional): Number of periods to look back. Defaults to 20.
            growth (bool, optional): Whether to calculate growth of the ratio. Defaults to False.
            lag (int | list[int], optional): The lag to use for growth calculation. Defaults to 1.
            rounding (int | None, optional): Number of decimal places to round to. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the consecutive growth periods
        """
        result = earnings_model.get_consecutive_number_of_growth(dataset, period)
        return self._process_ratio_result(
            pd.DataFrame(result),
            growth=growth,
            lag=lag,
            rounding=rounding or self._rounding,
        )


    @handle_errors
    def get_ebitda_margin_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the EBITDA Margin ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: EBITDA margin ratio values.
        """
        ebitda = self._income_statement[self._income_statement['full_name'] == 'EBITDA']['value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']

        if trailing:
            ebitda = ebitda.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).mean().T

        result = earnings_model.calculate_ebitda_margin(ebitda, revenue)
        result_df = result.to_frame(name='EBITDA Margin')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_roe_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Return on Equity (ROE) ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: ROE ratio values.
        """
        net_income = self._income_statement[self._income_statement['full_name'] == 'Net Income']['value']
        total_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Assets'][
            'value']
        total_liabilities = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Liabilities']['value']

        if trailing:
            net_income = net_income.T.rolling(trailing).mean().T
            total_assets = total_assets.T.rolling(trailing).mean().T
            total_liabilities = total_liabilities.T.rolling(trailing).mean().T

        result = earnings_model.get_roe(net_income, total_assets, total_liabilities)
        result_df = result.to_frame(name='ROE')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Year-over-Year Free Cash Flow Growth ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: FCF growth ratio values.
        """
        fcf = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Free Cash Flow']['value']

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T

        result = earnings_model.get_annual_growth_fcf(fcf)
        result_df = result.to_frame(name='FCF Growth YoY')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    ################ Quality Model Ratios ###############

    @handle_errors
    def collect_quality_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Quality-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Quality ratios calculated based on the specified parameters.
        """
        # Calculate all quality ratios
        aicr = self.get_aicr_ratio(trailing=trailing)
        profit_dip = self.get_profit_dip_ratio(trailing=trailing)
        roic_band = self.get_roic_band_ratio(trailing=trailing)
        cfo_band = self.get_cfo_band_ratio(trailing=trailing)
        fcf_dip = self.get_fcf_dip_ratio(trailing=trailing)
        negative_fcf = self.get_negative_fcf_ratio(trailing=trailing)
        fcf_profit_band = self.get_fcf_profit_band_ratio(trailing=trailing)

        # Combine all ratios
        self._quality_ratios = pd.concat([
            aicr, profit_dip, roic_band, cfo_band,
            fcf_dip, negative_fcf, fcf_profit_band
        ], axis=1)

        # Process and return the results
        return self._process_ratio_result(self._quality_ratios, growth, lag, rounding)

    @handle_errors
    def get_aicr_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Annual Intrinsic Compounding Rate (AICR) ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: AICR ratio values.
        """
        net_income = self._income_statement[self._income_statement['full_name'] == 'Net Income']['value']
        total_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Assets'][
            'value']
        total_liabilities = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Liabilities']['value']
        dividend_paid = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Dividend Paid']['value']

        if trailing:
            net_income = net_income.T.rolling(trailing).mean().T
            total_assets = total_assets.T.rolling(trailing).mean().T
            total_liabilities = total_liabilities.T.rolling(trailing).mean().T
            dividend_paid = dividend_paid.T.rolling(trailing).mean().T

        result = quality_model.get_intrinsic_compounding_rate(net_income, total_assets, total_liabilities,
                                                              dividend_paid)
        result_df = result.to_frame(name='Annual Intrinsic Compounding Rate')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_profit_dip_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Profit Dip ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Profit dip ratio values.
        """
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        total_expense = self._income_statement[self._income_statement['full_name'] == 'Total Expense']['value']

        if trailing:
            revenue = revenue.T.rolling(trailing).mean().T
            total_expense = total_expense.T.rolling(trailing).mean().T

        result = quality_model.get_dip_profit_last10yrs(revenue, total_expense)
        result_df = result.to_frame(name='Profit Dip Last 10Y')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_roic_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the ROIC Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: ROIC band ratio values.
        """
        ebit = self._income_statement[self._income_statement['full_name'] == 'EBIT']['value']
        tax_rate = self._income_statement[self._income_statement['full_name'] == 'Tax Rate']['value']
        total_equity = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Equity'][
            'value']
        short_term_debt = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Short-Term Debt']['value']
        long_term_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Long-Term Debt'][
            'value']

        if trailing:
            ebit = ebit.T.rolling(trailing).mean().T
            tax_rate = tax_rate.T.rolling(trailing).mean().T
            total_equity = total_equity.T.rolling(trailing).mean().T
            short_term_debt = short_term_debt.T.rolling(trailing).mean().T
            long_term_debt = long_term_debt.T.rolling(trailing).mean().T

        result = quality_model.get_roic_band(ebit, tax_rate, total_equity, short_term_debt, long_term_debt)
        result_df = pd.DataFrame([result])  # Convert dict to DataFrame
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cfo_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CFO Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: CFO band ratio values.
        """
        cfo = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Cash Flow from Operations']['value']
        capex = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Capital Expenditure']['value']
        total_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Debt']['value']

        if trailing:
            cfo = cfo.T.rolling(trailing).mean().T
            capex = capex.T.rolling(trailing).mean().T
            total_debt = total_debt.T.rolling(trailing).mean().T

        # Get start and end series for debt
        total_debt_start = total_debt.shift(1)
        total_debt_end = total_debt

        result = quality_model.get_cfo_band(cfo, capex, total_debt_start, total_debt_end)
        return self._process_ratio_result(result, growth, lag, rounding)

    @handle_errors
    def get_fcf_dip_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the FCF Dip ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: FCF dip ratio values.
        """
        fcf = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Free Cash Flow']['value']

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T

        result = quality_model.get_dip_fcf_last10yrs(fcf)
        result_df = result.to_frame(name='FCF Dip Last 10Y')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_negative_fcf_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Negative FCF ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Negative FCF ratio values.
        """
        fcf = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Free Cash Flow']['value']

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T

        result = quality_model.get_no_negative_fcf_last10yrs(fcf)
        result_df = result.to_frame(name='Negative FCF Last 10Y')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_profit_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the FCF to Profit Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: FCF to profit band ratio values.
        """
        fcf = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Free Cash Flow']['value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        total_expenses = self._income_statement[self._income_statement['full_name'] == 'Total Expense']['value']

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).mean().T
            total_expenses = total_expenses.T.rolling(trailing).mean().T

        result = quality_model.get_fcf_profit_band(fcf, revenue, total_expenses)
        result_df = result.to_frame(name='FCF to Profit Band')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    ###################### Valuation Model Ratios #######################


    @handle_errors
    def collect_valuation_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Valuation-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Valuation ratios calculated based on the specified parameters.
        """
        # Calculate all valuation ratios
        steady_state = self.get_steady_state_value_ratio(trailing=trailing)
        fair_value = self.get_fair_value_ratio(trailing=trailing)
        cmp_revenue = self.get_cmp_revenue_band_ratio(trailing=trailing)
        cmp_eps = self.get_cmp_eps_band_ratio(trailing=trailing)
        cmp_cfo = self.get_cmp_cfo_band_ratio(trailing=trailing)
        fcf_yield = self.get_fcf_yield_ratio(trailing=trailing)

        # Combine all ratios
        self._valuation_ratios = pd.concat([
            steady_state, fair_value, cmp_revenue,
            cmp_eps, cmp_cfo, fcf_yield
        ], axis=1)

        # Process and return the results
        return self._process_ratio_result(self._valuation_ratios, growth, lag, rounding)

    @handle_errors
    def get_steady_state_value_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Steady State Value ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Steady State Value ratio.
        """
        eps = self._income_statement[self._income_statement['full_name'] == 'Basic EPS']['value']
        wacc = self._income_statement[self._income_statement['full_name'] == 'WACC']['value']
        current_price = self._historical_data[self._historical_data['full_name'] == 'Close']['value'].iloc[-1]

        if trailing:
            eps = eps.T.rolling(trailing).mean().T
            wacc = wacc.T.rolling(trailing).mean().T

        result = valuation_model.get_steady_state_value(eps, wacc, current_price)
        result_df = result.to_frame(name='Steady State Value')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fair_value_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Fair Value vs Current Market Price ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: Fair Value ratio values.
        """
        net_income = self._income_statement[self._income_statement['full_name'] == 'Net Income']['value']
        total_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Assets'][
            'value']
        total_liabilities = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Total Liabilities']['value']
        eps = self._income_statement[self._income_statement['full_name'] == 'Basic EPS']['value']
        eps_quarterly = eps / 4  # Convert annual to quarterly
        current_price = self._historical_data[self._historical_data['full_name'] == 'Close']['value'].iloc[-1]
        current_assets = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Current Assets'][
            'value']
        current_liabilities = \
        self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Current Liabilities']['value']
        long_term_debt = self._balance_sheet_statement[self._balance_sheet_statement['full_name'] == 'Long-Term Debt'][
            'value']
        shares_outstanding = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding'][
            'value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        cogs = self._income_statement[self._income_statement['full_name'] == 'Cost of Goods Sold']['value']

        if trailing:
            net_income = net_income.T.rolling(trailing).mean().T
            total_assets = total_assets.T.rolling(trailing).mean().T
            total_liabilities = total_liabilities.T.rolling(trailing).mean().T
            eps = eps.T.rolling(trailing).mean().T
            eps_quarterly = eps_quarterly.T.rolling(trailing).mean().T
            current_assets = current_assets.T.rolling(trailing).mean().T
            current_liabilities = current_liabilities.T.rolling(trailing).mean().T
            long_term_debt = long_term_debt.T.rolling(trailing).mean().T
            shares_outstanding = shares_outstanding.T.rolling(trailing).mean().T
            revenue = revenue.T.rolling(trailing).mean().T
            cogs = cogs.T.rolling(trailing).mean().T

        result = valuation_model.get_fair_value_vs_cmp(
            net_income, total_assets, total_liabilities, eps, eps_quarterly,
            current_price, current_assets, current_liabilities, long_term_debt,
            shares_outstanding, revenue, cogs
        )
        result_df = result.to_frame(name='Fair Value vs CMP')
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_revenue_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CMP to Revenue Band ratio for last 3 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: CMP to Revenue Band ratio values.
        """
        price = self._historical_data[self._historical_data['full_name'] == 'Close']['value']
        revenue = self._income_statement[self._income_statement['full_name'] == 'Revenue']['value']
        shares_outstanding = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding'][
            'value']
        revenue_per_share = revenue / shares_outstanding

        if trailing:
            price = price.T.rolling(trailing).mean().T
            revenue_per_share = revenue_per_share.T.rolling(trailing).mean().T

        result = valuation_model.get_cmp_revenue_band_last3yrs(price, revenue_per_share)
        result_df = pd.DataFrame([result])  # Convert dict to DataFrame
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_eps_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CMP to EPS Band ratio for last 3 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: CMP to EPS Band ratio values.
        """
        net_income = self._income_statement[self._income_statement['full_name'] == 'Net Income']['value']
        shares_outstanding = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding'][
            'value']
        time_outstanding = self._income_statement[self._income_statement['full_name'] == 'Time Outstanding']['value']
        current_price = self._historical_data[self._historical_data['full_name'] == 'Close']['value'].iloc[-1]

        if trailing:
            net_income = net_income.T.rolling(trailing).mean().T
            shares_outstanding = shares_outstanding.T.rolling(trailing).mean().T
            time_outstanding = time_outstanding.T.rolling(trailing).mean().T

        result = valuation_model.get_cmp_eps_band_last3yrs(
            net_income, shares_outstanding, time_outstanding, current_price
        )
        result_df = pd.DataFrame([result])  # Convert dict to DataFrame
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_cfo_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CMP to CFO Band ratio for last 3 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: CMP to CFO Band ratio values.
        """
        cfo = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Cash Flow from Operations']['value']

        if trailing:
            cfo = cfo.T.rolling(trailing).mean().T

        result = valuation_model.get_cmp_cfo_band_last3yrs(cfo)
        result_df = pd.DataFrame([result])  # Convert dict to DataFrame
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_yield_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Free Cash Flow Yield ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.

        Returns:
            pd.DataFrame: FCF Yield ratio values.
        """
        cfo = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Cash Flow from Operations']['value']
        capex = self._cash_flow_statement[self._cash_flow_statement['full_name'] == 'Capital Expenditure']['value']
        shares_outstanding = self._income_statement[self._income_statement['full_name'] == 'Shares Outstanding'][
            'value']
        current_price = self._historical_data[self._historical_data['full_name'] == 'Close']['value'].iloc[-1]
        market_cap = shares_outstanding * current_price

        if trailing:
            cfo = cfo.T.rolling(trailing).mean().T
            capex = capex.T.rolling(trailing).mean().T
            market_cap = market_cap.T.rolling(trailing).mean().T

        result = valuation_model.get_fcf_yield(cfo, capex, market_cap)
        result_df = result.to_frame(name='FCF Yield')
        return self._process_ratio_result(result_df, growth, lag, rounding)






