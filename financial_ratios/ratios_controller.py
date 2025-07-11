"""Ratios Module"""

import pandas as pd

from financial_ratios.utils.helpers import calculate_growth, handle_errors, calculate_average, FrequencyType, freq
from . import financial_health_model, earnings_model, quality_model, valuation_model


class Ratios:
    """
    The Ratios Module contains financial health ratios that can be used to analyse companies.
    These ratios focus on assessing a company's financial health, solvency, and risk of bankruptcy.
    """

    def __init__(
        self,
        tickers: str | list[str],
        exchange: str | list[str],
        financial_data: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
    ):
        """
        Initializes the Ratios Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculations.
            financial_data (pd.DataFrame): DataFrame containing all required financial data.
                Required columns vary by ratio but should match standard names:
                - 'Long_Term_Debt': Long term debt values
                - 'Short_Term_Debt': Short term debt values
                - 'Total_Equity': Total equity values
                - 'EBITDA': Earnings before interest, taxes, depreciation and amortization
                - 'Interest_Expense': Interest expense values
                - 'Current_Assets': Total Current Assets values
                - 'Current_Liabilities': Total Current Liabilities values
                etc...
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
        """
        self._tickers = tickers
        self._exchange = exchange
        self._financial_data = financial_data
        self._rounding = rounding
        self._quarterly = quarterly
        
        # Initialize ratio storage
        self._financial_health_ratios = pd.DataFrame()
        self._financial_health_ratios_growth = pd.DataFrame()
        self._earning_ratios = pd.DataFrame()
        self._earning_ratios_growth = pd.DataFrame()
        self._quality_ratios = pd.DataFrame()
        self._quality_ratios_growth = pd.DataFrame()
        self._valuation_ratios = pd.DataFrame()
        self._valuation_ratios_growth = pd.DataFrame()


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
        lag: int | list[int] = 1
    ) -> pd.DataFrame:
        """
        Calculates and collects all Financial Health Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Required columns in financial_data:
            - Long_Term_Debt, Short_Term_Debt, Total_Equity: For debt to equity ratio
            - EBITDA, Interest_Expense: For interest coverage ratio
            - Current_Assets, Current_Liabilities: For current ratio
            - Total Inventories, Cost_Of_Goods_Sold, Accounts_Receivable, Revenue, Accounts_Payable: For cash conversion cycle
            - Total_Assets, EBIT, Diluted_Shares_Outstanding, Total_Liabilities, Retained_Earnings, Stock_Price: For Altman Z-score

        Returns:
            pd.DataFrame: Financial health ratios calculated based on the specified parameters.
        """

        # Calculate all financial health ratios
        if self._quarterly:
            interest_coverage = self.get_interest_coverage_ratio(freq=FrequencyType.TTM)
            self._financial_health_ratios = pd.concat([interest_coverage], axis=1)
        else:
            debt_equity = self.get_debt_to_equity_ratio(freq=FrequencyType.FY)
            current_ratio = self.get_current_ratio(freq=FrequencyType.FY)
            cash_conversion = self.get_cash_conversion_cycle(freq=FrequencyType.FY)
            altman_z = self.get_altman_z_score(freq=FrequencyType.FY)
            self._financial_health_ratios = pd.concat([debt_equity, current_ratio, cash_conversion, altman_z], axis=1)

        # Process and return the results
        return self._process_ratio_result(self._financial_health_ratios, growth, lag, rounding)

    @handle_errors
    def get_debt_to_equity_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the debt to equity ratio, which measures the proportion of
        a company's equity that is financed by debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Debt to equity ratio values.
        """
        # Get required series from financial data
        total_debt = self._financial_data['Total Debt']
        total_equity = self._financial_data['Total Equity']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year (April-March) calculations
                total_debt = total_debt.freq.FY(exchange=self._exchange)
                total_equity = total_equity.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                total_debt = total_debt.freq.TTM
                total_equity = total_equity.freq.TTM
        # Apply trailing window if specified (for backward compatibility)
        if trailing:
            total_debt = total_debt.rolling(trailing).mean()
            total_equity = total_equity.rolling(trailing).mean()
        # Name based on frequency used
        ratio_name = 'Debt to Equity'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        # Calculate ratio and convert to DataFrame
        result = financial_health_model.get_debt_to_equity_ratio(total_debt, total_equity)
        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the interest coverage ratio, which measures a company's
        ability to pay its interest expenses on outstanding debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Interest coverage ratio values.
        """

        # Get required series from financial data
        ebit = self._financial_data['EBIT']
        interest_expense = self._financial_data['Interest Expense']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                ebit = ebit.freq.FY(exchange=self._exchange)
                interest_expense = interest_expense.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                ebit = ebit.freq.TTM
                interest_expense = interest_expense.freq.TTM
        # Apply trailing window if specified (for backward compatibility)
        elif trailing:
            ebit = ebit.rolling(trailing).sum()
            interest_expense = interest_expense.rolling(trailing).sum()

        result = financial_health_model.get_interest_coverage_ratio(ebit, interest_expense)

        # Name based on frequency used
        ratio_name = 'Interest Coverage'
        if freq == FrequencyType.TTM or trailing:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_current_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the current ratio, which measures a company's ability to pay
        its short-term liabilities with its Total Current Assets.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Current ratio values.
        """
        # Get required series from financial data
        current_assets = self._financial_data['Total Current Assets']
        current_liabilities = self._financial_data['Total Current Liabilities']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                current_assets = current_assets.freq.FY(exchange=self._exchange)
                current_liabilities = current_liabilities.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                current_assets = current_assets.freq.TTM
                current_liabilities = current_liabilities.freq.TTM
        # Apply trailing window if specified (for backward compatibility)
        if trailing:
            current_assets = current_assets.rolling(trailing).mean()
            current_liabilities = current_liabilities.rolling(trailing).mean()

        result = financial_health_model.get_current_ratio(current_assets, current_liabilities)

        # Name based on frequency used
        ratio_name = 'Current Ratio'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cash_conversion_cycle(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        days: int | float | None = None,
        freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Cash Conversion Cycle, which measures how long it takes
        to convert investments in inventory into cash flows from sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            days: Number of days in period for calculations.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Cash conversion cycle values.
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        # Get required series from financial data
        inventory = self._financial_data['Total Inventories']
        cogs = self._financial_data['Cost of Goods Sold']
        accounts_receivable = self._financial_data['Accounts Receivable']
        revenue = self._financial_data['Revenue']
        accounts_payable = self._financial_data['Accounts Payable']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                inventory = inventory.freq.FY(exchange=self._exchange)
                cogs = cogs.freq.FY(exchange=self._exchange)
                accounts_receivable = accounts_receivable.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
                accounts_payable = accounts_payable.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                inventory = inventory.freq.TTM
                cogs = cogs.freq.TTM
                accounts_receivable = accounts_receivable.freq.TTM
                revenue = revenue.freq.TTM
                accounts_payable = accounts_payable.freq.TTM
        # Apply trailing window if specified (for backward compatibility)
        if trailing:
            inventory = inventory.rolling(trailing).mean()
            cogs = cogs.rolling(trailing).sum()
            accounts_receivable = accounts_receivable.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).sum()
            accounts_payable = accounts_payable.rolling(trailing).mean()

        result = financial_health_model.get_cash_conversion_cycle(
            inventory, cogs, accounts_receivable, revenue, accounts_payable, days
        )

        # Name based on frequency used
        ratio_name = 'Cash Conversion Cycle'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_altman_z_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Altman Z-Score, which predicts the probability that a company
        will go bankrupt within two years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Altman Z-Score values.
        """
        # Get required series from financial data
        current_assets = self._financial_data['Total Current Assets']
        current_liabilities = self._financial_data['Total Current Liabilities']
        total_assets = self._financial_data['Total Assets']
        ebit = self._financial_data['EBIT']
        diluted_shares = self._financial_data['Shares Outstanding']
        revenue = self._financial_data['Revenue']
        total_liabilities = self._financial_data['Total Liabilities']
        retained_earnings = self._financial_data['Retained Earnings']
        stock_price = self._financial_data['Stock Price']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                current_assets = current_assets.freq.FY(exchange=self._exchange)
                current_liabilities = current_liabilities.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
                ebit = ebit.freq.FY(exchange=self._exchange)
                diluted_shares = diluted_shares.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
                total_liabilities = total_liabilities.freq.FY(exchange=self._exchange)
                retained_earnings = retained_earnings.freq.FY(exchange=self._exchange)
                # Stock price doesn't get frequency transformation
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                current_assets = current_assets.freq.TTM
                current_liabilities = current_liabilities.freq.TTM
                total_assets = total_assets.freq.TTM
                ebit = ebit.freq.TTM
                diluted_shares = diluted_shares.freq.TTM
                revenue = revenue.freq.TTM
                total_liabilities = total_liabilities.freq.TTM
                retained_earnings = retained_earnings.freq.TTM
                # Stock price doesn't get frequency transformation
        # Apply trailing window if specified (for backward compatibility)
        if trailing:
            current_assets = current_assets.rolling(trailing).mean()
            current_liabilities = current_liabilities.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            ebit = ebit.rolling(trailing).sum()
            diluted_shares = diluted_shares.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).sum()
            total_liabilities = total_liabilities.rolling(trailing).mean()
            retained_earnings = retained_earnings.rolling(trailing).mean()

        result = financial_health_model.get_altman_z_score(
            current_assets, current_liabilities, total_assets,
            ebit, diluted_shares, revenue, total_liabilities, retained_earnings, stock_price
        )

        # Name based on frequency used
        ratio_name = 'Altman Z-Score'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)


    ################ Earnings Model Ratios ###############

    @handle_errors
    def collect_earning_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            days: int | float | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Earnings-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For revenue growth and related metrics
            - Basic_EPS: For EPS growth and related metrics
            - Gross_Margin: For gross margin metrics
            - EBITDA: For EBITDA-related metrics
            - Net_Income: For profitability metrics
            - Free_Cash_Flow: For cash flow metrics
            Additional columns may be required by individual ratio calculations.

        Returns:
            pd.DataFrame: Earnings ratios calculated based on the specified parameters.
        """
        # Calculate all earnings ratios with the appropriate frequency

        if self._quarterly:
            # Basic Growth Metrics
            revenue_growth = self.get_revenue_growth_ratio()
            eps_growth = self.get_eps_growth_ratio()
            # Consecutive growth metrics
            revenue_consecutive_growth = self.get_revenue_consecutive_growth_ratio()
            eps_consecutive_growth = self.get_eps_consecutive_growth_ratio()
            # Average growth analysis
            avg_revenue_growth = self.get_average_revenue_growth_ratio()
            avg_gross_margin = self.get_average_gross_margin_ratio()
            avg_gross_margin_growth = self.get_average_gross_margin_growth_ratio()
            avg_ebitda = self.get_average_ebitda_margin_ratio()
            avg_ebitda_growth = self.get_average_ebitda_margin_growth_ratio()
            avg_eps_growth = self.get_average_eps_growth_ratio()
            # Growth comparison metrics
            revenue_growth_vs_avg = self.get_revenue_growth_vs_average_growth_ratio(freq=FrequencyType.TTM)
            eps_growth_vs_avg = self.get_eps_growth_vs_average_growth_ratio(freq=FrequencyType.TTM)
            ebitda_growth_vs_avg = self.get_ebitda_margin_vs_average_ratio(freq=FrequencyType.TTM)
            gross_margin_growth_vs_avg = self.get_gross_margin_vs_average_ratio(freq=FrequencyType.TTM)
            # Return metrics
            roe = self.get_roe_ratio(freq=FrequencyType.TTM)
            roe_vs_avg = self.get_roe_vs_average_roe_ratio(freq=FrequencyType.TTM)
            roa = self.get_return_on_assets_ratio(freq=FrequencyType.TTM)
            roa_vs_avg = self.get_roa_vs_average_roa_ratio(freq=FrequencyType.TTM)
            # Estimate Comparison Metrics
            revenue_vs_estimate = self.get_revenue_vs_estimate_ratio()
            shares_outstanding_vs_estimate = self.get_shares_outstanding_vs_estimate_ratio()
            # Combine all ratios by logical categories
            self._earning_ratios = pd.concat([
                # Basic Growth Metrics
                revenue_growth,
                eps_growth,

                # Consecutive Growth Metrics
                revenue_consecutive_growth,
                eps_consecutive_growth,

                # Average Growth Analysis
                avg_revenue_growth,
                avg_gross_margin,
                avg_gross_margin_growth,
                avg_ebitda,
                avg_ebitda_growth,
                avg_eps_growth,

                # Growth Comparison Metrics
                revenue_growth_vs_avg,
                eps_growth_vs_avg,
                ebitda_growth_vs_avg,
                gross_margin_growth_vs_avg,

                # Return Metrics
                roe,
                roe_vs_avg,
                roa,
                roa_vs_avg,

                # Estimate Comparison Metrics
                revenue_vs_estimate,
                shares_outstanding_vs_estimate,
            ], axis=1)

        else:
            # Composite Scores
            piotroski_score = self.get_piotroski_score_ratio(freq=FrequencyType.FY)
            # Cash Flow Analysis
            fcf_growth = self.get_fcf_growth_ratio(freq=FrequencyType.FY)
            fcf_avg_growth = self.get_free_cash_flow_average_growth_ratio(freq=FrequencyType.FY)
            # Combine all ratios by logical categories
            self._earning_ratios = pd.concat([
                # Composite Scores
                piotroski_score,

                # Cash Flow Analysis
                fcf_growth,
                fcf_avg_growth
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
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Piotroski F-Score, a comprehensive scoring system that evaluates
        a company's financial health across 9 criteria.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For profitability assessment
            - Operating_Cash_Flow: For cash flow assessment
            - Total_Assets: For efficiency assessment
            - Total_Debt: For leverage assessment
            - Current_Assets, Current_Liabilities: For liquidity assessment
            - Shares_Outstanding: For dilution assessment
            - Revenue, Cost_Of_Goods_Sold: For operational efficiency assessment

        Returns:
            pd.DataFrame: Piotroski F-Score values ranging from 0-9
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        operating_cash_flow = self._financial_data['Operating Cash Flow']
        total_assets = self._financial_data['Total Assets']
        total_debt = self._financial_data['Total Debt']
        current_assets = self._financial_data['Total Current Assets']
        current_liabilities = self._financial_data['Total Current Liabilities']
        shares_outstanding = self._financial_data['Shares Outstanding']
        revenue = self._financial_data['Revenue']
        cogs = self._financial_data['Cost of Goods Sold']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                operating_cash_flow = operating_cash_flow.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
                total_debt = total_debt.freq.FY(exchange=self._exchange)
                current_assets = current_assets.freq.FY(exchange=self._exchange)
                current_liabilities = current_liabilities.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
                cogs = cogs.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                operating_cash_flow = operating_cash_flow.freq.TTM
                total_assets = total_assets.freq.TTM
                total_debt = total_debt.freq.TTM
                current_assets = current_assets.freq.TTM
                current_liabilities = current_liabilities.freq.TTM
                revenue = revenue.freq.TTM
                cogs = cogs.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            operating_cash_flow = operating_cash_flow.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_debt = total_debt.rolling(trailing).mean()
            current_assets = current_assets.rolling(trailing).mean()
            current_liabilities = current_liabilities.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()
            cogs = cogs.rolling(trailing).mean()

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

        # Name based on frequency used
        ratio_name = 'Piotroski F-Score'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        # Convert to DataFrame with appropriate name
        result_df = result.to_frame(name=ratio_name)

        # Process and return the results
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_revenue_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the period-over-period revenue growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For revenue values

        Returns:
            pd.DataFrame: Revenue growth rates
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            revenue = revenue.rolling(trailing).mean()

        # Calculate revenue growth using earnings model
        result = earnings_model.get_revenue_growth(revenue)

        # Name based on frequency used
        ratio_name = 'Revenue Growth'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        # Convert to DataFrame with appropriate name
        result_df = result.to_frame(name=ratio_name)

        # Process and return the results
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_eps_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the period-over-period EPS growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For earnings per share values

        Returns:
            pd.DataFrame: EPS growth rates
        """
        # Get required series from financial data
        eps = self._financial_data['Basic EPS']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                eps = eps.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                eps = eps.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            eps = eps.rolling(trailing).mean()

        # Calculate EPS growth using earnings model
        result = earnings_model.get_eps_growth(eps)

        # Name based on frequency used
        ratio_name = 'EPS Growth'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        # Convert to DataFrame with appropriate name
        result_df = result.to_frame(name=ratio_name)

        # Process and return the results
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_roe_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Return on Equity (ROE) ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For calculating shareholders' equity

        Returns:
            pd.DataFrame: ROE ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']
        total_liabilities = self._financial_data['Total Liabilities']

        # Calculate shareholders' equity
        shareholders_equity = total_assets - total_liabilities

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                shareholders_equity = shareholders_equity.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                shareholders_equity = shareholders_equity.freq.TTM / 4


        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            shareholders_equity = shareholders_equity.rolling(trailing).mean()

        result = earnings_model.get_return_on_equity(net_income, shareholders_equity)

        # Name based on frequency used
        ratio_name = 'Return on Equity'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Year-over-Year Free Cash Flow Growth ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Free_Cash_Flow: For calculating year-over-year FCF growth

        Returns:
            pd.DataFrame: FCF growth ratio values.
        """
        # Get required series from financial data
        fcf = self._financial_data['Free Cash Flow']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                fcf = fcf.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                fcf = fcf.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            fcf = fcf.rolling(trailing).mean()

        result = earnings_model.get_free_cash_flow_growth(fcf)

        # Name based on frequency used
        ratio_name = 'FCF Growth YoY'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_revenue_consecutive_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the number of consecutive periods with positive revenue growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For revenue values

        Returns:
            pd.DataFrame: Consecutive growth periods count.
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_revenue_consecutive_growth(revenue)

        # Name based on frequency used
        ratio_name = 'Revenue Consecutive Growth Periods'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_eps_consecutive_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the number of consecutive periods with positive EPS growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For earnings per share values

        Returns:
            pd.DataFrame: Consecutive growth periods count.
        """
        # Get required series from financial data
        eps = self._financial_data['Basic EPS']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                eps = eps.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                eps = eps.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            eps = eps.rolling(trailing).mean()

        result = earnings_model.get_eps_consecutive_growth(eps)

        # Name based on frequency used
        ratio_name = 'EPS Consecutive Growth Periods'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_revenue_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average revenue growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For revenue values

        Returns:
            pd.DataFrame: Average revenue growth rates.
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_average_revenue_growth(revenue)

        # Name based on frequency used
        ratio_name = 'Average Revenue Growth (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_gross_margin_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average gross margin.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Gross_Margin: For gross margin values

        Returns:
            pd.DataFrame: Average gross margin values.
        """
        # Get required series from financial data
        gross_margin = self._financial_data['Gross Margin']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                gross_margin = gross_margin.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                gross_margin = gross_margin.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            gross_margin = gross_margin.rolling(trailing).mean()

        result = earnings_model.get_average_gross_margin(gross_margin)

        # Name based on frequency used
        ratio_name = 'Average Gross Margin (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_gross_margin_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average gross margin growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Gross_Margin: For gross margin values

        Returns:
            pd.DataFrame: Average gross margin growth rates.
        """
        # Get required series from financial data
        gross_margin = self._financial_data['Gross Margin']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                gross_margin = gross_margin.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                gross_margin = gross_margin.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            gross_margin = gross_margin.rolling(trailing).mean()

        result = earnings_model.get_average_gross_margin_growth(gross_margin)

        # Name based on frequency used
        ratio_name = 'Average Gross Margin Growth (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_ebitda_margin_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average EBITDA.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - EBITDA: For EBITDA values

        Returns:
            pd.DataFrame: Average EBITDA values.
        """
        # Get required series from financial data
        ebitda = self._financial_data['EBITDA']
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                ebitda = ebitda.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                ebitda = ebitda.freq.TTM
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            ebitda = ebitda.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_average_ebitda_margin(ebitda, revenue)

        # Name based on frequency used
        ratio_name = 'Average EBITDA Margin (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_ebitda_margin_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average EBITDA growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - EBITDA: For EBITDA values

        Returns:
            pd.DataFrame: Average EBITDA growth rates.
        """
        # Get required series from financial data
        ebitda = self._financial_data['EBITDA']
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                ebitda = ebitda.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                ebitda = ebitda.freq.TTM
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            ebitda = ebitda.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_average_ebitda_margin_growth(ebitda, revenue)

        # Name based on frequency used
        ratio_name = 'Average EBITDA Margin Growth (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_average_eps_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average EPS growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For EPS values

        Returns:
            pd.DataFrame: Average EPS growth rates.
        """
        # Get required series from financial data
        eps = self._financial_data['Basic EPS']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                eps = eps.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                eps = eps.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            eps = eps.rolling(trailing).mean()

        result = earnings_model.get_average_eps_growth(eps)

        # Name based on frequency used
        ratio_name = 'Average EPS Growth (20p)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    # Growth Comparison Metrics

    @handle_errors
    def get_revenue_growth_vs_average_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current revenue growth to its 20-period trailing average growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For revenue values

        Returns:
            pd.DataFrame: Growth ratio values (current growth / average growth).
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_revenue_growth_vs_average_growth(revenue)

        # Name based on frequency used
        ratio_name = 'Revenue Growth vs Avg Growth'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_eps_growth_vs_average_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current EPS growth to its 20-period trailing average growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For EPS values

        Returns:
            pd.DataFrame: Growth ratio values (current growth / average growth).
        """
        # Get required series from financial data
        eps = self._financial_data['Basic EPS']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                eps = eps.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                eps = eps.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            eps = eps.rolling(trailing).mean()

        result = earnings_model.get_eps_growth_vs_average_growth(eps)

        # Name based on frequency used
        ratio_name = 'EPS Growth vs Avg Growth'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_ebitda_margin_vs_average_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current EBITDA growth to its 20-period trailing average growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - EBITDA: For EBITDA values

        Returns:
            pd.DataFrame: Growth ratio values (current growth / average growth).
        """
        # Get required series from financial data
        ebitda = self._financial_data['EBITDA']
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                ebitda = ebitda.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                ebitda = ebitda.freq.TTM
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            ebitda = ebitda.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_ebitda_margin_vs_average(ebitda, revenue)

        # Name based on frequency used
        ratio_name = 'EBITDA Margin vs Avg EBITDA'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_gross_margin_vs_average_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current gross margin growth to its 20-period trailing average growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Gross_Margin: For gross margin values

        Returns:
            pd.DataFrame: Growth ratio values (current growth / average growth).
        """
        # Get required series from financial data
        gross_profit = self._financial_data['Gross Profit']
        revenue = self._financial_data['Revenue']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                gross_profit = gross_profit.freq.FY(exchange=self._exchange)
                revenue = revenue.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                gross_profit = gross_profit.freq.TTM
                revenue = revenue.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            gross_profit = gross_profit.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

        result = earnings_model.get_gross_margin_vs_average(gross_profit, revenue)

        # Name based on frequency used
        ratio_name = 'Gross Margin Growth vs Avg Growth'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    # Return Metrics

    @handle_errors
    def get_roe_vs_average_roe_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current Return on Equity (ROE) to its 20-period trailing average.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For calculating shareholders' equity

        Returns:
            pd.DataFrame: ROE ratio values (current ROE / average ROE).
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']
        total_liabilities = self._financial_data['Total Liabilities']

        # Calculate shareholders' equity
        shareholders_equity = total_assets - total_liabilities

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                shareholders_equity = shareholders_equity.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                shareholders_equity = shareholders_equity.freq.TTM / 4

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()

        result = earnings_model.get_roe_vs_average_roe(net_income, shareholders_equity)

        # Name based on frequency used
        ratio_name = 'ROE vs Average ROE'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_return_on_assets_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate Return on Assets (ROA).

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets

        Returns:
            pd.DataFrame: ROA ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                total_assets = total_assets.freq.TTM / 4

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()

        result = earnings_model.get_return_on_assets(net_income, total_assets)

        # Name based on frequency used
        ratio_name = 'Return on Assets'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_roa_vs_average_roa_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of current Return on Assets (ROA) to its 20-period trailing average.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets

        Returns:
            pd.DataFrame: ROA ratio values (current ROA / average ROA).
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                total_assets = total_assets.freq.TTM / 4

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()

        result = earnings_model.get_roa_vs_average_roa(net_income, total_assets)

        # Name based on frequency used
        ratio_name = 'ROA vs Average ROA'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    # Estimate Comparison Metrics

    @handle_errors
    def get_revenue_vs_estimate_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of actual revenue to estimated revenue.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For actual revenue values
            - Revenue_Estimate: For estimated revenue values

        Returns:
            pd.DataFrame: Ratio of actual to estimated revenue.
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']
        revenue_estimate = self._financial_data['Revenue Estimate']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
                revenue_estimate = revenue_estimate.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM
                revenue_estimate = revenue_estimate.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            revenue = revenue.rolling(trailing).mean()
            revenue_estimate = revenue_estimate.rolling(trailing).mean()

        result = earnings_model.get_revenue_vs_estimate(revenue, revenue_estimate)

        # Name based on frequency used
        ratio_name = 'Revenue vs Estimate'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_shares_outstanding_vs_estimate_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ratio of actual shares outstanding to estimated shares outstanding.
        Shares outstanding is derived from net income / EPS.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For actual net income
            - Basic_EPS: For actual EPS
            - Net_Income_Estimate: For estimated net income
            - EPS_Estimate: For estimated EPS

        Returns:
            pd.DataFrame: Ratio of actual to estimated shares outstanding.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        eps = self._financial_data['Basic EPS']
        net_income_estimate = self._financial_data['Net Income Estimate']
        eps_estimate = self._financial_data['EPS Estimate']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                eps = eps.freq.FY(exchange=self._exchange)
                net_income_estimate = net_income_estimate.freq.FY(exchange=self._exchange)
                eps_estimate = eps_estimate.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                eps = eps.freq.TTM
                net_income_estimate = net_income_estimate.freq.TTM
                eps_estimate = eps_estimate.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            eps = eps.rolling(trailing).mean()
            net_income_estimate = net_income_estimate.rolling(trailing).mean()
            eps_estimate = eps_estimate.rolling(trailing).mean()

        result = earnings_model.get_shares_outstanding_vs_estimate(
            net_income, eps, net_income_estimate, eps_estimate
        )

        # Name based on frequency used
        ratio_name = 'Shares Outstanding vs Estimate'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    # Cash Flow Analysis

    @handle_errors
    def get_free_cash_flow_average_growth_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the 20-period trailing average free cash flow growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Free_Cash_Flow: For free cash flow values

        Returns:
            pd.DataFrame: Average FCF growth rate values.
        """
        # Get required series from financial data
        fcf = self._financial_data['Free Cash Flow']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                fcf = fcf.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                fcf = fcf.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            fcf = fcf.rolling(trailing).mean()

        result = earnings_model.get_free_cash_flow_average_growth(fcf)

        # Name based on frequency used
        ratio_name = 'Average FCF Growth (5yrs)'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    ################ Quality Model Ratios ###############

    @handle_errors
    def collect_quality_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            days: int | float | None = None
    ) -> pd.DataFrame:
        """
        Calculates and collects all Quality-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Quality ratios calculated based on the specified parameters.
        """
        # Calculate all quality ratios with the appropriate frequency
        aicr = self.get_aicr_ratio(freq=FrequencyType.FY)
        profit_dip = self.get_profit_dip_ratio(freq=FrequencyType.FY)
        roic_band = self.get_roic_band_ratio(freq=FrequencyType.FY)
        cfo_band = self.get_cfo_band_ratio(freq=FrequencyType.FY)
        fcf_dip = self.get_fcf_dip_ratio(freq=FrequencyType.FY)
        negative_fcf = self.get_negative_fcf_ratio(freq=FrequencyType.FY)
        cfo_profit = self.get_cfo_profit_ratio(freq=FrequencyType.FY)

        # Combine all ratios
        self._quality_ratios = pd.concat([
            aicr, profit_dip, roic_band, cfo_band,
            fcf_dip, negative_fcf, cfo_profit
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
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Annual Intrinsic Compounding Rate (AICR) ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For total liabilities
            - Dividend_Paid: For dividends paid

        Returns:
            pd.DataFrame: AICR ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']
        total_liabilities = self._financial_data['Total Liabilities']
        dividend_paid = self._financial_data['Dividends Paid']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
                total_liabilities = total_liabilities.freq.FY(exchange=self._exchange)
                dividend_paid = dividend_paid.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                total_assets = total_assets.freq.TTM
                total_liabilities = total_liabilities.freq.TTM
                dividend_paid = dividend_paid.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_liabilities = total_liabilities.rolling(trailing).mean()
            dividend_paid = dividend_paid.rolling(trailing).mean()

        result = quality_model.get_intrinsic_compounding_rate(net_income, total_assets, total_liabilities,
                                                              dividend_paid)

        # Name based on frequency used
        ratio_name = 'Annual Intrinsic Compounding Rate'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_profit_dip_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Profit Dip ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Revenue: For total revenue
            - Total_Expense: For total expenses

        Returns:
            pd.DataFrame: Profit dip ratio values.
        """
        # Get required series from financial data
        net_profit = self._financial_data['Net Income']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_profit = net_profit.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_profit = net_profit.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_profit = net_profit.rolling(trailing).mean()

        result = quality_model.get_dips_in_profit_over_10yrs(net_profit)

        # Name based on frequency used
        ratio_name = 'Profit Dip Last 10Y'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_roic_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the ROIC Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - EBIT: For earnings before interest and taxes
            - Tax_Rate: For effective tax rate
            - Total_Equity: For shareholders' equity
            - Short_Term_Debt: For short-term debt
            - Long_Term_Debt: For long-term debt

        Returns:
            pd.DataFrame: ROIC band ratio values.
        """
        # Get required series from financial data
        # Using calculated fields from field_normalizer
        invested_capital = self._financial_data['Invested Capital']
        ebit = self._financial_data['EBIT']
        tax_rate = self._financial_data['Tax Rate']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                invested_capital = invested_capital.freq.FY(exchange=self._exchange)
                ebit = ebit.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                invested_capital = invested_capital.freq.TTM
                ebit = ebit.freq.TTM

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            invested_capital = invested_capital.rolling(trailing).mean()
            ebit = ebit.rolling(trailing).mean()

        result = quality_model.get_roic_band(invested_capital,ebit,tax_rate)

        # Name based on frequency used
        ratio_name = 'ROIC Band'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cfo_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the CFO Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: CFO band ratio values.
        """
        cfo = self._financial_data['Operating Cash Flow']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                cfo = cfo.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                cfo = cfo.freq.TTM

        if trailing:
            cfo = cfo.T.rolling(trailing).mean().T

        result = quality_model.get_cfo_band(cfo)

        # Name based on frequency used
        ratio_name = 'CFO Band'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_dip_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the FCF Dip ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: FCF dip ratio values.
        """
        fcf = self._financial_data['Free Cash Flow']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                fcf = fcf.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                fcf = fcf.freq.TTM

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T

        result = quality_model.get_negative_dips_in_fcf_over_10yrs(fcf)

        # Name based on frequency used
        ratio_name = 'FCF Dip Last 10Y'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_negative_fcf_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Negative FCF ratio for the last 10 years.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: Negative FCF ratio values.
        """
        fcf = self._financial_data['Free Cash Flow']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                fcf = fcf.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                fcf = fcf.freq.TTM

        if trailing:
            fcf = fcf.T.rolling(trailing).mean().T

        result = quality_model.get_negative_fcf_years(fcf)

        # Name based on frequency used
        ratio_name = 'Negative FCF Last 10Y'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cfo_profit_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the FCF to Profit Band ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Returns:
            pd.DataFrame: FCF to profit band ratio values.
        """
        cfo = self._financial_data['Operating Cash Flow']
        net_profit = self._financial_data['Net Income']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                cfo = cfo.freq.FY(exchange=self._exchange)
                net_profit = net_profit.freq.FY(exchange=self._exchange)
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                cfo = cfo.freq.TTM
                net_profit = net_profit.freq.TTM

        if trailing:
            cfo = cfo.T.rolling(trailing).mean().T
            net_profit = net_profit.T.rolling(trailing).mean().T

        result = quality_model.get_cfo_to_net_profit(cfo, net_profit)

        # Name based on frequency used
        ratio_name = 'CFO to Profit'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    ###################### Valuation Model Ratios #######################


    @handle_errors
    def collect_valuation_ratios(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            days: int | float | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Valuation-related Ratios.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For earnings per share calculations
            - WACC: For weighted average cost of capital
            - Stock_Price: For current market price
            - Revenue: For revenue-based valuations
            - Operating_Cash_Flow: For cash flow-based valuations
            Additional columns may be required by individual ratio calculations.

        Returns:
            pd.DataFrame: Valuation ratios calculated based on the specified parameters.
        """
        # Calculate all valuation ratios with the appropriate frequency
        if self._quarterly:
            cmp_revenue = self.get_cmp_revenue_band_ratio(freq=FrequencyType.TTM)
            cmp_eps = self.get_cmp_eps_band_ratio(freq=FrequencyType.TTM)

            # Combine all ratios
            self._valuation_ratios = pd.concat([
                 cmp_revenue, cmp_eps], axis=1)
        else:
            steady_state = self.get_steady_state_value_ratio(freq=FrequencyType.FY)
            fair_value = self.get_fair_value_ratio(freq=FrequencyType.FY)
            cmp_cfo = self.get_cmp_cfo_band_ratio(freq=FrequencyType.FY)
            fcf_yield = self.get_fcf_yield_ratio(freq=FrequencyType.FY)
            self._valuation_ratios = pd.concat([
                steady_state, fair_value , cmp_cfo, fcf_yield], axis=1)




        # Process and return the results
        return self._process_ratio_result(self._valuation_ratios, growth, lag, rounding)

    @handle_errors
    def get_steady_state_value_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Steady State Value ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Basic_EPS: For earnings per share
            - WACC: For weighted average cost of capital
            - Stock_Price: For current market price

        Returns:
            pd.DataFrame: Steady State Value ratio.
        """
        # Get required series from financial data
        price = self._financial_data['Stock Price']
        wacc = self._financial_data['WACC']
        shares_outstanding = self._financial_data['Shares Outstanding']
        ebit = self._financial_data['EBIT']
        tax_rate = self._financial_data['Tax Rate']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                price = price.freq.FY(exchange=self._exchange)
                wacc = wacc.freq.FY(exchange=self._exchange)
                ebit = ebit.freq.FY(exchange=self._exchange)
                tax_rate = tax_rate.freq.FY(exchange=self._exchange)
                # Current price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                price = price.freq.TTM
                wacc = wacc.freq.TTM
                ebit = ebit.freq.TTM
                tax_rate = tax_rate.freq.TTM
                # Current price typically doesn't get frequency treatment

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            price = price.rolling(trailing).mean()
            wacc = wacc.rolling(trailing).mean()
            ebit = ebit.rolling(trailing).mean()
            tax_rate = tax_rate.rolling(trailing).mean()


        result = valuation_model.get_steady_state_value(price, wacc, shares_outstanding, ebit, tax_rate)

        # Name based on frequency used
        ratio_name = 'Steady State Value'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fair_value_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Fair Value vs Current Market Price ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For total liabilities
            - Basic_EPS: For earnings per share
            - Stock_Price: For current market price

        Returns:
            pd.DataFrame: Fair Value ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net Income']
        total_assets = self._financial_data['Total Assets']
        total_liabilities = self._financial_data['Total Liabilities']
        eps = self._financial_data['Basic EPS']
        current_price = self._financial_data['Stock Price']
        dividends_paid = self._financial_data['Dividends Paid']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                net_income = net_income.freq.FY(exchange=self._exchange)
                total_assets = total_assets.freq.FY(exchange=self._exchange)
                total_liabilities = total_liabilities.freq.FY(exchange=self._exchange)
                eps = eps.freq.FY(exchange=self._exchange)
                dividends_paid = dividends_paid.freq.FY(exchange=self._exchange)
                # Stock price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                net_income = net_income.freq.TTM
                total_assets = total_assets.freq.TTM
                total_liabilities = total_liabilities.freq.TTM
                eps = eps.freq.TTM
                dividends_paid = dividends_paid.freq.TTM
                # Stock price typically doesn't get frequency treatment

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_liabilities = total_liabilities.rolling(trailing).mean()
            eps = eps.rolling(trailing).mean()
            dividends_paid = dividends_paid.rolling(trailing).mean()

        result = valuation_model.get_fair_value_vs_market_price(
            net_income, total_assets, total_liabilities, eps, current_price, dividends_paid
        )

        # Name based on frequency used
        ratio_name = 'Fair Value vs Market Price'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_revenue_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Price to Revenue Band ratio for last 3 years.

        This metric shows how current P/S ratio deviates from its historical average.
        Useful for:
        - Companies with negative earnings where P/E can't be used
        - Comparing companies in the same industry
        - Identifying potential over/undervaluation based on historical trends

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Revenue: For total revenue values
            - Shares_Outstanding: For shares outstanding values

        Returns:
            pd.DataFrame: Price to Revenue band values (in standard deviations).
            Returns NaN for periods with zero revenue or shares, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock Price']
        revenue = self._financial_data['Revenue']
        shares_outstanding = self._financial_data['Shares Outstanding']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                revenue = revenue.freq.FY(exchange=self._exchange)
                # Stock price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                revenue = revenue.freq.TTM
                # Stock price typically doesn't get frequency treatment

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            price = price.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

        result = valuation_model.get_price_to_revenue_band(price, revenue, shares_outstanding)

        # Name based on frequency used
        ratio_name = 'Price to Revenue Band'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_eps_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Price to Earnings (P/E) Band ratio.

        This metric shows how current P/E ratio deviates from its historical average.
        One of the most widely used valuation metrics as it shows how much investors
        are willing to pay for each dollar of earnings.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Basic_EPS: For earnings per share values

        Returns:
            pd.DataFrame: Price to Earnings band values (in standard deviations).
            Returns NaN for periods with zero EPS, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock Price']
        eps = self._financial_data['Basic EPS']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                eps = eps.freq.FY(exchange=self._exchange)
                # Stock price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                eps = eps.freq.TTM
                # Stock price typically doesn't get frequency treatment

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            price = price.rolling(trailing).mean()
            eps = eps.rolling(trailing).mean()

        result = valuation_model.get_price_to_eps_band(price, eps)

        # Name based on frequency used
        ratio_name = 'Price to Earnings Band'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_cmp_cfo_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Price to Operating Cash Flow (P/CFO) Band ratio.

        This metric shows how current P/CFO ratio deviates from its historical average.
        Valuable because:
        - Cash flows are harder to manipulate than earnings
        - Helps identify companies trading at premium/discount to historical cash generation
        - Useful for companies with significant non-cash charges

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Operating_Cash_Flow: For Operating Cash Flow values
            - Shares_Outstanding: For shares outstanding values

        Returns:
            pd.DataFrame: P/CFO band values (in standard deviations).
            Returns NaN for periods with zero CFO or shares, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock Price']
        cfo = self._financial_data['Operating Cash Flow']
        shares_outstanding = self._financial_data['Shares Outstanding']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                cfo = cfo.freq.FY(exchange=self._exchange)
                # Stock price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                cfo = cfo.freq.TTM
                # Stock price typically doesn't get frequency treatment

        # Apply trailing if specified (for backward compatibility)
        if trailing:
            price = price.rolling(trailing).mean()
            cfo = cfo.rolling(trailing).mean()

        result = valuation_model.get_price_to_cfo_band(price, cfo, shares_outstanding)

        # Name based on frequency used
        ratio_name = 'Price to CFO Band'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name

        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)

    @handle_errors
    def get_fcf_yield_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
            freq: FrequencyType = None,
    ) -> pd.DataFrame:
        """
        Calculate the Free Cash Flow (FCF) Yield ratio.

        This metric measures how much free cash flow a company generates relative to its market value.
        Important because:
        - Shows how much cash is available for dividends, buybacks, or reinvestment
        - Higher yields may indicate undervaluation or strong cash generation ability
        - Useful for comparing companies across different sectors

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            freq (FrequencyType, optional): Frequency type to apply (FY for fiscal year, TTM for trailing twelve months).

        Required columns in financial_data:
            - Operating_Cash_Flow: For Operating Cash Flow
            - Capital_Expenditure: For capital expenditure
            - Shares_Outstanding: For shares outstanding
            - Stock_Price: For current market price

        Returns:
            pd.DataFrame: FCF Yield ratio in absolute percentage.
            Returns NaN for periods with zero market cap or insufficient data.
        """
        # Get required series from financial data
        fcf = self._financial_data['Free Cash Flow']
        shares_outstanding = self._financial_data['Shares Outstanding']
        price = self._financial_data['Stock Price']

        # Apply frequency transformation if requested
        if freq is not None:
            if freq == FrequencyType.FY:
                # Apply fiscal year calculations
                fcf = fcf.freq.FY(exchange=self._exchange)
                # Stock price typically doesn't get frequency treatment
            elif freq == FrequencyType.TTM:
                # Apply trailing twelve months calculations
                fcf = fcf.freq.TTM
                # Stock price typically doesn't get frequency treatment


        # Apply trailing if specified (for backward compatibility)
        if trailing:
            fcf = fcf.rolling(trailing).mean()
            price = price.rolling(trailing).mean()

        result = valuation_model.get_fcf_yield(fcf, price, shares_outstanding)
        
        # Name based on frequency used
        ratio_name = 'FCF Yield'
        if freq == FrequencyType.TTM:
            ratio_name = 'TTM ' + ratio_name
        elif freq == FrequencyType.FY:
            ratio_name = 'FY ' + ratio_name
        elif self._quarterly:
            ratio_name = 'QoQ ' + ratio_name
            
        result_df = result.to_frame(name=ratio_name)
        return self._process_ratio_result(result_df, growth, lag, rounding)






