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
                - 'Current_Assets': Current assets values
                - 'Current_Liabilities': Current liabilities values
                etc...
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
        """
        self._tickers = tickers
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

        Required columns in financial_data:
            - Long_Term_Debt, Short_Term_Debt, Total_Equity: For debt to equity ratio
            - EBITDA, Interest_Expense: For interest coverage ratio
            - Current_Assets, Current_Liabilities: For current ratio
            - Inventory, Cost_Of_Goods_Sold, Accounts_Receivable, Revenue, Accounts_Payable: For cash conversion cycle
            - Total_Assets, EBIT, Diluted_Shares_Outstanding, Total_Liabilities, Retained_Earnings, Stock_Price: For Altman Z-score

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
        self._financial_health_ratios = pd.concat([debt_equity, interest_coverage, current_ratio, cash_conversion, altman_z], axis=1)
        
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
        # Get required series from financial data
        total_debt = self._financial_data['Total_Debt']
        total_equity = self._financial_data['Total_Equity']

        if trailing:
            total_debt = total_debt.rolling(trailing).mean()
            total_equity = total_equity.rolling(trailing).mean()

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

        # Get required series from financial data
        ebitda = self._financial_data['EBITDA']
        interest_expense = self._financial_data['Interest_Expense']

        if trailing:
            ebitda = ebitda.rolling(trailing).sum()
            interest_expense = interest_expense.rolling(trailing).sum()

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
        # Get required series from financial data
        current_assets = self._financial_data['Current_Assets']
        current_liabilities = self._financial_data['Current_Liabilities']

        if trailing:
            current_assets = current_assets.rolling(trailing).mean()
            current_liabilities = current_liabilities.rolling(trailing).mean()

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

        # Get required series from financial data
        inventory = self._financial_data['Inventory']
        cogs = self._financial_data['Cost_Of_Goods_Sold']
        accounts_receivable = self._financial_data['Accounts_Receivable']
        revenue = self._financial_data['Revenue']
        accounts_payable = self._financial_data['Accounts_Payable']

        if trailing:
            inventory = inventory.rolling(trailing).mean()
            cogs = cogs.rolling(trailing).sum()
            accounts_receivable = accounts_receivable.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).sum()
            accounts_payable = accounts_payable.rolling(trailing).mean()

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
        # Get required series from financial data
        current_assets = self._financial_data['Current_Assets']
        current_liabilities = self._financial_data['Current_Liabilities']
        total_assets = self._financial_data['Total_Assets']
        ebit = self._financial_data['EBIT']
        diluted_shares = self._financial_data['Diluted_Shares_Outstanding']
        revenue = self._financial_data['Revenue']
        total_liabilities = self._financial_data['Total_Liabilities']
        retained_earnings = self._financial_data['Retained_Earnings']
        stock_price = self._financial_data['Stock_Price']

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

        Required columns in financial_data:
            - Revenue: For revenue growth and related metrics
            - Basic_EPS: For EPS growth and related metrics
            - Gross_Margin: For gross margin metrics
            - EBITDA: For EBITDA-related metrics
            Additional columns may be required by individual ratio calculations.

        Returns:
            pd.DataFrame: Earnings ratios calculated based on the specified parameters.
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']
        eps_basic = self._financial_data['Basic_EPS']
        gross_margin = self._financial_data['Gross_Margin']
        ebitda = self._financial_data['EBITDA']

        # Calculate all earnings ratios
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
        net_income = self._financial_data['Net_Income']
        operating_cash_flow = self._financial_data['Operating_Cash_Flow']
        total_assets = self._financial_data['Total_Assets']
        total_debt = self._financial_data['Total_Debt']
        current_assets = self._financial_data['Current_Assets']
        current_liabilities = self._financial_data['Current_Liabilities']
        shares_outstanding = self._financial_data['Shares_Outstanding']
        revenue = self._financial_data['Revenue']
        cogs = self._financial_data['Cost_Of_Goods_Sold']

        # Apply trailing if specified
        if trailing:
            net_income = net_income.rolling(trailing).mean()
            operating_cash_flow = operating_cash_flow.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_debt = total_debt.rolling(trailing).mean()
            current_assets = current_assets.rolling(trailing).mean()
            current_liabilities = current_liabilities.rolling(trailing).mean()
            shares_outstanding = shares_outstanding.rolling(trailing).mean()
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

        Required columns in financial_data:
            - EBITDA: For earnings before interest, taxes, depreciation, and amortization
            - Revenue: For total revenue

        Returns:
            pd.DataFrame: EBITDA margin ratio values.
        """
        # Get required series from financial data
        ebitda = self._financial_data['EBITDA']
        revenue = self._financial_data['Revenue']

        if trailing:
            ebitda = ebitda.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()

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

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For calculating shareholders' equity

        Returns:
            pd.DataFrame: ROE ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net_Income']
        total_assets = self._financial_data['Total_Assets']
        total_liabilities = self._financial_data['Total_Liabilities']

        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_liabilities = total_liabilities.rolling(trailing).mean()

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

        Required columns in financial_data:
            - Free_Cash_Flow: For calculating year-over-year FCF growth

        Returns:
            pd.DataFrame: FCF growth ratio values.
        """
        # Get required series from financial data
        fcf = self._financial_data['Free_Cash_Flow']

        if trailing:
            fcf = fcf.rolling(trailing).mean()

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

        Required columns in financial_data:
            - Net_Income: For net income
            - Total_Assets: For total assets
            - Total_Liabilities: For total liabilities
            - Dividend_Paid: For dividends paid

        Returns:
            pd.DataFrame: AICR ratio values.
        """
        # Get required series from financial data
        net_income = self._financial_data['Net_Income']
        total_assets = self._financial_data['Total_Assets']
        total_liabilities = self._financial_data['Total_Liabilities']
        dividend_paid = self._financial_data['Dividend_Paid']

        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_liabilities = total_liabilities.rolling(trailing).mean()
            dividend_paid = dividend_paid.rolling(trailing).mean()

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

        Required columns in financial_data:
            - Revenue: For total revenue
            - Total_Expense: For total expenses

        Returns:
            pd.DataFrame: Profit dip ratio values.
        """
        # Get required series from financial data
        revenue = self._financial_data['Revenue']
        total_expense = self._financial_data['Total_Expense']

        if trailing:
            revenue = revenue.rolling(trailing).mean()
            total_expense = total_expense.rolling(trailing).mean()

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
        ebit = self._financial_data['EBIT']
        tax_rate = self._financial_data['Tax_Rate']
        total_equity = self._financial_data['Total_Equity']
        short_term_debt = self._financial_data['Short_Term_Debt']
        long_term_debt = self._financial_data['Long_Term_Debt']

        if trailing:
            ebit = ebit.rolling(trailing).mean()
            tax_rate = tax_rate.rolling(trailing).mean()
            total_equity = total_equity.rolling(trailing).mean()
            short_term_debt = short_term_debt.rolling(trailing).mean()
            long_term_debt = long_term_debt.rolling(trailing).mean()

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

        Required columns in financial_data:
            - Basic_EPS: For earnings per share
            - WACC: For weighted average cost of capital
            - Stock_Price: For current market price

        Returns:
            pd.DataFrame: Steady State Value ratio.
        """
        # Get required series from financial data
        eps = self._financial_data['Basic_EPS']
        wacc = self._financial_data['WACC']
        current_price = self._financial_data['Stock_Price'].iloc[-1]  # Get latest price

        if trailing:
            eps = eps.rolling(trailing).mean()
            wacc = wacc.rolling(trailing).mean()

        # Pass current price as a Series to match model requirements
        current_price_series = pd.Series([current_price] * len(eps), index=eps.index)
        
        result = valuation_model.get_steady_state_value(eps, wacc, current_price_series)
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
        net_income = self._financial_data['Net_Income']
        total_assets = self._financial_data['Total_Assets']
        total_liabilities = self._financial_data['Total_Liabilities']
        eps = self._financial_data['Basic_EPS']
        current_price = self._financial_data['Stock_Price']

        if trailing:
            net_income = net_income.rolling(trailing).mean()
            total_assets = total_assets.rolling(trailing).mean()
            total_liabilities = total_liabilities.rolling(trailing).mean()
            eps = eps.rolling(trailing).mean()
            current_price = current_price.rolling(trailing).mean()

        result = valuation_model.get_fair_value_vs_market_price(
            net_income, total_assets, total_liabilities, eps, current_price
        )
        result_df = result.to_frame(name='Fair Value vs Market Price')
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

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Revenue: For total revenue values
            - Shares_Outstanding: For shares outstanding values

        Returns:
            pd.DataFrame: Price to Revenue band values (in standard deviations).
            Returns NaN for periods with zero revenue or shares, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock_Price']
        revenue = self._financial_data['Revenue']
        shares_outstanding = self._financial_data['Shares_Outstanding']

        if trailing:
            price = price.rolling(trailing).mean()
            revenue = revenue.rolling(trailing).mean()
            shares_outstanding = shares_outstanding.rolling(trailing).mean()

        try:
            result = valuation_model.get_price_to_revenue_band(price, revenue, shares_outstanding)
            result_df = result.to_frame(name='Price to Revenue Band')
            return self._process_ratio_result(result_df, growth, lag, rounding)
        except ValueError as e:
            # Handle insufficient data error
            return pd.DataFrame(columns=['Price to Revenue Band'])

    @handle_errors
    def get_cmp_eps_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
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

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Basic_EPS: For earnings per share values

        Returns:
            pd.DataFrame: Price to Earnings band values (in standard deviations).
            Returns NaN for periods with zero EPS, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock_Price']
        eps = self._financial_data['Basic_EPS']

        if trailing:
            price = price.rolling(trailing).mean()
            eps = eps.rolling(trailing).mean()

        try:
            result = valuation_model.get_price_to_eps_band(price, eps)
            result_df = result.to_frame(name='Price to Earnings Band')
            return self._process_ratio_result(result_df, growth, lag, rounding)
        except ValueError as e:
            # Handle insufficient data error
            return pd.DataFrame(columns=['Price to Earnings Band'])

    @handle_errors
    def get_cmp_cfo_band_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Price to Cash Flow from Operations (P/CFO) Band ratio.

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

        Required columns in financial_data:
            - Stock_Price: For stock price values
            - Operating_Cash_Flow: For Cash Flow from Operations values
            - Shares_Outstanding: For shares outstanding values

        Returns:
            pd.DataFrame: P/CFO band values (in standard deviations).
            Returns NaN for periods with zero CFO or shares, or insufficient data.
        """
        # Get required series from financial data
        price = self._financial_data['Stock_Price']
        cfo = self._financial_data['Operating_Cash_Flow']
        shares_outstanding = self._financial_data['Shares_Outstanding']

        if trailing:
            price = price.rolling(trailing).mean()
            cfo = cfo.rolling(trailing).mean()
            shares_outstanding = shares_outstanding.rolling(trailing).mean()

        try:
            result = valuation_model.get_price_to_cfo_band(price, cfo, shares_outstanding)
            result_df = result.to_frame(name='Price to CFO Band')
            return self._process_ratio_result(result_df, growth, lag, rounding)
        except ValueError as e:
            # Handle insufficient data error
            return pd.DataFrame(columns=['Price to CFO Band'])

    @handle_errors
    def get_fcf_yield_ratio(
            self,
            rounding: int | None = None,
            growth: bool = False,
            lag: int | list[int] = 1,
            trailing: int | None = None,
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

        Required columns in financial_data:
            - Operating_Cash_Flow: For cash flow from operations
            - Capital_Expenditure: For capital expenditure
            - Shares_Outstanding: For shares outstanding
            - Stock_Price: For current market price

        Returns:
            pd.DataFrame: FCF Yield ratio in absolute percentage.
            Returns NaN for periods with zero market cap or insufficient data.
        """
        # Get required series from financial data
        cfo = self._financial_data['Operating_Cash_Flow']
        capex = self._financial_data['Capital_Expenditure']
        fcf = cfo - capex  # Calculate Free Cash Flow
        price = self._financial_data['Stock_Price']
        shares_outstanding = self._financial_data['Shares_Outstanding']

        if trailing:
            fcf = fcf.rolling(trailing).mean()
            price = price.rolling(trailing).mean()
            shares_outstanding = shares_outstanding.rolling(trailing).mean()

        result = valuation_model.get_fcf_yield(fcf, price, shares_outstanding)
        result_df = result.to_frame(name='FCF Yield')
        return self._process_ratio_result(result_df, growth, lag, rounding)






