"""
Ratio Dependencies Module

This module provides comprehensive mapping of financial data field dependencies
required by ratio calculations. It defines which data points are needed for specific
financial ratio calculations without introducing circular dependencies.

Updated to support FrequencyType parameter used across ratio methods, enabling
calculations based on Fiscal Year (FY) or Trailing Twelve Month (TTM) data.
"""

from typing import Set, List, Dict, Any, Optional

# Mapping of ratio calculation methods to their required financial data fields
RATIO_FIELD_DEPENDENCIES = {
    # Financial Health Ratios
    'get_debt_to_equity_ratio': {'Total Debt', 'Total Equity'},
    'get_interest_coverage_ratio': {'EBITDA', 'Interest Expense'},
    'get_current_ratio': {'Total Current Assets', 'Total Current Liabilities'},
    'get_cash_conversion_cycle': {'Total Inventories', 'Cost of Goods Sold', 'Accounts Receivable',
                                  'Revenue', 'Accounts Payable'},
    'get_altman_z_score': {'Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'EBIT',
                           'Shares Outstanding', 'Revenue', 'Total Liabilities',
                           'Retained Earnings', 'Stock Price'},

    # Earnings Ratios
    'get_piotroski_score_ratio': {'Net Income', 'Operating Cash Flow', 'Total Assets', 'Total Debt',
                                 'Total Current Assets', 'Total Current Liabilities', 'Shares Outstanding', 'Gross Margin',
                                 'Total Assets Turnover'},
    'get_revenue_growth_ratio': {'Revenue'},
    'get_eps_growth_ratio': {'Basic EPS'},
    'get_roe_ratio': {'Net Income', 'Total Equity'},
    'get_fcf_growth_ratio': {'Free Cash Flow'},
    'get_revenue_consecutive_growth_ratio': {'Revenue'},
    'get_eps_consecutive_growth_ratio': {'Basic EPS'},
    'get_average_revenue_growth_ratio': {'Revenue'},
    'get_average_gross_margin_ratio': {'Gross Margin'},
    'get_average_gross_margin_growth_ratio': {'Gross Margin'},
    'get_average_ebitda_ratio': {'EBITDA'},
    'get_average_ebitda_growth_ratio': {'EBITDA'},
    'get_average_eps_growth_ratio': {'Basic EPS'},
    'get_revenue_growth_vs_average_growth_ratio': {'Revenue'},
    'get_eps_growth_vs_average_growth_ratio': {'Basic EPS'},
    'get_ebitda_growth_vs_average_growth_ratio': {'EBITDA'},
    'get_gross_margin_growth_vs_average_growth_ratio': {'Gross Margin'},
    'get_return_on_assets_ratio': {'Net Income', 'Total Assets'},
    'get_roe_vs_average_roe_ratio': {'Net Income', 'Total Equity'},
    'get_roa_vs_average_roa_ratio': {'Net Income', 'Total Assets'},
    'get_revenue_vs_estimate_ratio': {'Revenue', 'Revenue Estimate'},
    'get_shares_outstanding_vs_estimate_ratio': {'Net Income', 'Basic EPS', 'Net Income Estimate', 'EPS Estimate'},
    'get_free_cash_flow_average_growth_ratio': {'Free Cash Flow'},

    # Quality Ratios
    'get_aicr_ratio': {'Net Income', 'Total Assets', 'Total Liabilities', 'Dividends Paid'},
    'get_profit_dip_ratio': {'Net Income'},
    'get_roic_band_ratio': {'Invested Capital', 'NOPAT'},
    'get_cfo_band_ratio': {'Cash Flow from Operations'},
    'get_fcf_dip_ratio': {'Free Cash Flow'},
    'get_negative_fcf_ratio': {'Free Cash Flow'},
    'get_cfo_profit_band_ratio': {'Cash Flow from Operations', 'Net Income'},

    # Valuation Ratios
    'get_steady_state_value_ratio': {'Basic EPS', 'WACC', 'Stock Price'},
    'get_fair_value_ratio': {'Net Income', 'Total Assets', 'Total Liabilities', 'Basic EPS', 'Stock Price'},
    'get_cmp_revenue_band_ratio': {'Stock Price', 'Revenue', 'Shares Outstanding'},
    'get_cmp_eps_band_ratio': {'Stock Price', 'Basic EPS'},
    'get_cmp_cfo_band_ratio': {'Stock Price', 'Operating Cash Flow', 'Shares Outstanding'},
    'get_fcf_yield_ratio': {'Free Cash Flow', 'Stock Price', 'Shares Outstanding'}

}


def get_ratio_dependencies(ratio_name: str) -> Set[str]:
    """
    Get the financial data fields required for calculating a specific ratio.
    
    Args:
        ratio_name: The name of the ratio or collection method
        
    Returns:
        Set of financial data field names required for the calculation
    """
    return RATIO_FIELD_DEPENDENCIES.get(ratio_name, set())


def get_all_financial_dependencies() -> Set[str]:
    """
    Get all unique financial data fields required across all ratio calculations.
    
    Returns:
        Set of all unique field names required by any ratio calculation
    """
    all_fields = set()
    for fields in RATIO_FIELD_DEPENDENCIES.values():
        all_fields.update(fields)
    return all_fields


def get_dependencies_for_categories(categories: List[str]) -> Set[str]:
    """
    Get the combined financial data fields required for multiple ratio categories.
    
    Args:
        categories: List of ratio category collection names
                   (e.g., ['collect_financial_health_ratios', 'collect_earning_ratios'])
        
    Returns:
        Set of all financial data fields required for the specified categories
    """
    combined_fields = set()
    for category in categories:
        category_fields = get_ratio_dependencies(category)
        combined_fields.update(category_fields)
    return combined_fields


class FinancialDependencyRegistry:
    """
    Registry class for accessing financial data dependencies for ratio calculations.
    Provides methods to identify which financial data fields are required for different
    types of ratio calculations.
    """

    @classmethod
    def get_fields_for_ratio(cls, ratio_name: str) -> Set[str]:
        """
        Get the financial data fields required for a specific ratio calculation.
        
        Args:
            ratio_name: The name of the ratio method (e.g., 'get_debt_to_equity_ratio')
            
        Returns:
            Set of financial data field names required for the calculation
        """
        return get_ratio_dependencies(ratio_name)

    @classmethod
    def get_fields_for_collection(cls, collection_name: str) -> Set[str]:
        """
        Get financial data fields required for a ratio collection method.
        
        Args:
            collection_name: The name of the collection method (e.g., 'collect_financial_health_ratios')
            
        Returns:
            Set of field names required for the collection
        """
        return get_ratio_dependencies(collection_name)

    @classmethod
    def get_all_fields(cls) -> Set[str]:
        """
        Get all unique financial data fields required for any ratio calculation.
        
        Returns:
            Set of all financial data field names
        """
        return get_all_financial_dependencies()

    @classmethod
    def get_fields_for_categories(cls, categories: List[str]) -> Set[str]:
        """
        Get financial data fields required for multiple ratio categories.
        
        Args:
            categories: List of collection names
            
        Returns:
            Combined set of required financial data field names
        """
        return get_dependencies_for_categories(categories)

    @classmethod
    def check_fields_availability(cls, available_fields: Set[str], ratio_name: str) -> Dict[str, bool]:
        """
        Check if all financial data fields required for a ratio are available.
        
        Args:
            available_fields: Set of available field names in the financial data
            ratio_name: Name of the ratio to check
            
        Returns:
            Dictionary mapping field names to availability status
        """
        required_fields = cls.get_fields_for_ratio(ratio_name)
        return {field: field in available_fields for field in required_fields}

    @classmethod
    def get_missing_fields(cls, available_fields: Set[str], ratio_name: str) -> Set[str]:
        """
        Get financial data fields required for a ratio that are not available.
        
        Args:
            available_fields: Set of available field names in the financial data
            ratio_name: Name of the ratio to check
            
        Returns:
            Set of missing field names
        """
        required_fields = cls.get_fields_for_ratio(ratio_name)
        return required_fields - available_fields


# Create an instance for direct import
financial_dependencies = FinancialDependencyRegistry()
