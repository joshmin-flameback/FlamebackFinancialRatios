<div align="center">
  <img src="./assets/factorslab_logo.png" alt="Flameback Financial Ratios Logo" width="600px" />
</div>

# Flameback Financial Ratios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/flameback-financial-ratios.svg)](https://pypi.org/project/flameback-financial-ratios/)

A beginner-friendly Python package for calculating and analyzing financial ratios. This library helps you analyze company financials through various metrics including earnings, financial health, quality indicators, and valuation models.

## What Are Financial Ratios?

Financial ratios are mathematical comparisons of financial statement accounts or categories that help evaluate a company's financial performance. These ratios help investors and analysts interpret financial statements by providing a standardized way to compare different companies or different time periods for the same company.

## Features

- **Earnings Analysis**
  - Operating Margin
  - Net Profit Margin
  - Return on Equity (ROE)
  - Return on Assets (ROA)
  
- **Financial Health Metrics**
  - Current Ratio
  - Quick Ratio
  - Debt-to-Equity Ratio
  - Interest Coverage Ratio
  - Cash Conversion Cycle
  - Altman Z-Score
  
- **Quality Indicators**
  - Asset Turnover
  - Inventory Turnover
  - Receivables Turnover
  
- **Valuation Models**
  - Price-to-Earnings (P/E)
  - Price-to-Book (P/B)
  - Enterprise Value Multiples

## Installation

```bash
pip install https://github.com/joshmin-flameback/FlamebackFinancialRatios/archive/refs/heads/main.zip
```

## Data Requirements

The library uses a simplified interface with a single DataFrame for all financial data. This DataFrame should have the following standardized column names (use only the columns needed for your specific ratios):

### For Financial Health Ratios
- `Long_Term_Debt`, `Short_Term_Debt`, `Total_Equity` - For debt to equity ratio
- `EBITDA`, `Interest_Expense` - For interest coverage ratio
- `Current_Assets`, `Current_Liabilities` - For current ratio
- `Inventory`, `Cost_Of_Goods_Sold`, `Accounts_Receivable`, `Revenue`, `Accounts_Payable` - For cash conversion cycle
- `Total_Assets`, `EBIT`, `Diluted_Shares_Outstanding`, `Total_Liabilities`, `Retained_Earnings`, `Stock_Price` - For Altman Z-score

### For Earnings Ratios
- `Revenue` - For revenue growth and related metrics
- `Basic_EPS` - For EPS growth and related metrics
- `Gross_Margin` - For gross margin metrics
- `EBITDA` - For EBITDA-related metrics

## Quick Start Guide

### Basic Usage

```python
import pandas as pd
from financial_ratios import RatiosController

# Initialize the controller
controller = RatiosController()

# Create a DataFrame with financial data
financial_data = pd.DataFrame({
    'Year': [2019, 2020, 2021, 2022],
    'Revenue': [1000000, 1200000, 1350000, 1500000],
    'Total_Assets': [2000000, 2100000, 2300000, 2500000],
    'Total_Equity': [1500000, 1600000, 1700000, 1800000],
    # Add other required columns based on what ratios you want to calculate
})

# Calculate all financial ratios
all_ratios = controller.calculate_all_ratios(financial_data)

# Or calculate specific ratio categories
earnings_ratios = controller.calculate_earnings_ratios(financial_data)
health_ratios = controller.calculate_financial_health_ratios(financial_data)
quality_ratios = controller.calculate_quality_ratios(financial_data)
valuation_ratios = controller.calculate_valuation_ratios(financial_data)

# Display the results
print(all_ratios)
```

### How to Use a Single Ratio

If you only need to calculate a specific ratio, you can use the individual ratio calculation methods. Here are examples for common ratios:

#### Example 1: Calculating Current Ratio

```python
import pandas as pd
from financial_ratios import RatiosController

# Create a DataFrame with only the required fields for current ratio
financial_data = pd.DataFrame({
    'Year': [2021, 2022],
    'Current_Assets': [500000, 550000],
    'Current_Liabilities': [300000, 320000]
})

# Initialize controller
controller = RatiosController()

# Calculate current ratio
current_ratio = controller.financial_health.calculate_current_ratio(financial_data)
print(f"Current Ratio: {current_ratio}")
# Expected output: A DataFrame with Current_Ratio values for 2021 and 2022
```

#### Example 2: Calculating Debt-to-Equity Ratio

```python
import pandas as pd
from financial_ratios import RatiosController

# Create a DataFrame with only the required fields for debt-to-equity ratio
financial_data = pd.DataFrame({
    'Year': [2021, 2022],
    'Long_Term_Debt': [400000, 450000],
    'Short_Term_Debt': [100000, 120000],
    'Total_Equity': [1200000, 1300000]
})

# Initialize controller
controller = RatiosController()

# Calculate debt-to-equity ratio
debt_to_equity = controller.financial_health.calculate_debt_to_equity_ratio(financial_data)
print(f"Debt-to-Equity Ratio: {debt_to_equity}")
```

#### Example 3: Calculating Return on Equity (ROE)

```python
import pandas as pd
from financial_ratios import RatiosController

# Create a DataFrame with only the required fields for ROE
financial_data = pd.DataFrame({
    'Year': [2021, 2022],
    'Net_Income': [150000, 180000],
    'Total_Equity': [1200000, 1300000]
})

# Initialize controller
controller = RatiosController()

# Calculate ROE
roe = controller.earnings.calculate_return_on_equity(financial_data)
print(f"Return on Equity: {roe}")
```

## Tips for Beginners

1. **Start Small**: If you're just learning, begin by calculating one ratio at a time until you understand what it means and what data it requires.
   
2. **Data Preparation**: Make sure your financial data DataFrame has the correct column names as specified in the Data Requirements section above.
   
3. **Data Sources**: You can get financial data from company annual reports, financial websites like Yahoo Finance, or financial data APIs.
   
4. **Time Series Analysis**: Include multiple years in your DataFrame to track how ratios change over time.
   
5. **Data Validation**: Always check that your input data makes sense before calculating ratios to avoid incorrect results.

## Documentation

For detailed documentation, please visit our [Wiki](../../wiki).

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/joshmin-flameback/FlamebackFinancialRatios.git
cd FlamebackFinacialRatios
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

Before contributing, please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Create an [Issue](../../issues) for bug reports, feature requests, or questions
- Follow [@YourTwitterHandle](https://twitter.com/YourTwitterHandle) for announcements
- Add a ⭐️ star on GitHub to support the project!
