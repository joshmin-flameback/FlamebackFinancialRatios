# Flameback Financial Ratios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/flameback-financial-ratios.svg)](https://pypi.org/project/flameback-financial-ratios/)

A comprehensive Python package for calculating and analyzing financial ratios. This library provides tools for analyzing company financials through various metrics including earnings, financial health, quality indicators, and valuation models.

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
pip install flameback-financial-ratios
```

## Quick Start

```python
from financial_ratios import RatiosController

# Initialize the controller
controller = RatiosController()

# Calculate all financial ratios
ratios = controller.calculate_all_ratios(financial_data)

# Or calculate specific ratio categories
earnings = controller.calculate_earnings_ratios(financial_data)
health = controller.calculate_financial_health_ratios(financial_data)
quality = controller.calculate_quality_ratios(financial_data)
valuation = controller.calculate_valuation_ratios(financial_data)
```

## Documentation

For detailed documentation, please visit our [Wiki](../../wiki).

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FlamebackFinacialRatios.git
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
