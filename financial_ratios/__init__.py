"""
Financial Ratios Module

This module provides implementations of various financial ratios categorized into:
- Earnings Ratios
- Financial Health Ratios
- Piotroski Score
- Quality Ratios
- Valuation Ratios

It also provides utilities to identify which financial data fields are required for each ratio calculation.
"""

from .ratios_controller import Ratios
from . import (
    earnings_model,
    financial_health_model,
    quality_model,
    valuation_model,
)
from .utils import financial_dependencies, FinancialDependencyRegistry

__version__ = "0.1.0"

__all__ = [
    'Ratios',
    'earnings_model',
    'financial_health_model',
    'quality_model',
    'valuation_model',
    'financial_dependencies',
    'FinancialDependencyRegistry'
]
