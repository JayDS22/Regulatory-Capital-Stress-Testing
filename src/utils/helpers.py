"""
Helper functions for formatting, validation, and common operations.
"""

from typing import Optional


def format_currency(value: float, prefix: str = "$", decimals: int = 0) -> str:
    """Format a number as currency string."""
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:,.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:,.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{prefix}{value / 1e3:,.{decimals}f}K"
    return f"{prefix}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_bps(value: float) -> str:
    """Format a decimal as basis points."""
    return f"{value * 10000:.0f} bps"


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_ratio(value: float, name: str) -> float:
    """Validate that a value is between 0 and 1."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def annualize_quarterly(quarterly_value: float) -> float:
    """Convert a quarterly rate to annualized rate."""
    return (1 + quarterly_value) ** 4 - 1


def quarterly_from_annual(annual_value: float) -> float:
    """Convert an annual rate to quarterly rate."""
    return (1 + annual_value) ** 0.25 - 1
