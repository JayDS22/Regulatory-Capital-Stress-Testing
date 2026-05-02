"""
Data Validation Module
======================
Validates portfolio data integrity, completeness, and
regulatory compliance requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"Validation Status: {status}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class DataValidator:
    """Validate portfolio data for stress testing."""

    REQUIRED_LOAN_COLUMNS = [
        "loan_id", "asset_class", "rating", "ead", "pd", "lgd",
        "maturity_years", "risk_weight"
    ]

    VALID_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]

    VALID_ASSET_CLASSES = [
        "corporate", "retail_mortgage", "retail_revolving",
        "retail_other", "sovereign", "bank", "commercial_real_estate"
    ]

    def validate_loan_portfolio(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run comprehensive validation on a loan portfolio DataFrame.

        Args:
            df: Loan portfolio DataFrame.

        Returns:
            ValidationResult with errors, warnings, and statistics.
        """
        result = ValidationResult()

        # Check required columns
        missing_cols = set(self.REQUIRED_LOAN_COLUMNS) - set(df.columns)
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
            return result

        # Check for duplicates
        if df["loan_id"].duplicated().any():
            n_dupes = df["loan_id"].duplicated().sum()
            result.add_error(f"Found {n_dupes} duplicate loan_ids")

        # Check for nulls
        null_counts = df[self.REQUIRED_LOAN_COLUMNS].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                result.add_error(f"Column '{col}' has {count} null values")

        # Validate ranges
        if (df["pd"] < 0).any() or (df["pd"] > 1).any():
            result.add_error("PD values must be between 0 and 1")

        if (df["lgd"] < 0).any() or (df["lgd"] > 1).any():
            result.add_error("LGD values must be between 0 and 1")

        if (df["ead"] < 0).any():
            result.add_error("EAD values must be non-negative")

        if (df["maturity_years"] <= 0).any():
            result.add_error("Maturity must be positive")

        if (df["risk_weight"] < 0).any():
            result.add_error("Risk weights must be non-negative")

        # Validate categorical values
        invalid_ratings = set(df["rating"].unique()) - set(self.VALID_RATINGS)
        if invalid_ratings:
            result.add_error(f"Invalid ratings found: {invalid_ratings}")

        invalid_ac = set(df["asset_class"].unique()) - set(self.VALID_ASSET_CLASSES)
        if invalid_ac:
            result.add_error(f"Invalid asset classes found: {invalid_ac}")

        # Concentration warnings
        for ac in df["asset_class"].unique():
            pct = (df["asset_class"] == ac).mean()
            if pct > 0.50:
                result.add_warning(
                    f"High concentration in {ac}: {pct:.1%} of portfolio"
                )

        # Default rate warning
        default_rate = (df["rating"] == "Default").mean()
        if default_rate > 0.10:
            result.add_warning(f"High default rate: {default_rate:.1%}")

        # Collect statistics
        result.stats = {
            "n_loans": len(df),
            "total_ead": df["ead"].sum(),
            "total_rwa": df["rwa"].sum() if "rwa" in df.columns else None,
            "avg_pd": df["pd"].mean(),
            "avg_lgd": df["lgd"].mean(),
            "default_rate": default_rate,
            "asset_class_distribution": df["asset_class"].value_counts().to_dict(),
            "rating_distribution": df["rating"].value_counts().to_dict(),
        }

        return result

    def validate_market_positions(self, df: pd.DataFrame) -> ValidationResult:
        """Validate market risk positions."""
        result = ValidationResult()

        required = ["position_id", "instrument_type", "notional", "market_value", "daily_volatility"]
        missing = set(required) - set(df.columns)
        if missing:
            result.add_error(f"Missing columns: {missing}")
            return result

        if (df["notional"] <= 0).any():
            result.add_error("Notional values must be positive")

        if (df["daily_volatility"] < 0).any():
            result.add_error("Daily volatility must be non-negative")

        result.stats = {
            "n_positions": len(df),
            "total_notional": df["notional"].sum(),
            "total_market_value": df["market_value"].sum(),
            "avg_volatility": df["daily_volatility"].mean(),
        }

        return result
