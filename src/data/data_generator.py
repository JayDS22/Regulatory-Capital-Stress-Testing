"""
Synthetic Portfolio Data Generator
===================================
Generates realistic banking portfolio data for stress testing,
including loan-level attributes, counterparty information,
and market risk positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class PortfolioDataGenerator:
    """
    Generate synthetic banking portfolio data with realistic
    distributions for regulatory capital stress testing.
    """

    ASSET_CLASSES = [
        "corporate", "retail_mortgage", "retail_revolving",
        "retail_other", "sovereign", "bank", "commercial_real_estate"
    ]

    RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]

    RATING_PD_MAP = {
        "AAA": 0.0003, "AA": 0.0008, "A": 0.0020,
        "BBB": 0.0070, "BB": 0.0200, "B": 0.0500,
        "CCC": 0.1500, "Default": 1.0000,
    }

    RATING_WEIGHTS = [0.03, 0.07, 0.15, 0.30, 0.25, 0.12, 0.06, 0.02]

    SECTOR_NAMES = [
        "Technology", "Healthcare", "Energy", "Financial Services",
        "Consumer Discretionary", "Industrials", "Utilities",
        "Real Estate", "Materials", "Communications"
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_loan_portfolio(self, n_loans: int = 5000) -> pd.DataFrame:
        """
        Generate a synthetic loan portfolio with realistic attributes.

        Args:
            n_loans: Number of loans to generate.

        Returns:
            DataFrame with loan-level data.
        """
        ratings = self.rng.choice(
            self.RATINGS, size=n_loans, p=self.RATING_WEIGHTS
        )
        asset_classes = self.rng.choice(
            self.ASSET_CLASSES, size=n_loans,
            p=[0.25, 0.20, 0.10, 0.10, 0.05, 0.10, 0.20]
        )
        sectors = self.rng.choice(self.SECTOR_NAMES, size=n_loans)

        # Exposure at Default (EAD) - lognormal distribution
        ead = self.rng.lognormal(mean=14.5, sigma=1.8, size=n_loans)
        ead = np.clip(ead, 1e4, 5e9)

        # Probability of Default (PD) based on rating
        pd_values = np.array([self.RATING_PD_MAP[r] for r in ratings])
        pd_noise = self.rng.normal(0, 0.001, size=n_loans)
        pd_values = np.clip(pd_values + pd_noise, 0.0001, 1.0)

        # Loss Given Default (LGD)
        lgd_base = self._generate_lgd(asset_classes, n_loans)

        # Maturity in years
        maturity = self._generate_maturity(asset_classes, n_loans)

        # Risk weights (Basel standardized approach)
        risk_weights = self._assign_risk_weights(ratings, asset_classes)

        # Interest rates
        base_rate = 0.05
        spread = pd_values * 3 + self.rng.uniform(0.005, 0.03, size=n_loans)
        interest_rate = base_rate + spread

        # Geographic regions
        regions = self.rng.choice(
            ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"],
            size=n_loans, p=[0.40, 0.30, 0.15, 0.10, 0.05]
        )

        df = pd.DataFrame({
            "loan_id": [f"L{str(i).zfill(6)}" for i in range(n_loans)],
            "asset_class": asset_classes,
            "rating": ratings,
            "sector": sectors,
            "region": regions,
            "ead": np.round(ead, 2),
            "pd": np.round(pd_values, 6),
            "lgd": np.round(lgd_base, 4),
            "maturity_years": np.round(maturity, 2),
            "risk_weight": np.round(risk_weights, 4),
            "interest_rate": np.round(interest_rate, 4),
            "is_defaulted": (ratings == "Default").astype(int),
        })

        # Derived columns
        df["expected_loss"] = np.round(df["ead"] * df["pd"] * df["lgd"], 2)
        df["rwa"] = np.round(df["ead"] * df["risk_weight"], 2)

        return df

    def generate_market_positions(self, n_positions: int = 500) -> pd.DataFrame:
        """Generate synthetic trading book / market risk positions."""
        instrument_types = self.rng.choice(
            ["equity", "fixed_income", "fx", "commodity", "derivative"],
            size=n_positions, p=[0.25, 0.30, 0.20, 0.10, 0.15]
        )
        notional = self.rng.lognormal(mean=16, sigma=1.5, size=n_positions)
        notional = np.clip(notional, 1e5, 1e10)

        daily_vol = self.rng.uniform(0.005, 0.05, size=n_positions)
        duration = np.where(
            instrument_types == "fixed_income",
            self.rng.uniform(1, 15, size=n_positions),
            0
        )

        return pd.DataFrame({
            "position_id": [f"P{str(i).zfill(5)}" for i in range(n_positions)],
            "instrument_type": instrument_types,
            "notional": np.round(notional, 2),
            "market_value": np.round(notional * self.rng.uniform(0.8, 1.2, size=n_positions), 2),
            "daily_volatility": np.round(daily_vol, 6),
            "duration": np.round(duration, 2),
            "delta": np.round(self.rng.uniform(-1, 1, size=n_positions), 4),
            "gamma": np.round(self.rng.uniform(-0.1, 0.1, size=n_positions), 6),
        })

    def generate_macro_history(self, n_quarters: int = 40) -> pd.DataFrame:
        """Generate historical macroeconomic time series."""
        quarters = pd.date_range(
            end=pd.Timestamp("2025-12-31"), periods=n_quarters, freq="QE"
        )

        gdp_growth = np.cumsum(self.rng.normal(0.005, 0.008, n_quarters))
        gdp_growth = 0.02 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, n_quarters)) + \
                     self.rng.normal(0, 0.003, n_quarters)

        unemployment = 0.05 + 0.02 * np.sin(np.linspace(0, 3 * np.pi, n_quarters)) + \
                       np.abs(self.rng.normal(0, 0.005, n_quarters))

        fed_funds = 0.03 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, n_quarters)) + \
                    self.rng.normal(0, 0.002, n_quarters)
        fed_funds = np.clip(fed_funds, 0, 0.08)

        credit_spread = 0.015 + 0.01 * np.sin(np.linspace(0, 3 * np.pi, n_quarters)) + \
                        np.abs(self.rng.normal(0, 0.003, n_quarters))

        sp500_return = 0.02 + self.rng.normal(0, 0.04, n_quarters)
        housing_index = 100 + np.cumsum(self.rng.normal(0.5, 1.5, n_quarters))

        return pd.DataFrame({
            "quarter": quarters,
            "gdp_growth": np.round(gdp_growth, 4),
            "unemployment_rate": np.round(np.clip(unemployment, 0.03, 0.15), 4),
            "fed_funds_rate": np.round(fed_funds, 4),
            "credit_spread": np.round(credit_spread, 4),
            "sp500_quarterly_return": np.round(sp500_return, 4),
            "housing_price_index": np.round(housing_index, 2),
            "vix": np.round(15 + 10 * np.abs(self.rng.normal(0, 1, n_quarters)), 2),
            "cpi_yoy": np.round(0.02 + self.rng.normal(0, 0.005, n_quarters), 4),
        })

    def generate_income_statement(self, n_quarters: int = 9) -> pd.DataFrame:
        """Generate projected bank income statement items."""
        quarters = [f"Q{i+1}" for i in range(n_quarters)]
        base_nii = 8.5e9  # Net Interest Income
        base_non_ii = 3.2e9  # Non-Interest Income
        base_expenses = 6.5e9

        nii = [base_nii * (1 + self.rng.normal(0.005, 0.02)) for _ in range(n_quarters)]
        non_ii = [base_non_ii * (1 + self.rng.normal(0.003, 0.03)) for _ in range(n_quarters)]
        expenses = [base_expenses * (1 + self.rng.normal(0.002, 0.01)) for _ in range(n_quarters)]

        return pd.DataFrame({
            "quarter": quarters,
            "net_interest_income": np.round(nii, 0),
            "non_interest_income": np.round(non_ii, 0),
            "operating_expenses": np.round(expenses, 0),
            "pre_provision_net_revenue": np.round(
                np.array(nii) + np.array(non_ii) - np.array(expenses), 0
            ),
        })

    def _generate_lgd(self, asset_classes: np.ndarray, n: int) -> np.ndarray:
        """Generate LGD values based on asset class."""
        lgd_params = {
            "corporate": (0.45, 0.15),
            "retail_mortgage": (0.20, 0.10),
            "retail_revolving": (0.75, 0.10),
            "retail_other": (0.55, 0.15),
            "sovereign": (0.45, 0.20),
            "bank": (0.45, 0.15),
            "commercial_real_estate": (0.35, 0.12),
        }
        lgd = np.zeros(n)
        for ac in np.unique(asset_classes):
            mask = asset_classes == ac
            mu, sigma = lgd_params.get(ac, (0.45, 0.15))
            lgd[mask] = self.rng.beta(
                a=mu * 5, b=(1 - mu) * 5, size=mask.sum()
            )
        return np.clip(lgd, 0.01, 1.0)

    def _generate_maturity(self, asset_classes: np.ndarray, n: int) -> np.ndarray:
        """Generate maturity values based on asset class."""
        mat_params = {
            "corporate": (3.0, 1.5),
            "retail_mortgage": (20.0, 5.0),
            "retail_revolving": (1.0, 0.5),
            "retail_other": (3.0, 1.0),
            "sovereign": (7.0, 3.0),
            "bank": (2.0, 1.0),
            "commercial_real_estate": (10.0, 3.0),
        }
        maturity = np.zeros(n)
        for ac in np.unique(asset_classes):
            mask = asset_classes == ac
            mu, sigma = mat_params.get(ac, (3.0, 1.5))
            maturity[mask] = self.rng.normal(mu, sigma, size=mask.sum())
        return np.clip(maturity, 0.25, 30.0)

    def _assign_risk_weights(
        self, ratings: np.ndarray, asset_classes: np.ndarray
    ) -> np.ndarray:
        """Assign Basel III standardized risk weights."""
        rw_corporate = {
            "AAA": 0.20, "AA": 0.20, "A": 0.50, "BBB": 1.00,
            "BB": 1.00, "B": 1.50, "CCC": 1.50, "Default": 1.50,
        }
        rw_retail = 0.75
        rw_mortgage = 0.35
        rw_cre = 1.00

        n = len(ratings)
        rw = np.ones(n)

        for i in range(n):
            ac = asset_classes[i]
            r = ratings[i]
            if ac in ("retail_revolving", "retail_other"):
                rw[i] = rw_retail
            elif ac == "retail_mortgage":
                rw[i] = rw_mortgage
            elif ac == "commercial_real_estate":
                rw[i] = rw_cre
            elif ac == "sovereign":
                rw[i] = rw_corporate.get(r, 1.0) * 0.5
            else:
                rw[i] = rw_corporate.get(r, 1.0)

        return rw
