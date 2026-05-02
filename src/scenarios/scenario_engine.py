"""
Scenario Engine
===============
Generates and manages macroeconomic stress scenarios for
CCAR/DFAST regulatory stress testing exercises.

Implements:
- Fed-defined scenario templates (Baseline, Adverse, Severely Adverse)
- Custom scenario generation with macro variable paths
- Scenario interpolation and severity scaling
- Multi-quarter projection with mean reversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MacroScenario:
    """Represents a macroeconomic stress scenario."""
    name: str
    description: str
    n_quarters: int
    gdp_growth: np.ndarray
    unemployment_rate: np.ndarray
    fed_funds_rate: np.ndarray
    ten_year_treasury: np.ndarray
    credit_spread: np.ndarray
    sp500_return: np.ndarray
    housing_price_change: np.ndarray
    vix: np.ndarray
    cpi_inflation: np.ndarray
    probability: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scenario to DataFrame."""
        quarters = [f"Q{i+1}" for i in range(self.n_quarters)]
        return pd.DataFrame({
            "quarter": quarters,
            "gdp_growth": self.gdp_growth,
            "unemployment_rate": self.unemployment_rate,
            "fed_funds_rate": self.fed_funds_rate,
            "ten_year_treasury": self.ten_year_treasury,
            "credit_spread": self.credit_spread,
            "sp500_return": self.sp500_return,
            "housing_price_change": self.housing_price_change,
            "vix": self.vix,
            "cpi_inflation": self.cpi_inflation,
        })


class ScenarioEngine:
    """
    Generate and manage macroeconomic stress scenarios
    for regulatory capital stress testing.
    """

    # Base macro conditions (starting point)
    BASE_MACRO = {
        "gdp_growth": 0.023,
        "unemployment_rate": 0.042,
        "fed_funds_rate": 0.0525,
        "ten_year_treasury": 0.043,
        "credit_spread": 0.015,
        "sp500_return": 0.025,
        "housing_price_change": 0.03,
        "vix": 16.0,
        "cpi_inflation": 0.025,
    }

    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        self.config = config or {}
        self.rng = np.random.default_rng(seed)
        self.n_quarters = self.config.get("projection_horizon_quarters", 9)

    def generate_baseline(self) -> MacroScenario:
        """Generate baseline (expected) economic scenario."""
        n = self.n_quarters
        base = self.BASE_MACRO

        gdp = np.full(n, base["gdp_growth"])
        gdp += self.rng.normal(0, 0.002, n)

        unemployment = np.full(n, base["unemployment_rate"])
        unemployment += np.linspace(0, -0.003, n)

        fed_funds = np.full(n, base["fed_funds_rate"])
        fed_funds += np.linspace(0, -0.005, n)

        ten_yr = np.full(n, base["ten_year_treasury"])
        ten_yr += np.linspace(0, -0.003, n)

        credit_spread = np.full(n, base["credit_spread"])
        sp500 = np.full(n, base["sp500_return"])
        sp500 += self.rng.normal(0, 0.01, n)

        housing = np.full(n, base["housing_price_change"])
        vix = np.full(n, base["vix"])
        vix += self.rng.normal(0, 1, n)

        cpi = np.full(n, base["cpi_inflation"])

        return MacroScenario(
            name="Baseline",
            description="Expected economic conditions with moderate growth",
            n_quarters=n,
            gdp_growth=np.round(gdp, 4),
            unemployment_rate=np.round(np.clip(unemployment, 0.03, 0.15), 4),
            fed_funds_rate=np.round(np.clip(fed_funds, 0, 0.10), 4),
            ten_year_treasury=np.round(np.clip(ten_yr, 0.005, 0.10), 4),
            credit_spread=np.round(credit_spread, 4),
            sp500_return=np.round(sp500, 4),
            housing_price_change=np.round(housing, 4),
            vix=np.round(np.clip(vix, 10, 50), 1),
            cpi_inflation=np.round(cpi, 4),
            probability=0.60,
        )

    def generate_adverse(self) -> MacroScenario:
        """Generate adverse stress scenario."""
        n = self.n_quarters
        base = self.BASE_MACRO

        # GDP contracts for 3 quarters, then slow recovery
        gdp_path = np.array([
            -0.01, -0.02, -0.025, -0.015, -0.005,
            0.005, 0.01, 0.015, 0.018
        ])[:n]

        # Unemployment rises over 5 quarters, then plateaus
        unemp_path = base["unemployment_rate"] + np.array([
            0.005, 0.012, 0.020, 0.028, 0.032,
            0.030, 0.027, 0.023, 0.020
        ])[:n]

        # Fed cuts rates in response
        fed_path = base["fed_funds_rate"] + np.array([
            -0.005, -0.0125, -0.02, -0.025, -0.025,
            -0.02, -0.015, -0.01, -0.005
        ])[:n]

        ten_yr_path = base["ten_year_treasury"] + np.array([
            -0.003, -0.008, -0.012, -0.010, -0.008,
            -0.005, -0.003, -0.001, 0.002
        ])[:n]

        spread_path = base["credit_spread"] + np.array([
            0.005, 0.012, 0.020, 0.018, 0.015,
            0.012, 0.010, 0.008, 0.005
        ])[:n]

        sp500_path = np.array([
            -0.05, -0.10, -0.08, -0.03, 0.01,
            0.03, 0.04, 0.03, 0.02
        ])[:n]

        housing_path = np.array([
            0.01, -0.02, -0.05, -0.08, -0.06,
            -0.04, -0.02, 0.0, 0.01
        ])[:n]

        vix_path = np.array([
            22, 28, 35, 30, 25, 22, 20, 18, 17
        ])[:n].astype(float)

        cpi_path = np.array([
            0.025, 0.020, 0.015, 0.012, 0.010,
            0.012, 0.015, 0.018, 0.020
        ])[:n]

        return MacroScenario(
            name="Adverse",
            description="Moderate recession with elevated unemployment and market stress",
            n_quarters=n,
            gdp_growth=np.round(gdp_path, 4),
            unemployment_rate=np.round(np.clip(unemp_path, 0.03, 0.15), 4),
            fed_funds_rate=np.round(np.clip(fed_path, 0, 0.10), 4),
            ten_year_treasury=np.round(np.clip(ten_yr_path, 0.005, 0.10), 4),
            credit_spread=np.round(spread_path, 4),
            sp500_return=np.round(sp500_path, 4),
            housing_price_change=np.round(housing_path, 4),
            vix=np.round(vix_path, 1),
            cpi_inflation=np.round(cpi_path, 4),
            probability=0.30,
        )

    def generate_severely_adverse(self) -> MacroScenario:
        """Generate severely adverse stress scenario (deep recession)."""
        n = self.n_quarters
        base = self.BASE_MACRO

        gdp_path = np.array([
            -0.03, -0.055, -0.065, -0.045, -0.025,
            -0.01, 0.005, 0.01, 0.015
        ])[:n]

        unemp_path = base["unemployment_rate"] + np.array([
            0.01, 0.025, 0.045, 0.060, 0.065,
            0.060, 0.052, 0.045, 0.038
        ])[:n]

        fed_path = np.array([
            0.04, 0.025, 0.01, 0.005, 0.0025,
            0.0025, 0.005, 0.0075, 0.01
        ])[:n]

        ten_yr_path = np.array([
            0.035, 0.025, 0.015, 0.012, 0.015,
            0.018, 0.022, 0.025, 0.028
        ])[:n]

        spread_path = base["credit_spread"] + np.array([
            0.015, 0.035, 0.055, 0.050, 0.040,
            0.030, 0.022, 0.015, 0.010
        ])[:n]

        sp500_path = np.array([
            -0.12, -0.20, -0.15, -0.08, -0.02,
            0.05, 0.08, 0.06, 0.04
        ])[:n]

        housing_path = np.array([
            -0.02, -0.08, -0.15, -0.20, -0.18,
            -0.12, -0.06, -0.02, 0.01
        ])[:n]

        vix_path = np.array([
            30, 45, 65, 55, 40, 30, 25, 22, 18
        ])[:n].astype(float)

        cpi_path = np.array([
            0.020, 0.010, 0.005, 0.002, 0.000,
            0.005, 0.010, 0.015, 0.018
        ])[:n]

        return MacroScenario(
            name="Severely Adverse",
            description="Deep recession with severe financial market disruption and housing collapse",
            n_quarters=n,
            gdp_growth=np.round(gdp_path, 4),
            unemployment_rate=np.round(np.clip(unemp_path, 0.03, 0.20), 4),
            fed_funds_rate=np.round(np.clip(fed_path, 0, 0.10), 4),
            ten_year_treasury=np.round(np.clip(ten_yr_path, 0.005, 0.10), 4),
            credit_spread=np.round(spread_path, 4),
            sp500_return=np.round(sp500_path, 4),
            housing_price_change=np.round(housing_path, 4),
            vix=np.round(vix_path, 1),
            cpi_inflation=np.round(cpi_path, 4),
            probability=0.10,
        )

    def generate_all_scenarios(self) -> Dict[str, MacroScenario]:
        """Generate all three standard regulatory scenarios."""
        return {
            "baseline": self.generate_baseline(),
            "adverse": self.generate_adverse(),
            "severely_adverse": self.generate_severely_adverse(),
        }

    def generate_custom_scenario(
        self,
        name: str,
        gdp_shock: float,
        unemployment_shock: float,
        rate_shock: float,
        equity_shock: float,
        housing_shock: float,
        spread_shock: float,
        recovery_speed: float = 0.15,
    ) -> MacroScenario:
        """
        Generate a custom stress scenario with specified shock magnitudes
        and mean-reverting recovery.

        Args:
            name: Scenario name
            gdp_shock: Peak GDP growth shock (e.g., -0.05)
            unemployment_shock: Peak unemployment increase
            rate_shock: Interest rate change
            equity_shock: Equity market decline
            housing_shock: Housing price decline
            spread_shock: Credit spread widening
            recovery_speed: Mean reversion speed (0-1)
        """
        n = self.n_quarters
        base = self.BASE_MACRO

        def shock_path(peak_shock: float, base_val: float, is_additive: bool = True) -> np.ndarray:
            """Create shock path: ramp to peak, then mean-revert."""
            peak_q = min(3, n - 1)
            path = np.zeros(n)
            # Ramp to peak
            for i in range(peak_q + 1):
                path[i] = peak_shock * (i / peak_q)
            # Mean reversion
            for i in range(peak_q + 1, n):
                path[i] = path[i - 1] * (1 - recovery_speed)

            if is_additive:
                return base_val + path
            return path

        return MacroScenario(
            name=name,
            description=f"Custom scenario: GDP {gdp_shock:+.1%}, Unemp {unemployment_shock:+.1%}",
            n_quarters=n,
            gdp_growth=np.round(shock_path(gdp_shock, 0, False), 4),
            unemployment_rate=np.round(
                np.clip(shock_path(unemployment_shock, base["unemployment_rate"]), 0.03, 0.20), 4
            ),
            fed_funds_rate=np.round(
                np.clip(shock_path(rate_shock, base["fed_funds_rate"]), 0, 0.10), 4
            ),
            ten_year_treasury=np.round(
                np.clip(shock_path(rate_shock * 0.6, base["ten_year_treasury"]), 0.005, 0.10), 4
            ),
            credit_spread=np.round(
                np.clip(shock_path(spread_shock, base["credit_spread"]), 0.001, 0.10), 4
            ),
            sp500_return=np.round(shock_path(equity_shock, 0, False), 4),
            housing_price_change=np.round(shock_path(housing_shock, 0, False), 4),
            vix=np.round(np.clip(
                shock_path(abs(equity_shock) * 100, base["vix"]), 10, 80
            ), 1),
            cpi_inflation=np.round(
                shock_path(-abs(gdp_shock) * 0.3, base["cpi_inflation"]), 4
            ),
            probability=0.0,
        )

    def interpolate_scenarios(
        self,
        scenario_a: MacroScenario,
        scenario_b: MacroScenario,
        weight: float = 0.5,
    ) -> MacroScenario:
        """
        Create an interpolated scenario between two scenarios.
        weight=0 returns scenario_a, weight=1 returns scenario_b.
        """
        def interp(a, b):
            return (1 - weight) * a + weight * b

        return MacroScenario(
            name=f"Interpolated ({scenario_a.name} → {scenario_b.name}, w={weight:.2f})",
            description=f"Interpolation between {scenario_a.name} and {scenario_b.name}",
            n_quarters=self.n_quarters,
            gdp_growth=np.round(interp(scenario_a.gdp_growth, scenario_b.gdp_growth), 4),
            unemployment_rate=np.round(interp(scenario_a.unemployment_rate, scenario_b.unemployment_rate), 4),
            fed_funds_rate=np.round(interp(scenario_a.fed_funds_rate, scenario_b.fed_funds_rate), 4),
            ten_year_treasury=np.round(interp(scenario_a.ten_year_treasury, scenario_b.ten_year_treasury), 4),
            credit_spread=np.round(interp(scenario_a.credit_spread, scenario_b.credit_spread), 4),
            sp500_return=np.round(interp(scenario_a.sp500_return, scenario_b.sp500_return), 4),
            housing_price_change=np.round(interp(scenario_a.housing_price_change, scenario_b.housing_price_change), 4),
            vix=np.round(interp(scenario_a.vix, scenario_b.vix), 1),
            cpi_inflation=np.round(interp(scenario_a.cpi_inflation, scenario_b.cpi_inflation), 4),
            probability=0.0,
        )
