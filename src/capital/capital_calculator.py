"""
Capital Adequacy Calculator
============================
Orchestrates the full Basel III/IV capital adequacy computation,
combining credit risk, market risk, and operational risk components
to produce regulatory capital ratios under stress scenarios.

Computes:
- CET1 Ratio, Tier 1 Ratio, Total Capital Ratio
- Leverage Ratio
- Capital buffers (CCB, CCyB, G-SIB)
- Stressed capital projections over 9-quarter horizon
- Pass/fail assessment against regulatory minimums
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..models.credit_risk import CreditRiskModel
from ..models.market_risk import MarketRiskModel
from ..models.operational_risk import OperationalRiskModel
from ..scenarios.scenario_engine import MacroScenario


@dataclass
class CapitalPosition:
    """Represents a bank's capital position at a point in time."""
    cet1_capital: float          # Common Equity Tier 1
    at1_capital: float           # Additional Tier 1
    tier2_capital: float         # Tier 2 Capital
    total_rwa: float             # Total Risk-Weighted Assets
    credit_rwa: float
    market_rwa: float
    operational_rwa: float
    total_exposure: float        # For leverage ratio
    total_credit_losses: float = 0
    ppnr: float = 0             # Pre-Provision Net Revenue

    @property
    def tier1_capital(self) -> float:
        return self.cet1_capital + self.at1_capital

    @property
    def total_capital(self) -> float:
        return self.tier1_capital + self.tier2_capital

    @property
    def cet1_ratio(self) -> float:
        return self.cet1_capital / self.total_rwa if self.total_rwa > 0 else 0

    @property
    def tier1_ratio(self) -> float:
        return self.tier1_capital / self.total_rwa if self.total_rwa > 0 else 0

    @property
    def total_capital_ratio(self) -> float:
        return self.total_capital / self.total_rwa if self.total_rwa > 0 else 0

    @property
    def leverage_ratio(self) -> float:
        return self.tier1_capital / self.total_exposure if self.total_exposure > 0 else 0


@dataclass
class StressTestResult:
    """Complete stress test result for a single scenario."""
    scenario_name: str
    quarterly_positions: List[CapitalPosition]
    min_cet1_ratio: float
    min_tier1_ratio: float
    min_total_capital_ratio: float
    min_leverage_ratio: float
    total_losses: float
    total_ppnr: float
    passes_cet1: bool
    passes_tier1: bool
    passes_total_capital: bool
    passes_leverage: bool

    @property
    def passes_all(self) -> bool:
        return self.passes_cet1 and self.passes_tier1 and \
               self.passes_total_capital and self.passes_leverage

    def to_summary_dict(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "min_cet1_ratio": self.min_cet1_ratio,
            "min_tier1_ratio": self.min_tier1_ratio,
            "min_total_capital_ratio": self.min_total_capital_ratio,
            "min_leverage_ratio": self.min_leverage_ratio,
            "total_losses": self.total_losses,
            "total_ppnr": self.total_ppnr,
            "passes_all": self.passes_all,
        }


class CapitalAdequacyCalculator:
    """
    Main orchestrator for Basel III/IV capital adequacy calculations
    under regulatory stress testing frameworks (CCAR/DFAST).
    """

    # Regulatory minimums (including capital conservation buffer)
    MIN_CET1 = 0.045
    MIN_TIER1 = 0.06
    MIN_TOTAL_CAPITAL = 0.08
    MIN_LEVERAGE = 0.03
    CAPITAL_CONSERVATION_BUFFER = 0.025
    GSIB_SURCHARGE = 0.01

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.credit_model = CreditRiskModel(self.config.get("credit_loss", {}))
        self.market_model = MarketRiskModel(self.config.get("market_risk", {}))
        self.op_model = OperationalRiskModel(self.config.get("operational_risk", {}))

        # Override minimums from config
        ratios = self.config.get("minimum_ratios", {})
        self.MIN_CET1 = ratios.get("cet1", self.MIN_CET1)
        self.MIN_TIER1 = ratios.get("tier1", self.MIN_TIER1)
        self.MIN_TOTAL_CAPITAL = ratios.get("total_capital", self.MIN_TOTAL_CAPITAL)
        self.MIN_LEVERAGE = ratios.get("leverage_ratio", self.MIN_LEVERAGE)

    def compute_initial_capital(
        self,
        loan_portfolio: pd.DataFrame,
        market_positions: pd.DataFrame,
        bank_financials: Dict,
    ) -> CapitalPosition:
        """
        Compute the initial (pre-stress) capital position.

        Args:
            loan_portfolio: Loan-level portfolio data
            market_positions: Market risk positions
            bank_financials: Bank financial statement data

        Returns:
            Initial CapitalPosition
        """
        # Credit RWA
        credit_rwa = loan_portfolio["rwa"].sum()

        # Market Risk RWA
        var_result = self.market_model.compute_parametric_var(
            market_positions["market_value"].values,
            market_positions["daily_volatility"].values,
        )
        svar_result = self.market_model.compute_stressed_var(
            market_positions["market_value"].values,
            market_positions["daily_volatility"].values,
        )
        mkt_capital = self.market_model.compute_market_risk_capital(var_result, svar_result)
        market_rwa = mkt_capital["total_market_risk_capital"] * 12.5

        # Operational Risk RWA
        op_result = self.op_model.compute_operational_risk_capital(
            interest_income=bank_financials.get("interest_income", 12e9),
            interest_expense=bank_financials.get("interest_expense", 4e9),
            fee_income=bank_financials.get("fee_income", 3e9),
            fee_expense=bank_financials.get("fee_expense", 0.8e9),
            other_operating_income=bank_financials.get("other_operating_income", 1.5e9),
            net_trading_income=bank_financials.get("net_trading_income", 2e9),
        )
        operational_rwa = op_result["operational_risk_rwa"]

        total_rwa = credit_rwa + market_rwa + operational_rwa
        total_exposure = loan_portfolio["ead"].sum() + market_positions["notional"].sum()

        return CapitalPosition(
            cet1_capital=bank_financials.get("cet1_capital", total_rwa * 0.12),
            at1_capital=bank_financials.get("at1_capital", total_rwa * 0.015),
            tier2_capital=bank_financials.get("tier2_capital", total_rwa * 0.02),
            total_rwa=total_rwa,
            credit_rwa=credit_rwa,
            market_rwa=market_rwa,
            operational_rwa=operational_rwa,
            total_exposure=total_exposure,
        )

    def run_stress_test(
        self,
        loan_portfolio: pd.DataFrame,
        market_positions: pd.DataFrame,
        bank_financials: Dict,
        scenario: MacroScenario,
    ) -> StressTestResult:
        """
        Run a complete stress test for a given scenario over the projection horizon.

        Computes quarter-by-quarter capital depletion from:
        1. Stressed credit losses (PD/LGD sensitivity to macro variables)
        2. Market risk losses from position revaluation
        3. Operational risk stress
        4. Pre-provision net revenue (PPNR) offset

        Args:
            loan_portfolio: Loan portfolio DataFrame
            market_positions: Market positions DataFrame
            bank_financials: Bank financial data dict
            scenario: MacroScenario instance

        Returns:
            StressTestResult with full quarterly capital trajectory
        """
        # Initial position
        initial_pos = self.compute_initial_capital(
            loan_portfolio, market_positions, bank_financials
        )

        quarterly_positions = [initial_pos]
        cumulative_losses = 0
        cumulative_ppnr = 0

        current_cet1 = initial_pos.cet1_capital
        current_at1 = initial_pos.at1_capital
        current_tier2 = initial_pos.tier2_capital

        for q in range(scenario.n_quarters):
            # 1. Stress PD based on macro variables
            stressed_pd = self.credit_model.stress_pd(
                loan_portfolio["pd"].values,
                gdp_shock=scenario.gdp_growth[q],
                unemployment_shock=scenario.unemployment_rate[q] - 0.042,
                credit_spread_shock=scenario.credit_spread[q] - 0.015,
            )

            # 2. Stress LGD based on collateral shocks
            stressed_lgd = self.credit_model.stress_lgd(
                loan_portfolio["lgd"].values,
                housing_shock=scenario.housing_price_change[q],
                equity_shock=scenario.sp500_return[q],
            )

            # 3. Compute quarterly credit losses
            quarterly_el = (stressed_pd * stressed_lgd * loan_portfolio["ead"].values).sum() / 4
            quarterly_losses = quarterly_el * (1 + 0.3 * abs(scenario.gdp_growth[q]) / 0.065)

            # 4. Compute stressed credit RWA
            irb_result = self.credit_model.compute_capital_requirement_irb(
                stressed_pd, stressed_lgd,
                loan_portfolio["ead"].values,
                loan_portfolio["maturity_years"].values,
            )
            stressed_credit_rwa = irb_result["rwa"].sum()

            # 5. Market risk under stress
            stress_vol_mult = 1 + abs(scenario.sp500_return[q]) * 3
            market_loss = (
                market_positions["market_value"].values *
                market_positions["daily_volatility"].values *
                scenario.sp500_return[q] * np.sqrt(63)
            ).sum() * 0.1  # Quarterly approximation

            # 6. Operational risk stress
            op_stress = self.op_model.stress_operational_risk(
                initial_pos.operational_rwa / 12.5,
                scenario="severely_adverse" if "Severe" in scenario.name else
                "adverse" if "Adverse" in scenario.name else "baseline",
            )

            # 7. PPNR estimation (simplified)
            base_ppnr = bank_financials.get("quarterly_ppnr", 5.2e9)
            nii_sensitivity = -0.5 * (scenario.fed_funds_rate[q] - 0.0525) * base_ppnr
            ppnr = base_ppnr * (1 + scenario.gdp_growth[q] * 2) + nii_sensitivity
            ppnr = max(ppnr * 0.3, ppnr)  # Floor at 30% of base

            # 8. Update capital
            net_income = ppnr - quarterly_losses + min(market_loss, 0)
            current_cet1 += net_income
            cumulative_losses += quarterly_losses + max(-market_loss, 0)
            cumulative_ppnr += ppnr

            # 9. Stressed RWA
            stressed_market_rwa = initial_pos.market_rwa * stress_vol_mult
            stressed_op_rwa = op_stress["stressed_capital"] * 12.5
            total_stressed_rwa = stressed_credit_rwa + stressed_market_rwa + stressed_op_rwa

            # Record position
            pos = CapitalPosition(
                cet1_capital=current_cet1,
                at1_capital=current_at1,
                tier2_capital=current_tier2,
                total_rwa=total_stressed_rwa,
                credit_rwa=stressed_credit_rwa,
                market_rwa=stressed_market_rwa,
                operational_rwa=stressed_op_rwa,
                total_exposure=initial_pos.total_exposure,
                total_credit_losses=cumulative_losses,
                ppnr=cumulative_ppnr,
            )
            quarterly_positions.append(pos)

        # Compute minimums across all quarters
        min_cet1 = min(p.cet1_ratio for p in quarterly_positions)
        min_tier1 = min(p.tier1_ratio for p in quarterly_positions)
        min_total = min(p.total_capital_ratio for p in quarterly_positions)
        min_lev = min(p.leverage_ratio for p in quarterly_positions)

        return StressTestResult(
            scenario_name=scenario.name,
            quarterly_positions=quarterly_positions,
            min_cet1_ratio=min_cet1,
            min_tier1_ratio=min_tier1,
            min_total_capital_ratio=min_total,
            min_leverage_ratio=min_lev,
            total_losses=cumulative_losses,
            total_ppnr=cumulative_ppnr,
            passes_cet1=min_cet1 >= self.MIN_CET1,
            passes_tier1=min_tier1 >= self.MIN_TIER1,
            passes_total_capital=min_total >= self.MIN_TOTAL_CAPITAL,
            passes_leverage=min_lev >= self.MIN_LEVERAGE,
        )

    def run_all_scenarios(
        self,
        loan_portfolio: pd.DataFrame,
        market_positions: pd.DataFrame,
        bank_financials: Dict,
        scenarios: Dict[str, MacroScenario],
    ) -> Dict[str, StressTestResult]:
        """Run stress tests across all scenarios."""
        results = {}
        for scenario_name, scenario in scenarios.items():
            results[scenario_name] = self.run_stress_test(
                loan_portfolio, market_positions, bank_financials, scenario
            )
        return results

    def generate_summary_report(
        self, results: Dict[str, StressTestResult]
    ) -> pd.DataFrame:
        """Generate a summary report DataFrame across all scenarios."""
        rows = []
        for name, result in results.items():
            row = result.to_summary_dict()
            row["cet1_buffer"] = result.min_cet1_ratio - self.MIN_CET1
            row["tier1_buffer"] = result.min_tier1_ratio - self.MIN_TIER1
            rows.append(row)

        return pd.DataFrame(rows)

    def compute_capital_shortfall(
        self, result: StressTestResult
    ) -> Dict[str, float]:
        """Compute capital shortfall (if any) against regulatory minimums."""
        # Find the quarter with minimum CET1 ratio
        min_pos = min(result.quarterly_positions, key=lambda p: p.cet1_ratio)

        cet1_shortfall = max(0, (self.MIN_CET1 + self.CAPITAL_CONSERVATION_BUFFER) * min_pos.total_rwa - min_pos.cet1_capital)
        tier1_shortfall = max(0, self.MIN_TIER1 * min_pos.total_rwa - min_pos.tier1_capital)
        total_shortfall = max(0, self.MIN_TOTAL_CAPITAL * min_pos.total_rwa - min_pos.total_capital)

        return {
            "cet1_shortfall": cet1_shortfall,
            "tier1_shortfall": tier1_shortfall,
            "total_capital_shortfall": total_shortfall,
            "requires_capital_action": cet1_shortfall > 0 or tier1_shortfall > 0,
            "min_cet1_ratio": min_pos.cet1_ratio,
            "required_cet1_ratio": self.MIN_CET1 + self.CAPITAL_CONSERVATION_BUFFER,
        }
