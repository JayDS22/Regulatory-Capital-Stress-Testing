"""
Operational Risk Model
======================
Implements Basel III/IV Standardized Measurement Approach (SMA)
for operational risk capital calculation.
"""

import numpy as np
from typing import Dict, Optional


class OperationalRiskModel:
    """
    Operational risk capital model using the Basel III
    Standardized Measurement Approach (SMA).
    """

    # Basel III SMA bucket coefficients
    SMA_BUCKETS = [
        (1e9, 0.12),      # Bucket 1: BI <= 1B
        (3e9, 0.15),      # Bucket 2: 1B < BI <= 3B
        (10e9, 0.18),     # Bucket 3: 3B < BI <= 10B
        (30e9, 0.18),     # Bucket 4: 10B < BI <= 30B
        (float("inf"), 0.18),  # Bucket 5: BI > 30B
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.internal_loss_multiplier = self.config.get("internal_loss_multiplier", 1.0)

    def compute_business_indicator(
        self,
        interest_income: float,
        interest_expense: float,
        fee_income: float,
        fee_expense: float,
        other_operating_income: float,
        net_trading_income: float,
        dividend_income: float = 0,
    ) -> Dict[str, float]:
        """
        Compute the Business Indicator (BI) per Basel III SMA.

        BI = ILDC + SC + FC
        where:
        ILDC = Interest, Lease, and Dividend Component
        SC   = Services Component
        FC   = Financial Component
        """
        # Interest, Lease, Dividend Component
        ildc = abs(interest_income - interest_expense) + dividend_income

        # Services Component
        sc = max(fee_income, fee_expense) + max(other_operating_income, 0)

        # Financial Component
        fc = abs(net_trading_income)

        bi = ildc + sc + fc

        return {
            "business_indicator": bi,
            "ildc_component": ildc,
            "services_component": sc,
            "financial_component": fc,
        }

    def compute_bi_component_capital(self, business_indicator: float) -> float:
        """
        Compute the BI Component (BIC) capital charge.
        Uses marginal coefficients per bucket.
        """
        bic = 0.0
        remaining_bi = business_indicator
        prev_threshold = 0

        for threshold, coeff in self.SMA_BUCKETS:
            bucket_bi = min(remaining_bi, threshold - prev_threshold)
            if bucket_bi <= 0:
                break
            bic += bucket_bi * coeff
            remaining_bi -= bucket_bi
            prev_threshold = threshold

        return bic

    def compute_internal_loss_multiplier(
        self,
        avg_annual_loss: float,
        bi_component: float,
    ) -> float:
        """
        Compute the Internal Loss Multiplier (ILM).
        ILM = ln(exp(1) - 1 + (LC/BIC)^0.8)
        where LC = 15 * avg_annual_loss
        """
        if bi_component <= 0:
            return 1.0

        loss_component = 15 * avg_annual_loss
        ratio = loss_component / bi_component

        ilm = np.log(np.exp(1) - 1 + ratio ** 0.8)
        return max(ilm, 0.0)

    def compute_operational_risk_capital(
        self,
        interest_income: float,
        interest_expense: float,
        fee_income: float,
        fee_expense: float,
        other_operating_income: float,
        net_trading_income: float,
        avg_annual_op_loss: float = 0,
        dividend_income: float = 0,
    ) -> Dict[str, float]:
        """
        Full operational risk capital calculation per Basel III SMA.

        Returns:
            Dictionary with operational risk capital metrics.
        """
        bi_result = self.compute_business_indicator(
            interest_income, interest_expense,
            fee_income, fee_expense,
            other_operating_income, net_trading_income,
            dividend_income,
        )

        bi = bi_result["business_indicator"]
        bic = self.compute_bi_component_capital(bi)

        if avg_annual_op_loss > 0 and bi > 1e9:
            ilm = self.compute_internal_loss_multiplier(avg_annual_op_loss, bic)
        else:
            ilm = 1.0

        op_risk_capital = bic * ilm * self.internal_loss_multiplier

        # RWA = Capital * 12.5
        op_risk_rwa = op_risk_capital * 12.5

        return {
            **bi_result,
            "bi_component_capital": bic,
            "internal_loss_multiplier": ilm,
            "operational_risk_capital": op_risk_capital,
            "operational_risk_rwa": op_risk_rwa,
            "capital_as_pct_bi": op_risk_capital / bi if bi > 0 else 0,
        }

    def stress_operational_risk(
        self,
        base_capital: float,
        scenario: str = "baseline",
    ) -> Dict[str, float]:
        """
        Apply stress scenarios to operational risk capital.
        """
        stress_factors = {
            "baseline": 1.0,
            "adverse": 1.25,
            "severely_adverse": 1.60,
        }

        factor = stress_factors.get(scenario, 1.0)
        stressed_capital = base_capital * factor

        return {
            "base_capital": base_capital,
            "stress_factor": factor,
            "stressed_capital": stressed_capital,
            "scenario": scenario,
            "capital_increase": stressed_capital - base_capital,
        }
