"""
Credit Risk Model
=================
Implements Basel III/IV Internal Ratings-Based (IRB) approach for
credit risk capital calculation, including:
- PD estimation with macro-sensitivity
- LGD downturn estimation
- Expected and Unexpected Loss calculation
- Risk-Weighted Assets (RWA) computation
- Vasicek single-factor model for portfolio loss distribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


class CreditRiskModel:
    """
    Basel III/IV IRB Credit Risk Model for capital adequacy computation.
    Implements the Asymptotic Single Risk Factor (ASRF) framework.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.confidence_level = self.config.get("confidence_level", 0.999)
        self.asset_correlation_override = self.config.get("correlation_asset", None)

    def compute_asset_correlation(self, pd_values: np.ndarray, asset_class: str) -> np.ndarray:
        """
        Compute asset correlation per Basel III IRB formula.

        R = 0.12 * (1 - exp(-50*PD))/(1 - exp(-50))
          + 0.24 * (1 - (1 - exp(-50*PD))/(1 - exp(-50)))
        """
        if self.asset_correlation_override is not None:
            return np.full_like(pd_values, self.asset_correlation_override)

        exp_neg50_pd = np.exp(-50 * pd_values)
        exp_neg50 = np.exp(-50)

        weight = (1 - exp_neg50_pd) / (1 - exp_neg50)

        if asset_class in ("retail_mortgage",):
            r = 0.15  # Fixed correlation for residential mortgage
        elif asset_class in ("retail_revolving",):
            r = 0.04  # Fixed for qualifying revolving
        elif asset_class in ("retail_other",):
            r = 0.03 * weight + 0.16 * (1 - weight)
        else:
            # Corporate/Bank/Sovereign
            r = 0.12 * weight + 0.24 * (1 - weight)

        return np.full_like(pd_values, r) if np.isscalar(r) else r

    def compute_maturity_adjustment(self, pd_values: np.ndarray, maturity: np.ndarray) -> np.ndarray:
        """
        Basel III maturity adjustment factor.
        b(PD) = (0.11852 - 0.05478 * ln(PD))^2
        MA = (1 + (M - 2.5) * b) / (1 - 1.5 * b)
        """
        b = (0.11852 - 0.05478 * np.log(np.clip(pd_values, 1e-10, 1))) ** 2
        ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        return np.clip(ma, 0.5, 5.0)

    def compute_capital_requirement_irb(
        self,
        pd: np.ndarray,
        lgd: np.ndarray,
        ead: np.ndarray,
        maturity: np.ndarray,
        asset_class: str = "corporate"
    ) -> Dict[str, np.ndarray]:
        """
        Compute IRB capital requirements per Basel III formula.

        K = LGD * [N((1-R)^-0.5 * G(PD) + (R/(1-R))^0.5 * G(0.999)) - PD] * MA

        Args:
            pd: Probability of Default array
            lgd: Loss Given Default array
            ead: Exposure at Default array
            maturity: Effective maturity array
            asset_class: Asset class for correlation lookup

        Returns:
            Dictionary with capital metrics per exposure.
        """
        pd_clipped = np.clip(pd, 1e-10, 1 - 1e-10)

        # Asset correlation
        R = self.compute_asset_correlation(pd_clipped, asset_class)

        # Conditional PD under stressed scenario (Vasicek formula)
        G_pd = stats.norm.ppf(pd_clipped)
        G_conf = stats.norm.ppf(self.confidence_level)

        conditional_pd = stats.norm.cdf(
            (np.sqrt(1 / (1 - R)) * G_pd) + (np.sqrt(R / (1 - R)) * G_conf)
        )

        # Maturity adjustment
        if asset_class not in ("retail_mortgage", "retail_revolving", "retail_other"):
            ma = self.compute_maturity_adjustment(pd_clipped, maturity)
        else:
            ma = np.ones_like(pd_clipped)

        # Capital requirement per unit EAD
        K = lgd * (conditional_pd - pd_clipped) * ma
        K = np.clip(K, 0, 1)

        # Risk-weighted assets
        rwa = K * 12.5 * ead

        # Expected Loss
        el = pd_clipped * lgd * ead

        # Unexpected Loss
        ul = K * ead

        return {
            "capital_requirement_pct": K,
            "rwa": rwa,
            "expected_loss": el,
            "unexpected_loss": ul,
            "conditional_pd": conditional_pd,
            "asset_correlation": R if isinstance(R, np.ndarray) else np.full_like(pd, R),
            "maturity_adjustment": ma,
        }

    def compute_portfolio_loss_distribution(
        self,
        pd: np.ndarray,
        lgd: np.ndarray,
        ead: np.ndarray,
        n_simulations: int = 10000,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of portfolio credit losses
        using the single-factor Vasicek model.

        Returns:
            Dictionary with loss distribution statistics.
        """
        rng = np.random.default_rng(seed)
        R = 0.15  # Average asset correlation
        total_ead = ead.sum()
        n_exposures = len(pd)

        portfolio_losses = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Systematic factor
            Z = rng.standard_normal()

            # Idiosyncratic factors
            epsilon = rng.standard_normal(n_exposures)

            # Latent variable
            X = np.sqrt(R) * Z + np.sqrt(1 - R) * epsilon

            # Default indicator
            default_threshold = stats.norm.ppf(pd)
            defaults = (X < default_threshold).astype(float)

            # Portfolio loss
            portfolio_losses[sim] = np.sum(defaults * lgd * ead)

        return {
            "mean_loss": float(np.mean(portfolio_losses)),
            "median_loss": float(np.median(portfolio_losses)),
            "std_loss": float(np.std(portfolio_losses)),
            "var_95": float(np.percentile(portfolio_losses, 95)),
            "var_99": float(np.percentile(portfolio_losses, 99)),
            "var_999": float(np.percentile(portfolio_losses, 99.9)),
            "expected_shortfall_99": float(
                np.mean(portfolio_losses[portfolio_losses >= np.percentile(portfolio_losses, 99)])
            ),
            "max_loss": float(np.max(portfolio_losses)),
            "total_ead": float(total_ead),
            "loss_rate_mean": float(np.mean(portfolio_losses) / total_ead),
            "loss_rate_99": float(np.percentile(portfolio_losses, 99) / total_ead),
        }

    def stress_pd(
        self,
        base_pd: np.ndarray,
        gdp_shock: float = 0.0,
        unemployment_shock: float = 0.0,
        credit_spread_shock: float = 0.0,
    ) -> np.ndarray:
        """
        Apply macroeconomic stress to PD values.
        Uses a log-linear sensitivity model:
        ln(PD_stressed) = ln(PD_base) + β_gdp * ΔGDP + β_unemp * ΔUnemp + β_spread * ΔSpread

        Args:
            base_pd: Base probability of default values
            gdp_shock: GDP growth shock (e.g., -0.05 for 5% decline)
            unemployment_shock: Unemployment rate increase
            credit_spread_shock: Credit spread widening

        Returns:
            Stressed PD values
        """
        beta_gdp = -3.0        # GDP decline increases PD
        beta_unemp = 5.0       # Unemployment rise increases PD
        beta_spread = 8.0      # Spread widening increases PD

        log_pd = np.log(np.clip(base_pd, 1e-10, 1))
        log_pd_stressed = (
            log_pd
            + beta_gdp * gdp_shock
            + beta_unemp * unemployment_shock
            + beta_spread * credit_spread_shock
        )

        return np.clip(np.exp(log_pd_stressed), 1e-10, 1.0)

    def stress_lgd(
        self,
        base_lgd: np.ndarray,
        housing_shock: float = 0.0,
        equity_shock: float = 0.0,
    ) -> np.ndarray:
        """
        Apply stress to LGD values based on collateral value shocks.
        """
        # Housing price decline increases LGD on secured exposures
        lgd_adjustment = -0.3 * housing_shock + -0.15 * equity_shock
        stressed_lgd = base_lgd + lgd_adjustment
        return np.clip(stressed_lgd, 0.01, 1.0)
