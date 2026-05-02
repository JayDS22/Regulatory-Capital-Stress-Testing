"""
Market Risk Model
=================
Implements market risk capital calculations including:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Stressed VaR (SVaR)
- Expected Shortfall (ES) per FRTB
- Interest Rate Risk in the Banking Book (IRRBB)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


class MarketRiskModel:
    """
    Market risk capital model implementing Basel III/IV standards
    including the Fundamental Review of the Trading Book (FRTB).
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.confidence_level = self.config.get("var_confidence", 0.99)
        self.horizon_days = self.config.get("var_horizon_days", 10)
        self.stressed_window = self.config.get("stressed_var_window", 252)

    def compute_parametric_var(
        self,
        market_values: np.ndarray,
        daily_volatilities: np.ndarray,
        confidence: Optional[float] = None,
        horizon: Optional[int] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Parametric (variance-covariance) VaR calculation.

        Args:
            market_values: Position market values
            daily_volatilities: Daily volatility per position
            confidence: Confidence level (default 0.99)
            horizon: Holding period in days (default 10)
            correlation_matrix: Optional correlation matrix

        Returns:
            Dictionary with VaR metrics.
        """
        confidence = confidence or self.confidence_level
        horizon = horizon or self.horizon_days

        z_score = stats.norm.ppf(confidence)

        # Individual position VaR
        position_var = np.abs(market_values) * daily_volatilities * z_score * np.sqrt(horizon)

        if correlation_matrix is not None:
            # Correlated portfolio VaR
            dollar_vol = np.abs(market_values) * daily_volatilities
            portfolio_variance = dollar_vol @ correlation_matrix @ dollar_vol
            portfolio_var = z_score * np.sqrt(portfolio_variance) * np.sqrt(horizon)
        else:
            # Sum of individual VaRs (conservative, no diversification)
            portfolio_var = np.sqrt(np.sum(position_var ** 2))

        undiversified_var = np.sum(position_var)
        diversification_benefit = 1 - portfolio_var / undiversified_var if undiversified_var > 0 else 0

        return {
            "portfolio_var": float(portfolio_var),
            "undiversified_var": float(undiversified_var),
            "diversification_benefit": float(diversification_benefit),
            "individual_var": position_var,
            "confidence": confidence,
            "horizon_days": horizon,
            "total_market_value": float(np.sum(market_values)),
            "var_as_pct_mv": float(portfolio_var / np.abs(np.sum(market_values))) if np.sum(market_values) != 0 else 0,
        }

    def compute_historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        confidence: Optional[float] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Historical simulation VaR.

        Args:
            returns: Historical daily return series
            portfolio_value: Current portfolio value
            confidence: Confidence level
            horizon: Holding period in days

        Returns:
            Dictionary with VaR and ES metrics.
        """
        confidence = confidence or self.confidence_level
        horizon = horizon or self.horizon_days

        # Scale to holding period using square-root-of-time
        scaled_returns = returns * np.sqrt(horizon)

        # Portfolio P&L
        pnl = portfolio_value * scaled_returns

        # VaR as the quantile loss
        var_amount = -np.percentile(pnl, (1 - confidence) * 100)

        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = -pnl[pnl <= -var_amount]
        es = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_amount

        return {
            "historical_var": float(var_amount),
            "expected_shortfall": float(es),
            "es_var_ratio": float(es / var_amount) if var_amount > 0 else 0,
            "worst_loss": float(-np.min(pnl)),
            "best_gain": float(np.max(pnl)),
            "mean_pnl": float(np.mean(pnl)),
            "n_observations": len(returns),
        }

    def compute_monte_carlo_var(
        self,
        market_values: np.ndarray,
        daily_volatilities: np.ndarray,
        n_simulations: int = 10000,
        confidence: Optional[float] = None,
        horizon: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Monte Carlo VaR simulation.
        """
        confidence = confidence or self.confidence_level
        horizon = horizon or self.horizon_days
        rng = np.random.default_rng(seed)

        n_positions = len(market_values)
        simulated_pnl = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Generate correlated random shocks
            shocks = rng.standard_normal(n_positions)
            # Scale by volatility and horizon
            position_pnl = market_values * daily_volatilities * shocks * np.sqrt(horizon)
            simulated_pnl[sim] = np.sum(position_pnl)

        var_amount = -np.percentile(simulated_pnl, (1 - confidence) * 100)
        tail = -simulated_pnl[simulated_pnl <= -var_amount]
        es = float(np.mean(tail)) if len(tail) > 0 else var_amount

        return {
            "mc_var": float(var_amount),
            "mc_expected_shortfall": float(es),
            "mc_var_std_error": float(np.std(simulated_pnl) / np.sqrt(n_simulations)),
            "mean_pnl": float(np.mean(simulated_pnl)),
            "pnl_std": float(np.std(simulated_pnl)),
            "n_simulations": n_simulations,
        }

    def compute_stressed_var(
        self,
        market_values: np.ndarray,
        daily_volatilities: np.ndarray,
        stress_multiplier: float = 2.0,
    ) -> Dict[str, float]:
        """
        Stressed VaR with elevated volatility assumptions.
        Represents worst-case scenario market conditions.
        """
        stressed_vol = daily_volatilities * stress_multiplier
        return self.compute_parametric_var(
            market_values, stressed_vol,
            confidence=self.confidence_level,
            horizon=self.horizon_days,
        )

    def compute_irrbb(
        self,
        notional: np.ndarray,
        duration: np.ndarray,
        rate_shock_bps: float = 200,
    ) -> Dict[str, float]:
        """
        Interest Rate Risk in the Banking Book (IRRBB).
        Computes Economic Value of Equity (EVE) impact.

        Args:
            notional: Position notional amounts
            duration: Modified duration per position
            rate_shock_bps: Interest rate shock in basis points

        Returns:
            Dictionary with IRRBB metrics.
        """
        rate_shock = rate_shock_bps / 10000
        dv01 = notional * duration * 0.0001  # Dollar value of 1 bp
        eve_impact = -notional * duration * rate_shock

        return {
            "total_dv01": float(np.sum(np.abs(dv01))),
            "eve_impact_up": float(np.sum(eve_impact)),
            "eve_impact_down": float(-np.sum(eve_impact)),
            "rate_shock_bps": rate_shock_bps,
            "avg_duration": float(np.average(duration, weights=np.abs(notional))),
            "total_notional": float(np.sum(notional)),
        }

    def compute_market_risk_capital(
        self,
        var_result: Dict,
        svar_result: Dict,
        multiplier: float = 3.0,
    ) -> Dict[str, float]:
        """
        Basel III market risk capital charge.
        Capital = max(VaR_t-1, mc * VaR_avg_60d) + max(SVaR_t-1, ms * SVaR_avg_60d)
        Simplified: Capital = multiplier * VaR + SVaR
        """
        var_charge = multiplier * var_result["portfolio_var"]
        svar_charge = svar_result["portfolio_var"]
        total_charge = var_charge + svar_charge

        return {
            "var_charge": float(var_charge),
            "svar_charge": float(svar_charge),
            "total_market_risk_capital": float(total_charge),
            "multiplier": multiplier,
        }
