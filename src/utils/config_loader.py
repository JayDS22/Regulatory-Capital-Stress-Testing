"""
Configuration loader for the Regulatory Capital Stress Testing platform.
Handles YAML config parsing, validation, and default fallbacks.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG = {
    "capital": {
        "minimum_ratios": {
            "cet1": 0.045,
            "tier1": 0.06,
            "total_capital": 0.08,
            "leverage_ratio": 0.03,
        },
        "buffers": {
            "capital_conservation": 0.025,
            "countercyclical_min": 0.0,
            "countercyclical_max": 0.025,
            "gsib_surcharge": 0.01,
        },
        "risk_weights": {
            "sovereign_aaa": 0.0,
            "sovereign_aa": 0.0,
            "sovereign_a": 0.20,
            "sovereign_bbb": 0.50,
            "sovereign_below_bbb": 1.00,
            "bank_rated": 0.20,
            "corporate_aaa_aa": 0.20,
            "corporate_a": 0.50,
            "corporate_bbb": 1.00,
            "corporate_below_bbb": 1.50,
            "retail": 0.75,
            "residential_mortgage": 0.35,
            "commercial_real_estate": 1.00,
        },
    },
    "scenarios": {
        "baseline": {
            "name": "Baseline",
            "gdp_growth_shock": 0.0,
            "unemployment_shock": 0.0,
            "interest_rate_shock": 0.0,
            "credit_spread_shock": 0.0,
            "equity_market_shock": 0.0,
            "housing_price_shock": 0.0,
            "probability": 0.60,
        },
        "adverse": {
            "name": "Adverse",
            "gdp_growth_shock": -0.025,
            "unemployment_shock": 0.03,
            "interest_rate_shock": 0.015,
            "credit_spread_shock": 0.02,
            "equity_market_shock": -0.20,
            "housing_price_shock": -0.10,
            "probability": 0.30,
        },
        "severely_adverse": {
            "name": "Severely Adverse",
            "gdp_growth_shock": -0.065,
            "unemployment_shock": 0.06,
            "interest_rate_shock": -0.01,
            "credit_spread_shock": 0.05,
            "equity_market_shock": -0.45,
            "housing_price_shock": -0.30,
            "probability": 0.10,
        },
    },
    "models": {
        "credit_loss": {
            "method": "through_the_cycle",
            "correlation_asset": 0.15,
            "maturity_adjustment": True,
        },
        "market_risk": {
            "var_confidence": 0.99,
            "var_horizon_days": 10,
            "stressed_var_window": 252,
        },
        "operational_risk": {
            "method": "standardized_approach",
            "business_indicator_coefficient": 0.12,
        },
    },
    "reporting": {
        "projection_horizon_quarters": 9,
        "confidence_intervals": [0.95, 0.99],
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Falls back to default config if file not found.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing merged configuration.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        # Try default locations
        candidates = [
            Path("configs/stress_test_config.yaml"),
            Path("config.yaml"),
            Path(os.environ.get("STRESS_TEST_CONFIG", "")),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
        if file_config:
            config = _deep_merge(config, file_config)

    return config


def get_default_config() -> Dict[str, Any]:
    """Return the default configuration dictionary."""
    return DEFAULT_CONFIG.copy()


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
