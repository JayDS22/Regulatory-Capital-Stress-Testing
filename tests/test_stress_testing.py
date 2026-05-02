"""
Test Suite for Regulatory Capital Stress Testing Platform
=========================================================
Comprehensive tests for all modules: data, models, scenarios, capital, reporting.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_generator import PortfolioDataGenerator
from src.data.data_validator import DataValidator
from src.models.credit_risk import CreditRiskModel
from src.models.market_risk import MarketRiskModel
from src.models.operational_risk import OperationalRiskModel
from src.scenarios.scenario_engine import ScenarioEngine
from src.capital.capital_calculator import CapitalAdequacyCalculator
from src.reporting.report_generator import ReportGenerator
from src.utils.config_loader import load_config, get_default_config
from src.utils.helpers import format_currency, format_percentage, format_bps


# ══════════════════════════════════════════════════════════
# Data Generation Tests
# ══════════════════════════════════════════════════════════

class TestPortfolioDataGenerator:

    def setup_method(self):
        self.gen = PortfolioDataGenerator(seed=42)

    def test_loan_portfolio_shape(self):
        df = self.gen.generate_loan_portfolio(1000)
        assert len(df) == 1000
        assert "loan_id" in df.columns
        assert "ead" in df.columns
        assert "pd" in df.columns
        assert "lgd" in df.columns

    def test_loan_portfolio_pd_range(self):
        df = self.gen.generate_loan_portfolio(2000)
        assert df["pd"].min() >= 0
        assert df["pd"].max() <= 1

    def test_loan_portfolio_lgd_range(self):
        df = self.gen.generate_loan_portfolio(2000)
        assert df["lgd"].min() >= 0
        assert df["lgd"].max() <= 1

    def test_loan_portfolio_ead_positive(self):
        df = self.gen.generate_loan_portfolio(1000)
        assert (df["ead"] > 0).all()

    def test_loan_portfolio_rwa_computed(self):
        df = self.gen.generate_loan_portfolio(1000)
        assert "rwa" in df.columns
        assert "expected_loss" in df.columns

    def test_loan_portfolio_unique_ids(self):
        df = self.gen.generate_loan_portfolio(500)
        assert df["loan_id"].nunique() == 500

    def test_market_positions_shape(self):
        df = self.gen.generate_market_positions(200)
        assert len(df) == 200
        assert "position_id" in df.columns

    def test_macro_history_shape(self):
        df = self.gen.generate_macro_history(20)
        assert len(df) == 20
        assert "gdp_growth" in df.columns
        assert "unemployment_rate" in df.columns

    def test_income_statement(self):
        df = self.gen.generate_income_statement(9)
        assert len(df) == 9
        assert "pre_provision_net_revenue" in df.columns

    def test_reproducibility(self):
        gen1 = PortfolioDataGenerator(seed=123)
        gen2 = PortfolioDataGenerator(seed=123)
        df1 = gen1.generate_loan_portfolio(100)
        df2 = gen2.generate_loan_portfolio(100)
        pd.testing.assert_frame_equal(df1, df2)


# ══════════════════════════════════════════════════════════
# Data Validation Tests
# ══════════════════════════════════════════════════════════

class TestDataValidator:

    def setup_method(self):
        self.gen = PortfolioDataGenerator(seed=42)
        self.validator = DataValidator()

    def test_valid_portfolio(self):
        df = self.gen.generate_loan_portfolio(500)
        result = self.validator.validate_loan_portfolio(df)
        assert result.is_valid

    def test_missing_columns(self):
        df = pd.DataFrame({"loan_id": [1, 2], "ead": [100, 200]})
        result = self.validator.validate_loan_portfolio(df)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_invalid_pd(self):
        df = self.gen.generate_loan_portfolio(100)
        df.loc[0, "pd"] = -0.5
        result = self.validator.validate_loan_portfolio(df)
        assert not result.is_valid

    def test_market_validation(self):
        df = self.gen.generate_market_positions(100)
        result = self.validator.validate_market_positions(df)
        assert result.is_valid

    def test_stats_populated(self):
        df = self.gen.generate_loan_portfolio(500)
        result = self.validator.validate_loan_portfolio(df)
        assert result.stats["n_loans"] == 500
        assert result.stats["total_ead"] > 0


# ══════════════════════════════════════════════════════════
# Credit Risk Model Tests
# ══════════════════════════════════════════════════════════

class TestCreditRiskModel:

    def setup_method(self):
        self.model = CreditRiskModel()

    def test_asset_correlation_range(self):
        pd_values = np.array([0.01, 0.05, 0.10, 0.20])
        R = self.model.compute_asset_correlation(pd_values, "corporate")
        assert np.all(R >= 0)
        assert np.all(R <= 1)

    def test_maturity_adjustment(self):
        pd_values = np.array([0.01, 0.05])
        maturity = np.array([2.5, 5.0])
        ma = self.model.compute_maturity_adjustment(pd_values, maturity)
        assert len(ma) == 2
        # MA at 2.5 years should be close to 1
        assert abs(ma[0] - 1.0) < 0.5

    def test_irb_capital_computation(self):
        pd = np.array([0.01, 0.05, 0.10])
        lgd = np.array([0.45, 0.45, 0.45])
        ead = np.array([1e6, 1e6, 1e6])
        maturity = np.array([2.5, 2.5, 2.5])

        result = self.model.compute_capital_requirement_irb(pd, lgd, ead, maturity)
        assert "rwa" in result
        assert "expected_loss" in result
        assert np.all(result["capital_requirement_pct"] >= 0)
        assert np.all(result["capital_requirement_pct"] <= 1)

    def test_higher_pd_means_higher_capital(self):
        lgd = np.array([0.45])
        ead = np.array([1e6])
        maturity = np.array([2.5])

        r1 = self.model.compute_capital_requirement_irb(np.array([0.01]), lgd, ead, maturity)
        r2 = self.model.compute_capital_requirement_irb(np.array([0.10]), lgd, ead, maturity)
        assert r2["rwa"][0] > r1["rwa"][0]

    def test_stress_pd_increases_with_gdp_shock(self):
        base_pd = np.array([0.02, 0.05])
        stressed = self.model.stress_pd(base_pd, gdp_shock=-0.05)
        assert np.all(stressed > base_pd)

    def test_stress_pd_range(self):
        base_pd = np.array([0.01, 0.50, 0.99])
        stressed = self.model.stress_pd(base_pd, gdp_shock=-0.10, unemployment_shock=0.05)
        assert np.all(stressed >= 0)
        assert np.all(stressed <= 1)

    def test_portfolio_loss_distribution(self):
        pd = np.array([0.02] * 100)
        lgd = np.array([0.45] * 100)
        ead = np.array([1e6] * 100)

        result = self.model.compute_portfolio_loss_distribution(
            pd, lgd, ead, n_simulations=1000
        )
        assert result["mean_loss"] > 0
        assert result["var_99"] > result["mean_loss"]
        assert result["var_999"] >= result["var_99"]


# ══════════════════════════════════════════════════════════
# Market Risk Model Tests
# ══════════════════════════════════════════════════════════

class TestMarketRiskModel:

    def setup_method(self):
        self.model = MarketRiskModel()

    def test_parametric_var(self):
        mv = np.array([1e6, 2e6, 1.5e6])
        vol = np.array([0.02, 0.03, 0.015])
        result = self.model.compute_parametric_var(mv, vol)
        assert result["portfolio_var"] > 0
        assert result["diversification_benefit"] >= 0

    def test_historical_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        result = self.model.compute_historical_var(returns, 1e8)
        assert result["historical_var"] > 0
        assert result["expected_shortfall"] >= result["historical_var"]

    def test_monte_carlo_var(self):
        mv = np.array([1e6, 2e6])
        vol = np.array([0.02, 0.03])
        result = self.model.compute_monte_carlo_var(mv, vol, n_simulations=5000)
        assert result["mc_var"] > 0

    def test_stressed_var_higher(self):
        mv = np.array([1e6, 2e6])
        vol = np.array([0.02, 0.03])
        normal = self.model.compute_parametric_var(mv, vol)
        stressed = self.model.compute_stressed_var(mv, vol, stress_multiplier=2.0)
        assert stressed["portfolio_var"] > normal["portfolio_var"]

    def test_irrbb(self):
        notional = np.array([1e8, 2e8])
        duration = np.array([5.0, 3.0])
        result = self.model.compute_irrbb(notional, duration, rate_shock_bps=200)
        assert result["total_dv01"] > 0
        assert result["eve_impact_up"] != 0


# ══════════════════════════════════════════════════════════
# Operational Risk Model Tests
# ══════════════════════════════════════════════════════════

class TestOperationalRiskModel:

    def setup_method(self):
        self.model = OperationalRiskModel()

    def test_business_indicator(self):
        result = self.model.compute_business_indicator(
            interest_income=12e9, interest_expense=4e9,
            fee_income=3e9, fee_expense=0.8e9,
            other_operating_income=1.5e9, net_trading_income=2e9,
        )
        assert result["business_indicator"] > 0

    def test_bi_component_capital(self):
        bic = self.model.compute_bi_component_capital(5e9)
        assert bic > 0

    def test_full_op_risk_capital(self):
        result = self.model.compute_operational_risk_capital(
            interest_income=12e9, interest_expense=4e9,
            fee_income=3e9, fee_expense=0.8e9,
            other_operating_income=1.5e9, net_trading_income=2e9,
        )
        assert result["operational_risk_capital"] > 0
        assert result["operational_risk_rwa"] > 0

    def test_stress_factors(self):
        base = 1e9
        adverse = self.model.stress_operational_risk(base, "adverse")
        severe = self.model.stress_operational_risk(base, "severely_adverse")
        assert adverse["stressed_capital"] > base
        assert severe["stressed_capital"] > adverse["stressed_capital"]


# ══════════════════════════════════════════════════════════
# Scenario Engine Tests
# ══════════════════════════════════════════════════════════

class TestScenarioEngine:

    def setup_method(self):
        self.engine = ScenarioEngine(seed=42)

    def test_baseline_generation(self):
        scenario = self.engine.generate_baseline()
        assert scenario.name == "Baseline"
        assert len(scenario.gdp_growth) == 9

    def test_adverse_generation(self):
        scenario = self.engine.generate_adverse()
        assert scenario.name == "Adverse"
        assert scenario.gdp_growth[2] < 0  # Recession

    def test_severely_adverse_generation(self):
        scenario = self.engine.generate_severely_adverse()
        assert scenario.name == "Severely Adverse"
        assert scenario.gdp_growth[2] < scenario.gdp_growth[0]  # Deeper recession

    def test_all_scenarios(self):
        scenarios = self.engine.generate_all_scenarios()
        assert len(scenarios) == 3
        assert "baseline" in scenarios
        assert "adverse" in scenarios
        assert "severely_adverse" in scenarios

    def test_scenario_to_dataframe(self):
        scenario = self.engine.generate_baseline()
        df = scenario.to_dataframe()
        assert len(df) == 9
        assert "gdp_growth" in df.columns

    def test_custom_scenario(self):
        scenario = self.engine.generate_custom_scenario(
            name="Custom Recession",
            gdp_shock=-0.04,
            unemployment_shock=0.04,
            rate_shock=-0.02,
            equity_shock=-0.30,
            housing_shock=-0.15,
            spread_shock=0.03,
        )
        assert scenario.name == "Custom Recession"
        assert len(scenario.gdp_growth) == 9

    def test_scenario_interpolation(self):
        baseline = self.engine.generate_baseline()
        adverse = self.engine.generate_adverse()
        interp = self.engine.interpolate_scenarios(baseline, adverse, weight=0.5)
        # Interpolated GDP should be between baseline and adverse
        for i in range(9):
            min_gdp = min(baseline.gdp_growth[i], adverse.gdp_growth[i])
            max_gdp = max(baseline.gdp_growth[i], adverse.gdp_growth[i])
            assert min_gdp <= interp.gdp_growth[i] <= max_gdp


# ══════════════════════════════════════════════════════════
# Capital Adequacy Tests
# ══════════════════════════════════════════════════════════

class TestCapitalAdequacyCalculator:

    def setup_method(self):
        self.gen = PortfolioDataGenerator(seed=42)
        self.loans = self.gen.generate_loan_portfolio(1000)
        self.positions = self.gen.generate_market_positions(100)
        self.financials = {
            "cet1_capital": 85e9,
            "at1_capital": 12e9,
            "tier2_capital": 18e9,
            "interest_income": 12e9,
            "interest_expense": 4e9,
            "fee_income": 3e9,
            "fee_expense": 0.8e9,
            "other_operating_income": 1.5e9,
            "net_trading_income": 2e9,
            "quarterly_ppnr": 5.2e9,
        }
        self.calculator = CapitalAdequacyCalculator()
        self.engine = ScenarioEngine(seed=42)

    def test_initial_capital_computation(self):
        pos = self.calculator.compute_initial_capital(
            self.loans, self.positions, self.financials
        )
        assert pos.cet1_capital == 85e9
        assert pos.total_rwa > 0
        assert pos.cet1_ratio > 0

    def test_stress_test_single_scenario(self):
        scenario = self.engine.generate_baseline()
        result = self.calculator.run_stress_test(
            self.loans, self.positions, self.financials, scenario
        )
        assert result.scenario_name == "Baseline"
        assert len(result.quarterly_positions) == 10  # Initial + 9 quarters
        assert result.min_cet1_ratio > 0

    def test_stress_test_all_scenarios(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        assert len(results) == 3

    def test_adverse_worse_than_baseline(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        assert results["adverse"].min_cet1_ratio <= results["baseline"].min_cet1_ratio
        assert results["severely_adverse"].min_cet1_ratio <= results["adverse"].min_cet1_ratio

    def test_summary_report(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        summary = self.calculator.generate_summary_report(results)
        assert len(summary) == 3
        assert "min_cet1_ratio" in summary.columns

    def test_capital_shortfall(self):
        scenario = self.engine.generate_baseline()
        result = self.calculator.run_stress_test(
            self.loans, self.positions, self.financials, scenario
        )
        shortfall = self.calculator.compute_capital_shortfall(result)
        assert "cet1_shortfall" in shortfall
        assert "requires_capital_action" in shortfall


# ══════════════════════════════════════════════════════════
# Reporting Tests
# ══════════════════════════════════════════════════════════

class TestReportGenerator:

    def setup_method(self):
        self.gen = PortfolioDataGenerator(seed=42)
        self.loans = self.gen.generate_loan_portfolio(500)
        self.positions = self.gen.generate_market_positions(50)
        self.financials = {
            "cet1_capital": 85e9, "at1_capital": 12e9, "tier2_capital": 18e9,
            "interest_income": 12e9, "interest_expense": 4e9,
            "fee_income": 3e9, "fee_expense": 0.8e9,
            "other_operating_income": 1.5e9, "net_trading_income": 2e9,
            "quarterly_ppnr": 5.2e9,
        }
        self.calculator = CapitalAdequacyCalculator()
        self.engine = ScenarioEngine(seed=42)
        self.reporter = ReportGenerator("/tmp/test_outputs")

    def test_capital_trajectory(self):
        scenario = self.engine.generate_baseline()
        result = self.calculator.run_stress_test(
            self.loans, self.positions, self.financials, scenario
        )
        trajectory = self.reporter.generate_capital_trajectory(result)
        assert len(trajectory) == 10
        assert "cet1_ratio" in trajectory.columns

    def test_loss_decomposition(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        losses = self.reporter.generate_loss_decomposition(results)
        assert len(losses) == 3

    def test_json_export(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        path = self.reporter.export_to_json(results)
        assert Path(path).exists()

    def test_excel_export(self):
        scenarios = self.engine.generate_all_scenarios()
        results = self.calculator.run_all_scenarios(
            self.loans, self.positions, self.financials, scenarios
        )
        path = self.reporter.export_to_excel(results, self.calculator)
        assert Path(path).exists()


# ══════════════════════════════════════════════════════════
# Utility Tests
# ══════════════════════════════════════════════════════════

class TestUtils:

    def test_format_currency(self):
        assert "B" in format_currency(5e9)
        assert "M" in format_currency(5e6)
        assert "K" in format_currency(5e3)

    def test_format_percentage(self):
        assert format_percentage(0.125) == "12.50%"

    def test_format_bps(self):
        assert "150" in format_bps(0.015)

    def test_config_loader(self):
        config = get_default_config()
        assert "capital" in config
        assert "scenarios" in config


# ══════════════════════════════════════════════════════════
# Integration Test
# ══════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end pipeline integration test."""

    def test_full_pipeline(self):
        from main import run_stress_test_pipeline

        result = run_stress_test_pipeline(
            output_dir="/tmp/test_integration",
            n_loans=500,
            n_market_positions=50,
            seed=42,
            verbose=False,
        )

        assert result is not None
        assert "results" in result
        assert len(result["results"]) == 3
        assert "loan_portfolio" in result
        assert "scenarios" in result

        # Verify all scenarios produced results
        for name in ["baseline", "adverse", "severely_adverse"]:
            assert name in result["results"]
            r = result["results"][name]
            assert r.min_cet1_ratio > 0
            assert len(r.quarterly_positions) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
