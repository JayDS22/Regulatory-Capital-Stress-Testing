"""
Regulatory Capital Stress Testing - Main Pipeline
==================================================
End-to-end execution of the CCAR/DFAST stress testing pipeline.

Usage:
    python main.py [--config CONFIG_PATH] [--output OUTPUT_DIR] [--scenarios baseline,adverse,severely_adverse]
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.data.data_generator import PortfolioDataGenerator
from src.data.data_validator import DataValidator
from src.models.credit_risk import CreditRiskModel
from src.models.market_risk import MarketRiskModel
from src.models.operational_risk import OperationalRiskModel
from src.scenarios.scenario_engine import ScenarioEngine
from src.capital.capital_calculator import CapitalAdequacyCalculator
from src.reporting.report_generator import ReportGenerator


def run_stress_test_pipeline(
    config_path: str = None,
    output_dir: str = "data/outputs",
    n_loans: int = 5000,
    n_market_positions: int = 500,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Execute the full regulatory capital stress testing pipeline.

    Steps:
    1. Load configuration
    2. Generate synthetic portfolio data
    3. Validate data integrity
    4. Generate macroeconomic scenarios
    5. Run stress tests across all scenarios
    6. Generate reports

    Args:
        config_path: Path to YAML config file
        output_dir: Output directory for reports
        n_loans: Number of loans in synthetic portfolio
        n_market_positions: Number of market risk positions
        seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Dictionary with all results and reports
    """
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("  REGULATORY CAPITAL STRESS TESTING PLATFORM")
        print("  Basel III/IV | CCAR/DFAST Framework")
        print("=" * 70)

    # ─── Step 1: Load Configuration ───
    if verbose:
        print("\n[1/6] Loading configuration...")
    config = load_config(config_path)

    # ─── Step 2: Generate Portfolio Data ───
    if verbose:
        print(f"[2/6] Generating synthetic portfolio (n_loans={n_loans}, n_positions={n_market_positions})...")

    generator = PortfolioDataGenerator(seed=seed)
    loan_portfolio = generator.generate_loan_portfolio(n_loans)
    market_positions = generator.generate_market_positions(n_market_positions)
    macro_history = generator.generate_macro_history(40)
    income_stmt = generator.generate_income_statement(9)

    # Bank financials for capital calculation
    bank_financials = {
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

    if verbose:
        total_ead = loan_portfolio["ead"].sum()
        total_rwa = loan_portfolio["rwa"].sum()
        print(f"    Portfolio EAD: ${total_ead/1e9:.1f}B | RWA: ${total_rwa/1e9:.1f}B")
        print(f"    Market Positions: {len(market_positions)} | Notional: ${market_positions['notional'].sum()/1e9:.1f}B")

    # ─── Step 3: Validate Data ───
    if verbose:
        print("[3/6] Validating data integrity...")

    validator = DataValidator()
    loan_validation = validator.validate_loan_portfolio(loan_portfolio)
    market_validation = validator.validate_market_positions(market_positions)

    if not loan_validation.is_valid:
        print(f"  WARNING: Loan validation issues:\n{loan_validation.summary()}")
    if not market_validation.is_valid:
        print(f"  WARNING: Market validation issues:\n{market_validation.summary()}")

    if verbose:
        print(f"    Loan Portfolio: {'VALID' if loan_validation.is_valid else 'ISSUES FOUND'}")
        print(f"    Market Positions: {'VALID' if market_validation.is_valid else 'ISSUES FOUND'}")

    # ─── Step 4: Generate Scenarios ───
    if verbose:
        print("[4/6] Generating macroeconomic stress scenarios...")

    scenario_engine = ScenarioEngine(
        config=config.get("reporting", {}), seed=seed
    )
    scenarios = scenario_engine.generate_all_scenarios()

    if verbose:
        for name, scenario in scenarios.items():
            print(f"    {scenario.name}: GDP [{scenario.gdp_growth[0]:+.1%} → {scenario.gdp_growth[-1]:+.1%}], "
                  f"Unemp [{scenario.unemployment_rate[0]:.1%} → {scenario.unemployment_rate[-1]:.1%}]")

    # ─── Step 5: Run Stress Tests ───
    if verbose:
        print("[5/6] Running stress tests across all scenarios...")

    calculator = CapitalAdequacyCalculator(config.get("capital", {}))
    results = calculator.run_all_scenarios(
        loan_portfolio, market_positions, bank_financials, scenarios
    )

    if verbose:
        for name, result in results.items():
            status = "✓ PASS" if result.passes_all else "✗ FAIL"
            print(f"    {result.scenario_name}: CET1={result.min_cet1_ratio:.2%} | {status}")

    # ─── Step 6: Generate Reports ───
    if verbose:
        print("[6/6] Generating reports...")

    reporter = ReportGenerator(output_dir)

    # Export reports
    json_path = reporter.export_to_json(results)
    excel_path = reporter.export_to_excel(results, calculator)

    if verbose:
        print(f"    JSON Report: {json_path}")
        print(f"    Excel Report: {excel_path}")

    # Print console summary
    if verbose:
        reporter.print_summary(results, calculator)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Pipeline completed in {elapsed:.2f}s")
        print("=" * 70 + "\n")

    return {
        "config": config,
        "loan_portfolio": loan_portfolio,
        "market_positions": market_positions,
        "macro_history": macro_history,
        "income_statement": income_stmt,
        "scenarios": scenarios,
        "results": results,
        "calculator": calculator,
        "validation": {
            "loans": loan_validation,
            "market": market_validation,
        },
        "reports": {
            "json": json_path,
            "excel": excel_path,
        },
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regulatory Capital Stress Testing Platform"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output", type=str, default="data/outputs",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--n-loans", type=int, default=5000,
        help="Number of loans in synthetic portfolio"
    )
    parser.add_argument(
        "--n-positions", type=int, default=500,
        help="Number of market risk positions"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    run_stress_test_pipeline(
        config_path=args.config,
        output_dir=args.output,
        n_loans=args.n_loans,
        n_market_positions=args.n_positions,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
