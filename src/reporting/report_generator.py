"""
Report Generator
================
Generates regulatory stress test reports including:
- Capital ratio trajectories
- Loss waterfall analysis
- Risk decomposition
- Pass/fail assessment
- Excel and JSON output
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from ..capital.capital_calculator import StressTestResult, CapitalAdequacyCalculator


class ReportGenerator:
    """Generate regulatory stress testing reports."""

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_capital_trajectory(
        self, result: StressTestResult
    ) -> pd.DataFrame:
        """
        Generate quarter-by-quarter capital ratio trajectory.
        """
        rows = []
        for i, pos in enumerate(result.quarterly_positions):
            rows.append({
                "quarter": f"Q{i}" if i == 0 else f"Q{i}",
                "period": "Initial" if i == 0 else f"Quarter {i}",
                "cet1_capital": pos.cet1_capital,
                "tier1_capital": pos.tier1_capital,
                "total_capital": pos.total_capital,
                "total_rwa": pos.total_rwa,
                "credit_rwa": pos.credit_rwa,
                "market_rwa": pos.market_rwa,
                "operational_rwa": pos.operational_rwa,
                "cet1_ratio": pos.cet1_ratio,
                "tier1_ratio": pos.tier1_ratio,
                "total_capital_ratio": pos.total_capital_ratio,
                "leverage_ratio": pos.leverage_ratio,
                "cumulative_losses": pos.total_credit_losses,
                "cumulative_ppnr": pos.ppnr,
            })
        return pd.DataFrame(rows)

    def generate_loss_decomposition(
        self, results: Dict[str, StressTestResult]
    ) -> pd.DataFrame:
        """Decompose losses by risk type across scenarios."""
        rows = []
        for name, result in results.items():
            final_pos = result.quarterly_positions[-1]
            initial_pos = result.quarterly_positions[0]

            credit_loss_impact = final_pos.total_credit_losses
            rwa_increase = final_pos.total_rwa - initial_pos.total_rwa
            ppnr_offset = final_pos.ppnr

            rows.append({
                "scenario": name,
                "total_credit_losses": credit_loss_impact,
                "rwa_change": rwa_increase,
                "rwa_change_pct": rwa_increase / initial_pos.total_rwa if initial_pos.total_rwa > 0 else 0,
                "total_ppnr": ppnr_offset,
                "net_capital_impact": ppnr_offset - credit_loss_impact,
                "cet1_depletion": initial_pos.cet1_capital - min(
                    p.cet1_capital for p in result.quarterly_positions
                ),
            })
        return pd.DataFrame(rows)

    def generate_pass_fail_summary(
        self,
        results: Dict[str, StressTestResult],
        calculator: CapitalAdequacyCalculator,
    ) -> pd.DataFrame:
        """Generate pass/fail assessment against regulatory minimums."""
        rows = []
        for name, result in results.items():
            shortfall = calculator.compute_capital_shortfall(result)
            rows.append({
                "scenario": name,
                "min_cet1_ratio": result.min_cet1_ratio,
                "min_cet1_required": calculator.MIN_CET1,
                "cet1_buffer": result.min_cet1_ratio - calculator.MIN_CET1,
                "passes_cet1": result.passes_cet1,
                "min_tier1_ratio": result.min_tier1_ratio,
                "passes_tier1": result.passes_tier1,
                "min_total_ratio": result.min_total_capital_ratio,
                "passes_total": result.passes_total_capital,
                "min_leverage_ratio": result.min_leverage_ratio,
                "passes_leverage": result.passes_leverage,
                "passes_all": result.passes_all,
                "cet1_shortfall": shortfall["cet1_shortfall"],
                "requires_action": shortfall["requires_capital_action"],
            })
        return pd.DataFrame(rows)

    def export_to_json(
        self,
        results: Dict[str, StressTestResult],
        filename: str = "stress_test_results.json",
    ) -> str:
        """Export results to JSON format."""
        output = {}
        for name, result in results.items():
            trajectory = self.generate_capital_trajectory(result)
            output[name] = {
                "summary": result.to_summary_dict(),
                "trajectory": trajectory.to_dict(orient="records"),
            }

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        return str(filepath)

    def export_to_excel(
        self,
        results: Dict[str, StressTestResult],
        calculator: CapitalAdequacyCalculator,
        filename: str = "stress_test_report.xlsx",
    ) -> str:
        """Export comprehensive results to Excel."""
        filepath = self.output_dir / filename

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Summary sheet
            summary = calculator.generate_summary_report(results)
            summary.to_excel(writer, sheet_name="Summary", index=False)

            # Pass/Fail
            pf = self.generate_pass_fail_summary(results, calculator)
            pf.to_excel(writer, sheet_name="Pass_Fail", index=False)

            # Loss decomposition
            losses = self.generate_loss_decomposition(results)
            losses.to_excel(writer, sheet_name="Loss_Decomposition", index=False)

            # Per-scenario trajectories
            for name, result in results.items():
                traj = self.generate_capital_trajectory(result)
                sheet_name = name.replace("_", " ").title()[:31]
                traj.to_excel(writer, sheet_name=sheet_name, index=False)

        return str(filepath)

    def print_summary(
        self,
        results: Dict[str, StressTestResult],
        calculator: CapitalAdequacyCalculator,
    ):
        """Print a formatted summary to console."""
        print("\n" + "=" * 80)
        print("  REGULATORY CAPITAL STRESS TEST RESULTS")
        print("  Basel III/IV Framework | CCAR/DFAST Compliance")
        print("=" * 80)

        for name, result in results.items():
            status = "PASS ✓" if result.passes_all else "FAIL ✗"
            print(f"\n  Scenario: {result.scenario_name}")
            print(f"  Status:   {status}")
            print(f"  {'─' * 50}")
            print(f"  Min CET1 Ratio:          {result.min_cet1_ratio:.2%}  (min: {calculator.MIN_CET1:.1%})")
            print(f"  Min Tier 1 Ratio:        {result.min_tier1_ratio:.2%}  (min: {calculator.MIN_TIER1:.1%})")
            print(f"  Min Total Capital Ratio:  {result.min_total_capital_ratio:.2%}  (min: {calculator.MIN_TOTAL_CAPITAL:.1%})")
            print(f"  Min Leverage Ratio:       {result.min_leverage_ratio:.2%}  (min: {calculator.MIN_LEVERAGE:.1%})")
            print(f"  Total Losses:             ${result.total_losses / 1e9:.2f}B")
            print(f"  Total PPNR:               ${result.total_ppnr / 1e9:.2f}B")

            shortfall = calculator.compute_capital_shortfall(result)
            if shortfall["requires_capital_action"]:
                print(f"  ⚠ CET1 Shortfall:        ${shortfall['cet1_shortfall'] / 1e9:.2f}B")

        print("\n" + "=" * 80)
