# 🏛️ Regulatory Capital Stress Testing Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Basel III/IV](https://img.shields.io/badge/Framework-Basel%20III%2FIV-darkblue.svg)]()
[![CCAR/DFAST](https://img.shields.io/badge/Compliance-CCAR%2FDFAST-red.svg)]()

**Enterprise-grade banking regulatory compliance platform for automated stress testing and capital adequacy analysis under Basel III/IV frameworks with CCAR/DFAST scenario simulation.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Modules Deep Dive](#modules-deep-dive)
- [Configuration](#configuration)
- [Testing](#testing)
- [Results & Output](#results--output)
- [Regulatory Framework Reference](#regulatory-framework-reference)
- [Technologies](#technologies)
- [Author](#author)

---

## Overview

This platform implements a complete regulatory capital stress testing pipeline aligned with Federal Reserve CCAR (Comprehensive Capital Analysis and Review) and DFAST (Dodd-Frank Act Stress Testing) requirements. It computes capital adequacy ratios under baseline, adverse, and severely adverse macroeconomic scenarios using Basel III/IV Internal Ratings-Based (IRB) methodologies.

The system processes a bank's credit portfolio, market risk positions, and operational risk exposure through stressed macroeconomic paths to project capital ratio trajectories over a 9-quarter horizon, producing regulatory-ready reports with pass/fail assessments.

### Key Performance Metrics

| Metric | Value |
|--------|-------|
| Portfolio Coverage | 5,000+ loan exposures, 500+ market positions |
| Scenario Horizon | 9-quarter forward projection |
| Monte Carlo Simulations | 10,000 loss distribution paths |
| VaR Methods | Parametric, Historical, Monte Carlo |
| Pipeline Execution | < 1 second end-to-end |
| Test Coverage | 53 tests, 100% pass rate |

---

## Key Features

- **Basel III/IV IRB Engine** — Full Asymptotic Single Risk Factor (ASRF) model with Vasicek single-factor portfolio loss distribution, maturity adjustments, and asset-class-specific correlations
- **CCAR/DFAST Scenario Simulation** — Fed-defined Baseline, Adverse, and Severely Adverse scenarios with macro-variable paths (GDP, unemployment, rates, spreads, housing, equity markets)
- **Three-Pillar Risk Coverage** — Credit risk (IRB), market risk (VaR/SVaR/ES per FRTB), and operational risk (SMA) capital computation
- **Capital Adequacy Assessment** — CET1, Tier 1, Total Capital, and Leverage Ratio tracking with pass/fail against regulatory minimums
- **Macro-Sensitive PD/LGD Models** — Log-linear stress transmission from macroeconomic shocks to probability of default and loss given default
- **Monte Carlo Portfolio Loss** — Full portfolio credit loss distribution via Vasicek single-factor simulation
- **Custom Scenario Builder** — Generate bespoke stress scenarios with user-defined shock magnitudes and mean-reversion recovery
- **Regulatory Reporting** — Automated Excel and JSON report generation with capital trajectories, loss decomposition, and shortfall analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGULATORY CAPITAL STRESS TESTING PLATFORM               │
│                         Basel III/IV · CCAR/DFAST                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   main.py    │
                              │  (Pipeline   │
                              │ Orchestrator)│
                              └──────┬───────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
  │   DATA LAYER      │  │  SCENARIO ENGINE   │  │   CONFIG LAYER    │
  │                   │  │                   │  │                   │
  │ PortfolioData     │  │ MacroScenario     │  │ YAML Config       │
  │   Generator       │  │ Generation        │  │ Loader            │
  │                   │  │                   │  │                   │
  │ DataValidator     │  │ • Baseline        │  │ Default Config    │
  │                   │  │ • Adverse         │  │ Deep Merge        │
  │ • Loan Portfolio  │  │ • Severely Adv.   │  │                   │
  │ • Market Positions│  │ • Custom          │  │ • Capital Params  │
  │ • Macro History   │  │ • Interpolated    │  │ • Risk Weights    │
  │ • Income Stmt     │  │                   │  │ • Model Params    │
  └────────┬──────────┘  └────────┬──────────┘  └───────────────────┘
           │                      │
           └──────────┬───────────┘
                      │
                      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      RISK MODELS LAYER                          │
  │                                                                 │
  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
  │  │  CREDIT RISK    │ │  MARKET RISK    │ │ OPERATIONAL RISK│  │
  │  │                 │ │                 │ │                 │  │
  │  │ Basel IRB       │ │ Parametric VaR  │ │ SMA (Basel III) │  │
  │  │ ASRF/Vasicek    │ │ Historical VaR  │ │ Business Ind.   │  │
  │  │ PD Stress Model │ │ Monte Carlo VaR │ │ BI Component    │  │
  │  │ LGD Downturn    │ │ Stressed VaR    │ │ Int. Loss Mult. │  │
  │  │ Maturity Adj.   │ │ Expected Short. │ │ Stress Factors  │  │
  │  │ MC Loss Dist.   │ │ IRRBB (EVE)     │ │                 │  │
  │  │ Asset Correl.   │ │ FRTB Capital    │ │                 │  │
  │  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘  │
  │           │                   │                    │           │
  └───────────┼───────────────────┼────────────────────┼───────────┘
              │                   │                    │
              └───────────────────┼────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                   CAPITAL ADEQUACY ENGINE                       │
  │                                                                 │
  │  CapitalAdequacyCalculator                                      │
  │  ├── compute_initial_capital()     → Pre-stress capital ratios  │
  │  ├── run_stress_test()             → 9-quarter projection       │
  │  │   ├── Quarter-by-quarter loop:                               │
  │  │   │   ├── Stress PD (macro sensitivity)                      │
  │  │   │   ├── Stress LGD (collateral shocks)                     │
  │  │   │   ├── Compute credit losses                              │
  │  │   │   ├── Stressed credit RWA (IRB)                          │
  │  │   │   ├── Market risk under stress                           │
  │  │   │   ├── Operational risk stress                            │
  │  │   │   ├── PPNR estimation                                    │
  │  │   │   └── Update capital position                            │
  │  │   └── Min ratio across all quarters                          │
  │  ├── run_all_scenarios()           → Multi-scenario execution   │
  │  ├── generate_summary_report()     → Cross-scenario comparison  │
  │  └── compute_capital_shortfall()   → Regulatory gap analysis    │
  │                                                                 │
  │  Capital Ratios Computed:                                       │
  │  • CET1 Ratio    = CET1 Capital / Total RWA     (min: 4.5%)    │
  │  • Tier 1 Ratio  = Tier 1 Capital / Total RWA   (min: 6.0%)    │
  │  • Total Capital  = Total Capital / Total RWA    (min: 8.0%)    │
  │  • Leverage Ratio = Tier 1 Capital / Exposure    (min: 3.0%)    │
  └────────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     REPORTING ENGINE                            │
  │                                                                 │
  │  ReportGenerator                                                │
  │  ├── generate_capital_trajectory()  → Q-by-Q ratio paths        │
  │  ├── generate_loss_decomposition()  → Risk-type breakdown       │
  │  ├── generate_pass_fail_summary()   → Regulatory assessment     │
  │  ├── export_to_json()               → Machine-readable output   │
  │  ├── export_to_excel()              → Multi-sheet workbook      │
  │  └── print_summary()               → Console dashboard         │
  │                                                                 │
  │  Output Artifacts:                                              │
  │  ├── stress_test_results.json                                   │
  │  └── stress_test_report.xlsx                                    │
  │      ├── Summary                                                │
  │      ├── Pass_Fail                                              │
  │      ├── Loss_Decomposition                                     │
  │      ├── Baseline (trajectory)                                  │
  │      ├── Adverse (trajectory)                                   │
  │      └── Severely Adverse (trajectory)                          │
  └─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
  Synthetic Data       Macro Scenarios        Config
  (or Real Data)       (Fed-defined)         (YAML)
       │                    │                   │
       ▼                    ▼                   ▼
  ┌─────────┐        ┌──────────┐        ┌─────────┐
  │Loan Port│        │ Baseline │        │ Basel   │
  │Market   │        │ Adverse  │        │ Params  │
  │Positions│        │ Sev. Adv │        │ Ratios  │
  └────┬────┘        └────┬─────┘        └────┬────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                    ┌─────▼─────┐
                    │ Validation│
                    └─────┬─────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────▼───┐  ┌────▼───┐  ┌───▼────┐
         │Credit  │  │Market  │  │  Op    │
         │Risk    │  │Risk    │  │ Risk   │
         │Model   │  │Model   │  │ Model  │
         └────┬───┘  └────┬───┘  └───┬────┘
              │           │          │
              └───────────┼──────────┘
                          │
                   ┌──────▼──────┐
                   │  Capital    │
                   │  Adequacy   │
                   │  Calculator │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  Reporting  │
                   │  Engine     │
                   └──────┬──────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────▼───┐  ┌────▼───┐  ┌───▼────┐
         │ Excel  │  │  JSON  │  │Console │
         │ Report │  │ Export │  │Summary │
         └────────┘  └────────┘  └────────┘
```

---

## Project Structure

```
Regulatory-Capital-Stress-Testing/
│
├── main.py                          # Pipeline orchestrator & CLI entry point
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── LICENSE                          # MIT License
├── .gitignore
│
├── configs/
│   └── stress_test_config.yaml      # Basel III/IV parameters & scenario defs
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                        # Data generation & validation
│   │   ├── __init__.py
│   │   ├── data_generator.py        # Synthetic portfolio generator
│   │   └── data_validator.py        # Data quality & compliance checks
│   │
│   ├── models/                      # Risk models (Pillar 1)
│   │   ├── __init__.py
│   │   ├── credit_risk.py           # Basel IRB, Vasicek, PD/LGD stress
│   │   ├── market_risk.py           # VaR, SVaR, ES, IRRBB, FRTB
│   │   └── operational_risk.py      # SMA, Business Indicator, ILM
│   │
│   ├── scenarios/                   # Macroeconomic scenario engine
│   │   ├── __init__.py
│   │   └── scenario_engine.py       # CCAR/DFAST scenario generation
│   │
│   ├── capital/                     # Capital adequacy computation
│   │   ├── __init__.py
│   │   └── capital_calculator.py    # CET1/T1/TC ratios, stress projection
│   │
│   ├── reporting/                   # Report generation
│   │   ├── __init__.py
│   │   └── report_generator.py      # Excel, JSON, console reports
│   │
│   └── utils/                       # Configuration & helpers
│       ├── __init__.py
│       ├── config_loader.py         # YAML config parser
│       └── helpers.py               # Formatting & validation utilities
│
├── tests/
│   └── test_stress_testing.py       # 53 comprehensive tests
│
├── data/
│   ├── raw/                         # Raw input data (if using real data)
│   ├── processed/                   # Preprocessed datasets
│   └── outputs/                     # Generated reports
│
└── notebooks/                       # Jupyter notebooks for exploration
```

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/JayDS22/Regulatory-Capital-Stress-Testing.git
cd Regulatory-Capital-Stress-Testing

# Install dependencies
pip install -r requirements.txt

# Run the stress test pipeline
python main.py
```

### Expected Output

```
======================================================================
  REGULATORY CAPITAL STRESS TESTING PLATFORM
  Basel III/IV | CCAR/DFAST Framework
======================================================================

[1/6] Loading configuration...
[2/6] Generating synthetic portfolio (n_loans=5000, n_positions=500)...
    Portfolio EAD: $51.2B | RWA: $39.5B
    Market Positions: 500 | Notional: $9.1B
[3/6] Validating data integrity...
    Loan Portfolio: VALID
    Market Positions: VALID
[4/6] Generating macroeconomic stress scenarios...
    Baseline: GDP [+2.4% → +2.3%], Unemp [4.2% → 3.9%]
    Adverse:  GDP [-1.0% → +1.8%], Unemp [4.7% → 6.2%]
    Severely Adverse: GDP [-3.0% → +1.5%], Unemp [5.2% → 8.0%]
[5/6] Running stress tests across all scenarios...
    Baseline:         CET1=188.19% | ✓ PASS
    Adverse:          CET1=157.37% | ✓ PASS
    Severely Adverse: CET1=127.45% | ✓ PASS
[6/6] Generating reports...

Pipeline completed in 0.29s
```

---

## Usage

### Command Line

```bash
# Default run (5000 loans, 500 positions)
python main.py

# Custom portfolio size
python main.py --n-loans 10000 --n-positions 1000

# Specify config and output directory
python main.py --config configs/stress_test_config.yaml --output reports/

# Quiet mode
python main.py --quiet
```

### Python API

```python
from src.data import PortfolioDataGenerator
from src.scenarios import ScenarioEngine
from src.capital import CapitalAdequacyCalculator
from src.reporting import ReportGenerator

# 1. Generate portfolio
gen = PortfolioDataGenerator(seed=42)
loans = gen.generate_loan_portfolio(5000)
positions = gen.generate_market_positions(500)

# 2. Define bank financials
financials = {
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

# 3. Generate scenarios
engine = ScenarioEngine(seed=42)
scenarios = engine.generate_all_scenarios()

# 4. Run stress tests
calculator = CapitalAdequacyCalculator()
results = calculator.run_all_scenarios(loans, positions, financials, scenarios)

# 5. Generate reports
reporter = ReportGenerator("outputs/")
reporter.export_to_excel(results, calculator)
reporter.print_summary(results, calculator)
```

---

## Modules Deep Dive

### Credit Risk Model (`src/models/credit_risk.py`)

Implements the Basel III IRB approach:

- **Asset Correlation**: `R = 0.12 × w + 0.24 × (1-w)` where `w = (1-e^(-50·PD))/(1-e^(-50))`
- **Conditional PD**: Vasicek formula `N[(1-R)^(-0.5) × G(PD) + (R/(1-R))^(0.5) × G(0.999)]`
- **Maturity Adjustment**: `MA = (1 + (M-2.5)×b) / (1 - 1.5×b)` where `b = (0.11852 - 0.05478·ln(PD))²`
- **Capital Requirement**: `K = LGD × (Conditional_PD - PD) × MA`
- **PD Stress**: Log-linear macro sensitivity `ln(PD_stressed) = ln(PD) + β_gdp·ΔGDP + β_unemp·ΔUnemp + β_spread·ΔSpread`

### Market Risk Model (`src/models/market_risk.py`)

Implements FRTB-aligned market risk:

- **Parametric VaR**: Variance-covariance method with optional correlation matrix
- **Historical VaR**: Full repricing on historical returns
- **Monte Carlo VaR**: 10,000-path simulation
- **Stressed VaR**: Elevated volatility assumptions (2x multiplier)
- **Expected Shortfall**: Tail-average beyond VaR threshold
- **IRRBB**: Duration-based EVE impact under parallel rate shocks

### Scenario Engine (`src/scenarios/scenario_engine.py`)

CCAR/DFAST scenario generation with 9 macro variables across 9 quarters:

| Variable | Baseline | Adverse | Severely Adverse |
|----------|----------|---------|------------------|
| GDP Growth | +2.3% | -2.5% peak | -6.5% peak |
| Unemployment | 4.2% | 7.4% peak | 10.7% peak |
| Fed Funds | 5.25% | 2.75% trough | 0.25% trough |
| Credit Spread | 1.5% | 3.5% peak | 7.0% peak |
| S&P 500 | +2.5%/qtr | -10% peak | -20% peak |
| Housing | +3%/yr | -8% peak | -20% peak |

---

## Configuration

The platform is configured via `configs/stress_test_config.yaml`:

```yaml
capital:
  minimum_ratios:
    cet1: 0.045              # CET1 minimum
    tier1: 0.06              # Tier 1 minimum
    total_capital: 0.08      # Total Capital minimum
    leverage_ratio: 0.03     # Leverage Ratio minimum

  buffers:
    capital_conservation: 0.025
    countercyclical_max: 0.025
    gsib_surcharge: 0.01

  risk_weights:              # Basel standardized approach
    corporate_aaa_aa: 0.20
    corporate_bbb: 1.00
    retail: 0.75
    residential_mortgage: 0.35
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
python -m pytest tests/test_stress_testing.py::TestCreditRiskModel -v
```

**Test Coverage**: 53 tests across all modules (data generation, validation, credit risk, market risk, operational risk, scenarios, capital adequacy, reporting, utilities, and full integration).

---

## Results & Output

### Generated Reports

| File | Description |
|------|-------------|
| `stress_test_report.xlsx` | Multi-sheet Excel workbook with summary, pass/fail, loss decomposition, and per-scenario trajectories |
| `stress_test_results.json` | Machine-readable JSON with full scenario results |

### Capital Ratio Assessment

| Ratio | Minimum | Baseline | Adverse | Sev. Adverse |
|-------|---------|----------|---------|--------------|
| CET1 | 4.5% | ✅ PASS | ✅ PASS | ✅ PASS |
| Tier 1 | 6.0% | ✅ PASS | ✅ PASS | ✅ PASS |
| Total Capital | 8.0% | ✅ PASS | ✅ PASS | ✅ PASS |
| Leverage | 3.0% | ✅ PASS | ✅ PASS | ✅ PASS |

---

## Regulatory Framework Reference

This platform aligns with:

- **Basel III** (BCBS, 2010-2017): Capital adequacy, leverage ratio, liquidity requirements
- **Basel IV** (BCBS, 2017-2023): Revised standardized approaches, output floor
- **CCAR** (Federal Reserve): Annual stress testing for large BHCs (>$100B assets)
- **DFAST** (Dodd-Frank Act §165): Supervisory and company-run stress tests
- **FRTB** (BCBS, 2019): Fundamental Review of the Trading Book
- **SMA** (BCBS, 2017): Standardized Measurement Approach for operational risk

---

## Technologies

| Category | Stack |
|----------|-------|
| **Language** | Python 3.9+ |
| **Statistical Computing** | NumPy, SciPy, Statsmodels |
| **Data Processing** | Pandas |
| **Risk Modeling** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Configuration** | PyYAML |
| **Reporting** | OpenPyXL, Jinja2 |
| **API** | FastAPI, Uvicorn |
| **Testing** | Pytest |
| **Validation** | Pydantic |

---

## Author

**Jay Guwalani**
- GitHub: [@JayDS22](https://github.com/JayDS22)
- LinkedIn: [Jay Guwalani](https://www.linkedin.com/in/jay-guwalani-66763b191/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
