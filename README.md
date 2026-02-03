# Deer Population Management Simulator

A research-grade simulation tool for evaluating deer culling strategies under biological uncertainty and real-world operational constraints.

Built with **Streamlit**, this application models structured deer populations and provides decision-support through scenario analysis, strategy comparison, and policy optimisation.

---

## What this project is

This repository implements a **structured population dynamics model** covering three deer species — muntjac, roe, and fallow — each divided into sex and age classes.

The simulator integrates:

* Density-dependent survival and reproduction
* Migration effects
* Financial constraints with nonlinear harvest costs
* Species-level and class-level culling limits
* Policy-driven harvest allocation
* Ensemble simulation to reflect ecological uncertainty

Management strategies are evaluated using a scalar objective that balances:

* Total cost
* Time to population stabilisation
* Deviation from target abundance
* Total harvest

The full conceptual and mathematical description of the model is provided in **`4_Model_description.py`**.

---

## Core Capabilities

* **Scenario analysis** — explore the outcomes of a single management strategy
* **Strategy comparison** — evaluate competing policies on a consistent scoring scale
* **Policy optimisation** — search the management space using grid or random search
* **Uncertainty quantification** — run ensembles with perturbed biological parameters
* **Constraint diagnostics** — identify when budgets or caps become binding

The simulation engine runs multiple biological draws and aggregates results into summary statistics and uncertainty bands, enabling more robust interpretation than single-run models.

---

## Repository Structure

```
Home.py                          # Streamlit entry point
model.py                         # Core population and harvest mechanics
sim_runner.py                    # Scenario execution, scoring, and ensemble aggregation

1_Analyse_a_single_strategy.py   # Explore one strategy
2_Compare_multiple_strategies.py # Compare policies
3_Find_the_optimal_strategy.py   # Optimisation workflow

4_Model_description.py           # Conceptual + mathematical model explanation
5_Model_parameters.py            # Live parameter reference
```

---

## Installation

```bash
pip install -r requirements.txt
streamlit run Home.py
```

---

## Disclaimer

This software is provided for research and demonstration purposes only.
No responsibility is accepted for real-world management decisions based on this tool.

