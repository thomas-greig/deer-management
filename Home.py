import streamlit as st
from auth import require_password

require_password()

st.set_page_config(
    page_title="Your App Title",
    layout="wide"
)


st.title("Instructions")

st.markdown(
    """
Use this tool to explore deer management policies under constraints (targets, caps, budgets) and compare outcomes using a consistent scoring framework.

---

## 1) Scenario analysis (single scenario)
Use this page to **run one scenario** and inspect trajectories and uncertainty.

**Set model context**
- **Years** (simulation horizon)
- **Initial abundances** (Uttlesford preset or manual)
- **Biology mode**: *fixed* (single run) or *ensemble* (uncertainty bands)

**Set management controls**
- **Cull policy** (age–sex allocation weights applied to all species)
- **Cull intensity** (fraction of annual surplus removed, before constraints)
- **Targets** (post-cull species totals)
- **Caps**: per-species absolute annual cull caps + per-class fraction cap **ρ**
- **Budget cap** (optional): total annual £ cap across species

**What you get**
- Species totals and (optionally) bands over time
- Annual realised cull and annual cost
- Age–sex structure plots (composition / abundance)
- **Scenario score + component breakdown**
  - Note: score is computed from normalised components (cost, time to stabilise, late deviation, total cull)

**Scoring reference scales**
- You can choose **fixed reference scales** (comparable across different runs) or **scenario-derived scales** (only comparable within the same constraint context).

---

## 2) Scenario comparison (many scenarios, same context)
Use this page to **compare multiple management choices** under the same shared assumptions.

**Add scenarios (vary these per scenario)**
- Policy, intensity, **ρ**
- Targets, caps, budget cap
- Custom scenario label

**Shared settings (held constant across scenarios)**
- Years, initial abundances, biology mode, ensemble draws

**What you get**
- Overlaid plots for totals / cull / cost
- Scenario table with raw metrics + normalised components + total score
- Downloadable outputs (including per-class cull plan where available)

**Important**
- This page uses **fixed scoring reference scales** for fair comparison across scenarios.

---

## 3) Policy recommendation (optimise controls)
Use this page to **search for a good policy** given fixed assumptions and constraints.

**Fixed settings (held constant during optimisation)**
- Years, initial abundances
- Targets, biology mode (+ ensemble draws if used)
- Budget cap on/off (and level, if enabled)
- Cost curves and demographic schedules are fixed

**Decision variables (optimised within bounds)**
- Cull policy (optional to optimise; otherwise fixed)
- Cull intensity (optional to optimise; otherwise fixed)
- Species cull caps (min/max bounds per species)
- Per-class cap **ρ** is fixed (not optimised)

**Methods**
- **Random search** (recommended default): scales better for multi-dimensional search
- **Grid search**: can become huge quickly; the UI warns/blocks oversized grids

**What you get**
- Recommended controls + ranked candidate table
- Plots for top candidates
- Diagnostics on constraint binding (budget/caps frequently binding)

---

### Practical tips
- Use **fixed biology** when debugging or iterating quickly; switch to **ensemble** when you need uncertainty bands.
"""
)
