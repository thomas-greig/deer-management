import json
import streamlit as st
import pandas as pd
import numpy as np

import model
from auth import require_password
require_password()



# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Model description", layout="wide")
st.title("Model description")

# -------------------------
# Model description (Markdown + KaTeX-friendly math)
# -------------------------
st.markdown(
    r"""
This page describes the population model used throughout the tool.

## Simple explanation of model

This section provides a simplified explanation of how population dynamics and management decisions are modelled.  
For a more detailed and explicit explanation, see the **Detailed mathematical formulation of the model** section below.

The model consists of **30 codependent classes of deer**:  
**3 species × 2 sexes × 5 age groups**  
(<1, 1–2, 2–5, 5–8, 8+ years).

The abundance (number of deer) in each class is affected by natural processes and management decisions, as described below.

---

### Biological processes

**Survival**  
All classes have a chance of dying naturally each year. Mortality is higher at the start and end of the normal lifespan. Deer that survive age forward by exactly one year and transition into the next age class where applicable.

**Reproduction**  
Female classes may produce offspring each year. Adults have a higher probability of reproducing than yearlings, while juveniles do not reproduce. Offspring are born with an equal male–female ratio and all enter the population in the `<1` year age class.

**Density pressure**  
Competition within and between species is modelled such that higher total deer abundance reduces survival and fertility. Density pressure is species-specific and recalculated annually.

**Net migration**  
Yearling and adult deer are assumed to migrate from higher-density areas to lower-density areas. Neighbouring districts are assumed to maintain a constant density equal to the modelled district’s initial density. As a result, when local abundance falls below its initial level, deer are added at a rate proportional to this difference.

---

### Management decisions

The number of deer culled each year is calculated as the **surplus multiplied by the cull intensity**, before constraints are applied.

**The surplus**  
The difference between the abundance of each species at the start of a year and its target abundance.

**The cull intensity**  
The fraction of the surplus that is targeted for removal each year, subject to constraints.

**Practical and ethical constraints**
- **Annual budget cap**  
  Cull cost increases nonlinearly with the number of deer removed, reflecting increasing logistical difficulty at higher harvest intensities.
- **Absolute annual species cap**  
  A maximum number of individuals of each species that can be culled per year.
- **Relative annual age–sex cap**  
  A maximum fraction of each age–sex class that can be removed per year.

---

### Model calibration

The model is calibrated so that, in the absence of culling, total abundance remains approximately constant while the population slowly ages. This provides a fair baseline for comparing management strategies and reflects the fact that real-world abundance estimates already incorporate some level of adult culling.

---

### Model parameter averaging

Key numerical parameters controlling biological processes are estimated from the literature but are uncertain. To avoid results depending on any single parameter set, outcomes are averaged across an ensemble of simulations with varied parameters representing plausible ecological scenarios.

---

### Strategy evaluation

Each strategy is evaluated using a **scalar score**, where lower scores indicate more desirable outcomes.  
The score incorporates the following components, listed from highest to lowest importance:

1. Total cost (£)
2. Time taken for the population to stabilise
3. Deviation between final abundance and target abundance
4. Total number of deer culled

---

## Detailed mathematical formulation of the model

This section provides a precise mathematical description of the population dynamics and management processes implemented in the simulation model, extending the simplified explanation above.

---

### State variables and indexing

Species are indexed by  
$ s \in \{\text{muntjac}, \text{roe}, \text{fallow}\} $.

For each species, the population state at discrete time $ t \in \mathbb{N} $ is represented by a 10-dimensional non-negative vector  

$$
\mathbf{N}_s(t) \in \mathbb{R}_{\ge 0}^{10},
$$

ordered as  

$$
(M0, F0, M1, F1, M2a, F2a, M2b, F2b, M2c, F2c).
$$

Total abundance for species $ s $ is  

$$
N_s(t) = \mathbf{1}^\top \mathbf{N}_s(t).
$$

---

### Density pressure and interspecific competition

Each species experiences density pressure from both its own population and other species:

$$
P_s(t) = \sum_{s'} \alpha_{s s'} \, N_{s'}(t).
$$

Density dependence is applied using Beverton–Holt–type multipliers:

$$
g(P_s; K_s, \beta) = \frac{1}{1 + \beta \, P_s / K_s},
$$

applied separately to fertility ($\beta_f$), juvenile survival ($\beta_{s0}$), and yearling/adult survival ($\beta_{s,\text{adult}}$).

---

### Reproduction

Let $ f_{s,i} $ denote baseline fertility for female age class $ i $.  
Total births:

$$
B_s(t) = g(P_s(t); K_s, \beta_f) \sum_i f_{s,i} N_{s,i}(t).
$$

Offspring are split evenly by sex:

$$
N_{s,M0}(t+1) = 0.5\, B_s(t), \quad
N_{s,F0}(t+1) = 0.5\, B_s(t).
$$

---

### Survival and ageing

Juvenile survival is density-dependent via $\beta_{s0}$; yearling and adult survival via $\beta_{s,\text{adult}}$.

Example transitions:

$$
N_{s,M1}(t+1) =
g(P_s(t); K_s, \beta_{s0}) \, s_{s,M0} \, N_{s,M0}(t),
$$

$$
N_{s,M2a}(t+1) =
g(P_s(t); K_s, \beta_{s,\text{adult}}) \, s_{s,M1} \, N_{s,M1}(t).
$$

The terminal age class accumulates survivors:

$$
N_{s,M2c}(t+1) =
g(P_s(t); K_s, \beta_{s,\text{adult}})
\bigl(s_{s,M2b} N_{s,M2b}(t) + s_{s,M2c} N_{s,M2c}(t)\bigr).
$$

---

### Net migration

Net migration is modelled as:

$$
\Delta_s(t) = \gamma_s \bigl(N_{s,0} - N_s(t)\bigr).
$$

Positive values add migrants to male and female yearlings; negative values remove individuals proportionally from yearling and adult classes.

---

### Cull policy and constraints

Desired cull:

$$
H_s^{\text{des}}(t)
= \min\!\left\{
c \max(N_s(t) - N_s^\star, 0),\;
H_s^{\max}
\right\}.
$$

Cull is allocated across classes using a fixed weight vector $ \mathbf{w} $, subject to:

$$
H_{s,i}(t) \le \rho \, N_{s,i}(t).
$$

---

### Budget constraint and cost function

Cull cost:

$$
C_s(h_s) = a_s h_s + b_s h_s^{p_s}.
$$

If total cost exceeds the annual budget $ B $, harvests are reduced using a Lagrange-multiplier allocation that prioritises species with larger overshoot relative to target.

---

### Scenario scoring and objective function

Each strategy is scored using four components:

- **Total harvest**
- **Total cost**
- **Time to stabilisation**
- **Late-time deviation from targets**

Each component is normalised using reference scales and combined as:

$$
S =
w_{\text{cull}} \tilde H +
w_{\text{cost}} \tilde C +
w_{\text{time}} \tilde T +
w_{\text{steady}} \tilde D.
$$

Lower scores indicate better performance.

---

### Full update step

For each species and year:
1. Compute density pressure.
2. Apply reproduction, survival, and ageing.
3. Apply net migration.
4. Subtract realised cull (after all caps).

---

### Uncertainty and ensemble averaging

Key biological parameters are uncertain. To account for this, policies are evaluated across an ensemble of perturbed systems:

- Carrying capacities (±20%)
- Density-dependence strengths (±20%)
- Competition coefficients (±15%)
- Migration coupling strengths (±30%)

Each policy is simulated across all ensemble members. Scores are computed per realisation and summarised across the ensemble (mean, spread, worst-case), ensuring robustness to ecological uncertainty.
"""
)

