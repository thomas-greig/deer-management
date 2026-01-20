import streamlit as st
from auth import require_password
#require_password()


st.set_page_config(page_title="Home", layout="wide")
st.title("Deer cull modelling tool")

st.markdown(
    r"""
## About this tool
This tool allows you to explore how different deer culling strategies affect species abundance under financial and feasability constraints (targets, caps, budgets). The tool was built to model Uttlesford, a district of lowland habitat in East Anglia characterised by
intensive arable agriculture, fragmented woodland cover, high road density, and a substantial human population. It models 3 species of deer (Muntjac, Roe and Fallow) split into 2 sex and 5 age categories each. Management strategies can be compared using a consistent scoring framework based on the financial cost of the cull, the number of years it takes for the population to stabalise at the target abundance levels, the difference between the the final abundance and the target abundance, and the total number of deer culled.

A simple explanation of how the model works, along with a more more in-depth mathematical explanation can be found in the Model Description page. The parameter values used by the model are also visible in the Model Parameters page.

There are three ways you can explore the results of different culling strategies:
1. Analyse a single culling strategy
2. Compare multiple strategies
3. Find the optimal strategy

You will find a page for each in the menu on the left hand side.

---

## Instructions

1. **Set model context**
    1. **Years** (i.e. the simulation horizon)
    2. **Initial abundances** (use Uttlesford presets or set manually)

2. **Set management controls**
    1. **Cull policy** (this controls which age and sex classes are prioritised in the cull, the weights are applied to all species equally)
    2.**Cull intensity** (this controls the fraction of annual surplus, the difference between current abundance and target abundance, that is removed, before caps are applied)
    3. **Targets** (this is the post-cull species abundance you are aiming for)
    4. **Caps** (you can set absolute value cull caps for each species and/or a fractional cap per age and sex class. These reflect ethical or feasibility constraints) 
    5. **Budget cap** (you can set a total annual £ cap across all species)

3. **Manage uncertainty and model stability**
    1. **Biology mode** (set this to **fixed** if you want quick results; switch to **ensemble** when you need uncertainty bands)
    2. **Ensemble size** (this controls how many model runs are performed using slightly different sets of biological input parameters)

4. **Analyse results**
    1. Species abundance over time
    2. Annual realised cull and annual cost over time
    3. Age–sex and species abundance and composition plots over time
    4. Strategy score and rank [Comparison and optimisation pages only]
    5. Comparison plots for total abundance / cull / cost [Comparison page only]
    6. Recommended management plan [Comparison and optimisation pages only]
    7. Scenario table with raw metrics + normalised components + total score [Comparison and optimisation pages only]
    8. Top strategy plots for total abundance / cull / cost [Comparison and optimisation pages only]
    9. Diagnostics on constraint binding (budget/caps frequently binding) [Optimisation page only]
"""
)
