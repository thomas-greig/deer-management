import streamlit as st
from auth import require_password
require_password()


st.set_page_config(page_title="Home", layout="wide")
st.title("Deer cull modedlling tool")

st.markdown(
    r"""
## About this tool
This tool allows you to explore how different deer culling strategies affect species abundance under financial and feasability constraints (targets, caps, budgets). The tool was built to model Uttlesford, a district of lowland habitat in East Anglia characterised by by
intensive arable agriculture, fragmented woodland cover, high road density, and a substantial human population. It models 3 species of deer (Muntjac, Roe and Fallow) split into 2 sex and 5 age categories each. Management strategies can be compared using a consistent scoring framework based on the financial cost of the cull, the number of years it takes for the population to stabalise at the target abundance levels, the difference between the the final abundance and the target abundance, and the total number of deer culled.

A simple explanation of how the model works, along with a more more in-depth mathematical explanation can be found in the Model Description page. The parameter values used by the model are also visible in the Model Parameters page.

There are three ways you can explore the results of different culling strategies:
- Analyse a single strategy
- Compare multiple strategies
- Find the optimal strategy

You will find a page for each in the menu on the left hand side.

---

## Instructions

**Set model context**
- **Years** (simulation horizon)
- **Initial abundances** (Uttlesford preset or manual)

**Set management controls**
- **Cull policy** (this controls which age and sex classes are prioritised in the cull, the weights are applied to all species equally)
- **Cull intensity** (this controls the fraction of annual surplus, the difference between current abundance and target abundance, that is removed, before constraints)
- **Targets** (this is the post-cull species abundance you are aiming for)
- **Caps** (you can set absolute value cull caps for each species and/or a fractional cap per age and sex class. These reflect ethical or feasability constraints) 
- **Budget cap** (you can set a total annual £ cap across all species)

**Manage uncertainty and model stability**
- **Biology mode** (set this to **fixed** if you want quick results; switch to **ensemble** when you need uncertainty bands.

**Analyse results**
- Species abundance over time
- Annual realised cull and annual cost over time
- Age–sex and species abundance and composition plots over time
- Strategy score and rank [Comparison and optimisation pages only]
- Comparison plots for total abundance / cull / cost [Comparison page only]
- Recommended management plan [Comparison and optimisation pages only]
- Scenario table with raw metrics + normalised components + total score [Comparison and optimisation pages only]
- Top strategy plots for total abundance / cull / cost [Comparison and optimisation pages only]
- Diagnostics on constraint binding (budget/caps frequently binding) [Optimisation page only]
"""
)
