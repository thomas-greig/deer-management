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
st.set_page_config(page_title="Model parameters", layout="wide")
st.title("Model description and parameters")

defaults = model.build_defaults()
SPECIES = list(getattr(model, "SPECIES", ["muntjac", "roe", "fallow"]))


# =========================
# FORMATTING HELPERS 
# =========================
def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _species_dict_to_tuple(x):
    """If x looks like a species-keyed dict, return tuple ordered by SPECIES; else None."""
    if not isinstance(x, dict):
        return None
    norm_map = {_norm_key(k): v for k, v in x.items()}

    out = []
    hits = 0
    for sp in SPECIES:
        spn = _norm_key(sp)
        if spn in norm_map:
            out.append(norm_map[spn])
            hits += 1
        else:
            # allow keys like "roe deer"
            found = None
            for k_norm, v in norm_map.items():
                if spn in k_norm:
                    found = v
                    break
            out.append(found)
            if found is not None:
                hits += 1

    # treat as species dict only if at least 2 matches (avoid false positives)
    return tuple(out) if hits >= 2 else None


def _fmt_num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if abs(xf) >= 1000:
            return f"{xf:,.0f}"
        return f"{xf:.4g}"
    return str(x)


def _fmt_seq(seq) -> str:
    if seq is None:
        return ""
    arr = np.asarray(seq).reshape(-1)
    return "(" + ", ".join(_fmt_num(v) for v in arr) + ")"


def _fmt_matrix(mat) -> str:
    if mat is None:
        return ""
    a = np.asarray(mat, dtype=float)
    rows = []
    for r in a:
        rows.append("(" + ", ".join(_fmt_num(float(v)) for v in r) + ")")
    return "[" + ", ".join(rows) + "]"


def _fmt_value(x) -> str:
    """General formatter used for all Value cells."""
    if x is None:
        return ""
    sp_tuple = _species_dict_to_tuple(x)
    if sp_tuple is not None:
        return _fmt_seq(sp_tuple)
    if isinstance(x, (list, tuple, np.ndarray)):
        return _fmt_seq(x)
    if isinstance(x, (int, float, np.integer, np.floating, bool, str)):
        return _fmt_num(x)
    if isinstance(x, dict):
        try:
            return json.dumps(x, indent=2, default=str)
        except Exception:
            return str(x)
    return str(x)


def _kv_df(rows: list[tuple[str, object]]) -> pd.DataFrame:
    """Build a 2-col Parameter/Value dataframe with values pre-formatted as strings."""
    return pd.DataFrame(
        [{"Parameter": p, "Value": _fmt_value(v)} for (p, v) in rows]
    )


# =========================
# Policy selector (updates live)
# =========================
st.markdown(
    "This page lists the parameters that correspond to the model parameter table in the report, "
    "but values are pulled live from the codebase (`model.build_defaults()`)."
)

weight_scenarios = defaults.get("weight_scenarios", {}) or {}
policy_names = list(weight_scenarios.keys())

if policy_names:
    default_policy = "Female priority, no juveniles"
    if default_policy not in policy_names:
        default_policy = policy_names[0]
    picked_policy = st.selectbox(
        "Cull allocation weights policy (used below)",
        policy_names,
        index=policy_names.index(default_policy),
    )
    picked_weights = np.asarray(weight_scenarios[picked_policy], dtype=float).reshape(-1)
else:
    picked_policy = "(no policies found)"
    picked_weights = np.zeros(10, dtype=float)
    st.warning("No policy weight scenarios found in defaults['weight_scenarios'].")

st.markdown("---")


# =========================
# Pull live values
# =========================

years_default = 10
cull_intensity_default = 0.7
rho_default = 0.30

targets = defaults.get("default_targets_total")
caps = defaults.get("ANNUAL_CULL_LIMITS")
budget = defaults.get("ANNUAL_BUDGET_TOTAL")

base_params = defaults.get("base_params", {}) or {}

def _sp(sp: str) -> dict:
    d = base_params.get(sp, {})
    return d if isinstance(d, dict) else {}

def _sp_survival(sp: str):
    return _sp(sp).get("survival")

def _sp_fertility(sp: str):
    return _sp(sp).get("fertility")

def _sp_param(sp: str, k: str):
    return _sp(sp).get(k)

male_birth_fraction = _sp_param(SPECIES[0], "male_birth_fraction")

K_vals = {sp: _sp_param(sp, "K") for sp in SPECIES}
beta_f_vals = {sp: _sp_param(sp, "beta_f") for sp in SPECIES}
beta_s0_vals = {sp: _sp_param(sp, "beta_s0") for sp in SPECIES}
beta_s_adult_vals = {sp: _sp_param(sp, "beta_s_adult") for sp in SPECIES}

alpha = defaults.get("base_alpha", None)
gamma = defaults.get("base_gamma", None)

uttlesford_totals = defaults.get("uttlesford_totals")
male_frac_default = defaults.get("male_frac_default")
age_fracs_default = defaults.get("age_fracs_default")

cost_params = defaults.get("COST_PARAMS", {}) or {}

def _cost_tuple_or_raw(sp: str):
    d = cost_params.get(sp)
    if isinstance(d, dict) and any(k in d for k in ("a", "b", "p")):
        return (d.get("a"), d.get("b"), d.get("p"))
    return d


# =========================
# Display
# =========================
st.subheader("Management-defined parameters (configurable; defaults shown)")
mgmt_rows = [
    ("Simulation horizon (years)", years_default),
    ("Cull intensity (scales annual desired harvest)", cull_intensity_default),
    ("Maximum annual harvest fraction per age–sex class ρ", rho_default),
    ("Target post-harvest abundance thresholds (muntjac, roe, fallow)", targets),
    (f"Cull allocation weights by age–sex class w ({picked_policy})", picked_weights),
    ("Absolute annual harvest caps by species (muntjac, roe, fallow)", caps),
    ("Total annual management budget cap B (£)", budget),
    ("Nonlinear harvest cost parameters (muntjac) (a,b,p)", _cost_tuple_or_raw("muntjac")),
    ("Nonlinear harvest cost parameters (roe) (a,b,p)", _cost_tuple_or_raw("roe")),
    ("Nonlinear harvest cost parameters (fallow) (a,b,p)", _cost_tuple_or_raw("fallow")),
]
st.dataframe(_kv_df(mgmt_rows), use_container_width=True, hide_index=True)

st.subheader("Demographic parameters (fixed)")
demo_rows = [
    ("Baseline annual survival probabilities (muntjac) (s_M0,...,s_F2c)", _sp_survival("muntjac")),
    ("Baseline annual survival probabilities (roe) (s_M0,...,s_F2c)", _sp_survival("roe")),
    ("Baseline annual survival probabilities (fallow) (s_M0,...,s_F2c)", _sp_survival("fallow")),
    ("Female fertility rates by age class (muntjac) (f_F0,f_F1,f_F2a,f_F2b,f_F2c)", _sp_fertility("muntjac")),
    ("Female fertility rates by age class (roe) (f_F0,f_F1,f_F2a,f_F2b,f_F2c)", _sp_fertility("roe")),
    ("Female fertility rates by age class (fallow) (f_F0,f_F1,f_F2a,f_F2b,f_F2c)", _sp_fertility("fallow")),
    ("Male fraction of offspring at birth (all species)", male_birth_fraction),
]
st.dataframe(_kv_df(demo_rows), use_container_width=True, hide_index=True)

st.subheader("Density dependence and interspecific interactions (fixed)")
dd_rows = [
    ("Carrying capacities (K_muntjac, K_roe, K_fallow)", K_vals),
    ("Density-dependence on fertility β_f (muntjac, roe, fallow)", beta_f_vals),
    ("Density-dependence on juvenile survival β_s0 (muntjac, roe, fallow)", beta_s0_vals),
    ("Density-dependence on adult survival β_{s,adult} (muntjac, roe, fallow)", beta_s_adult_vals),
    ("Competition matrix α (rows/cols: muntjac, roe, fallow)", _fmt_matrix(alpha)),
    ("Net migration coupling strengths γ (muntjac, roe, fallow)", gamma),
]
st.dataframe(_kv_df(dd_rows), use_container_width=True, hide_index=True)

st.subheader("Initial conditions (configurable; defaults shown)")
init_rows = [
    ("Initial species abundances at t=0 (muntjac, roe, fallow)", uttlesford_totals),
    ("Initial male fraction within each age class", male_frac_default),
    ("Initial age-class proportions (<1, 1–2, 2–5, 5–8, 8+)", age_fracs_default),
]
st.dataframe(_kv_df(init_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Ensemble draw ranges (fixed)")
st.markdown(
    """
These multiplicative perturbation ranges are applied around the fixed base parameters in ensemble mode:

- `K_range = (0.8, 1.2)`
- `beta_range = (0.8, 1.2)`
- `alpha_range = (0.85, 1.15)`
- `gamma_range = (0.7, 1.3)`
"""
)
