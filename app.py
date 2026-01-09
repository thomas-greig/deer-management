import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import model
from sim_runner import run_scenario

st.set_page_config(page_title="Deer management simulator", layout="wide")
st.title("Optimal Deer Management — interactive simulator")

defaults = model.build_defaults()
SPECIES_DISPLAY = {"muntjac": "Muntjac", "roe": "Roe", "fallow": "Fallow"}


def build_policy_table_10(weight_scenarios: dict[str, model.np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, w in weight_scenarios.items():
        r = {"Policy": name}
        for idx, lab in enumerate(model.STATE_LABELS):
            r[lab] = float(w[idx])
        rows.append(r)
    return pd.DataFrame(rows)


df_policy_10 = build_policy_table_10(defaults["weight_scenarios"])


def fig_series_with_band(year, s: dict, title: str, y_label: str, target: float | None = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(year, s["mean"])
    ax.fill_between(year, s["lo"], s["hi"], alpha=0.2)
    if target is not None:
        ax.axhline(float(target), linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


st.sidebar.header("Controls")

# -------------------------
# Model definition
# -------------------------
st.sidebar.subheader("Model definition")
years = st.sidebar.slider("Years", min_value=3, max_value=50, value=10, step=1)

use_uttlesford = st.sidebar.checkbox(
    "Use Uttlesford abundance estimates",
    value=True,
    help="Preset: (N0,muntjac, N0,roe, N0,fallow) = (2403, 1540, 2157)",
)
utt = defaults["uttlesford_totals"]

if use_uttlesford:
    n0_muntjac = float(utt["muntjac"])
    n0_roe = float(utt["roe"])
    n0_fallow = float(utt["fallow"])
    st.sidebar.caption(f"Uttlesford: Muntjac={n0_muntjac:.0f}, Roe={n0_roe:.0f}, Fallow={n0_fallow:.0f}")
else:
    st.sidebar.markdown("**Initial abundances**")
    n0_muntjac = st.sidebar.number_input("N0 Muntjac", min_value=0.0, value=float(utt["muntjac"]), step=10.0)
    n0_roe = st.sidebar.number_input("N0 Roe", min_value=0.0, value=float(utt["roe"]), step=10.0)
    n0_fallow = st.sidebar.number_input("N0 Fallow", min_value=0.0, value=float(utt["fallow"]), step=10.0)

init_totals = {"muntjac": float(n0_muntjac), "roe": float(n0_roe), "fallow": float(n0_fallow)}

# -------------------------
# Management decisions
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Management decisions")

budget_enabled = st.sidebar.checkbox("Enable budget cap", value=True)

annual_budget_total = defaults["ANNUAL_BUDGET_TOTAL"] if budget_enabled else None
if budget_enabled:
    annual_budget_total = st.sidebar.number_input(
        "Annual budget cap (£)",
        min_value=0.0,
        value=40_000.0,
        step=1000.0,
    )
else:
    st.sidebar.caption("Budget cap disabled")

policy_names = list(defaults["weight_scenarios"].keys())
default_policy = "Female priority, no juveniles"
default_policy_index = policy_names.index(default_policy) if default_policy in policy_names else 0
weight_name = st.sidebar.selectbox("Cull policy (all species)", policy_names, index=default_policy_index)

# Cull intensity (Option A semantics)
cull_intensity = st.sidebar.slider(
    "Cull intensity",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help=(
        "Each year, the model computes the current surplus max(total − target, 0) and attempts to remove cull_intensity × surplus that year (subject to caps and budget)."
    ),
)

default_targets = defaults["default_targets_total"]
st.sidebar.markdown("**Target post-cull abundances**")
target_muntjac = st.sidebar.number_input("Target Muntjac", min_value=0.0, value=float(default_targets["muntjac"]), step=10.0)
target_roe = st.sidebar.number_input("Target Roe", min_value=0.0, value=float(default_targets["roe"]), step=10.0)
target_fallow = st.sidebar.number_input("Target Fallow", min_value=0.0, value=float(default_targets["fallow"]), step=10.0)
targets_total = {"muntjac": float(target_muntjac), "roe": float(target_roe), "fallow": float(target_fallow)}

default_caps = defaults["ANNUAL_CULL_LIMITS"]
st.sidebar.markdown("**Annual cull caps per species**")
cap_muntjac = st.sidebar.number_input("Cull cap Muntjac", min_value=0.0, value=float(default_caps["muntjac"]), step=10.0)
cap_roe = st.sidebar.number_input("Cull cap Roe", min_value=0.0, value=float(default_caps["roe"]), step=10.0)
cap_fallow = st.sidebar.number_input("Cull cap Fallow", min_value=0.0, value=float(default_caps["fallow"]), step=10.0)
annual_cull_limits = {"muntjac": float(cap_muntjac), "roe": float(cap_roe), "fallow": float(cap_fallow)}

rho = st.sidebar.slider("Max cull fraction per class", 0.0, 0.5, 0.30, 0.01)

# -------------------------
# Modelling stability
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Modelling stability")

bio_mode = st.sidebar.selectbox("Biology mode", ["fixed", "ensemble"], index=1)
n_draws = 1
if bio_mode == "ensemble":
    n_draws = st.sidebar.slider("Ensemble draws", 1, 50, 15, 1)
else:
    st.sidebar.caption("Fixed biology (single run)")

st.sidebar.markdown("---")
run_single_btn = st.sidebar.button("Run single scenario", type="primary")


# -------------------------
# Main
# -------------------------
if run_single_btn:
    init_tuple = (init_totals["muntjac"], init_totals["roe"], init_totals["fallow"])
    targ_tuple = (targets_total["muntjac"], targets_total["roe"], targets_total["fallow"])
    caps_tuple = (annual_cull_limits["muntjac"], annual_cull_limits["roe"], annual_cull_limits["fallow"])

    with st.spinner("Running model..."):
        df_metrics, totals_stats, cull_stats, cost_stats, is_ensemble = run_scenario(
            years=int(years),
            bio_mode=str(bio_mode),
            n_draws=int(n_draws),
            rho=float(rho),
            cull_intensity=float(cull_intensity),
            weight_name=str(weight_name),
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
            init_totals_tuple=init_tuple,
            targets_tuple=targ_tuple,
            caps_tuple=caps_tuple,
        )

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Scenario inputs")
        st.write(
            {
                "Model definition": {
                    "years": int(years),
                    "initial_abundances": {"Muntjac": init_totals["muntjac"], "Roe": init_totals["roe"], "Fallow": init_totals["fallow"]},
                    "use_uttlesford_abundance_estimates": bool(use_uttlesford),
                },
                "Management decisions": {
                    "annual_budget_cap": (float(annual_budget_total) if annual_budget_total is not None else None),
                    "cull_policy_all_species": weight_name,
                    "cull_intensity": float(cull_intensity),
                    "target_post_cull_abundances": {"Muntjac": targets_total["muntjac"], "Roe": targets_total["roe"], "Fallow": targets_total["fallow"]},
                    "annual_cull_caps": {"Muntjac": annual_cull_limits["muntjac"], "Roe": annual_cull_limits["roe"], "Fallow": annual_cull_limits["fallow"]},
                    "max_cull_fraction_per_class": float(rho),
                },
                "Modelling stability": {"biology_mode": bio_mode, "ensemble_draws": int(n_draws)},
            }
        )

        st.subheader("Policy weights (10-class)")
        st.dataframe(df_policy_10, use_container_width=True)

        if is_ensemble:
            st.subheader("Summary across draws")
            st.dataframe(df_metrics.describe(), use_container_width=True)

        st.subheader("Draw-by-draw metrics")
        st.dataframe(df_metrics, use_container_width=True)

        st.download_button(
            "Download metrics CSV",
            data=df_metrics.to_csv(index=False).encode("utf-8"),
            file_name="metrics.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Time series")

        if is_ensemble:
            st.caption(
                "Ensemble plots show the **mean** across draws. The shaded band is the **10th–90th percentile** "
                "across draws at each year (computed pointwise over the ensemble)."
            )

        year0 = totals_stats["year"]
        st.pyplot(fig_series_with_band(year0, totals_stats["all_total"], "All species total abundance (mean + 10–90% band)", "All-species total"))

        st.markdown("### Species totals")
        for sp in model.SPECIES:
            col = f"{sp}_total"
            st.pyplot(
                fig_series_with_band(
                    year0,
                    totals_stats[col],
                    f"{SPECIES_DISPLAY[sp]} total (mean + 10–90% band; dashed = target)",
                    "Abundance",
                    target=float(targets_total[sp]),
                )
            )

        st.markdown("### Cull & cost")
        year1 = cull_stats["year"]
        st.pyplot(fig_series_with_band(year1, cull_stats, "Total realised cull per year (mean + 10–90% band)", "Cull"))
        st.pyplot(fig_series_with_band(year1, cost_stats, "Total cost per year (mean + 10–90% band)", "Cost (£)"))

else:
    st.info("Set parameters in the sidebar and click **Run single scenario**.")
