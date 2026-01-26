import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import model
from sim_runner import run_scenario
from auth import require_password
require_password()


st.set_page_config(page_title="Scenario analysis", layout="wide")
st.title("Scenario analysis")

defaults = model.build_defaults()
SPECIES_DISPLAY = {"muntjac": "Muntjac", "roe": "Roe", "fallow": "Fallow"}



def fig_series_with_band(
    year, s: dict, title: str, y_label: str, target: float | None = None, y0: bool = True
) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(year, s["mean"])
    ax.fill_between(year, s["lo"], s["hi"], alpha=0.2)
    if target is not None:
        ax.axhline(float(target), linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y0:
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    return fig


def fig_class_split_over_time(year, sp_stats: dict, title: str, show_band: bool):
    fig, ax = plt.subplots()
    for lab in model.STATE_LABELS:
        s = sp_stats[lab]
        ax.plot(year, s["mean"], label=lab)
        if show_band:
            ax.fill_between(year, s["lo"], s["hi"], alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of species total")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_stacked_abundance(year, series_by_label: dict, title: str, y_label: str, show_band: bool = False):
    """
    Stacked column chart from mean series. Optional: overlay 10-90 band on the TOTAL only.
    series_by_label: label -> {"mean","lo","hi"} arrays
    """
    fig, ax = plt.subplots()

    bottoms = np.zeros_like(year, dtype=float)
    total_lo = np.zeros_like(year, dtype=float)
    total_hi = np.zeros_like(year, dtype=float)

    for lab in model.STATE_LABELS:
        s = series_by_label[lab]
        vals = np.asarray(s["mean"], dtype=float)
        ax.bar(year, vals, bottom=bottoms, label=lab)
        bottoms = bottoms + vals
        total_lo = total_lo + np.asarray(s["lo"], dtype=float)
        total_hi = total_hi + np.asarray(s["hi"], dtype=float)

    if show_band:
        ax.fill_between(year, total_lo, total_hi, alpha=0.15)

    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_stacked_species_totals(year, totals_stats: dict, title: str, y_label: str):
    fig, ax = plt.subplots()
    bottoms = np.zeros_like(year, dtype=float)
    for sp in model.SPECIES:
        s = totals_stats[f"{sp}_total"]
        vals = np.asarray(s["mean"], dtype=float)
        ax.bar(year, vals, bottom=bottoms, label=SPECIES_DISPLAY.get(sp, sp))
        bottoms = bottoms + vals
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

# -------------------------
# Model definition
# -------------------------
st.sidebar.subheader("Model definition")
years = st.sidebar.slider("Years", min_value=3, max_value=50, value=10, step=1)

use_uttlesford = st.sidebar.checkbox(
    "Use Uttlesford abundance estimates",
    value=True,
    help="Preset: (Initial Muntjac, Initial Roe, Initial Fallow) = (2403, 1540, 2157)",
)
utt = defaults["uttlesford_totals"]

if use_uttlesford:
    n0_muntjac = float(utt["muntjac"])
    n0_roe = float(utt["roe"])
    n0_fallow = float(utt["fallow"])
else:
    st.sidebar.markdown("**Initial abundances**")
    n0_muntjac = st.sidebar.number_input("Initial Muntjac", min_value=0.0, value=float(utt["muntjac"]), step=10.0)
    n0_roe = st.sidebar.number_input("Initial Roe", min_value=0.0, value=float(utt["roe"]), step=10.0)
    n0_fallow = st.sidebar.number_input("Initial Fallow", min_value=0.0, value=float(utt["fallow"]), step=10.0)

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
weight_name = st.sidebar.selectbox("Cull policy (all species)", policy_names, index=default_policy_index, help="Determines which age/sex classes are prioritised in the cull")

cull_intensity = st.sidebar.slider(
    "Cull intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help=(
        "The fraction of the surplus (current abundance - target abundance) that is targeted for removal each year, subject to caps and budget."
    ),
)

default_targets = defaults["default_targets_total"]
st.sidebar.markdown("**Post-cull abundance targets**")
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

rho = st.sidebar.slider(
    "Max annual cull fraction per class",
    0.0, 1.0, 1.0, 0.01,
    help=(
        "Limit on the proportion of any single age/sex class that can be culled in one year. "
    ),
)


# -------------------------
# Modelling stability
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Modelling stability")

bio_mode = st.sidebar.selectbox("Biology mode", ["fixed", "ensemble"], index=1, help="Use fixed for a quick run with the default biological parameters. Use ensemble to create uncertainty bands based on multiple sets of parameter values")
n_draws = 1
if bio_mode == "ensemble":
    n_draws = st.sidebar.slider("Ensemble size", 1, 50, 40, 1)
else:
    st.sidebar.caption("Fixed biology (single run)")

st.sidebar.subheader("Density dependence")

enable_density = st.sidebar.checkbox(
    "Enable density dependence",
    value=True,
    help="If unticked, all density-dependent effects on survival and reproduction are removed."
)

disable_density = not enable_density

# -------------------------
# Scoring reference scales
# -------------------------

st.sidebar.subheader("Scoring reference scales")

use_fixed_refs = st.sidebar.checkbox(
    "Use fixed scoring scales",
    value=True,
    help=(
        "Keep this ON when comparing scenarios across different policies, caps, or budgets. "
        "Turn it OFF only if you have changed caps or the budget substantially and want the "
        "score scaled relative to this scenario’s own constraints. "
        "Scores will then NOT be comparable across runs."
    ),
)

fixed_refs = None
if use_fixed_refs:
    fixed_refs = model.score_reference_scales(
        years=int(years),
        annual_budget_total=float(defaults["ANNUAL_BUDGET_TOTAL"])
        if defaults.get("ANNUAL_BUDGET_TOTAL") is not None
        else None,
        annual_cull_limits=dict(defaults["ANNUAL_CULL_LIMITS"]),
        cost_params=defaults["COST_PARAMS"],
        steady_ref=float(defaults.get("score_refs", {}).get("steady_ref", 1.0)),
    )
else:
    st.sidebar.caption(
        "OFF: Scores are scaled to this scenario’s constraints "
        "(use only for within-scenario sensitivity analysis)."
    )


st.sidebar.markdown("---")
run_single_btn = st.sidebar.button("Run single scenario", type="primary")


# -------------------------
# Run + persist results
# -------------------------
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_single_btn:
    init_tuple = (init_totals["muntjac"], init_totals["roe"], init_totals["fallow"])
    targ_tuple = (targets_total["muntjac"], targets_total["roe"], targets_total["fallow"])
    caps_tuple = (annual_cull_limits["muntjac"], annual_cull_limits["roe"], annual_cull_limits["fallow"])

    with st.spinner("Running model..."):
        (
            df_metrics,
            totals_stats,
            cull_stats,
            cost_stats,
            _cull_by_class_stats,  
            class_split_stats,
            class_abundance_stats,
            is_ensemble,
            _df_cull_last_draw,
        ) = run_scenario(
            years=int(years),
            bio_mode=str(bio_mode),
            disable_density_dependence=bool(disable_density),
            n_draws=int(n_draws),
            rho=float(rho),
            cull_intensity=float(cull_intensity),
            weight_name=str(weight_name),
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
            init_totals_tuple=init_tuple,
            targets_tuple=targ_tuple,
            caps_tuple=caps_tuple,
            fixed_score_refs=fixed_refs,
        )

    st.session_state["last_run"] = dict(
        inputs=dict(
            years=int(years),
            init_totals=init_totals,
            use_uttlesford=bool(use_uttlesford),
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
            weight_name=str(weight_name),
            cull_intensity=float(cull_intensity),
            targets_total=targets_total,
            annual_cull_limits=annual_cull_limits,
            rho=float(rho),
            bio_mode=str(bio_mode),
            n_draws=int(n_draws),
            use_fixed_score_refs=bool(use_fixed_refs),
            fixed_score_refs=(dict(fixed_refs) if fixed_refs is not None else None),
        ),
        df_metrics=df_metrics,
        totals_stats=totals_stats,
        cull_stats=cull_stats,
        cost_stats=cost_stats,
        class_split_stats=class_split_stats,
        class_abundance_stats=class_abundance_stats,
        is_ensemble=bool(is_ensemble),
    )

# -------------------------
# Main rendering (from cached last_run)
# -------------------------
last = st.session_state["last_run"]
if last is None:
    st.info("Set parameters in the sidebar and click **Run single scenario**.")
else:
    df_metrics = last["df_metrics"]
    totals_stats = last["totals_stats"]
    cull_stats = last["cull_stats"]
    cost_stats = last["cost_stats"]
    class_split_stats = last["class_split_stats"]
    class_abundance_stats = last["class_abundance_stats"]
    is_ensemble = last["is_ensemble"]
    inputs = last["inputs"]

    left, right = st.columns([1, 2], gap="large")

    with left:
        if is_ensemble:
            st.subheader("Summary across ensemble draws")
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
                "Ensemble plots show the **mean** across ensemble draws. The shaded band is the **10th–90th percentile** "
                "across draws at each year (computed pointwise over the ensemble)."
            )

        year0 = totals_stats["year"]
        st.pyplot(
            fig_series_with_band(
                year0,
                totals_stats["all_total"],
                "All species total abundance (mean + 10–90% band)",
                "All-species total",
                y0=True,
            )
        )

        st.markdown("### Species totals")
        for sp in model.SPECIES:
            col = f"{sp}_total"
            st.pyplot(
                fig_series_with_band(
                    year0,
                    totals_stats[col],
                    f"{SPECIES_DISPLAY[sp]} total (mean + 10–90% band; dashed = target)",
                    "Abundance",
                    target=float(inputs["targets_total"][sp]),
                    y0=True,
                )
            )

        st.markdown("### Cull & cost")
        year1 = cull_stats["year"]
        st.pyplot(fig_series_with_band(year1, cull_stats, "Total realised cull per year (mean + 10–90% band)", "Cull", y0=True))
        st.pyplot(fig_series_with_band(year1, cost_stats, "Total cost per year (mean + 10–90% band)", "Cost (£)", y0=True))

        st.markdown("### Class split over time")
        st.caption("For the selected species, each line is the proportion of the population in that age/sex bin (sums to ~1 each year).")

        sp_pick = st.selectbox(
            "Species (class split)",
            model.SPECIES,
            format_func=lambda x: SPECIES_DISPLAY.get(x, x),
            key="class_split_species",
        )

        show_band = bool(is_ensemble) and st.checkbox("Show 10–90% band (class split)", value=False, key="class_split_band")

        st.pyplot(
            fig_class_split_over_time(
                class_split_stats["year"],
                class_split_stats[sp_pick],
                f"{SPECIES_DISPLAY.get(sp_pick, sp_pick)}: class split over time",
                bool(show_band),
            )
        )

        st.markdown("### Abundance composition (stacked columns)")

        st.pyplot(
            fig_stacked_species_totals(
                year0,
                totals_stats,
                "All-species total abundance split by species (stacked mean)",
                "Abundance",
            )
        )

        st.caption("Stacked bars below show **absolute class abundances** for the selected species.")

        sp_pick2 = st.selectbox(
            "Species (stacked class abundance)",
            model.SPECIES,
            format_func=lambda x: SPECIES_DISPLAY.get(x, x),
            key="stacked_species",
        )

        show_band2 = bool(is_ensemble) and st.checkbox("Show 10–90% band on total only (stacked)", value=False, key="stacked_band")

        st.pyplot(
            fig_stacked_abundance(
                class_abundance_stats["year"],
                class_abundance_stats[sp_pick2],
                f"{SPECIES_DISPLAY.get(sp_pick2, sp_pick2)}: class abundance (stacked mean)",
                "Abundance",
                show_band=bool(show_band2),
            )
        )
