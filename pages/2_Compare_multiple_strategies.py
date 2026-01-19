import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import model
from sim_runner import run_scenario
from auth import require_password

from viz import (
    fig_series_with_band,
    fig_class_split_over_time,
    fig_stacked_abundance,
    fig_stacked_species_totals,
)

#require_password()

st.set_page_config(page_title="Compare multiple management strategies", layout="wide")
st.title("Compare multiple management strategies")

defaults = model.build_defaults()
SPECIES_DISPLAY = {"muntjac": "Muntjac", "roe": "Roe", "fallow": "Fallow"}

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

# Persist last run so selectboxes don't wipe outputs
if "cmp_last_run" not in st.session_state:
    st.session_state["cmp_last_run"] = None


# -----------------------------
# Overlay plot helpers (kept as-is)
# -----------------------------
def fig_overlay_with_band(year, series_list, title: str, y_label: str) -> plt.Figure:
    fig, ax = plt.subplots()
    for label, s in series_list:
        ax.plot(year, s["mean"], label=label)
        ax.fill_between(year, s["lo"], s["hi"], alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_overlay(year, series_list, title: str, y_label: str) -> plt.Figure:
    fig, ax = plt.subplots()
    for label, s in series_list:
        ax.plot(year, s["mean"], label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# Plan table helpers
# -----------------------------
def _plan_table_for_species(plan: dict, sp: str, show_band: bool) -> pd.DataFrame:
    """
    Human-readable plan:
      rows: Year
      cols: STATE_LABELS
      cell: mean OR "mean (lo–hi)" depending on show_band
    """
    years = plan["year"]
    out = {"Year": years}
    for lab in model.STATE_LABELS:
        m = np.asarray(plan[sp][lab]["mean"])
        if show_band:
            lo = np.asarray(plan[sp][lab]["lo"])
            hi = np.asarray(plan[sp][lab]["hi"])
            out[lab] = [
                f"{int(np.rint(m[i]))} ({int(np.rint(lo[i]))}–{int(np.rint(hi[i]))})"
                for i in range(len(years))
            ]
        else:
            out[lab] = [f"{int(np.rint(m[i]))}" for i in range(len(years))]
    return pd.DataFrame(out)


def _plan_csv_for_species(plan: dict, sp: str, which: str) -> pd.DataFrame:
    """
    Numeric plan for download.
    which in {"mean","lo","hi"}.
    Returns integer columns.
    """
    years = plan["year"]
    out = {"year": years}
    for lab in model.STATE_LABELS:
        arr = np.asarray(plan[sp][lab][which])
        out[lab] = np.rint(arr).astype(int)
    return pd.DataFrame(out)


# -----------------------------
# Add strategy UI
# -----------------------------
st.subheader("Define and add a management strategy")

colA, colB = st.columns(2)

with colA:
    budget_enabled = st.checkbox("Enable budget cap", value=True, key="cmp_budget_enabled")
    annual_budget_total = defaults["ANNUAL_BUDGET_TOTAL"] if budget_enabled else None
    if budget_enabled:
        annual_budget_total = st.number_input(
            "Annual budget cap (£)",
            min_value=0.0,
            value=40_000.0,
            step=1000.0,
            key="cmp_budget",
        )

    policy_names = list(defaults["weight_scenarios"].keys())
    default_policy = "Female priority, no juveniles"
    default_policy_index = policy_names.index(default_policy) if default_policy in policy_names else 0
    weight_name = st.selectbox("Cull policy (all species)", policy_names, index=default_policy_index, key="cmp_policy", help="Determines which age/sex classes are prioritised in the cull")

    cull_intensity = st.slider(
        "Cull intensity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="cmp_intensity",
        help=(
            "The fraction of the surplus (current abundance - target abundance) that is targeted for removal each year, subject to caps and budget."
        ),
    )

    rho = st.slider("Max annual cull fraction per class", 0.0, 1.0, 0.50, 0.01, key="cmp_rho", help="Limit on the proportion of any single age/sex class that can be culled in one year.")

with colB:
    default_targets = defaults["default_targets_total"]
    st.markdown("**Post-cull abundance targets**")
    target_m = st.number_input(
        "Target Muntjac", min_value=0.0, value=float(default_targets["muntjac"]), step=10.0, key="cmp_t_m"
    )
    target_r = st.number_input("Target Roe", min_value=0.0, value=float(default_targets["roe"]), step=10.0, key="cmp_t_r")
    target_f = st.number_input(
        "Target Fallow", min_value=0.0, value=float(default_targets["fallow"]), step=10.0, key="cmp_t_f"
    )

    default_caps = defaults["ANNUAL_CULL_LIMITS"]
    st.markdown("**Annual cull caps per species**")
    cap_m = st.number_input(
        "Cull cap Muntjac", min_value=0.0, value=float(default_caps["muntjac"]), step=10.0, key="cmp_c_m"
    )
    cap_r = st.number_input("Cull cap Roe", min_value=0.0, value=float(default_caps["roe"]), step=10.0, key="cmp_c_r")
    cap_f = st.number_input(
        "Cull cap Fallow", min_value=0.0, value=float(default_caps["fallow"]), step=10.0, key="cmp_c_f"
    )

st.markdown("**Strategy label**")
budget_tag = f"£{annual_budget_total:,.0f}" if annual_budget_total is not None else "no budget cap"
label_default = f"{weight_name} | intensity={cull_intensity:.2f} | budget={budget_tag}"
label = st.text_input("Label", value=label_default, key="cmp_label")

if st.button("Add strategy", type="primary"):
    st.session_state["scenarios"].append(
        dict(
            label=label.strip() if label.strip() else f"Strategy {len(st.session_state['scenarios']) + 1}",
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
            weight_name=str(weight_name),
            cull_intensity=float(cull_intensity),
            targets_tuple=(float(target_m), float(target_r), float(target_f)),
            caps_tuple=(float(cap_m), float(cap_r), float(cap_f)),
            rho=float(rho),
        )
    )
    st.success("Strategy added.")

st.markdown("---")


# -----------------------------
# Shared settings
# -----------------------------
st.subheader("Define the context (settings held constant across strategies)")

col1, col2, col3 = st.columns(3)
utt = defaults["uttlesford_totals"]

with col1:
    years = st.slider("Years", 3, 50, 10, 1, key="cmp_years")

with col2:
    use_uttlesford = st.checkbox("Use Uttlesford abundance estimates", value=True, key="cmp_use_utt")
    if use_uttlesford:
        init_tuple = (float(utt["muntjac"]), float(utt["roe"]), float(utt["fallow"]))
        st.caption(
            f"Uttlesford: Initial Muntjac={init_tuple[0]:.0f}, Initial Roe={init_tuple[1]:.0f}, Initial Fallow={init_tuple[2]:.0f}"
        )
    else:
        n0_m = st.number_input("Initial Muntjac", min_value=0.0, value=float(utt["muntjac"]), step=10.0, key="cmp_n0_m")
        n0_r = st.number_input("Initial Roe", min_value=0.0, value=float(utt["roe"]), step=10.0, key="cmp_n0_r")
        n0_f = st.number_input("Initial Fallow", min_value=0.0, value=float(utt["fallow"]), step=10.0, key="cmp_n0_f")
        init_tuple = (float(n0_m), float(n0_r), float(n0_f))

with col3:
    bio_mode = st.selectbox("Biology mode", ["fixed", "ensemble"], index=1, key="cmp_bio_mode", help="Use fixed for a quick run with the default biological parameters. Use ensemble to create uncertainty bands based on multiple sets of parameter values")
    n_draws = 1
    if bio_mode == "ensemble":
        n_draws = st.slider("Ensemble draws", 1, 50, 40, 1, key="cmp_draws")
    else:
        st.caption("Fixed biology (single run)")


# -----------------------------
# Fixed refs for fair comparison
# -----------------------------
st.markdown("---")
st.caption(
    "For fair strategy comparison, all strategies are scored using the same reference scales (derived from the defaults), "
    "even if their caps/budgets differ."
)

fixed_refs = model.score_reference_scales(
    years=int(years),
    annual_budget_total=float(defaults["ANNUAL_BUDGET_TOTAL"]),
    annual_cull_limits=dict(defaults["ANNUAL_CULL_LIMITS"]),
    cost_params=defaults["COST_PARAMS"],
    steady_ref=float(defaults.get("score_refs", {}).get("steady_ref", 1.0)),
)

st.markdown("---")


# -----------------------------
# Strategy list
# -----------------------------
st.subheader("Strategy list")

if not st.session_state["scenarios"]:
    st.info("Add at least one strategy above.")
else:
    df_list = pd.DataFrame(st.session_state["scenarios"])
    st.dataframe(
        df_list[["label", "weight_name", "cull_intensity", "annual_budget_total", "targets_tuple", "caps_tuple", "rho"]],
        use_container_width=True,
        hide_index=True,
    )

    colx, coly, colz = st.columns(3)
    with colx:
        if st.button("Clear all strategies"):
            st.session_state["scenarios"] = []
            st.session_state["cmp_last_run"] = None
            st.rerun()

    with coly:
        delete_idx = st.number_input(
            "Delete strategy index (0-based)",
            min_value=0,
            max_value=max(0, len(st.session_state["scenarios"]) - 1),
            value=0,
            step=1,
        )
        if st.button("Delete selected"):
            if st.session_state["scenarios"]:
                st.session_state["scenarios"].pop(int(delete_idx))
                st.session_state["cmp_last_run"] = None
                st.rerun()

    with colz:
        run_all = st.button("Run all strategies", type="primary")

    if run_all:
        results = []
        overlays_all = []
        overlays_cull = []
        overlays_cost = []

        # Per-strategy plan and extra stats so we can plot the winner like the policy page
        scenario_plans: dict[str, dict] = {}
        scenario_extras: dict[str, dict] = {}

        with st.spinner("Running scenarios..."):
            for sc in st.session_state["scenarios"]:
                (
                    df_metrics,
                    totals_stats,
                    cull_stats,
                    cost_stats,
                    cull_by_class_stats,
                    class_split_stats,
                    class_abundance_stats,
                    is_ens,
                    _df_cull_last_draw,
                ) = run_scenario(
                    years=int(years),
                    bio_mode=str(bio_mode),
                    n_draws=int(n_draws),
                    rho=float(sc["rho"]),
                    cull_intensity=float(sc["cull_intensity"]),
                    weight_name=str(sc["weight_name"]),
                    annual_budget_total=(float(sc["annual_budget_total"]) if sc["annual_budget_total"] is not None else None),
                    init_totals_tuple=init_tuple,
                    targets_tuple=tuple(sc["targets_tuple"]),
                    caps_tuple=tuple(sc["caps_tuple"]),
                    fixed_score_refs=fixed_refs,
                )

                summary = df_metrics.mean(numeric_only=True).to_dict()
                summary.update(
                    dict(
                        label=sc["label"],
                        weight_name=sc["weight_name"],
                        cull_intensity=sc["cull_intensity"],
                        rho=sc["rho"],
                        annual_budget_total=sc["annual_budget_total"],
                        target_muntjac=sc["targets_tuple"][0],
                        target_roe=sc["targets_tuple"][1],
                        target_fallow=sc["targets_tuple"][2],
                        cap_muntjac=sc["caps_tuple"][0],
                        cap_roe=sc["caps_tuple"][1],
                        cap_fallow=sc["caps_tuple"][2],
                    )
                )
                results.append(summary)

                overlays_all.append((sc["label"], totals_stats["all_total"]))
                overlays_cull.append((sc["label"], cull_stats))
                overlays_cost.append((sc["label"], cost_stats))

                lbl = str(sc["label"])
                scenario_plans[lbl] = cull_by_class_stats
                scenario_extras[lbl] = dict(
                    totals_stats=totals_stats,
                    cull_stats=cull_stats,
                    cost_stats=cost_stats,
                    class_split_stats=class_split_stats,
                    class_abundance_stats=class_abundance_stats,
                    is_ensemble=bool(is_ens),
                )

        df_cmp = pd.DataFrame(results)

        # Cache everything needed for rendering (prevents reset on widget changes)
        st.session_state["cmp_last_run"] = dict(
            df_cmp=df_cmp,
            overlays_all=overlays_all,
            overlays_cull=overlays_cull,
            overlays_cost=overlays_cost,
            scenario_plans=scenario_plans,
            scenario_extras=scenario_extras,
            years=int(years),
            bio_mode=str(bio_mode),
            n_draws=int(n_draws),
            init_tuple=tuple(init_tuple),
        )

        st.success("Done.")


# -----------------------------
# Render cached results (prevents reset on widget change)
# -----------------------------
if st.session_state["cmp_last_run"] is not None:
    cache = st.session_state["cmp_last_run"]

    df_cmp = cache["df_cmp"]
    overlays_all = cache["overlays_all"]
    overlays_cull = cache["overlays_cull"]
    overlays_cost = cache["overlays_cost"]
    scenario_plans = cache["scenario_plans"]
    scenario_extras = cache["scenario_extras"]

    years_r = int(cache.get("years", years))
    bio_mode_r = str(cache.get("bio_mode", bio_mode))
    n_draws_r = int(cache.get("n_draws", n_draws))
    init_tuple_r = tuple(cache.get("init_tuple", init_tuple))

    # -----------------------------
    # Comparison table
    # -----------------------------
    st.subheader("Comparison table (averaged across draws)")

    cols = [
        "label",
        "score",
        "total_cull",
        "total_cost",
        "t_stable",
        "steady_dev_late",
        "weight_name",
        "cull_intensity",
        "rho",
        "annual_budget_total",
        "target_muntjac",
        "target_roe",
        "target_fallow",
        "cap_muntjac",
        "cap_roe",
        "cap_fallow",
    ]
    cols = [c for c in cols if c in df_cmp.columns]
    st.dataframe(df_cmp[cols].sort_values("score"), use_container_width=True, hide_index=True)

    st.download_button(
        "Download comparison CSV",
        data=df_cmp.to_csv(index=False).encode("utf-8"),
        file_name="scenario_comparison.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Comparison plots
    # -----------------------------
    st.markdown("---")
    st.subheader("Comparison plots")

    if bio_mode_r == "ensemble":
        st.caption(
            "Each line is the strategy mean across draws. The shaded band is the 10th–90th percentile across draws (pointwise by year)."
        )
        y_all = scenario_extras[list(scenario_extras.keys())[0]]["totals_stats"]["year"]
        y_flow = overlays_cull[0][1]["year"]
        st.pyplot(fig_overlay_with_band(y_all, overlays_all, "All-species total abundance", "All-species total"))
        st.pyplot(fig_overlay_with_band(y_flow, overlays_cull, "Total realised cull per year", "Cull"))
        st.pyplot(fig_overlay_with_band(y_flow, overlays_cost, "Total cost per year", "Cost (£)"))
    else:
        st.caption("Fixed biology: each line is deterministic for its strategy.")
        y_all = scenario_extras[list(scenario_extras.keys())[0]]["totals_stats"]["year"]
        y_flow = overlays_cull[0][1]["year"]
        st.pyplot(fig_overlay(y_all, overlays_all, "All-species total abundance", "All-species total"))
        st.pyplot(fig_overlay(y_flow, overlays_cull, "Total realised cull per year", "Cull"))
        st.pyplot(fig_overlay(y_flow, overlays_cost, "Total cost per year", "Cost (£)"))

    # -----------------------------
    # Recommended management plan (best strategy)
    # -----------------------------
    st.markdown("---")
    st.subheader("Recommended management plan (best strategy)")

    if len(df_cmp) == 0:
        st.info("No results to recommend from.")
    else:
        best_row = df_cmp.sort_values("score").iloc[0]
        best_label = str(best_row["label"])
        plan = scenario_plans.get(best_label)
        extra = scenario_extras.get(best_label)

        init_m, init_r, init_f = init_tuple_r
        draws_text = f"{int(n_draws_r)} draw(s)" if str(bio_mode_r) == "ensemble" else "single run"

        st.markdown(
            f"""
**Shared settings**
- Horizon: **{int(years_r)} years**
- Biology mode: **{str(bio_mode_r)}** ({draws_text})
- Initial totals: Muntjac **{init_m:.0f}**, Roe **{init_r:.0f}**, Fallow **{init_f:.0f}**
"""
        )

        st.markdown(
            f"""
**Best strategy controls**
- Label: **{best_label}**
- Cull policy: **{best_row.get('weight_name', '')}**
- Cull intensity: **{float(best_row.get('cull_intensity', np.nan)):.3f}**
- Max annual cull fraction per class: **{float(best_row.get('rho', np.nan)):.2f}**
- Annual budget cap: **{best_row.get('annual_budget_total', None)}**
- Targets: Muntjac **{best_row.get('target_muntjac', np.nan):.0f}**, Roe **{best_row.get('target_roe', np.nan):.0f}**, Fallow **{best_row.get('target_fallow', np.nan):.0f}**
- Annual cull caps: Muntjac **{best_row.get('cap_muntjac', np.nan):.0f}**, Roe **{best_row.get('cap_roe', np.nan):.0f}**, Fallow **{best_row.get('cap_fallow', np.nan):.0f}**
"""
        )

        if plan is None:
            st.warning("Could not find the per-class cull plan for the best strategy (unexpected).")
        else:
            sp_pick = st.selectbox(
                "Species for plan table",
                model.SPECIES,
                format_func=lambda s: SPECIES_DISPLAY.get(s, s),
                key="best_plan_species",
            )

            with st.expander("Year-by-year cull plan by class (table)", expanded=False):
                is_ens = (bio_mode_r == "ensemble")
                if is_ens:
                    st.caption(
                        "Realised cull schedule by year and class, reported as the mean across biology draws with a 10–90% uncertainty band."
                    )
                else:
                    st.caption(
                        "Realised cull schedule by year and class, reported as a single deterministic run (no uncertainty band)."
                    )

                st.dataframe(_plan_table_for_species(plan, sp_pick, show_band=is_ens), use_container_width=True, hide_index=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        "Download mean plan CSV",
                        data=_plan_csv_for_species(plan, sp_pick, "mean").to_csv(index=False).encode("utf-8"),
                        file_name=f"scenario_plan_{sp_pick}_mean.csv",
                        mime="text/csv",
                    )

                if bio_mode_r == "ensemble":
                    with c2:
                        st.download_button(
                            "Download P10 plan CSV",
                            data=_plan_csv_for_species(plan, sp_pick, "lo").to_csv(index=False).encode("utf-8"),
                            file_name=f"scenario_plan_{sp_pick}_p10.csv",
                            mime="text/csv",
                        )
                    with c3:
                        st.download_button(
                            "Download P90 plan CSV",
                            data=_plan_csv_for_species(plan, sp_pick, "hi").to_csv(index=False).encode("utf-8"),
                            file_name=f"scenario_plan_{sp_pick}_p90.csv",
                            mime="text/csv",
                        )

        if extra is None:
            st.warning("Best strategy plots unavailable (missing cached stats).")
        else:
            st.markdown("---")
            st.subheader("Best strategy — plots")

            year_tot = extra["totals_stats"]["year"]
            year_flow = extra["cull_stats"]["year"]

            st.pyplot(
                fig_series_with_band(
                    year_tot,
                    extra["totals_stats"]["all_total"],
                    "All-species total abundance (mean + 10–90% band)",
                    "Abundance",
                    target=None,
                    y0=True,
                )
            )
            st.pyplot(
                fig_series_with_band(
                    year_flow,
                    extra["cull_stats"],
                    "Total realised cull per year (mean + 10–90% band)",
                    "Cull",
                    target=None,
                    y0=True,
                )
            )
            st.pyplot(
                fig_series_with_band(
                    year_flow,
                    extra["cost_stats"],
                    "Total cost per year (mean + 10–90% band)",
                    "Cost (£)",
                    target=None,
                    y0=True,
                )
            )

            st.pyplot(
                fig_stacked_species_totals(
                    year_tot,
                    extra["totals_stats"],
                    "All-species total abundance split by species (stacked mean)",
                    "Abundance",
                    species_display=SPECIES_DISPLAY,
                )
            )

            st.markdown("### Species detail")
            sp_detail = st.selectbox(
                "Species (detail plots)",
                model.SPECIES,
                format_func=lambda x: SPECIES_DISPLAY.get(x, x),
                key="cmp_species_detail_best",
            )

            targets_total = {
                "muntjac": float(best_row.get("target_muntjac", np.nan)),
                "roe": float(best_row.get("target_roe", np.nan)),
                "fallow": float(best_row.get("target_fallow", np.nan)),
            }

            st.pyplot(
                fig_series_with_band(
                    year_tot,
                    extra["totals_stats"][f"{sp_detail}_total"],
                    f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)} total (mean + 10–90% band; dashed = target)",
                    "Abundance",
                    target=targets_total.get(sp_detail, None),
                    y0=True,
                )
            )

            show_band_detail = bool(extra.get("is_ensemble", False)) and st.checkbox(
                "Show 10–90% band on class split/abundance",
                value=False,
                key="cmp_detail_show_band_best",
            )

            st.pyplot(
                fig_class_split_over_time(
                    extra["class_split_stats"]["year"],
                    extra["class_split_stats"][sp_detail],
                    f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)}: class split over time",
                    bool(show_band_detail),
                )
            )

            st.pyplot(
                fig_stacked_abundance(
                    extra["class_abundance_stats"]["year"],
                    extra["class_abundance_stats"][sp_detail],
                    f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)}: class abundance (stacked mean)",
                    "Abundance",
                    show_band=bool(show_band_detail),
                )
            )

    # -----------------------------
    # Score decomposition (collapsible, at bottom)
    # -----------------------------
    st.markdown("---")
    with st.expander("Score decomposition (advanced)", expanded=False):
        need = ["score", "contrib_cost", "contrib_steady", "contrib_time", "contrib_cull"]
        if all(c in df_cmp.columns for c in need):
            st.caption("Because reference scales are fixed across strategies, these contribution percentages are directly comparable.")
            d = df_cmp[["label"] + need + ["n_cost", "n_steady", "n_time", "n_cull"]].copy()
            denom = d["score"].replace(0.0, np.nan)
            for c in ["contrib_cost", "contrib_steady", "contrib_time", "contrib_cull"]:
                d[c + "_%"] = 100.0 * d[c] / denom
            st.dataframe(d.sort_values("score"), use_container_width=True, hide_index=True)
        else:
            st.info("Score decomposition columns not available in results.")
