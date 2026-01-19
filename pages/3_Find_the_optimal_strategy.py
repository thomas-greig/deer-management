import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

st.set_page_config(page_title="Find the optimal management strategy", layout="wide")
st.title("Find the optimal management strategy")

defaults = model.build_defaults()
SPECIES_DISPLAY = {"muntjac": "Muntjac", "roe": "Roe", "fallow": "Fallow"}

# -----------------------------
# Persist last optimisation run across reruns (so selectboxes don't wipe outputs)
# -----------------------------
if "polrec_results_sorted" not in st.session_state:
    st.session_state["polrec_results_sorted"] = None
if "polrec_df" not in st.session_state:
    st.session_state["polrec_df"] = None
if "polrec_meta" not in st.session_state:
    st.session_state["polrec_meta"] = None


# -----------------------------
# Helpers
# -----------------------------
def _plot_abundance(year, totals_stats_by_candidate):
    fig, ax = plt.subplots()
    for cand_label, totals_stats in totals_stats_by_candidate:
        ax.plot(year, totals_stats["all_total"]["mean"], label=cand_label)
    ax.set_title("All-species total abundance")
    ax.set_xlabel("Year")
    ax.set_ylabel("Abundance")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_series(year, series_by_candidate, title, ylabel):
    fig, ax = plt.subplots()
    for cand_label, s in series_by_candidate:
        ax.plot(year, s["mean"], label=cand_label)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig


def _binding_flags(df_cull_last_draw: pd.DataFrame | None, caps: dict[str, float], annual_budget_total: float | None) -> dict:
    if df_cull_last_draw is None or df_cull_last_draw.empty:
        return {"budget_binding": False, "cap_binding_muntjac": False, "cap_binding_roe": False, "cap_binding_fallow": False}

    out = {}
    # Budget binding is already provided by the model
    if "budget_binding" in df_cull_last_draw.columns:
        out["budget_binding"] = bool(df_cull_last_draw["budget_binding"].fillna(False).any())
    else:
        if annual_budget_total is None or annual_budget_total <= 0:
            out["budget_binding"] = False
        else:
            out["budget_binding"] = bool((df_cull_last_draw["cost_total_all"] >= 0.999 * annual_budget_total).any())

    # Cap binding (approx): realised close to cap in any year
    for sp in model.SPECIES:
        col = f"{sp}_realised"
        cap = float(caps.get(sp, 0.0))
        if cap <= 0 or col not in df_cull_last_draw.columns:
            out[f"cap_binding_{sp}"] = False
        else:
            out[f"cap_binding_{sp}"] = bool((df_cull_last_draw[col] >= 0.999 * cap).any())

    return out


def _plan_table_for_species(plan: dict, sp: str, show_band: bool) -> pd.DataFrame:
    """
    Returns a human-readable table:
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
    which in {"mean","lo","hi"}; returns numeric table for download.
    Returns integer columns.
    """
    years = plan["year"]
    out = {"year": years}
    for lab in model.STATE_LABELS:
        arr = np.asarray(plan[sp][lab][which])
        out[lab] = np.rint(arr).astype(int)
    return pd.DataFrame(out)




# -----------------------------
# Frozen controls
# -----------------------------
# -----------------------------
# Fixed context + fixed strategy
# -----------------------------
st.subheader("Fixed management strategy controls")
st.caption(
    "These settings are held fixed during optimisation."
)

# --- Fixed management strategy controls ---
mcol1, mcol2 = st.columns([1, 1])

with mcol1:
    rho = st.slider(
        "Max annual cull fraction per class",
        0.0, 1.0, 0.50, 0.01,
        help="Limit on the proportion of any single age/sex class that can be culled in one year.",
    )

with mcol2:
    budget_enabled = st.checkbox("Enable annual budget cap", value=True)
    annual_budget_total = None
    if budget_enabled:
        annual_budget_total = st.number_input(
            "Annual budget cap (£)",
            min_value=0.0,
            value=float(defaults["ANNUAL_BUDGET_TOTAL"]),
            step=1000.0,
        )

st.markdown("**Post-cull abundance targets**")
tcol1, tcol2, tcol3 = st.columns(3)
dt = defaults["default_targets_total"]
with tcol1:
    target_m = st.number_input("Target Muntjac", min_value=0.0, value=float(dt["muntjac"]), step=10.0)
with tcol2:
    target_r = st.number_input("Target Roe", min_value=0.0, value=float(dt["roe"]), step=10.0)
with tcol3:
    target_f = st.number_input("Target Fallow", min_value=0.0, value=float(dt["fallow"]), step=10.0)

targets_tuple = (float(target_m), float(target_r), float(target_f))


# -----------------------------
# Fixed context controls
# -----------------------------
st.subheader("Fixed context controls")
st.caption(
    "These settings are held fixed during optimisation.")

utt = defaults["uttlesford_totals"]

ccol1, ccol2, ccol3 = st.columns([1, 1.2, 1])

with ccol1:
    years = st.slider("Years", 3, 50, 10, 1)

with ccol2:
    use_utt = st.checkbox("Use Uttlesford abundance estimates", value=True)
    if use_utt:
        init_tuple = (float(utt["muntjac"]), float(utt["roe"]), float(utt["fallow"]))
        st.caption(
            f"Initial: Muntjac={init_tuple[0]:.0f}, Roe={init_tuple[1]:.0f}, Fallow={init_tuple[2]:.0f}"
        )
    else:
        n0_m = st.number_input("Initial Muntjac", min_value=0.0, value=float(utt["muntjac"]), step=10.0)
        n0_r = st.number_input("Initial Roe", min_value=0.0, value=float(utt["roe"]), step=10.0)
        n0_f = st.number_input("Initial Fallow", min_value=0.0, value=float(utt["fallow"]), step=10.0)
        init_tuple = (float(n0_m), float(n0_r), float(n0_f))

with ccol3:
    bio_mode = st.selectbox("Biology mode", ["fixed", "ensemble"], index=0)
    n_draws = 1
    if bio_mode == "ensemble":
        n_draws = st.slider("Ensemble draws", 1, 50, 15, 1)


# -----------------------------
# Score weighting (as before, but compact)
# -----------------------------
sk = defaults["score_kwargs"]

mode = st.radio(
    "Score weighting",
    ["Use default priorities", "Adjust priorities (relative to defaults)"],
    index=0,
    help="Adjust the relative contributions of the 4 elements of the evaluation score",
)

priority_levels = [
    "Very low priority",
    "Low priority",
    "Medium priority",
    "High priority",
    "Very high priority",
]
priority_mult = {
    "Very low priority": 0.70,
    "Low priority": 0.85,
    "Medium priority": 1.00,
    "High priority": 1.15,
    "Very high priority": 1.30,
}

if mode == "Use default priorities":
    w_cull = float(sk["w_cull"])
    w_time = float(sk["w_time"])
    w_steady = float(sk["w_steady"])
    w_cost = float(sk["w_cost"])
else:
    c1, c2 = st.columns(2)
    with c1:
        p_cull = st.selectbox("Cull", priority_levels, index=2)
        p_time = st.selectbox("Time to stabilise", priority_levels, index=2)
    with c2:
        p_steady = st.selectbox("Steady-state deviation", priority_levels, index=2)
        p_cost = st.selectbox("Cost", priority_levels, index=2)

    w_cull = float(sk["w_cull"]) * priority_mult[p_cull]
    w_time = float(sk["w_time"]) * priority_mult[p_time]
    w_steady = float(sk["w_steady"]) * priority_mult[p_steady]
    w_cost = float(sk["w_cost"]) * priority_mult[p_cost]

# Stability settings are intentionally frozen to defaults for comparability
stable_tol_rel = float(sk["stable_tol_rel"])
stable_window = int(sk["stable_window"])
late_window = int(sk["late_window"])

score_kwargs_override = dict(
    w_cull=float(w_cull),
    w_time=float(w_time),
    w_steady=float(w_steady),
    w_cost=float(w_cost),
    stable_tol_rel=float(stable_tol_rel),
    stable_window=int(stable_window),
    late_window=int(late_window),
)

st.markdown("---")


# -----------------------------
# Optimised controls + search space
# -----------------------------
st.subheader("Select which management controls to optimise")

policies = list(defaults["weight_scenarios"].keys())
default_policy = "Female priority, no juveniles"
default_policy_index = policies.index(default_policy) if default_policy in policies else 0

col1, col2 = st.columns(2)
with col1:
    optimise_policy = st.checkbox("Optimise cull policy", value=True, help="Determine which age/sex classes should be prioritised in the cull")
    if not optimise_policy:
        fixed_policy = st.selectbox("Cull policy (fixed)", policies, index=default_policy_index)

with col2:
    optimise_intensity = st.checkbox("Optimise cull intensity", value=True, help="Determine the fraction of the surplus (current abundance - target abundance) that is targeted for removal each year, subject to caps and budget.")
    if not optimise_intensity:
        fixed_intensity = st.slider("Cull intensity (fixed)", 0.0, 1.0, 1.0, 0.05)

st.markdown("**Annual cull caps per species optimation bounds**")
dcaps = defaults["ANNUAL_CULL_LIMITS"]
b1, b2, b3 = st.columns(3)
with b1:
    cap_m_min = st.number_input("Muntjac cap min", min_value=0.0, value=0.0, step=10.0)
    cap_m_max = st.number_input("Muntjac cap max", min_value=0.0, value=float(dcaps["muntjac"]), step=10.0)
with b2:
    cap_r_min = st.number_input("Roe cap min", min_value=0.0, value=0.0, step=10.0)
    cap_r_max = st.number_input("Roe cap max", min_value=0.0, value=float(dcaps["roe"]), step=10.0)
with b3:
    cap_f_min = st.number_input("Fallow cap min", min_value=0.0, value=0.0, step=10.0)
    cap_f_max = st.number_input("Fallow cap max", min_value=0.0, value=float(dcaps["fallow"]), step=10.0)

st.markdown("---")

st.subheader("Choose optimisation method")
method = st.selectbox("Method", ["Grid search", "Random search"], index=1, help="Chooses how the optimisation algorithm explores candidate management strategies. Grid search is systematic but time-consuming; random search is faster and usually good enough")

MAX_GRID_CANDIDATES = 15_000

if method == "Grid search":
    st.caption("Grid search evaluates a structured set of candidates. Large grids can be slow.")
    g1, g2 = st.columns(2)
    with g1:
        n_intensity = st.slider("Cull intensity grid points", 2, 15, 6, 1, help="Only used if cull intensity is optimised.")
    with g2:
        n_caps = st.slider("Cap grid points per species", 2, 15, 6, 1)

    n_pol = len(policies) if optimise_policy else 1
    n_int = int(n_intensity) if optimise_intensity else 1
    n_cap = int(n_caps)
    est_candidates = int(n_pol * n_int * (n_cap**3))

    st.caption(f"Estimated candidates to evaluate: **{est_candidates:,}**")

    grid_too_big = est_candidates > MAX_GRID_CANDIDATES
    if grid_too_big:
        st.warning(
            f"That grid is likely to be slow/unresponsive ({est_candidates:,} candidates). "
            f"Reduce grid points or switch to Random search. "
            f"(Safety limit: {MAX_GRID_CANDIDATES:,}.)"
        )
else:
    st.caption("Random search is recommended as grid search can take a very long time.")
    max_evals = st.slider("Number of candidates", 20, 2000, 300, 10)
    est_candidates = int(max_evals)
    grid_too_big = False

show_top_n = st.slider("Show top N candidates", 5, 100, 10, 1)
run_btn = st.button("Run optimisation", type="primary")


# -----------------------------
# Fixed references for THIS optimisation run
# -----------------------------
frozen_caps_for_ref = {"muntjac": float(dcaps["muntjac"]), "roe": float(dcaps["roe"]), "fallow": float(dcaps["fallow"])}
ref_budget_for_ref = float(defaults["ANNUAL_BUDGET_TOTAL"]) if defaults.get("ANNUAL_BUDGET_TOTAL") is not None else None

fixed_refs_for_opt = model.score_reference_scales(
    years=int(years),
    annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else ref_budget_for_ref),
    annual_cull_limits=frozen_caps_for_ref,
    cost_params=defaults["COST_PARAMS"],
    steady_ref=float(defaults.get("score_refs", {}).get("steady_ref", 1.0)),
)


# -----------------------------
# Run optimisation
# -----------------------------
if run_btn:
    t0 = time.time()

    candidates = []

    policy_choices = policies if optimise_policy else [str(fixed_policy)]
    if optimise_intensity:
        intensity_choices = None  # generated below
    else:
        intensity_choices = [float(fixed_intensity)]

    def linspace_minmax(a, b, n):
        a = float(a)
        b = float(b)
        if n <= 1:
            return [a]
        if b < a:
            a, b = b, a
        return list(np.linspace(a, b, int(n)))

    cap_m_grid = linspace_minmax(cap_m_min, cap_m_max, (n_caps if method == "Grid search" else 0) or 0)
    cap_r_grid = linspace_minmax(cap_r_min, cap_r_max, (n_caps if method == "Grid search" else 0) or 0)
    cap_f_grid = linspace_minmax(cap_f_min, cap_f_max, (n_caps if method == "Grid search" else 0) or 0)

    if method.startswith("Grid search"):
        intensity_grid = linspace_minmax(0.0, 1.0, n_intensity) if optimise_intensity else intensity_choices
        for pol in policy_choices:
            for inten in intensity_grid:
                for cm in cap_m_grid:
                    for cr in cap_r_grid:
                        for cf in cap_f_grid:
                            candidates.append(
                                dict(
                                    policy_name=str(pol),
                                    cull_intensity=float(inten),
                                    cap_muntjac=float(cm),
                                    cap_roe=float(cr),
                                    cap_fallow=float(cf),
                                )
                            )
    else:
        rng = np.random.default_rng(123)
        for _ in range(int(max_evals)):
            pol = rng.choice(policy_choices)
            inten = float(rng.uniform(0.0, 1.0)) if optimise_intensity else float(intensity_choices[0])

            cm = float(rng.uniform(min(cap_m_min, cap_m_max), max(cap_m_min, cap_m_max)))
            cr = float(rng.uniform(min(cap_r_min, cap_r_max), max(cap_r_min, cap_r_max)))
            cf = float(rng.uniform(min(cap_f_min, cap_f_max), max(cap_f_min, cap_f_max)))

            candidates.append(
                dict(
                    policy_name=str(pol),
                    cull_intensity=float(inten),
                    cap_muntjac=float(cm),
                    cap_roe=float(cr),
                    cap_fallow=float(cf),
                )
            )

    if len(candidates) == 0:
        st.error("No candidates to evaluate. Check bounds/grid settings.")
        st.stop()

    results = []
    prog = st.progress(0)
    status = st.empty()

    for i, cand in enumerate(candidates, start=1):
        status.write(f"Evaluating candidate {i:,}/{len(candidates):,}...")

        caps_tuple = (cand["cap_muntjac"], cand["cap_roe"], cand["cap_fallow"])

        (
            df_metrics,
            totals_stats,
            cull_stats,
            cost_stats,
            cull_by_class_stats,
            class_split_stats,
            class_abundance_stats,
            is_ensemble,
            df_cull_last_draw,
        ) = run_scenario(
            years=int(years),
            bio_mode=str(bio_mode),
            n_draws=int(n_draws),
            rho=float(rho),
            cull_intensity=float(cand["cull_intensity"]),
            weight_name=str(cand["policy_name"]),
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
            init_totals_tuple=init_tuple,
            targets_tuple=targets_tuple,
            caps_tuple=caps_tuple,
            score_kwargs_override=score_kwargs_override,
            fixed_score_refs=fixed_refs_for_opt,
        )

        mean_score = float(df_metrics["score"].mean())
        flags = _binding_flags(
            df_cull_last_draw,
            caps={"muntjac": cand["cap_muntjac"], "roe": cand["cap_roe"], "fallow": cand["cap_fallow"]},
            annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
        )

        results.append(
            dict(
                mean_score=mean_score,
                policy_name=cand["policy_name"],
                cull_intensity=cand["cull_intensity"],
                cap_muntjac=cand["cap_muntjac"],
                cap_roe=cand["cap_roe"],
                cap_fallow=cand["cap_fallow"],
                budget_binding=flags["budget_binding"],
                cap_binding_muntjac=flags["cap_binding_muntjac"],
                cap_binding_roe=flags["cap_binding_roe"],
                cap_binding_fallow=flags["cap_binding_fallow"],
                totals_stats=totals_stats,
                cull_stats=cull_stats,
                cost_stats=cost_stats,
                cull_by_class_stats=cull_by_class_stats,
                class_split_stats=class_split_stats,
                class_abundance_stats=class_abundance_stats,
                is_ensemble=bool(is_ensemble),
            )
        )

        prog.progress(int(100 * i / len(candidates)))

    status.write(f"Done. Evaluated {len(candidates):,} candidates in {time.time() - t0:.1f}s.")
    prog.empty()

    results_sorted = sorted(results, key=lambda r: float(r["mean_score"]))

    df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k
                not in (
                    "totals_stats",
                    "cull_stats",
                    "cost_stats",
                    "cull_by_class_stats",
                    "class_split_stats",
                    "class_abundance_stats",
                )
            }
            for r in results_sorted
        ]
    )
    df.insert(0, "Candidate", np.arange(1, len(df) + 1))

    # Store for reruns
    st.session_state["polrec_results_sorted"] = results_sorted
    st.session_state["polrec_df"] = df
    st.session_state["polrec_meta"] = dict(
        years=int(years),
        rho=float(rho),
        bio_mode=str(bio_mode),
        n_draws=int(n_draws),
        init_tuple=tuple(init_tuple),
        targets_tuple=tuple(targets_tuple),
        annual_budget_total=(float(annual_budget_total) if annual_budget_total is not None else None),
        show_top_n=int(show_top_n),
    )


# -----------------------------
# Render cached results (prevents reset on widget change)
# -----------------------------
if st.session_state["polrec_results_sorted"] is None:
    st.info("Run optimisation to see results.")
else:
    results_sorted = st.session_state["polrec_results_sorted"]
    df = st.session_state["polrec_df"]
    meta = st.session_state["polrec_meta"] or {}

    years_r = int(meta.get("years", years))
    rho_r = float(meta.get("rho", rho))
    bio_mode_r = str(meta.get("bio_mode", bio_mode))
    n_draws_r = int(meta.get("n_draws", n_draws))
    init_tuple_r = tuple(meta.get("init_tuple", init_tuple))
    targets_tuple_r = tuple(meta.get("targets_tuple", targets_tuple))
    annual_budget_total_r = meta.get("annual_budget_total", annual_budget_total)
    show_top_n_r = int(meta.get("show_top_n", show_top_n))

    # -----------------------------
    # Top candidates table (collapsible)
    # -----------------------------
    with st.expander("Top candidates (table)", expanded=False):
        st.caption(
            "Each row is one candidate set of management controls (cull policy, intensity, and per-species annual caps), ranked by mean score. "
            "The binding flags are quick diagnostics for which constraints were active:\n"
            "- **budget_binding**: at least one year hit the annual budget cap.\n"
            "- **cap_binding_<species>**: at least one year hit that species’ annual cull cap.\n"
            "These flags are computed from a single representative simulation draw, so treat them as indicative."
        )
        st.dataframe(df.head(int(show_top_n_r)), use_container_width=True, hide_index=True)

    # -----------------------------
    # Best strategy
    # -----------------------------
    best = results_sorted[0]
    st.markdown("---")
    st.subheader("Recommended management plan (best strategy)")



    # Fixed strategy controls summary
    init_m, init_r, init_f = init_tuple_r
    targ_m, targ_r, targ_f = targets_tuple_r
    budget_text = "Disabled" if annual_budget_total_r is None else f"£{float(annual_budget_total_r):,.0f} / year"
    draws_text = f"{int(n_draws_r)} draw(s)" if str(bio_mode_r) == "ensemble" else "single run"

    st.markdown(
        f"""
**Fixed strategy controls (used in optimisation)**
- Horizon: **{int(years_r)} years**
- Biology mode: **{str(bio_mode_r)}** ({draws_text})
- Max annual cull fraction per class: **{float(rho_r):.2f}**
- Initial totals: Muntjac **{init_m:.0f}**, Roe **{init_r:.0f}**, Fallow **{init_f:.0f}**
- Targets (post-cull totals): Muntjac **{targ_m:.0f}**, Roe **{targ_r:.0f}**, Fallow **{targ_f:.0f}**
- Annual budget cap: **{budget_text}**
"""
    )

    st.markdown(
        f"""
**Optimised management controls**
- Cull policy: **{best['policy_name']}**
- Cull intensity: **{best['cull_intensity']:.3f}**
- Annual cull caps: Muntjac **{best['cap_muntjac']:.1f}**, Roe **{best['cap_roe']:.1f}**, Fallow **{best['cap_fallow']:.1f}**
"""
    )

    # -----------------------------
    # Cull plan table (collapsible)
    # -----------------------------
    plan = best["cull_by_class_stats"]
    sp_pick_plan = st.selectbox(
        "Species for plan table",
        model.SPECIES,
        format_func=lambda s: SPECIES_DISPLAY.get(s, s),
        key="polrec_plan_species",
    )

    if bio_mode_r == "ensemble":
        st.caption(
            "This is the model’s realised cull schedule by year and class, reported as the mean across biology draws "
            "with a 10–90% uncertainty band."
        )
    else:
        st.caption(
            "This is the model’s realised cull schedule by year and class, reported as a single deterministic run (no uncertainty band)."
        )

    with st.expander("Year-by-year cull plan by class", expanded=False):
        is_ens = (bio_mode_r == "ensemble")
        st.dataframe(_plan_table_for_species(plan, sp_pick_plan, show_band=is_ens), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download mean plan CSV",
                data=_plan_csv_for_species(plan, sp_pick_plan, "mean").to_csv(index=False).encode("utf-8"),
                file_name=f"policy_plan_{sp_pick_plan}_mean.csv",
                mime="text/csv",
            )

        if bio_mode_r == "ensemble":
            with c2:
                st.download_button(
                    "Download P10 plan CSV",
                    data=_plan_csv_for_species(plan, sp_pick_plan, "lo").to_csv(index=False).encode("utf-8"),
                    file_name=f"policy_plan_{sp_pick_plan}_p10.csv",
                    mime="text/csv",
                )
            with c3:
                st.download_button(
                    "Download P90 plan CSV",
                    data=_plan_csv_for_species(plan, sp_pick_plan, "hi").to_csv(index=False).encode("utf-8"),
                    file_name=f"policy_plan_{sp_pick_plan}_p90.csv",
                    mime="text/csv",
                )

    # -----------------------------
    # Best strategy plots (requested)
    # -----------------------------
    st.markdown("---")
    st.subheader("Best strategy — plots")

    year_tot = best["totals_stats"]["year"]
    year_flow = best["cull_stats"]["year"]

    # 4) Best strategy plots: all-species total abundance, total realised cull, total cost
    st.pyplot(
        fig_series_with_band(
            year_tot,
            best["totals_stats"]["all_total"],
            "All-species total abundance (mean + 10–90% band)",
            "Abundance",
            target=None,
            y0=True,
        )
    )
    st.pyplot(
        fig_series_with_band(
            year_flow,
            best["cull_stats"],
            "Total realised cull per year (mean + 10–90% band)",
            "Cull",
            target=None,
            y0=True,
        )
    )

    st.pyplot(
        fig_series_with_band(
            year_flow,
            best["cost_stats"],
            "Total cost per year (mean + 10–90% band)",
            "Cost (£)",
            target=None,
            y0=True,
        )
    )

    # 5) Move all-species total abundance split by species into best strategy plots
    st.pyplot(
        fig_stacked_species_totals(
            year_tot,
            best["totals_stats"],
            "All-species total abundance split by species (stacked mean)",
            "Abundance",
            species_display=SPECIES_DISPLAY,
        )
    )

    # 6) Group species abundance + class split + class abundance under one species selector
    st.markdown("### Species detail")
    sp_detail = st.selectbox(
        "Species (detail plots)",
        model.SPECIES,
        format_func=lambda x: SPECIES_DISPLAY.get(x, x),
        key="polrec_species_detail_best",
    )

    targets_total = {"muntjac": float(targets_tuple_r[0]), "roe": float(targets_tuple_r[1]), "fallow": float(targets_tuple_r[2])}

    st.pyplot(
        fig_series_with_band(
            year_tot,
            best["totals_stats"][f"{sp_detail}_total"],
            f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)} total (mean + 10–90% band; dashed = target)",
            "Abundance",
            target=targets_total[sp_detail],
            y0=True,
        )
    )

    show_band_detail = bool(best.get("is_ensemble", False)) and st.checkbox(
        "Show 10–90% band on class split/abundance",
        value=False,
        key="polrec_detail_show_band_best",
    )

    st.pyplot(
        fig_class_split_over_time(
            best["class_split_stats"]["year"],
            best["class_split_stats"][sp_detail],
            f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)}: class split over time",
            bool(show_band_detail),
        )
    )

    st.pyplot(
        fig_stacked_abundance(
            best["class_abundance_stats"]["year"],
            best["class_abundance_stats"][sp_detail],
            f"{SPECIES_DISPLAY.get(sp_detail, sp_detail)}: class abundance (stacked mean)",
            "Abundance",
            show_band=bool(show_band_detail),
        )
    )

    # -----------------------------
    # Top / middle / bottom candidate plots (collapsible)
    # -----------------------------
    st.markdown("---")

    n = len(results_sorted)
    if n > 0:
        idx_top = 0
        idx_mid = n // 2
        idx_bot = n - 1

        # If n is small, avoid duplicates by keeping unique indices in order.
        idxs = []
        for j in (idx_top, idx_mid, idx_bot):
            if j not in idxs:
                idxs.append(j)

        picks = [results_sorted[j] for j in idxs]
        labels = []
        for j in idxs:
            if j == idx_top:
                labels.append(f"Top (rank 1/{n})")
            elif j == idx_bot:
                labels.append(f"Bottom (rank {n}/{n})")
            else:
                labels.append(f"Middle (rank {j+1}/{n})")

        with st.expander("Top / middle / bottom candidates — plots", expanded=False):
            totals_by = [(labels[k], r["totals_stats"]) for k, r in enumerate(picks)]
            cull_by = [(labels[k], r["cull_stats"]) for k, r in enumerate(picks)]
            cost_by = [(labels[k], r["cost_stats"]) for k, r in enumerate(picks)]

            year0 = picks[0]["totals_stats"]["year"]
            year1 = picks[0]["cull_stats"]["year"]

            st.pyplot(_plot_abundance(year0, totals_by))
            st.pyplot(_plot_series(year1, cull_by, "Total realised cull per year", "Cull"))
            st.pyplot(_plot_series(year1, cost_by, "Total cost per year", "Cost (£)"))

    # -----------------------------
    # Top 3 candidate plots (collapsible)
    # -----------------------------
    st.markdown("---")
    top3 = results_sorted[:3]
    if len(top3) > 0:
        with st.expander("Top 3 candidates — plots", expanded=False):
            totals_by = [(f"Candidate {i}", r["totals_stats"]) for i, r in enumerate(top3, start=1)]
            cull_by = [(f"Candidate {i}", r["cull_stats"]) for i, r in enumerate(top3, start=1)]
            cost_by = [(f"Candidate {i}", r["cost_stats"]) for i, r in enumerate(top3, start=1)]

            year0 = top3[0]["totals_stats"]["year"]
            year1 = top3[0]["cull_stats"]["year"]

            st.pyplot(_plot_abundance(year0, totals_by))
            st.pyplot(_plot_series(year1, cull_by, "Total realised cull per year", "Cull"))
            st.pyplot(_plot_series(year1, cost_by, "Total cost per year", "Cost (£)"))
