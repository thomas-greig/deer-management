import numpy as np
import pandas as pd
import streamlit as st

import model


@st.cache_data(show_spinner=False)
def run_scenario(
    *,
    years: int,
    bio_mode: str,
    n_draws: int,
    rho: float,
    cull_intensity: float,
    weight_name: str,
    annual_budget_total: float | None,
    init_totals_tuple: tuple[float, float, float],
    targets_tuple: tuple[float, float, float],
    caps_tuple: tuple[float, float, float],
):
    """
    Runs ONE scenario and returns:
      - df_metrics (draw-by-draw)
      - totals_stats (mean/10-90 band for totals, year 0..years)
      - cull_stats   (mean/10-90 band, year 1..years)
      - cost_stats   (mean/10-90 band, year 1..years)
      - is_ensemble
    """
    defaults = model.build_defaults()

    score_kwargs = defaults["score_kwargs"]
    weight_scenarios = defaults["weight_scenarios"]

    age_fracs = defaults["age_fracs_default"]
    male_frac = defaults["male_frac_default"]

    init_totals = {"muntjac": init_totals_tuple[0], "roe": init_totals_tuple[1], "fallow": init_totals_tuple[2]}
    N0 = {sp: model.make_initial_state_from_total(init_totals[sp], age_fracs, male_frac) for sp in model.SPECIES}

    # Migration baseline equals chosen initial totals
    N_OUT = dict(init_totals)

    targets_total = {"muntjac": targets_tuple[0], "roe": targets_tuple[1], "fallow": targets_tuple[2]}
    annual_cull_limits = {"muntjac": caps_tuple[0], "roe": caps_tuple[1], "fallow": caps_tuple[2]}

    cull_weights = weight_scenarios[weight_name]

    if bio_mode == "fixed":
        bios = [defaults["biology_fixed"]]
    else:
        rng = np.random.default_rng(123)
        bios = model.draw_biology_ensemble(
            base_params=defaults["base_params"],
            base_alpha=defaults["base_alpha"],
            base_gamma=defaults["base_gamma"],
            n_draws=n_draws,
            rng=rng,
            K_range=(0.8, 1.2),
            beta_range=(0.8, 1.2),
            alpha_range=(0.85, 1.15),
            gamma_range=(0.7, 1.3),
        )

    metrics = []
    totals_all_draws = []   # year 0..years (DataFrame)
    cull_all_draws = []     # year 1..years (np array)
    cost_all_draws = []     # year 1..years (np array)

    for idx, bio in enumerate(bios):
        df_classes, df_cull = model.simulate(
            N0=N0,
            params=bio["params"],
            alpha=bio["alpha"],
            targets_total=targets_total,
            N_out=N_OUT,
            gamma_mig=bio["gamma_mig"],
            cull_weights=cull_weights,
            max_cull_frac_per_class=rho,
            cull_intensity=cull_intensity,
            annual_cull_limits=annual_cull_limits,
            annual_budget_total=annual_budget_total,
            cost_params=defaults["COST_PARAMS"],
            years=years,
        )

        totals_df = model.compute_totals(df_classes)
        totals_all_draws.append(totals_df)

        if df_cull is None or df_cull.empty:
            cull_series = np.zeros(years, dtype=float)
            cost_series = np.zeros(years, dtype=float)
        else:
            year_index = np.arange(1, years + 1)
            tmp = df_cull.set_index("year")
            cull_series = tmp.reindex(year_index)["realised_total_all"].fillna(0.0).to_numpy(dtype=float)
            if "cost_total_all" in tmp.columns:
                cost_series = tmp.reindex(year_index)["cost_total_all"].fillna(0.0).to_numpy(dtype=float)
            else:
                cost_series = np.zeros(years, dtype=float)

        cull_all_draws.append(cull_series)
        cost_all_draws.append(cost_series)

        total_cull, total_cost, t_stable, steady_dev_late = model.score_components_v4(
            df_classes=df_classes,
            df_cull=df_cull,
            targets_total=targets_total,
            stable_tol_rel=score_kwargs["stable_tol_rel"],
            stable_window=score_kwargs["stable_window"],
            late_window=score_kwargs["late_window"],
        )

        score = model.scenario_score_v4(
            df_classes=df_classes,
            df_cull=df_cull,
            targets_total=targets_total,
            years=years,
            w_cull=score_kwargs["w_cull"],
            w_time=score_kwargs["w_time"],
            w_steady=score_kwargs["w_steady"],
            w_cost=score_kwargs["w_cost"],
            stable_tol_rel=score_kwargs["stable_tol_rel"],
            stable_window=score_kwargs["stable_window"],
            late_window=score_kwargs["late_window"],
        )

        metrics.append(
            dict(
                draw=int(idx),
                score=float(score),
                total_cull=float(total_cull),
                total_cost=float(total_cost),
                t_stable=int(t_stable),
                steady_dev_late=float(steady_dev_late),
            )
        )

    df_metrics = pd.DataFrame(metrics)

    # summaries: mean + 10–90% band (pointwise by year)
    years_axis = totals_all_draws[0]["year"].to_numpy()

    def stack_totals(col: str) -> np.ndarray:
        return np.vstack([tdf[col].to_numpy() for tdf in totals_all_draws])

    totals_stats = {"year": years_axis}
    for col in ["all_total", "muntjac_total", "roe_total", "fallow_total"]:
        X = stack_totals(col)
        totals_stats[col] = {
            "mean": X.mean(axis=0),
            "lo": np.quantile(X, 0.10, axis=0),
            "hi": np.quantile(X, 0.90, axis=0),
        }

    year1 = np.arange(1, years + 1)
    C = np.vstack(cull_all_draws)
    K = np.vstack(cost_all_draws)

    cull_stats = {"year": year1, "mean": C.mean(axis=0), "lo": np.quantile(C, 0.10, axis=0), "hi": np.quantile(C, 0.90, axis=0)}
    cost_stats = {"year": year1, "mean": K.mean(axis=0), "lo": np.quantile(K, 0.10, axis=0), "hi": np.quantile(K, 0.90, axis=0)}

    return df_metrics, totals_stats, cull_stats, cost_stats, (bio_mode == "ensemble")
