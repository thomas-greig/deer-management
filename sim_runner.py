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
    disable_density_dependence: bool = False,
    score_kwargs_override: dict | None = None,
    # if provided, these refs are used for normalisation (fair comparison / fixed ruler)
    fixed_score_refs: dict | None = None,
):
    """
    Runs one scenario and returns:
      (df_metrics,
       totals_stats,
       cull_stats,
       cost_stats,
       cull_by_class_stats,     
       class_split_stats,
       class_abundance_stats,
       is_ensemble,
       df_cull_last_draw)

    df_metrics includes:
      - score
      - raw metrics: total_cull, total_cost, t_stable, steady_dev_late
      - normalised metrics: n_cull, n_cost, n_time, n_steady
      - contributions: contrib_cull, contrib_cost, contrib_time, contrib_steady
      - reference scales used: total_cull_ref, total_cost_ref, time_ref, steady_ref

    fixed_score_refs must contain keys:
      {"total_cull_ref","total_cost_ref","time_ref","steady_ref"}

    
      This script assumes model.simulate() returns df_cull with per-class realised cull columns:
        f"{sp}_{lab}_cull" for sp in SPECIES and lab in STATE_LABELS
      (30 columns total).
    """
    defaults = model.build_defaults()

    # Optional: force density dependence off 
    if disable_density_dependence:
        for sp in model.SPECIES:
            defaults["base_params"][sp]["beta_f"] = 0.0
            defaults["base_params"][sp]["beta_s0"] = 0.0
            defaults["base_params"][sp]["beta_s_adult"] = 0.0
            defaults["biology_fixed"]["params"][sp]["beta_f"] = 0.0
            defaults["biology_fixed"]["params"][sp]["beta_s0"] = 0.0
            defaults["biology_fixed"]["params"][sp]["beta_s_adult"] = 0.0

    # Score params (override-able)
    score_kwargs = dict(defaults["score_kwargs"])
    if score_kwargs_override:
        score_kwargs.update(score_kwargs_override)

    # Inputs -> dicts
    age_fracs = defaults["age_fracs_default"]
    male_frac = defaults["male_frac_default"]

    init_totals = {
        "muntjac": float(init_totals_tuple[0]),
        "roe": float(init_totals_tuple[1]),
        "fallow": float(init_totals_tuple[2]),
    }
    N0 = {sp: model.make_initial_state_from_total(init_totals[sp], age_fracs, male_frac) for sp in model.SPECIES}

    N_OUT = dict(init_totals)  # migration baseline equals initial totals

    targets_total = {
        "muntjac": float(targets_tuple[0]),
        "roe": float(targets_tuple[1]),
        "fallow": float(targets_tuple[2]),
    }
    annual_cull_limits = {
        "muntjac": float(caps_tuple[0]),
        "roe": float(caps_tuple[1]),
        "fallow": float(caps_tuple[2]),
    }

    # Cull policy weights
    cull_weights = defaults["weight_scenarios"][str(weight_name)]

    # Biology draws (seed fixed to 123)
    if str(bio_mode) == "fixed":
        bios = [defaults["biology_fixed"]]
    else:
        rng = np.random.default_rng(123)
        bios = model.draw_biology_ensemble(
            base_params=defaults["base_params"],
            base_alpha=defaults["base_alpha"],
            base_gamma=defaults["base_gamma"],
            n_draws=int(n_draws),
            rng=rng,
            K_range=(0.8, 1.2),
            beta_range=(0.8, 1.2),
            alpha_range=(0.85, 1.15),
            gamma_range=(0.7, 1.3),
        )

    # Validate fixed refs if provided
    fixed_refs = None
    if fixed_score_refs is not None:
        fixed_refs = dict(fixed_score_refs)
        for k in ["total_cull_ref", "total_cost_ref", "time_ref", "steady_ref"]:
            if k not in fixed_refs:
                raise ValueError(f"fixed_score_refs missing key: {k}")
        fixed_refs["total_cull_ref"] = max(float(fixed_refs["total_cull_ref"]), 1.0)
        fixed_refs["total_cost_ref"] = max(float(fixed_refs["total_cost_ref"]), 1.0)
        fixed_refs["time_ref"] = max(float(fixed_refs["time_ref"]), 1.0)
        fixed_refs["steady_ref"] = max(float(fixed_refs["steady_ref"]), 1e-12)

    metrics = []
    totals_all_draws = []
    cull_all_draws = []
    cost_all_draws = []
    classes_all_draws = []

    # per-draw per-class cull time series (year 1..years)
    cull_class_all_draws: list[pd.DataFrame] = []

    df_cull_last_draw = None

    # expected per-class cull columns
    class_cols = [f"{sp}_{lab}_cull" for sp in model.SPECIES for lab in model.STATE_LABELS]

    for idx, bio in enumerate(bios):
        df_classes, df_cull = model.simulate(
            N0=N0,
            params=bio["params"],
            alpha=bio["alpha"],
            targets_total=targets_total,
            N_out=N_OUT,
            gamma_mig=bio["gamma_mig"],
            cull_weights=cull_weights,
            cull_intensity=float(cull_intensity),
            max_cull_frac_per_class=float(rho),
            annual_cull_limits=annual_cull_limits,
            annual_budget_total=annual_budget_total,
            cost_params=defaults["COST_PARAMS"],
            years=int(years),
        )

        df_cull_last_draw = df_cull
        classes_all_draws.append(df_classes)

        totals_df = model.compute_totals(df_classes)
        totals_all_draws.append(totals_df)

        year_index = np.arange(1, int(years) + 1)

        # Per-year realised cull/cost
        if df_cull is None or df_cull.empty:
            cull_series = np.zeros(int(years), dtype=float)
            cost_series = np.zeros(int(years), dtype=float)

            df_cull_classes = pd.DataFrame({"year": year_index})
            for c in class_cols:
                df_cull_classes[c] = 0.0
        else:
            tmp = df_cull.set_index("year").reindex(year_index)

            cull_series = (
                tmp.get("realised_total_all", pd.Series(index=year_index, data=0.0))
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            cost_series = (
                tmp.get("cost_total_all", pd.Series(index=year_index, data=0.0))
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            # per-class realised cull (30 series)
            missing = [c for c in class_cols if c not in tmp.columns]
            if missing:
                raise KeyError(
                    "df_cull is missing per-class cull columns. "
                    "Update model.simulate() to include columns like f'{sp}_{lab}_cull'. "
                    f"Missing examples: {missing[:5]}"
                )

            df_cull_classes = tmp[class_cols].fillna(0.0).reset_index().rename(columns={"index": "year"})
            df_cull_classes["year"] = year_index

        cull_all_draws.append(cull_series)
        cost_all_draws.append(cost_series)
        cull_class_all_draws.append(df_cull_classes)

        # --- Raw components (v4)
        total_cull, total_cost, t_stable, steady_dev_late = model.score_components_v4(
            df_classes=df_classes,
            df_cull=df_cull,
            targets_total=targets_total,
            stable_tol_rel=float(score_kwargs["stable_tol_rel"]),
            stable_window=int(score_kwargs["stable_window"]),
            late_window=int(score_kwargs["late_window"]),
        )

        # --- Reference scales: fixed if provided, else scenario-derived
        if fixed_refs is not None:
            refs = fixed_refs
        else:
            refs = model.score_reference_scales(
                years=int(years),
                annual_budget_total=annual_budget_total,
                annual_cull_limits=annual_cull_limits,
                cost_params=defaults["COST_PARAMS"],
                steady_ref=float(defaults.get("score_refs", {}).get("steady_ref", 1.0)),
            )

        # --- Normalised metrics
        n_cull = float(total_cull) / float(refs["total_cull_ref"])
        n_cost = float(total_cost) / float(refs["total_cost_ref"])
        n_time = float(t_stable) / float(refs["time_ref"])
        n_steady = float(steady_dev_late) / float(refs["steady_ref"])

        # --- Contributions and total score
        contrib_cull = float(score_kwargs["w_cull"]) * n_cull
        contrib_cost = float(score_kwargs["w_cost"]) * n_cost
        contrib_time = float(score_kwargs["w_time"]) * n_time
        contrib_steady = float(score_kwargs["w_steady"]) * n_steady
        score = contrib_cull + contrib_cost + contrib_time + contrib_steady

        metrics.append(
            dict(
                draw=int(idx),
                score=float(score),
                # raw
                total_cull=float(total_cull),
                total_cost=float(total_cost),
                t_stable=int(t_stable),
                steady_dev_late=float(steady_dev_late),
                # refs
                total_cull_ref=float(refs["total_cull_ref"]),
                total_cost_ref=float(refs["total_cost_ref"]),
                time_ref=float(refs["time_ref"]),
                steady_ref=float(refs["steady_ref"]),
                # normalised
                n_cull=float(n_cull),
                n_cost=float(n_cost),
                n_time=float(n_time),
                n_steady=float(n_steady),
                # contributions
                contrib_cull=float(contrib_cull),
                contrib_cost=float(contrib_cost),
                contrib_time=float(contrib_time),
                contrib_steady=float(contrib_steady),
            )
        )

    df_metrics = pd.DataFrame(metrics)

    # ---------- Ensemble summaries ----------
    years_axis = totals_all_draws[0]["year"].to_numpy()

    def stack_totals(col: str) -> np.ndarray:
        return np.vstack([tdf[col].to_numpy(dtype=float) for tdf in totals_all_draws])

    totals_stats = {"year": years_axis}
    for col in ["all_total", "muntjac_total", "roe_total", "fallow_total"]:
        X = stack_totals(col)
        totals_stats[col] = {
            "mean": X.mean(axis=0),
            "lo": np.quantile(X, 0.10, axis=0),
            "hi": np.quantile(X, 0.90, axis=0),
        }

    year1 = np.arange(1, int(years) + 1)
    C = np.vstack(cull_all_draws)
    K = np.vstack(cost_all_draws)

    cull_stats = {
        "year": year1,
        "mean": C.mean(axis=0),
        "lo": np.quantile(C, 0.10, axis=0),
        "hi": np.quantile(C, 0.90, axis=0),
    }
    cost_stats = {
        "year": year1,
        "mean": K.mean(axis=0),
        "lo": np.quantile(K, 0.10, axis=0),
        "hi": np.quantile(K, 0.90, axis=0),
    }

    # NEW: per-class cull stats (mean + 10–90%)
    cull_by_class_stats: dict = {"year": year1}
    for sp in model.SPECIES:
        block = {}
        for lab in model.STATE_LABELS:
            col = f"{sp}_{lab}_cull"
            X = np.vstack([df[col].to_numpy(dtype=float) for df in cull_class_all_draws])
            block[lab] = {
                "mean": X.mean(axis=0),
                "lo": np.quantile(X, 0.10, axis=0),
                "hi": np.quantile(X, 0.90, axis=0),
            }
        cull_by_class_stats[sp] = block

    # ----- class split stats -----
    class_split_stats: dict = {"year": years_axis}

    def stack_props(sp: str, lab: str) -> np.ndarray:
        cols = [f"{sp}_{x}" for x in model.STATE_LABELS]
        out = []
        for dfc in classes_all_draws:
            num = dfc[f"{sp}_{lab}"].to_numpy(dtype=float)
            den = dfc[cols].sum(axis=1).to_numpy(dtype=float)
            den = np.maximum(den, 1e-12)
            out.append(num / den)
        return np.vstack(out)

    for sp in model.SPECIES:
        block = {}
        for lab in model.STATE_LABELS:
            X = stack_props(sp, lab)
            block[lab] = {
                "mean": X.mean(axis=0),
                "lo": np.quantile(X, 0.10, axis=0),
                "hi": np.quantile(X, 0.90, axis=0),
            }
        class_split_stats[sp] = block

    # ----- class abundance stats -----
    class_abundance_stats: dict = {"year": years_axis}

    def stack_abs(sp: str, lab: str) -> np.ndarray:
        return np.vstack([dfc[f"{sp}_{lab}"].to_numpy(dtype=float) for dfc in classes_all_draws])

    for sp in model.SPECIES:
        block = {}
        for lab in model.STATE_LABELS:
            X = stack_abs(sp, lab)
            block[lab] = {
                "mean": X.mean(axis=0),
                "lo": np.quantile(X, 0.10, axis=0),
                "hi": np.quantile(X, 0.90, axis=0),
            }
        class_abundance_stats[sp] = block

    is_ensemble = (str(bio_mode) == "ensemble")

    return (
        df_metrics,
        totals_stats,
        cull_stats,
        cost_stats,
        cull_by_class_stats,  # NEW
        class_split_stats,
        class_abundance_stats,
        is_ensemble,
        df_cull_last_draw,
    )
