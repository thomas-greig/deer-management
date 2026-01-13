from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# OPTIMAL DEER MANAGEMENT — STREAMLIT-READY MODULE
# + Policies (user-specified fixed weights)
# + Density-dependent adult survival via beta_s_adult
# + NEW: Cull intensity parameter (scales annual desire)
# ============================================================

SPECIES = ["muntjac", "roe", "fallow"]

STATE_LABELS = [
    "M0_<1",
    "F0_<1",
    "M1_1-2",
    "F1_1-2",
    "M2a_2-5",
    "F2a_2-5",
    "M2b_5-8",
    "F2b_5-8",
    "M2c_8+",
    "F2c_8+",
]
IDX = {
    "M0": 0,
    "F0": 1,
    "M1": 2,
    "F1": 3,
    "M2a": 4,
    "F2a": 5,
    "M2b": 6,
    "F2b": 7,
    "M2c": 8,
    "F2c": 9,
}

# Defaults (can be overridden by app controls)
ANNUAL_CULL_LIMITS = {"muntjac": 600.0, "roe": 500.0, "fallow": 400.0}
ANNUAL_BUDGET_TOTAL = 40_000.0  # UPDATED default control
COST_PARAMS = {
    "muntjac": {"linear": 80.0, "scale": 0.0583333333, "power": 2.0},
    "roe": {"linear": 70.0, "scale": 0.0700000000, "power": 2.0},
    "fallow": {"linear": 60.0, "scale": 0.0750000000, "power": 2.0},
}

# Migration removal weights
MIGRATION_REMOVE_WEIGHTS = np.zeros(10)
for k in ["M1", "F1", "M2a", "F2a", "M2b", "F2b", "M2c", "F2c"]:
    MIGRATION_REMOVE_WEIGHTS[IDX[k]] = 1.0 / 8.0


# -----------------------
# HELPERS
# -----------------------
def density_factor(P: float, K: float, beta: float) -> float:
    """Beverton–Holt style multiplier: g = 1 / (1 + beta * P / K) in (0,1]."""
    if beta <= 0:
        return 1.0
    K = max(float(K), 1e-9)
    return 1.0 / (1.0 + float(beta) * float(P) / K)


def allocate_cull_by_weights(N: np.ndarray, total_cull: float, weights: np.ndarray) -> np.ndarray:
    """
    Allocate desired total cull across classes by weights, respecting availability.
    If a class is short, leftover is reallocated among remaining eligible classes.
    """
    N = np.asarray(N, dtype=float)
    weights = np.asarray(weights, dtype=float)
    H = np.zeros_like(N)

    remaining = float(max(total_cull, 0.0))
    eligible = (weights > 0) & (N > 0)

    while remaining > 1e-9 and eligible.any():
        w = weights * eligible
        w_sum = w.sum()
        if w_sum <= 0:
            break

        proposal = remaining * (w / w_sum)
        cap = N - H
        take = np.minimum(proposal, cap)

        H += take
        remaining -= take.sum()

        eligible = (weights > 0) & ((N - H) > 1e-9)

    return H


def cap_total_harvest(H: np.ndarray, max_total: float | None) -> np.ndarray:
    """If sum(H) > max_total, scale proportionally to hit the cap."""
    if max_total is None:
        return H
    max_total = float(max_total)
    if max_total < 0:
        max_total = 0.0
    tot = float(np.sum(H))
    if tot <= 1e-12 or tot <= max_total:
        return H
    return H * (max_total / tot)


def annual_harvest_cost(h: float, cost_params: dict) -> float:
    """cost(h) = linear*h + scale*h^power."""
    h = float(max(h, 0.0))
    linear = float(cost_params.get("linear", 0.0))
    scale = float(cost_params.get("scale", 0.0))
    power = float(cost_params.get("power", 1.0))
    power = max(power, 1.0)
    return linear * h + scale * (h**power)


def total_cost_from_species_harvests(h_by_sp: dict[str, float], cost_params: dict[str, dict] | None) -> float:
    if cost_params is None:
        return 0.0
    return float(sum(annual_harvest_cost(h_by_sp.get(sp, 0.0), cost_params.get(sp, {})) for sp in SPECIES))


def optimal_species_harvest_under_budget_A2(
    overshoot: dict[str, float],
    h_caps: dict[str, float],
    budget_total: float | None,
    cost_params: dict[str, dict] | None,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> tuple[dict[str, float], bool]:
    """
    A2 allocator (species-level): allocate h_s in [0, h_cap_s] under total annual budget.

    This does not choose the policy over time, it only resolves a single-year budget
    conflict across species given per-species feasible caps.
    """
    if budget_total is None or cost_params is None:
        return {sp: float(h_caps.get(sp, 0.0)) for sp in SPECIES}, False

    B = float(budget_total)
    if B < 0:
        B = 0.0

    h_take_caps = {sp: float(max(h_caps.get(sp, 0.0), 0.0)) for sp in SPECIES}
    c_caps = total_cost_from_species_harvests(h_take_caps, cost_params)
    if c_caps <= B + 1e-9:
        return h_take_caps, False

    if B <= 1e-12:
        return {sp: 0.0 for sp in SPECIES}, True

    def h_given_lambda(lam: float) -> dict[str, float]:
        lam = float(max(lam, 1e-18))
        out: dict[str, float] = {}
        for sp in SPECIES:
            w = float(max(overshoot.get(sp, 0.0), 0.0))
            cap = float(max(h_caps.get(sp, 0.0), 0.0))
            if w <= 0.0 or cap <= 0.0:
                out[sp] = 0.0
                continue

            cp = cost_params.get(sp, {})
            linear = float(cp.get("linear", 0.0))
            scale = float(cp.get("scale", 0.0))
            power = float(cp.get("power", 1.0))
            power = max(power, 1.0)

            if scale <= 0.0 or power <= 1.0:
                if linear <= 1e-12:
                    out[sp] = cap
                else:
                    out[sp] = cap if (w / lam) > linear else 0.0
                continue

            rhs = (w / lam) - linear
            if rhs <= 0.0:
                out[sp] = 0.0
                continue

            denom = scale * power
            if denom <= 1e-18:
                out[sp] = 0.0
                continue

            h_star = (rhs / denom) ** (1.0 / (power - 1.0))
            out[sp] = float(min(max(h_star, 0.0), cap))
        return out

    lam_lo = 1e-12
    lam_hi = 1.0
    for _ in range(80):
        h_hi = h_given_lambda(lam_hi)
        c_hi = total_cost_from_species_harvests(h_hi, cost_params)
        if c_hi <= B + 1e-9:
            break
        lam_hi *= 2.0
    else:
        return {sp: 0.0 for sp in SPECIES}, True

    best_h = h_given_lambda(lam_hi)
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        h_mid = h_given_lambda(lam_mid)
        c_mid = total_cost_from_species_harvests(h_mid, cost_params)

        if c_mid > B:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
            best_h = h_mid

        if abs(lam_hi - lam_lo) <= tol * max(1.0, lam_hi):
            break

    c_best = total_cost_from_species_harvests(best_h, cost_params)
    budget_binding = c_best >= (B - 1e-3)
    return best_h, bool(budget_binding)


def make_initial_state_from_total(
    total: float,
    age_fracs=(0.20, 0.25, 0.35, 0.15, 0.05),
    male_frac=0.50,
) -> np.ndarray:
    """Build a 10-class initial state from total abundance."""
    a0, a1, a2a, a2b, a2c = age_fracs
    if not np.isclose(a0 + a1 + a2a + a2b + a2c, 1.0):
        raise ValueError("age_fracs must sum to 1")

    M0 = total * a0 * male_frac
    F0 = total * a0 * (1 - male_frac)

    M1 = total * a1 * male_frac
    F1 = total * a1 * (1 - male_frac)

    M2a = total * a2a * male_frac
    F2a = total * a2a * (1 - male_frac)

    M2b = total * a2b * male_frac
    F2b = total * a2b * (1 - male_frac)

    M2c = total * a2c * male_frac
    F2c = total * a2c * (1 - male_frac)

    return np.array([M0, F0, M1, F1, M2a, F2a, M2b, F2b, M2c, F2c], dtype=float)


def apply_net_migration(N_vec: np.ndarray, net_mig: float) -> np.ndarray:
    """
    net_mig > 0: add to M1/F1 (50:50)
    net_mig < 0: remove from (M1,F1, adults) using MIGRATION_REMOVE_WEIGHTS with reallocation.
    """
    N_vec = np.asarray(N_vec, dtype=float).copy()

    if net_mig > 0:
        N_vec[IDX["M1"]] += 0.5 * net_mig
        N_vec[IDX["F1"]] += 0.5 * net_mig
        return N_vec

    if net_mig < 0:
        outflow = -net_mig
        removal = allocate_cull_by_weights(N_vec, outflow, MIGRATION_REMOVE_WEIGHTS)
        removal = np.minimum(removal, N_vec)
        N_vec -= removal
        return np.maximum(N_vec, 0.0)

    return N_vec


def normalise_weights(w: np.ndarray) -> np.ndarray:
    """Ensure weights are nonnegative and sum to 1 if positive; otherwise all zeros."""
    w = np.asarray(w, dtype=float).copy()
    w[w < 0] = 0.0
    s = w.sum()
    if s > 0:
        w /= s
    return w


# -----------------------
# BIOLOGY
# -----------------------
def step_species(N: np.ndarray, sp_params: dict, pressure: float, harvest: np.ndarray | None = None) -> np.ndarray:
    """
    One-year update for a single species.

    N order (10):
      [M0,F0,M1,F1,M2a,F2a,M2b,F2b,M2c,F2c]
    """
    N = np.asarray(N, dtype=float)

    s = np.asarray(sp_params["survival"], dtype=float)      # length 10
    fert = np.asarray(sp_params["fertility"], dtype=float)  # length 5: [F0,F1,F2a,F2b,F2c]
    K = float(sp_params["K"])
    beta_f = float(sp_params.get("beta_f", 0.0))
    beta_s0 = float(sp_params.get("beta_s0", 0.0))
    beta_s_adult = float(sp_params.get("beta_s_adult", 0.0))
    male_birth_fraction = float(sp_params.get("male_birth_fraction", 0.5))

    g_f = density_factor(pressure, K, beta_f)
    g_s0 = density_factor(pressure, K, beta_s0)
    g_sA = density_factor(pressure, K, beta_s_adult)  # adult survival multiplier

    # births
    F0 = N[IDX["F0"]]
    F1 = N[IDX["F1"]]
    F2a = N[IDX["F2a"]]
    F2b = N[IDX["F2b"]]
    F2c = N[IDX["F2c"]]
    births = g_f * (fert[0] * F0 + fert[1] * F1 + fert[2] * F2a + fert[3] * F2b + fert[4] * F2c)

    M0_next = male_birth_fraction * births
    F0_next = (1.0 - male_birth_fraction) * births

    # juvenile survival (density-dependent)
    M1_next = g_s0 * s[IDX["M0"]] * N[IDX["M0"]]
    F1_next = g_s0 * s[IDX["F0"]] * N[IDX["F0"]]

    # adult survival transitions (density-dependent)
    M2a_next = g_sA * s[IDX["M1"]] * N[IDX["M1"]]
    F2a_next = g_sA * s[IDX["F1"]] * N[IDX["F1"]]

    M2b_next = g_sA * s[IDX["M2a"]] * N[IDX["M2a"]]
    F2b_next = g_sA * s[IDX["F2a"]] * N[IDX["F2a"]]

    M2c_next = g_sA * (s[IDX["M2b"]] * N[IDX["M2b"]] + s[IDX["M2c"]] * N[IDX["M2c"]])
    F2c_next = g_sA * (s[IDX["F2b"]] * N[IDX["F2b"]] + s[IDX["F2c"]] * N[IDX["F2c"]])

    N_next = np.array(
        [M0_next, F0_next, M1_next, F1_next, M2a_next, F2a_next, M2b_next, F2b_next, M2c_next, F2c_next],
        dtype=float,
    )

    if harvest is not None:
        harvest = np.asarray(harvest, dtype=float)
        harvest = np.minimum(harvest, N_next)
        N_next = N_next - harvest

    return np.maximum(N_next, 0.0)


# -----------------------
# POLICY
# -----------------------
def harvest_policy(
    N_by_species: dict[str, np.ndarray],
    targets_total: dict[str, float],
    cull_weights: np.ndarray,
    max_cull_frac_per_class: float,
    *,
    cull_intensity: float = 1.0,  # NEW
    annual_cull_limits: dict[str, float] | None = None,
    annual_budget_total: float | None = None,
    cost_params: dict[str, dict] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float], dict[str, float], bool]:
    """
    desired_total = cull_intensity * max(total - target, 0);

    Then:
      1) allocate by weights
      2) cap per class by rho
      3) cap total annual cull per species by absolute limits
      4) if annual budget is provided, allocate across species using A2

    Returns:
      harvest_vectors, desired_total, realised_total, per_sp_cost, budget_binding
    """
    cull_intensity = float(cull_intensity)
    cull_intensity = min(max(cull_intensity, 0.0), 1.0)

    harvest_vectors: dict[str, np.ndarray] = {}
    desired_total: dict[str, float] = {}
    realised_total: dict[str, float] = {}
    h_caps: dict[str, float] = {}

    for sp, N in N_by_species.items():
        total = float(N.sum())
        target = float(targets_total[sp])
        desire = cull_intensity * max(total - target, 0.0)  # NEW

        H = allocate_cull_by_weights(N, desire, cull_weights)
        H = np.minimum(H, max_cull_frac_per_class * N)

        if annual_cull_limits is not None:
            H = cap_total_harvest(H, annual_cull_limits.get(sp, None))

        harvest_vectors[sp] = H
        desired_total[sp] = float(desire)
        h_caps[sp] = float(H.sum())
        realised_total[sp] = float(H.sum())

    h_alloc, budget_binding = optimal_species_harvest_under_budget_A2(
        overshoot=desired_total,
        h_caps=h_caps,
        budget_total=annual_budget_total,
        cost_params=cost_params,
    )

    for sp in SPECIES:
        cap = float(max(h_caps.get(sp, 0.0), 0.0))
        want = float(max(h_alloc.get(sp, 0.0), 0.0))
        if cap <= 1e-12 or want <= 1e-12:
            harvest_vectors[sp] = harvest_vectors[sp] * 0.0
            realised_total[sp] = 0.0
        else:
            scale = min(want / cap, 1.0)
            harvest_vectors[sp] = harvest_vectors[sp] * scale
            realised_total[sp] = float(harvest_vectors[sp].sum())

    per_sp_cost: dict[str, float] = {}
    for sp in SPECIES:
        if cost_params is None:
            per_sp_cost[sp] = 0.0
        else:
            per_sp_cost[sp] = annual_harvest_cost(realised_total[sp], cost_params.get(sp, {}))

    return harvest_vectors, desired_total, realised_total, per_sp_cost, bool(budget_binding)


# -----------------------
# SIMULATION
# -----------------------
def simulate(
    N0: dict[str, np.ndarray],
    params: dict[str, dict],
    alpha: np.ndarray,
    targets_total: dict[str, float],
    N_out: dict[str, float],
    gamma_mig: dict[str, float],
    cull_weights: np.ndarray,
    max_cull_frac_per_class: float,
    *,
    cull_intensity: float = 1.0,  # NEW
    annual_cull_limits: dict[str, float] | None = None,
    annual_budget_total: float | None = None,
    cost_params: dict[str, dict] | None = None,
    years: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    alpha: (3,3) competition matrix aligned with SPECIES order
      pressure_i = sum_j alpha[i,j] * N_j_total
    """
    N = {sp: np.asarray(N0[sp], dtype=float).copy() for sp in SPECIES}

    class_rows = []
    cull_rows = []

    for t in range(years + 1):
        row = {"year": t}
        for sp in SPECIES:
            vec = N[sp]
            for k, lab in enumerate(STATE_LABELS):
                row[f"{sp}_{lab}"] = vec[k]
        class_rows.append(row)

        if t == years:
            break

        totals = np.array([N[sp].sum() for sp in SPECIES], dtype=float)

        harvest_vecs, desired, realised, per_sp_cost, budget_binding = harvest_policy(
            N_by_species=N,
            targets_total=targets_total,
            cull_weights=cull_weights,
            max_cull_frac_per_class=max_cull_frac_per_class,
            cull_intensity=cull_intensity,  # NEW
            annual_cull_limits=annual_cull_limits,
            annual_budget_total=annual_budget_total,
            cost_params=cost_params,
        )

        cull_rows.append(
            {
                "year": t + 1,

                # NEW: realised per-class cull (30 categories)
                **{
                    f"{sp}_{lab}_cull": float(harvest_vecs[sp][j])
                    for sp in SPECIES
                    for j, lab in enumerate(STATE_LABELS)
                },

                **{f"{sp}_desired": desired[sp] for sp in SPECIES},
                **{f"{sp}_realised": realised[sp] for sp in SPECIES},
                **{f"{sp}_cost": per_sp_cost[sp] for sp in SPECIES},
                "desired_total_all": sum(desired.values()),
                "realised_total_all": sum(realised.values()),
                "cost_total_all": sum(per_sp_cost.values()),
                "annual_budget_total": float(annual_budget_total) if annual_budget_total is not None else np.nan,
                "budget_binding": bool(budget_binding),
            }
        )


        N_next = {}
        for i, sp in enumerate(SPECIES):
            pressure = float(np.sum(alpha[i, :] * totals))
            N_next[sp] = step_species(N[sp], params[sp], pressure, harvest=harvest_vecs[sp])
            N_in = float(N_next[sp].sum())
            net_mig = float(gamma_mig[sp]) * (float(N_out[sp]) - N_in)
            N_next[sp] = apply_net_migration(N_next[sp], net_mig)
        N = N_next

    df_classes = pd.DataFrame(class_rows)
    df_cull = pd.DataFrame(cull_rows)
    return df_classes, df_cull


# -----------------------
# TOTALS + OBJECTIVE
# -----------------------
def compute_totals(df_classes: pd.DataFrame) -> pd.DataFrame:
    """year, species totals, all-species total."""
    out = pd.DataFrame({"year": df_classes["year"].to_numpy()})
    total_all = np.zeros(len(out), dtype=float)
    for sp in SPECIES:
        cols = [f"{sp}_{lab}" for lab in STATE_LABELS]
        s_tot = df_classes[cols].sum(axis=1).to_numpy()
        out[f"{sp}_total"] = s_tot
        total_all += s_tot
    out["all_total"] = total_all
    return out


def time_to_stabilise(series: np.ndarray, tol_rel: float = 0.005, window: int = 3, start_year: int = 1) -> int:
    """First year t such that for 'window' consecutive years, rel change <= tol_rel."""
    x = np.asarray(series, dtype=float)
    eps = 1e-9
    last_year = len(x) - 1

    for t in range(max(start_year, 1), last_year + 1):
        ok = True
        for k in range(t - window + 1, t + 1):
            if k <= 0:
                ok = False
                break
            rel = abs(x[k] - x[k - 1]) / max(abs(x[k - 1]), eps)
            if rel > tol_rel:
                ok = False
                break
        if ok and (t - window + 1) >= start_year:
            return t
    return last_year


def steady_deviation_late_window(
    totals_df: pd.DataFrame,
    targets_total: dict[str, float],
    late_window: int = 3,
) -> float:
    """Mean across species of mean squared relative deviation from targets over last L years."""
    eps = 1e-9
    years = totals_df["year"].to_numpy()
    last_year = int(years[-1])
    start = max(last_year - late_window + 1, 0)

    dev = 0.0
    for sp in SPECIES:
        target = float(targets_total[sp])
        vals = totals_df[f"{sp}_total"].to_numpy()[start : last_year + 1]
        rel = (vals - target) / (target + eps)
        dev += float(np.mean(rel**2))
    return dev


def score_components_v4(
    df_classes: pd.DataFrame,
    df_cull: pd.DataFrame,
    targets_total: dict[str, float],
    stable_tol_rel: float = 0.005,
    stable_window: int = 3,
    late_window: int = 3,
) -> tuple[float, float, int, float]:
    """Return (total_cull, total_cost, t_stable, steady_dev_late)."""
    totals = compute_totals(df_classes)
    all_total = totals["all_total"].to_numpy()

    t_stable = time_to_stabilise(all_total, tol_rel=stable_tol_rel, window=stable_window, start_year=1)
    steady_dev_late = steady_deviation_late_window(totals, targets_total, late_window=late_window)

    total_cull = 0.0 if df_cull.empty else float(df_cull["realised_total_all"].sum())
    total_cost = 0.0 if df_cull.empty else float(df_cull.get("cost_total_all", 0.0).sum())
    return float(total_cull), float(total_cost), int(t_stable), float(steady_dev_late)


def score_reference_scales(
    *,
    years: int,
    annual_budget_total: float | None,
    annual_cull_limits: dict[str, float],
    cost_params: dict[str, dict] | None,
    steady_ref: float = 1.0,
) -> dict[str, float]:
    """
    Reference scales used to normalise score metrics so the *weights* are interpretable.

    Defaults:
      - total_cull_ref: sum of per-species annual caps × years
      - total_cost_ref: (annual budget × years) if budget is provided,
                        else cost of culling at caps each year × years
      - time_ref: years
      - steady_ref: 1 (steady deviation is already dimensionless)
    """
    years_i = max(int(years), 1)

    cull_ref = float(sum(max(float(annual_cull_limits.get(sp, 0.0)), 0.0) for sp in SPECIES)) * float(years_i)
    cull_ref = max(cull_ref, 1.0)

    if annual_budget_total is not None:
        cost_ref = float(max(float(annual_budget_total), 0.0)) * float(years_i)
        cost_ref = max(cost_ref, 1.0)
    else:
        if cost_params is None:
            cost_ref = 1.0
        else:
            per_year_caps = {sp: float(max(float(annual_cull_limits.get(sp, 0.0)), 0.0)) for sp in SPECIES}
            per_year_cost = total_cost_from_species_harvests(per_year_caps, cost_params)
            cost_ref = max(float(per_year_cost) * float(years_i), 1.0)

    time_ref = float(years_i)
    steady_ref = max(float(steady_ref), 1e-12)

    return {
        "total_cull_ref": cull_ref,
        "total_cost_ref": cost_ref,
        "time_ref": time_ref,
        "steady_ref": steady_ref,
    }



# -----------------------
# BIOLOGICAL PARAMETER DRAWS (optional ensemble)
# -----------------------
def make_params_variant(base_params: dict[str, dict], K_scale: float, beta_scale: float) -> dict[str, dict]:
    out = {}
    for sp, d in base_params.items():
        out[sp] = dict(d)
        out[sp]["K"] = float(d["K"]) * float(K_scale)
        out[sp]["beta_f"] = float(d.get("beta_f", 0.0)) * float(beta_scale)
        out[sp]["beta_s0"] = float(d.get("beta_s0", 0.0)) * float(beta_scale)
        out[sp]["beta_s_adult"] = float(d.get("beta_s_adult", 0.0)) * float(beta_scale)
    return out


def scale_alpha(base_alpha: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(base_alpha, dtype=float) * float(scale)


def scale_gamma(base_gamma: dict[str, float], scale: float) -> dict[str, float]:
    return {sp: float(v) * float(scale) for sp, v in base_gamma.items()}


def draw_biology_ensemble(
    base_params: dict[str, dict],
    base_alpha: np.ndarray,
    base_gamma: dict[str, float],
    n_draws: int,
    rng: np.random.Generator,
    K_range=(0.8, 1.2),
    beta_range=(0.8, 1.2),
    alpha_range=(0.85, 1.15),
    gamma_range=(0.7, 1.3),
) -> list[dict]:
    loK, hiK = K_range
    lob, hib = beta_range
    loa, hia = alpha_range
    log, hig = gamma_range

    ensemble = []
    for _ in range(int(n_draws)):
        K_sc = float(rng.uniform(loK, hiK))
        b_sc = float(rng.uniform(lob, hib))
        a_sc = float(rng.uniform(loa, hia))
        g_sc = float(rng.uniform(log, hig))

        params = make_params_variant(base_params, K_scale=K_sc, beta_scale=b_sc)
        alpha = scale_alpha(base_alpha, a_sc)
        gamma = scale_gamma(base_gamma, g_sc)

        ensemble.append({"params": params, "alpha": alpha, "gamma_mig": gamma})
    return ensemble


# -----------------------
# POLICIES (RENAMED KEYS, SAME WEIGHTS)
# -----------------------
def policy_from_group_weights(*, F0=0.0, M0=0.0, F1=0.0, M1=0.0, F_adult=0.0, M_adult=0.0) -> np.ndarray:
    w = np.zeros(10, dtype=float)
    w[IDX["F0"]] = float(F0)
    w[IDX["M0"]] = float(M0)
    w[IDX["F1"]] = float(F1)
    w[IDX["M1"]] = float(M1)

    for k in ["F2a", "F2b", "F2c"]:
        w[IDX[k]] = float(F_adult) / 3.0
    for k in ["M2a", "M2b", "M2c"]:
        w[IDX[k]] = float(M_adult) / 3.0

    return normalise_weights(w)


def build_policy_scenarios() -> dict[str, np.ndarray]:
    return {
        "Balanced": policy_from_group_weights(
            F0=0.1666666667, M0=0.1666666667, F1=0.1666666667, M1=0.1666666667, F_adult=0.1666666667, M_adult=0.1666666667
        ),
        "Balanced sex, no juveniles": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.25, M1=0.25, F_adult=0.25, M_adult=0.25),
        "Female priority": policy_from_group_weights(F0=0.2333333333, M0=0.1, F1=0.2333333333, M1=0.1, F_adult=0.2333333333, M_adult=0.1),
        # was "Female mid priority, no juveniles"
        "Female priority, no juveniles": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.3, M1=0.2, F_adult=0.3, M_adult=0.2),
        # was "Female high priority, no juveniles"
        "Female high priority, no juveniles": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.4, M1=0.1, F_adult=0.4, M_adult=0.1),
        # was "Female mid priority, low juvenile, low yearlings"
        "Female priority, no juvenile, low yearlings": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.1333333333, M1=0.06666666667, F_adult=0.5, M_adult=0.3),
        # was "Female high priority, no juveniles, low yearlings"
        "Female high priority, no juveniles, low yearlings": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.06666666667, M1=0.03333333333, F_adult=0.7, M_adult=0.2),
        # was "Female mid priority, no juveniles or yearlings"
        "Female priority, no juvenile, no yearlings": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.0, M1=0.0, F_adult=0.6, M_adult=0.4),
        # was "Female high priority, no juveniles or yearlings"
        "Female high priority, no juveniles, no yearlings": policy_from_group_weights(F0=0.0, M0=0.0, F1=0.0, M1=0.0, F_adult=0.8, M_adult=0.2),
    }


# -----------------------
# DEFAULTS
# -----------------------
def build_defaults() -> dict:
    uttlesford_totals = {"muntjac": 2403.0, "roe": 1540.0, "fallow": 2157.0}

    base_params = {
        "muntjac": {
            "survival": np.array([0.85, 0.85, 0.90, 0.90, 0.95, 0.95, 0.90, 0.90, 0.75, 0.75], dtype=float),
            "fertility": np.array([0.05, 0.60, 1.50, 1.50, 1.50], dtype=float),
            "male_birth_fraction": 0.50,
            "K": 10774.3,
            "beta_f": 1.25,
            "beta_s0": 1.52,
            "beta_s_adult": 0.15 * 1.52,
        },
        "fallow": {
            "survival": np.array([0.90, 0.90, 0.95, 0.95, 0.97, 0.97, 0.94, 0.94, 0.80, 0.80], dtype=float),
            "fertility": np.array([0.00, 0.50, 0.85, 0.85, 0.85], dtype=float),
            "male_birth_fraction": 0.50,
            "K": 10992.2,
            "beta_f": 1.00,
            "beta_s0": 1.14,
            "beta_s_adult": 0.15 * 1.14,
        },
        "roe": {
            "survival": np.array([0.85, 0.85, 0.90, 0.90, 0.92, 0.92, 0.80, 0.80, 0.60, 0.60], dtype=float),
            "fertility": np.array([0.00, 0.50, 1.40, 1.40, 1.40], dtype=float),
            "male_birth_fraction": 0.50,
            "K": 14235.0,
            "beta_f": 1.125,
            "beta_s0": 1.33,
            "beta_s_adult": 0.15 * 1.33,
        },
    }

    base_alpha = np.array(
        [
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ],
        dtype=float,
    )

    base_gamma = {"muntjac": 0.05, "roe": 0.08, "fallow": 0.03}
    default_targets_total = {"muntjac": 1550.0, "roe": 1000.0, "fallow": 1250.0}

    weight_scenarios = build_policy_scenarios()

    score_kwargs = {
        "w_cost": 3.0,
        "w_steady": 20.0,
        "w_time": 1.0,
        "w_cull": 0.1,
        "stable_tol_rel": 0.005,
        "stable_window": 3,
        "late_window": 3,
    }

    biology_fixed = {"params": base_params, "alpha": base_alpha, "gamma_mig": base_gamma}

    age_fracs_default = (0.20, 0.25, 0.35, 0.15, 0.05)
    male_frac_default = 0.50

    score_refs = {"steady_ref": 1.0}


    return dict(
        uttlesford_totals=uttlesford_totals,
        default_targets_total=default_targets_total,
        base_params=base_params,
        base_alpha=base_alpha,
        base_gamma=base_gamma,
        biology_fixed=biology_fixed,
        weight_scenarios=weight_scenarios,
        score_kwargs=score_kwargs,
        score_refs=score_refs,  # <-- add this line
        age_fracs_default=age_fracs_default,
        male_frac_default=male_frac_default,
        ANNUAL_CULL_LIMITS=ANNUAL_CULL_LIMITS,
        ANNUAL_BUDGET_TOTAL=ANNUAL_BUDGET_TOTAL,
        COST_PARAMS=COST_PARAMS,
    )

