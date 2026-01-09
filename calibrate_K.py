from __future__ import annotations

import copy
import numpy as np

from model import (
    SPECIES,
    make_initial_state_from_total,
    simulate,
    build_defaults,
)

def _totals_over_time(df_classes):
    """Return dict sp -> np.array totals length years+1."""
    out = {}
    for sp in SPECIES:
        cols = [c for c in df_classes.columns if c.startswith(f"{sp}_")]
        out[sp] = df_classes[cols].sum(axis=1).to_numpy(dtype=float)
    return out

def _drift_objective(totals_by_sp, N0_totals, years, burn_in=2):
    """
    Objective: mean squared relative deviation from initial totals,
    averaged over years burn_in..years (inclusive).
    """
    eps = 1e-9
    J = 0.0
    n = 0
    for sp in SPECIES:
        x = totals_by_sp[sp]
        x0 = float(N0_totals[sp])
        for t in range(burn_in, years + 1):
            rel = (x[t] - x0) / (x0 + eps)
            J += rel * rel
            n += 1
    return float(J / max(n, 1))

def _end_rel_error(totals_by_sp, N0_totals, years):
    """Signed relative error at final year: (N_T - N0)/N0."""
    eps = 1e-9
    err = {}
    for sp in SPECIES:
        xT = float(totals_by_sp[sp][years])
        x0 = float(N0_totals[sp])
        err[sp] = (xT - x0) / (x0 + eps)
    return err

def run_no_harvest(params, alpha, gamma_mig, N0, N_out, years):
    """
    Run a simulation with no harvest by setting targets absurdly high.
    IMPORTANT: also set cull_intensity=0.0 so the "surplus_remaining" logic cannot
    generate any desired harvest (your simulate() now uses intensity-based desired_override).
    """
    targets_total = {sp: 1e18 for sp in SPECIES}
    dummy_weights = np.zeros(10, dtype=float)

    df_classes, _df_cull = simulate(
        N0=N0,
        params=params,
        alpha=alpha,
        targets_total=targets_total,
        N_out=N_out,
        gamma_mig=gamma_mig,
        cull_weights=dummy_weights,
        max_cull_frac_per_class=0.0,
        cull_intensity=0.0,          # <-- CRITICAL given your current model.py
        annual_cull_limits=None,
        annual_budget_total=None,
        cost_params=None,
        years=years,
    )
    return df_classes

def calibrate_Ks_to_stationary_totals(
    base_params: dict,
    base_alpha: np.ndarray,
    base_gamma: dict,
    N0_totals: dict[str, float],
    *,
    years: int = 20,
    outer_iters: int = 6,
    bisection_iters: int = 18,
    K_scale_bounds=(0.3, 3.5),
    burn_in: int = 2,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Calibrate K for each species so that with no harvest, totals stay close to initial totals.
    Uses coordinate descent: update each K using bisection on a scale factor.

    Returns: dict sp -> calibrated K
    """
    age_fracs_default = (0.20, 0.25, 0.35, 0.15, 0.05)
    male_frac_default = 0.50
    N0 = {sp: make_initial_state_from_total(N0_totals[sp], age_fracs_default, male_frac_default) for sp in SPECIES}

    # Pin migration target to initial totals during calibration (avoid migration-induced drift)
    N_out = {sp: float(N0_totals[sp]) for sp in SPECIES}

    params = copy.deepcopy(base_params)

    def eval_current():
        df = run_no_harvest(params, base_alpha, base_gamma, N0, N_out, years)
        totals_by_sp = _totals_over_time(df)
        J = _drift_objective(totals_by_sp, N0_totals, years, burn_in=burn_in)
        err = _end_rel_error(totals_by_sp, N0_totals, years)
        return J, err

    J0, err0 = eval_current()
    if verbose:
        print(f"Initial no-harvest drift objective J={J0:.6g}, end-year rel errors={err0}")

    for it in range(outer_iters):
        if verbose:
            print(f"\n=== Outer iteration {it+1}/{outer_iters} ===")

        for sp in SPECIES:
            K0 = float(params[sp]["K"])

            lo_sc, hi_sc = K_scale_bounds
            lo = np.log(K0 * lo_sc)
            hi = np.log(K0 * hi_sc)

            def signed_err_at_logK(logK):
                params[sp]["K"] = float(np.exp(logK))
                df = run_no_harvest(params, base_alpha, base_gamma, N0, N_out, years)
                totals_by_sp = _totals_over_time(df)
                err = _end_rel_error(totals_by_sp, N0_totals, years)[sp]
                J = _drift_objective(totals_by_sp, N0_totals, years, burn_in=burn_in)
                return float(err), float(J)

            e_lo, J_lo = signed_err_at_logK(lo)
            e_hi, J_hi = signed_err_at_logK(hi)

            # If both ends have same sign, pick the better objective endpoint.
            if e_lo == 0.0:
                best_logK = lo
            elif e_hi == 0.0:
                best_logK = hi
            elif (e_lo > 0 and e_hi > 0) or (e_lo < 0 and e_hi < 0):
                best_logK = lo if J_lo <= J_hi else hi
                params[sp]["K"] = float(np.exp(best_logK))
                if verbose:
                    print(
                        f"{sp}: could not bracket sign (e_lo={e_lo:.3g}, e_hi={e_hi:.3g}); "
                        f"choosing endpoint K={params[sp]['K']:.6g}"
                    )
                continue
            else:
                best_logK = None
                best_J = None

                for _ in range(bisection_iters):
                    mid = 0.5 * (lo + hi)
                    e_mid, J_mid = signed_err_at_logK(mid)

                    if best_logK is None or J_mid < best_J:
                        best_logK = mid
                        best_J = J_mid

                    # root-bracketing step
                    if (e_lo <= 0 and e_mid <= 0) or (e_lo >= 0 and e_mid >= 0):
                        lo = mid
                        e_lo = e_mid
                    else:
                        hi = mid
                        e_hi = e_mid

                params[sp]["K"] = float(np.exp(best_logK))

            if verbose:
                J_now, err_now = eval_current()
                print(f"{sp}: set K={params[sp]['K']:.6g} | J={J_now:.6g} | end rel err={err_now[sp]:.4g}")

        J_now, err_now = eval_current()
        if verbose:
            print(f"After outer iter {it+1}: J={J_now:.6g}, end-year rel errors={err_now}")

    return {sp: float(params[sp]["K"]) for sp in SPECIES}


if __name__ == "__main__":
    defaults = build_defaults()

    base_params = defaults["base_params"]
    base_alpha = defaults["base_alpha"]
    base_gamma = defaults["base_gamma"]

    # Use Uttlesford totals as the calibration reference
    N0_totals = defaults["uttlesford_totals"]

    Ks = calibrate_Ks_to_stationary_totals(
        base_params=base_params,
        base_alpha=base_alpha,
        base_gamma=base_gamma,
        N0_totals=N0_totals,
        years=20,
        outer_iters=6,
        verbose=True,
    )

    print("\nCalibrated K defaults:")
    for sp, K in Ks.items():
        print(f"  {sp}: K = {K:.6g}")
