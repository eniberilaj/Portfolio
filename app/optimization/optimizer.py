"""
Control-rod insertion optimiser.
Objective: minimise the power peaking factor while staying near criticality.

Honest note: with a single rod-bank knob this is effectively a 1-D problem, so what
runs below is a shrinking-step coordinate search (try ±step, keep the better point,
halve the step) rather than a full Nelder–Mead simplex. Same spirit — derivative-free
local search — but simpler, and plenty for one variable. The objective and the plumbing
are written so it generalises to a real multi-bank simplex later.
"""
from __future__ import annotations
import time
import numpy as np
from app.physics.diffusion import CoreParams, solve_core


def _objective(x: np.ndarray, base: CoreParams) -> float:
    """What we're minimising. Three competing terms, weighted by hand:
      • peaking factor      — the thing we actually want low (flatter flux)
      • (keff − 1)²         — heavily penalise drifting off criticality
      • std(x)              — mild nudge toward smooth, non-jagged rod patterns
    Each evaluation costs a full core solve, which is why max_iter stays small."""
    p = CoreParams(
        rod_insertion   = float(np.clip(x[0], 0.0, 0.95)),
        enrichment      = base.enrichment,
        coolant_density = base.coolant_density,
        moderator_temp  = base.moderator_temp,
        power_demand    = base.power_demand,
        refine          = base.refine,
        burnup_map      = base.burnup_map,
    )
    res  = solve_core(p)
    pk   = res["peaking_factor"]
    keff = res["keff"]
    rough = float(np.std(x))
    return pk + 35.0 * (keff - 1.0)**2 + 0.3 * rough


def optimize(base: CoreParams, max_iter: int = 30) -> dict:
    t0   = time.perf_counter()
    x0   = np.array([base.rod_insertion])   # start from the current rod position

    # Baseline solve, so we can report "before vs after".
    res0 = solve_core(base)
    pk0  = res0["peaking_factor"]

    # Shrinking-step coordinate search: at each iteration probe both directions,
    # move to whichever improves the objective, then shrink the step (×0.85) so we
    # home in. Clipped to [0.01, 0.94] to keep the rods physically in range.
    x_best = x0.copy()
    f_best = _objective(x0, base)
    history = [f_best]
    n_evals = 1

    step = 0.10
    for it in range(max_iter):
        for sign in (+1, -1):
            xc = np.clip(x_best + sign * step, 0.01, 0.94)
            fc = _objective(xc, base)
            n_evals += 1
            if fc < f_best:
                f_best = fc
                x_best = xc
        history.append(f_best)
        step *= 0.85           # tighten the search around the best point so far

    # Final solve with optimal params
    p_opt = CoreParams(
        rod_insertion   = float(x_best[0]),
        enrichment      = base.enrichment,
        coolant_density = base.coolant_density,
        moderator_temp  = base.moderator_temp,
        power_demand    = base.power_demand,
        refine          = base.refine,
        burnup_map      = base.burnup_map,
    )
    res1 = solve_core(p_opt)

    return {
        "before": {
            "peaking":   pk0,
            "keff":      res0["keff"],
            "power_map": res0["power_map"],
            "mask":      res0["active_mask"],
        },
        "after": {
            "peaking":   res1["peaking_factor"],
            "keff":      res1["keff"],
            "power_map": res1["power_map"],
            "mask":      res1["active_mask"],
        },
        "x_opt":              x_best.tolist(),
        "n_evals":            n_evals,
        "history":            [round(h, 5) for h in history],
        "optimization_score": round((pk0 - res1["peaking_factor"]) / max(pk0, 1e-6) * 100, 2),
        "wall_time_s":        round(time.perf_counter() - t0, 2),
    }
