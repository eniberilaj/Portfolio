"""
Fuel burnup and depletion model -- simplified 3-batch reload scheme.
====================================================================
As fuel sits in the core it slowly "burns up": fissile U-235 depletes and the
reactivity it provides drifts down over the ~18-month cycle. Real plants manage
this by reloading in batches — fresh assemblies in some positions, once- and
twice-burned ones in others — so the core stays critical for as long as possible.

I model that the cheap way: three radial batches with different burn rates, and a
single empirical k_eff(t) curve for the whole cycle. It's not a depletion code
(no Bateman chains, no isotopics) — just enough to show the end-of-cycle countdown
and the radial burnup pattern that batch reloading produces.

EFPD = Effective Full-Power Days (calendar days scaled by how hard the core ran).
"""
from __future__ import annotations
import numpy as np

CYCLE_LENGTH_EFPD = 540.0   # nominal cycle length (~18 months at full power)
_MESH = 45                  # core grid, matches the diffusion solver


def _assembly_positions():
    """Every grid cell inside the circular core boundary (radius 21 about centre)."""
    cx, cy = 22, 22
    R = 21.0
    pos = []
    for i in range(_MESH):
        for j in range(_MESH):
            if (i - cx)**2 + (j - cy)**2 <= R**2:
                pos.append((i, j))
    return pos


_POSITIONS = _assembly_positions()
_BATCH_RADIUS = [0.33, 0.65, 1.0]   # radial fraction cutoffs for the three batches


def _batch_map():
    """Assign each assembly to a batch by radius: inner third = batch 2 (freshest,
    burns fastest), middle ring = batch 1, outer = batch 0. This in/out shuffling is
    roughly how real reloads flatten the power and stretch the cycle."""
    cx, cy = 22, 22
    R = 21.0
    bmap = np.zeros((_MESH, _MESH), dtype=int)
    for i, j in _POSITIONS:
        r_frac = np.sqrt((i - cx)**2 + (j - cy)**2) / R
        if r_frac < _BATCH_RADIUS[0]:
            bmap[i, j] = 2
        elif r_frac < _BATCH_RADIUS[1]:
            bmap[i, j] = 1
        else:
            bmap[i, j] = 0
    return bmap


# Precompute the geometry once at import — it never changes between calls.
_BATCH_MAP = _batch_map()


def burnup_state(cycle_day: float, power_map=None) -> dict:
    """Snapshot of core depletion at a given day in the cycle."""
    cycle_day = float(np.clip(cycle_day, 0, CYCLE_LENGTH_EFPD))
    frac      = cycle_day / CYCLE_LENGTH_EFPD     # 0 at start of cycle, 1 at end
    batch_base = np.array([0.0, 0.0, 0.0])        # all batches start the *cycle* at 0 here
    batch_rate = np.array([40.0, 20.0, 10.0])     # GWd/tU accrued over a cycle, per batch
    burnup_map = np.zeros((_MESH, _MESH))
    active_map = np.zeros((_MESH, _MESH), dtype=bool)
    for i, j in _POSITIONS:
        b = _BATCH_MAP[i, j]
        active_map[i, j] = True
        # weight the local burn rate by how much power that spot actually makes
        # (high-flux assemblies deplete faster); +0.5 floor so cold edges still age.
        pf = 1.0
        if power_map is not None and power_map.shape == (_MESH, _MESH):
            pf = max(float(power_map[i, j]), 0.0) + 0.5
        burnup_map[i, j] = batch_base[b] + batch_rate[b] * frac * pf
    avg_burnup = float(burnup_map[active_map].mean())
    # Empirical k_eff letdown: starts hot (1.045) and decays to just-critical by
    # end of cycle. The ^0.65 makes the drop steeper early, flatter later — the
    # shape you get once xenon settles and only slow depletion remains.
    k0, k_eoc = 1.045, 1.001
    keff = k0 - (k0 - k_eoc) * frac ** 0.65
    # Solve that curve for the day k_eff hits 1.0 — the cycle's natural end (EOC).
    eoc_efpd = float(CYCLE_LENGTH_EFPD * ((k0 - 1.000) / (k0 - k_eoc)) ** (1.0/0.65))
    days_arr = np.linspace(0, CYCLE_LENGTH_EFPD, 120)
    keff_arr = k0 - (k0 - k_eoc) * (days_arr / CYCLE_LENGTH_EFPD) ** 0.65
    batch_progress = [frac, min(1.0, frac + 0.05), min(1.0, frac + 0.12)]
    return {
        "cycle_day":             round(cycle_day, 1),
        "burnup_map":            np.round(burnup_map, 2).tolist(),
        "avg_burnup":            round(avg_burnup, 2),
        "keff":                  round(keff, 5),
        "eoc_efpd":              round(eoc_efpd, 1),
        "days_remaining":        round(max(0.0, eoc_efpd - cycle_day), 1),
        "reactivity_margin_pcm": round((keff - 1.0) * 1e5, 0),
        "utilization_pct":       round(avg_burnup / 50.0 * 100.0, 1),
        "letdown": {
            "days": np.round(days_arr, 1).tolist(),
            "keff": np.round(keff_arr, 5).tolist(),
        },
        "batch_progress": [round(p, 3) for p in batch_progress],
    }
