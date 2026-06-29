"""
Six-group point-kinetics solver.
=================================
This is the "how fast does power change in time" half of the reactor, as opposed
to diffusion.py which is the "where is the power in space" half. Point kinetics
collapses the whole core to a single number n(t) (power relative to nominal) plus
six delayed-neutron precursor populations Cᵢ — the tiny fraction of neutrons that
arrive seconds-to-minutes late and are the entire reason a reactor is controllable
by hand rather than detonating.

Point kinetics equations (n = P/P₀ normalised):

    dn/dt  = [(ρ(t) − β)/Λ] · n(t) + Σᵢ λᵢ Cᵢ(t)
    dCᵢ/dt = (βᵢ/Λ)         · n(t) − λᵢ · Cᵢ(t)   i = 1…6

The thing to notice: if reactivity ρ ever reaches β the prompt term blows up
(prompt-critical) and the delayed neutrons can't save you — that's why β matters
so much and why reactivity is measured in dollars (ρ/β).

DNP constants: IAEA benchmark (U-235 thermal fission).
Λ = 50 μs  (PWR prompt neutron generation time).
Integrator: fixed-step RK4 (sub-ms resolution), thinned to ≤ 600 output points.
"""
from __future__ import annotations
import numpy as np

# ── 6-group delayed neutron parameters (U-235, thermal fission) ──────────
# βᵢ = fraction of fission neutrons from precursor group i; λᵢ = its decay constant
# [1/s]. Group 1 is the slow, long-lived one; group 6 the fast, short-lived one.
BETA_I   = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA_I = np.array([0.0124,   0.0305,   0.1110,   0.3010,   1.1400,   3.0100])
BETA     = float(BETA_I.sum())   # total delayed fraction 0.006502 (650.2 pcm) = "$1"
LAMBDA_P = 5.0e-5                # prompt neutron generation time Λ (s)
GROUPS   = 6


def _ic(n0: float = 1.0) -> np.ndarray:
    """Equilibrium initial conditions.

    Start the precursors at their steady-state values (set dCᵢ/dt = 0 and solve),
    so the reactor sits perfectly still until reactivity is inserted — no spurious
    transient just from a bad starting point.
    """
    C0 = (BETA_I / (LAMBDA_I * LAMBDA_P)) * n0
    y0 = np.empty(1 + GROUPS)
    y0[0]  = n0
    y0[1:] = C0
    return y0


def _rho(t: float, cfg: dict) -> float:
    """Reactivity ρ(t) in Δk/k — the forcing function that drives everything.

    Four shapes the UI can pick. Everything stays at 0 until t_insert so you can
    see the steady baseline first. (rho_pcm is in pcm = 1e-5 Δk/k, hence the 1e-5.)
    """
    rtype = cfg["type"]
    t0    = cfg["t_insert"]
    r0    = cfg["rho_pcm"] * 1e-5

    if rtype == "step":                      # instant jump and hold
        return r0 if t >= t0 else 0.0
    elif rtype == "ramp":                    # linear ramp, clamped at the target
        rate = cfg.get("ramp_rate", 10.0) * 1e-5
        if t < t0: return 0.0
        return float(np.clip(rate * (t - t0), min(0.0, r0), max(0.0, r0)))
    elif rtype == "sine":                    # oscillation, to probe the frequency response
        if t < t0: return 0.0
        return r0 * float(np.sin(2.0 * np.pi * cfg.get("freq", 0.5) * (t - t0)))
    elif rtype == "scram":                   # emergency shutdown: slam in −$15 of rods
        return (-15.0 * BETA) if t >= t0 else 0.0
    return 0.0


def _rhs(t: float, y: np.ndarray, cfg: dict) -> np.ndarray:
    """Right-hand side of the ODE system: y = [n, C₁..C₆] -> dy/dt."""
    n, C = float(y[0]), y[1:]
    rh   = _rho(t, cfg)
    dy   = np.empty(1 + GROUPS)
    # power: prompt term (ρ−β)/Λ·n  +  delayed neutrons trickling back from precursors
    dy[0]  = ((rh - BETA) / LAMBDA_P) * n + float(np.dot(LAMBDA_I, C))
    # each precursor group: produced ∝ power, decays at its own rate λᵢ
    dy[1:] = (BETA_I / LAMBDA_P) * n - LAMBDA_I * C
    return dy


def _integrate(cfg: dict, duration: float, n_steps: int):
    """Plain fixed-step RK4. The equations are stiff (Λ is microseconds while the
    slow precursor is ~80 s), so the step has to stay small — hence n_steps≈2000
    and the output thinning later rather than taking big steps."""
    dt    = duration / n_steps
    t_arr = np.linspace(0.0, duration, n_steps + 1)
    Y     = np.zeros((n_steps + 1, 1 + GROUPS))
    Y[0]  = _ic(1.0)

    for i in range(n_steps):
        t, y = t_arr[i], Y[i]
        k1 = _rhs(t,        y,               cfg)
        k2 = _rhs(t+dt/2,   y+(dt/2)*k1,     cfg)
        k3 = _rhs(t+dt/2,   y+(dt/2)*k2,     cfg)
        k4 = _rhs(t+dt,     y+dt*k3,          cfg)
        Y[i+1] = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if Y[i+1, 0] < 0.0: Y[i+1, 0] = 0.0   # power can't go negative; clamp roundoff

    return t_arr, Y


def solve(rho_type="step", rho_pcm=50.0, t_insert=1.0, duration=30.0,
          ramp_rate=10.0, freq=0.5, n_steps=2000) -> dict:
    cfg   = {"type": rho_type, "t_insert": t_insert, "rho_pcm": rho_pcm,
             "ramp_rate": ramp_rate, "freq": freq}
    t, Y  = _integrate(cfg, duration, n_steps)
    n_t   = Y[:, 0]
    C_t   = Y[:, 1:]

    # Thin the dense solution down to ≤600 points — plenty for a smooth chart,
    # and keeps the JSON payload small.
    sl        = slice(None, None, max(1, len(t) // 600))
    rho_arr   = np.array([_rho(ti, cfg) * 1e5 for ti in t[sl]])   # back to pcm for display
    del_contrib = C_t[sl].dot(LAMBDA_I)                            # total delayed source

    # A few summary numbers for the readout panel:
    peak_idx   = int(np.argmax(n_t))                              # when/where power peaked
    tail       = n_t[max(0, len(n_t) - max(1, len(n_t)//10)):]    # last 10% of the run
    stable_pwr = float(np.mean(tail))                             # settled power level

    # Reactor period: time to change power by a factor of e. Estimated from the
    # initial slope right after insertion; clamped so a near-flat response doesn't
    # report a silly ±infinity period.
    eps = max(1, n_steps // 100)
    dn  = n_t[eps] - n_t[0]
    prd = float(t[eps] / max(abs(dn) / max(n_t[0], 1e-9), 1e-9))
    prd = max(-9999.0, min(9999.0, prd))

    return {
        "t":               np.round(t[sl], 5).tolist(),
        "n":               np.round(n_t[sl], 7).tolist(),
        "precursors":      [np.round(C_t[sl, i] / max(float(C_t[0, i]), 1e-30), 6).tolist()
                            for i in range(GROUPS)],
        "rho_pcm":         np.round(rho_arr, 3).tolist(),
        "delayed_contrib": np.round(del_contrib, 8).tolist(),
        "peak_power":      round(float(n_t[peak_idx]), 6),
        "peak_time_s":     round(float(t[peak_idx]), 4),
        "stable_power":    round(stable_pwr, 6),
        "reactor_period_s": round(prd, 2),
        "rho_input_pcm":   rho_pcm,
        "rho_type":        rho_type,
        "beta_i_pcm":      np.round(BETA_I * 1e5, 3).tolist(),
        "lambda_i":        LAMBDA_I.tolist(),
        "beta_total_pcm":  round(BETA * 1e5, 2),
        "lambda_p_us":     round(LAMBDA_P * 1e6, 1),
    }
