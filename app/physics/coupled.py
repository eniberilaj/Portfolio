"""
Coupled multiphysics engine.
============================================================================
Phase 1 — unified ReactorState + step(dt) time integrator.
Phase 2 — two-way TH <-> neutronics coupling via Picard (fixed-point) iteration
          driving a 1-D AXIAL two-group diffusion eigenvalue solve (matrix-free
          variable-coefficient CG) against a 1-D axial enthalpy model with
          Doppler (fuel-temp) and moderator-density feedback.  Produces a
          bottom-peaked axial flux as the upper core heats and voids.
Phase 3 — scenario macros (SCRAM, LOFA, asymmetric rod drop) as state mutations
          advanced by the unified engine.

Zero-dependency: numpy + stdlib only.  Matrix-free, vectorised array slicing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np

from app.numerics.linalg import jcg_solve
from app.physics.diffusion import CoreParams, solve_core

# ── Axial geometry / plant (4-loop PWR class) ──────────────────────────────
H_CM      = 366.0        # active fuel height (cm)
T_IN_NOM  = 286.0        # core inlet (°C)
DT_NOM    = 36.0         # nominal full-power coolant rise (°C)
RHO0      = 0.72         # reference coolant density (g/cc)
TF0       = 600.0        # reference fuel temperature (°C, Doppler datum)

# Axial two-group base XS (cm^-1, D in cm) — consistent with the 2-D library.
_AX_BASE = dict(D1=1.268, D2=0.370, sig_a1=0.00805, sig_a2=0.0820,
                sig_s12=0.01760, nu_sig_f1=0.00485, nu_sig_f2=0.10060)

# Feedback coefficients (calibrated for a visibly bottom-peaked shape, stable).
A_DOPPLER  = 2.6e-3      # resonance capture vs sqrt(T_fuel)   (fast absorption)
A_MOD_SCAT = 1.00        # down-scatter scales ~linearly with density
A_MOD_LEAK = 1.00        # fast leakage rises as density drops
ROD_WORTH  = 0.045       # per-node fast absorption when rodded (cm^-1 equiv.)


# ════════════════════════════════════════════════════════════════════════
#  Phase 1 — unified state
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ReactorState:
    power: float = 1.0                 # normalised core power (P/P0)
    rod_insertion: float = 0.22        # 0..1, banks from top
    enrichment: float = 3.2
    inlet_temp: float = T_IN_NOM
    mdot_frac: float = 1.0             # coolant mass-flow fraction
    n_axial: int = 40
    # filled by the coupled solve:
    axial_power: np.ndarray = field(default_factory=lambda: np.ones(40))
    axial_tcool: np.ndarray = field(default_factory=lambda: np.full(40, T_IN_NOM))
    axial_tfuel: np.ndarray = field(default_factory=lambda: np.full(40, TF0))
    axial_rho:   np.ndarray = field(default_factory=lambda: np.full(40, RHO0))
    flux_thermal: np.ndarray = field(default_factory=lambda: np.ones(40))
    keff: float = 1.0
    # point-kinetics state (for the transient engine)
    n_kin: float = 1.0
    precursors: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def mutate(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


# ════════════════════════════════════════════════════════════════════════
#  Axial neutronics — 1-D, two-group, variable-coefficient, matrix-free CG
# ════════════════════════════════════════════════════════════════════════
def _axial_operator(D: np.ndarray, Sr: np.ndarray, dz: float):
    """Matrix-free -(d/dz)(D dφ/dz) + Sr·φ with zero-flux ends. SPD → CG."""
    N = len(D)
    Dp = np.empty(N); Dm = np.empty(N)
    Dp[:-1] = 0.5 * (D[:-1] + D[1:]); Dp[-1] = D[-1]
    Dm[1:]  = 0.5 * (D[1:] + D[:-1]); Dm[0]  = D[0]
    cp = Dp / dz**2
    cm = Dm / dz**2
    diag = cp + cm + Sr

    def matvec(x):
        y = diag * x
        y[1:]  -= cm[1:]  * x[:-1]      # lower neighbour (ghost=0 at node 0)
        y[:-1] -= cp[:-1] * x[1:]       # upper neighbour (ghost=0 at node N-1)
        return y

    return diag, matvec


def _solve_axial(xsn: dict, dz: float):
    """1-D two-group eigenvalue solve: power iteration + Jacobi-CG inner solves."""
    Sr1 = xsn["sig_a1"] + xsn["sig_s12"]
    Sr2 = xsn["sig_a2"]
    diag1, mv1 = _axial_operator(xsn["D1"], Sr1, dz)
    diag2, mv2 = _axial_operator(xsn["D2"], Sr2, dz)

    nu1, nu2, s12 = xsn["nu_sig_f1"], xsn["nu_sig_f2"], xsn["sig_s12"]
    N = len(diag1)
    phi1 = np.ones(N); phi2 = np.ones(N); k = 1.0
    inner = 0
    for it in range(90):
        src = (nu1 * phi1 + nu2 * phi2) / k
        phi1n, i1, _ = jcg_solve(diag1, mv1, src, tol=1e-7, max_iter=200)
        phi1n = np.maximum(phi1n, 0.0)
        phi2n, i2, _ = jcg_solve(diag2, mv2, s12 * phi1n, tol=1e-7, max_iter=200)
        phi2n = np.maximum(phi2n, 0.0)
        inner += i1 + i2
        Fn = float((nu1 * phi1n + nu2 * phi2n).sum())
        Fo = float((nu1 * phi1  + nu2 * phi2 ).sum())
        kn = k * Fn / max(Fo, 1e-30)
        norm = max(float(phi2n.max()), 1e-30)
        phi1n /= norm; phi2n /= norm
        dk = abs(kn - k)
        phi1, phi2, k = phi1n, phi2n, kn
        if dk < 1e-6 and it > 3:
            break
    power = nu1 * phi1 + nu2 * phi2
    return phi1, phi2, float(k), power, inner


# ── Axial thermal-hydraulics (enthalpy rise + density + fuel temp) ─────────
def _axial_th(power_shape, mdot_frac, power_level, inlet_temp):
    """Given a normalised axial power shape, return coolant T, density, fuel T."""
    N = len(power_shape)
    pf = np.clip(power_shape, 0.0, None)
    pf = pf / max(pf.sum(), 1e-30)                  # power fraction per node
    dT_total = DT_NOM * power_level / max(mdot_frac, 0.05)
    # coolant heats from the bottom (node 0) upward — cumulative enthalpy rise
    t_cool = inlet_temp + dT_total * np.cumsum(pf)
    # density: linearised drop with temperature, steep near saturation (voiding)
    over = np.clip((t_cool - 310.0) / 35.0, 0.0, 1.0)
    rho  = RHO0 * (1.0 - 0.18 * np.clip((t_cool - inlet_temp) / 60.0, 0, 1)
                        - 0.30 * over**2)
    rho  = np.clip(rho, 0.30, RHO0)
    # fuel temperature: coolant + conduction rise proportional to local power
    q_node = pf * power_level
    t_fuel = t_cool + 1.9e4 * q_node / N
    return t_cool, rho, t_fuel


def _axial_feedback(t_fuel, rho, rod_insertion, enrichment, N):
    """Per-node macroscopic XS with Doppler + moderator-density + rod feedback."""
    xsn = {k: np.full(N, v, dtype=float) for k, v in _AX_BASE.items()}
    # enrichment (uniform)
    enr_fac = max(0.5, 1.0 + (enrichment - 3.2) / 3.2 * 0.35)
    xsn["nu_sig_f1"] *= enr_fac
    xsn["nu_sig_f2"] *= enr_fac
    # Doppler broadening of resonance capture (fast absorption ↑ with sqrt(Tf))
    xsn["sig_a1"] *= (1.0 + A_DOPPLER * (np.sqrt(np.maximum(t_fuel, 1.0)) - np.sqrt(TF0)))
    # moderator density: less down-scatter + more fast leakage where less dense
    rr = np.clip(rho / RHO0, 0.30, 1.05)
    xsn["sig_s12"] *= np.maximum(0.30, A_MOD_SCAT * rr)
    xsn["D1"]      /= np.maximum(0.40, rr)
    xsn["sig_a2"]  *= (0.85 + 0.15 * rr)          # slightly less thermal capture when voided
    # control rods insert from the TOP — top fraction gets absorber
    rod_nodes = int(round(np.clip(rod_insertion, 0, 1) * N))
    if rod_nodes > 0:
        xsn["sig_a1"][N - rod_nodes:] += ROD_WORTH
    return xsn


# ════════════════════════════════════════════════════════════════════════
#  Phase 2 — Picard fixed-point TH <-> neutronics coupling
# ════════════════════════════════════════════════════════════════════════
def solve_coupled(power_level=1.0, mdot_frac=1.0, inlet_temp=T_IN_NOM,
                  rod_insertion=0.22, enrichment=3.2, n_axial=40,
                  max_picard=40, relax=0.5):
    t0 = time.perf_counter()
    N  = int(np.clip(n_axial, 16, 80))
    dz = H_CM / N

    power = np.ones(N)                       # initial flat fission shape
    hist_k, hist_tmax, hist_ao = [], [], []
    inner_total = 0
    p_it = 0
    for p_it in range(max_picard):
        t_cool, rho, t_fuel = _axial_th(power, mdot_frac, power_level, inlet_temp)
        xsn = _axial_feedback(t_fuel, rho, rod_insertion, enrichment, N)
        phi1, phi2, k, power_new, inner = _solve_axial(xsn, dz)
        inner_total += inner
        pn = power_new / max(power_new.mean(), 1e-30)
        power = relax * pn + (1.0 - relax) * power

        half = N // 2
        Pb = float(power[:half].sum()); Pt = float(power[half:].sum())
        ao = (Pt - Pb) / max(Pt + Pb, 1e-30)        # axial offset (<0 ⇒ bottom-peaked)
        hist_k.append(k); hist_tmax.append(float(t_fuel.max())); hist_ao.append(ao)

        if (p_it > 2 and abs(hist_k[-1] - hist_k[-2]) < 1e-6
                and abs(hist_tmax[-1] - hist_tmax[-2]) < 0.05):
            break

    z = np.linspace(0, H_CM / 100.0, N)             # metres
    phi_th = phi2 / max(phi2.max(), 1e-30)
    return {
        "z_m":            np.round(z, 3).tolist(),
        "axial_power":    np.round(power / power.max(), 4).tolist(),
        "flux_thermal":   np.round(phi_th, 4).tolist(),
        "flux_fast":      np.round(phi1 / max(phi1.max(), 1e-30), 4).tolist(),
        "t_coolant":      np.round(t_cool, 2).tolist(),
        "t_fuel":         np.round(t_fuel, 1).tolist(),
        "density":        np.round(rho, 4).tolist(),
        "keff":           round(k, 5),
        "reactivity_pcm": round((k - 1.0) / k * 1e5, 0),
        "axial_offset":   round(hist_ao[-1], 4),
        "peak_node":      int(np.argmax(power)),
        "peak_fraction":  round(float(np.argmax(power)) / (N - 1), 3),
        "t_fuel_max":     round(float(t_fuel.max()), 0),
        "t_out":          round(float(t_cool[-1]), 1),
        "min_density":    round(float(rho.min()), 4),
        "convergence": {
            "keff":     [round(x, 6) for x in hist_k],
            "t_fuel":   [round(x, 1) for x in hist_tmax],
            "axial_offset": [round(x, 4) for x in hist_ao],
            "picard_iters": p_it + 1,
            "inner_iters":  inner_total,
        },
        "time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


# ════════════════════════════════════════════════════════════════════════
#  Phase 1 — unified step(dt): point kinetics + lumped feedback
# ════════════════════════════════════════════════════════════════════════
BETA_I   = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA_I = np.array([0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100])
BETA     = float(BETA_I.sum())
LAMBDA_P = 5.0e-5
A_FUEL   = -2.8e-5      # Doppler Δk/k per °C  (≈ −2.8 pcm/°C)
A_COOL   = -22.0e-5 / 5 # moderator Δk/k per °C of coolant rise (≈ −4.4 pcm/°C)


def _decay_heat(frac_since_trip_s):
    """ANS-style decay-heat fraction (~6.6% prompt, decaying)."""
    s = max(frac_since_trip_s, 0.1)
    return 0.066 * s ** -0.2


def simulate_scenario(scenario="scram", duration=40.0, n_steps=4000):
    """Phase 3 transient macros advanced by point kinetics + lumped TH."""
    dt = duration / n_steps
    Tf0, Tc0 = TF0, T_IN_NOM + DT_NOM / 2.0
    n = 1.0
    C = (BETA_I / (LAMBDA_I * LAMBDA_P))
    Tf, Tc = Tf0, Tc0
    mdot = 1.0
    Cf, Cc, UA = 2.5e5, 1.2e5, 3411e3 / 320.0
    P0 = 3411e3

    t_arr, n_arr, tf_arr, tc_arr, rho_arr, rho_pcm = [], [], [], [], [], []
    trip_t = 2.0

    for i in range(n_steps + 1):
        t = i * dt
        # ── external + feedback reactivity ──
        rod_rho = 0.0
        if scenario == "scram":
            rod_rho = -0.085 if t >= trip_t else 0.0          # deep insertion
        elif scenario == "rod_eject":
            rod_rho = 0.0035 if t >= trip_t else 0.0          # (illustrative +)
        if scenario == "lofa":
            mdot = float(np.exp(-(max(0.0, t - trip_t)) / 5.0)) if t >= trip_t else 1.0
            mdot = max(mdot, 0.06)

        fb = A_FUEL * (Tf - Tf0) + A_COOL * (Tc - Tc0)
        rho = rod_rho + fb

        # ── point kinetics (implicit-friendly small dt RK2) ──
        def deriv(nv, Cv):
            dn = ((rho - BETA) / LAMBDA_P) * nv + float(np.dot(LAMBDA_I, Cv))
            dC = (BETA_I / LAMBDA_P) * nv - LAMBDA_I * Cv
            return dn, dC
        dn1, dC1 = deriv(n, C)
        dn2, dC2 = deriv(max(n + dt * dn1, 0.0), C + dt * dC1)
        n = max(n + dt * 0.5 * (dn1 + dn2), 0.0)
        C = C + dt * 0.5 * (dC1 + dC2)

        # decay heat floor after a trip (fission gone, heat remains)
        if scenario == "scram" and t >= trip_t:
            n = max(n, _decay_heat(t - trip_t))

        # ── lumped two-node thermal with (possibly decaying) flow ──
        P = n * P0
        dTf = (P - UA * (Tf - Tc)) / Cf
        dTc = (UA * (Tf - Tc) - 2.0 * mdot * (P0 / (2.0 * DT_NOM)) * (Tc - T_IN_NOM)) / Cc
        Tf += dt * dTf
        Tc += dt * dTc

        if i % max(1, n_steps // 500) == 0:
            t_arr.append(round(t, 4)); n_arr.append(round(n, 6))
            tf_arr.append(round(Tf, 1)); tc_arr.append(round(Tc, 1))
            rho_c = RHO0 * (1 - 0.18 * np.clip((Tc - T_IN_NOM) / 60, 0, 1)
                                - 0.30 * np.clip((Tc - 310) / 35, 0, 1) ** 2)
            rho_arr.append(round(float(np.clip(rho_c, 0.3, RHO0)), 4))
            rho_pcm.append(round(rho * 1e5, 1))

    return {
        "scenario": scenario, "trip_time_s": trip_t,
        "t": t_arr, "power": n_arr, "t_fuel": tf_arr, "t_coolant": tc_arr,
        "density": rho_arr, "reactivity_pcm": rho_pcm,
        "peak_power": round(max(n_arr), 4),
        "final_power": round(n_arr[-1], 5),
        "peak_tfuel": round(max(tf_arr), 0),
    }


# ── Phase 3 — asymmetric rod drop (2-D radial flux tilt + Fq spike) ────────
def asymmetric_rod_drop(enrichment=3.2):
    """Drop one RCCA bank fully on one side; show flux tilt + peaking spike."""
    base = solve_core(CoreParams(rod_insertion=0.22, enrichment=enrichment, refine=3))
    # Re-solve with a single dropped bank by exploiting the 2-D solver's per-cell
    # rod map: emulate asymmetry with a higher-insertion proxy on one side.
    dropped = _solve_2d_asymmetric(enrichment)
    return {
        "before": {"power_map": base["power_map"], "peaking": base["peaking_factor"],
                   "keff": base["keff"], "mask": base["active_mask"]},
        "after":  dropped,
    }


def _solve_2d_asymmetric(enrichment):
    """One bank fully inserted, others nominal → asymmetric absorber pattern."""
    from app.physics.diffusion import _feedback, _make_stencil, _CORE_RADIUS
    p = CoreParams(rod_insertion=0.22, enrichment=enrichment, refine=3)
    xs = _feedback(p)
    N = 45
    cx, cy = N // 2, N // 2
    R = N * 0.5
    ii, jj = np.mgrid[0:N, 0:N]
    active = (np.sqrt((ii - cx)**2 + (jj - cy)**2) <= R * 0.98).astype(float)
    rod_map = np.zeros((N, N))
    banks = list(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    for bi, ang in enumerate(banks):
        depth = 0.95 if bi == 0 else 0.22                  # bank 0 fully dropped
        ri = int(cy + R * 0.55 * np.sin(ang)); rj = int(cx + R * 0.55 * np.cos(ang))
        rr = max(2, int(R * 0.12))
        di = np.arange(-rr, rr + 1)
        DI, DJ = np.meshgrid(di, di, indexing="ij")
        m = DI**2 + DJ**2 <= rr**2
        rows = np.clip(ri + DI[m], 0, N - 1); cols = np.clip(rj + DJ[m], 0, N - 1)
        np.add.at(rod_map, (rows, cols), depth)
    diag1, diag2, A1, A2 = _make_stencil(N, xs, rod_map, np.zeros((N, N)), np.zeros((N, N)))
    phi1 = active.ravel() * .5 + .1; phi2 = active.ravel() * .7 + .1; k = 1.0
    nu1, nu2, s12 = xs["nu_sig_f1"], xs["nu_sig_f2"], xs["sig_s12"]
    for _ in range(60):
        src = (nu1 * phi1 + nu2 * phi2) / k
        p1, _, _ = jcg_solve(diag1, A1, src * active.ravel(), tol=1e-5, max_iter=300)
        p1 = np.maximum(p1, 0) * active.ravel()
        p2, _, _ = jcg_solve(diag2, A2, s12 * p1, tol=1e-5, max_iter=300)
        p2 = np.maximum(p2, 0) * active.ravel()
        Fn = float((nu1 * p1 + nu2 * p2).sum()); Fo = float((nu1 * phi1 + nu2 * phi2).sum())
        kn = k * Fn / max(Fo, 1e-30)
        nrm = max(float(p2.max()), 1e-30); p1 /= nrm; p2 /= nrm
        dk = abs(kn - k); phi1, phi2, k = p1, p2, kn
        if dk < 1e-5: break
    fis = (nu1 * phi1 + nu2 * phi2).reshape(N, N) * active
    fmean = float(fis[active > 0].mean()) if active.any() else 1.0
    pk = fis / max(fmean, 1e-30)
    return {"power_map": np.round(pk, 3).tolist(),
            "peaking": round(float(pk.max()), 4),
            "keff": round(k, 5),
            "rod_map": np.round(rod_map * active, 3).tolist(),
            "mask": (active > 0).tolist()}