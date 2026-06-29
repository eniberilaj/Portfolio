"""
Thermal-hydraulic single-channel model.
=======================================
The companion to the neutronics solver: once diffusion tells me how much power
each part of the core makes, *this* tells me how hot the fuel and coolant get.

The idea is deliberately simple. I march coolant up one representative channel,
adding the heat it picks up along the way, then stack the resistances from the
coolant film through the cladding, gas gap and fuel pellet to get the fuel
centreline temperature. That's enough to watch the safety margins (DNBR, fuel
melt) move when you tug a slider, without pretending to be a full sub-channel code.

Standard library + NumPy only.
"""
from __future__ import annotations
import numpy as np
from app.numerics.linalg import rk4

# NumPy 2.0 renamed np.trapz -> np.trapezoid. Grab whichever exists so this runs
# on both old and new installs (learned this the hard way after an upgrade broke it).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# ── Plant constants, sized to a 4-loop Westinghouse-class PWR ──
RATED_MWT = 3411.0     # rated thermal power [MW]
H         = 3.66       # active fuel height [m]
HE        = 4.10       # extrapolated height (a bit taller than H so the cosine
                       #   power shape doesn't go to zero right at the fuel ends)
MDOT      = 17000.0    # total core coolant mass flow [kg/s]
CP        = 5.5        # coolant heat capacity [kJ/kg·K] (hot pressurised water)
N_RODS    = 50952      # number of fuel rods in the core
T_IN_NOM  = 286.0      # nominal core inlet temperature [°C]
P_SYS     = 15.5       # primary system pressure [MPa]
T_SAT     = 344.8      # saturation temperature at P_SYS [°C] — the boiling limit
                       #   we never want the hot channel to reach


def simulate(power_pct=100.0, inlet_temp=T_IN_NOM,
             flow_pct=100.0, peaking=1.45, n_z=80):
    """Steady-state axial temperatures up a single representative channel."""
    q_total = RATED_MWT * power_pct / 100.0   # actual thermal power at this setting
    mdot    = MDOT * flow_pct / 100.0         # actual coolant flow
    z       = np.linspace(0, H, n_z)          # axial grid, bottom -> top of fuel

    # Chopped-cosine axial power shape, then renormalise so its average over the
    # height is exactly 1 — that way q_lin_avg below stays the true core average.
    shape   = np.cos(np.pi * (z - H / 2) / HE)
    shape  /= _trapz(shape, z) / H

    # Linear heat rate [kW/m]: average per rod, shaped axially, then the hot rod
    # (peaking factor) which is the one that actually limits us.
    q_lin_avg = q_total * 1e3 / (N_RODS * H)
    q_lin     = q_lin_avg * shape
    q_lin_hot = q_lin * peaking

    # Coolant temperature rise = running integral of the heat picked up so far,
    # divided by flow·Cp (trapezoidal cumulative sum up the channel).
    dT = (np.concatenate([[0],
           np.cumsum((q_lin[1:] + q_lin[:-1]) / 2 * np.diff(z))])
          * N_RODS / (mdot * CP))
    t_cool = inlet_temp + dT
    t_hot  = inlet_temp + dT * peaking * 0.85   # hot channel runs hotter; 0.85 is
                                                #   a crude enthalpy-rise fudge

    # Walk the heat outward-to-in through a stack of thermal resistances
    # [K·m/kW]: coolant film -> cladding -> gas gap -> fuel pellet. The fuel
    # centreline is the hottest point and the one that mustn't melt.
    R_film, R_clad, R_gap, R_fuel = 0.55, 0.45, 8.5, 26.5
    t_clad      = t_hot + q_lin_hot * (R_film + R_clad / 2)
    t_fuel_surf = t_hot + q_lin_hot * (R_film + R_clad + R_gap)
    t_fuel_cl   = t_fuel_surf + q_lin_hot * R_fuel

    # Rough secondary-side power conversion: a Carnot ceiling from the steam temp,
    # knocked down by 0.715 for real turbine/generator losses. Just for the MWe readout.
    t_out   = float(t_cool[-1])
    t_steam = min(t_out - 38.0, 290.0)
    eta_c   = 1 - (33 + 273.15) / (t_steam + 273.15)
    eta     = eta_c * 0.715
    mwe     = q_total * eta

    # Departure-from-nucleate-boiling ratio — how much margin before the hot spot boils.
    dnbr = _dnbr_estimate(q_lin_hot.max(), t_hot.max(), mdot)

    return {
        "z_m":                  np.round(z, 3).tolist(),
        "coolant_temp":         np.round(t_cool, 2).tolist(),
        "hot_channel_temp":     np.round(t_hot, 2).tolist(),
        "clad_surface_temp":    np.round(t_clad, 2).tolist(),
        "fuel_centerline_temp": np.round(t_fuel_cl, 1).tolist(),
        "linear_power_kw_m":    np.round(q_lin_hot, 2).tolist(),
        "lin_power_kw":         np.round(q_lin_hot, 2).tolist(),
        "clad_surface_peak":    round(float(t_clad.max()), 1),
        "fuel_cl_peak":         round(float(t_fuel_cl.max()), 0),
        "delta_t_coolant":      round(t_out - inlet_temp, 1),
        "t_inlet":              round(inlet_temp, 1),
        "t_outlet":             round(t_out, 1),
        "delta_t":              round(t_out - inlet_temp, 1),
        "t_sat_margin":         round(T_SAT - float(t_hot.max()), 1),
        "max_fuel_centerline":  round(float(t_fuel_cl.max()), 0),
        "max_clad_temp":        round(float(t_clad.max()), 1),
        "thermal_power_mw":     round(q_total, 0),
        "electric_power_mw":    round(mwe, 0),
        "efficiency_pct":       round(eta * 100, 2),
        "flow_kg_s":            round(mdot, 0),
        "dnbr":                 dnbr,
        "system_pressure_mpa":  P_SYS,
        "primary_loop": {
            "core_inlet_c":   round(inlet_temp, 1),
            "core_outlet_c":  round(t_out, 1),
            "hot_leg_c":      round(t_out - 2.0, 1),
            "cold_leg_c":     round(inlet_temp + 1.5, 1),
            "sg_secondary_c": round(t_out - 55.0, 1),
            "flow_kg_s":      round(mdot, 0),
        },
    }


def _dnbr_estimate(q_peak_kw_m, t_max, mdot):
    """A toy DNBR: critical heat flux over actual peak flux.

    Not a real correlation (W-3 etc. need local quality and a proper geometry) —
    just a monotonic stand-in that rises with flow and with subcooling margin
    (T_sat − T) so the number moves the right way when you change the sliders.
    Clamped to a sane 0.5–6 range.
    """
    q_chf = 95.0 * (mdot / MDOT) ** 0.8 * max(0.05, (T_SAT - t_max) / 25.0) ** 0.33
    return round(float(np.clip(q_chf / max(q_peak_kw_m, 1.0), 0.5, 6.0)), 2)


def transient_multichannel(perturbation="power_step", magnitude=10.0,
                           duration_s=120.0, n_snapshots=60, n_z=80):
    """
    Multi-channel axial thermal transient with animated snapshots.
    Three channels: hot (Fq=1.45), average (1.0), cold (0.65).

    Trick that keeps this cheap: I solve ONE lumped two-node ODE (fuel+coolant) for
    the *time* response, then for each snapshot I just blend between the known
    start and end steady-state axial profiles by how far that lumped response has
    travelled. So the expensive bit is a tiny ODE, not a full space-time solve.
    """
    R_film, R_clad, R_gap, R_fuel = 0.55, 0.45, 8.5, 26.5
    channel_defs = [("hot", 1.45), ("average", 1.00), ("cold", 0.65)]

    # state 0 = before the disturbance, state 1 = after. Each perturbation type
    # nudges one of (power, flow, inlet temp); everything else holds.
    q0, m0, Tin0 = RATED_MWT, MDOT, T_IN_NOM
    q1, m1, Tin1 = q0, m0, Tin0

    if perturbation == "power_step":
        q1 = q0 * (1.0 + magnitude / 100.0)
    elif perturbation == "flow_step":
        m1 = m0 * max(0.3, 1.0 + magnitude / 100.0)
    elif perturbation == "inlet_temp":
        Tin1 = Tin0 + magnitude
    elif perturbation == "rod_withdraw":
        q1 = q0 * (1.0 + 0.8 * abs(magnitude) / 100.0)

    z     = np.linspace(0.0, H, n_z)
    shape = np.cos(np.pi * (z - H / 2.0) / HE)
    shape /= _trapz(shape, z) / H

    def steady_profile(q_mw, mdot, T_in, peak):
        q_l = q_mw * 1e3 / (N_RODS * H) * shape * peak
        dT  = (np.concatenate([[0.0],
               np.cumsum((q_l[1:] + q_l[:-1]) / 2.0 * np.diff(z))])
               * N_RODS / (mdot * CP))
        tc  = T_in + dT
        tfc = tc + q_l * (R_film + R_clad + R_gap + R_fuel)
        return {
            "tc": tc, "tfc": tfc, "t_out": float(tc[-1]),
            "dnbr": _dnbr_estimate(float(q_l.max()), float(tc.max()), mdot),
            "q_l": q_l,
        }

    # Lumped two-node model: fuel node and coolant node, each with a heat capacity
    # (Cf, Cc) and coupled by a conductance UA. Picked so the time constants land
    # in the right ballpark for a PWR (fuel responds in seconds, coolant faster).
    UA  = q0 * 1e3 / 320.0
    Cf, Cc = 2.5e5, 1.2e5
    Tc0 = Tin0 + q0 * 1e3 / (2.0 * m0 * CP)   # initial coolant node temp
    Tf0 = Tc0  + q0 * 1e3 / UA                # initial fuel node temp (hotter)
    P1  = q1 * 1e3

    def ode(t, y):
        # ramp the power over the first 2 s instead of a hard step (avoids an ugly
        # discontinuity in the trace), then hold at the new level.
        P = P1 if t > 2.0 else (q0 * 1e3 + (P1 - q0 * 1e3) * (t / 2.0))
        return np.array([
            (P - UA * (y[0] - y[1])) / Cf,
            (UA * (y[0] - y[1]) - 2.0 * m1 * CP * (y[1] - Tin0)) / Cc,
        ])

    t_rk, Y_rk = rk4(ode, np.array([Tf0, Tc0]), 0.0, duration_s,
                      max(400, n_snapshots * 6))
    t_snap  = np.linspace(0.0, duration_s, n_snapshots)
    Tc_lump = np.interp(t_snap, t_rk, Y_rk[:, 1])

    channels_out = {}
    for ch_name, peak in channel_defs:
        p0 = steady_profile(q0, m0, Tin0, peak)   # axial profile before
        p1 = steady_profile(q1, m1, Tin1, peak)   # axial profile after
        # f in [0,1] = how far this channel has moved from start to end, read off
        # the lumped coolant response. f=0 -> profile p0, f=1 -> profile p1.
        dT_out = p1["t_out"] - p0["t_out"]
        f_arr  = np.clip(
            (Tc_lump - Tc_lump[0]) / max(abs(dT_out), 0.5), 0.0, None)

        snaps_tc, snaps_tfc, t_out_tr, dnbr_tr, fpeak_tr = [], [], [], [], []
        for fi in f_arr:
            tc  = p0["tc"]  + fi * (p1["tc"]  - p0["tc"])
            tfc = p0["tfc"] + fi * (p1["tfc"] - p0["tfc"])
            ql  = p0["q_l"] + fi * (p1["q_l"] - p0["q_l"])
            mi  = m0 + fi * (m1 - m0)
            dn  = _dnbr_estimate(float(ql.max()), float(tc.max()),
                                 max(float(mi), 100.0))
            snaps_tc.append(np.round(tc, 2).tolist())
            snaps_tfc.append(np.round(tfc, 2).tolist())
            t_out_tr.append(round(float(tc[-1]), 2))
            dnbr_tr.append(round(dn, 3))
            fpeak_tr.append(round(float(tfc.max()), 1))

        channels_out[ch_name] = {
            "t_cool_snapshots": snaps_tc,
            "t_fuel_snapshots": snaps_tfc,
            "t_out_trace":      t_out_tr,
            "dnbr_trace":       dnbr_tr,
            "fuel_peak_trace":  fpeak_tr,
            "t_out_initial":    round(p0["t_out"], 2),
            "t_out_final":      round(p1["t_out"], 2),
            "dnbr_initial":     round(p0["dnbr"], 3),
            "dnbr_final":       round(p1["dnbr"], 3),
        }

    return {
        "t_snap":       np.round(t_snap, 2).tolist(),
        "z_m":          np.round(z, 3).tolist(),
        "channels":     channels_out,
        "perturbation": perturbation,
        "magnitude":    magnitude,
        "duration_s":   duration_s,
        "n_snapshots":  n_snapshots,
        "t_sat":        T_SAT,
        "p_sys_mpa":    P_SYS,
    }


def transient(power_step_pct=10.0, duration_s=600.0):
    """Lumped two-node (fuel + coolant) transient after a step change.

    The bare-bones version of the model above: no axial profiles, just the two
    coupled ODEs integrated with RK4. Handy for a quick fuel/coolant temperature
    trace after a power step.
    """
    P0   = RATED_MWT * 1e3
    P1   = P0 * (1 + power_step_pct / 100.0)
    Cf   = 2.5e5
    Cc   = 1.2e5
    UA   = P0 / 320.0
    Tin  = T_IN_NOM
    flow = MDOT * CP

    def f(t, y):
        Tf, Tc = y
        P   = P1 if t > 30.0 else P0
        dTf = (P - UA * (Tf - Tc)) / Cf
        dTc = (UA * (Tf - Tc) - 2 * flow * (Tc - Tin)) / Cc
        return np.array([dTf, dTc])

    Tc0 = Tin + P0 / (2 * flow)
    Tf0 = Tc0 + P0 / UA
    t, y = rk4(f, np.array([Tf0, Tc0]), 0.0, duration_s, 1200)
    sl = slice(None, None, 4)
    return {
        "t":                np.round(t[sl], 1).tolist(),
        "T_fuel":           np.round(y[sl, 0], 2).tolist(),
        "T_cool":           np.round(y[sl, 1], 2).tolist(),
        "t_s":              np.round(t[sl], 1).tolist(),
        "fuel_avg_temp":    np.round(y[sl, 0], 2).tolist(),
        "coolant_avg_temp": np.round(y[sl, 1], 2).tolist(),
        "step_pct":         power_step_pct,
        "step_time_s":      30.0,
        "integrator":       "RK4, h=0.5 s, 1200 steps",
    }