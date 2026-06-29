"""Reactor Intelligence router -- /api/reactor/*

Thin glue between the frontend and the physics. Each api_* function reads the
current operating point, runs the relevant engine (diffusion, thermal, kinetics,
burnup…), and returns a JSON-able dict. No physics lives here — this file is just
wiring, so the engines stay independently testable.
"""
from __future__ import annotations
import numpy as np
from app.state import get_state, update_state
from app.physics.diffusion import CoreParams, solve_core
from app.physics import thermal
from app.physics.burnup import burnup_state
from app.physics import kinetics, fluctuations
from app.optimization.optimizer import optimize
from app.computing.benchmarks import run_all as run_benchmarks
from app.experiments import registry
from app.physics import coupled
from app.physics import xenon

# The neutron-diffusion solve is the expensive step (~40 ms) and several endpoints
# (core, thermal, fuel, kpis…) all need the same result for the same operating
# point. Cache the last solve so one slider change doesn't re-solve five times.
_solve_cache: dict = {}


def _core_params(state, refine=3):
    """Bundle the live operating point + current burnup into the diffusion solver's
    input struct. refine controls mesh resolution (3 = 45×45 for display, 2 = coarser
    for the optimisation loop where speed matters more than detail)."""
    bu = np.array(burnup_state(state["cycle_day"])["burnup_map"])
    return CoreParams(
        rod_insertion=state["rod_insertion"], enrichment=state["enrichment"],
        coolant_density=state["coolant_density"], moderator_temp=state["moderator_temp"],
        power_demand=state["power_demand"], refine=refine, burnup_map=bu,
    )


def api_xenon(q, body):
    b = body or {}
    st = get_state()
    return xenon.simulate_transient(
        state=st, 
        scenario=b.get("scenario", "scram"), 
        duration_h=float(b.get("duration", 72))
    )

def _solve_current(state):
    """Solve for this operating point, memoised on the full state tuple. We only
    keep the most recent result (clear-then-store) — the user is on one operating
    point at a time, so a 1-entry cache is all that's worth keeping."""
    key = tuple(sorted(state.items()))
    if key not in _solve_cache:
        _solve_cache.clear()
        _solve_cache[key] = solve_core(_core_params(state))
    return _solve_cache[key]


def api_state(q, body):
    return update_state(body) if body else get_state()


def api_core(q, body):
    state = update_state(body) if body else get_state()
    return {"state": state, **_solve_current(state)}


def api_thermal(q, body):
    state = update_state(body) if body else get_state()
    core  = _solve_current(state)
    return thermal.simulate(
        power_pct=state["power_demand"] * 100.0, inlet_temp=state["inlet_temp"],
        flow_pct=state["flow_pct"], peaking=max(core["peaking_factor"], 1.0),
    )


def api_transient(q, body):
    step = float((body or {}).get("step_pct", q.get("step_pct", [10])[0]))
    return thermal.transient(power_step_pct=step)


def api_fuel(q, body):
    state = update_state(body) if body else get_state()
    core  = _solve_current(state)
    return burnup_state(state["cycle_day"], power_map=np.array(core["power_map"]))


def api_kpis(q, body):
    state = get_state()
    core  = _solve_current(state)
    th    = thermal.simulate(
        power_pct=state["power_demand"] * 100.0, inlet_temp=state["inlet_temp"],
        flow_pct=state["flow_pct"], peaking=max(core["peaking_factor"], 1.0),
    )
    fuel = burnup_state(state["cycle_day"], power_map=np.array(core["power_map"]))
    pk   = core["peaking_factor"]
    # "Health" and "safety" are presentation scores for the dashboard gauges, not
    # real licensing metrics — hand-tuned so they sit near 100 at nominal and drop
    # off as you push the core. Health penalises a peaked flux (pk above 1.5) and
    # being far from critical (keff off 1.0); safety rewards DNBR margin, boiling
    # margin, and cycle life remaining. The clips just keep the needles on-dial.
    health = float(np.clip(
        100 - 18 * max(pk - 1.5, 0) ** 1.4 * 10
            - max(abs(core["keff"] - 1.0) - 0.005, 0) * 800, 60, 99.5))
    safety = float(np.clip(
        40 * min(th["dnbr"] / 1.3 - 1, 1.0)
        + 30 * min(th["t_sat_margin"] / 12.0, 1.0)
        + 30 * min(fuel["days_remaining"] / 30.0, 1.0), 50, 99.9))
    return {
        "state": state,
        "reactor_power_pct":      state["power_demand"] * 100.0,
        "thermal_power_mw":       th["thermal_power_mw"],
        "electric_power_mw":      th["electric_power_mw"],
        "thermal_efficiency_pct": th["efficiency_pct"],
        "keff":                   core["keff"],
        "reactivity_pcm":         core["reactivity_pcm"],
        "peaking_factor":         pk,
        "core_health_pct":        round(health, 1),
        "reactivity_margin_pcm":  fuel["reactivity_margin_pcm"],
        "fuel_utilization_pct":   fuel["utilization_pct"],
        "avg_burnup":             fuel["avg_burnup"],
        "safety_score":           round(safety, 1),
        "days_remaining":         fuel["days_remaining"],
        "cycle_day":              state["cycle_day"],
        "dnbr":                   th["dnbr"],
        "t_outlet":               th["t_outlet"],
        "t_sat_margin":           th["t_sat_margin"],
    }


def api_optimize(q, body):
    # Wrap the rod-pattern optimisation in an "experiment" record so the run shows
    # up in the registry with before/after results (and a 'failed' status if it throws).
    state = get_state()
    exp = registry.create("reactor", "optimization", "RCCA pattern optimization",
                          {"max_iter": int((body or {}).get("max_iter", 30))})
    try:
        res = optimize(_core_params(state, refine=2),
                       max_iter=int((body or {}).get("max_iter", 30)))
        registry.finish(exp["id"], {
            "peaking_before": res["before"]["peaking"],
            "peaking_after":  res["after"]["peaking"],
            "keff_after":     res["after"]["keff"],
        })
        res["experiment_id"] = exp["id"]
        return res
    except Exception:
        registry.update(exp["id"], status="failed")
        raise


def api_benchmarks(q, body):
    return run_benchmarks()


def api_coupled(q, body):
    b = body or {}
    st = get_state()
    return coupled.solve_coupled(
        power_level=float(b.get("power_level", st.get("power_demand", 1.0))),
        mdot_frac=float(b.get("mdot_frac", 1.0)),
        inlet_temp=float(b.get("inlet_temp", 286.0)),
        rod_insertion=float(b.get("rod_insertion", st.get("rod_insertion", 0.22))),
        enrichment=float(b.get("enrichment", st.get("enrichment", 3.2))),
        n_axial=int(b.get("n_axial", 40)),
    )
 
def api_scenario(q, body):
    b = body or {}
    return coupled.simulate_scenario(
        scenario=b.get("scenario", "scram"),
        duration=float(b.get("duration", 40.0)),
        n_steps=int(b.get("n_steps", 4000)),
    )
 
def api_rod_drop(q, body):
    b = body or {}
    return coupled.asymmetric_rod_drop(
        enrichment=float(b.get("enrichment", get_state().get("enrichment", 3.2))),
    )




def api_flux_noise(q, body):
    b    = body or {}
    core = _solve_current(get_state())
    return fluctuations.generate(
        core_result=core,
        n_steps=int(b.get("n_steps", 300)),
        dt=float(b.get("dt", 0.005)),
        amplitude=float(b.get("amplitude", 0.015)),
        correlation_length=float(b.get("correlation_length", 1.8)),
        tau_c=float(b.get("tau_c", 0.10)),
        seed=int(b.get("seed", 42)),
    )


def api_kinetics(q, body):
    b = body or {}
    return kinetics.solve(
        rho_type=b.get("rho_type", "step"),
        rho_pcm=float(b.get("rho_pcm", 50.0)),
        t_insert=float(b.get("t_insert", 1.0)),
        duration=float(b.get("duration", 30.0)),
        ramp_rate=float(b.get("ramp_rate", 10.0)),
        freq=float(b.get("freq", 0.5)),
        n_steps=int(b.get("n_steps", 2000)),
    )


def api_live_transients(q, body):
    b = body or {}
    return thermal.transient_multichannel(
        perturbation=b.get("perturbation", "power_step"),
        magnitude=float(b.get("magnitude", 10.0)),
        duration_s=float(b.get("duration_s", 120.0)),
        n_snapshots=int(b.get("n_snapshots", 60)),
    )


# The path -> handler table the server merges into its global dispatch map.
ROUTES = {
    "/api/reactor/state":           api_state,
    "/api/reactor/core":            api_core,
    "/api/reactor/thermal":         api_thermal,
    "/api/reactor/transient":       api_transient,
    "/api/reactor/fuel":            api_fuel,
    "/api/reactor/kpis":            api_kpis,
    "/api/reactor/optimize":        api_optimize,
    "/api/reactor/benchmarks":      api_benchmarks,
    "/api/reactor/flux_noise":      api_flux_noise,
    "/api/reactor/kinetics":        api_kinetics,
    "/api/reactor/live_transients": api_live_transients,
    "/api/reactor/coupled":         api_coupled,
    "/api/reactor/scenario":        api_scenario,
    "/api/reactor/rod_drop":        api_rod_drop,
    "/api/reactor/xenon":           api_xenon,
}
