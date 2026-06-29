"""
Global reactor plant state — shared mutable dict with thread-safe update.

The reactor's operating point (rod position, enrichment, …) needs to be readable
and writable from any request thread. Since the server is threaded, I guard the
single shared dict with a lock so concurrent reads/writes can't tear. Tiny, but it
beats sprinkling locks all over the API modules.
"""
from __future__ import annotations
import threading

_LOCK = threading.Lock()

# Nominal operating point — also what reset_state() restores to.

_DEFAULT: dict = {
    "rod_insertion":    0.22,   # fraction 0–1
    "enrichment":       3.2,    # w/o U-235
    "coolant_density":  0.72,   # g/cm³
    "moderator_temp":   305.0,  # °C
    "power_demand":     1.00,   # fraction 0–1.05
    "inlet_temp":       286.0,  # °C
    "flow_pct":         100.0,  # % of rated
    "cycle_day":        185,    # EFPD
}

_STATE: dict = dict(_DEFAULT)


def get_state() -> dict:
    # Hand back a *copy* so callers can't mutate the shared dict behind the lock.
    with _LOCK:
        return dict(_STATE)


def update_state(patch: dict) -> dict:
    with _LOCK:
        for k, v in (patch or {}).items():
            # Only accept known keys, and coerce the incoming value to the existing
            # type — JSON from the browser arrives as strings/floats, so e.g.
            # cycle_day stays an int. Quietly ignores anything unexpected.
            if k in _STATE:
                _STATE[k] = type(_STATE[k])(v)
        return dict(_STATE)


def reset_state() -> dict:
    with _LOCK:
        _STATE.update(_DEFAULT)
        return dict(_STATE)
