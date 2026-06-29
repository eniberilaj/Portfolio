"""
SpaceLab router — LEO Constellation Kinetic-Impact & Collision-Cascade Simulator.

REST
  GET /api/spacelab/config        default sim config + shell descriptors for the UI

WebSocket
  /stream/orbit?density=&accel=   live stream.  Protocol:
      → text  {type:'init', n, n_active, re, kinds:<base64 uint8>}   (once, on open)
      → bin   float32[n*3]  ECI positions in Earth-radii                (every step)
      → text  {type:'tele', t, conjunctions, debris, density, pc, ...}  (throttled)
      ← text  {cmd:'set', density?, accel?, paused?}                    (control)
      ← text  {cmd:'impact'}                                            (kinetic strike)

Positions are packed with ndarray.tobytes() — a contiguous little-endian float32 buffer
the browser drops straight into the InstancedMesh matrices with zero parsing.
"""
from __future__ import annotations
import base64
import json
import threading
import time

import numpy as np

from app import ws
from app.physics import orbital
from app.physics.orbital import RE

# default population / streaming knobs
_DEFAULTS = {
    "density": 12000,        # target object count
    "accel": 60.0,          # sim-seconds advanced per streamed step
    "rate_hz": 24.0,        # streamed steps per wall-clock second
    "cell_km": 12.0,        # spatial-hash voxel side
    "thresh_km": 6.0,       # conjunction distance threshold
}
_MAX_OBJECTS = 22000
_MIN_OBJECTS = 3000
_MAX_STREAMS = 6            # guard against runaway connections
_active_streams = 0
_stream_lock = threading.Lock()


def api_config(q, body):
    """Static descriptors so the dashboard can render real numbers before connecting."""
    shells = [{"alt_km": a, "inc_deg": i, "share": s} for (a, i, s) in orbital._SHELLS]
    return {
        "defaults": _DEFAULTS,
        "limits": {"min": _MIN_OBJECTS, "max": _MAX_OBJECTS},
        "shells": shells,
        "constants": {"RE_km": RE / 1e3, "J2": orbital.J2,
                      "mu": orbital.MU},
        "engine": "pure NumPy · vectorised Kepler + J2 secular · spatial-hash collisions",
    }


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class _Controls:
    """Thread-shared control state mutated by the reader, read by the streamer."""
    def __init__(self, density, accel):
        self.density = density
        self.accel = accel
        self.paused = False
        self.want_impact = False
        self.want_reset = False


def _reader(socket: "ws.WebSocket", ctrl: "_Controls"):
    """Background thread: drain inbound control frames until the socket closes."""
    while socket.open:
        frame = socket.recv()
        if frame is None:
            break
        opcode, data = frame
        if opcode != ws.OP_TEXT:
            continue
        try:
            msg = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        cmd = msg.get("cmd")
        if cmd == "set":
            if "density" in msg:
                ctrl.density = int(_clamp(int(msg["density"]), _MIN_OBJECTS, _MAX_OBJECTS))
                ctrl.want_reset = True
            if "accel" in msg:
                ctrl.accel = float(_clamp(float(msg["accel"]), 1.0, 1200.0))
            if "paused" in msg:
                ctrl.paused = bool(msg["paused"])
        elif cmd == "impact":
            ctrl.want_impact = True
        elif cmd == "reset":
            ctrl.want_reset = True


def stream_orbit(handler, query):
    """WebSocket entry point for /stream/orbit. Runs the live physics loop."""
    global _active_streams
    socket = ws.handshake(handler)
    if socket is None:
        return
    with _stream_lock:
        if _active_streams >= _MAX_STREAMS:
            socket.send_text(json.dumps({"type": "error", "msg": "too many streams"}))
            socket.close()
            return
        _active_streams += 1

    try:
        density = int(_clamp(int(query.get("density", [_DEFAULTS["density"]])[0]),
                             _MIN_OBJECTS, _MAX_OBJECTS))
        accel = float(_clamp(float(query.get("accel", [_DEFAULTS["accel"]])[0]),
                             1.0, 1200.0))
        ctrl = _Controls(density, accel)
        reader = threading.Thread(target=_reader, args=(socket, ctrl), daemon=True)
        reader.start()
        _run_sim(socket, ctrl)
    finally:
        socket.close()
        with _stream_lock:
            _active_streams -= 1


def _build(density):
    pop = orbital.build_population(density)
    prop = orbital.Propagator(pop)
    cm = orbital.CascadeModel(pop["n_debris0"])
    return pop, prop, cm


def _send_init(socket, pop):
    kinds_b64 = base64.b64encode(pop["kind"].astype(np.uint8).tobytes()).decode()
    socket.send_text(json.dumps({
        "type": "init",
        "n": int(pop["n"]),
        "n_active": int(pop["n_active"]),
        "n_debris": int(pop["n_debris0"]),
        "re": 1.0,                     # positions are streamed in Earth-radii
        "kinds": kinds_b64,
        "alt_bins_km": (orbital._ALT_BINS / 1e3).tolist(),   # altitude-profile bin edges
    }))


def _run_sim(socket, ctrl: "_Controls"):
    pop, prop, cm = _build(ctrl.density)
    _send_init(socket, pop)
    active = (~pop["reserve"]) & (~pop["decayed"])

    cell = _DEFAULTS["cell_km"] * 1e3
    thresh = _DEFAULTS["thresh_km"] * 1e3
    interval = 1.0 / _DEFAULTS["rate_hz"]

    t_sim = 0.0
    step = 0
    reentries = 0
    next_tick = time.perf_counter()

    while socket.open:
        # ── handle control intents ──
        if ctrl.want_reset:
            ctrl.want_reset = False
            pop, prop, cm = _build(ctrl.density)
            _send_init(socket, pop)
            active = (~pop["reserve"]) & (~pop["decayed"])
            t_sim = 0.0
            step = 0
            reentries = 0
        if ctrl.want_impact:
            ctrl.want_impact = False
            rel = prop.kinetic_impact(t_sim)
            active = (~pop["reserve"]) & (~pop["decayed"])
            socket.send_text(json.dumps({"type": "event", "kind": "impact",
                                         "released": rel, "t": round(t_sim, 1)}))

        decayed_now = 0
        if not ctrl.paused:
            t_sim += ctrl.accel
            step += 1

        # ── propagate + pack positions (Earth-radii, float32, contiguous) ──
        pos = prop.step(t_sim)                  # (N,3) metres

        if not ctrl.paused:
            burnt = prop.apply_drag(ctrl.accel)   # decay + re-entry burn-up
            if burnt.size:
                decayed_now = int(burnt.size)
                reentries += decayed_now
                active = (~pop["reserve"]) & (~pop["decayed"])
                # tell the browser where to flash a re-entry streak (small sample)
                rpos = (pos[burnt][:18] / RE).astype(np.float32)
                socket.send_text(json.dumps({"type": "event", "kind": "reentry",
                                             "n": decayed_now, "pos": rpos.round(4).tolist()}))

        # decayed objects collapse to the origin → the frontend hides them
        pos = pos.copy()
        pos[pop["decayed"]] = 0.0
        buf = (pos.astype(np.float32) / np.float32(RE))
        socket.send_bytes(np.ascontiguousarray(buf).tobytes())

        # ── collision + cascade telemetry (throttled to every 2nd step) ──
        if step % 2 == 0 and not ctrl.paused:
            n_conj, peak, mean_occ, _ = orbital.conjunctions(pos[active], cell, thresh)
            tele = cm.update(n_conj, mean_occ, ctrl.accel, removed=decayed_now)
            hist = orbital.altitude_histogram(pos[active]).tolist()
            tele.update(type="tele", t=round(t_sim, 1), step=step,
                        n=int(pop["n"]), peak=int(peak), accel=ctrl.accel,
                        active=int(active.sum()), reentries=reentries,
                        decay_rate=round(decayed_now / max(ctrl.accel, 1e-6), 5),
                        alt_hist=hist)
            socket.send_text(json.dumps(tele))

        # ── pace the wall-clock so the browser gets a steady cadence ──
        next_tick += interval
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()      # fell behind; resync


# HTTP routes (the WebSocket route is registered separately as WS_ROUTES)
ROUTES = {"/api/spacelab/config": api_config}
WS_ROUTES = {"/stream/orbit": stream_orbit}
