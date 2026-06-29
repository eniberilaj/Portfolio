"""
LEO Constellation Kinetic-Impact & Collision-Cascade engine — pure, vectorised NumPy.

No external physics libraries (no sgp4 / astropy / numba). Every quantity below is a
NumPy array of shape (N,) or (N,3); there is not a single Python `for` loop in any hot
path. The whole population of 20 000+ objects is advanced with array algebra only.

What lives here
---------------
1.  build_population()       Walker-delta constellations + a background debris belt.
2.  Propagator.step(t)       Keplerian two-body motion + **J2 secular precession** of the
                             ascending node (Ω̇), the argument of perigee (ω̇) and the mean
                             anomaly (Ṁ).  Kepler's equation is solved with a fully
                             vectorised **Newton–Raphson** iteration.
3.  coe2rv / rv2coe          Classical-elements ⇄ Cartesian state, vectorised — used both
                             for rendering and for the kinetic-impact breakup (apply Δv to a
                             state, then recover the new osculating elements).
4.  conjunctions()           **Vectorised spatial hashing**: np.floor discretises the ECI
                             cloud into a grid, np.lexsort orders it, and same-cell
                             adjacency yields candidate close approaches — no pairwise loop.
5.  CascadeModel             A Kessler-style actuarial accumulator: turns the live
                             conjunction rate + spatial density into a projected cascading
                             debris count and a Probability-of-Cascade  P_c.

Units are SI internally (metres, seconds, radians).  Positions are returned in metres;
the frontend scales them to Earth-radii for the WebGL globe.
"""
from __future__ import annotations
import numpy as np

# ── Physical constants (WGS-84 / EGM) ────────────────────────────────────────
MU  = 3.986004418e14      # Earth gravitational parameter   [m^3 s^-2]
RE  = 6.378137e6          # Earth equatorial radius          [m]
J2  = 1.08262668e-3       # second zonal harmonic            [-]
TWO_PI = 2.0 * np.pi

# LEO band the simulator works in (altitudes above the surface)
ALT_MIN = 300e3
ALT_MAX = 1600e3

# ── atmospheric-drag / orbital-decay model ───────────────────────────────────
# Piecewise-exponential atmosphere: density falls off with a scale height that
# itself grows with altitude (a coarse fit to the US Standard Atmosphere / NRLMSISE
# envelope). Drag is what removes debris from LEO — the natural counterweight to
# the Kessler cascade — so a re-entry sink makes the long-run risk honest.
_RHO0 = 2.8e-10           # reference density at H_REF [kg/m^3]  (~200 km)
_H_REF = 200e3            # reference altitude [m]
_H_SCALE = 45e3           # exponential scale height [m] (coarse 150–900 km fit)
_REENTRY_ALT = 100e3      # perigee below this → atmospheric burn-up [m]
_DRAG_GAIN = 22.0         # demo-acceleration of the (otherwise multi-year) decay

# altitude bins for the live population-vs-altitude profile [m]
_ALT_BINS = np.linspace(150e3, 1700e3, 17)


def atmosphere_density(alt: np.ndarray) -> np.ndarray:
    """Coarse exponential atmosphere density [kg/m^3] for an altitude array [m].

    A single exponential ρ = ρ₀·exp(−(h−h_ref)/H): steep enough that drag bites hard
    below ~400 km and is negligible above ~900 km — which is exactly why the low belt
    self-cleans while the high belt is effectively permanent.
    """
    return _RHO0 * np.exp(-(np.maximum(alt, 0.0) - _H_REF) / _H_SCALE)


def altitude_histogram(pos: np.ndarray) -> np.ndarray:
    """Object count per altitude band from ECI positions [m] — the debris-belt profile."""
    r = np.sqrt(np.einsum("ij,ij->i", pos, pos))
    counts, _ = np.histogram(r - RE, bins=_ALT_BINS)
    return counts


# ════════════════════════════════════════════════════════════════════════════
#  Kepler solver  —  vectorised Newton–Raphson
# ════════════════════════════════════════════════════════════════════════════
def solve_kepler(M: np.ndarray, e: np.ndarray, iters: int = 6) -> np.ndarray:
    """Solve  M = E − e·sin E  for the eccentric anomaly E, for the whole array at once.

    Newton–Raphson:  E ← E − f/f'  with  f = E − e sinE − M,  f' = 1 − e cosE.
    The starting guess  E₀ = M + e·sinM  is second-order accurate, so a handful of
    fixed iterations drive every element below machine-ish tolerance for LEO (e ≲ 0.1).
    Runs as pure array algebra — `iters` is tiny and constant, so this is O(N), loop-free
    over the population.
    """
    M = np.mod(M, TWO_PI)
    E = M + e * np.sin(M)                       # vectorised seed
    for _ in range(iters):                      # fixed, population-independent
        f  = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E  = E - f / fp
    return E


# ════════════════════════════════════════════════════════════════════════════
#  Classical orbital elements  ⇄  Cartesian state   (all vectorised)
# ════════════════════════════════════════════════════════════════════════════
def coe2rv(a, e, i, raan, argp, nu, want_v: bool = False):
    """Classical elements → ECI position (and optionally velocity).

    Builds the perifocal state then rotates it through the 3-1-3 sequence
    R = Rz(Ω)·Rx(i)·Rz(ω).  All inputs are (N,) arrays; output r is (N,3).
    """
    p = a * (1.0 - e * e)                        # semi-latus rectum
    r = p / (1.0 + e * np.cos(nu))              # radius

    cnu, snu = np.cos(nu), np.sin(nu)
    # perifocal position
    xp = r * cnu
    yp = r * snu

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i),    np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    # rows of Rz(Ω)·Rx(i)·Rz(ω)
    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    r31 = sw * si
    r32 = cw * si

    x = r11 * xp + r12 * yp
    y = r21 * xp + r22 * yp
    z = r31 * xp + r32 * yp
    pos = np.stack((x, y, z), axis=-1)

    if not want_v:
        return pos

    # perifocal velocity  (√(μ/p)·[−sinν , e+cosν])
    sqmu_p = np.sqrt(MU / p)
    vxp = -sqmu_p * snu
    vyp = sqmu_p * (e + cnu)
    vx = r11 * vxp + r12 * vyp
    vy = r21 * vxp + r22 * vyp
    vz = r31 * vxp + r32 * vyp
    vel = np.stack((vx, vy, vz), axis=-1)
    return pos, vel


def rv2coe(r: np.ndarray, v: np.ndarray):
    """Cartesian state → classical elements (vectorised).  r,v are (N,3).

    Returns (a, e, i, raan, argp, nu).  Used after a kinetic-impact Δv to recover the
    osculating elements of every fragment so they can be fed straight back into the
    secular-J2 propagator.  Standard Vallado algorithm, expressed with array reductions.
    """
    eps = 1e-12
    rmag = np.linalg.norm(r, axis=1)
    vmag = np.linalg.norm(v, axis=1)

    h = np.cross(r, v)                           # specific angular momentum
    hmag = np.linalg.norm(h, axis=1)
    khat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(khat, h)                     # node vector
    nmag = np.linalg.norm(nvec, axis=1)

    rdotv = np.einsum('ij,ij->i', r, v)
    # eccentricity vector
    evec = ((vmag ** 2 - MU / rmag)[:, None] * r - rdotv[:, None] * v) / MU
    e = np.linalg.norm(evec, axis=1)

    energy = vmag ** 2 / 2.0 - MU / rmag
    a = -MU / (2.0 * energy)

    i = np.arccos(np.clip(h[:, 2] / np.maximum(hmag, eps), -1.0, 1.0))

    raan = np.arccos(np.clip(nvec[:, 0] / np.maximum(nmag, eps), -1.0, 1.0))
    raan = np.where(nvec[:, 1] < 0.0, TWO_PI - raan, raan)

    argp = np.arccos(np.clip(
        np.einsum('ij,ij->i', nvec, evec) / np.maximum(nmag * e, eps), -1.0, 1.0))
    argp = np.where(evec[:, 2] < 0.0, TWO_PI - argp, argp)

    nu = np.arccos(np.clip(
        np.einsum('ij,ij->i', evec, r) / np.maximum(e * rmag, eps), -1.0, 1.0))
    nu = np.where(rdotv < 0.0, TWO_PI - nu, nu)

    # equatorial / circular guards so node-degenerate orbits stay finite
    raan = np.where(nmag < eps, 0.0, raan)
    argp = np.where((nmag < eps) | (e < eps), 0.0, argp)
    return a, e, i, raan, argp, nu


# ════════════════════════════════════════════════════════════════════════════
#  Population builder  —  Walker-delta shells + background debris belt
# ════════════════════════════════════════════════════════════════════════════
# Realistic-ish LEO shells (altitude km, inclination deg, share of the constellation)
_SHELLS = [
    (550.0,  53.0,  0.34),   # Starlink-like
    (570.0,  70.0,  0.14),
    (560.0,  97.6,  0.16),   # sun-synchronous
    (1200.0, 87.9,  0.18),   # OneWeb-like
    (780.0,  86.4,  0.18),   # Iridium-like
]


def _walker_shell(n, alt_km, inc_deg, rng):
    """One Walker-delta shell of n satellites: P planes evenly in RAAN, S per plane
    evenly in mean anomaly, with an inter-plane phasing offset. Fully vectorised."""
    n = int(max(n, 1))
    planes = int(max(2, round(np.sqrt(n * 1.4))))   # roughly square-ish grid
    per = int(np.ceil(n / planes))
    plane_idx = np.repeat(np.arange(planes), per)[:n]
    slot_idx = np.tile(np.arange(per), planes)[:n]

    a = np.full(n, RE + alt_km * 1e3)
    e = rng.uniform(0.0002, 0.0025, n)              # near-circular
    i = np.full(n, np.radians(inc_deg)) + rng.normal(0, np.radians(0.04), n)
    raan = np.mod(plane_idx * (TWO_PI / planes), TWO_PI)
    phasing = (TWO_PI / n) * plane_idx              # Walker phasing F-term
    M = np.mod(slot_idx * (TWO_PI / per) + phasing, TWO_PI)
    argp = rng.uniform(0, TWO_PI, n)
    return a, e, i, raan, argp, M


def build_population(n_target: int = 12000, debris_frac: float = 0.45, seed: int = 7):
    """Assemble the initial constellation.

    Returns a dict of (N,) element arrays plus a `kind` array
    (0 = active satellite, 1 = pre-existing debris) and a `reserve` mask flagging
    dormant fragment slots used by the kinetic-impact breakup.  N ≥ n_target.
    """
    rng = np.random.default_rng(seed)
    n_active = int(round(n_target * (1.0 - debris_frac)))
    n_debris = int(round(n_target * debris_frac))

    A, E, I, RA, AR, M, K = [], [], [], [], [], [], []

    # ── active satellites across the realistic shells ──
    for alt, inc, share in _SHELLS:
        ns = int(round(n_active * share))
        a, e, i, raan, argp, m = _walker_shell(ns, alt, inc, rng)
        A.append(a); E.append(e); I.append(i); RA.append(raan); AR.append(argp); M.append(m)
        K.append(np.zeros(ns, dtype=np.uint8))

    # ── background debris belt: broad, inclined, slightly eccentric ──
    #    sample perigee/apogee altitudes so perigee can never fall below the atmosphere
    hp = rng.uniform(ALT_MIN, ALT_MAX, n_debris)          # perigee altitude [m]
    ha = hp + rng.uniform(0.0, 600e3, n_debris)           # apogee ≥ perigee
    rp = RE + hp
    ra = RE + ha
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    i = np.radians(rng.uniform(20.0, 110.0, n_debris))
    raan = rng.uniform(0, TWO_PI, n_debris)
    argp = rng.uniform(0, TWO_PI, n_debris)
    m = rng.uniform(0, TWO_PI, n_debris)
    A.append(a); E.append(e); I.append(i); RA.append(raan); AR.append(argp); M.append(m)
    K.append(np.ones(n_debris, dtype=np.uint8))

    # ── dormant reserve fragments for kinetic-impact events (start coincident with a
    #    target satellite so they are invisible until released) ──
    n_res = 1400
    tgt = 0                                          # released around population index 0
    a = np.full(n_res, A[0][0])
    e = np.full(n_res, E[0][0])
    i = np.full(n_res, I[0][0])
    raan = np.full(n_res, RA[0][0])
    argp = np.full(n_res, AR[0][0])
    m = np.full(n_res, M[0][0])
    A.append(a); E.append(e); I.append(i); RA.append(raan); AR.append(argp); M.append(m)
    K.append(np.full(n_res, 2, dtype=np.uint8))      # kind 2 = dormant reserve

    pop = dict(
        a=np.concatenate(A).astype(np.float64),
        e=np.concatenate(E).astype(np.float64),
        i=np.concatenate(I).astype(np.float64),
        raan=np.concatenate(RA).astype(np.float64),
        argp=np.concatenate(AR).astype(np.float64),
        M0=np.concatenate(M).astype(np.float64),
        kind=np.concatenate(K),
    )
    n = pop["a"].size
    pop["epoch"] = np.zeros(n)                       # per-object epoch [s]
    pop["reserve"] = pop["kind"] == 2               # dormant slots
    pop["decayed"] = np.zeros(n, dtype=bool)        # burnt up on re-entry
    # ballistic coefficient BC = Cd·A/m [m²/kg]: intact satellites are compact, while
    # tumbling debris has a high area-to-mass ratio and so decays far faster.
    bc = np.empty(n)
    k = pop["kind"]
    bc[k == 0] = rng.uniform(0.004, 0.012, int(np.count_nonzero(k == 0)))   # active
    bc[k == 1] = rng.uniform(0.020, 0.160, int(np.count_nonzero(k == 1)))   # debris
    bc[k == 2] = 0.010                                                      # reserve
    pop["bc"] = bc
    pop["n_active"] = int(np.count_nonzero(pop["kind"] == 0))
    pop["n_debris0"] = int(np.count_nonzero(pop["kind"] == 1))
    pop["n"] = n
    return pop


# ════════════════════════════════════════════════════════════════════════════
#  Propagator  —  two-body + J2 secular precession (vectorised)
# ════════════════════════════════════════════════════════════════════════════
class Propagator:
    """Holds the population and advances it analytically to any absolute time t.

    The J2 secular rates are evaluated once per orbit-element set (they only change
    when the impact breakup mutates elements), so a `step` is just three linear
    Ω/ω/M updates, a vectorised Kepler solve and one coe2rv rotation.
    """

    def __init__(self, pop: dict):
        self.pop = pop
        self._recompute_rates()

    def _recompute_rates(self):
        p = self.pop
        n_mean = np.sqrt(MU / p["a"] ** 3)          # mean motion [rad/s]
        sl = p["a"] * (1.0 - p["e"] ** 2)           # semi-latus rectum
        k = 1.5 * J2 * (RE / sl) ** 2 * n_mean      # common J2 factor
        ci = np.cos(p["i"]); si2 = np.sin(p["i"]) ** 2
        self.n_mean = n_mean
        self.raan_dot = -k * ci                                       # node regression
        self.argp_dot = k * (2.0 - 2.5 * si2)                        # apsidal rotation
        self.m_dot = n_mean + k * np.sqrt(1.0 - p["e"] ** 2) * (1.0 - 1.5 * si2)

    def step(self, t: float, want_v: bool = False):
        """Return ECI positions (N,3) [m] at absolute time t (and velocities if asked)."""
        p = self.pop
        dt = t - p["epoch"]
        raan = p["raan"] + self.raan_dot * dt
        argp = p["argp"] + self.argp_dot * dt
        M = p["M0"] + self.m_dot * dt
        E = solve_kepler(M, p["e"])
        # eccentric → true anomaly
        nu = 2.0 * np.arctan2(np.sqrt(1.0 + p["e"]) * np.sin(E / 2.0),
                              np.sqrt(1.0 - p["e"]) * np.cos(E / 2.0))
        return coe2rv(p["a"], p["e"], p["i"], raan, argp, nu, want_v=want_v)

    # ── atmospheric drag → orbital decay → re-entry ──────────────────────────
    def apply_drag(self, dt: float):
        """Advance the semi-analytic drag decay by `dt` seconds and burn up re-entries.

        Drag dominates near perigee, so density is evaluated there. The semi-major
        axis bleeds off as  da/dt ≈ −G·(Cd A/m)·ρ·√(μa)  and the orbit circularises;
        once the perigee drops below ~100 km the object is flagged decayed (burnt up).
        Returns the integer indices that re-entered this step (for the visual flash).
        """
        p = self.pop
        live = (~p["reserve"]) & (~p["decayed"])
        if not live.any():
            return np.empty(0, dtype=np.int64)

        a, e = p["a"], p["e"]
        hp = a * (1.0 - e) - RE                     # perigee altitude
        rho = atmosphere_density(hp)
        da = -_DRAG_GAIN * p["bc"] * rho * np.sqrt(MU * np.maximum(a, RE)) * dt
        frac = np.clip(-da / np.maximum(a, 1.0), 0.0, 0.5)   # shrink fraction
        a_new = np.maximum(a + da, RE + _REENTRY_ALT)
        e_new = np.clip(e * (1.0 - 0.7 * frac), 0.0, 0.95)   # drag circularises

        p["a"] = np.where(live, a_new, a)
        p["e"] = np.where(live, e_new, e)

        hp_new = p["a"] * (1.0 - p["e"]) - RE
        newly = live & (hp_new < _REENTRY_ALT)
        idx = np.where(newly)[0]
        if idx.size:
            p["decayed"][idx] = True
        self._recompute_rates()
        return idx

    # ── kinetic-impact / ASAT breakup ────────────────────────────────────────
    def kinetic_impact(self, t: float, n_frag: int = 900, dv: float = 180.0, seed=None):
        """Detonate the dormant reserve into a fragment cloud at time t.

        The reserve slots ride coincident with a target satellite, so we read the
        target's state (r,v), add an isotropic Δv shell (NASA-EVOLVE-style spread),
        convert back to elements with rv2coe, and re-epoch them at t.  Returns the
        number of fragments actually released.
        """
        p = self.pop
        idx = np.where(p["reserve"])[0]
        if idx.size == 0:
            return 0
        idx = idx[:n_frag]
        rng = np.random.default_rng(seed)

        # state of the reserve slots right now (they still track the target)
        r, v = self.step(t, want_v=True)
        r = r[idx]; v = v[idx]

        # isotropic Δv shell with a log-normal speed spread
        u = rng.normal(size=(idx.size, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        speed = dv * rng.lognormal(mean=0.0, sigma=0.5, size=idx.size)
        v_new = v + u * speed[:, None]

        a, e, i, raan, argp, nu = rv2coe(r, v_new)
        # mean anomaly at epoch from true anomaly
        E = np.arctan2(np.sqrt(1.0 - e ** 2) * np.sin(nu), e + np.cos(nu))
        M = E - e * np.sin(E)

        # keep only bound, sane fragments (reject hyperbolic / re-entry / sub-orbital)
        perigee = a * (1.0 - e)
        good = (e < 0.95) & (a > RE + 150e3) & (perigee > RE + 120e3) & np.isfinite(a)
        sel = idx[good]
        p["a"][sel] = a[good]
        p["e"][sel] = e[good]
        p["i"][sel] = i[good]
        p["raan"][sel] = raan[good]
        p["argp"][sel] = argp[good]
        p["M0"][sel] = np.mod(M[good], TWO_PI)
        p["epoch"][sel] = t
        p["kind"][sel] = 3                           # kind 3 = fresh impact debris
        p["bc"][sel] = rng.uniform(0.05, 0.6, int(sel.size))   # light frags → faster decay
        p["decayed"][sel] = False
        p["reserve"][sel] = False
        self._recompute_rates()
        return int(sel.size)


# ════════════════════════════════════════════════════════════════════════════
#  Collision engine  —  vectorised spatial hashing (no pairwise loop)
# ════════════════════════════════════════════════════════════════════════════
def conjunctions(pos: np.ndarray, cell: float = 12e3, d_thresh: float = 6e3):
    """Find close-approach conjunctions in an ECI cloud via spatial hashing.

    Algorithm (entirely array ops):
      1. np.floor maps each (x,y,z) into an integer grid cell of side `cell`.
      2. The 3 integer coords are packed into one 1-D key.
      3. np.lexsort orders objects so that cell-mates become adjacent.
      4. Equal-key adjacency (and one extra shift, to catch >2 occupants) gives
         candidate pairs sharing a voxel — *without* an O(N²) pairwise scan.
      5. Candidates are filtered by true Euclidean separation < `d_thresh`.

    Returns (n_conj, peak_occupancy, mean_occupancy, pair_midpoints[K,3]).
    """
    n = pos.shape[0]
    if n < 2:
        return 0, 0, 0.0, np.empty((0, 3))

    g = np.floor(pos / cell).astype(np.int64)        # (N,3) voxel coords
    gmin = g.min(axis=0)
    g -= gmin                                         # non-negative
    dims = g.max(axis=0) + 1
    # linear voxel key (row-major); int64 keeps it exact for LEO-sized grids
    key = (g[:, 0].astype(np.int64) * dims[1] + g[:, 1]) * dims[2] + g[:, 2]

    order = np.lexsort((g[:, 2], g[:, 1], g[:, 0]))  # lexsort by voxel coords
    ks = key[order]

    # per-cell occupancy via run-length on the sorted keys
    uniq, counts = np.unique(key, return_counts=True)
    peak = int(counts.max())
    occupied = uniq.size
    mean_occ = float(n / max(occupied, 1))

    # candidate pairs: consecutive sorted entries in the same voxel (shifts 1 & 2)
    cand_a = []
    cand_b = []
    for s in (1, 2):                                 # constant 2 iterations, not O(N)
        same = ks[s:] == ks[:-s]
        cand_a.append(order[:-s][same])
        cand_b.append(order[s:][same])
    ia = np.concatenate(cand_a)
    ib = np.concatenate(cand_b)

    if ia.size == 0:
        return 0, peak, mean_occ, np.empty((0, 3))

    d2 = np.sum((pos[ia] - pos[ib]) ** 2, axis=1)
    hit = d2 < d_thresh * d_thresh
    n_conj = int(np.count_nonzero(hit))
    mids = 0.5 * (pos[ia[hit]] + pos[ib[hit]]) if n_conj else np.empty((0, 3))
    return n_conj, peak, mean_occ, mids


# ════════════════════════════════════════════════════════════════════════════
#  Cascade model  —  Kessler-style actuarial accumulator
# ════════════════════════════════════════════════════════════════════════════
class CascadeModel:
    """Turns the live conjunction stream into actuarial risk numbers.

    State machine (per step, dt seconds of sim time):
      • collisions  ~  conjunctions · P(hit | conjunction)
      • each collision injects  FRAG  catastrophic fragments
      • the standing debris population feeds back (Kessler): collision rate grows
        with the square of spatial density, so P_c climbs super-linearly.
    P_c is a saturating hazard:  P_c = 1 − exp(−Λ),  Λ = cumulative collision risk.
    """
    P_HIT = 0.012            # conditional collision probability per conjunction
    FRAG = 120               # fragments per catastrophic collision (NASA breakup model)
    DENS_CRIT = 9.0          # critical mean occupancy that defines a runaway regime

    def __init__(self, n_debris0: int):
        self.debris = float(n_debris0)
        self.n_debris0 = float(max(n_debris0, 1))
        self.cum_collisions = 0.0
        self.hazard = 0.0
        self.pc = 0.0
        self.rate = 0.0

    def update(self, n_conj: int, mean_occ: float, dt: float, removed: int = 0):
        # density feedback factor (≥1, grows ~quadratically past critical density)
        fb = 1.0 + (mean_occ / self.DENS_CRIT) ** 2
        collisions = n_conj * self.P_HIT * fb
        self.rate = collisions / max(dt, 1e-6)
        self.cum_collisions += collisions
        # source (fragmentation) minus sink (atmospheric re-entry) — the Kessler balance
        self.debris += collisions * self.FRAG
        self.debris = max(0.0, self.debris - float(removed))

        # hazard accumulates with the debris-weighted collision risk
        growth = (self.debris / self.n_debris0)
        self.hazard += collisions * growth * 0.0015
        self.pc = 1.0 - np.exp(-self.hazard)
        return self.telemetry(n_conj, mean_occ)

    def telemetry(self, n_conj, mean_occ):
        return {
            "conjunctions": int(n_conj),
            "debris": int(round(self.debris)),
            "cum_collisions": round(self.cum_collisions, 3),
            "density": round(mean_occ, 3),
            "rate": round(self.rate, 4),
            "pc": round(float(self.pc), 5),
        }


# ════════════════════════════════════════════════════════════════════════════
#  Self-test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    pop = build_population(12000)
    prop = Propagator(pop)
    print(f"population N = {pop['n']}  (active {pop['n_active']}, "
          f"debris {pop['n_debris0']}, reserve {int(pop['reserve'].sum())})")

    # round-trip coe<->rv sanity check
    r, v = prop.step(0.0, want_v=True)
    a, e, i, raan, argp, nu = rv2coe(r, v)
    da = np.nanmax(np.abs(a - pop["a"]))
    print(f"coe2rv->rv2coe  max |Δa| = {da:.3e} m")

    # propagation timing + altitude conservation
    t0 = time.perf_counter()
    for k in range(20):
        pos = prop.step(k * 60.0)
    dt_ms = (time.perf_counter() - t0) / 20 * 1e3
    alt = (np.linalg.norm(pos, axis=1) - RE) / 1e3
    print(f"step: {dt_ms:.2f} ms/frame   alt range {alt.min():.0f}–{alt.max():.0f} km")

    # collision engine — dormant reserves are parked/coincident, exclude them
    active = ~pop["reserve"]
    t0 = time.perf_counter()
    n_conj, peak, mean_occ, mids = conjunctions(pos[active])
    print(f"conjunctions: {n_conj}  peak/cell {peak}  mean-occ {mean_occ:.2f}  "
          f"({(time.perf_counter()-t0)*1e3:.2f} ms)")

    # cascade + kinetic impact
    cm = CascadeModel(pop["n_debris0"])
    print("telemetry:", cm.update(n_conj, mean_occ, 60.0))
    released = prop.kinetic_impact(20 * 60.0)
    print(f"kinetic impact released {released} fragments")
