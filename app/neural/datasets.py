"""
Synthetic dataset generators for the Neural Lab.

Toy 2-D datasets to train the from-scratch MLP on. They're deliberately the
classic "hard for a linear model" shapes — spirals, moons, XOR — so the decision
boundary the network learns is actually interesting to watch evolve. Each one is
just a bit of trig plus Gaussian noise; `noise` controls how much the classes blur
into each other.
"""
from __future__ import annotations
import numpy as np

TYPES = ["two_spirals", "moons", "gaussian_blobs", "concentric_rings",
         "xor", "damped_wave"]

# Generated datasets are kept server-side by id so a later /train call can refer
# back to the exact same points (the browser only gets a copy to plot).
_DS: dict = {}


def generate(ds_type="two_spirals", n=600, noise=0.12, seed=0,
             classes=3) -> dict:
    rng = np.random.default_rng(seed)
    if ds_type == "two_spirals":
        X, y = _spirals(n, noise, rng)
    elif ds_type == "moons":
        X, y = _moons(n, noise, rng)
    elif ds_type == "gaussian_blobs":
        X, y = _blobs(n, noise, rng, classes)
    elif ds_type == "concentric_rings":
        X, y = _rings(n, noise, rng, classes)
    elif ds_type == "xor":
        X, y = _xor(n, noise, rng)
    elif ds_type == "damped_wave":
        X, y = _wave(n, noise, rng)
    else:
        raise ValueError(f"Unknown ds_type: {ds_type!r}")

    ds_id = f"ds_{len(_DS):04d}"
    _DS[ds_id] = {"X": X, "y": y, "type": ds_type, "n": n,
                  "noise": noise, "seed": seed}
    return {
        "id":    ds_id,
        "type":  ds_type,
        "n":     int(n),
        "x":     np.round(X, 5).tolist(),
        "y":     y.tolist(),
        "dim":   int(X.shape[1]),
        "classes": int(len(np.unique(y))),
    }


def get(ds_id: str) -> dict:
    return _DS[ds_id]


def list_all() -> list:
    return [{"id": k, "type": v["type"], "n": v["n"]} for k, v in _DS.items()]


def latest_id() -> str | None:
    return list(_DS.keys())[-1] if _DS else None


# ── Generators ─────────────────────────────────────────────────────────────

def _spirals(n, noise, rng):
    # Two interleaved Archimedean spirals (class 1 is class 0 rotated by π). The
    # classic non-linearly-separable benchmark — a single line can't split these.
    half = n // 2
    theta = np.linspace(0, 4*np.pi, half)
    r = np.linspace(0.1, 1.0, half)
    X0 = np.c_[r*np.cos(theta), r*np.sin(theta)]
    X1 = np.c_[r*np.cos(theta+np.pi), r*np.sin(theta+np.pi)]
    X  = np.vstack([X0, X1]) + rng.standard_normal((n, 2)) * noise
    y  = np.array([0]*half + [1]*half)
    return X, y


def _moons(n, noise, rng):
    half = n // 2
    t = np.linspace(0, np.pi, half)
    X0 = np.c_[np.cos(t), np.sin(t)]
    X1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X  = np.vstack([X0, X1]) + rng.standard_normal((n, 2)) * noise
    y  = np.array([0]*half + [1]*half)
    return X, y


def _blobs(n, noise, rng, k):
    k  = max(2, min(k, 5))
    cx = rng.uniform(-1, 1, (k, 2))
    per = n // k
    Xs, ys = [], []
    for i in range(k):
        ni = per if i < k-1 else n - per*(k-1)
        Xs.append(cx[i] + rng.standard_normal((ni, 2)) * noise * 1.5)
        ys.extend([i]*ni)
    return np.vstack(Xs), np.array(ys)


def _rings(n, noise, rng, k):
    k   = max(2, min(k, 4))
    radii = np.linspace(0.3, 1.2, k)
    per = n // k
    Xs, ys = [], []
    for i, r in enumerate(radii):
        ni = per if i < k-1 else n - per*(k-1)
        a  = rng.uniform(0, 2*np.pi, ni)
        ri = r + rng.standard_normal(ni) * noise * 0.5
        Xs.append(np.c_[ri*np.cos(a), ri*np.sin(a)])
        ys.extend([i]*ni)
    return np.vstack(Xs), np.array(ys)


def _xor(n, noise, rng):
    # Label = which diagonal quadrant you're in (the textbook XOR problem that
    # famously killed the single-layer perceptron).
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.standard_normal((n, 2)) * noise
    return X, y


def _wave(n, noise, rng):
    # A 1-D regression target (damped sine) rather than a classification set —
    # used by the labs that demonstrate function fitting instead of boundaries.
    X = rng.uniform(-3, 3, (n, 1))
    y = np.exp(-0.3*np.abs(X[:,0])) * np.sin(X[:,0]) + rng.standard_normal(n)*noise
    return X, y
